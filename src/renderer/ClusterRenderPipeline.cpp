#include "renderer/ClusterRenderPipeline.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "core/Logger.h"
#include "geometry/CacheFileManager.h"
#include "geometry/ClusterFormat.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    bool ClusterRenderPipeline::Init(const ClusterRenderPipelineCreateInfo& createInfo) {
        Shutdown();

        m_Device = createInfo.device;
        m_RenderExtent = createInfo.renderExtent;
        m_Allocator = createInfo.allocator;
        m_CommandPool = createInfo.commandPool;
        m_Queue = createInfo.queue;
        m_VisBufferClusterIDImage = createInfo.visBufferClusterIDImage;
        m_VisBufferClusterIDView = createInfo.visBufferClusterIDView;
        m_VisBufferTriangleIDImage = createInfo.visBufferTriangleIDImage;
        m_VisBufferTriangleIDView = createInfo.visBufferTriangleIDView;
        m_DepthImage = createInfo.depthImage;
        m_DepthImageView = createInfo.depthImageView;

        // =========================================================================================
        // STEP 1 -- Read the consolidated .cache file's header and both tables (CPU side).
        // =========================================================================================
        geometry::CacheFileManager cacheManager;

        geometry::CacheFileHeader header{};
        if (!cacheManager.ReadHeader(createInfo.cacheFilePath, header)) {
            Logger::Log(LogLevel::Error, std::format("[ClusterRenderPipeline] Failed to read cache header '{}'.", createInfo.cacheFilePath.string()));
            return false;
        }

        std::vector<geometry::ClusterIndexEntry> indexEntries;
        std::vector<geometry::DAGNodeEntry> dagEntries;
        if (!cacheManager.ReadClusterIndexTable(createInfo.cacheFilePath, header, indexEntries)
            || !cacheManager.ReadDAGTable(createInfo.cacheFilePath, header, dagEntries)
            || indexEntries.empty() || indexEntries.size() != dagEntries.size()) {
            Logger::Log(LogLevel::Error, "[ClusterRenderPipeline] Failed to read cache cluster/DAG tables.");
            return false;
        }

        uint32_t totalClusterCount = static_cast<uint32_t>(indexEntries.size());
        Logger::Log(LogLevel::Info, std::format("[ClusterRenderPipeline] Cache tables read: {} clusters.", totalClusterCount));

        // =========================================================================================
        // STEP 2 -- Streaming: read the whole geometry section into one host-visible staging
        // buffer. Startup-only bulk load through the exact same page-granular path a per-frame
        // streamer would use (BindPage per 4 KB page below) -- see the class comment's streaming
        // scope note for why the per-frame async residency loop is not active yet.
        // =========================================================================================
        VkDeviceSize geometrySectionSize = header.totalFileSizeBytes - header.geometryDataBaseOffset;

        GpuBuffer stagingBuffer;
        stagingBuffer.Create(createInfo.allocator, geometrySectionSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);

        {
            std::ifstream file(createInfo.cacheFilePath, std::ios::binary);
            if (!file.is_open()) {
                Logger::Log(LogLevel::Error, "[ClusterRenderPipeline] Failed to open cache file for the geometry section read.");
                return false;
            }
            file.seekg(static_cast<std::streamoff>(header.geometryDataBaseOffset));
            file.read(static_cast<char*>(stagingBuffer.MappedData()), static_cast<std::streamsize>(geometrySectionSize));
            if (!file) {
                Logger::Log(LogLevel::Error, "[ClusterRenderPipeline] Short read on the cache geometry section.");
                return false;
            }
        }

        // =========================================================================================
        // STEP 3 -- Initialize the GPU-side stages that the setup command buffer below records into.
        // =========================================================================================
        // Logical address space spans the whole file (virtualAddress is a file offset); physical
        // capacity is exactly one page per cluster -- the whole scene stays resident (see the
        // class comment), so no eviction can ever occur.
        uint32_t maxLogicalPages = static_cast<uint32_t>(header.totalFileSizeBytes / geometry::kPageSizeBytes);
        m_PagePool.Init(createInfo.allocator, maxLogicalPages, totalClusterCount);
        m_Decompression.Init(createInfo.device, createInfo.allocator, totalClusterCount, m_PagePool.GetPhysicalPoolBuffer());

        m_HZB.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
            createInfo.depthImageView, createInfo.renderExtent);

        // Candidate set = DAG leaves only (full detail) -- see the class comment's LOD scope note.
        uint32_t leafCount = 0;
        for (const geometry::DAGNodeEntry& node : dagEntries) {
            if (node.level == 0) {
                ++leafCount;
            }
        }
        if (leafCount == 0) {
            Logger::Log(LogLevel::Error, "[ClusterRenderPipeline] Cache contains no leaf clusters.");
            return false;
        }

        m_OcclusionCulling.Init(createInfo.device, createInfo.allocator, leafCount,
            m_HZB.GetFullView(), m_HZB.GetMipExtent(0), m_HZB.GetMipLevelCount());

        // =========================================================================================
        // STEP 4 -- One setup command buffer: page-table clear, then for EVERY cluster (all DAG
        // levels, so a future GPU LOD cut finds its coarser levels already resident) one 4 KB page
        // bind (staging -> physical pool) + vertex decompression + index expansion, then the
        // persisted-visibility clear. One blocking submit, at startup only -- the per-frame path
        // never submits or waits mid-frame.
        // =========================================================================================
        {
            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = createInfo.commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;

            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            m_PagePool.ClearPageTable(cmd);

            for (const geometry::ClusterIndexEntry& entry : indexEntries) {
                bool bound = m_PagePool.BindPage(cmd, entry.virtualAddress, stagingBuffer.Handle(),
                    entry.virtualAddress - header.geometryDataBaseOffset, geometry::kPageSizeBytes);
                if (!bound) {
                    Logger::Log(LogLevel::Error, std::format("[ClusterRenderPipeline] BindPage failed for cluster {}.", entry.clusterID));
                    vkEndCommandBuffer(cmd);
                    vkFreeCommandBuffers(m_Device, createInfo.commandPool, 1, &cmd);
                    return false;
                }

                // The logical->physical mapping is CPU-side bookkeeping, valid the moment
                // BindPage() records -- so the matching decompression dispatch can target the
                // exact physical slot immediately, in the same command buffer.
                uint32_t physicalPage = m_PagePool.GetPhysicalPageIndex(entry.virtualAddress);
                m_Decompression.DecompressPage(cmd, physicalPage,
                    maths::vec3{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] },
                    maths::vec3{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] });
            }

            m_OcclusionCulling.ClearPersistedVisibility(cmd);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(createInfo.queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(createInfo.queue));
            vkFreeCommandBuffers(m_Device, createInfo.commandPool, 1, &cmd);
        }

        stagingBuffer.Destroy();
        Logger::Log(LogLevel::Info, "[ClusterRenderPipeline] All cluster pages streamed + decompressed.");

        // =========================================================================================
        // STEP 5 -- Build and upload the leaf clusters' culling metadata. firstIndex/vertexOffset
        // follow GeometryDecompressionPass's documented physical-page-slot layout contract, which
        // is exactly what ClusterRaster.vert / ClusterSoftwareRaster.comp decode back.
        // =========================================================================================
        std::vector<ClusterCullMetadata> metadata;
        metadata.reserve(leafCount);
        // DEBUG (temporary): ClusterCullMetadata deliberately has no entity ID field (the GPU
        // culling/routing code never needs one), so this CPU-side shadow array -- index-aligned
        // with `metadata`/the GPU cluster-slot indices -- is how the debug outcome histogram
        // (see RecordFrame's periodic dump) maps a cluster slot back to which primitive it came
        // from, using the same ClusterIndexEntry::entityID already read from disk.
        m_ClusterSlotToEntityID.reserve(leafCount);
        for (size_t i = 0; i < indexEntries.size(); ++i) {
            if (dagEntries[i].level != 0) {
                continue;
            }
            const geometry::ClusterIndexEntry& entry = indexEntries[i];
            uint32_t physicalPage = m_PagePool.GetPhysicalPageIndex(entry.virtualAddress);

            ClusterCullMetadata meta{};
            meta.boundsMin = { entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] };
            meta.boundsMax = { entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] };
            meta.sphereCenter = { entry.sphereCenter[0], entry.sphereCenter[1], entry.sphereCenter[2] };
            meta.sphereRadius = entry.sphereRadius;
            // Inverse of BuildIndexEntry's int8 quantization (VirtualGeometryCacheTest.cpp): the
            // axis is re-normalized after dequantization since per-component rounding denormalizes it.
            maths::vec3 coneAxis{ entry.coneAxisX / 127.0f, entry.coneAxisY / 127.0f, entry.coneAxisZ / 127.0f };
            meta.coneAxis = coneAxis.Normalize();
            meta.coneCutoff = entry.coneCutoff / 127.0f;
            meta.indexCount = entry.indexCount;
            meta.firstIndex = physicalPage * geometry::kMaxClusterIndices;
            meta.vertexOffset = physicalPage * geometry::kMaxClusterVertices;
            meta.clusterID = entry.clusterID;
            metadata.push_back(meta);
            m_ClusterSlotToEntityID.push_back(entry.entityID);
        }
        m_ClusterCount = static_cast<uint32_t>(metadata.size());

        m_OcclusionCulling.UploadClusterMetadata(createInfo.commandPool, createInfo.queue, metadata);
        Logger::Log(LogLevel::Info, "[ClusterRenderPipeline] Culling metadata uploaded; initializing raster/resolve passes...");

        // =========================================================================================
        // STEP 6 -- Initialize the hybrid rasterization + resolve stages against the buffers the
        // culling pass owns.
        // =========================================================================================
        std::array<VkFormat, 2> visBufferFormats{ createInfo.visBufferFormat, createInfo.visBufferFormat };
        m_HardwareRaster.Init(createInfo.device, m_OcclusionCulling.GetClusterMetadataBuffer(),
            m_PagePool.GetPhysicalPoolBuffer(), visBufferFormats, createInfo.depthFormat);

        m_SoftwareRaster.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
            createInfo.renderExtent, m_OcclusionCulling.GetClusterMetadataBuffer(),
            m_PagePool.GetPhysicalPoolBuffer(), m_OcclusionCulling.GetSoftwareClusterListBuffer());

        m_Resolve.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
            createInfo.renderExtent, m_OcclusionCulling.GetClusterMetadataBuffer(),
            m_PagePool.GetPhysicalPoolBuffer(), createInfo.visBufferClusterIDView,
            createInfo.visBufferTriangleIDView, createInfo.depthImageView,
            m_SoftwareRaster.GetVisBufferAtomicView());

        Logger::Log(LogLevel::Info, std::format(
            "[ClusterRenderPipeline] Initialized: {} clusters streamed from cache ({} leaf candidates, {} logical pages).",
            totalClusterCount, m_ClusterCount, maxLogicalPages));
        return true;
    }

    void ClusterRenderPipeline::Shutdown() {
        // Reverse initialization order; each Shutdown() is null-safe on a never-initialized pass.
        m_Resolve.Shutdown();
        m_SoftwareRaster.Shutdown();
        m_HardwareRaster.Shutdown();
        m_OcclusionCulling.Shutdown();
        m_HZB.Shutdown();
        m_Decompression.Shutdown();
        m_PagePool.Shutdown();

        m_ClusterCount = 0;
        m_ClusterSlotToEntityID.clear();
        m_FrameCounter = 0;
        m_DebugOutcomeDumped = false;
        m_Allocator = VK_NULL_HANDLE;
        m_CommandPool = VK_NULL_HANDLE;
        m_Queue = VK_NULL_HANDLE;
        m_VisBufferClusterIDImage = VK_NULL_HANDLE;
        m_VisBufferClusterIDView = VK_NULL_HANDLE;
        m_VisBufferTriangleIDImage = VK_NULL_HANDLE;
        m_VisBufferTriangleIDView = VK_NULL_HANDLE;
        m_DepthImage = VK_NULL_HANDLE;
        m_DepthImageView = VK_NULL_HANDLE;
        m_RenderExtent = { 0, 0 };
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterRenderPipeline::BeginVisBufferRendering(VkCommandBuffer cmd, bool clearAttachments) const {
        VkAttachmentLoadOp loadOp = clearAttachments ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;

        VkRenderingAttachmentInfo colorAttachments[2]{};
        colorAttachments[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachments[0].imageView = m_VisBufferClusterIDView;
        colorAttachments[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachments[0].loadOp = loadOp;
        colorAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // 0xFFFFFFFF sentinel: matches geometry::kInvalidClusterID and ClusterResolve.comp's
        // "nothing rasterized here" hardware-path check.
        colorAttachments[0].clearValue.color.uint32[0] = 0xFFFFFFFFu;

        colorAttachments[1] = colorAttachments[0];
        colorAttachments[1].imageView = m_VisBufferTriangleIDView;

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = m_DepthImageView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = loadOp;
        // STORE (unlike the old flat path's DONT_CARE): this depth is consumed twice after each
        // raster pass -- by the HZB rebuilds and by the resolve pass's per-pixel depth arbitration.
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, m_RenderExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 2;
        renderingInfo.pColorAttachments = colorAttachments;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);
    }

    // DEBUG (temporary): see the class comment above RecordFrame()'s trigger site.
    void ClusterRenderPipeline::DumpDebugOutcomeHistogram(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) const {
        VkDeviceSize readbackSize = static_cast<VkDeviceSize>(sizeof(uint32_t)) * m_ClusterCount;

        GpuBuffer stagingBuffer;
        stagingBuffer.Create(allocator, readbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU, /*mapped=*/true);

        VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd));

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Make the previous frame's culling-shader writes to the debug outcome buffer visible to
        // this copy -- see RecordFrame()'s trigger comment for why this is guaranteed complete.
        VkMemoryBarrier2 preCopyBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        preCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preCopyBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        preCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        preCopyBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

        VkDependencyInfo preCopyDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preCopyDependency.memoryBarrierCount = 1;
        preCopyDependency.pMemoryBarriers = &preCopyBarrier;
        vkCmdPipelineBarrier2(cmd, &preCopyDependency);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = readbackSize;
        vkCmdCopyBuffer(cmd, m_OcclusionCulling.GetDebugOutcomeBuffer(), stagingBuffer.Handle(), 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
        vkFreeCommandBuffers(device, commandPool, 1, &cmd);

        const uint32_t* outcomes = static_cast<const uint32_t*>(stagingBuffer.MappedData());

        static constexpr const char* kOutcomeNames[8] = {
            "frustum-rejected", "backface-rejected", "occluded-early(pending)", "occluded-late(final)",
            "drawn-hw-early", "drawn-hw-late", "drawn-sw-early", "drawn-sw-late"
        };

        // Fixed 12 entities x 8 outcome codes -- see ClusterHZBOcclusionCull.comp's DebugOutcomeSSBO
        // encoding comment for the code list.
        uint32_t countsByEntityAndOutcome[12][8] = {};
        for (uint32_t slot = 0; slot < m_ClusterCount; ++slot) {
            uint32_t entityID = m_ClusterSlotToEntityID[slot];
            uint32_t outcome = outcomes[slot];
            if (entityID < 12 && outcome < 8) {
                ++countsByEntityAndOutcome[entityID][outcome];
            }
        }

        Logger::Log(LogLevel::Info, std::format("[ClusterRenderPipeline] DEBUG per-entity culling outcome histogram (frame {}):", m_FrameCounter));
        for (uint32_t entityID = 0; entityID < 12; ++entityID) {
            std::string line = std::format("[ClusterRenderPipeline]   meshID={}: ", entityID);
            for (uint32_t outcome = 0; outcome < 8; ++outcome) {
                uint32_t count = countsByEntityAndOutcome[entityID][outcome];
                if (count > 0) {
                    line += std::format("{}={} ", kOutcomeNames[outcome], count);
                }
            }
            Logger::Log(LogLevel::Info, line);
        }
    }

    void ClusterRenderPipeline::RecordFrame(VkCommandBuffer cmd, const CameraPushConstants& camera,
        const maths::vec3& cameraPositionWorld, VkImage swapchainImage) {
        assert(m_ClusterCount > 0 && "RecordFrame called before a successful Init");

        // DEBUG (temporary): dump a per-entity culling-outcome histogram once, after the scene has
        // reached steady state (frame 60). Safe to do here, before this frame's own `cmd` is
        // recorded/submitted: main.cpp's single-frame-in-flight loop already waited on the previous
        // frame's fence before calling RecordFrame(), so the previous frame's debugOutcome writes
        // are guaranteed complete and this blocking readback cannot race anything.
        ++m_FrameCounter;
        if (m_FrameCounter >= 55 && m_FrameCounter <= 65) {
            DumpDebugOutcomeHistogram(m_Device, m_Allocator, m_CommandPool, m_Queue);
        }

        // Every stage of this frame consumes the SAME combined matrix -- this is what makes the
        // resolve pass's screen-space triangle re-projection bit-identical to the rasterizers'.
        maths::mat4 viewProj = camera.proj * camera.view;

        ClusterCullViewParams viewParams{};
        viewParams.frustumPlanes = ExtractFrustumPlanes(viewProj);
        viewParams.cameraPositionWorld = cameraPositionWorld;

        // abs(proj.m[5]): PerspectiveVulkan stores -1/tan(fovY/2) there (Y flip), and the screen
        // size classification needs the positive scale.
        float projScaleY = std::abs(camera.proj.m[5]);

        // =========================================================================================
        // [1] Per-frame worklist clears. Each Record*() carries its own CLEAR->COMPUTE barrier, so
        // the culling/raster dispatches below can never read a stale counter.
        // =========================================================================================
        m_OcclusionCulling.RecordClearFrame(cmd);
        m_SoftwareRaster.RecordClear(cmd);

        // =========================================================================================
        // [2] EARLY cull: every leaf candidate vs frustum/backface + LAST frame's HZB (rebuilt at
        // the end of the previous RecordFrame from that frame's complete depth). Its trailing
        // barrier makes the early draw list/count visible to DRAW_INDIRECT and the pending +
        // software lists visible to later COMPUTE.
        // =========================================================================================
        m_OcclusionCulling.RecordEarlyPass(cmd, viewParams, viewProj, projScaleY, m_ClusterCount, kSoftwareRasterThresholdPixels);

        // =========================================================================================
        // [3] Attachment layout acquisition. All three images are re-acquired with
        // oldLayout = UNDEFINED (content discarded): both raster passes fully re-render every
        // frame, and cross-frame GPU ordering is already guaranteed by the frame fence, so no
        // previous-frame content ever needs preserving here. srcStageMask still names the stages
        // that touched these images at the end of the previous frame (resolve's compute reads and
        // the second HZB rebuild's depth sampling) purely as an execution dependency, keeping the
        // transition correct even if the frame pacing later moves to multiple frames in flight.
        // =========================================================================================
        {
            VkImageMemoryBarrier2 barriers[3]{};

            barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barriers[0].srcAccessMask = 0; // UNDEFINED discard: nothing to make available.
            barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].image = m_VisBufferClusterIDImage;
            barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            barriers[1] = barriers[0];
            barriers[1].image = m_VisBufferTriangleIDImage;

            barriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barriers[2].srcAccessMask = 0;
            barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barriers[2].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barriers[2].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[2].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[2].image = m_DepthImage;
            barriers[2].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 3;
            depInfo.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =========================================================================================
        // [4] EARLY hardware raster: draws exactly the early cull's survivors via
        // vkCmdDrawIndexedIndirectCount into the VisBuffer + depth (CLEAR load ops).
        // =========================================================================================
        BeginVisBufferRendering(cmd, /*clearAttachments=*/true);
        m_HardwareRaster.RecordDraw(cmd, camera, m_RenderExtent, m_Decompression.GetDecompressedIndexPoolBuffer(),
            m_OcclusionCulling.GetEarlyIndirectCommandBuffer(), m_OcclusionCulling.GetEarlyDrawCountBuffer(), m_ClusterCount);
        vkCmdEndRendering(cmd);

        // =========================================================================================
        // [5] Depth -> sampled-readable for the HZB rebuild: the early pass's depth writes (late
        // fragment tests) must complete and be visible before HZBBuildInit.comp samples the image.
        // =========================================================================================
        {
            VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_DepthImage;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =========================================================================================
        // [6] Mid-frame HZB rebuild from the early-pass depth. The early cull's own HZB sampling
        // (of last frame's pyramid) is ordered before Generate()'s first imageStore by the
        // execution dependency RecordEarlyPass's trailing barrier already created (srcStage
        // COMPUTE covers all prior compute, dst includes COMPUTE) -- no WAR hazard. Generate()'s
        // internal per-mip barriers end at STORAGE_READ; the extra memory barrier below extends
        // visibility to SAMPLED_READ, which is how the late cull actually reads the pyramid
        // (textureLod through a combined image sampler) -- the exact contract documented on
        // ClusterOcclusionCullingPass step 5.
        // =========================================================================================
        m_HZB.Generate(cmd);
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =========================================================================================
        // [7] LATE cull: GPU-sized indirect dispatch over exactly the pending list, against the
        // fresh HZB. RecordLatePass's trailing barrier covers the late draw list for
        // DRAW_INDIRECT; the extra barrier below additionally makes its software-cluster-list
        // writes visible to COMPUTE (the software raster's dispatch-args build + raster reads),
        // which that trailing barrier does not cover.
        // =========================================================================================
        m_OcclusionCulling.RecordBuildLateDispatchArgs(cmd);
        m_OcclusionCulling.RecordLatePass(cmd);
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =========================================================================================
        // [8] Re-arm the attachments for the LATE raster: depth returns to attachment layout
        // (contents PRESERVED -- the late draws must depth-test against the early geometry), and
        // both VisBuffer images get a write-after-write dependency between the two rendering
        // passes (no layout change; without it the two passes' color writes are unordered).
        // =========================================================================================
        {
            VkImageMemoryBarrier2 barriers[3]{};

            barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
            barriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].image = m_VisBufferClusterIDImage;
            barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            barriers[1] = barriers[0];
            barriers[1].image = m_VisBufferTriangleIDImage;

            // WAR on the depth image (HZB sampled it in [6]; the late raster writes it): a read-
            // before-write hazard needs only the execution dependency plus the layout transition
            // -- there are no prior writes to make available, hence srcAccessMask = 0.
            barriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[2].srcAccessMask = 0;
            barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barriers[2].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            barriers[2].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barriers[2].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[2].image = m_DepthImage;
            barriers[2].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 3;
            depInfo.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =========================================================================================
        // [9] LATE hardware raster (loadOp = LOAD) -- draws the disocclusions the early pass
        // could not confirm, on top of the early output.
        // =========================================================================================
        BeginVisBufferRendering(cmd, /*clearAttachments=*/false);
        m_HardwareRaster.RecordDraw(cmd, camera, m_RenderExtent, m_Decompression.GetDecompressedIndexPoolBuffer(),
            m_OcclusionCulling.GetLateIndirectCommandBuffer(), m_OcclusionCulling.GetLateDrawCountBuffer(), m_ClusterCount);
        vkCmdEndRendering(cmd);

        // =========================================================================================
        // [10] Software raster of every micro-triangle cluster (early- and late-routed entries are
        // both in the list by now). Writes only its own R64 atomic image -- fully independent of
        // the hardware attachments, so no ordering against [9] is required beyond what its own
        // internal barriers already record.
        // =========================================================================================
        m_SoftwareRaster.RecordRaster(cmd, viewProj);

        // =========================================================================================
        // [11] Hand the hardware VisBuffer + depth to the resolve pass: color attachments become
        // GENERAL storage images (the layout the resolve descriptors were written with), depth
        // becomes sampled-readable again.
        // =========================================================================================
        {
            VkImageMemoryBarrier2 barriers[3]{};

            barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            barriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].image = m_VisBufferClusterIDImage;
            barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            barriers[1] = barriers[0];
            barriers[1].image = m_VisBufferTriangleIDImage;

            barriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barriers[2].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barriers[2].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            barriers[2].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[2].image = m_DepthImage;
            barriers[2].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 3;
            depInfo.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =========================================================================================
        // [12] Resolve: per-pixel hardware-vs-software arbitration + barycentric reconstruction +
        // material evaluation into the resolve pass's own RGBA8 output image. The software
        // rasterizer's atomic writes are already visible (RecordRaster's trailing COMPUTE ->
        // COMPUTE barrier).
        // =========================================================================================
        m_Resolve.RecordResolve(cmd, viewProj);

        // =========================================================================================
        // [13] Second HZB rebuild, from the now-complete (early + late) depth: this is the pyramid
        // NEXT frame's early pass tests against. WAR ordering vs the late cull's sampling of the
        // previous pyramid content is transitively guaranteed (multiple COMPUTE -> COMPUTE
        // execution dependencies sit in between), but is restated here explicitly so the
        // dependency does not silently vanish if an intermediate pass's barriers change. Depth is
        // already in DEPTH_STENCIL_READ_ONLY_OPTIMAL from [11], which Generate() accepts.
        // =========================================================================================
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = 0; // WAR only: the prior accesses were reads, nothing to flush.
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
        m_HZB.Generate(cmd);
        {
            // Next frame's early cull samples the pyramid; same STORAGE_WRITE -> SAMPLED_READ
            // extension as [6]. Barriers order against subsequent commands in submission order on
            // the same queue, so this correctly covers the next command buffer too.
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =========================================================================================
        // [14] Blit the resolved image to the swapchain image and hand it to present. The resolve
        // output stays in GENERAL (valid blit source layout); its own trailing barrier targeted
        // the COPY stage, but vkCmdBlitImage executes in the BLIT stage, so visibility is
        // re-extended here explicitly.
        // =========================================================================================
        {
            VkMemoryBarrier2 resolveToBlitBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            resolveToBlitBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            resolveToBlitBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            resolveToBlitBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            resolveToBlitBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

            VkImageMemoryBarrier2 swapchainToBlitBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            swapchainToBlitBarrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            swapchainToBlitBarrier.srcAccessMask = 0;
            swapchainToBlitBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            swapchainToBlitBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            swapchainToBlitBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            swapchainToBlitBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapchainToBlitBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainToBlitBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainToBlitBarrier.image = swapchainImage;
            swapchainToBlitBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &resolveToBlitBarrier;
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &swapchainToBlitBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        VkImageBlit blitRegion{};
        blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blitRegion.srcOffsets[1] = { static_cast<int32_t>(m_RenderExtent.width), static_cast<int32_t>(m_RenderExtent.height), 1 };
        blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blitRegion.dstOffsets[1] = { static_cast<int32_t>(m_RenderExtent.width), static_cast<int32_t>(m_RenderExtent.height), 1 };
        // Same extent both sides -- the "blit" is a 1:1 copy that also performs the
        // RGBA8 -> B8G8R8A8 component reordering the swapchain format requires.
        vkCmdBlitImage(cmd, m_Resolve.GetOutputColorImage(), VK_IMAGE_LAYOUT_GENERAL,
            swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_NEAREST);

        {
            VkImageMemoryBarrier2 presentBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            presentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            presentBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            presentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            presentBarrier.dstAccessMask = 0;
            presentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            presentBarrier.image = swapchainImage;
            presentBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &presentBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
    }

}
