#include "renderer/passes/VirtualShadowMapPass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <format>
#include <limits>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h"
#include "io/CacheFileManager.h"
#include "renderer/vulkan/VulkanPipeline.h"

namespace renderer {

    namespace {

        // Byte-identical to ShadowMapPass.cpp's own ShadowCaptureConstants -- both pipelines share
        // the exact same ShadowMapCapture.vert unchanged (see this pass' own class comment).
        struct ShadowCaptureConstants {
            maths::mat4 lightViewProj;
        };
        static_assert(sizeof(ShadowCaptureConstants) == 64,
            "ShadowCaptureConstants must match ShadowMapCapture.vert's push_constant block exactly");

        constexpr float kHalfPi = 1.57079632679489661923f;

        // Standard cubemap face directions/up-vectors (Vulkan/D3D convention: +X,-X,+Y,-Y,+Z,-Z),
        // consumed identically by shadow_point_sampling.glsl's ShadowCubeFaceIndex so a world
        // position's selected face here always matches the face a shadow lookup samples.
        constexpr maths::vec3 kFaceDirs[6] = {
            {  1.0f,  0.0f,  0.0f }, { -1.0f,  0.0f,  0.0f },
            {  0.0f,  1.0f,  0.0f }, {  0.0f, -1.0f,  0.0f },
            {  0.0f,  0.0f,  1.0f }, {  0.0f,  0.0f, -1.0f },
        };
        constexpr maths::vec3 kFaceUps[6] = {
            { 0.0f, -1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
            { 0.0f,  0.0f, 1.0f }, { 0.0f,  0.0f, -1.0f },
            { 0.0f, -1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
        };

    } // namespace

    bool VirtualShadowMapPass::Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
        VkCommandPool commandPool, VkQueue queue, const std::filesystem::path& cacheFilePath) {
        kSunBaseRadius = config::lumen::VSM_SUN_BASE_RADIUS;
        kPhysicalPageCapacity = config::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
        kMaxPagesRenderedPerFrame = config::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;

        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Read every fallback mesh's geometry + derive the scene's bounding sphere.
        // Byte-for-byte the same procedure as ShadowMapPass::Init's own STEP 1 (own, separate
        // buffers -- see this pass' class comment on why a self-contained copy, not a shared read).
        // =====================================================================================
        geometry::CacheFileManager cacheManager;
        geometry::CacheFileHeader header{};
        if (!cacheManager.ReadHeader(cacheFilePath, header)) {
            LOG_ERROR(std::format("[VirtualShadowMapPass] Failed to read cache header '{}'.", cacheFilePath.string()));
            return false;
        }

        std::vector<geometry::FallbackMeshIndexEntry> fallbackTable;
        if (!cacheManager.ReadFallbackMeshTable(cacheFilePath, header, fallbackTable)) {
            LOG_ERROR("[VirtualShadowMapPass] Failed to read the fallback-mesh table.");
            return false;
        }

        maths::vec3 sceneBoundsMin{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        maths::vec3 sceneBoundsMax{ -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };

        std::vector<geometry::FallbackVertex> hostVertices;
        std::vector<uint32_t> hostIndices;
        for (const geometry::FallbackMeshIndexEntry& entry : fallbackTable) {
            std::vector<geometry::FallbackVertex> entityVertices;
            std::vector<uint32_t> entityIndices;
            if (!cacheManager.ReadFallbackMeshGeometry(cacheFilePath, entry, entityVertices, entityIndices)) {
                LOG_ERROR(std::format(
                    "[VirtualShadowMapPass] Failed to read fallback mesh geometry for entityID={}.", entry.entityID));
                return false;
            }

            const uint32_t indexBase = static_cast<uint32_t>(hostVertices.size());
            hostVertices.insert(hostVertices.end(), entityVertices.begin(), entityVertices.end());
            for (uint32_t index : entityIndices) {
                hostIndices.push_back(index + indexBase);
            }

            const maths::vec3 entityMin{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] };
            const maths::vec3 entityMax{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] };
            sceneBoundsMin.x = std::min(sceneBoundsMin.x, entityMin.x);
            sceneBoundsMin.y = std::min(sceneBoundsMin.y, entityMin.y);
            sceneBoundsMin.z = std::min(sceneBoundsMin.z, entityMin.z);
            sceneBoundsMax.x = std::max(sceneBoundsMax.x, entityMax.x);
            sceneBoundsMax.y = std::max(sceneBoundsMax.y, entityMax.y);
            sceneBoundsMax.z = std::max(sceneBoundsMax.z, entityMax.z);
        }

        m_TotalIndexCount = static_cast<uint32_t>(hostIndices.size());

        if (fallbackTable.empty()) {
            m_SceneBoundsCenter = maths::vec3{ 0.0f, 0.0f, 0.0f };
            m_SceneBoundsRadius = 0.0f;
        } else {
            m_SceneBoundsCenter = (sceneBoundsMin + sceneBoundsMax) * 0.5f;
            m_SceneBoundsRadius = (sceneBoundsMax - sceneBoundsMin).Length() * 0.5f;
        }

        // =====================================================================================
        // STEP 2 -- Combined vertex/index GPU buffers (position-only reads, same convention as
        // ShadowMapPass) + one-time upload.
        // =====================================================================================
        VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(hostVertices.size()) * sizeof(geometry::FallbackVertex);
        VkDeviceSize indexBytes = static_cast<VkDeviceSize>(hostIndices.size()) * sizeof(uint32_t);

        if (vertexBytes > 0 && indexBytes > 0) {
            m_VertexBuffer.Create(allocator, vertexBytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_IndexBuffer.Create(allocator, indexBytes,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocResultInfo{};
            VkDeviceSize stagingSize = vertexBytes + indexBytes;
            VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            stagingInfo.size = stagingSize;
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagingAllocInfo{};
            stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VK_CHECK(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));
            std::memcpy(stagingAllocResultInfo.pMappedData, hostVertices.data(), static_cast<size_t>(vertexBytes));
            std::memcpy(static_cast<char*>(stagingAllocResultInfo.pMappedData) + vertexBytes, hostIndices.data(), static_cast<size_t>(indexBytes));

            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            VkCommandBuffer uploadCmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &uploadCmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(uploadCmd, &beginInfo);

            VkBufferCopy vertexCopy{ 0, 0, vertexBytes };
            vkCmdCopyBuffer(uploadCmd, stagingBuffer, m_VertexBuffer.Handle(), 1, &vertexCopy);
            VkBufferCopy indexCopy{ vertexBytes, 0, indexBytes };
            vkCmdCopyBuffer(uploadCmd, stagingBuffer, m_IndexBuffer.Handle(), 1, &indexCopy);

            VkMemoryBarrier2 uploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            uploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            uploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            uploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            uploadBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
            VkDependencyInfo uploadDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            uploadDep.memoryBarrierCount = 1;
            uploadDep.pMemoryBarriers = &uploadBarrier;
            vkCmdPipelineBarrier2(uploadCmd, &uploadDep);

            vkEndCommandBuffer(uploadCmd);
            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &uploadCmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(m_Device, commandPool, 1, &uploadCmd);
            vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        }

        // =====================================================================================
        // STEP 3 -- Physical page pool + feedback buffer (own dedicated instance, see class
        // comment on why not shared with the geometry streaming system's own).
        // =====================================================================================
        if (!m_Pool.Init(physicalDevice, device, allocator, commandPool, queue, kTotalVSMCount, kPhysicalPageCapacity)) {
            LOG_ERROR("[VirtualShadowMapPass] Failed to initialize VirtualShadowMapPool.");
            return false;
        }
        m_Feedback.Init(allocator, kFeedbackCapacity);

        // =====================================================================================
        // STEP 4 -- Sun clipmap: each level's PROJECTION is fixed for this pass' entire lifetime
        // (radius_L never changes -- see class comment on why these windows never re-center); only
        // the VIEW half (sunDirection-dependent) is recomputed every RecordBeginFrame() call.
        // =====================================================================================
        for (uint32_t level = 0; level < kSunLevelCount; ++level) {
            float radius = std::max(kSunBaseRadius * float(1u << level), 1.0e-3f);
            float nearPlane = radius * kSunNearMarginFactor;
            float farPlane = radius * (2.0f + kSunFarMarginFactor);
            m_SunLevelProj[level] = maths::mat4::OrthoVulkan(radius, radius, nearPlane, farPlane);
        }

        // =====================================================================================
        // STEP 5 -- Depth-only capture pipeline: identical setup to ShadowMapPass::Init's own
        // STEP 4, reusing the exact same ShadowMapCapture.vert (already fully generic -- just
        // `gl_Position = pc.lightViewProj * vec4(inPosition, 1.0)`).
        // =====================================================================================
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(ShadowCaptureConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 0;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ShadowMapCapture.vert.spv");

        VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vertModule;
        stage.pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(geometry::FallbackVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attribute{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(geometry::FallbackVertex, position) };

        VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = 1;
        vertexInput.pVertexAttributeDescriptions = &attribute;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        // No back-face culling -- same self-shadowing rationale as ShadowMapPass's own pipeline.
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.attachmentCount = 0;

        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 0;
        pipelineRendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

        VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.stageCount = 1;
        pipelineInfo.pStages = &stage;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.pNext = &pipelineRendering;
        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));

        vkDestroyShaderModule(m_Device, vertModule, nullptr);

        // =====================================================================================
        // STEP 6 -- VSM matrix UBOs: persistently-mapped CPU_TO_GPU, plain memcpy every
        // RecordBeginFrame() call -- same idiom as renderer::SurfaceCachePass::UpdateLighting's own
        // m_LightingUBO (no descriptor-set update needed after the initial bind, no barrier needed
        // since the host write always happens-before this frame's vkQueueSubmit).
        // =====================================================================================
        m_SunLevelsUBO.Create(allocator, sizeof(m_SunLevelViewProj),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        m_PointFacesUBO.Create(allocator, sizeof(m_PointFaceViewProj),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        LOG_INFO(std::format(
            "[VirtualShadowMapPass] Initialized: {} fallback mesh(es) ({} vertices, {} indices), "
            "{} sun clipmap level(s), up to {} point light(s) x 6 faces, scene bounding sphere radius={:.3f}.",
            fallbackTable.size(), hostVertices.size(), hostIndices.size(), kSunLevelCount, kMaxPointLights, m_SceneBoundsRadius));
        return true;
    }

    void VirtualShadowMapPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
        }
        m_SunLevelsUBO.Destroy();
        m_PointFacesUBO.Destroy();
        m_Feedback.Shutdown();
        m_Pool.Shutdown();
        m_VertexBuffer.Destroy();
        m_IndexBuffer.Destroy();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_TotalIndexCount = 0;
        m_SceneBoundsRadius = 0.0f;
        m_ActivePointLightCount = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void VirtualShadowMapPass::RecordBeginFrame(VkCommandBuffer cmd, const maths::vec3& sunDirection,
        const SceneLights& sceneLights, const maths::vec3& cameraPosition) {
        (void)cameraPosition; // Unused -- see class comment on why the sun clipmap windows are fixed, not camera-following.

        // --- Recompute this frame's sun clipmap view-projection matrices (view depends on
        // sunDirection; projection was fixed once at Init(), see class comment). ---
        const maths::vec3 lightDir = sunDirection.Normalize();
        maths::vec3 up{ 0.0f, 1.0f, 0.0f };
        if (std::abs(lightDir.Dot(up)) > 0.999f) {
            up = maths::vec3{ 0.0f, 0.0f, 1.0f };
        }
        for (uint32_t level = 0; level < kSunLevelCount; ++level) {
            float radius = std::max(kSunBaseRadius * float(1u << level), 1.0e-3f);
            float nearPlane = radius * kSunNearMarginFactor;
            maths::vec3 eye = m_SceneBoundsCenter - lightDir * (radius + nearPlane);
            maths::mat4 view = maths::mat4::LookAt(eye, m_SceneBoundsCenter, up);
            m_SunLevelViewProj[level] = m_SunLevelProj[level] * view;
        }

        // --- Recompute this frame's point light cube face view-projection matrices. ---
        m_ActivePointLightCount = std::min(sceneLights.pointLightCount, kMaxPointLights);
        for (uint32_t slot = 0; slot < m_ActivePointLightCount; ++slot) {
            const PointLight& light = sceneLights.pointLights[slot];
            float farPlane = std::max(light.radius, 0.1f);
            float nearPlane = std::max(farPlane * 0.01f, 0.02f);
            maths::mat4 proj = maths::mat4::PerspectiveVulkan(kHalfPi, 1.0f, nearPlane, farPlane);
            for (uint32_t face = 0; face < 6; ++face) {
                maths::mat4 view = maths::mat4::LookAt(light.position, light.position + kFaceDirs[face], kFaceUps[face]);
                m_PointFaceViewProj[slot * 6 + face] = proj * view;
            }
        }

        std::memcpy(m_SunLevelsUBO.MappedData(), m_SunLevelViewProj, sizeof(m_SunLevelViewProj));
        std::memcpy(m_PointFacesUBO.MappedData(), m_PointFaceViewProj, sizeof(m_PointFaceViewProj));

        // --- Clear this frame's feedback counter before any consumer can write into it. ---
        m_Feedback.RecordClear(cmd);

        // --- Process LAST frame's page-miss reports (one-frame lag, see class comment): dedup,
        // allocate + render up to kMaxPagesRenderedPerFrame this frame. ---
        // config::lumen::BUILD_SHADOWS temporary kill-switch: skipping this entire block leaves
        // every shadow page permanently non-resident, which shadow_sun_sampling.glsl/
        // shadow_point_sampling.glsl already interpret as "fully lit" -- see EngineConfig.h's own
        // comment on this flag. m_SunLevelsUBO/m_PointFacesUBO above still get this frame's valid
        // view-projection matrices either way, so re-enabling this flag later needs no other change.
        bool renderedAnyPage = false;
        if (config::lumen::BUILD_SHADOWS) {
            std::vector<uint32_t> missedPages = m_Feedback.ReadRequestedClusterIDs();
            // Priority = -(vsmIndex): sun clipmap levels are indexed finest-first (kSunLevelCount
            // levels, see RecordBeginFrame's own radius = kSunBaseRadius * 2^level loop above), so
            // a lower vsmIndex is both a finer sun cascade AND, since point-light cube faces are
            // indexed immediately after every sun level, always prioritized ahead of any point
            // light's shadow pages -- one monotonic key covers both cases without a branch.
            std::vector<float> missedPriorities;
            missedPriorities.reserve(missedPages.size());
            for (uint32_t logicalPageID : missedPages) {
                uint32_t vsmIndex = logicalPageID / kShadowPagesPerVSM;
                missedPriorities.push_back(-float(vsmIndex));
            }
            m_RequestQueue.SubmitFrameRequests(missedPages, missedPriorities);

            for (uint32_t processed = 0; processed < kMaxPagesRenderedPerFrame; ++processed) {
                uint32_t logicalPageID = 0;
                if (!m_RequestQueue.PopNextRequest(logicalPageID)) {
                    break;
                }

                uint32_t vsmIndex = logicalPageID / kShadowPagesPerVSM;
                uint32_t localPageIndex = logicalPageID % kShadowPagesPerVSM;

                const maths::mat4* viewProj = nullptr;
                if (vsmIndex < kSunLevelCount) {
                    viewProj = &m_SunLevelViewProj[vsmIndex];
                } else {
                    uint32_t faceSlot = vsmIndex - kSunLevelCount;
                    if (faceSlot >= m_ActivePointLightCount * 6u) {
                        // Stale report for a now-inactive point light slot -- nothing to render.
                        m_RequestQueue.MarkRequestCompleted(logicalPageID);
                        continue;
                    }
                    viewProj = &m_PointFaceViewProj[faceSlot];
                }

                uint32_t physicalLayer = m_Pool.AllocatePage(cmd, logicalPageID);
                if (physicalLayer == kInvalidShadowPhysicalPage) {
                    m_RequestQueue.MarkRequestCompleted(logicalPageID); // Let a future miss report retry it.
                    continue;
                }

                RenderPage(cmd, vsmIndex, localPageIndex, physicalLayer, *viewProj);
                renderedAnyPage = true;
                m_RequestQueue.MarkRequestCompleted(logicalPageID);
            }
        }

        if (renderedAnyPage) {
            // Makes this call's page depth writes visible to a later fragment/vertex-shader
            // sampled read THIS SAME frame (SurfaceCacheCapture.frag, ClusterResolve.comp/
            // ClusterResolveBinned.comp) -- unlike the feedback buffer's own one-frame lag, a page
            // rendered this frame must be immediately visible, not next frame.
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
    }

    void VirtualShadowMapPass::RecordEndFrame(VkCommandBuffer cmd) {
        // Captures this frame's page-miss reports (written by whatever ran between
        // RecordBeginFrame() and this call) for RecordBeginFrame() to consume next frame -- see
        // this class' own header comment for the full one-frame-lag contract.
        m_Feedback.RecordReadback(cmd);
    }

    void VirtualShadowMapPass::RenderPage(VkCommandBuffer cmd, uint32_t vsmIndex, uint32_t localPageIndex,
        uint32_t physicalLayer, const maths::mat4& viewProj) {
        (void)vsmIndex; // Rendering itself is VSM-agnostic -- only `viewProj` (already resolved by the caller) matters.
        uint32_t pageX = localPageIndex % kShadowPagesPerAxis;
        uint32_t pageY = localPageIndex / kShadowPagesPerAxis;

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = m_Pool.GetPhysicalLayerView(physicalLayer);
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, { kShadowPageTexels, kShadowPageTexels } };
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = &depthAttachment;
        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

        // Virtual-viewport technique: a viewport LARGER than the 128x128 physical page, offset so
        // only page (pageX,pageY)'s own 128x128 region of the level/face's full 2048x2048 virtual
        // resolution falls inside [0,128)x[0,128) -- Vulkan explicitly permits viewport dimensions
        // exceeding the bound attachment (see VirtualShadowMapPool.h's own comment on why this is
        // guaranteed safe without a capability query on any conformant Vulkan 1.3 device); the
        // scissor below clips the actual rasterized output to the physical page's real bounds.
        constexpr float kVirtualResolution = float(kShadowPagesPerAxis * kShadowPageTexels); // 2048.
        VkViewport viewport{};
        viewport.x = -float(pageX) * float(kShadowPageTexels);
        viewport.y = -float(pageY) * float(kShadowPageTexels);
        viewport.width = kVirtualResolution;
        viewport.height = kVirtualResolution;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{ { 0, 0 }, { kShadowPageTexels, kShadowPageTexels } };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        if (m_TotalIndexCount > 0) {
            ShadowCaptureConstants pc{};
            pc.lightViewProj = viewProj;
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            VkBuffer vertexBuffer = m_VertexBuffer.Handle();
            VkDeviceSize vertexOffset0 = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset0);
            vkCmdBindIndexBuffer(cmd, m_IndexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

            // Same "one draw covers every entity" reasoning as ShadowMapPass::RecordCapture -- see
            // this pass' own class comment on why redrawing the whole (tiny) scene per page,
            // clipped by viewport+scissor, is not worth avoiding via per-page culling.
            vkCmdDrawIndexed(cmd, m_TotalIndexCount, 1, 0, 0, 0);
        }

        vkCmdEndRendering(cmd);
    }

}
