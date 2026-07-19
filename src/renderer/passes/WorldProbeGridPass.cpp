#include "renderer/passes/WorldProbeGridPass.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/Logger.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte layout match for WorldProbeInjectPushConstants in WorldProbeInject.comp.
        // Phase 6 (UE5.8 parity roadmap): replaced the old `gridOriginX/Y/Z` (the pre-toroidal
        // "texel (0,0,0) corner" convention) with the same `dispatchMinTexel`/`dispatchWorldMin`
        // split GlobalSDFCompositePC already uses -- `dispatchMinTexel` is this dispatch's WRAPPED
        // destination texel start, `dispatchWorldMin` is the matching TRUE world position (no wrap
        // arithmetic needed shader-side, exactly like GlobalSDFComposite.comp's own mode==0/1
        // branches).
        struct WorldProbeInjectPushConstants {
            int32_t dispatchMinTexelX = 0, dispatchMinTexelY = 0, dispatchMinTexelZ = 0;
            float dispatchWorldMinX = 0.0f, dispatchWorldMinY = 0.0f, dispatchWorldMinZ = 0.0f;
            float probeSpacing = 0.0f;
            uint32_t entityCount = 0;
            uint32_t frameIndex = 0;
            uint32_t traceMode = 0;
            // gridResolution: renderer::WorldProbeGridPass::kGridResolution is runtime-configurable
            // (config::lumen::PROBE_GRID_RESOLUTION, assigned at the top of Init() -- see that
            // function) -- the same tier-scaled pattern renderer::GlobalSDFPass::kClipmapResolution
            // now also follows (config::lumen::GLOBAL_SDF_CLIPMAP_RESOLUTION). Either way the
            // shader's own bounds check needs this value passed explicitly every dispatch, since a
            // push constant cannot read a C++-side global directly.
            int32_t gridResolution = 0;
            // Atmos weather system, Subtask 5.
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
        };
        static_assert(sizeof(WorldProbeInjectPushConstants) == 56,
            "WorldProbeInjectPushConstants must match WorldProbeInject.comp's push_constant block exactly");

        constexpr uint32_t kDispatchWorkgroupSize = 4; // Matches WorldProbeInject.comp's local_size_x/y/z.

        uint32_t DispatchGroupCount(uint32_t extent) {
            return (extent + kDispatchWorkgroupSize - 1u) / kDispatchWorkgroupSize;
        }

        // One contiguous, wrap-resolved piece of an axis range -- byte-for-byte mirror of
        // renderer::GlobalSDFPass.cpp's own WrappedPiece/SplitWrappedRange (duplicated locally
        // rather than shared, matching this codebase's established per-pass self-containment
        // convention). `worldVoxelStart` here is a world-PROBE-index (not an SDF voxel index), but
        // the wrap arithmetic is identical.
        struct WrappedPiece {
            int32_t worldVoxelStart = 0;
            int32_t wrappedTexelStart = 0;
            int32_t count = 0;
        };

        std::vector<WrappedPiece> SplitWrappedRange(int32_t minInclusive, int32_t maxExclusive, int32_t resolution) {
            std::vector<WrappedPiece> pieces;
            int32_t remaining = maxExclusive - minInclusive;
            int32_t cur = minInclusive;
            while (remaining > 0) {
                int32_t wrapped = ((cur % resolution) + resolution) % resolution;
                int32_t pieceLen = std::min(remaining, resolution - wrapped);
                pieces.push_back(WrappedPiece{ cur, wrapped, pieceLen });
                cur += pieceLen;
                remaining -= pieceLen;
            }
            return pieces;
        }

    } // namespace

    bool WorldProbeGridPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
        const SurfaceCacheRayTracingPass& rtPass) {
        kGridResolution = config::lumen::PROBE_GRID_RESOLUTION;
        kProbeSpacing = config::lumen::PROBE_SPACING;
        kProbeSampleDirections = config::lumen::PROBE_SAMPLE_DIRECTIONS;

        // Self-reinit (see ShadowMapPass's own migration comment for the identical pattern):
        // Shutdown() clears m_Device/m_Allocator, which RenderPass<WorldProbeGridPass>::Init()
        // already set to this call's values just before invoking this function -- restore them.
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- the grid image itself: kGridResolution^3, rgba16f, storage + sampled (the
        // former for this pass' own imageStore, the latter for world_probe_sampling.glsl's
        // consumers), kept VK_IMAGE_LAYOUT_GENERAL for its entire lifetime.
        // =====================================================================================
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.format = kGridFormat;
        imageInfo.extent = { kGridResolution, kGridResolution, kGridResolution };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_GridImage, &m_GridAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_GridImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = kGridFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_GridView));
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_GridView, nullptr);
            vmaDestroyImage(m_Allocator, m_GridImage, m_GridAllocation);
            m_GridView = VK_NULL_HANDLE;
            m_GridImage = VK_NULL_HANDLE;
            m_GridAllocation = VK_NULL_HANDLE;
        });

        // Phase 6 (UE5.8 parity roadmap): NEAREST, not LINEAR -- now that the grid is addressed
        // TOROIDALLY (see the class comment), hardware trilinear filtering would incorrectly blend
        // across the wrap seam (a texel near index 0 and a texel near index kGridResolution-1 can
        // represent world-probe-indices that are nowhere near each other once the window has
        // shifted) -- the exact same reasoning renderer::SDFRayMarchPass's own clipmap sampling
        // already relies on for renderer::GlobalSDFPass. world_probe_sampling.glsl's
        // SampleWorldProbeGrid() no longer uses this sampler's filtering at all (manual 8-corner
        // texelFetch blend instead, wrap-aware) -- kept NEAREST+CLAMP_TO_EDGE purely so a valid
        // sampler object still exists for the combined-image-sampler descriptor binding.
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_GridSampler));
        RegisterResource([this] { vkDestroySampler(m_Device, m_GridSampler, nullptr); m_GridSampler = VK_NULL_HANDLE; });

        // One-time UNDEFINED -> GENERAL transition (mirrors ClusterResolvePass::Init's own one-shot
        // pattern) -- stays GENERAL for this image's entire lifetime.
        VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_GridImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        // =====================================================================================
        // STEP 2 -- set 0's layout: grid output + the shared TLAS/vertex/index/draw-range
        // resources (a smaller set-0 than SurfaceCacheGIInjectPass's -- see the class comment on
        // why probes don't read the 4 per-texel Surface Cache atlases).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[6]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        // Atmos weather system, Subtask 5 -- see SetAtmosSkyView()'s own comment.
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorPoolSize poolSizes[4] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, // g_SkyViewLUT (Atmos Subtask 5).
        };
        auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device,
            std::span{ bindings, 6 }, std::span{ poolSizes, 4 });
        m_SetLayout = descSet.layout;
        m_DescriptorPool = descSet.pool;
        m_Set = descSet.set;
        RegisterResource([this] {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            m_DescriptorPool = VK_NULL_HANDLE;
            m_SetLayout = VK_NULL_HANDLE;
            m_Set = VK_NULL_HANDLE;
        });

        VkDescriptorImageInfo gridStorageInfo{ VK_NULL_HANDLE, m_GridView, VK_IMAGE_LAYOUT_GENERAL };

        VkWriteDescriptorSet writes[1]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m_Set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].pImageInfo = &gridStorageInfo;

        vkUpdateDescriptorSets(m_Device, 1, writes, 0, nullptr);

        VulkanUtils::WriteSharedGeometryBindings(m_Device, m_Set, 1, rtPass.GetTLASHandle(),
            surfaceCache.GetVertexBuffer(), surfaceCache.GetIndexBuffer(), rtPass.GetDrawRangeBuffer());

        // =====================================================================================
        // STEP 3 -- pipeline layout (set 0 above + set 1 mesh SDF trace scene + set 2 surface
        // cache sampling, both shared unmodified from traceContext, exactly like
        // SurfaceCacheGIInjectPass's own 3-set layout) + compute pipeline.
        // =====================================================================================
        VkDescriptorSetLayout setLayouts[3] = {
            m_SetLayout,
            traceContext.GetMeshSdfTraceSetLayout(),
            traceContext.GetSurfaceCacheSamplingSetLayout()
        };

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(WorldProbeInjectPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 3;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));
        RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); m_PipelineLayout = VK_NULL_HANDLE; });

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/WorldProbeInject.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        RegisterResource([this] { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); m_Pipeline = VK_NULL_HANDLE; });

        m_GridOriginWorld = maths::vec3{ 0.0f, 0.0f, 0.0f };
        m_FrameIndex = 0;
        m_SnappedCenterProbe[0] = m_SnappedCenterProbe[1] = m_SnappedCenterProbe[2] = 0;
        m_HasValidWindow = false;
        m_PendingSlabs.clear();
        // Re-registered so a LATER Shutdown() call (not just this InitImpl's own leading
        // self-reinit Shutdown()) still resets this POD state, matching the original Shutdown()'s
        // own unconditional reset of it.
        RegisterResource([this] {
            m_GridOriginWorld = maths::vec3{};
            m_FrameIndex = 0;
            m_SnappedCenterProbe[0] = m_SnappedCenterProbe[1] = m_SnappedCenterProbe[2] = 0;
            m_HasValidWindow = false;
            m_PendingSlabs.clear();
        });

        LOG_INFO(std::format("[WorldProbeGridPass] Initialized: {}^3 grid, {} world-unit spacing (toroidal streaming).",
            kGridResolution, kProbeSpacing));
        return true;
    }

    void WorldProbeGridPass::SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler) {
        VkDescriptorImageInfo skyViewInfo{ skyViewLUTSampler, skyViewLUTView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = m_Set; write.dstBinding = 5; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo = &skyViewInfo;
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
    }

    // Shutdown() is inherited from RenderPass<WorldProbeGridPass>: runs the RegisterResource()
    // cleanups above in reverse (POD state reset -> pipeline -> pipeline layout -> descriptor
    // pool+layout -> sampler -> grid view+image), the same dependency-safe order the hand-written
    // Shutdown() used.

    void WorldProbeGridPass::EnqueueDirtyRegionsForGrid(const maths::vec3& cameraPositionWorld) {
        const int32_t res = static_cast<int32_t>(kGridResolution);
        const int32_t halfRes = res / 2;

        // Snap the camera's world position down to the nearest whole probe-index, per axis -- no
        // coarser "chunk" quantization on top of that (unlike GlobalSDFPass::kSnapChunkVoxels):
        // this is a single-resolution grid, not a multi-level clipmap where a coarser snap step
        // reduces how many LEVELS re-window per frame, so there is no equivalent reason to
        // introduce one here (see the class comment's own note on this).
        auto snapAxis = [&](float worldPos) -> int32_t {
            return static_cast<int32_t>(std::floor(worldPos / kProbeSpacing));
            };
        int32_t newCenter[3] = {
            snapAxis(cameraPositionWorld.x), snapAxis(cameraPositionWorld.y), snapAxis(cameraPositionWorld.z)
        };

        if (!m_HasValidWindow) {
            DirtySlab slab{};
            for (int axis = 0; axis < 3; ++axis) {
                slab.probeMin[axis] = newCenter[axis] - halfRes;
                slab.probeMax[axis] = newCenter[axis] + halfRes;
            }
            m_PendingSlabs.push_back(slab);
            m_SnappedCenterProbe[0] = newCenter[0];
            m_SnappedCenterProbe[1] = newCenter[1];
            m_SnappedCenterProbe[2] = newCenter[2];
            m_HasValidWindow = true;
        } else {
            for (int axis = 0; axis < 3; ++axis) {
                int32_t delta = newCenter[axis] - m_SnappedCenterProbe[axis];
                if (delta == 0) {
                    continue;
                }

                DirtySlab slab{};
                for (int otherAxis = 0; otherAxis < 3; ++otherAxis) {
                    if (otherAxis == axis) continue;
                    // Full extent on the two axes that did not move this call -- see
                    // GlobalSDFPass::EnqueueDirtyRegionsForLevel's own identical comment: any
                    // diagonal motion this frame is covered by the UNION of the per-axis slabs
                    // enqueued here, one per moved axis.
                    slab.probeMin[otherAxis] = newCenter[otherAxis] - halfRes;
                    slab.probeMax[otherAxis] = newCenter[otherAxis] + halfRes;
                }

                if (std::abs(delta) >= res) {
                    // Moved farther than the whole covered window in one update -- cheaper to
                    // refill the entire window on this axis than to compute a slab wider than it.
                    slab.probeMin[axis] = newCenter[axis] - halfRes;
                    slab.probeMax[axis] = newCenter[axis] + halfRes;
                } else if (delta > 0) {
                    slab.probeMin[axis] = m_SnappedCenterProbe[axis] + halfRes;
                    slab.probeMax[axis] = newCenter[axis] + halfRes;
                } else {
                    slab.probeMin[axis] = newCenter[axis] - halfRes;
                    slab.probeMax[axis] = m_SnappedCenterProbe[axis] - halfRes;
                }
                m_PendingSlabs.push_back(slab);
            }

            m_SnappedCenterProbe[0] = newCenter[0];
            m_SnappedCenterProbe[1] = newCenter[1];
            m_SnappedCenterProbe[2] = newCenter[2];
        }

        // GetGridOriginWorld()'s new contract (see that method's own comment): the world-space
        // minimum corner of the window currently centered on m_SnappedCenterProbe.
        float halfExtent = static_cast<float>(halfRes) * kProbeSpacing;
        m_GridOriginWorld = maths::vec3{
            static_cast<float>(m_SnappedCenterProbe[0]) * kProbeSpacing - halfExtent,
            static_cast<float>(m_SnappedCenterProbe[1]) * kProbeSpacing - halfExtent,
            static_cast<float>(m_SnappedCenterProbe[2]) * kProbeSpacing - halfExtent
        };
    }

    void WorldProbeGridPass::RecordSlab(VkCommandBuffer cmd, const DirtySlab& slab,
        const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld) {
        const int32_t res = static_cast<int32_t>(kGridResolution);

        std::vector<WrappedPiece> piecesX = SplitWrappedRange(slab.probeMin[0], slab.probeMax[0], res);
        std::vector<WrappedPiece> piecesY = SplitWrappedRange(slab.probeMin[1], slab.probeMax[1], res);
        std::vector<WrappedPiece> piecesZ = SplitWrappedRange(slab.probeMin[2], slab.probeMax[2], res);

        for (const WrappedPiece& px : piecesX) {
            for (const WrappedPiece& py : piecesY) {
                for (const WrappedPiece& pz : piecesZ) {
                    WorldProbeInjectPushConstants pc{};
                    pc.dispatchMinTexelX = px.wrappedTexelStart;
                    pc.dispatchMinTexelY = py.wrappedTexelStart;
                    pc.dispatchMinTexelZ = pz.wrappedTexelStart;
                    pc.dispatchWorldMinX = static_cast<float>(px.worldVoxelStart) * kProbeSpacing;
                    pc.dispatchWorldMinY = static_cast<float>(py.worldVoxelStart) * kProbeSpacing;
                    pc.dispatchWorldMinZ = static_cast<float>(pz.worldVoxelStart) * kProbeSpacing;
                    pc.probeSpacing = kProbeSpacing;
                    pc.entityCount = traceContext.GetEntityCount();
                    pc.frameIndex = m_FrameIndex;
                    pc.traceMode = traceMode;
                    pc.gridResolution = res;
                    pc.sunDirX = sunDirectionWorld.x; pc.sunDirY = sunDirectionWorld.y; pc.sunDirZ = sunDirectionWorld.z;
                    vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

                    uint32_t groupsX = DispatchGroupCount(static_cast<uint32_t>(px.count));
                    uint32_t groupsY = DispatchGroupCount(static_cast<uint32_t>(py.count));
                    uint32_t groupsZ = DispatchGroupCount(static_cast<uint32_t>(pz.count));
                    vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);

                    // WorldProbeInject.comp writes disjoint texels across different (px,py,pz)
                    // combinations, but a barrier is still required between dispatches on the same
                    // queue whenever a later one could otherwise execute concurrently with an
                    // earlier one's still-in-flight writes -- explicit, not assumed, per CLAUDE.md's
                    // synchronization rule (mirrors GlobalSDFPass::RecordSlab's own barrier-after-
                    // every-dispatch discipline).
                    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    depInfo.memoryBarrierCount = 1;
                    depInfo.pMemoryBarriers = &barrier;
                    vkCmdPipelineBarrier2(cmd, &depInfo);
                }
            }
        }
    }

    void WorldProbeGridPass::DrainAndRecordSlabs(VkCommandBuffer cmd,
        const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld) {
        uint32_t processed = 0;
        while (processed < kMaxDirtySlabsPerCall && !m_PendingSlabs.empty()) {
            DirtySlab slab = m_PendingSlabs.front();
            m_PendingSlabs.pop_front();
            RecordSlab(cmd, slab, traceContext, traceMode, sunDirectionWorld);
            ++processed;
        }
    }

    void WorldProbeGridPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld,
        const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld) {
        EnqueueDirtyRegionsForGrid(cameraPositionWorld);

        // Pipeline + descriptor sets never change across the (possibly several) dispatches
        // DrainAndRecordSlabs() below records -- bound once here rather than per-dispatch (unlike
        // GlobalSDFPass::RecordSlab, which must rebind per-entity/per-mode; this pass has no such
        // per-dispatch resource variation).
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        VkDescriptorSet sets[3] = { m_Set, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 3, sets, 0, nullptr);

        DrainAndRecordSlabs(cmd, traceContext, traceMode, sunDirectionWorld);

        ++m_FrameIndex;
    }

}
