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
        // F1: +levelSpacing (was implicitly `probeSpacing`, now varies per level -- see
        // WorldProbeGridPass::GetLevelSpacing's own comment). `dispatchMinTexel` is this dispatch's
        // WRAPPED destination texel start, `dispatchWorldMin` is the matching TRUE world position
        // (no wrap arithmetic needed shader-side), mirroring GlobalSDFCompositePC's own identical
        // split.
        struct WorldProbeInjectPushConstants {
            int32_t dispatchMinTexelX = 0, dispatchMinTexelY = 0, dispatchMinTexelZ = 0;
            float dispatchWorldMinX = 0.0f, dispatchWorldMinY = 0.0f, dispatchWorldMinZ = 0.0f;
            float levelSpacing = 0.0f;
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
        // STEP 1 -- set 0's layout: grid + occlusion output + the shared TLAS/vertex/index/
        // draw-range resources (F1: +1 storage-image binding vs. the pre-F1 single-image layout,
        // for the occlusion texel -- see the class comment's own "probe occlusion" note) + the
        // Atmos Sky-View LUT. ONE layout, shared by every level's own descriptor set (allocated
        // below, STEP 3).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[7]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };               // g_ProbeGrid (irradiance)
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };               // g_ProbeOcclusion (F1)
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        // Atmos weather system, Subtask 5 -- see SetAtmosSkyView()'s own comment.
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 7;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        // Pool sized for kLevelCount independent sets of the layout above (mirrors
        // GlobalSDFPass::InitImpl's own `poolInfo.maxSets = kLevelCount + entitySetCount` pattern).
        VkDescriptorPoolSize poolSizes[4] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kLevelCount },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 * kLevelCount },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * kLevelCount },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 * kLevelCount }, // g_SkyViewLUT (Atmos Subtask 5).
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = kLevelCount;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));
        RegisterResource([this] {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            m_DescriptorPool = VK_NULL_HANDLE;
            m_SetLayout = VK_NULL_HANDLE;
        });

        // =====================================================================================
        // STEP 2 -- shared NEAREST + CLAMP_TO_EDGE sampler (see the header's own comment on why
        // NEAREST -- toroidal wrap addressing means hardware trilinear filtering would incorrectly
        // blend across the wrap seam; world_probe_sampling.glsl does its own manual 8-corner
        // texelFetch blend instead). One sampler, reused by every level.
        // =====================================================================================
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

        // =====================================================================================
        // STEP 3 -- per level: grid + occlusion images/views, one-time UNDEFINED -> GENERAL
        // transitions, one descriptor set (from the shared layout/pool above) with every binding
        // written.
        // =====================================================================================
        for (uint32_t level = 0; level < kLevelCount; ++level) {
            ClipmapLevel& lvl = m_Levels[level];

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
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &lvl.gridImage, &lvl.gridAllocation, nullptr));

            VkImageCreateInfo occImageInfo = imageInfo;
            occImageInfo.format = kOcclusionFormat;
            VK_CHECK(vmaCreateImage(allocator, &occImageInfo, &allocInfo, &lvl.occlusionImage, &lvl.occlusionAllocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = lvl.gridImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            viewInfo.format = kGridFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &lvl.gridView));

            VkImageViewCreateInfo occViewInfo = viewInfo;
            occViewInfo.image = lvl.occlusionImage;
            occViewInfo.format = kOcclusionFormat;
            VK_CHECK(vkCreateImageView(m_Device, &occViewInfo, nullptr, &lvl.occlusionView));

            RegisterResource([this, level] {
                ClipmapLevel& l = m_Levels[level];
                vkDestroyImageView(m_Device, l.gridView, nullptr);
                vmaDestroyImage(m_Allocator, l.gridImage, l.gridAllocation);
                vkDestroyImageView(m_Device, l.occlusionView, nullptr);
                vmaDestroyImage(m_Allocator, l.occlusionImage, l.occlusionAllocation);
                l.gridView = VK_NULL_HANDLE; l.gridImage = VK_NULL_HANDLE; l.gridAllocation = VK_NULL_HANDLE;
                l.occlusionView = VK_NULL_HANDLE; l.occlusionImage = VK_NULL_HANDLE; l.occlusionAllocation = VK_NULL_HANDLE;
            });

            // One-time UNDEFINED -> GENERAL transitions (mirrors ClusterResolvePass::Init's own
            // one-shot pattern) -- stay GENERAL for each image's entire lifetime.
            VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, lvl.gridImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, lvl.occlusionImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            VkDescriptorSetAllocateInfo dsAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            dsAllocInfo.descriptorPool = m_DescriptorPool;
            dsAllocInfo.descriptorSetCount = 1;
            dsAllocInfo.pSetLayouts = &m_SetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &dsAllocInfo, &lvl.descriptorSet));

            VkDescriptorImageInfo gridStorageInfo{ VK_NULL_HANDLE, lvl.gridView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo occStorageInfo{ VK_NULL_HANDLE, lvl.occlusionView, VK_IMAGE_LAYOUT_GENERAL };

            VkWriteDescriptorSet writes[2]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[0].dstSet = lvl.descriptorSet; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].pImageInfo = &gridStorageInfo;
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            writes[1].dstSet = lvl.descriptorSet; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo = &occStorageInfo;
            vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

            VulkanUtils::WriteSharedGeometryBindings(m_Device, lvl.descriptorSet, 2, rtPass.GetTLASHandle(),
                surfaceCache.GetVertexBuffer(), surfaceCache.GetIndexBuffer(), rtPass.GetDrawRangeBuffer());

            lvl.originWorld = maths::vec3{ 0.0f, 0.0f, 0.0f };
            lvl.snappedCenterProbe[0] = lvl.snappedCenterProbe[1] = lvl.snappedCenterProbe[2] = 0;
            lvl.hasValidWindow = false;
        }
        // Re-registered so a LATER Shutdown() call (not just this InitImpl's own leading self-reinit
        // Shutdown()) still resets every level's POD state, matching the original single-level
        // Shutdown()'s own unconditional reset of it.
        RegisterResource([this] {
            for (uint32_t level = 0; level < kLevelCount; ++level) {
                m_Levels[level].originWorld = maths::vec3{};
                m_Levels[level].snappedCenterProbe[0] = m_Levels[level].snappedCenterProbe[1] = m_Levels[level].snappedCenterProbe[2] = 0;
                m_Levels[level].hasValidWindow = false;
            }
            m_FrameIndex = 0;
            m_PendingSlabs.clear();
        });

        // =====================================================================================
        // STEP 4 -- pipeline layout (set 0 above + set 1 mesh SDF trace scene + set 2 surface
        // cache sampling, both shared unmodified from traceContext, exactly like
        // SurfaceCacheGIInjectPass's own 3-set layout) + ONE compute pipeline, shared by every
        // level (only the bound set 0 differs per dispatch -- see RecordSlab()).
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

        m_FrameIndex = 0;
        m_PendingSlabs.clear();

        LOG_INFO(std::format("[WorldProbeGridPass] Initialized: {} levels x {}^3 grid, base spacing {} (toroidal streaming, F1 occlusion).",
            kLevelCount, kGridResolution, kProbeSpacing));
        return true;
    }

    void WorldProbeGridPass::SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler) {
        for (uint32_t level = 0; level < kLevelCount; ++level) {
            VkDescriptorImageInfo skyViewInfo{ skyViewLUTSampler, skyViewLUTView, VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.dstSet = m_Levels[level].descriptorSet; write.dstBinding = 6; write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo = &skyViewInfo;
            vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
        }
    }

    // Shutdown() is inherited from RenderPass<WorldProbeGridPass>: runs the RegisterResource()
    // cleanups above in reverse (POD state reset -> pipeline -> pipeline layout -> per-level
    // images/views -> sampler -> descriptor pool+layout), the same dependency-safe order the
    // hand-written Shutdown() used.

    void WorldProbeGridPass::EnqueueDirtyRegionsForLevel(uint32_t level, const maths::vec3& cameraPositionWorld) {
        ClipmapLevel& lvl = m_Levels[level];
        const int32_t res = static_cast<int32_t>(kGridResolution);
        const int32_t halfRes = res / 2;
        const float spacing = GetLevelSpacing(level);

        // Snap the camera's world position down to the nearest whole probe-index, per axis -- no
        // coarser "chunk" quantization on top of that (unlike GlobalSDFPass::kSnapChunkVoxels): each
        // level here is a single-resolution grid, not a further-subdivided clipmap, so there is no
        // equivalent reason to introduce one (see the class comment's own note on this).
        auto snapAxis = [&](float worldPos) -> int32_t {
            return static_cast<int32_t>(std::floor(worldPos / spacing));
            };
        int32_t newCenter[3] = {
            snapAxis(cameraPositionWorld.x), snapAxis(cameraPositionWorld.y), snapAxis(cameraPositionWorld.z)
        };

        if (!lvl.hasValidWindow) {
            DirtySlab slab{};
            slab.level = level;
            for (int axis = 0; axis < 3; ++axis) {
                slab.probeMin[axis] = newCenter[axis] - halfRes;
                slab.probeMax[axis] = newCenter[axis] + halfRes;
            }
            m_PendingSlabs.push_back(slab);
            lvl.snappedCenterProbe[0] = newCenter[0];
            lvl.snappedCenterProbe[1] = newCenter[1];
            lvl.snappedCenterProbe[2] = newCenter[2];
            lvl.hasValidWindow = true;
        } else {
            for (int axis = 0; axis < 3; ++axis) {
                int32_t delta = newCenter[axis] - lvl.snappedCenterProbe[axis];
                if (delta == 0) {
                    continue;
                }

                DirtySlab slab{};
                slab.level = level;
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
                    slab.probeMin[axis] = lvl.snappedCenterProbe[axis] + halfRes;
                    slab.probeMax[axis] = newCenter[axis] + halfRes;
                } else {
                    slab.probeMin[axis] = newCenter[axis] - halfRes;
                    slab.probeMax[axis] = lvl.snappedCenterProbe[axis] - halfRes;
                }
                m_PendingSlabs.push_back(slab);
            }

            lvl.snappedCenterProbe[0] = newCenter[0];
            lvl.snappedCenterProbe[1] = newCenter[1];
            lvl.snappedCenterProbe[2] = newCenter[2];
        }

        // GetGridOriginWorld(level)'s own contract: the world-space minimum corner of the window
        // currently centered on lvl.snappedCenterProbe, at THIS level's own spacing.
        float halfExtent = static_cast<float>(halfRes) * spacing;
        lvl.originWorld = maths::vec3{
            static_cast<float>(lvl.snappedCenterProbe[0]) * spacing - halfExtent,
            static_cast<float>(lvl.snappedCenterProbe[1]) * spacing - halfExtent,
            static_cast<float>(lvl.snappedCenterProbe[2]) * spacing - halfExtent
        };
    }

    void WorldProbeGridPass::RecordSlab(VkCommandBuffer cmd, const DirtySlab& slab,
        const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld) {
        const int32_t res = static_cast<int32_t>(kGridResolution);
        const float spacing = GetLevelSpacing(slab.level);

        std::vector<WrappedPiece> piecesX = SplitWrappedRange(slab.probeMin[0], slab.probeMax[0], res);
        std::vector<WrappedPiece> piecesY = SplitWrappedRange(slab.probeMin[1], slab.probeMax[1], res);
        std::vector<WrappedPiece> piecesZ = SplitWrappedRange(slab.probeMin[2], slab.probeMax[2], res);

        VkDescriptorSet sets[3] = { m_Levels[slab.level].descriptorSet, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 3, sets, 0, nullptr);

        for (const WrappedPiece& px : piecesX) {
            for (const WrappedPiece& py : piecesY) {
                for (const WrappedPiece& pz : piecesZ) {
                    WorldProbeInjectPushConstants pc{};
                    pc.dispatchMinTexelX = px.wrappedTexelStart;
                    pc.dispatchMinTexelY = py.wrappedTexelStart;
                    pc.dispatchMinTexelZ = pz.wrappedTexelStart;
                    pc.dispatchWorldMinX = static_cast<float>(px.worldVoxelStart) * spacing;
                    pc.dispatchWorldMinY = static_cast<float>(py.worldVoxelStart) * spacing;
                    pc.dispatchWorldMinZ = static_cast<float>(pz.worldVoxelStart) * spacing;
                    pc.levelSpacing = spacing;
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
        for (uint32_t level = 0; level < kLevelCount; ++level) {
            EnqueueDirtyRegionsForLevel(level, cameraPositionWorld);
        }

        // Pipeline never changes across the (possibly several) dispatches DrainAndRecordSlabs()
        // below records -- bound once here; the descriptor set DOES change per slab (level-
        // dependent), so RecordSlab() itself rebinds set 0 before each slab's own dispatches.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);

        DrainAndRecordSlabs(cmd, traceContext, traceMode, sunDirectionWorld);

        ++m_FrameIndex;
    }

}
