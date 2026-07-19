#include "renderer/passes/ScreenProbeGIPass.h"

#include <array>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte std140 mirror of ScreenProbeViewParamsUBO in every ScreenProbe*.comp shader.
        struct ScreenProbeViewParamsUBO {
            maths::mat4 invViewProj;
            maths::mat4 prevViewProj;
            float viewportWidth = 0.0f;
            float viewportHeight = 0.0f;
            float probeCountX = 0.0f;
            float probeCountY = 0.0f;
        };
        static_assert(sizeof(ScreenProbeViewParamsUBO) == 144,
            "ScreenProbeViewParamsUBO must match every ScreenProbe*.comp shader's UBO exactly (std140 layout)");

        // Phase 6 (UE5.8 parity roadmap): +tileSize/+dispatchMode -- see ScreenProbeTrace.comp's
        // own comment on the shared fine/coarse entry point these two fields select between.
        struct TracePushConstants {
            uint32_t probeCountX = 0, probeCountY = 0;
            uint32_t entityCount = 0;
            uint32_t traceMode = 0;
            uint32_t frameIndex = 0;
            uint32_t tileSize = 0;
            uint32_t dispatchMode = 0; // 0 = direct grid, 1 = indirect compacted list.
        };
        static_assert(sizeof(TracePushConstants) == 28, "TracePushConstants must match ScreenProbeTrace.comp's push_constant block exactly");

        // Phase 6: +tileSize/+dispatchMode, same meaning as TracePushConstants above -- see
        // ScreenProbeTemporal.comp's own comment on why this shader flattened to a 1D dispatch.
        struct TemporalPushConstants {
            uint32_t probeCountX = 0, probeCountY = 0;
            uint32_t tileSize = 0;
            uint32_t dispatchMode = 0;
        };
        static_assert(sizeof(TemporalPushConstants) == 16, "TemporalPushConstants must match ScreenProbeTemporal.comp's push_constant block exactly");

        // Phase 6: +coarseProbeCountX/Y -- see ScreenProbeGather.comp's own comment on why the
        // coarse grid's size can't be read from the shared ScreenProbeViewParamsUBO (which only
        // ever holds the FINE grid's size).
        // F9: no more Debug-only `debugViewMode` field -- see ScreenProbeGIPass.h's own RecordGather()
        // comment on why that visualization now lives entirely at the renderer::GICompositePass level.
        struct GatherPushConstants {
            uint32_t probeCountX = 0, probeCountY = 0;
            uint32_t coarseProbeCountX = 0, coarseProbeCountY = 0;
        };

        // Phase 6: ScreenProbeClassify.comp's own push-constant block. `fineTileSize` is
        // kProbeTileSize (runtime-configurable, see ScreenProbeGIPass.h) -- the shader needs it
        // explicitly to reconstruct each fine child tile's center pixel exactly the way
        // ScreenProbeTrace.comp's own pc.tileSize-driven reconstruction does (the two must agree
        // pixel-for-pixel, see this shader's own class comment).
        struct ClassifyPushConstants {
            uint32_t coarseProbeCountX = 0, coarseProbeCountY = 0;
            uint32_t fineProbeCountX = 0, fineProbeCountY = 0;
            uint32_t viewportWidth = 0, viewportHeight = 0;
            uint32_t fineTileSize = 0;
        };
        static_assert(sizeof(ClassifyPushConstants) == 28, "ClassifyPushConstants must match ScreenProbeClassify.comp's push_constant block exactly");

        // Phase 6: the FINE grid's own probe images need TRANSFER_SRC_BIT (RecordTrace()'s own
        // history-copy-forward, see class comment) on top of renderer::VulkanUtils::
        // CreateStorageSampledImage2D's usual STORAGE|SAMPLED|TRANSFER_DST set -- the coarse grid
        // never participates in that copy, so its own images use the shared helper unmodified.
        // Otherwise identical to that helper's own image-creation recipe.
        void CreateFineProbeImage(VmaAllocator allocator, VkDevice device, VkFormat format, VkExtent2D extent,
            VkImage& outImage, VmaAllocation& outAllocation, VkImageView& outView) {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = { extent.width, extent.height, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage, &outAllocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = outImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outView));
        }

    } // namespace

    bool ScreenProbeGIPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
        const SurfaceCacheRayTracingPass& rtPass, const ClusterResolvePass& resolvePass) {
        kProbeTileSize = config::lumen::SCREEN_PROBE_TILE_SIZE;
        kCoarseProbeTileSize = kProbeTileSize * 2u;
        kProbeRayCount = config::lumen::SCREEN_PROBE_RAY_COUNT;
        kTemporalAlpha = config::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;

        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;
        m_ProbeCountX = (renderExtent.width + kProbeTileSize - 1u) / kProbeTileSize;
        m_ProbeCountY = (renderExtent.height + kProbeTileSize - 1u) / kProbeTileSize;
        VkExtent2D probeExtent{ m_ProbeCountX, m_ProbeCountY };

        // Phase 6: the always-on COARSE grid -- see the class comment's hierarchy description.
        m_ProbeCoarseCountX = (renderExtent.width + kCoarseProbeTileSize - 1u) / kCoarseProbeTileSize;
        m_ProbeCoarseCountY = (renderExtent.height + kCoarseProbeTileSize - 1u) / kCoarseProbeTileSize;
        VkExtent2D probeCoarseExtent{ m_ProbeCoarseCountX, m_ProbeCoarseCountY };

        // =====================================================================================
        // STEP 0 (F9) -- this pass' own full-render-resolution indirect-only output image --
        // mirrors renderer::ScreenTracePass::Init's own output-image recipe exactly (STORAGE_BIT
        // for RecordGather's own imageStore, SAMPLED_BIT for renderer::GICompositePass's later
        // sampled read, TRANSFER_DST_BIT for parity with every other GI output image in this
        // codebase even though nothing currently vkCmdClear's this one), kept VK_IMAGE_LAYOUT_GENERAL
        // for its entire lifetime.
        // =====================================================================================
        {
            VkImageCreateInfo outputImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            outputImageInfo.imageType = VK_IMAGE_TYPE_2D;
            outputImageInfo.format = kOutputFormat;
            outputImageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
            outputImageInfo.mipLevels = 1;
            outputImageInfo.arrayLayers = 1;
            outputImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            outputImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            outputImageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            outputImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            outputImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo outputAllocInfo{};
            outputAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &outputImageInfo, &outputAllocInfo, &m_OutputImage, &m_OutputAllocation, nullptr));

            VkImageViewCreateInfo outputViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            outputViewInfo.image = m_OutputImage;
            outputViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            outputViewInfo.format = kOutputFormat;
            outputViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &outputViewInfo, nullptr, &m_OutputView));

            VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_OutputImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // =====================================================================================
        // STEP 1 -- Probe ping-pong images (2 fine slots + 2 coarse slots, 5 fields each) + shared
        // sampler + view-params UBO.
        // =====================================================================================
        for (ProbeSlot& slot : m_Slots) {
            CreateFineProbeImage(allocator, device, kSHFormat, probeExtent, slot.shRImage, slot.shRAllocation, slot.shRView);
            CreateFineProbeImage(allocator, device, kSHFormat, probeExtent, slot.shGImage, slot.shGAllocation, slot.shGView);
            CreateFineProbeImage(allocator, device, kSHFormat, probeExtent, slot.shBImage, slot.shBAllocation, slot.shBView);
            CreateFineProbeImage(allocator, device, kWorldPosFormat, probeExtent, slot.worldPosImage, slot.worldPosAllocation, slot.worldPosView);
            CreateFineProbeImage(allocator, device, kNormalFormat, probeExtent, slot.normalImage, slot.normalAllocation, slot.normalView);
        }
        for (ProbeSlot& slot : m_CoarseSlots) {
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kSHFormat, probeCoarseExtent, slot.shRImage, slot.shRAllocation, slot.shRView);
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kSHFormat, probeCoarseExtent, slot.shGImage, slot.shGAllocation, slot.shGView);
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kSHFormat, probeCoarseExtent, slot.shBImage, slot.shBAllocation, slot.shBView);
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kWorldPosFormat, probeCoarseExtent, slot.worldPosImage, slot.worldPosAllocation, slot.worldPosView);
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kNormalFormat, probeCoarseExtent, slot.normalImage, slot.normalAllocation, slot.normalView);
        }

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_ProbeSampler));

        m_ViewParamsBuffer.Create(allocator, sizeof(ScreenProbeViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // One-time UNDEFINED -> GENERAL transition + neutral-default clear, mirrors
        // renderer::SurfaceCachePass::Init's own STEP 3 pattern: SH channels start at zero (no
        // contribution yet), worldPos.a = 0 (never traced -- see kWorldPosFormat's own "validity"
        // comment), normal = oct-encoded +Y (an arbitrary but valid direction). Both fine and
        // coarse slots get this same neutral clear -- the coarse grid starts invalid too, becoming
        // valid after its first (always-on, unconditional) RecordTrace().
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkClearColorValue zeroClear{}; zeroClear.float32[0] = 0.0f; zeroClear.float32[1] = 0.0f; zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 0.0f;
            VkClearColorValue normalClear{}; normalClear.float32[0] = 0.5f; normalClear.float32[1] = 0.5f; normalClear.float32[2] = 0.0f; normalClear.float32[3] = 0.0f;

            for (ProbeSlot& slot : m_Slots) {
                struct { VkImage image; const VkClearColorValue* clear; } clears[5] = {
                    { slot.shRImage, &zeroClear }, { slot.shGImage, &zeroClear }, { slot.shBImage, &zeroClear },
                    { slot.worldPosImage, &zeroClear }, { slot.normalImage, &normalClear }
                };
                for (auto& entry : clears) {
                    VulkanUtils::ClearComputeImageToGeneral(cmd, entry.image, *entry.clear);
                }
            }
            for (ProbeSlot& slot : m_CoarseSlots) {
                struct { VkImage image; const VkClearColorValue* clear; } clears[5] = {
                    { slot.shRImage, &zeroClear }, { slot.shGImage, &zeroClear }, { slot.shBImage, &zeroClear },
                    { slot.worldPosImage, &zeroClear }, { slot.normalImage, &normalClear }
                };
                for (auto& entry : clears) {
                    VulkanUtils::ClearComputeImageToGeneral(cmd, entry.image, *entry.clear);
                }
            }
        });

        // =====================================================================================
        // STEP 1B (Phase 6) -- the compacted fine-tile list + its 2 indirect-dispatch-args buffers.
        // Created here (before STEP 2 needs to bind m_ActiveFineTileListBuffer into every Trace
        // descriptor set) -- see ScreenProbeGIPass.h's own comment on each buffer's exact shape.
        // =====================================================================================
        {
            // {uint count; uint coords[fineProbeCount];} -- worst case every fine tile active at
            // once, so this never needs runtime resizing (see the class comment's own sizing note).
            VkDeviceSize activeFineTileListSize = sizeof(uint32_t) + static_cast<VkDeviceSize>(m_ProbeCountX) * m_ProbeCountY * sizeof(uint32_t);
            m_ActiveFineTileListBuffer.Create(allocator, activeFineTileListSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_FineTraceDispatchArgsBuffer.Create(allocator, sizeof(uint32_t) * 3,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_FineTemporalDispatchArgsBuffer.Create(allocator, sizeof(uint32_t) * 3,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }

        // =====================================================================================
        // STEP 2 -- Trace pipeline: set 0 (13 bindings -- Phase 6 added binding 12, the compacted
        // fine-tile list -- 4 slot-indexed variants: m_TraceSet[0/1] fine, m_CoarseTraceSet[0/1]
        // coarse, all 4 sharing this ONE layout) + set 1 (mesh SDF trace scene) + set 2 (surface
        // cache sampling), both shared unmodified from traceContext.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[13]{};
            for (uint32_t b = 0; b <= 6; ++b) {
                bindings[b].binding = b;
                bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                bindings[b].descriptorCount = 1;
                bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[11] = { 11, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[12] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // Phase 6: g_ActiveFineTileList.

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 13;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_TraceSetLayout));

            VkDescriptorPoolSize poolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7 * 4 },
                { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 * 4 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * 4 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 4 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 4;
            poolInfo.poolSizeCount = 4;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_TraceDescriptorPool));

            VkDescriptorSetLayout setLayouts[4] = { m_TraceSetLayout, m_TraceSetLayout, m_TraceSetLayout, m_TraceSetLayout };
            VkDescriptorSet allocatedSets[4]{};
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_TraceDescriptorPool;
            setAllocInfo.descriptorSetCount = 4;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, allocatedSets));
            m_TraceSet[0] = allocatedSets[0]; m_TraceSet[1] = allocatedSets[1];
            m_CoarseTraceSet[0] = allocatedSets[2]; m_CoarseTraceSet[1] = allocatedSets[3];

            VkAccelerationStructureKHR tlasHandle = rtPass.GetTLASHandle();
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo activeFineTileListInfo{ m_ActiveFineTileListBuffer.Handle(), 0, m_ActiveFineTileListBuffer.Size() };

            auto writeTraceSet = [&](VkDescriptorSet set, const ProbeSlot& slot) {
                VkDescriptorImageInfo shRInfo{ VK_NULL_HANDLE, slot.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo shGInfo{ VK_NULL_HANDLE, slot.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo shBInfo{ VK_NULL_HANDLE, slot.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo worldPosInfo{ VK_NULL_HANDLE, slot.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo normalInfo{ VK_NULL_HANDLE, slot.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkWriteDescriptorSet writes[8]{};
                VkDescriptorImageInfo* imageInfos[7] = { &shRInfo, &shGInfo, &shBInfo, &worldPosInfo, &normalInfo, &gbufferNormalInfo, &gbufferDepthInfo };
                for (uint32_t b = 0; b <= 6; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, imageInfos[b], nullptr, nullptr };
                }
                writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 11, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 8, writes, 0, nullptr);

                VulkanUtils::WriteSharedGeometryBindings(m_Device, set, 7, tlasHandle,
                    surfaceCache.GetVertexBuffer(), surfaceCache.GetIndexBuffer(), rtPass.GetDrawRangeBuffer());

                VkWriteDescriptorSet activeListWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &activeFineTileListInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 1, &activeListWrite, 0, nullptr);
                };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                writeTraceSet(m_TraceSet[slotIndex], m_Slots[slotIndex]);
                writeTraceSet(m_CoarseTraceSet[slotIndex], m_CoarseSlots[slotIndex]);
            }

            VkDescriptorSetLayout traceSetLayouts[3] = { m_TraceSetLayout, traceContext.GetMeshSdfTraceSetLayout(), traceContext.GetSurfaceCacheSamplingSetLayout() };
            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TracePushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 3;
            pipelineLayoutInfo.pSetLayouts = traceSetLayouts;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_TracePipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ScreenProbeTrace.comp.spv");
            m_TracePipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_TracePipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 3 -- Temporal pipeline: set 0 (12 bindings -- Phase 6 added binding 11, the
        // compacted fine-tile list -- 4 slot-indexed variants: m_TemporalSet[0/1] fine,
        // m_CoarseTemporalSet[0/1] coarse, same one-layout-for-both convention as Trace above).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[12]{};
            for (uint32_t b = 0; b <= 4; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            for (uint32_t b = 5; b <= 9; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[10] = { 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // Phase 6: g_ActiveFineTileList.

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 12;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_TemporalSetLayout));

            VkDescriptorPoolSize poolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 * 4 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 * 4 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 4 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 * 4 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 4;
            poolInfo.poolSizeCount = 4;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_TemporalDescriptorPool));

            VkDescriptorSetLayout setLayouts[4] = { m_TemporalSetLayout, m_TemporalSetLayout, m_TemporalSetLayout, m_TemporalSetLayout };
            VkDescriptorSet allocatedSets[4]{};
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_TemporalDescriptorPool;
            setAllocInfo.descriptorSetCount = 4;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, allocatedSets));
            m_TemporalSet[0] = allocatedSets[0]; m_TemporalSet[1] = allocatedSets[1];
            m_CoarseTemporalSet[0] = allocatedSets[2]; m_CoarseTemporalSet[1] = allocatedSets[3];

            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorBufferInfo activeFineTileListInfo{ m_ActiveFineTileListBuffer.Handle(), 0, m_ActiveFineTileListBuffer.Size() };

            auto writeTemporalSet = [&](VkDescriptorSet set, const ProbeSlot& current, const ProbeSlot& history) {
                VkDescriptorImageInfo curSHR{ VK_NULL_HANDLE, current.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHG{ VK_NULL_HANDLE, current.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHB{ VK_NULL_HANDLE, current.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curWorldPos{ VK_NULL_HANDLE, current.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curNormal{ VK_NULL_HANDLE, current.normalView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histSHR{ m_ProbeSampler, history.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histSHG{ m_ProbeSampler, history.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histSHB{ m_ProbeSampler, history.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histWorldPos{ m_ProbeSampler, history.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histNormal{ m_ProbeSampler, history.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkDescriptorImageInfo* storageInfos[5] = { &curSHR, &curSHG, &curSHB, &curWorldPos, &curNormal };
                VkDescriptorImageInfo* sampledInfos[5] = { &histSHR, &histSHG, &histSHB, &histWorldPos, &histNormal };

                VkWriteDescriptorSet writes[12]{};
                for (uint32_t b = 0; b <= 4; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b], nullptr, nullptr };
                }
                for (uint32_t b = 5; b <= 9; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, b, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampledInfos[b - 5], nullptr, nullptr };
                }
                writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &activeFineTileListInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 12, writes, 0, nullptr);
                };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                writeTemporalSet(m_TemporalSet[slotIndex], m_Slots[slotIndex], m_Slots[1 - slotIndex]);
                writeTemporalSet(m_CoarseTemporalSet[slotIndex], m_CoarseSlots[slotIndex], m_CoarseSlots[1 - slotIndex]);
            }

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TemporalPushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_TemporalSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_TemporalPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ScreenProbeTemporal.comp.spv");
            m_TemporalPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_TemporalPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 4 -- Gather pipeline: set 0 (15 bindings, 2 slot-indexed variants -- Phase 6 added
        // bindings 5-9, the COARSE grid's own current-slot images, read within the SAME invocation
        // as the fine grid's, so no separate coarse gather set is needed -- see
        // ScreenProbeGather.comp's own multi-tier fallback).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[15]{};
            for (uint32_t b = 0; b <= 9; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            for (uint32_t b = 10; b <= 13; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[14] = { 14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 15;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_GatherSetLayout));

            VkDescriptorPoolSize poolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 * 2 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 3;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_GatherDescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_GatherSetLayout, m_GatherSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_GatherDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_GatherSet));

            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferAlbedoInfo{ VK_NULL_HANDLE, resolvePass.GetOutputAlbedoView(), VK_IMAGE_LAYOUT_GENERAL };
            // F9: this pass' OWN output image (was resolvePass.GetOutputColorView() pre-deletion,
            // read-modify-written directly -- see ScreenProbeGather.comp's own header comment for
            // why this is now a plain write into a dedicated image instead).
            VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, m_OutputView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                const ProbeSlot& current = m_Slots[slotIndex];
                const ProbeSlot& coarseCurrent = m_CoarseSlots[slotIndex];
                VkDescriptorImageInfo curSHR{ m_ProbeSampler, current.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHG{ m_ProbeSampler, current.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHB{ m_ProbeSampler, current.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curWorldPos{ m_ProbeSampler, current.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curNormal{ m_ProbeSampler, current.normalView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curCoarseSHR{ m_ProbeSampler, coarseCurrent.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curCoarseSHG{ m_ProbeSampler, coarseCurrent.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curCoarseSHB{ m_ProbeSampler, coarseCurrent.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curCoarseWorldPos{ m_ProbeSampler, coarseCurrent.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curCoarseNormal{ m_ProbeSampler, coarseCurrent.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkDescriptorImageInfo* sampledInfos[10] = {
                    &curSHR, &curSHG, &curSHB, &curWorldPos, &curNormal,
                    &curCoarseSHR, &curCoarseSHG, &curCoarseSHB, &curCoarseWorldPos, &curCoarseNormal
                };
                VkDescriptorImageInfo* storageInfos[4] = { &gbufferNormalInfo, &gbufferDepthInfo, &gbufferAlbedoInfo, &outputColorInfo };

                VkWriteDescriptorSet writes[15]{};
                for (uint32_t b = 0; b <= 9; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampledInfos[b], nullptr, nullptr };
                }
                for (uint32_t b = 10; b <= 13; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b - 10], nullptr, nullptr };
                }
                writes[14] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], 14, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 15, writes, 0, nullptr);
            }

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GatherPushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_GatherSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_GatherPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ScreenProbeGather.comp.spv");
            m_GatherPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_GatherPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 5 (Phase 6) -- BuildDispatchIndirectArgs.comp instance, reused twice per RecordTrace()/
        // RecordTemporal() pair (once for the fine Trace dispatch, once for the fine Temporal
        // dispatch -- different workgroupSize/perElementMultiplier each time, see RecordTrace()'s
        // own body) -- shared layout/pipeline helper, same convention renderer::
        // ClusterOcclusionCullingPass's own RecordBuildLateDispatchArgs uses for this exact shader.
        // Set 0 binding 0 aliases m_ActiveFineTileListBuffer's leading `count` word only
        // (SourceCountSSBO's own binding-0 layout is `{uint count;}`, which sits at that exact
        // same offset 0 in m_ActiveFineTileListBuffer).
        // =====================================================================================
        {
            m_BuildArgsSetLayout = VulkanPipeline::CreateBuildDispatchIndirectArgsSetLayout(m_Device);

            VkDescriptorPoolSize poolSizes[1] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 } }; // 2 bindings x 2 sets.
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_BuildArgsDescriptorPool));

            VkDescriptorSetLayout buildArgsSetLayouts[2] = { m_BuildArgsSetLayout, m_BuildArgsSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_BuildArgsDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = buildArgsSetLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_BuildArgsSet));

            VkDescriptorBufferInfo sourceCountInfo{ m_ActiveFineTileListBuffer.Handle(), 0, sizeof(uint32_t) };
            VkDescriptorBufferInfo traceArgsInfo{ m_FineTraceDispatchArgsBuffer.Handle(), 0, m_FineTraceDispatchArgsBuffer.Size() };
            VkDescriptorBufferInfo temporalArgsInfo{ m_FineTemporalDispatchArgsBuffer.Handle(), 0, m_FineTemporalDispatchArgsBuffer.Size() };

            VkWriteDescriptorSet traceWrites[2]{};
            traceWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsSet[0], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sourceCountInfo, nullptr };
            traceWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsSet[0], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &traceArgsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, traceWrites, 0, nullptr);

            VkWriteDescriptorSet temporalWrites[2]{};
            temporalWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsSet[1], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sourceCountInfo, nullptr };
            temporalWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsSet[1], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &temporalArgsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, temporalWrites, 0, nullptr);

            VulkanPipeline::CreateBuildDispatchIndirectArgsPipeline(m_Device, m_BuildArgsSetLayout, m_BuildArgsPipelineLayout, m_BuildArgsPipeline);
        }

        // =====================================================================================
        // STEP 6 (Phase 6) -- Classify pipeline: set 0 only (GBuffer normal/depth readonly +
        // view-params UBO + the active-list SSBO, read-write) -- see ScreenProbeClassify.comp's
        // own class comment. One set only (not slot-indexed): this pass never touches any
        // ping-pong probe image, only the shared GBuffer + the compacted list it produces.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[4]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 4;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ClassifySetLayout));

            VkDescriptorPoolSize poolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 3;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_ClassifyDescriptorPool));

            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_ClassifyDescriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_ClassifySetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_ClassifySet));

            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorBufferInfo activeFineTileListInfo{ m_ActiveFineTileListBuffer.Handle(), 0, m_ActiveFineTileListBuffer.Size() };

            VkWriteDescriptorSet writes[4]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferNormalInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferDepthInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &activeFineTileListInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClassifyPushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ClassifySetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ClassifyPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ScreenProbeClassify.comp.spv");
            m_ClassifyPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ClassifyPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        m_CurrentSlotIndex = 0;
        LOG_INFO(std::format("[ScreenProbeGIPass] Initialized: {} x {} probes ({} x {} coarse).",
            m_ProbeCountX, m_ProbeCountY, m_ProbeCoarseCountX, m_ProbeCoarseCountY));
        return true;
    }

    void ScreenProbeGIPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_ClassifyPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ClassifyPipeline, nullptr);
            if (m_ClassifyPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ClassifyPipelineLayout, nullptr);
            if (m_ClassifyDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_ClassifyDescriptorPool, nullptr);
            if (m_ClassifySetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ClassifySetLayout, nullptr);

            if (m_BuildArgsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_BuildArgsPipeline, nullptr);
            if (m_BuildArgsPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_BuildArgsPipelineLayout, nullptr);
            if (m_BuildArgsDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_BuildArgsDescriptorPool, nullptr);
            if (m_BuildArgsSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_BuildArgsSetLayout, nullptr);

            if (m_GatherPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_GatherPipeline, nullptr);
            if (m_GatherPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_GatherPipelineLayout, nullptr);
            if (m_GatherDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_GatherDescriptorPool, nullptr);
            if (m_GatherSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_GatherSetLayout, nullptr);

            if (m_TemporalPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_TemporalPipeline, nullptr);
            if (m_TemporalPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_TemporalPipelineLayout, nullptr);
            if (m_TemporalDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_TemporalDescriptorPool, nullptr);
            if (m_TemporalSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_TemporalSetLayout, nullptr);

            if (m_TracePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_TracePipeline, nullptr);
            if (m_TracePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_TracePipelineLayout, nullptr);
            if (m_TraceDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_TraceDescriptorPool, nullptr);
            if (m_TraceSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_TraceSetLayout, nullptr);

            if (m_ProbeSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ProbeSampler, nullptr);

            // F9: this pass' own output image.
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);

            for (ProbeSlot& slot : m_Slots) {
                if (slot.shRView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shRView, nullptr);
                if (slot.shGView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shGView, nullptr);
                if (slot.shBView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shBView, nullptr);
                if (slot.worldPosView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.worldPosView, nullptr);
                if (slot.normalView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.normalView, nullptr);
            }
            for (ProbeSlot& slot : m_CoarseSlots) {
                if (slot.shRView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shRView, nullptr);
                if (slot.shGView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shGView, nullptr);
                if (slot.shBView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shBView, nullptr);
                if (slot.worldPosView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.worldPosView, nullptr);
                if (slot.normalView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.normalView, nullptr);
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            for (ProbeSlot& slot : m_Slots) {
                vmaDestroyImage(m_Allocator, slot.shRImage, slot.shRAllocation);
                vmaDestroyImage(m_Allocator, slot.shGImage, slot.shGAllocation);
                vmaDestroyImage(m_Allocator, slot.shBImage, slot.shBAllocation);
                vmaDestroyImage(m_Allocator, slot.worldPosImage, slot.worldPosAllocation);
                vmaDestroyImage(m_Allocator, slot.normalImage, slot.normalAllocation);
            }
            for (ProbeSlot& slot : m_CoarseSlots) {
                vmaDestroyImage(m_Allocator, slot.shRImage, slot.shRAllocation);
                vmaDestroyImage(m_Allocator, slot.shGImage, slot.shGAllocation);
                vmaDestroyImage(m_Allocator, slot.shBImage, slot.shBAllocation);
                vmaDestroyImage(m_Allocator, slot.worldPosImage, slot.worldPosAllocation);
                vmaDestroyImage(m_Allocator, slot.normalImage, slot.normalAllocation);
            }
            // F9: this pass' own output image.
            vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
        }
        m_ViewParamsBuffer.Destroy();
        m_ActiveFineTileListBuffer.Destroy();
        m_FineTraceDispatchArgsBuffer.Destroy();
        m_FineTemporalDispatchArgsBuffer.Destroy();

        m_ClassifyPipeline = VK_NULL_HANDLE; m_ClassifyPipelineLayout = VK_NULL_HANDLE; m_ClassifyDescriptorPool = VK_NULL_HANDLE; m_ClassifySetLayout = VK_NULL_HANDLE;
        m_ClassifySet = VK_NULL_HANDLE;
        m_BuildArgsPipeline = VK_NULL_HANDLE; m_BuildArgsPipelineLayout = VK_NULL_HANDLE; m_BuildArgsDescriptorPool = VK_NULL_HANDLE; m_BuildArgsSetLayout = VK_NULL_HANDLE;
        m_BuildArgsSet[0] = VK_NULL_HANDLE; m_BuildArgsSet[1] = VK_NULL_HANDLE;
        m_GatherPipeline = VK_NULL_HANDLE; m_GatherPipelineLayout = VK_NULL_HANDLE; m_GatherDescriptorPool = VK_NULL_HANDLE; m_GatherSetLayout = VK_NULL_HANDLE;
        m_GatherSet[0] = VK_NULL_HANDLE; m_GatherSet[1] = VK_NULL_HANDLE;
        m_TemporalPipeline = VK_NULL_HANDLE; m_TemporalPipelineLayout = VK_NULL_HANDLE; m_TemporalDescriptorPool = VK_NULL_HANDLE; m_TemporalSetLayout = VK_NULL_HANDLE;
        m_TemporalSet[0] = VK_NULL_HANDLE; m_TemporalSet[1] = VK_NULL_HANDLE;
        m_CoarseTemporalSet[0] = VK_NULL_HANDLE; m_CoarseTemporalSet[1] = VK_NULL_HANDLE;
        m_TracePipeline = VK_NULL_HANDLE; m_TracePipelineLayout = VK_NULL_HANDLE; m_TraceDescriptorPool = VK_NULL_HANDLE; m_TraceSetLayout = VK_NULL_HANDLE;
        m_TraceSet[0] = VK_NULL_HANDLE; m_TraceSet[1] = VK_NULL_HANDLE;
        m_CoarseTraceSet[0] = VK_NULL_HANDLE; m_CoarseTraceSet[1] = VK_NULL_HANDLE;
        m_ProbeSampler = VK_NULL_HANDLE;
        m_OutputView = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE;
        m_OutputAllocation = VK_NULL_HANDLE;
        m_Slots[0] = ProbeSlot{};
        m_Slots[1] = ProbeSlot{};
        m_CoarseSlots[0] = ProbeSlot{};
        m_CoarseSlots[1] = ProbeSlot{};
        m_CurrentSlotIndex = 0;
        m_ProbeCountX = 0;
        m_ProbeCountY = 0;
        m_ProbeCoarseCountX = 0;
        m_ProbeCoarseCountY = 0;
        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ScreenProbeGIPass::RecordUpdateViewParams(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj) {
        ScreenProbeViewParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.prevViewProj = prevViewProj;
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        ubo.probeCountX = static_cast<float>(m_ProbeCountX);
        ubo.probeCountY = static_cast<float>(m_ProbeCountY);
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);
    }

    void ScreenProbeGIPass::RecordTrace(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
        uint32_t entityCount, uint32_t traceMode, uint32_t frameIndex) {
        // This frame writes into whichever slot held the PREVIOUS frame's history -- see this
        // method's own header comment.
        m_CurrentSlotIndex = 1 - m_CurrentSlotIndex;
        uint32_t historySlotIndex = 1 - m_CurrentSlotIndex;

        // =====================================================================================
        // Phase 6 -- "report" copy-forward (FINE slots only): the new current slot starts this
        // frame as a bit-exact copy of the history slot, so any fine tile ScreenProbeClassify.comp
        // does NOT select below still carries forward its exact last-known value -- see this
        // class' own header comment for why this is required for correctness, not an optimization.
        // =====================================================================================
        {
            VkImage historyImages[5] = {
                m_Slots[historySlotIndex].shRImage, m_Slots[historySlotIndex].shGImage, m_Slots[historySlotIndex].shBImage,
                m_Slots[historySlotIndex].worldPosImage, m_Slots[historySlotIndex].normalImage
            };
            VkImage currentImages[5] = {
                m_Slots[m_CurrentSlotIndex].shRImage, m_Slots[m_CurrentSlotIndex].shGImage, m_Slots[m_CurrentSlotIndex].shBImage,
                m_Slots[m_CurrentSlotIndex].worldPosImage, m_Slots[m_CurrentSlotIndex].normalImage
            };

            VkImageMemoryBarrier2 preCopyBarriers[10]{};
            for (uint32_t i = 0; i < 5; ++i) {
                preCopyBarriers[i] = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                preCopyBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                preCopyBarriers[i].srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                preCopyBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                preCopyBarriers[i].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                preCopyBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                preCopyBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                preCopyBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                preCopyBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                preCopyBarriers[i].image = historyImages[i];
                preCopyBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }
            for (uint32_t i = 0; i < 5; ++i) {
                preCopyBarriers[5 + i] = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                // Conservative (read OR write): this slot was "history" (read-only, by the
                // previous frame's own Temporal reprojection) as of the start of this frame, but a
                // broader src mask costs nothing and keeps this barrier correct even if that
                // assumption ever changes.
                preCopyBarriers[5 + i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                preCopyBarriers[5 + i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                preCopyBarriers[5 + i].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                preCopyBarriers[5 + i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                preCopyBarriers[5 + i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                preCopyBarriers[5 + i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                preCopyBarriers[5 + i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                preCopyBarriers[5 + i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                preCopyBarriers[5 + i].image = currentImages[i];
                preCopyBarriers[5 + i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }
            VkDependencyInfo preCopyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            preCopyDep.imageMemoryBarrierCount = 10;
            preCopyDep.pImageMemoryBarriers = preCopyBarriers;
            vkCmdPipelineBarrier2(cmd, &preCopyDep);

            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.extent = { m_ProbeCountX, m_ProbeCountY, 1 };
            for (uint32_t i = 0; i < 5; ++i) {
                vkCmdCopyImage(cmd, historyImages[i], VK_IMAGE_LAYOUT_GENERAL, currentImages[i], VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
            }

            VkImageMemoryBarrier2 postCopyBarriers[5]{};
            for (uint32_t i = 0; i < 5; ++i) {
                postCopyBarriers[i] = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                postCopyBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                postCopyBarriers[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                postCopyBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                postCopyBarriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                postCopyBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                postCopyBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                postCopyBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                postCopyBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                postCopyBarriers[i].image = currentImages[i];
                postCopyBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }
            VkDependencyInfo postCopyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            postCopyDep.imageMemoryBarrierCount = 5;
            postCopyDep.pImageMemoryBarriers = postCopyBarriers;
            vkCmdPipelineBarrier2(cmd, &postCopyDep);
        }

        // =====================================================================================
        // Phase 6 -- classify (always direct, whole coarse grid) -> compact fine-tile list ->
        // build this frame's 2 indirect dispatch args (Trace's own below, Temporal's own for its
        // later RecordTemporal() call).
        // =====================================================================================
        vkCmdFillBuffer(cmd, m_ActiveFineTileListBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ClassifyPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ClassifyPipelineLayout, 0, 1, &m_ClassifySet, 0, nullptr);
        ClassifyPushConstants classifyPC{};
        classifyPC.coarseProbeCountX = m_ProbeCoarseCountX;
        classifyPC.coarseProbeCountY = m_ProbeCoarseCountY;
        classifyPC.fineProbeCountX = m_ProbeCountX;
        classifyPC.fineProbeCountY = m_ProbeCountY;
        classifyPC.viewportWidth = m_RenderExtent.width;
        classifyPC.viewportHeight = m_RenderExtent.height;
        classifyPC.fineTileSize = kProbeTileSize;
        vkCmdPushConstants(cmd, m_ClassifyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(classifyPC), &classifyPC);
        uint32_t classifyGroupsX = (m_ProbeCoarseCountX + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        uint32_t classifyGroupsY = (m_ProbeCoarseCountY + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        vkCmdDispatch(cmd, classifyGroupsX, classifyGroupsY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipeline);
        // Trace's own args: one full workgroup PER active tile (workgroupSize=1 -- see
        // ScreenProbeTrace.comp's own indirect mode, which reads one tile per gl_WorkGroupID.x, no
        // flattening needed).
        uint32_t traceArgsPushConstants[2] = { 1u, 1u };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipelineLayout, 0, 1, &m_BuildArgsSet[0], 0, nullptr);
        vkCmdPushConstants(cmd, m_BuildArgsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(traceArgsPushConstants), traceArgsPushConstants);
        vkCmdDispatch(cmd, 1, 1, 1);
        // Temporal's own args: ceil(count/64) workgroups (workgroupSize=64 -- matches
        // ScreenProbeTemporal.comp's own local_size_x, see that shader's own comment for why it's
        // now a flattened 1D dispatch).
        uint32_t temporalArgsPushConstants[2] = { 64u, 1u };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipelineLayout, 0, 1, &m_BuildArgsSet[1], 0, nullptr);
        vkCmdPushConstants(cmd, m_BuildArgsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(temporalArgsPushConstants), temporalArgsPushConstants);
        vkCmdDispatch(cmd, 1, 1, 1);

        // The dispatch args must be visible to this method's OWN vkCmdDispatchIndirect below
        // (Trace's) and RecordTemporal()'s later one -- classified the same as
        // vkCmdDrawIndirect's indirect-buffer read (mirrors
        // ClusterOcclusionCullingPass::RecordBuildLateDispatchArgs's own identical barrier).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

        // =====================================================================================
        // Coarse Trace -- always direct, full grid, mode=0, tileSize=kCoarseProbeTileSize.
        // =====================================================================================
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TracePipeline);
        VkDescriptorSet coarseSets[3] = { m_CoarseTraceSet[m_CurrentSlotIndex], traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TracePipelineLayout, 0, 3, coarseSets, 0, nullptr);
        TracePushConstants coarsePC{};
        coarsePC.probeCountX = m_ProbeCoarseCountX;
        coarsePC.probeCountY = m_ProbeCoarseCountY;
        coarsePC.entityCount = entityCount;
        coarsePC.traceMode = traceMode;
        coarsePC.frameIndex = frameIndex;
        coarsePC.tileSize = kCoarseProbeTileSize;
        coarsePC.dispatchMode = 0;
        vkCmdPushConstants(cmd, m_TracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(coarsePC), &coarsePC);
        vkCmdDispatch(cmd, m_ProbeCoarseCountX, m_ProbeCoarseCountY, 1);

        // =====================================================================================
        // Fine Trace -- indirect, only the tiles ScreenProbeClassify.comp flagged, mode=1,
        // tileSize=kProbeTileSize.
        // =====================================================================================
        VkDescriptorSet fineSets[3] = { m_TraceSet[m_CurrentSlotIndex], traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TracePipelineLayout, 0, 3, fineSets, 0, nullptr);
        TracePushConstants finePC{};
        finePC.probeCountX = m_ProbeCountX;
        finePC.probeCountY = m_ProbeCountY;
        finePC.entityCount = entityCount;
        finePC.traceMode = traceMode;
        finePC.frameIndex = frameIndex;
        finePC.tileSize = kProbeTileSize;
        finePC.dispatchMode = 1;
        vkCmdPushConstants(cmd, m_TracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(finePC), &finePC);
        vkCmdDispatchIndirect(cmd, m_FineTraceDispatchArgsBuffer.Handle(), 0);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    void ScreenProbeGIPass::RecordTemporal(VkCommandBuffer cmd) {
        // Coarse Temporal -- always direct, flattened 1D over the full coarse grid, mode=0.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TemporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TemporalPipelineLayout, 0, 1, &m_CoarseTemporalSet[m_CurrentSlotIndex], 0, nullptr);
        TemporalPushConstants coarsePC{};
        coarsePC.probeCountX = m_ProbeCoarseCountX;
        coarsePC.probeCountY = m_ProbeCoarseCountY;
        coarsePC.tileSize = kCoarseProbeTileSize;
        coarsePC.dispatchMode = 0;
        vkCmdPushConstants(cmd, m_TemporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(coarsePC), &coarsePC);
        uint32_t coarseTotal = m_ProbeCoarseCountX * m_ProbeCoarseCountY;
        uint32_t coarseGroupCount = (coarseTotal + 63u) / 64u; // local_size_x = 64.
        vkCmdDispatch(cmd, coarseGroupCount, 1, 1);

        // Fine Temporal -- indirect, only the tiles this frame's RecordTrace() selected, mode=1.
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TemporalPipelineLayout, 0, 1, &m_TemporalSet[m_CurrentSlotIndex], 0, nullptr);
        TemporalPushConstants finePC{};
        finePC.probeCountX = m_ProbeCountX;
        finePC.probeCountY = m_ProbeCountY;
        finePC.tileSize = kProbeTileSize;
        finePC.dispatchMode = 1;
        vkCmdPushConstants(cmd, m_TemporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(finePC), &finePC);
        vkCmdDispatchIndirect(cmd, m_FineTemporalDispatchArgsBuffer.Handle(), 0);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void ScreenProbeGIPass::RecordGather(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GatherPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GatherPipelineLayout, 0, 1, &m_GatherSet[m_CurrentSlotIndex], 0, nullptr);

        GatherPushConstants pc{};
        pc.probeCountX = m_ProbeCountX;
        pc.probeCountY = m_ProbeCountY;
        pc.coarseProbeCountX = m_ProbeCoarseCountX;
        pc.coarseProbeCountY = m_ProbeCoarseCountY;
        vkCmdPushConstants(cmd, m_GatherPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupCountX = (m_RenderExtent.width + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    }

}
