#include "renderer/passes/ReflectionPass.h"

#include <format>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/Logger.h"
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte std140 mirror of ReflectionViewParamsUBO in every Reflection*.comp shader.
        struct ReflectionViewParamsUBO {
            maths::mat4 invViewProj;
            maths::mat4 prevViewProj;
            float cameraPositionWorldX = 0.0f, cameraPositionWorldY = 0.0f, cameraPositionWorldZ = 0.0f;
            float _pad0 = 0.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(ReflectionViewParamsUBO) == 160,
            "ReflectionViewParamsUBO must match every Reflection*.comp shader's UBO exactly (std140 layout)");

        struct TracePushConstants {
            uint32_t entityCount = 0;
            uint32_t traceMode = 0;
            uint32_t frameIndex = 0;
        };
        static_assert(sizeof(TracePushConstants) == 12, "TracePushConstants must match ReflectionTrace.comp's push_constant block exactly");

    } // namespace

    bool ReflectionPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
        const SurfaceCacheRayTracingPass& rtPass, const ClusterResolvePass& resolvePass) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        // =====================================================================================
        // STEP 1 -- Ping-pong images (2 slots x 3 fields, full resolution) + shared sampler + UBO.
        // =====================================================================================
        for (ReflectionSlot& slot : m_Slots) {
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kRadianceFormat, renderExtent, slot.radianceImage, slot.radianceAllocation, slot.radianceView);
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kWorldPosFormat, renderExtent, slot.worldPosImage, slot.worldPosAllocation, slot.worldPosView);
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kNormalFormat, renderExtent, slot.normalImage, slot.normalAllocation, slot.normalView);
        }

        // Phase PP4: single fixed hit-mask image (not part of the ping-pong slots above -- see
        // GetHitMaskView()'s own comment for why).
        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kHitMaskFormat, renderExtent, m_HitMaskImage, m_HitMaskAllocation, m_HitMaskView);

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
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_ReflectionSampler));

        m_ViewParamsBuffer.Create(allocator, sizeof(ReflectionViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // One-time UNDEFINED -> GENERAL transition + neutral-default clear, mirrors
        // ScreenProbeGIPass::Init's own STEP 1 pattern: radiance/worldPos start at a=0 (never
        // traced -- see kWorldPosFormat's own "validity" comment), normal = oct-encoded +Y (an
        // arbitrary but valid direction).
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkClearColorValue zeroClear{}; zeroClear.float32[0] = 0.0f; zeroClear.float32[1] = 0.0f; zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 0.0f;
            VkClearColorValue normalClear{}; normalClear.float32[0] = 0.5f; normalClear.float32[1] = 0.5f; normalClear.float32[2] = 0.0f; normalClear.float32[3] = 0.0f;
            for (ReflectionSlot& slot : m_Slots) {
                struct { VkImage image; const VkClearColorValue* clear; } clears[3] = {
                    { slot.radianceImage, &zeroClear }, { slot.worldPosImage, &zeroClear }, { slot.normalImage, &normalClear }
                };
                for (auto& entry : clears) {
                    VulkanUtils::ClearComputeImageToGeneral(cmd, entry.image, *entry.clear);
                }
            }
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_HitMaskImage, zeroClear);
            });

        // =====================================================================================
        // STEP 2 -- Trace pipeline: set 0 (11 bindings, 2 slot-indexed variants) + set 1 (mesh SDF
        // trace scene) + set 2 (surface cache sampling), both shared unmodified from traceContext.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[12]{};
            for (uint32_t b : { 0u, 1u, 2u, 3u, 9u, 10u, 11u }) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[8] = { 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 12;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_TraceSetLayout));

            VkDescriptorPoolSize poolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7 * 2 },
                { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 * 2 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 4;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_TraceDescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_TraceSetLayout, m_TraceSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_TraceDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_TraceSet));

            VkAccelerationStructureKHR tlasHandle = rtPass.GetTLASHandle();
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferRoughnessMetallicInfo{ VK_NULL_HANDLE, resolvePass.GetOutputRoughnessMetallicView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo hitMaskInfo{ VK_NULL_HANDLE, m_HitMaskView, VK_IMAGE_LAYOUT_GENERAL };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                const ReflectionSlot& slot = m_Slots[slotIndex];
                VkDescriptorImageInfo radianceInfo{ VK_NULL_HANDLE, slot.radianceView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo worldPosInfo{ VK_NULL_HANDLE, slot.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo normalInfo{ VK_NULL_HANDLE, slot.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkWriteDescriptorSet writes[8]{};
                writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &radianceInfo, nullptr, nullptr };
                writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferNormalInfo, nullptr, nullptr };
                writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferDepthInfo, nullptr, nullptr };
                writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferRoughnessMetallicInfo, nullptr, nullptr };
                writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 8, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &worldPosInfo, nullptr, nullptr };
                writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normalInfo, nullptr, nullptr };
                writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hitMaskInfo, nullptr, nullptr };
                vkUpdateDescriptorSets(m_Device, 8, writes, 0, nullptr);

                VulkanUtils::WriteSharedGeometryBindings(m_Device, m_TraceSet[slotIndex], 4, tlasHandle,
                    surfaceCache.GetVertexBuffer(), surfaceCache.GetIndexBuffer(), rtPass.GetDrawRangeBuffer());
            }

            VkDescriptorSetLayout traceSetLayouts[3] = { m_TraceSetLayout, traceContext.GetMeshSdfTraceSetLayout(), traceContext.GetSurfaceCacheSamplingSetLayout() };
            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TracePushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 3;
            pipelineLayoutInfo.pSetLayouts = traceSetLayouts;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_TracePipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ReflectionTrace.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_TracePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_TracePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 3 -- Temporal pipeline: set 0 (8 bindings, 2 slot-indexed variants).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[8]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 8;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_TemporalSetLayout));

            VkDescriptorPoolSize poolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * 2 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 3;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_TemporalDescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_TemporalSetLayout, m_TemporalSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_TemporalDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_TemporalSet));

            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorImageInfo gbufferRoughnessMetallicInfo{ VK_NULL_HANDLE, resolvePass.GetOutputRoughnessMetallicView(), VK_IMAGE_LAYOUT_GENERAL };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                const ReflectionSlot& current = m_Slots[slotIndex];
                const ReflectionSlot& history = m_Slots[1 - slotIndex];

                VkDescriptorImageInfo curRadiance{ VK_NULL_HANDLE, current.radianceView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curWorldPos{ VK_NULL_HANDLE, current.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curNormal{ VK_NULL_HANDLE, current.normalView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histRadiance{ m_ReflectionSampler, history.radianceView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histWorldPos{ m_ReflectionSampler, history.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histNormal{ m_ReflectionSampler, history.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkWriteDescriptorSet writes[8]{};
                writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &curRadiance, nullptr, nullptr };
                writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &curWorldPos, nullptr, nullptr };
                writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &curNormal, nullptr, nullptr };
                writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &histRadiance, nullptr, nullptr };
                writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &histWorldPos, nullptr, nullptr };
                writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &histNormal, nullptr, nullptr };
                writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferRoughnessMetallicInfo, nullptr, nullptr };
                writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 7, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 8, writes, 0, nullptr);
            }

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_TemporalSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_TemporalPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ReflectionTemporal.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_TemporalPipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_TemporalPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 4 -- Gather pipeline: set 0 (9 bindings, 2 slot-indexed variants). Bindings 7/8
        // (Substrate integration): this pixel's materialID GBuffer image + the material params
        // SSBO renderer::ClusterResolvePass already filled -- see ReflectionGather.comp's own
        // binding comments. Both are the SAME resource for both slot variants, unlike the
        // per-slot curRadiance binding.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[9]{};
            for (uint32_t b = 0; b <= 5; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 9;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_GatherSetLayout));

            VkDescriptorPoolSize poolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 * 2 }
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
            VkDescriptorImageInfo gbufferRoughnessMetallicInfo{ VK_NULL_HANDLE, resolvePass.GetOutputRoughnessMetallicView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, resolvePass.GetOutputColorView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorImageInfo gbufferMaterialIDInfo{ VK_NULL_HANDLE, resolvePass.GetOutputMaterialIDView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo materialParamsInfo{ resolvePass.GetMaterialParamsBuffer(), 0, VK_WHOLE_SIZE };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                const ReflectionSlot& current = m_Slots[slotIndex];
                VkDescriptorImageInfo curRadiance{ VK_NULL_HANDLE, current.radianceView, VK_IMAGE_LAYOUT_GENERAL };

                VkDescriptorImageInfo* storageInfos[6] = { &curRadiance, &gbufferNormalInfo, &gbufferDepthInfo, &gbufferAlbedoInfo, &gbufferRoughnessMetallicInfo, &outputColorInfo };

                VkWriteDescriptorSet writes[9]{};
                for (uint32_t b = 0; b <= 5; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b], nullptr, nullptr };
                }
                writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], 6, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufferMaterialIDInfo, nullptr, nullptr };
                writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 9, writes, 0, nullptr);
            }

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_GatherSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_GatherPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ReflectionGather.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_GatherPipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_GatherPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        m_CurrentSlotIndex = 0;
        LOG_INFO(std::format("[ReflectionPass] Initialized: {} x {} full-resolution reflection buffers.", m_RenderExtent.width, m_RenderExtent.height));
        return true;
    }

    void ReflectionPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
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

            if (m_ReflectionSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ReflectionSampler, nullptr);

            for (ReflectionSlot& slot : m_Slots) {
                if (slot.radianceView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.radianceView, nullptr);
                if (slot.worldPosView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.worldPosView, nullptr);
                if (slot.normalView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.normalView, nullptr);
            }
            if (m_HitMaskView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_HitMaskView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            for (ReflectionSlot& slot : m_Slots) {
                vmaDestroyImage(m_Allocator, slot.radianceImage, slot.radianceAllocation);
                vmaDestroyImage(m_Allocator, slot.worldPosImage, slot.worldPosAllocation);
                vmaDestroyImage(m_Allocator, slot.normalImage, slot.normalAllocation);
            }
            vmaDestroyImage(m_Allocator, m_HitMaskImage, m_HitMaskAllocation);
        }
        m_ViewParamsBuffer.Destroy();

        m_GatherPipeline = VK_NULL_HANDLE; m_GatherPipelineLayout = VK_NULL_HANDLE; m_GatherDescriptorPool = VK_NULL_HANDLE; m_GatherSetLayout = VK_NULL_HANDLE;
        m_GatherSet[0] = VK_NULL_HANDLE; m_GatherSet[1] = VK_NULL_HANDLE;
        m_TemporalPipeline = VK_NULL_HANDLE; m_TemporalPipelineLayout = VK_NULL_HANDLE; m_TemporalDescriptorPool = VK_NULL_HANDLE; m_TemporalSetLayout = VK_NULL_HANDLE;
        m_TemporalSet[0] = VK_NULL_HANDLE; m_TemporalSet[1] = VK_NULL_HANDLE;
        m_TracePipeline = VK_NULL_HANDLE; m_TracePipelineLayout = VK_NULL_HANDLE; m_TraceDescriptorPool = VK_NULL_HANDLE; m_TraceSetLayout = VK_NULL_HANDLE;
        m_TraceSet[0] = VK_NULL_HANDLE; m_TraceSet[1] = VK_NULL_HANDLE;
        m_ReflectionSampler = VK_NULL_HANDLE;
        m_Slots[0] = ReflectionSlot{};
        m_Slots[1] = ReflectionSlot{};
        m_HitMaskImage = VK_NULL_HANDLE; m_HitMaskAllocation = VK_NULL_HANDLE; m_HitMaskView = VK_NULL_HANDLE;
        m_CurrentSlotIndex = 0;
        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ReflectionPass::RecordUpdateViewParams(VkCommandBuffer cmd, const maths::mat4& viewProj,
        const maths::mat4& prevViewProj, const maths::vec3& cameraPositionWorld) {
        ReflectionViewParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.prevViewProj = prevViewProj;
        ubo.cameraPositionWorldX = cameraPositionWorld.x;
        ubo.cameraPositionWorldY = cameraPositionWorld.y;
        ubo.cameraPositionWorldZ = cameraPositionWorld.z;
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);
    }

    void ReflectionPass::RecordTrace(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
        uint32_t entityCount, uint32_t traceMode, uint32_t frameIndex) {
        // This frame writes into whichever slot held the PREVIOUS frame's history -- see this
        // method's own header comment.
        m_CurrentSlotIndex = 1 - m_CurrentSlotIndex;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TracePipeline);
        VkDescriptorSet sets[3] = { m_TraceSet[m_CurrentSlotIndex], traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TracePipelineLayout, 0, 3, sets, 0, nullptr);

        TracePushConstants pc{};
        pc.entityCount = entityCount;
        pc.traceMode = traceMode;
        pc.frameIndex = frameIndex;
        vkCmdPushConstants(cmd, m_TracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupCountX = (m_RenderExtent.width + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    void ReflectionPass::RecordTemporal(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TemporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TemporalPipelineLayout, 0, 1, &m_TemporalSet[m_CurrentSlotIndex], 0, nullptr);

        uint32_t groupCountX = (m_RenderExtent.width + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void ReflectionPass::RecordGather(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GatherPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GatherPipelineLayout, 0, 1, &m_GatherSet[m_CurrentSlotIndex], 0, nullptr);

        uint32_t groupCountX = (m_RenderExtent.width + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    }

}
