#include "renderer/passes/MegaLightsPass.h"

#include <cstring>
#include <format>

#include "core/Logger.h"
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte std140 mirror of MegaLightsViewParamsUBO in MegaLightsShade.comp.
        struct MegaLightsViewParamsUBO {
            maths::mat4 invViewProj;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float _pad0 = 0.0f, _pad1 = 0.0f;
        };
        static_assert(sizeof(MegaLightsViewParamsUBO) == 80,
            "MegaLightsViewParamsUBO must match MegaLightsShade.comp's own UBO exactly (std140 layout)");

    } // namespace

    bool MegaLightsPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, const ClusterResolvePass& resolvePass,
        const SurfaceCacheRayTracingPass& rtPass, const MegaLightsData& lightsData) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;
        m_LightCount = lightsData.count;

        // =====================================================================================
        // STEP 1 -- Light SSBO: host-visible, persistently mapped, filled once here (static
        // procedural population -- never re-uploaded per frame, same convention as renderer::
        // SurfaceCacheTraceContext's own host-visible entity/card buffers).
        // =====================================================================================
        constexpr VkDeviceSize kHeaderBytes = 16; // uint lightCount + 3 reserved uint (16-byte-aligns the trailing array, matches megalights_ris.glsl's own MegaLightsSSBO layout).
        VkDeviceSize lightBufferBytes = kHeaderBytes + static_cast<VkDeviceSize>(kMaxMegaLights) * sizeof(MegaLight);
        m_LightBuffer.Create(allocator, lightBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        {
            uint8_t* dst = static_cast<uint8_t*>(m_LightBuffer.MappedData());
            uint32_t header[4] = { m_LightCount, 0u, 0u, 0u };
            std::memcpy(dst, header, sizeof(header));
            std::memcpy(dst + kHeaderBytes, lightsData.lights.data(), kMaxMegaLights * sizeof(MegaLight));
        }

        // =====================================================================================
        // STEP 2 -- Raw shade radiance image (rgba16f linear HDR -- see MegaLightsPass::
        // kRadianceFormat's own comment for why this format is forced) + the dedicated
        // ATrousDenoisePass instance.
        // =====================================================================================
        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kRadianceFormat, renderExtent,
            m_RawRadianceImage, m_RawRadianceAllocation, m_RawRadianceView);
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkClearColorValue zeroClear{}; zeroClear.float32[0] = 0.0f; zeroClear.float32[1] = 0.0f; zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 0.0f;
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_RawRadianceImage, zeroClear);
            });

        m_Denoiser.Init(device, allocator, commandPool, queue, renderExtent,
            m_RawRadianceView, resolvePass.GetOutputDepthView(), resolvePass.GetOutputNormalView());

        m_ViewParamsBuffer.Create(allocator, sizeof(MegaLightsViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // =====================================================================================
        // STEP 3 -- Shade pipeline: set 0 (8 bindings, single set -- no ping-pong, Phase A has no
        // temporal state).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[8]{};
            for (uint32_t b : { 0u, 1u, 2u, 3u, 4u }) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 8;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ShadeSetLayout));

            VkDescriptorPoolSize poolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 },
                { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 4;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_ShadeDescriptorPool));

            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_ShadeDescriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_ShadeSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_ShadeSet));

            VkDescriptorImageInfo shadeRadianceInfo{ VK_NULL_HANDLE, m_RawRadianceView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferAlbedoInfo{ VK_NULL_HANDLE, resolvePass.GetOutputAlbedoView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferRoughnessMetallicInfo{ VK_NULL_HANDLE, resolvePass.GetOutputRoughnessMetallicView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo lightBufferInfo{ m_LightBuffer.Handle(), 0, m_LightBuffer.Size() };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

            VkDescriptorImageInfo* storageInfos[5] = { &shadeRadianceInfo, &gbufferNormalInfo, &gbufferDepthInfo, &gbufferAlbedoInfo, &gbufferRoughnessMetallicInfo };
            VkWriteDescriptorSet writes[7]{};
            for (uint32_t b = 0; b < 5; ++b) {
                writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b], nullptr, nullptr };
            }
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadeSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);

            // Binding 5 (acceleration structure) written separately -- VkWriteDescriptorSetAccelerationStructureKHR
            // needs its own pNext chain, same pattern VulkanUtils::WriteSharedGeometryBindings uses
            // internally (not called here directly since this shader needs only the TLAS, not the
            // vertex/index/draw-range buffers that helper also writes -- a shadow-visibility-only
            // query never reconstructs the hit surface, see MegaLightsShade.comp's own TraceShadowRay
            // comment).
            VkAccelerationStructureKHR tlasHandle = rtPass.GetTLASHandle();
            VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
            accelWrite.accelerationStructureCount = 1;
            accelWrite.pAccelerationStructures = &tlasHandle;
            VkWriteDescriptorSet accelDescriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            accelDescriptorWrite.pNext = &accelWrite;
            accelDescriptorWrite.dstSet = m_ShadeSet;
            accelDescriptorWrite.dstBinding = 5;
            accelDescriptorWrite.descriptorCount = 1;
            accelDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(m_Device, 1, &accelDescriptorWrite, 0, nullptr);

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ShadeSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ShadePipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/MegaLightsShade.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_ShadePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_ShadePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 4 -- Composite pipeline: set 0 (2 bindings). g_DenoisedRadiance reads m_Denoiser's
        // own output (already visible via ATrous's own trailing barrier); g_OutputColor
        // read-modify-writes resolvePass's output color image, same target renderer::ReflectionPass
        // ::RecordGather already RMWs into.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[2]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 2;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_CompositeSetLayout));

            VkDescriptorPoolSize poolSizes[1] = { { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 } };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_CompositeDescriptorPool));

            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_CompositeDescriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_CompositeSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_CompositeSet));

            VkDescriptorImageInfo denoisedInfo{ VK_NULL_HANDLE, m_Denoiser.GetOutputView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, resolvePass.GetOutputColorView(), VK_IMAGE_LAYOUT_GENERAL };

            VkWriteDescriptorSet writes[2]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &denoisedInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputColorInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_CompositeSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_CompositePipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/MegaLightsComposite.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_CompositePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_CompositePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        LOG_INFO(std::format("[MegaLightsPass] Initialized: {} lights, {} candidates/pixel, {} x {}.",
            m_LightCount, 16u, m_RenderExtent.width, m_RenderExtent.height));
        return true;
    }

    void MegaLightsPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_CompositePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_CompositePipeline, nullptr);
            if (m_CompositePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_CompositePipelineLayout, nullptr);
            if (m_CompositeDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_CompositeDescriptorPool, nullptr);
            if (m_CompositeSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_CompositeSetLayout, nullptr);

            if (m_ShadePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ShadePipeline, nullptr);
            if (m_ShadePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ShadePipelineLayout, nullptr);
            if (m_ShadeDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_ShadeDescriptorPool, nullptr);
            if (m_ShadeSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ShadeSetLayout, nullptr);

            if (m_RawRadianceView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_RawRadianceView, nullptr);
        }
        m_Denoiser.Shutdown();
        if (m_Allocator != VK_NULL_HANDLE && m_RawRadianceImage != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_RawRadianceImage, m_RawRadianceAllocation);
        }
        m_ViewParamsBuffer.Destroy();
        m_LightBuffer.Destroy();

        m_CompositePipeline = VK_NULL_HANDLE; m_CompositePipelineLayout = VK_NULL_HANDLE; m_CompositeDescriptorPool = VK_NULL_HANDLE; m_CompositeSetLayout = VK_NULL_HANDLE; m_CompositeSet = VK_NULL_HANDLE;
        m_ShadePipeline = VK_NULL_HANDLE; m_ShadePipelineLayout = VK_NULL_HANDLE; m_ShadeDescriptorPool = VK_NULL_HANDLE; m_ShadeSetLayout = VK_NULL_HANDLE; m_ShadeSet = VK_NULL_HANDLE;
        m_RawRadianceImage = VK_NULL_HANDLE; m_RawRadianceAllocation = VK_NULL_HANDLE; m_RawRadianceView = VK_NULL_HANDLE;
        m_LightCount = 0;
        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void MegaLightsPass::RecordShade(VkCommandBuffer cmd, const maths::mat4& viewProj, uint32_t frameIndex) {
        MegaLightsViewParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;

        // --- Shade: RIS + 1 shadow ray per pixel, writes raw noisy radiance. ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ShadePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ShadePipelineLayout, 0, 1, &m_ShadeSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_ShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &frameIndex);
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // Raw radiance image must be visible to the denoiser's own combined-image-sampler read.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // --- Denoise: 5 À-Trous iterations (ATrousDenoisePass::RecordDenoise, unmodified). ---
        m_Denoiser.RecordDenoise(cmd);

        // --- Composite: additive RMW into resolvePass's output color image. ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompositePipelineLayout, 0, 1, &m_CompositeSet, 0, nullptr);
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
