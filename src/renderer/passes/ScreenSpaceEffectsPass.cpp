#include "renderer/passes/ScreenSpaceEffectsPass.h"

#include <cmath>
#include <format>

#include "core/Logger.h"
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/ReflectionPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte std140 mirror of GTAO.comp's GTAOParamsUBO.
        struct GTAOParamsUBO {
            maths::mat4 invViewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f;
            float radiusWorld = 1.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float projScale = 1.0f;
            float intensity = 1.0f;
            float power = 1.5f;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(GTAOParamsUBO) == 112, "GTAOParamsUBO must match GTAO.comp's GTAOParamsUBO exactly (std140 layout)");

        // Byte-for-byte std140 mirror of ContactShadows.comp's ContactShadowParamsUBO.
        struct ContactShadowParamsUBO {
            maths::mat4 invViewProj;
            maths::mat4 viewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f;
            float shadowLengthWorld = 1.0f;
            float sunDirX = 0.0f, sunDirY = -1.0f, sunDirZ = 0.0f;
            float intensity = 0.8f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float thicknessWorld = 0.3f;
            float _pad0 = 0.0f;
        };
        static_assert(sizeof(ContactShadowParamsUBO) == 176, "ContactShadowParamsUBO must match ContactShadows.comp's ContactShadowParamsUBO exactly (std140 layout)");

        // Byte-for-byte std140 mirror of SSRFallback.comp's SSRFallbackParamsUBO.
        struct SSRFallbackParamsUBO {
            maths::mat4 invViewProj;
            maths::mat4 viewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f;
            float maxDistanceWorld = 20.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float thicknessWorld = 0.5f;
            float intensity = 1.0f;
            // Atmos weather system, Subtask 5.
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
            float _pad0 = 0.0f;
        };
        static_assert(sizeof(SSRFallbackParamsUBO) == 176, "SSRFallbackParamsUBO must match SSRFallback.comp's SSRFallbackParamsUBO exactly (std140 layout)");

        VkPipeline CreateComputePipeline(VkDevice device, VkPipelineLayout layout, const char* shaderPath) {
            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(device, shaderPath);
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = layout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VkPipeline pipeline = VK_NULL_HANDLE;
            VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return pipeline;
        }

    } // namespace

    void ScreenSpaceEffectsPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, const ClusterResolvePass& resolvePass, const ReflectionPass& reflectionPass) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        // =====================================================================================
        // Owned AO output image.
        // =====================================================================================
        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kAOFormat, renderExtent, m_AOImage, m_AOAllocation, m_AOView);
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkClearColorValue whiteClear{}; whiteClear.float32[0] = 1.0f; whiteClear.float32[1] = 1.0f; whiteClear.float32[2] = 1.0f; whiteClear.float32[3] = 1.0f;
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_AOImage, whiteClear); // Neutral default: fully unoccluded.
            });

        m_AOParamsBuffer.Create(allocator, sizeof(GTAOParamsUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ContactShadowParamsBuffer.Create(allocator, sizeof(ContactShadowParamsUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_SSRFallbackParamsBuffer.Create(allocator, sizeof(SSRFallbackParamsUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        VkDescriptorImageInfo depthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo normalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo roughnessMetallicInfo{ VK_NULL_HANDLE, resolvePass.GetOutputRoughnessMetallicView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo albedoInfo{ VK_NULL_HANDLE, resolvePass.GetOutputAlbedoView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, resolvePass.GetOutputColorView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo aoOutputInfo{ VK_NULL_HANDLE, m_AOView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo hitMaskInfo{ VK_NULL_HANDLE, reflectionPass.GetHitMaskView(), VK_IMAGE_LAYOUT_GENERAL };

        // =====================================================================================
        // Stage 1 -- GTAO: set 0 (4 bindings). g_GBufferDepth / g_GBufferNormal / g_AOOutput / UBO.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[4]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 4;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_AOSetLayout));

            VkDescriptorPoolSize poolSizes[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 } };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 2;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_AODescriptorPool));

            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_AODescriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_AOSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_AOSet));

            VkDescriptorBufferInfo aoParamsInfo{ m_AOParamsBuffer.Handle(), 0, m_AOParamsBuffer.Size() };
            VkWriteDescriptorSet writes[4]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_AOSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &depthInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_AOSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normalInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_AOSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &aoOutputInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_AOSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &aoParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_AOSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_AOPipelineLayout));
            m_AOPipeline = CreateComputePipeline(m_Device, m_AOPipelineLayout, "shaders/GTAO.comp.spv");
        }

        // =====================================================================================
        // Stage 2 -- Contact Shadows: set 0 (4 bindings). g_GBufferDepth / g_GBufferNormal /
        // g_OutputColor (RMW) / UBO.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[4]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 4;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ContactShadowSetLayout));

            VkDescriptorPoolSize poolSizes[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 } };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 2;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_ContactShadowDescriptorPool));

            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_ContactShadowDescriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_ContactShadowSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_ContactShadowSet));

            VkDescriptorBufferInfo contactShadowParamsInfo{ m_ContactShadowParamsBuffer.Handle(), 0, m_ContactShadowParamsBuffer.Size() };
            VkWriteDescriptorSet writes[4]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ContactShadowSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &depthInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ContactShadowSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normalInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ContactShadowSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputColorInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ContactShadowSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &contactShadowParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ContactShadowSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ContactShadowPipelineLayout));
            m_ContactShadowPipeline = CreateComputePipeline(m_Device, m_ContactShadowPipelineLayout, "shaders/ContactShadows.comp.spv");
        }

        // =====================================================================================
        // Stage 3 -- SSR Fallback: set 0 (7 bindings). g_GBufferNormal / g_GBufferDepth /
        // g_GBufferRoughnessMetallic / g_GBufferAlbedo / g_HitMask / g_OutputColor (RMW) / UBO.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[8]{};
            for (uint32_t b = 0; b <= 4; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            // Atmos weather system, Subtask 5 -- see SetAtmosSkyView()'s own comment.
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 8;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SSRFallbackSetLayout));

            VkDescriptorPoolSize poolSizes[3] = { { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 3;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_SSRFallbackDescriptorPool));

            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_SSRFallbackDescriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_SSRFallbackSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_SSRFallbackSet));

            VkDescriptorImageInfo* storageInfos[6] = { &normalInfo, &depthInfo, &roughnessMetallicInfo, &albedoInfo, &hitMaskInfo, &outputColorInfo };
            VkDescriptorBufferInfo ssrParamsInfo{ m_SSRFallbackParamsBuffer.Handle(), 0, m_SSRFallbackParamsBuffer.Size() };

            VkWriteDescriptorSet writes[7]{};
            for (uint32_t b = 0; b <= 5; ++b) {
                writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SSRFallbackSet, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b], nullptr, nullptr };
            }
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SSRFallbackSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &ssrParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_SSRFallbackSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_SSRFallbackPipelineLayout));
            m_SSRFallbackPipeline = CreateComputePipeline(m_Device, m_SSRFallbackPipelineLayout, "shaders/SSRFallback.comp.spv");
        }

        LOG_INFO(std::format("[ScreenSpaceEffectsPass] Initialized: {} x {} GTAO/Contact Shadows/SSR Fallback.", m_RenderExtent.width, m_RenderExtent.height));
    }

    void ScreenSpaceEffectsPass::SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler) {
        VkDescriptorImageInfo skyViewInfo{ skyViewLUTSampler, skyViewLUTView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SSRFallbackSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &skyViewInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
    }

    void ScreenSpaceEffectsPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_SSRFallbackPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_SSRFallbackPipeline, nullptr);
            if (m_SSRFallbackPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_SSRFallbackPipelineLayout, nullptr);
            if (m_SSRFallbackDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_SSRFallbackDescriptorPool, nullptr);
            if (m_SSRFallbackSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SSRFallbackSetLayout, nullptr);

            if (m_ContactShadowPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ContactShadowPipeline, nullptr);
            if (m_ContactShadowPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ContactShadowPipelineLayout, nullptr);
            if (m_ContactShadowDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_ContactShadowDescriptorPool, nullptr);
            if (m_ContactShadowSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ContactShadowSetLayout, nullptr);

            if (m_AOPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_AOPipeline, nullptr);
            if (m_AOPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_AOPipelineLayout, nullptr);
            if (m_AODescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_AODescriptorPool, nullptr);
            if (m_AOSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_AOSetLayout, nullptr);

            if (m_AOView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_AOView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_AOImage, m_AOAllocation);
        }

        m_AOParamsBuffer.Destroy();
        m_ContactShadowParamsBuffer.Destroy();
        m_SSRFallbackParamsBuffer.Destroy();

        m_SSRFallbackPipeline = VK_NULL_HANDLE; m_SSRFallbackPipelineLayout = VK_NULL_HANDLE; m_SSRFallbackDescriptorPool = VK_NULL_HANDLE; m_SSRFallbackSetLayout = VK_NULL_HANDLE; m_SSRFallbackSet = VK_NULL_HANDLE;
        m_ContactShadowPipeline = VK_NULL_HANDLE; m_ContactShadowPipelineLayout = VK_NULL_HANDLE; m_ContactShadowDescriptorPool = VK_NULL_HANDLE; m_ContactShadowSetLayout = VK_NULL_HANDLE; m_ContactShadowSet = VK_NULL_HANDLE;
        m_AOPipeline = VK_NULL_HANDLE; m_AOPipelineLayout = VK_NULL_HANDLE; m_AODescriptorPool = VK_NULL_HANDLE; m_AOSetLayout = VK_NULL_HANDLE; m_AOSet = VK_NULL_HANDLE;
        m_AOImage = VK_NULL_HANDLE; m_AOAllocation = VK_NULL_HANDLE; m_AOView = VK_NULL_HANDLE;

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ScreenSpaceEffectsPass::RecordAmbientOcclusion(VkCommandBuffer cmd, const maths::mat4& viewProj,
        const maths::vec3& cameraPositionWorld, float fovYRadians, const Settings& settings) {
        GTAOParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.cameraPosX = cameraPositionWorld.x;
        ubo.cameraPosY = cameraPositionWorld.y;
        ubo.cameraPosZ = cameraPositionWorld.z;
        ubo.radiusWorld = settings.aoRadiusWorld;
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        ubo.projScale = static_cast<float>(m_RenderExtent.height) / (2.0f * std::tan(fovYRadians * 0.5f));
        ubo.intensity = settings.aoIntensity;
        ubo.power = settings.aoPower;
        vkCmdUpdateBuffer(cmd, m_AOParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_AOPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_AOPipelineLayout, 0, 1, &m_AOSet, 0, nullptr);

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void ScreenSpaceEffectsPass::RecordContactShadows(VkCommandBuffer cmd, const maths::mat4& viewProj,
        const maths::vec3& cameraPositionWorld, const maths::vec3& sunDirection, const Settings& settings) {
        ContactShadowParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.viewProj = viewProj;
        ubo.cameraPosX = cameraPositionWorld.x;
        ubo.cameraPosY = cameraPositionWorld.y;
        ubo.cameraPosZ = cameraPositionWorld.z;
        ubo.shadowLengthWorld = settings.contactShadowLengthWorld;
        ubo.sunDirX = sunDirection.x;
        ubo.sunDirY = sunDirection.y;
        ubo.sunDirZ = sunDirection.z;
        ubo.intensity = settings.contactShadowIntensity;
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        ubo.thicknessWorld = settings.contactShadowThicknessWorld;
        vkCmdUpdateBuffer(cmd, m_ContactShadowParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ContactShadowPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ContactShadowPipelineLayout, 0, 1, &m_ContactShadowSet, 0, nullptr);

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    void ScreenSpaceEffectsPass::RecordSSRFallback(VkCommandBuffer cmd, const maths::mat4& viewProj,
        const maths::vec3& cameraPositionWorld, const maths::vec3& sunDirectionWorld, const Settings& settings) {
        SSRFallbackParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.viewProj = viewProj;
        ubo.cameraPosX = cameraPositionWorld.x;
        ubo.cameraPosY = cameraPositionWorld.y;
        ubo.cameraPosZ = cameraPositionWorld.z;
        ubo.maxDistanceWorld = settings.ssrFallbackMaxDistanceWorld;
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        ubo.thicknessWorld = settings.ssrFallbackThicknessWorld;
        ubo.intensity = settings.ssrFallbackIntensity;
        ubo.sunDirX = sunDirectionWorld.x; ubo.sunDirY = sunDirectionWorld.y; ubo.sunDirZ = sunDirectionWorld.z;
        vkCmdUpdateBuffer(cmd, m_SSRFallbackParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SSRFallbackPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SSRFallbackPipelineLayout, 0, 1, &m_SSRFallbackSet, 0, nullptr);

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    }

}
