#include "renderer/passes/AtmosCloudsPass.h"

#include <algorithm>
#include <array>
#include <format>

#include "core/Logger.h"
#include "renderer/passes/AtmosClimatePass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of AtmosClouds.comp's AtmosCloudsPC push_constant block.
        struct AtmosCloudsPC {
            int32_t mode = 0;
            int32_t outputWidth = 0;
            int32_t outputHeight = 0;
            float cameraPositionX = 0.0f, cameraPositionY = 0.0f, cameraPositionZ = 0.0f;
            float cameraForwardX = 0.0f, cameraForwardY = 0.0f, cameraForwardZ = 0.0f;
            float cameraRightX = 0.0f, cameraRightY = 0.0f, cameraRightZ = 0.0f;
            float cameraUpX = 0.0f, cameraUpY = 0.0f, cameraUpZ = 0.0f;
            float tanHalfFovY = 0.0f;
            float aspectRatio = 0.0f;
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
            float sunColorR = 0.0f, sunColorG = 0.0f, sunColorB = 0.0f;
            float sunIlluminance = 0.0f;
        };
        static_assert(sizeof(AtmosCloudsPC) == 96,
            "AtmosCloudsPC must match AtmosClouds.comp's own push_constant block exactly");

        // Byte-for-byte mirror of AtmosCloudShadows.comp's AtmosCloudShadowsPC push_constant block.
        struct AtmosCloudShadowsPC {
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
            float _pad0 = 0.0f;
        };
        static_assert(sizeof(AtmosCloudShadowsPC) == 16,
            "AtmosCloudShadowsPC must match AtmosCloudShadows.comp's own push_constant block exactly");

    } // namespace

    bool AtmosCloudsPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, const AtmosClimatePass& atmosClimate) {
        m_OutputExtent = { std::max(1u, renderExtent.width / kResolutionDivisor), std::max(1u, renderExtent.height / kResolutionDivisor) };

        // --- 2 noise textures, generated once below, never rewritten again. ---
        auto createNoiseImage = [&](uint32_t resolution, Image3D& outImage) {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_3D;
            imageInfo.format = kNoiseFormat;
            imageInfo.extent = { resolution, resolution, resolution };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage.image, &outImage.allocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = outImage.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            viewInfo.format = kNoiseFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outImage.view));
        };
        createNoiseImage(kShapeNoiseResolution, m_ShapeNoise);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_ShapeNoise.view, nullptr);
            vmaDestroyImage(m_Allocator, m_ShapeNoise.image, m_ShapeNoise.allocation);
        });
        createNoiseImage(kDetailNoiseResolution, m_DetailNoise);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_DetailNoise.view, nullptr);
            vmaDestroyImage(m_Allocator, m_DetailNoise.image, m_DetailNoise.allocation);
        });

        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kOutputFormat, m_OutputExtent,
            m_CloudOutput.image, m_CloudOutput.allocation, m_CloudOutput.view);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_CloudOutput.view, nullptr);
            vmaDestroyImage(m_Allocator, m_CloudOutput.image, m_CloudOutput.allocation);
        });

        // Atmos Subtask 5: Cloud Shadow Map, fixed 512x512 regardless of render extent (see class
        // comment -- this covers a fixed real-world-unit patch, not a screen-space buffer).
        VkExtent2D shadowExtent{ kCloudShadowMapResolution, kCloudShadowMapResolution };
        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kShadowFormat, shadowExtent,
            m_CloudShadowMap.image, m_CloudShadowMap.allocation, m_CloudShadowMap.view);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_CloudShadowMap.view, nullptr);
            vmaDestroyImage(m_Allocator, m_CloudShadowMap.image, m_CloudShadowMap.allocation);
        });

        VkClearColorValue zeroClear{}; zeroClear.float32[0] = zeroClear.float32[1] = zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 1.0f;
        VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_ShapeNoise.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_DetailNoise.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_CloudOutput.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_CloudShadowMap.image, zeroClear);
            });

        // REPEAT, not CLAMP: the noise textures tile continuously as clouds scroll across many
        // multiples of their own world-space footprint -- see AtmosClouds.comp's own world-to-noise-
        // UVW scaling comment.
        VkSamplerCreateInfo noiseSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        noiseSamplerInfo.magFilter = VK_FILTER_LINEAR;
        noiseSamplerInfo.minFilter = VK_FILTER_LINEAR;
        noiseSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        noiseSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        noiseSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        noiseSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        noiseSamplerInfo.minLod = 0.0f;
        noiseSamplerInfo.maxLod = 0.0f;
        noiseSamplerInfo.compareEnable = VK_FALSE;
        noiseSamplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &noiseSamplerInfo, nullptr, &m_NoiseSampler));
        RegisterResource([this] { vkDestroySampler(m_Device, m_NoiseSampler, nullptr); });

        VkSamplerCreateInfo linearSamplerInfo = noiseSamplerInfo;
        linearSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        linearSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        linearSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(m_Device, &linearSamplerInfo, nullptr, &m_LinearSampler));
        RegisterResource([this] { vkDestroySampler(m_Device, m_LinearSampler, nullptr); });

        // --- Descriptor set: 6 bindings -- see AtmosClouds.comp's own binding comments. ---
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{ {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_ShapeNoiseStorage
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_ShapeNoiseSampler
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_DetailNoiseStorage
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_DetailNoiseSampler
            { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_CloudOutputStorage
            { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // AtmosGlobalsUBO
        } };
        std::array<VkDescriptorPoolSize, 3> poolSizes{ {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
        } };
        auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device, bindings, poolSizes);
        m_SetLayout = descSet.layout;
        m_DescriptorPool = descSet.pool;
        m_Set = descSet.set;
        RegisterResource([this] {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        });

        VkDescriptorImageInfo shapeStorageInfo{ VK_NULL_HANDLE, m_ShapeNoise.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo shapeSamplerInfo{ m_NoiseSampler, m_ShapeNoise.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo detailStorageInfo{ VK_NULL_HANDLE, m_DetailNoise.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo detailSamplerInfo{ m_NoiseSampler, m_DetailNoise.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outputStorageInfo{ VK_NULL_HANDLE, m_CloudOutput.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo atmosGlobalsInfo{ atmosClimate.GetGlobalsBufferHandle(), 0, atmosClimate.GetGlobalsBufferSize() };

        std::array<VkWriteDescriptorSet, 6> writes{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shapeStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shapeSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &detailStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &detailSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &atmosGlobalsInfo, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AtmosCloudsPC) };
        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &m_SetLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_PipelineLayout));
        RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); });

        VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/AtmosClouds.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shader);
        vkDestroyShaderModule(m_Device, shader, nullptr);
        RegisterResource([this] { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); });

        // =========================================================================================
        // Atmos Subtask 5: Cloud Shadow Map pipeline -- a SEPARATE descriptor set/pipeline layout/
        // pipeline (different shader entry point, different bindings) from the noise-gen/raymarch
        // one above, reusing m_ShapeNoise/m_DetailNoise (bound again here, into this own set) and
        // m_NoiseSampler. See AtmosCloudShadows.comp's own binding comments.
        // =========================================================================================
        {
            std::array<VkDescriptorSetLayoutBinding, 4> shadowBindings{ {
                { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_ShadowMapStorage
                { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_ShapeNoiseSampler
                { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_DetailNoiseSampler
                { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // AtmosGlobalsUBO
            } };
            std::array<VkDescriptorPoolSize, 3> shadowPoolSizes{ {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
            } };
            auto shadowDescSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device, shadowBindings, shadowPoolSizes);
            m_ShadowSetLayout = shadowDescSet.layout;
            m_ShadowDescriptorPool = shadowDescSet.pool;
            m_ShadowSet = shadowDescSet.set;
            RegisterResource([this] {
                vkDestroyDescriptorPool(m_Device, m_ShadowDescriptorPool, nullptr);
                vkDestroyDescriptorSetLayout(m_Device, m_ShadowSetLayout, nullptr);
            });

            VkDescriptorImageInfo shadowStorageInfo{ VK_NULL_HANDLE, m_CloudShadowMap.view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo shadowShapeSamplerInfo{ m_NoiseSampler, m_ShapeNoise.view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo shadowDetailSamplerInfo{ m_NoiseSampler, m_DetailNoise.view, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo shadowAtmosGlobalsInfo{ atmosClimate.GetGlobalsBufferHandle(), 0, atmosClimate.GetGlobalsBufferSize() };
            std::array<VkWriteDescriptorSet, 4> shadowWrites{ {
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadowSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &shadowStorageInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadowSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowShapeSamplerInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadowSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowDetailSamplerInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ShadowSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &shadowAtmosGlobalsInfo, nullptr },
            } };
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(shadowWrites.size()), shadowWrites.data(), 0, nullptr);

            VkPushConstantRange shadowPushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AtmosCloudShadowsPC) };
            VkPipelineLayoutCreateInfo shadowPlInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            shadowPlInfo.setLayoutCount = 1;
            shadowPlInfo.pSetLayouts = &m_ShadowSetLayout;
            shadowPlInfo.pushConstantRangeCount = 1;
            shadowPlInfo.pPushConstantRanges = &shadowPushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &shadowPlInfo, nullptr, &m_ShadowPipelineLayout));
            RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_ShadowPipelineLayout, nullptr); });

            VkShaderModule shadowShader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/AtmosCloudShadows.comp.spv");
            m_ShadowPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ShadowPipelineLayout, shadowShader);
            vkDestroyShaderModule(m_Device, shadowShader, nullptr);
            RegisterResource([this] { vkDestroyPipeline(m_Device, m_ShadowPipeline, nullptr); });
        }

        // --- One-time noise generation (modes 0/1), dispatched here, never again. ---
        VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

            AtmosCloudsPC pc{};
            pc.mode = 0;
            pc.outputWidth = static_cast<int32_t>(kShapeNoiseResolution);
            pc.outputHeight = static_cast<int32_t>(kShapeNoiseResolution);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t shapeGroups = (kShapeNoiseResolution + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
            vkCmdDispatch(cmd, shapeGroups, shapeGroups, shapeGroups);

            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            pc.mode = 1;
            pc.outputWidth = static_cast<int32_t>(kDetailNoiseResolution);
            pc.outputHeight = static_cast<int32_t>(kDetailNoiseResolution);
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t detailGroups = (kDetailNoiseResolution + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
            vkCmdDispatch(cmd, detailGroups, detailGroups, detailGroups);

            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            });

        LOG_INFO(std::format("[AtmosCloudsPass] Initialized (shape {}^3, detail {}^3, cloud output {}x{}, shadow map {}x{}).",
            kShapeNoiseResolution, kDetailNoiseResolution, m_OutputExtent.width, m_OutputExtent.height,
            kCloudShadowMapResolution, kCloudShadowMapResolution));
        return true;
    }

    // Shutdown() is inherited from RenderPass<AtmosCloudsPass>: runs the RegisterResource() cleanups
    // above in reverse (cloud-shadow pipeline -> cloud-shadow pipeline layout -> cloud-shadow
    // descriptor pool+layout -> raymarch pipeline -> raymarch pipeline layout -> raymarch descriptor
    // pool+layout -> linear sampler -> noise sampler -> shadow map image -> cloud output image ->
    // detail noise image -> shape noise image), the same dependency-safe order the hand-written
    // Shutdown() used. Note: unlike the original, m_OutputExtent is not reset to {0,0} on Shutdown()
    // -- it is set unconditionally at the top of InitImpl() on every (re-)Init anyway, and was never
    // read between a Shutdown() and the next Init() in this codebase's actual call pattern.

    void AtmosCloudsPass::DispatchMode(VkCommandBuffer cmd, int32_t mode, const maths::vec3& cameraPosition,
        const maths::vec3& cameraForward, const maths::vec3& cameraRight, const maths::vec3& cameraUp,
        float tanHalfFovY, float aspectRatio, const maths::vec3& sunDirectionWorld,
        const maths::vec3& sunColor, float sunIlluminanceLux) {
        AtmosCloudsPC pc{};
        pc.mode = mode;
        pc.outputWidth = static_cast<int32_t>(m_OutputExtent.width);
        pc.outputHeight = static_cast<int32_t>(m_OutputExtent.height);
        pc.cameraPositionX = cameraPosition.x; pc.cameraPositionY = cameraPosition.y; pc.cameraPositionZ = cameraPosition.z;
        pc.cameraForwardX = cameraForward.x; pc.cameraForwardY = cameraForward.y; pc.cameraForwardZ = cameraForward.z;
        pc.cameraRightX = cameraRight.x; pc.cameraRightY = cameraRight.y; pc.cameraRightZ = cameraRight.z;
        pc.cameraUpX = cameraUp.x; pc.cameraUpY = cameraUp.y; pc.cameraUpZ = cameraUp.z;
        pc.tanHalfFovY = tanHalfFovY;
        pc.aspectRatio = aspectRatio;
        pc.sunDirX = sunDirectionWorld.x; pc.sunDirY = sunDirectionWorld.y; pc.sunDirZ = sunDirectionWorld.z;
        pc.sunColorR = sunColor.x; pc.sunColorG = sunColor.y; pc.sunColorB = sunColor.z;
        pc.sunIlluminance = sunIlluminanceLux;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupsX = (m_OutputExtent.width + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
        uint32_t groupsY = (m_OutputExtent.height + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    void AtmosCloudsPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPosition,
        const maths::vec3& cameraForward, const maths::vec3& upHint,
        float fovYRadians, float aspectRatio, const maths::vec3& sunDirectionWorld,
        const maths::vec3& sunColor, float sunIlluminanceLux) {

        const maths::vec3 forward = cameraForward.Normalize();
        const maths::vec3 right = forward.Cross(upHint).Normalize();
        const maths::vec3 up = right.Cross(forward);

        DispatchMode(cmd, 2, cameraPosition, forward, right, up, std::tan(fovYRadians * 0.5f), aspectRatio,
            sunDirectionWorld, sunColor, sunIlluminanceLux);

        // Atmos Subtask 5: Cloud Shadow Map, regenerated every frame (tracks the live sun direction
        // and wind-scrolled cloud position, unlike the noise textures) -- independent pipeline/set,
        // so this dispatch has no ordering dependency on the raymarch dispatch just above (both only
        // read m_ShapeNoise/m_DetailNoise, never write them after Init()).
        {
            AtmosCloudShadowsPC shadowPC{};
            shadowPC.sunDirX = sunDirectionWorld.x;
            shadowPC.sunDirY = sunDirectionWorld.y;
            shadowPC.sunDirZ = sunDirectionWorld.z;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ShadowPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ShadowPipelineLayout, 0, 1, &m_ShadowSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_ShadowPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shadowPC), &shadowPC);
            uint32_t shadowGroups = (kCloudShadowMapResolution + kShadowWorkgroupSize - 1u) / kShadowWorkgroupSize;
            vkCmdDispatch(cmd, shadowGroups, shadowGroups, 1);
        }

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
