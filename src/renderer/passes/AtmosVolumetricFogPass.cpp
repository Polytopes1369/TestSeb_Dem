#include "renderer/passes/AtmosVolumetricFogPass.h"

#include <array>
#include <cmath>
#include <format>

#include "core/Logger.h"
#include "renderer/passes/AtmosClimatePass.h"
#include "renderer/passes/MegaLightsPass.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of AtmosVolumetricFog.comp's AtmosVolumetricFogPC push_constant block.
        struct AtmosVolumetricFogPC {
            int32_t mode = 0;
            float cameraPositionX = 0.0f, cameraPositionY = 0.0f, cameraPositionZ = 0.0f;
            float cameraForwardX = 0.0f, cameraForwardY = 0.0f, cameraForwardZ = 0.0f;
            float cameraRightX = 0.0f, cameraRightY = 0.0f, cameraRightZ = 0.0f;
            float cameraUpX = 0.0f, cameraUpY = 0.0f, cameraUpZ = 0.0f;
            float tanHalfFovY = 0.0f;
            float aspectRatio = 0.0f;
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
            float sunColorR = 0.0f, sunColorG = 0.0f, sunColorB = 0.0f;
            float sunIlluminance = 0.0f;
            uint32_t frameIndex = 0u;
        };
        static_assert(sizeof(AtmosVolumetricFogPC) == 92,
            "AtmosVolumetricFogPC must match AtmosVolumetricFog.comp's own push_constant block exactly");

    } // namespace

    void AtmosVolumetricFogPass::CreateFroxelImage(VmaAllocator allocator, VkDevice device, VkFormat format, FroxelImage& outImage) {
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.format = format;
        imageInfo.extent = { kGridWidth, kGridHeight, kGridDepth };
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
        viewInfo.format = format;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outImage.view));
    }

    bool AtmosVolumetricFogPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const AtmosClimatePass& atmosClimate, const MegaLightsPass& megaLights, const VirtualShadowMapPass& vsm) {
        m_Device = device;
        m_Allocator = allocator;

        CreateFroxelImage(allocator, device, kMediaFormat, m_MediaProps);
        CreateFroxelImage(allocator, device, kRadianceFormat, m_RawLight);
        CreateFroxelImage(allocator, device, kRadianceFormat, m_IntegratedFog);

        VkClearColorValue zeroClear{}; zeroClear.float32[0] = zeroClear.float32[1] = zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 1.0f;
        VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_MediaProps.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_RawLight.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_IntegratedFog.image, zeroClear);
            });

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_FogSampler));

        // --- Descriptor set: 11 bindings -- see AtmosVolumetricFog.comp's own binding comments. ---
        std::array<VkDescriptorSetLayoutBinding, 11> bindings{ {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_MediaPropsStorage
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_MediaPropsSampler
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_RawLightStorage
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_RawLightSampler
            { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_IntegratedFogStorage
            { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // AtmosGlobalsUBO
            { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // MegaLightsSSBO
            { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Shadow atlas
            { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // Shadow page table
            { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // Shadow feedback
            { 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },        // Shadow sun levels
        } };
        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        std::array<VkDescriptorPoolSize, 4> poolSizes{ {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 }, // AtmosGlobalsUBO + shadow sun levels.
        } };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo allocSet{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocSet.descriptorPool = m_DescriptorPool;
        allocSet.descriptorSetCount = 1;
        allocSet.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocSet, &m_Set));

        VkDescriptorImageInfo mediaStorageInfo{ VK_NULL_HANDLE, m_MediaProps.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo mediaSamplerInfo{ m_FogSampler, m_MediaProps.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo rawStorageInfo{ VK_NULL_HANDLE, m_RawLight.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo rawSamplerInfo{ m_FogSampler, m_RawLight.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo integratedStorageInfo{ VK_NULL_HANDLE, m_IntegratedFog.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo atmosGlobalsInfo{ atmosClimate.GetGlobalsBufferHandle(), 0, atmosClimate.GetGlobalsBufferSize() };
        VkDescriptorBufferInfo megaLightsInfo{ megaLights.GetLightBufferHandle(), 0, megaLights.GetLightBufferSize() };
        VkDescriptorImageInfo shadowAtlasInfo{ vsm.GetPhysicalAtlasSampler(), vsm.GetPhysicalAtlasView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo shadowPageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo shadowFeedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo shadowSunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 11> writes{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mediaStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mediaSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rawStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rawSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &integratedStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &atmosGlobalsInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &megaLightsInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowAtlasInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowPageTableInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowFeedbackInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &shadowSunLevelsInfo, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AtmosVolumetricFogPC) };
        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &m_SetLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/AtmosVolumetricFog.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shader);
        vkDestroyShaderModule(m_Device, shader, nullptr);

        LOG_INFO(std::format("[AtmosVolumetricFogPass] Initialized ({}x{}x{} froxel grid).", kGridWidth, kGridHeight, kGridDepth));
        return true;
    }

    void AtmosVolumetricFogPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_FogSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_FogSampler, nullptr);
            if (m_MediaProps.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_MediaProps.view, nullptr);
            if (m_RawLight.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_RawLight.view, nullptr);
            if (m_IntegratedFog.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_IntegratedFog.view, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            if (m_MediaProps.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_MediaProps.image, m_MediaProps.allocation);
            if (m_RawLight.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_RawLight.image, m_RawLight.allocation);
            if (m_IntegratedFog.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_IntegratedFog.image, m_IntegratedFog.allocation);
        }

        m_MediaProps = {};
        m_RawLight = {};
        m_IntegratedFog = {};
        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Set = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_FogSampler = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void AtmosVolumetricFogPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPosition,
        const maths::vec3& cameraForward, const maths::vec3& upHint,
        float fovYRadians, float aspectRatio, const maths::vec3& sunDirectionWorld,
        const maths::vec3& sunColor, float sunIlluminanceLux, uint32_t frameIndex) {

        // Same orthonormal-basis derivation as renderer::SDFRayMarchPass::RecordRayMarch's own
        // comment (itself matching renderer::SurfaceCachePass::UpdateVisibility).
        const maths::vec3 forward = cameraForward.Normalize();
        const maths::vec3 right = forward.Cross(upHint).Normalize();
        const maths::vec3 up = right.Cross(forward);

        AtmosVolumetricFogPC pc{};
        pc.cameraPositionX = cameraPosition.x; pc.cameraPositionY = cameraPosition.y; pc.cameraPositionZ = cameraPosition.z;
        pc.cameraForwardX = forward.x; pc.cameraForwardY = forward.y; pc.cameraForwardZ = forward.z;
        pc.cameraRightX = right.x; pc.cameraRightY = right.y; pc.cameraRightZ = right.z;
        pc.cameraUpX = up.x; pc.cameraUpY = up.y; pc.cameraUpZ = up.z;
        pc.tanHalfFovY = std::tan(fovYRadians * 0.5f);
        pc.aspectRatio = aspectRatio;
        pc.sunDirX = sunDirectionWorld.x; pc.sunDirY = sunDirectionWorld.y; pc.sunDirZ = sunDirectionWorld.z;
        pc.sunColorR = sunColor.x; pc.sunColorG = sunColor.y; pc.sunColorB = sunColor.z;
        pc.sunIlluminance = sunIlluminanceLux;
        pc.frameIndex = frameIndex;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

        const uint32_t groups3DX = (kGridWidth + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
        const uint32_t groups3DY = (kGridHeight + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
        const uint32_t groups3DZ = (kGridDepth + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;

        // --- Kernel 0: InjectMediaProps ---
        pc.mode = 0;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups3DX, groups3DY, groups3DZ);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // --- Kernel 1: InjectLight (reads m_MediaProps, writes m_RawLight) ---
        pc.mode = 1;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups3DX, groups3DY, groups3DZ);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // --- Kernel 2: AccumulateFog (one thread per (x, y) column, reads m_MediaProps + m_RawLight, writes m_IntegratedFog) ---
        // groups3DZ deliberately NOT included here (only 1 Z-group dispatched) -- see this class'
        // own kWorkgroupSize3D comment for why mode 2 only needs a 2D dispatch despite sharing the
        // 3D-shaped workgroup.
        pc.mode = 2;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups3DX, groups3DY, 1);

        // Trailing barrier: makes m_IntegratedFog visible to this frame's later consumer
        // (PostProcessComposite.comp), which reads it via COMPUTE_SHADER combined-image-sampler.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
