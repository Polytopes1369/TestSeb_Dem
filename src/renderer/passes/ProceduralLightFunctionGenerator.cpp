#include "renderer/passes/ProceduralLightFunctionGenerator.h"

#include <format>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    void ProceduralLightFunctionGenerator::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // --- Allocate every slot's image + view: usable both as a compute storage target
        // (generation, below) and as a sampled image afterward (light_functions.glsl /
        // caustics_projection.glsl). Same "STORAGE | SAMPLED, one-shot GENERAL -> SHADER_READ_ONLY_
        // OPTIMAL transition" shape as renderer::ProceduralMaskGenerator::Init(). ---
        m_Images.resize(kTotalSlots);
        m_Allocations.resize(kTotalSlots);
        m_ImageViews.resize(kTotalSlots);

        for (uint32_t i = 0; i < kTotalSlots; ++i) {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = kFormat;
            imageInfo.extent = { kTextureSize, kTextureSize, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo imageAllocInfo{};
            imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &imageAllocInfo, &m_Images[i], &m_Allocations[i], nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_Images[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = kFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_ImageViews[i]));
        }

        // REPEAT (not CLAMP): both the gobo patterns (a spot light's projected UV can tile several
        // times across its cone) and the caustics pattern (continuously UV-scrolled over time by
        // caustics_projection.glsl's own ComputeCausticsModulation) must wrap seamlessly.
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_Sampler));

        // =====================================================================================
        // One-time generation dispatch: descriptor set/pipeline are purely local to this
        // function -- no consumer pass ever needs them, only the finished image contents.
        // =====================================================================================
        VkDescriptorSetLayoutBinding genBinding{};
        genBinding.binding = 0;
        genBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        genBinding.descriptorCount = kTotalSlots;
        genBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayout genSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayoutCreateInfo genLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        genLayoutInfo.bindingCount = 1;
        genLayoutInfo.pBindings = &genBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &genLayoutInfo, nullptr, &genSetLayout));

        VkDescriptorPoolSize genPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kTotalSlots };
        VkDescriptorPool genDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorPoolCreateInfo genPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        genPoolInfo.maxSets = 1;
        genPoolInfo.poolSizeCount = 1;
        genPoolInfo.pPoolSizes = &genPoolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &genPoolInfo, nullptr, &genDescriptorPool));

        VkDescriptorSet genDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo genSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        genSetAlloc.descriptorPool = genDescriptorPool;
        genSetAlloc.descriptorSetCount = 1;
        genSetAlloc.pSetLayouts = &genSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &genSetAlloc, &genDescriptorSet));

        std::vector<VkDescriptorImageInfo> genImageInfos(kTotalSlots);
        for (uint32_t i = 0; i < kTotalSlots; ++i) {
            genImageInfos[i].imageView = m_ImageViews[i];
            genImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        VkWriteDescriptorSet genWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        genWrite.dstSet = genDescriptorSet;
        genWrite.dstBinding = 0;
        genWrite.descriptorCount = kTotalSlots;
        genWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        genWrite.pImageInfo = genImageInfos.data();
        vkUpdateDescriptorSets(m_Device, 1, &genWrite, 0, nullptr);

        VkPipelineLayout genPipelineLayout = VK_NULL_HANDLE;
        VkPipelineLayoutCreateInfo genPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        genPipelineLayoutInfo.setLayoutCount = 1;
        genPipelineLayoutInfo.pSetLayouts = &genSetLayout;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &genPipelineLayoutInfo, nullptr, &genPipelineLayout));

        VkShaderModule genShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/GenerateLightFunctionTextures.comp.spv");
        VkPipeline genPipeline = VulkanPipeline::CreateComputePipeline(m_Device, genPipelineLayout, genShaderModule);
        vkDestroyShaderModule(m_Device, genShaderModule, nullptr);

        // --- One-time command buffer: UNDEFINED -> GENERAL, dispatch (one Z-slice per slot), then
        // GENERAL -> SHADER_READ_ONLY_OPTIMAL. Blocking submit, startup-only -- mirrors
        // ProceduralMaskGenerator::Init()'s own identical one-shot transition pattern. ---
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            std::vector<VkImageMemoryBarrier2> toGeneral(kTotalSlots);
            for (uint32_t i = 0; i < kTotalSlots; ++i) {
                toGeneral[i] = VkImageMemoryBarrier2{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                toGeneral[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                toGeneral[i].srcAccessMask = 0;
                toGeneral[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                toGeneral[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                toGeneral[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                toGeneral[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                toGeneral[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGeneral[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toGeneral[i].image = m_Images[i];
                toGeneral[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }
            VkDependencyInfo toGeneralDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            toGeneralDep.imageMemoryBarrierCount = kTotalSlots;
            toGeneralDep.pImageMemoryBarriers = toGeneral.data();
            vkCmdPipelineBarrier2(cmd, &toGeneralDep);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, genPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, genPipelineLayout, 0, 1, &genDescriptorSet, 0, nullptr);
            uint32_t groupCount = (kTextureSize + 7u) / 8u;
            vkCmdDispatch(cmd, groupCount, groupCount, kTotalSlots);

            std::vector<VkImageMemoryBarrier2> toShaderReadOnly(kTotalSlots);
            for (uint32_t i = 0; i < kTotalSlots; ++i) {
                toShaderReadOnly[i] = VkImageMemoryBarrier2{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                toShaderReadOnly[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                toShaderReadOnly[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                toShaderReadOnly[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                toShaderReadOnly[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                toShaderReadOnly[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                toShaderReadOnly[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                toShaderReadOnly[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toShaderReadOnly[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toShaderReadOnly[i].image = m_Images[i];
                toShaderReadOnly[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }
            VkDependencyInfo toReadOnlyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            toReadOnlyDep.imageMemoryBarrierCount = kTotalSlots;
            toReadOnlyDep.pImageMemoryBarriers = toShaderReadOnly.data();
            vkCmdPipelineBarrier2(cmd, &toReadOnlyDep);
        });

        // --- Generation-only resources: no longer needed once every slot is populated. ---
        vkDestroyPipeline(m_Device, genPipeline, nullptr);
        vkDestroyPipelineLayout(m_Device, genPipelineLayout, nullptr);
        vkDestroyDescriptorPool(m_Device, genDescriptorPool, nullptr); // Frees genDescriptorSet implicitly.
        vkDestroyDescriptorSetLayout(m_Device, genSetLayout, nullptr);

        // --- Ready-to-bind image infos: slots 0..kLightFunctionSlotCount-1 for the bindless Light
        // Function array, slot kCausticsSlot alone for the caustics texture. ---
        m_LightFunctionImageInfos.resize(kLightFunctionSlotCount);
        for (uint32_t i = 0; i < kLightFunctionSlotCount; ++i) {
            m_LightFunctionImageInfos[i].sampler = m_Sampler;
            m_LightFunctionImageInfos[i].imageView = m_ImageViews[i];
            m_LightFunctionImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        m_CausticsImageInfo.sampler = m_Sampler;
        m_CausticsImageInfo.imageView = m_ImageViews[kCausticsSlot];
        m_CausticsImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        LOG_INFO(std::format("[ProceduralLightFunctionGenerator] Generated {} light-function texture(s) + 1 caustics texture ({}x{}, RGBA8_UNORM).",
            kLightFunctionSlotCount, kTextureSize, kTextureSize));
    }

    void ProceduralLightFunctionGenerator::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Sampler != VK_NULL_HANDLE) {
                vkDestroySampler(m_Device, m_Sampler, nullptr);
            }
            for (VkImageView view : m_ImageViews) {
                if (view != VK_NULL_HANDLE) {
                    vkDestroyImageView(m_Device, view, nullptr);
                }
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            for (size_t i = 0; i < m_Images.size(); ++i) {
                vmaDestroyImage(m_Allocator, m_Images[i], m_Allocations[i]);
            }
        }
        m_Images.clear();
        m_Allocations.clear();
        m_ImageViews.clear();
        m_LightFunctionImageInfos.clear();
        m_CausticsImageInfo = VkDescriptorImageInfo{};
        m_Sampler = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

}
