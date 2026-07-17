#include "renderer/passes/DepthOfFieldPass.h"

#include <array>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {
        constexpr uint32_t kWorkgroupSize = 8;
    }

    void DepthOfFieldPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D displayExtent, VkImageView hdrColorView, VkImageView depthView) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_DisplayExtent = displayExtent;

        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kOutputFormat;
        imageInfo.extent = { displayExtent.width, displayExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_OutputImage, &m_OutputAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_OutputImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kOutputFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputView));

        VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_OutputImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

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
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_LinearSampler));

        m_ParamsBuffer.Create(allocator, sizeof(ParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        std::array<VkDescriptorSetLayoutBinding, 4> bindings{ {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_HDRColor
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_Depth
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_Output
            { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // DepthOfFieldParamsUBO
        } };
        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        std::array<VkDescriptorPoolSize, 3> poolSizes{ {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
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

        // g_Depth (binding 1) is renderer::ClusterResolvePass's own fixed, never-ping-ponged depth
        // view -- written once here; g_HDRColor (binding 0) is bound by UpdateSourceDescriptor()
        // below instead (renderer::TAATSRPass's own output view identity changes every frame).
        VkDescriptorImageInfo depthInfo{ m_LinearSampler, depthView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, m_OutputView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo paramsInfo{ m_ParamsBuffer.Handle(), 0, m_ParamsBuffer.Size() };

        std::array<VkWriteDescriptorSet, 3> writes{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &paramsInfo, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/DepthOfField.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shader);
        vkDestroyShaderModule(m_Device, shader, nullptr);

        UpdateSourceDescriptor(hdrColorView);

        LOG_INFO("[DepthOfFieldPass] Initialized (Phase PP3: physical-camera CoC + Poisson-disk gather).");
    }

    void DepthOfFieldPass::UpdateSourceDescriptor(VkImageView hdrColorView) {
        VkDescriptorImageInfo sourceInfo{ m_LinearSampler, hdrColorView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sourceInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
    }

    void DepthOfFieldPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_LinearSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_LinearSampler, nullptr);
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE && m_OutputImage != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
        }

        m_ParamsBuffer.Destroy();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Set = VK_NULL_HANDLE;
        m_LinearSampler = VK_NULL_HANDLE;
        m_OutputView = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE;
        m_OutputAllocation = VK_NULL_HANDLE;

        m_DisplayExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void DepthOfFieldPass::RecordGenerate(VkCommandBuffer cmd, const maths::mat4& invViewProj,
        const maths::vec3& cameraPositionWorld, float aperture, const Settings& settings) {
        ParamsUBO params{};
        params.invViewProj = invViewProj;
        params.cameraPosX = cameraPositionWorld.x;
        params.cameraPosY = cameraPositionWorld.y;
        params.cameraPosZ = cameraPositionWorld.z;
        params.aperture = aperture;
        params.focalLengthMM = settings.focalLengthMM;
        params.focusDistanceWorldUnits = settings.focusDistanceWorldUnits;
        params.maxCoCRadiusPixels = settings.maxCoCRadiusPixels;
        vkCmdUpdateBuffer(cmd, m_ParamsBuffer.Handle(), 0, sizeof(ParamsUBO), &params);

        VkMemoryBarrier2 uploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        uploadBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        VkDependencyInfo uploadDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        uploadDep.memoryBarrierCount = 1;
        uploadDep.pMemoryBarriers = &uploadBarrier;
        vkCmdPipelineBarrier2(cmd, &uploadDep);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);
        uint32_t groupsX = (m_DisplayExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupsY = (m_DisplayExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo outputDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        outputDep.memoryBarrierCount = 1;
        outputDep.pMemoryBarriers = &outputBarrier;
        vkCmdPipelineBarrier2(cmd, &outputDep);
    }

}
