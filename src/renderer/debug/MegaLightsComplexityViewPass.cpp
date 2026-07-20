#include "renderer/debug/MegaLightsComplexityViewPass.h"
#ifndef NDEBUG

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer::debug {

    namespace {

        // Byte-for-byte mirror of MegaLightsComplexityView.comp's own ComplexityPC push-constant
        // block: a std430-packed mat4 (64 bytes) followed by a uvec2 (8 bytes). 72 bytes total,
        // comfortably under Vulkan's 128-byte guaranteed push-constant budget.
        struct ComplexityPC {
            maths::mat4 invViewProj;
            uint32_t viewportWidth = 0;
            uint32_t viewportHeight = 0;
        };
        static_assert(sizeof(ComplexityPC) == 72, "ComplexityPC must match MegaLightsComplexityView.comp's own push-constant block exactly");

        // One empty g_Lights header's worth of bytes (lightCount + 3 reserved -- see
        // megalights_ris.glsl's own MegaLightsSSBO comment): the Init()-time dummy reads as
        // lightCount == 0, so even an accidental pre-RecordBake dispatch would loop zero times.
        constexpr VkDeviceSize kDummyLightBufferBytes = 16;

    } // namespace

    void MegaLightsComplexityViewPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        // =====================================================================================
        // Owned output image (rgba8, one pixel per render-extent pixel -- the heatmap is a
        // full-resolution screen-space visualization, unlike ParticleDebugViewPass's fixed grid).
        // =====================================================================================
        {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = kOutputFormat;
            imageInfo.extent = { m_RenderExtent.width, m_RenderExtent.height, 1 };
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

            // Kept in VK_IMAGE_LAYOUT_GENERAL for its entire lifetime -- this pass' own compute
            // shader imageStore's into it (RecordBake) and renderer::debug::DebugBufferViewPass::
            // RecordView later samples it, matching every other Buffer Viewer candidate image's
            // established "always GENERAL" convention.
            VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_OutputImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // Init()-time dummies -- see this class' own header comment. The dummy light buffer is
        // GPU_ONLY zero-fill (VMA guarantees no contents; a zeroed header reads lightCount == 0,
        // inert by construction), the dummy depth image a 1x1 r32f storage image in GENERAL.
        m_DummyLightBuffer.Create(allocator, kDummyLightBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = VK_FORMAT_R32_SFLOAT;
            imageInfo.extent = { 1, 1, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_DummyDepthImage, &m_DummyDepthAllocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_DummyDepthImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R32_SFLOAT;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_DummyDepthView));

            VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_DummyDepthImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }

        // =====================================================================================
        // Pipeline: set 0 (3 bindings) -- g_Lights (rewritten every RecordBake) / g_Output /
        // g_GBufferDepth (also rewritten every RecordBake, same "cheap unconditional rewrite"
        // convention as DebugBufferViewPass::RecordView's own g_Source).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 3;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 } };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_Set));

        VkDescriptorBufferInfo lightInfo{ m_DummyLightBuffer.Handle(), 0, m_DummyLightBuffer.Size() };
        VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, m_OutputView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo depthInfo{ VK_NULL_HANDLE, m_DummyDepthView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet writes[3]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &depthInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 3, writes, 0, nullptr);

        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComplexityPC) };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/MegaLightsComplexityView.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[MegaLightsComplexityViewPass] Initialized.");
    }

    void MegaLightsComplexityViewPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_DummyDepthView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_DummyDepthView, nullptr);
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_DummyDepthImage, m_DummyDepthAllocation);
            vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
        }
        m_DummyLightBuffer.Destroy();

        m_Pipeline = VK_NULL_HANDLE; m_PipelineLayout = VK_NULL_HANDLE; m_DescriptorPool = VK_NULL_HANDLE; m_SetLayout = VK_NULL_HANDLE; m_Set = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE; m_OutputAllocation = VK_NULL_HANDLE; m_OutputView = VK_NULL_HANDLE;
        m_DummyDepthImage = VK_NULL_HANDLE; m_DummyDepthAllocation = VK_NULL_HANDLE; m_DummyDepthView = VK_NULL_HANDLE;

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void MegaLightsComplexityViewPass::RecordBake(VkCommandBuffer cmd, VkBuffer lightBuffer, VkDeviceSize lightBufferSizeBytes,
        VkImageView gbufferDepthView, const maths::mat4& invViewProj) {
        VkDescriptorBufferInfo lightInfo{ lightBuffer, 0, lightBufferSizeBytes };
        VkDescriptorImageInfo depthInfo{ VK_NULL_HANDLE, gbufferDepthView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &depthInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

        ComplexityPC pc{};
        pc.invViewProj = invViewProj;
        pc.viewportWidth = m_RenderExtent.width;
        pc.viewportHeight = m_RenderExtent.height;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // Makes this bake's own output visible to renderer::debug::DebugBufferViewPass::RecordView's
        // own g_Source COMBINED_IMAGE_SAMPLER read -- identical contract/rationale as
        // ParticleDebugViewPass::RecordBake's own trailing barrier.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
#endif
