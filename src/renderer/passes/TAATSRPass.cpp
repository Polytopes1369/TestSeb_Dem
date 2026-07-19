#include "renderer/passes/TAATSRPass.h"

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

#include <cstring>
#include <format>
#include "core/EngineConfig.h"

namespace renderer {

    void TAATSRPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, VkExtent2D displayExtent,
        VkImageView lowResColorView, VkImageView lowResDepthView) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;
        m_DisplayExtent = displayExtent;

        LOG_INFO(std::format("[TAATSRPass] Initializing: Render Scale {}x{} -> Display {}x{}",
            renderExtent.width, renderExtent.height, displayExtent.width, displayExtent.height));

        // 1. Create ping-pong history images in display (native) resolution
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kHistoryFormat;
        imageInfo.extent = { displayExtent.width, displayExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // STORAGE_BIT: written by compute. SAMPLED_BIT: read by compute next frame. TRANSFER_SRC_BIT: blitted to swapchain.
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
#ifndef NDEBUG
        imageInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // For debug overlay text drawing if targeted directly
#endif
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        for (uint32_t i = 0; i < 2; ++i) {
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_HistoryImages[i], &m_HistoryAllocations[i], nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_HistoryImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = kHistoryFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_HistoryImageViews[i]));
        }

        // Perform one-time UNDEFINED -> GENERAL layout transition for both history images
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 barriers[2]{};
            for (uint32_t i = 0; i < 2; ++i) {
                barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                barriers[i].srcAccessMask = 0;
                barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].image = m_HistoryImages[i];
                barriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 2;
            depInfo.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // 2. Samplers
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

        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_NearestSampler));

        // 3. UBO (mapped = true)
        m_ViewParamsBuffer.Create(allocator, sizeof(TAATSRViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, true);

        // 4. Descriptor set layout:
        // Binding 0: low-res color (Sampled)
        // Binding 1: low-res depth (Sampled)
        // Binding 2: history input (Sampled)
        // Binding 3: history output (Storage)
        // Binding 4: view parameters (Uniform)
        VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 5;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        // Descriptor Pool
        VkDescriptorPoolSize poolSizes[3] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 * 2 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout setLayouts[2] = { m_SetLayout, m_SetLayout };
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 2;
        setAllocInfo.pSetLayouts = setLayouts;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_DescriptorSets));

        UpdateDescriptorSets(lowResColorView, lowResDepthView);

        // 5. Create compute pipeline
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/TAATSR.comp.spv");
        VkComputePipelineCreateInfo computePipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        computePipelineInfo.layout = m_PipelineLayout;
        computePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computePipelineInfo.stage.module = shaderModule;
        computePipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &computePipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);
    }

    void TAATSRPass::Shutdown() {
        if (!m_Device) return;

        if (m_Pipeline) {
            vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            m_Pipeline = VK_NULL_HANDLE;
        }
        if (m_PipelineLayout) {
            vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            m_PipelineLayout = VK_NULL_HANDLE;
        }
        if (m_DescriptorPool) {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            m_DescriptorPool = VK_NULL_HANDLE;
        }
        if (m_SetLayout) {
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            m_SetLayout = VK_NULL_HANDLE;
        }
        if (m_LinearSampler) {
            vkDestroySampler(m_Device, m_LinearSampler, nullptr);
            m_LinearSampler = VK_NULL_HANDLE;
        }
        if (m_NearestSampler) {
            vkDestroySampler(m_Device, m_NearestSampler, nullptr);
            m_NearestSampler = VK_NULL_HANDLE;
        }
        m_ViewParamsBuffer.Destroy();

        for (uint32_t i = 0; i < 2; ++i) {
            if (m_HistoryImageViews[i]) {
                vkDestroyImageView(m_Device, m_HistoryImageViews[i], nullptr);
                m_HistoryImageViews[i] = VK_NULL_HANDLE;
            }
            if (m_HistoryImages[i]) {
                vmaDestroyImage(m_Allocator, m_HistoryImages[i], m_HistoryAllocations[i]);
                m_HistoryImages[i] = VK_NULL_HANDLE;
                m_HistoryAllocations[i] = VK_NULL_HANDLE;
            }
        }
        m_Device = VK_NULL_HANDLE;
    }

    void TAATSRPass::UpdateDescriptorSets(VkImageView lowResColorView, VkImageView lowResDepthView) {
        // [0] reads m_HistoryImageViews[1] (historyInput), writes m_HistoryImageViews[0] (historyOutput)
        // [1] reads m_HistoryImageViews[0] (historyInput), writes m_HistoryImageViews[1] (historyOutput)
        for (uint32_t i = 0; i < 2; ++i) {
            VkDescriptorImageInfo colorInfo{ m_NearestSampler, lowResColorView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo depthInfo{ m_NearestSampler, lowResDepthView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo historyInputInfo{ m_LinearSampler, m_HistoryImageViews[1 - i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo historyOutputInfo{ VK_NULL_HANDLE, m_HistoryImageViews[i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo uboInfo{ m_ViewParamsBuffer.Handle(), 0, sizeof(TAATSRViewParamsUBO) };

            VkWriteDescriptorSet writes[5]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &colorInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &historyInputInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &historyOutputInfo, nullptr, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr };

            vkUpdateDescriptorSets(m_Device, 5, writes, 0, nullptr);
        }
    }

    void TAATSRPass::RecordPass(VkCommandBuffer cmd,
        const maths::mat4& viewProj,
        const maths::mat4& prevViewProj,
        const maths::mat4& invViewProj,
        float jitterX, float jitterY,
        uint32_t frameIndex,
        bool resetHistory) {

        // 1. Update Uniform Buffer
        TAATSRViewParamsUBO ubo{};
        ubo.viewProj = viewProj;
        ubo.prevViewProj = prevViewProj;
        ubo.invViewProj = invViewProj;
        ubo.renderExtent = { static_cast<float>(m_RenderExtent.width), static_cast<float>(m_RenderExtent.height) };
        ubo.displayExtent = { static_cast<float>(m_DisplayExtent.width), static_cast<float>(m_DisplayExtent.height) };
        ubo.jitterOffset = { jitterX, jitterY };
        ubo.frameIndex = frameIndex;
        ubo.resetHistory = resetHistory ? 1u : 0u;
        ubo.blendAlpha = config::temporal::BLEND_ALPHA;
        ubo.blendAlphaStatic = config::temporal::BLEND_ALPHA_STATIC;
        ubo.varianceClampFactor = config::temporal::VARIANCE_CLAMP_FACTOR;
        std::memcpy(m_ViewParamsBuffer.MappedData(), &ubo, sizeof(TAATSRViewParamsUBO));

        // Barrier for UBO upload
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        // 2. Pre-dispatch barriers: ensure low-res inputs and history images are ready.
        // We sync:
        // - low-res inputs (written in compute by prior passes) to compute reads.
        // - history input (written last frame by compute) to compute reads.
        // - history output (target slot) to compute storage writes.
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // 3. Bind Pipeline and Descriptor Set
        // m_CurrentHistoryIndex is the index we WRITE to.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSets[m_CurrentHistoryIndex], 0, nullptr);

        // 4. Dispatch compute shader (one thread per native output pixel, workgroups of 8x8)
        uint32_t groupCountX = (m_DisplayExtent.width + 7) / 8;
        uint32_t groupCountY = (m_DisplayExtent.height + 7) / 8;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // 5. Post-dispatch barrier: make the newly written history image visible for subsequent swapchain blits (TRANSFER read).
        {
            VkImageMemoryBarrier2 blitBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            blitBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            blitBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            blitBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            blitBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            blitBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            blitBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            blitBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            blitBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            blitBarrier.image = m_HistoryImages[m_CurrentHistoryIndex];
            blitBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &blitBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // 6. Ping-pong swap for next frame
        m_CurrentHistoryIndex = 1 - m_CurrentHistoryIndex;
    }

} // namespace renderer
