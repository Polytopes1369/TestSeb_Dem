#include "renderer/ATrousDenoisePass.h"

#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/Logger.h"

namespace renderer {

    namespace {

        // Mirrors every other pass' own copy of these two helpers -- see GlobalSDFPass.cpp's own
        // comment on this codebase's per-pass self-containment convention.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("ATrousDenoisePass: failed to open SPIR-V file: " + filename);
            }
            size_t fileSize = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            file.close();
            return buffer;
        }

        VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code) {
            VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &module));
            return module;
        }

        // Byte-for-byte layout match for ATrousPushConstants in ATrousDenoise.comp.
        struct ATrousPushConstants {
            int32_t stepSize = 1;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(ATrousPushConstants) == 16,
            "ATrousPushConstants must match ATrousDenoise.comp's push_constant block exactly");

        // Step sizes {1,2,4,8,16} -- see ATrousDenoisePass::kIterations' own comment.
        constexpr int32_t kStepSizes[ATrousDenoisePass::kIterations] = { 1, 2, 4, 8, 16 };

    } // namespace

    void ATrousDenoisePass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, VkImageView noisyInputView, VkImageView depthView, VkImageView normalView) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        // --- 2 ping-pong images ---
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kFormat;
        imageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_SRC_BIT: this pass' output can be the final blit source (renderer::
        // ClusterRenderPipeline's own `blitSourceImage` selection). COLOR_ATTACHMENT_BIT, Debug
        // builds only: renderer::debug::DebugTextOverlay::RecordDraw draws its stat overlay
        // directly onto this image via a graphics pipeline / vkCmdBeginRendering (see that class'
        // own comment) whenever ClusterRenderPipeline's `applyDenoise` selects it -- a Release
        // build never calls RecordDraw at all, so this extra usage flag would be pure waste there.
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#ifndef NDEBUG
            | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
#endif
            ;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        for (uint32_t i = 0; i < 2; ++i) {
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_PingImages[i], &m_PingAllocations[i], nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_PingImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = kFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_PingViews[i]));
        }

        // One-time UNDEFINED -> GENERAL transition for both ping images (mirrors
        // ClusterResolvePass::Init's own one-shot pattern).
        {
            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;

            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

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
                barriers[i].image = m_PingImages[i];
                barriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 2;
            depInfo.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));

            vkFreeCommandBuffers(m_Device, commandPool, 1, &cmd);
        }

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_NearestSampler));

        // --- Descriptor set layout: 4 bindings (input, output, depth, normal), shared by all 3
        // per-iteration sets below (only the underlying image VIEWS differ between them). ---
        VkDescriptorSetLayoutBinding bindings[4]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_Input
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_Output
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_Depth
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_Normal

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 4;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[2] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 3 }, // input + depth + normal, x3 sets.
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 3;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout setLayouts[3] = { m_SetLayout, m_SetLayout, m_SetLayout };
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 3;
        setAllocInfo.pSetLayouts = setLayouts;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_Sets));

        // Per-set (input, output) pairs: [0] external -> ping[0]; [1] ping[0] -> ping[1];
        // [2] ping[1] -> ping[0] -- see RecordDenoise()'s own iteration-order comment.
        VkImageView setInputs[3] = { noisyInputView, m_PingViews[0], m_PingViews[1] };
        VkImageView setOutputs[3] = { m_PingViews[0], m_PingViews[1], m_PingViews[0] };

        for (uint32_t i = 0; i < 3; ++i) {
            VkDescriptorImageInfo inputInfo{ m_NearestSampler, setInputs[i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, setOutputs[i], VK_IMAGE_LAYOUT_GENERAL };
            // GENERAL, not DEPTH_STENCIL_READ_ONLY_OPTIMAL: `depthView` is renderer::
            // ClusterResolvePass::GetOutputDepthView(), a plain COLOR-aspect R32_SFLOAT GBuffer
            // image (the winning hw-vs-sw arbitrated NDC depth, not a real depth-attachment image)
            // kept in VK_IMAGE_LAYOUT_GENERAL for its entire lifetime, like every other GBuffer
            // image renderer::ClusterResolvePass owns.
            VkDescriptorImageInfo depthInfo{ m_NearestSampler, depthView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo normalInfo{ m_NearestSampler, normalView, VK_IMAGE_LAYOUT_GENERAL };

            VkWriteDescriptorSet writes[4]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &inputInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(ATrousPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_SetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = CreateShaderModule(m_Device, ReadShaderFile("shaders/ATrousDenoise.comp.spv"));
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[ATrousDenoisePass] Initialized 5-pass A-Trous spatial denoiser.");
    }

    void ATrousDenoisePass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_NearestSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_NearestSampler, nullptr);
            for (uint32_t i = 0; i < 2; ++i) {
                if (m_PingViews[i] != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_PingViews[i], nullptr);
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            for (uint32_t i = 0; i < 2; ++i) {
                vmaDestroyImage(m_Allocator, m_PingImages[i], m_PingAllocations[i]);
            }
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        for (uint32_t i = 0; i < 3; ++i) m_Sets[i] = VK_NULL_HANDLE;
        m_NearestSampler = VK_NULL_HANDLE;
        for (uint32_t i = 0; i < 2; ++i) {
            m_PingImages[i] = VK_NULL_HANDLE;
            m_PingAllocations[i] = VK_NULL_HANDLE;
            m_PingViews[i] = VK_NULL_HANDLE;
        }

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ATrousDenoisePass::RecordDenoise(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;

        for (uint32_t i = 0; i < kIterations; ++i) {
            // Set selection -- see the header's own m_Sets comment for the exact (src, dst) pair
            // each index represents.
            uint32_t setIndex = (i == 0) ? 0 : ((i % 2 == 1) ? 1 : 2);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Sets[setIndex], 0, nullptr);

            ATrousPushConstants pc{};
            pc.stepSize = kStepSizes[i];
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

            vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

            // Every iteration's output must be visible before the NEXT iteration's read of that
            // same image (the last iteration's barrier instead serves the caller's own next read
            // of GetOutputView()) -- one barrier per iteration covers both cases identically.
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
    }

}
