#include "renderer/debug/ParticleDebugViewPass.h"
#ifndef NDEBUG

#include "core/Logger.h"
#include "renderer/passes/ParticleSystemPass.h" // kMaxParticles -- see this file's own static_assert below.
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer::debug {

    namespace {

        // This pass' own core invariant (see ParticleDebugViewPass.h's own kGridWidth/kGridHeight
        // comment): one pixel per particle slot, so the grid must hold EXACTLY kMaxParticles
        // pixels -- checked at compile time so a future change to either constant fails the build
        // instead of silently producing an incomplete/overflowing debug view.
        static_assert(static_cast<uint32_t>(ParticleDebugViewPass::kGridWidth) * static_cast<uint32_t>(ParticleDebugViewPass::kGridHeight)
            == ParticleSystemPass::kMaxParticles,
            "ParticleDebugViewPass's grid must have exactly one pixel per renderer::ParticleSystemPass::kMaxParticles slot");

        // Matches renderer::GpuParticle's own static_assert (ParticleSystemPass.h) -- sizes the
        // Init()-time dummy buffer to exactly one real particle's worth of bytes.
        constexpr VkDeviceSize kDummyParticleBufferBytes = 80;

        // Byte-for-byte mirror of ParticleDebugView.comp's own ParticleDebugViewPC push-constant block.
        struct ParticleDebugViewPC {
            uint32_t particleCount = 0; // renderer::ParticleSystemPass::kMaxParticles.
            uint32_t maxEmitters = 0;   // renderer::ParticleSystemPass::kMaxEmitters.
        };
        static_assert(sizeof(ParticleDebugViewPC) == 8, "ParticleDebugViewPC must match ParticleDebugView.comp's own push-constant block exactly");

    } // namespace

    void ParticleDebugViewPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // Owned output image (rgba8, kGridWidth x kGridHeight -- see class comment).
        // =====================================================================================
        {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = kOutputFormat;
            imageInfo.extent = { kGridWidth, kGridHeight, 1 };
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
            // shader both imageStore's into it (RecordBake) and it is later sampled from
            // (renderer::debug::DebugBufferViewPass::RecordView's own g_Source), matching every
            // other Buffer Viewer candidate buffer's own established "always GENERAL" convention.
            VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_OutputImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // Init()-time dummy the g_Particles binding starts out pointing at -- see Init()'s own
        // header comment. Zero-filled is fine: never actually dispatched against before the first
        // real RecordBake() call rewrites the descriptor.
        m_DummyParticleBuffer.Create(allocator, kDummyParticleBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // =====================================================================================
        // Pipeline: set 0 (2 bindings). g_Particles (rewritten every RecordBake() call) / g_Output.
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 2;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 } };
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

        VkDescriptorBufferInfo particleInfo{ m_DummyParticleBuffer.Handle(), 0, m_DummyParticleBuffer.Size() };
        VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, m_OutputView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &particleInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ParticleDebugViewPC) };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleDebugView.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[ParticleDebugViewPass] Initialized.");
    }

    void ParticleDebugViewPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
        }
        m_DummyParticleBuffer.Destroy();

        m_Pipeline = VK_NULL_HANDLE; m_PipelineLayout = VK_NULL_HANDLE; m_DescriptorPool = VK_NULL_HANDLE; m_SetLayout = VK_NULL_HANDLE; m_Set = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE; m_OutputAllocation = VK_NULL_HANDLE; m_OutputView = VK_NULL_HANDLE;

        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ParticleDebugViewPass::RecordBake(VkCommandBuffer cmd, VkBuffer particleBuffer, VkDeviceSize particleBufferSizeBytes, uint32_t maxEmitters) {
        VkDescriptorBufferInfo particleInfo{ particleBuffer, 0, particleBufferSizeBytes };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &particleInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

        ParticleDebugViewPC pc{};
        pc.particleCount = ParticleSystemPass::kMaxParticles;
        pc.maxEmitters = maxEmitters;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupCountX = (kGridWidth + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupCountY = (kGridHeight + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // Makes this bake's own output visible to renderer::debug::DebugBufferViewPass::RecordView's
        // own g_Source COMBINED_IMAGE_SAMPLER read (a sampled read, not a storage-buffer read --
        // hence VK_ACCESS_2_SHADER_SAMPLED_READ_BIT here, unlike the SHADER_STORAGE_READ_BIT this
        // codebase's other compute-to-compute handoffs use).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
#endif
