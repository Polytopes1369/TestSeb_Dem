#ifndef NDEBUG

#include "renderer/debug/ClusterTriangleStatsPass.h"

#include <format>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"

namespace renderer::debug {


    void ClusterTriangleStatsPass::Init(VkDevice device, VmaAllocator allocator, uint32_t maxClusters,
        VkBuffer clusterMetadataBuffer, VkBuffer earlyIndirectCommandBuffer, VkBuffer earlyDrawCountBuffer,
        VkBuffer lateIndirectCommandBuffer, VkBuffer lateDrawCountBuffer, VkBuffer softwareClusterListBuffer,
        VkBuffer earlyIndirectCommandOpaqueBuffer, VkBuffer earlyDrawCountOpaqueBuffer,
        VkBuffer lateIndirectCommandOpaqueBuffer, VkBuffer lateDrawCountOpaqueBuffer,
        VkBuffer softwareClusterListOpaqueBuffer) {
        Shutdown();

        m_Device = device;
        m_MaxClusters = maxClusters;

        m_StatsBuffer.Create(allocator, 2 * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        m_StatsReadbackBuffer.Create(allocator, 2 * sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);

        VkDescriptorSetLayoutBinding bindings[12]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // EarlyIndirectCommandsSSBO (masked)
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // EarlyDrawCountSSBO (masked)
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // LateIndirectCommandsSSBO (masked)
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // LateDrawCountSSBO (masked)
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // ClusterCullMetadataSSBO
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // SoftwareClusterListSSBO (masked)
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // TriangleCountStatsSSBO
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // EarlyIndirectCommandsOpaqueSSBO
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // EarlyDrawCountOpaqueSSBO
        bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // LateIndirectCommandsOpaqueSSBO
        bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // LateDrawCountOpaqueSSBO
        bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SoftwareClusterListOpaqueSSBO

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 12;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAlloc.descriptorPool = m_DescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_DescriptorSet));

        VkDescriptorBufferInfo earlyIndirectInfo{ earlyIndirectCommandBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo earlyDrawCountInfo{ earlyDrawCountBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo lateIndirectInfo{ lateIndirectCommandBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo lateDrawCountInfo{ lateDrawCountBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo clusterMetadataInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo softwareListInfo{ softwareClusterListBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo statsInfo{ m_StatsBuffer.Handle(), 0, m_StatsBuffer.Size() };
        VkDescriptorBufferInfo earlyIndirectOpaqueInfo{ earlyIndirectCommandOpaqueBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo earlyDrawCountOpaqueInfo{ earlyDrawCountOpaqueBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo lateIndirectOpaqueInfo{ lateIndirectCommandOpaqueBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo lateDrawCountOpaqueInfo{ lateDrawCountOpaqueBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo softwareListOpaqueInfo{ softwareClusterListOpaqueBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[12]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyIndirectInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyDrawCountInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateIndirectInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateDrawCountInfo, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetadataInfo, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &softwareListInfo, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &statsInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyIndirectOpaqueInfo, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyDrawCountOpaqueInfo, nullptr };
        writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateIndirectOpaqueInfo, nullptr };
        writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateDrawCountOpaqueInfo, nullptr };
        writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &softwareListOpaqueInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 12, writes, 0, nullptr);

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(uint32_t); // mode

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ComputeTriangleStats.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shaderModule);
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO(std::format("[ClusterTriangleStatsPass] Initialized triangle stats pass: maxClusters={}", maxClusters));
    }

    void ClusterTriangleStatsPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[ClusterTriangleStatsPass] Shutting down triangle stats pass...");
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;

        m_StatsBuffer.Destroy();
        m_StatsReadbackBuffer.Destroy();
        m_MaxClusters = 0;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterTriangleStatsPass::RecordClear(VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_StatsBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void ClusterTriangleStatsPass::RecordCompute(VkCommandBuffer cmd) {
        uint32_t groupCount = (m_MaxClusters + kWorkgroupSize - 1) / kWorkgroupSize;
        if (groupCount == 0) {
            return;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

        for (uint32_t mode = 0; mode < 6; ++mode) {
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &mode);
            vkCmdDispatch(cmd, groupCount, 1, 1);
        }

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void ClusterTriangleStatsPass::RecordReadback(VkCommandBuffer cmd) {
        // RecordCompute()'s own trailing barrier already made the device buffer's writes visible
        // to VK_PIPELINE_STAGE_2_COPY_BIT / VK_ACCESS_2_TRANSFER_READ_BIT, so the copy below needs
        // no further src-side barrier -- only the dst-side one ordering the copy before the host
        // read (mirrors renderer::FeedbackBuffer::RecordReadback() exactly).
        VkBufferCopy copyRegion{ 0, 0, 2 * sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_StatsBuffer.Handle(), m_StatsReadbackBuffer.Handle(), 1, &copyRegion);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void ClusterTriangleStatsPass::ReadStats(uint32_t& hwTriangleCount, uint32_t& swTriangleCount) const {
        const uint32_t* data = static_cast<const uint32_t*>(m_StatsReadbackBuffer.MappedData());
        hwTriangleCount = data[0];
        swTriangleCount = data[1];
    }

}

#endif // NDEBUG
