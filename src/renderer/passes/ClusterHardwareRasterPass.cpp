#include "renderer/passes/ClusterHardwareRasterPass.h"

#include <format>

#include "core/Logger.h"
#include "renderer/passes/GeometryDecompressionPass.h"
#include "renderer/vulkan/VulkanPipeline.h"

namespace renderer {


    void ClusterHardwareRasterPass::Init(VkDevice device, VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
        VkBuffer wpoGlobalsBuffer, const std::vector<VkDescriptorImageInfo>& maskImageInfos,
        const std::array<VkFormat, 2>& visBufferColorFormats, VkFormat depthFormat,
        VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer) {
        Shutdown();

        m_Device = device;
        uint32_t maskTextureCount = static_cast<uint32_t>(maskImageInfos.size());

        // --- Descriptor set layout: 6 bindings -- 0/1/2/4/5 vertex-stage-only, 3 (the bindless mask
        // array) fragment-stage-only, since ClusterRaster.vert never samples it. ---
        VkDescriptorSetLayoutBinding bindings[6]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // ClusterCullMetadataSSBO
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // CompressedClusterPoolSSBO
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // WPOGlobalsUBO
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // g_MaskTextures[] (mask_sampling.glsl)
        bindings[3].descriptorCount = maskTextureCount;
        bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // EntityTransformBuffer
        bindings[4].descriptorCount = 1;
        bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[5].binding = 5;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // EntityDataBuffer
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 6;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_DescriptorSetLayout));

        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_DescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_DescriptorSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, &m_DescriptorSet));

        VkDescriptorBufferInfo metadataInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo compressedInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo wpoGlobalsInfo{ wpoGlobalsBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo entityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[6]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_DescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &metadataInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_DescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &compressedInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = m_DescriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].pBufferInfo = &wpoGlobalsInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = m_DescriptorSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = maskTextureCount;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = maskImageInfos.data();

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = m_DescriptorSet;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &entityTransformInfo;

        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = m_DescriptorSet;
        writes[5].dstBinding = 5;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].pBufferInfo = &entityDataInfo;

        vkUpdateDescriptorSets(m_Device, 6, writes, 0, nullptr);

        // --- Pipeline layout + pipeline: reuses VulkanPipeline::CreateGraphicsPipeline exactly as
        // draw.vert/draw.frag's own pipeline does (empty vertex input state, dynamic viewport/
        // scissor, back-face culling, 2-attachment VisBuffer color blend state, depth test/write
        // enabled) -- see VulkanPipeline.cpp. ---
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(CameraPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterRaster.vert.spv");
        VkShaderModule maskedFragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterRaster.frag.spv");
        VkShaderModule opaqueFragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterRasterOpaque.frag.spv");

        m_MaskedPipeline = VulkanPipeline::CreateGraphicsPipeline(m_Device, m_PipelineLayout, vertModule, maskedFragModule, visBufferColorFormats, depthFormat);
        m_OpaquePipeline = VulkanPipeline::CreateGraphicsPipeline(m_Device, m_PipelineLayout, vertModule, opaqueFragModule, visBufferColorFormats, depthFormat);

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, maskedFragModule, nullptr);
        vkDestroyShaderModule(m_Device, opaqueFragModule, nullptr);

        LOG_INFO(std::format("[ClusterHardwareRasterPass] Initialized hardware raster pass: maskTextures={}", maskTextureCount));
    }

    void ClusterHardwareRasterPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[ClusterHardwareRasterPass] Shutting down hardware raster pass...");
            if (m_OpaquePipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_OpaquePipeline, nullptr);
            }
            if (m_MaskedPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_MaskedPipeline, nullptr);
            }
            if (m_PipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            }
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_DescriptorSet -- not freed individually.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_DescriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_Device, m_DescriptorSetLayout, nullptr);
            }
        }

        m_OpaquePipeline = VK_NULL_HANDLE;
        m_MaskedPipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_DescriptorSetLayout = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterHardwareRasterPass::RecordDraw(VkCommandBuffer cmd, const CameraPushConstants& camera, VkExtent2D renderExtent,
        VkBuffer decompressedIndexPoolBuffer, VkBuffer indirectCommandBuffer, VkBuffer drawCountBuffer,
        uint32_t maxDrawCount, bool opaque) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, opaque ? m_OpaquePipeline : m_MaskedPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(CameraPushConstants), &camera);

        // GeometryDecompressionPass::kDecompressedIndexType is VK_INDEX_TYPE_UINT32 -- see that
        // class's comment for why UINT32 was chosen over UINT16 (no shaderInt16 feature/extension
        // dependency).
        vkCmdBindIndexBuffer(cmd, decompressedIndexPoolBuffer, 0, GeometryDecompressionPass::kDecompressedIndexType);

        VkViewport viewport{};
        viewport.width = static_cast<float>(renderExtent.width);
        viewport.height = static_cast<float>(renderExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = renderExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // One indirect multi-draw over every surviving cluster this frame -- the actual sub-draw
        // count is read from drawCountBuffer entirely on the GPU at execution time; maxDrawCount
        // only bounds how many VkDrawIndexedIndirectCommand-sized slots the driver may iterate,
        // matching the culling pass's own buffer capacity (see
        // renderer::ClusterCullingPass::GetMaxClusters() / ClusterOcclusionCullingPass::GetMaxClusters()).
        vkCmdDrawIndexedIndirectCount(cmd, indirectCommandBuffer, 0, drawCountBuffer, 0, maxDrawCount, sizeof(VkDrawIndexedIndirectCommand));
    }

}
