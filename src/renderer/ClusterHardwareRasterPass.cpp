#include "renderer/ClusterHardwareRasterPass.h"

#include <fstream>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/GeometryDecompressionPass.h" // kDecompressedIndexType
#include "renderer/VulkanPipeline.h"

namespace renderer {

    namespace {

        // Mirrors HZBPass::ReadShaderFile / every other pass's own copy -- duplicated rather than
        // shared because this class is deliberately self-contained (no VulkanContext dependency),
        // matching this codebase's existing per-pass convention.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("ClusterHardwareRasterPass: failed to open SPIR-V file: " + filename);
            }
            size_t fileSize = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            file.close();
            return buffer;
        }

    } // namespace

    void ClusterHardwareRasterPass::Init(VkDevice device, VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
        const std::array<VkFormat, 2>& visBufferColorFormats, VkFormat depthFormat) {
        Shutdown();

        m_Device = device;

        // --- Descriptor set layout: 2 bindings, both vertex-stage-only (ClusterRaster.frag reads
        // no descriptors at all). ---
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // ClusterCullMetadataSSBO
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // CompressedClusterPoolSSBO
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_DescriptorSetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_DescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_DescriptorSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, &m_DescriptorSet));

        VkDescriptorBufferInfo metadataInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo compressedInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[2]{};
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

        vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

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

        std::vector<char> vertCode = ReadShaderFile("shaders/ClusterRaster.vert.spv");
        std::vector<char> fragCode = ReadShaderFile("shaders/ClusterRaster.frag.spv");
        VkShaderModule vertModule = VulkanPipeline::CreateShaderModule(m_Device, vertCode);
        VkShaderModule fragModule = VulkanPipeline::CreateShaderModule(m_Device, fragCode);

        m_Pipeline = VulkanPipeline::CreateGraphicsPipeline(m_Device, m_PipelineLayout, vertModule, fragModule, visBufferColorFormats, depthFormat);

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, fragModule, nullptr);
    }

    void ClusterHardwareRasterPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
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

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_DescriptorSetLayout = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterHardwareRasterPass::RecordDraw(VkCommandBuffer cmd, const CameraPushConstants& camera, VkExtent2D renderExtent,
        VkBuffer decompressedIndexPoolBuffer, VkBuffer indirectCommandBuffer, VkBuffer drawCountBuffer, uint32_t maxDrawCount) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
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
