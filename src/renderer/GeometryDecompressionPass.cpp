#include "renderer/GeometryDecompressionPass.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h" // geometry::kMaxClusterVertices, geometry::kPageSizeBytes

namespace renderer {

    namespace {

        // Mirrors VulkanContext::ReadShaderFile (VulkanContext.cpp) -- duplicated rather than
        // shared because this class is deliberately self-contained (no VulkanContext dependency),
        // matching GpuGeometryPagePool's own independence from the rest of the renderer.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("GeometryDecompressionPass: failed to open SPIR-V file: " + filename);
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

        // Byte-for-byte layout match for DecompressPushConstants in
        // DecompressClusterVertices.comp: a uint plus 12 bytes of padding (16 bytes total, so the
        // following vec4 stays 16-byte-aligned), then two vec4 bounds (xyz used, w unused).
        struct DecompressPushConstants {
            uint32_t physicalPageIndex;
            uint32_t _pad0;
            uint32_t _pad1;
            uint32_t _pad2;
            float boundsMin[4];
            float boundsMax[4];
        };
        static_assert(sizeof(DecompressPushConstants) == 48,
            "DecompressPushConstants must match DecompressClusterVertices.comp's push_constant block exactly");

    } // namespace

    void GeometryDecompressionPass::Init(VkDevice device, VmaAllocator allocator, uint32_t maxPhysicalPages, VkBuffer compressedPhysicalPoolBuffer) {
        Shutdown();

        m_Device = device;
        m_MaxPhysicalPages = maxPhysicalPages;

        m_DecompressedVertexPool.Create(
            allocator,
            static_cast<VkDeviceSize>(maxPhysicalPages) * geometry::kMaxClusterVertices * kDecompressedVertexStrideBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // INDEX_BUFFER_BIT in addition to STORAGE_BUFFER_BIT: this pool is written as an ordinary
        // SSBO by DecompressClusterIndices.comp, but read directly by the fixed-function index
        // fetch stage once bound via vkCmdBindIndexBuffer (see renderer::ClusterHardwareRasterPass).
        m_DecompressedIndexPool.Create(
            allocator,
            static_cast<VkDeviceSize>(maxPhysicalPages) * geometry::kMaxClusterIndices * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Descriptor set layout: binding 0 is the compressed physical pool (read-only input,
        // shared by both dispatches), binding 1 is the decompressed final vertex pool (write-only,
        // DecompressClusterVertices.comp only), binding 2 is the decompressed final index pool
        // (write-only, DecompressClusterIndices.comp only). Both pipelines share this one set --
        // each simply never references the other's output binding.
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_DescriptorSetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
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

        VkDescriptorBufferInfo compressedInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo decompressedVertexInfo{ m_DecompressedVertexPool.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo decompressedIndexInfo{ m_DecompressedIndexPool.Handle(), 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_DescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &compressedInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_DescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &decompressedVertexInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = m_DescriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &decompressedIndexInfo;

        vkUpdateDescriptorSets(m_Device, 3, writes, 0, nullptr);

        // Shared pipeline layout: both DecompressClusterVertices.comp and
        // DecompressClusterIndices.comp declare a push-constant block that fits within this same
        // 48-byte range (the index shader's block is a strict prefix of it, only reading
        // physicalPageIndex) -- see DecompressPushConstants below.
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(DecompressPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        std::vector<char> vertexShaderCode = ReadShaderFile("shaders/DecompressClusterVertices.comp.spv");
        VkShaderModule vertexShaderModule = CreateShaderModule(m_Device, vertexShaderCode);

        VkComputePipelineCreateInfo vertexPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        vertexPipelineInfo.layout = m_PipelineLayout;
        vertexPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        vertexPipelineInfo.stage.module = vertexShaderModule;
        vertexPipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &vertexPipelineInfo, nullptr, &m_Pipeline));

        vkDestroyShaderModule(m_Device, vertexShaderModule, nullptr);

        std::vector<char> indexShaderCode = ReadShaderFile("shaders/DecompressClusterIndices.comp.spv");
        VkShaderModule indexShaderModule = CreateShaderModule(m_Device, indexShaderCode);

        VkComputePipelineCreateInfo indexPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        indexPipelineInfo.layout = m_PipelineLayout;
        indexPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        indexPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        indexPipelineInfo.stage.module = indexShaderModule;
        indexPipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &indexPipelineInfo, nullptr, &m_IndexPipeline));

        vkDestroyShaderModule(m_Device, indexShaderModule, nullptr);
    }

    void GeometryDecompressionPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            }
            if (m_IndexPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_IndexPipeline, nullptr);
            }
            if (m_PipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            }
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_DescriptorSet -- it was allocated from
                // this pool and never individually freed.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_DescriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_Device, m_DescriptorSetLayout, nullptr);
            }
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_IndexPipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_DescriptorSetLayout = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
        m_MaxPhysicalPages = 0;

        m_DecompressedVertexPool.Destroy();
        m_DecompressedIndexPool.Destroy();
    }

    void GeometryDecompressionPass::DecompressPage(VkCommandBuffer cmd, uint32_t physicalPageIndex, const maths::vec3& boundsMin, const maths::vec3& boundsMax) {
        assert(physicalPageIndex < m_MaxPhysicalPages && "physicalPageIndex out of range for the final vertex/index pools");

        DecompressPushConstants pushConstants{};
        pushConstants.physicalPageIndex = physicalPageIndex;
        pushConstants.boundsMin[0] = boundsMin.x;
        pushConstants.boundsMin[1] = boundsMin.y;
        pushConstants.boundsMin[2] = boundsMin.z;
        pushConstants.boundsMin[3] = 0.0f;
        pushConstants.boundsMax[0] = boundsMax.x;
        pushConstants.boundsMax[1] = boundsMax.y;
        pushConstants.boundsMax[2] = boundsMax.z;
        pushConstants.boundsMax[3] = 0.0f;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

        // Exactly one workgroup each: DecompressClusterVertices.comp's local_size_x is
        // geometry::kMaxClusterVertices (one thread per vertex), DecompressClusterIndices.comp's is
        // geometry::kMaxClusterIndices (one thread per local triangle-list index) -- both dispatch
        // over the one cluster resident at `physicalPageIndex`.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdDispatch(cmd, 1, 1, 1);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_IndexPipeline);
        vkCmdDispatch(cmd, 1, 1, 1);

        // Both compute writes must be visible to whatever later reads them: the vertex pool to a
        // later vertex-shader SSBO read (renderer::ClusterHardwareRasterPass decodes compressed
        // positions on the fly, but a fully-decompressed vertex pool read is still a documented,
        // supported consumer -- see the class comment), and the index pool to the fixed-function
        // index-fetch stage once bound via vkCmdBindIndexBuffer
        // (VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT / VK_ACCESS_2_INDEX_READ_BIT).
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

}
