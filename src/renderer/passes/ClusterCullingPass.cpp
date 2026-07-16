#include "renderer/passes/ClusterCullingPass.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        constexpr uint32_t kWorkgroupSize = 64; // Matches ClusterFrustumCull.comp's local_size_x.

    } // namespace

    std::array<std::array<float, 4>, 6> ExtractFrustumPlanes(const maths::mat4& viewProj) {
        const std::array<float, 16>& m = viewProj.m;

        // maths::mat4 is stored column-major (m[col * 4 + row], see mat4::operator* /
        // PerspectiveVulkan) -- the same convention draw.vert's mat4 uniforms use. Row i of the
        // matrix as a plain vec4 is therefore assembled by striding across columns at a fixed row
        // offset.
        std::array<float, 4> row0{ m[0], m[4], m[8], m[12] };
        std::array<float, 4> row1{ m[1], m[5], m[9], m[13] };
        std::array<float, 4> row2{ m[2], m[6], m[10], m[14] };
        std::array<float, 4> row3{ m[3], m[7], m[11], m[15] };

        auto add = [](const std::array<float, 4>& a, const std::array<float, 4>& b) {
            return std::array<float, 4>{ a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3] };
        };
        auto sub = [](const std::array<float, 4>& a, const std::array<float, 4>& b) {
            return std::array<float, 4>{ a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3] };
        };

        // Gribb-Hartmann extraction, for Vulkan's [0, 1] normalized device depth range (NOT
        // OpenGL's [-1, 1] convention -- verified against maths::mat4::PerspectiveVulkan: at
        // view-space z = -zNear, clip.z evaluates to exactly 0, so the near plane is row2 alone,
        // not row3 + row2 as the classic OpenGL-oriented derivation of the paper uses).
        std::array<std::array<float, 4>, 6> planes{
            add(row3, row0), // left:   clip.x >= -clip.w
            sub(row3, row0), // right:  clip.x <=  clip.w
            add(row3, row1), // bottom: clip.y >= -clip.w
            sub(row3, row1), // top:    clip.y <=  clip.w
            row2,            // near:   clip.z >=  0
            sub(row3, row2), // far:    clip.z <=  clip.w
        };

        // Normalize each plane so plane.xyz is a unit outward normal and plane.w is the true
        // signed distance from the origin -- required for BoxOutsidePlane's dot-product test in
        // ClusterFrustumCull.comp to compare against 0.0 correctly regardless of the combined
        // matrix's scale.
        for (auto& plane : planes) {
            float length = std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
            if (length > 1.0e-8f) {
                float invLength = 1.0f / length;
                plane[0] *= invLength;
                plane[1] *= invLength;
                plane[2] *= invLength;
                plane[3] *= invLength;
            }
        }

        return planes;
    }

    void ClusterCullingPass::Init(VkDevice device, VmaAllocator allocator, uint32_t maxClusters) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_MaxClusters = maxClusters;

        LOG_INFO(std::format("[ClusterCullingPass] Initializing pass: maxClusters={}", maxClusters));

        // --- Buffers ---
        // Cluster metadata: filled by UploadClusterMetadata() via a staged copy, so it needs
        // TRANSFER_DST_BIT in addition to STORAGE_BUFFER_BIT for the culling shader's readonly SSBO read.
        m_ClusterMetadataBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(ClusterCullMetadata)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // View params: written once per RecordCull() call via vkCmdUpdateBuffer, which requires
        // TRANSFER_DST_BIT on its destination buffer per the Vulkan spec.
        m_ViewParamsBuffer.Create(
            allocator,
            sizeof(ClusterCullViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Indirect command output: written by the culling shader (STORAGE_BUFFER_BIT) and read
        // directly by a later vkCmdDrawIndexedIndirect(Count) (INDIRECT_BUFFER_BIT).
        m_IndirectCommandBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Draw count: atomically incremented by the culling shader (STORAGE_BUFFER_BIT), reset
        // every frame via vkCmdFillBuffer (TRANSFER_DST_BIT), and usable directly as a
        // vkCmdDrawIndexedIndirectCount countBuffer (INDIRECT_BUFFER_BIT).
        m_DrawCountBuffer.Create(
            allocator,
            sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // --- Descriptor set layout: 4 bindings, all compute-visible, matching
        // ClusterFrustumCull.comp's set = 0 bindings 0..3 exactly. ---
        VkDescriptorSetLayoutBinding bindings[4]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // ClusterCullMetadataSSBO
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // CullingViewParamsUBO
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // IndirectCommandsSSBO
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // DrawCountSSBO
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 4;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAlloc.descriptorPool = m_DescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_DescriptorSet));

        VkDescriptorBufferInfo clusterInfo{ m_ClusterMetadataBuffer.Handle(), 0, m_ClusterMetadataBuffer.Size() };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorBufferInfo indirectInfo{ m_IndirectCommandBuffer.Handle(), 0, m_IndirectCommandBuffer.Size() };
        VkDescriptorBufferInfo drawCountInfo{ m_DrawCountBuffer.Handle(), 0, m_DrawCountBuffer.Size() };

        VkWriteDescriptorSet writes[4]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_DescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &clusterInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_DescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo = &viewParamsInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = m_DescriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &indirectInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = m_DescriptorSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &drawCountInfo;

        vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);

        // --- Pipeline layout + pipeline ---
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(uint32_t); // Matches ClusterCullPushConstants::clusterCount in the shader.

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterFrustumCull.comp.spv");

        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);
    }

    void ClusterCullingPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[ClusterCullingPass] Shutting down pass...");
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
            if (m_SetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            }
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;

        // GpuBuffer::Destroy() is null-safe (no-op on an already-empty instance).
        m_ClusterMetadataBuffer.Destroy();
        m_ViewParamsBuffer.Destroy();
        m_IndirectCommandBuffer.Destroy();
        m_DrawCountBuffer.Destroy();

        m_MaxClusters = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterCullingPass::UploadClusterMetadata(VkCommandPool commandPool, VkQueue queue, const std::vector<ClusterCullMetadata>& clusters) {
        assert(clusters.size() <= m_MaxClusters && "ClusterCullingPass::UploadClusterMetadata: candidate list exceeds maxClusters");
        if (clusters.empty()) {
            return;
        }

        // m_ClusterMetadataBuffer is VMA_MEMORY_USAGE_GPU_ONLY (not host-visible), so uploading
        // the CPU-authored candidate list requires a temporary host-visible staging buffer plus an
        // explicit GPU-side copy, mirroring VulkanContext::UploadEntityData's one-time-submit
        // staging pattern exactly.
        VkDeviceSize uploadSize = static_cast<VkDeviceSize>(sizeof(ClusterCullMetadata)) * clusters.size();

        VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        stagingInfo.size = uploadSize;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocResultInfo{};
        VK_CHECK(vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));
        std::memcpy(stagingAllocResultInfo.pMappedData, clusters.data(), static_cast<size_t>(uploadSize));

        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = uploadSize;
            vkCmdCopyBuffer(cmd, stagingBuffer, m_ClusterMetadataBuffer.Handle(), 1, &copyRegion);

            // The copy's writes must be visible to the culling shader's readonly SSBO reads before its
            // next dispatch.
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        });

        vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
    }

    void ClusterCullingPass::RecordClear(VkCommandBuffer cmd) {
        // Only the single draw-count word needs clearing every frame -- the shader only ever
        // reads/writes g_IndirectCommands.commands[] up to g_DrawCount.count, so stale entries
        // beyond that from a previous frame's larger draw are never observed.
        vkCmdFillBuffer(cmd, m_DrawCountBuffer.Handle(), 0, sizeof(uint32_t), 0u);

        // The clear's write must be visible to the culling shader's atomicAdd (both a read and a
        // write of the same word) before its dispatch runs.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    void ClusterCullingPass::RecordCull(VkCommandBuffer cmd, const ClusterCullViewParams& viewParams, uint32_t clusterCount) {
        assert(clusterCount <= m_MaxClusters && "ClusterCullingPass::RecordCull: clusterCount exceeds maxClusters");

        // --- Step 1: upload this frame's frustum planes + camera position into the view-params UBO ---
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ClusterCullViewParams), &viewParams);

        // vkCmdUpdateBuffer is classified as a copy/transfer operation by the Vulkan
        // synchronization chapter (VK_PIPELINE_STAGE_2_COPY_BIT / VK_ACCESS_2_TRANSFER_WRITE_BIT),
        // distinct from vkCmdFillBuffer's VK_PIPELINE_STAGE_2_CLEAR_BIT used in RecordClear().
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        // --- Step 2: dispatch one invocation per candidate cluster ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &clusterCount);

        uint32_t groupCount = (clusterCount + kWorkgroupSize - 1) / kWorkgroupSize;
        if (groupCount > 0) {
            vkCmdDispatch(cmd, groupCount, 1, 1);
        }

        // --- Step 3: make every surviving cluster's indirect command + the final draw count
        // visible to a subsequent vkCmdDrawIndexedIndirect(Count). ---
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

}
