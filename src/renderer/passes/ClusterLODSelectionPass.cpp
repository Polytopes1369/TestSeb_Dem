#include "renderer/passes/ClusterLODSelectionPass.h"

#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>
#ifndef NDEBUG
#include <unordered_map>
#endif

#include "core/Logger.h"
#include "geometry/ClusterDAG.h" // geometry::kInvalidClusterID
#include "geometry/GpuPageTable.h" // geometry::GpuPageTable::LogicalAddressToPageID
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of DAGScreenErrorViewParams in ClusterDAGScreenError.comp (std140):
        // 2 mat4 (128 bytes) + 4 floats (16 bytes) = 144 bytes, already a multiple of 16 -- no
        // manual padding needed.
        struct DAGScreenErrorViewParamsUBO {
            maths::mat4 view;
            maths::mat4 proj;
            float pixelThreshold = 1.0f;
            float fovYRadians = 0.0f;
            float viewportHeight = 0.0f;
            float aspectRatio = 0.0f;
        };
        static_assert(sizeof(DAGScreenErrorViewParamsUBO) == 144,
            "DAGScreenErrorViewParamsUBO must match DAGScreenErrorViewParams in ClusterDAGScreenError.comp exactly (std140 layout)");

    } // namespace

    void ClusterLODSelectionPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkBuffer pageTableBuffer, VkBuffer entityTransformBuffer, uint32_t leafCount,
        const std::vector<geometry::ClusterIndexEntry>& indexEntries,
        const std::vector<geometry::DAGNodeEntry>& dagEntries,
        VkBuffer entityDataBuffer) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_TotalNodeCount = static_cast<uint32_t>(indexEntries.size());
        m_LeafCount = leafCount;

        // =====================================================================================
        // Build the CPU-side DAGNodePayload / LODNodeMetadata arrays from the full index/DAG
        // tables (every DAG level, index-aligned 1:1 -- see ClusterFormat.h's on-disk contract).
        // =====================================================================================
        std::vector<DAGNodePayload> dagNodes(m_TotalNodeCount);
        std::vector<LODNodeMetadata> lodNodes(m_TotalNodeCount);
        for (uint32_t i = 0; i < m_TotalNodeCount; ++i) {
            const geometry::ClusterIndexEntry& entry = indexEntries[i];
            const geometry::DAGNodeEntry& dagNode = dagEntries[i];

            // Inverse of BuildIndexEntry's int8 quantization (VirtualGeometryCacheTest.cpp): the
            // axis is re-normalized after dequantization since per-component rounding
            // denormalizes it -- same reasoning as ClusterRenderPipeline's own metadata build.
            maths::vec3 coneAxis{ entry.coneAxisX / 127.0f, entry.coneAxisY / 127.0f, entry.coneAxisZ / 127.0f };
            coneAxis = coneAxis.Normalize();
            float coneCutoff = entry.coneCutoff / 127.0f;

            maths::vec3 sphereCenter{ entry.sphereCenter[0], entry.sphereCenter[1], entry.sphereCenter[2] };
            maths::vec3 boundsMin{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] };
            maths::vec3 boundsMax{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] };
            uint32_t logicalPageID = geometry::GpuPageTable::LogicalAddressToPageID(entry.virtualAddress);

            DAGNodePayload& node = dagNodes[i];
            node.sphereCenter = sphereCenter;
            node.sphereRadius = entry.sphereRadius;
            node.clusterID = dagNode.clusterID;
            node.parentClusterID0 = dagNode.parentClusterID[0];
            node.parentClusterID1 = dagNode.parentClusterID[1];
            node.childClusterID0 = dagNode.childClusterID[0];
            node.childClusterID1 = dagNode.childClusterID[1];
            node.childClusterID2 = dagNode.childClusterID[2];
            node.childClusterID3 = dagNode.childClusterID[3];
            node.level = dagNode.level;
            node.clusterError = dagNode.clusterError;
            node.parentError = dagNode.parentError;
            node.maxWPOAmplitude = entry.maxWPOAmplitude;
            node.entityID = entry.entityID;
            // Baked directly into the cache file at write time (geometry::DAGNodeEntry::
            // parentSphereCenter, geometry::ClusterDAGGroup::groupSphereCenter) instead of derived
            // here from a single parentClusterID lookup -- with up to 2 possible parents sharing
            // this one group, there is no longer a single "the parent's" ClusterIndexEntry to read
            // forward from; see ClusterDAGScreenError.comp's DAGNodePayload::parentSphereCenter
            // comment for why this must be one unambiguous shared value.
            node.parentSphereCenter = maths::vec3{
                dagNode.parentSphereCenter[0], dagNode.parentSphereCenter[1], dagNode.parentSphereCenter[2] };

            LODNodeMetadata& lodNode = lodNodes[i];
            lodNode.boundsMin = boundsMin;
            lodNode.boundsMax = boundsMax;
            lodNode.sphereCenter = sphereCenter;
            lodNode.sphereRadius = entry.sphereRadius;
            lodNode.coneAxis = coneAxis;
            lodNode.coneCutoff = coneCutoff;
            lodNode.indexCount = entry.indexCount;
            lodNode.clusterID = entry.clusterID;
            lodNode.parentClusterID0 = dagNode.parentClusterID[0];
            lodNode.parentClusterID1 = dagNode.parentClusterID[1];
            lodNode.logicalPageID = logicalPageID;
            lodNode.maskTextureIndex = entry.maskTextureIndex;
            lodNode.maxWPOAmplitude = entry.maxWPOAmplitude;
            lodNode.entityID = entry.entityID;
            lodNode.materialID = entry.materialID;
        }

#ifndef NDEBUG
        // Retain a CPU-side copy purely for DebugLogDAGCutGaps()'s ancestor walk -- see this
        // class's own header comment. Never referenced by any GPU-facing code path.
        m_DebugDagNodesCopy = dagNodes;
#endif

        // =====================================================================================
        // Buffers.
        // =====================================================================================
        m_DAGNodesBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(DAGNodePayload)) * m_TotalNodeCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_LODNodeMetadataBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(LODNodeMetadata)) * m_TotalNodeCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        // TRANSFER_SRC_BIT is only ever needed by RecordDebugReadback() (Debug-only, see this
        // class's own comment) -- adding it unconditionally in Release would cost nothing
        // functionally, but keeping it Debug-gated matches this codebase's "zero debug footprint
        // in Release" rule at the bit-flag level too.
        VkBufferUsageFlags decisionBufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
#ifndef NDEBUG
        decisionBufferUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
#endif
        m_DAGDecisionBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(uint32_t)) * m_TotalNodeCount,
            decisionBufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
        m_DAGLocalErrorBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(float)) * m_TotalNodeCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_DAGParentErrorBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(float)) * m_TotalNodeCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ForceDrawBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(uint32_t)) * m_TotalNodeCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        // 2x uint32 (forceDrawSubstitutions, ancestorWalkExhausted). TRANSFER_SRC_BIT only needed
        // in Debug (see ReadLODFallbackStats()) -- same bit-flag-level "zero debug footprint in
        // Release" pattern as decisionBufferUsage above.
        VkBufferUsageFlags lodFallbackStatsUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
#ifndef NDEBUG
        lodFallbackStatsUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
#endif
        m_LODFallbackStatsBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(uint32_t)) * 2,
            lodFallbackStatsUsage, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ViewParamsBuffer.Create(allocator, sizeof(DAGScreenErrorViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_CandidateMetadataBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(ClusterCullMetadata)) * m_LeafCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_CandidateCountBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_EarlyDispatchArgsBuffer.Create(allocator, sizeof(VkDispatchIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

#ifndef NDEBUG
        m_DebugDecisionReadbackBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(uint32_t)) * m_TotalNodeCount,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
        m_LODFallbackStatsReadbackBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(uint32_t)) * 2,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
#endif

        m_FeedbackBuffer.Init(allocator, m_TotalNodeCount);

        // =====================================================================================
        // One-time upload of the DAGNodesSSBO/LODNodeMetadataSSBO contents, via one shared
        // staging buffer + one command buffer (two copies), blocking submit -- Init()-time only,
        // mirroring every other pass's own one-shot setup pattern in this codebase.
        // =====================================================================================
        {
            VkDeviceSize dagNodesSize = static_cast<VkDeviceSize>(sizeof(DAGNodePayload)) * m_TotalNodeCount;
            VkDeviceSize lodNodesSize = static_cast<VkDeviceSize>(sizeof(LODNodeMetadata)) * m_TotalNodeCount;

            VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            stagingInfo.size = dagNodesSize + lodNodesSize;
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

            VmaAllocationCreateInfo stagingAllocInfo{};
            stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocResultInfo{};
            VK_CHECK(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));

            std::memcpy(stagingAllocResultInfo.pMappedData, dagNodes.data(), static_cast<size_t>(dagNodesSize));
            std::memcpy(static_cast<char*>(stagingAllocResultInfo.pMappedData) + dagNodesSize, lodNodes.data(), static_cast<size_t>(lodNodesSize));

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkBufferCopy dagNodesCopy{ 0, 0, dagNodesSize };
                vkCmdCopyBuffer(cmd, stagingBuffer, m_DAGNodesBuffer.Handle(), 1, &dagNodesCopy);
                VkBufferCopy lodNodesCopy{ dagNodesSize, 0, lodNodesSize };
                vkCmdCopyBuffer(cmd, stagingBuffer, m_LODNodeMetadataBuffer.Handle(), 1, &lodNodesCopy);

                VulkanUtils::RecordMemoryBarrier(cmd,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            });
            vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        }

        // =====================================================================================
        // Descriptor pool, shared by all 4 sets.
        // =====================================================================================
        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 + 7 + 7 + 2 }; // ScreenError + Fallback + Compact + BuildArgs SSBOs (ScreenError gained an EntityTransformBuffer binding for its rotated-sphereCenter LOD estimate, plus (Phase 1, Nanite advanced) an EntityDataBuffer binding for its own displacement-bound inflation; Compact gained an EntityDataBuffer binding for transparent-entity exclusion; Fallback gained a LODFallbackStatsSSBO binding for residency-fallback diagnostics plus a FeedbackTouchBufferSSBO binding for resident-touch reports, see GpuGeometryPagePool's TouchPages).
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };            // DAGViewParamsUBO.

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 4;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        // =====================================================================================
        // Set 1 / Pipeline 1: ClusterDAGScreenError.comp -- bindings 0..4.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[7]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DAGNodesSSBO
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DAGDecisionSSBO
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DAGLocalErrorSSBO
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DAGParentErrorSSBO
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DAGViewParamsUBO
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityTransformBuffer
            // Phase 1 (Nanite advanced): binding 6 -- the same entityDataBuffer handle
            // ClusterLODCompact.comp's own set (below) already borrows, wired into this second
            // descriptor set so this shader can inflate its projected-error bound for entities with
            // enhanced-displacement/spline-deformation flags set (see ClusterDAGScreenError.comp's
            // own main() comment).
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityDataBuffer

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 7;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ScreenErrorSetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_ScreenErrorSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_ScreenErrorDescriptorSet));

            VkDescriptorBufferInfo dagNodesInfo{ m_DAGNodesBuffer.Handle(), 0, m_DAGNodesBuffer.Size() };
            VkDescriptorBufferInfo decisionInfo{ m_DAGDecisionBuffer.Handle(), 0, m_DAGDecisionBuffer.Size() };
            VkDescriptorBufferInfo localErrorInfo{ m_DAGLocalErrorBuffer.Handle(), 0, m_DAGLocalErrorBuffer.Size() };
            VkDescriptorBufferInfo parentErrorInfo{ m_DAGParentErrorBuffer.Handle(), 0, m_DAGParentErrorBuffer.Size() };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo screenErrorEntityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet writes[7]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScreenErrorDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dagNodesInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScreenErrorDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &decisionInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScreenErrorDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &localErrorInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScreenErrorDescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &parentErrorInfo, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScreenErrorDescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScreenErrorDescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScreenErrorDescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &screenErrorEntityDataInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ScreenErrorSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ScreenErrorPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterDAGScreenError.comp.spv");
            m_ScreenErrorPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ScreenErrorPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // Set 2 / Pipeline 2: ClusterLODResidencyFallback.comp -- bindings 0..6.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[7]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DAGDecisionSSBO
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // LODNodeMetadataSSBO
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // GeometryPageTableSSBO (borrowed)
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // ForceDrawSSBO
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // FeedbackBufferSSBO (borrowed)
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // LODFallbackStatsSSBO (diagnostics)
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // FeedbackTouchBufferSSBO (borrowed) -- resident-touch reports for GpuGeometryPagePool's LRU.

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 7;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ResidencyFallbackSetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_ResidencyFallbackSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_ResidencyFallbackDescriptorSet));

            VkDescriptorBufferInfo decisionInfo{ m_DAGDecisionBuffer.Handle(), 0, m_DAGDecisionBuffer.Size() };
            VkDescriptorBufferInfo lodNodesInfo{ m_LODNodeMetadataBuffer.Handle(), 0, m_LODNodeMetadataBuffer.Size() };
            VkDescriptorBufferInfo pageTableInfo{ pageTableBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo forceDrawInfo{ m_ForceDrawBuffer.Handle(), 0, m_ForceDrawBuffer.Size() };
            VkDescriptorBufferInfo feedbackInfo{ m_FeedbackBuffer.GetDeviceBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo lodFallbackStatsInfo{ m_LODFallbackStatsBuffer.Handle(), 0, m_LODFallbackStatsBuffer.Size() };
            VkDescriptorBufferInfo feedbackTouchInfo{ m_FeedbackBuffer.GetTouchDeviceBuffer(), 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet writes[7]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResidencyFallbackDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &decisionInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResidencyFallbackDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lodNodesInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResidencyFallbackDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResidencyFallbackDescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &forceDrawInfo, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResidencyFallbackDescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResidencyFallbackDescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lodFallbackStatsInfo, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResidencyFallbackDescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackTouchInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ResidencyFallbackSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ResidencyFallbackPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterLODResidencyFallback.comp.spv");
            m_ResidencyFallbackPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ResidencyFallbackPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // Set 3 / Pipeline 3: ClusterLODCompact.comp -- bindings 0..6.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[7]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DAGDecisionSSBO
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // LODNodeMetadataSSBO
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // GeometryPageTableSSBO (borrowed)
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // ForceDrawSSBO
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // CandidateCountSSBO
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // CandidateMetadataSSBO
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityDataBuffer (borrowed) -- transparent-entity exclusion.

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 7;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_CompactSetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_CompactSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_CompactDescriptorSet));

            VkDescriptorBufferInfo decisionInfo{ m_DAGDecisionBuffer.Handle(), 0, m_DAGDecisionBuffer.Size() };
            VkDescriptorBufferInfo lodNodesInfo{ m_LODNodeMetadataBuffer.Handle(), 0, m_LODNodeMetadataBuffer.Size() };
            VkDescriptorBufferInfo pageTableInfo{ pageTableBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo forceDrawInfo{ m_ForceDrawBuffer.Handle(), 0, m_ForceDrawBuffer.Size() };
            VkDescriptorBufferInfo candidateCountInfo{ m_CandidateCountBuffer.Handle(), 0, m_CandidateCountBuffer.Size() };
            VkDescriptorBufferInfo candidateMetadataInfo{ m_CandidateMetadataBuffer.Handle(), 0, m_CandidateMetadataBuffer.Size() };
            VkDescriptorBufferInfo entityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet writes[7]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &decisionInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lodNodesInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &forceDrawInfo, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &candidateCountInfo, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &candidateMetadataInfo, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_CompactSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_CompactPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterLODCompact.comp.spv");
            m_CompactPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_CompactPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // Set 4 / Pipeline 4: BuildDispatchIndirectArgs.comp -- own instance, sourced from
        // CandidateCountSSBO (binding 0 aliases m_CandidateCountBuffer's leading -- and only --
        // word), matching every other pass's own use of this shared shader.
        // =====================================================================================
        {
            m_BuildArgsSetLayout = VulkanPipeline::CreateBuildDispatchIndirectArgsSetLayout(m_Device);
            VulkanPipeline::CreateBuildDispatchIndirectArgsPipeline(
                m_Device, m_BuildArgsSetLayout, m_BuildArgsPipelineLayout, m_BuildArgsPipeline);

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_BuildArgsSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_BuildArgsDescriptorSet));

            VkDescriptorBufferInfo sourceCountInfo{ m_CandidateCountBuffer.Handle(), 0, sizeof(uint32_t) };
            VkDescriptorBufferInfo dispatchArgsInfo{ m_EarlyDispatchArgsBuffer.Handle(), 0, m_EarlyDispatchArgsBuffer.Size() };

            VkWriteDescriptorSet writes[2]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sourceCountInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dispatchArgsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);
        }

        LOG_INFO(std::format("[ClusterLODSelectionPass] Initialized: {} total DAG nodes, {} max candidates.",
            m_TotalNodeCount, m_LeafCount));
    }

    void ClusterLODSelectionPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[ClusterLODSelectionPass] Shutting down pass...");
            if (m_ScreenErrorPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ScreenErrorPipeline, nullptr);
            if (m_ResidencyFallbackPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ResidencyFallbackPipeline, nullptr);
            if (m_CompactPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_CompactPipeline, nullptr);
            if (m_BuildArgsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_BuildArgsPipeline, nullptr);

            if (m_ScreenErrorPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ScreenErrorPipelineLayout, nullptr);
            if (m_ResidencyFallbackPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ResidencyFallbackPipelineLayout, nullptr);
            if (m_CompactPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_CompactPipelineLayout, nullptr);
            if (m_BuildArgsPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_BuildArgsPipelineLayout, nullptr);

            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees every set allocated from it.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_ScreenErrorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ScreenErrorSetLayout, nullptr);
            if (m_ResidencyFallbackSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ResidencyFallbackSetLayout, nullptr);
            if (m_CompactSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_CompactSetLayout, nullptr);
            if (m_BuildArgsSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_BuildArgsSetLayout, nullptr);
        }

        m_ScreenErrorPipeline = VK_NULL_HANDLE;
        m_ResidencyFallbackPipeline = VK_NULL_HANDLE;
        m_CompactPipeline = VK_NULL_HANDLE;
        m_BuildArgsPipeline = VK_NULL_HANDLE;
        m_ScreenErrorPipelineLayout = VK_NULL_HANDLE;
        m_ResidencyFallbackPipelineLayout = VK_NULL_HANDLE;
        m_CompactPipelineLayout = VK_NULL_HANDLE;
        m_BuildArgsPipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_ScreenErrorSetLayout = VK_NULL_HANDLE;
        m_ResidencyFallbackSetLayout = VK_NULL_HANDLE;
        m_CompactSetLayout = VK_NULL_HANDLE;
        m_BuildArgsSetLayout = VK_NULL_HANDLE;
        m_ScreenErrorDescriptorSet = VK_NULL_HANDLE;
        m_ResidencyFallbackDescriptorSet = VK_NULL_HANDLE;
        m_CompactDescriptorSet = VK_NULL_HANDLE;
        m_BuildArgsDescriptorSet = VK_NULL_HANDLE;

        m_FeedbackBuffer.Shutdown();

        // GpuBuffer::Destroy() is null-safe (no-op on an already-empty instance).
        m_DAGNodesBuffer.Destroy();
        m_LODNodeMetadataBuffer.Destroy();
        m_DAGDecisionBuffer.Destroy();
        m_DAGLocalErrorBuffer.Destroy();
        m_DAGParentErrorBuffer.Destroy();
        m_ForceDrawBuffer.Destroy();
        m_LODFallbackStatsBuffer.Destroy();
        m_ViewParamsBuffer.Destroy();
        m_CandidateMetadataBuffer.Destroy();
        m_CandidateCountBuffer.Destroy();
        m_EarlyDispatchArgsBuffer.Destroy();

#ifndef NDEBUG
        m_DebugDecisionReadbackBuffer.Destroy();
        m_DebugDagNodesCopy.clear();
        m_LODFallbackStatsReadbackBuffer.Destroy();
#endif

        m_TotalNodeCount = 0;
        m_LeafCount = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterLODSelectionPass::RecordClear(VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_CandidateCountBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_ForceDrawBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(cmd, m_LODFallbackStatsBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);
        m_FeedbackBuffer.RecordClear(cmd);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    void ClusterLODSelectionPass::RecordEvaluateAndCompact(VkCommandBuffer cmd, const ViewParams& viewParams) {
#ifndef NDEBUG
        // Log the PREVIOUS call's residency-fallback stats before recording this frame's own
        // work -- mirrors GetFeedbackBuffer()'s one-frame-lag contract (the copy this same
        // function records further down is only guaranteed visible to the host after the GPU work
        // it was part of has completed, i.e. by the time this function runs again next frame).
        LODFallbackStats stats = ReadLODFallbackStats();
        if (stats.ancestorWalkExhausted > 0) {
            LOG_WARNING(std::format("[ClusterLODSelectionPass] {} node(s) had NO resident ancestor within MAX_ANCESTOR_WALK last frame -- their screen region was a genuine hole. {} node(s) were drawn via a coarser resident-ancestor substitution (visible as wrong LOD).",
                stats.ancestorWalkExhausted, stats.forceDrawSubstitutions));
        } else if (stats.forceDrawSubstitutions > 0) {
            LOG_INFO(std::format("[ClusterLODSelectionPass] {} node(s) drawn via a coarser resident-ancestor substitution last frame (non-resident fine geometry still streaming in).",
                stats.forceDrawSubstitutions));
        }
#endif

        DAGScreenErrorViewParamsUBO uboParams{};
        uboParams.view = viewParams.view;
        uboParams.proj = viewParams.proj;
        uboParams.pixelThreshold = viewParams.pixelErrorThreshold;
        uboParams.fovYRadians = viewParams.fovYRadians;
        uboParams.viewportHeight = viewParams.viewportHeight;
        uboParams.aspectRatio = viewParams.aspectRatio;
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(DAGScreenErrorViewParamsUBO), &uboParams);

        {
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);
        }

        uint32_t groupCount = (m_TotalNodeCount + kWorkgroupSize - 1) / kWorkgroupSize;

        // --- Dispatch 1: ClusterDAGScreenError.comp ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScreenErrorPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScreenErrorPipelineLayout, 0, 1, &m_ScreenErrorDescriptorSet, 0, nullptr);
        if (groupCount > 0) {
            vkCmdDispatch(cmd, groupCount, 1, 1);
        }

        {
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // --- Dispatch 2: ClusterLODResidencyFallback.comp ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ResidencyFallbackPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ResidencyFallbackPipelineLayout, 0, 1, &m_ResidencyFallbackDescriptorSet, 0, nullptr);
        if (groupCount > 0) {
            vkCmdDispatch(cmd, groupCount, 1, 1);
        }

        {
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }

#ifndef NDEBUG
        // Copy this frame's LODFallbackStatsSSBO into the host-readable buffer ReadLODFallbackStats()
        // reads at the top of the NEXT call -- mirrors renderer::FeedbackBuffer::RecordReadback's
        // two-barrier copy-then-host-visibility pattern exactly. Independent of dispatch 3 (which
        // never touches this buffer), so no ordering constraint against it either way.
        {
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = m_LODFallbackStatsBuffer.Size();
            vkCmdCopyBuffer(cmd, m_LODFallbackStatsBuffer.Handle(), m_LODFallbackStatsReadbackBuffer.Handle(), 1, &copyRegion);

            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        }
#endif

        // --- Dispatch 3: ClusterLODCompact.comp ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompactPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompactPipelineLayout, 0, 1, &m_CompactDescriptorSet, 0, nullptr);
        if (groupCount > 0) {
            vkCmdDispatch(cmd, groupCount, 1, 1);
        }

        // Make the candidate metadata + count visible to RecordBuildEarlyDispatchArgs() and a
        // later compute read (renderer::ClusterOcclusionCullingPass's early pass).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void ClusterLODSelectionPass::RecordBuildEarlyDispatchArgs(VkCommandBuffer cmd) {
        // One thread per candidate -- perElementMultiplier = 1, matching
        // renderer::ClusterOcclusionCullingPass::RecordBuildLateDispatchArgs's own use of this
        // shared shader.
        uint32_t pushConstants[2] = { kWorkgroupSize, 1u };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipelineLayout, 0, 1, &m_BuildArgsDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_BuildArgsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
        vkCmdDispatch(cmd, 1, 1, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

#ifndef NDEBUG
    void ClusterLODSelectionPass::RecordDebugReadback(VkCommandBuffer cmd) {
        // Barrier #1: this frame's ClusterDAGScreenError.comp writes into m_DAGDecisionBuffer must
        // complete and be visible before the copy reads it.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        VkBufferCopy copyRegion{ 0, 0, m_DAGDecisionBuffer.Size() };
        vkCmdCopyBuffer(cmd, m_DAGDecisionBuffer.Handle(), m_DebugDecisionReadbackBuffer.Handle(), 1, &copyRegion);

        // Barrier #2: exactly renderer::FeedbackBuffer::RecordReadback()'s own host-visibility
        // barrier -- HOST_COHERENT memory still requires this execution/visibility dependency
        // before a host read, per the Vulkan spec.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    }

    ClusterLODSelectionPass::LODFallbackStats ClusterLODSelectionPass::ReadLODFallbackStats() const {
        const uint32_t* mapped = static_cast<const uint32_t*>(m_LODFallbackStatsReadbackBuffer.MappedData());
        if (mapped == nullptr) {
            return LODFallbackStats{}; // Init() not called yet (e.g. the very first frame).
        }
        LODFallbackStats stats{};
        stats.forceDrawSubstitutions = mapped[0];
        stats.ancestorWalkExhausted = mapped[1];
        return stats;
    }

    void ClusterLODSelectionPass::DebugLogDAGCutGaps() const {
        constexpr uint32_t kDagDecisionDraw = 0u; // Matches DAG_DECISION_DRAW in ClusterDAGScreenError.comp.

        const uint32_t* decisions = static_cast<const uint32_t*>(m_DebugDecisionReadbackBuffer.MappedData());
        if (decisions == nullptr || m_DebugDagNodesCopy.size() != m_TotalNodeCount) {
            LOG_WARNING("[ClusterLODSelectionPass][DebugDAGCutGaps] No data to analyze yet (readback not recorded, or Init() not called).");
            return;
        }

        struct EntityGapStats {
            uint32_t leafCount = 0;
            uint32_t gapCount = 0; // Leaves whose entire ancestor chain has zero DRAW decisions.
            std::vector<uint32_t> exampleGapClusterIDs; // First few, for logging.
        };
        std::unordered_map<uint32_t, EntityGapStats> statsByEntity;

        for (uint32_t i = 0; i < m_TotalNodeCount; ++i) {
            const DAGNodePayload& node = m_DebugDagNodesCopy[i];
            // Slot 0 is always filled first (geometry::VirtualGeometryCacheTest.cpp's
            // BuildIndexEntry/DAGNodeEntry-flatten loop assigns childClusterID sequentially from
            // a group's memberClusterIndices), so an invalid slot 0 alone already proves zero
            // children -- no need to also check slots 1-3.
            bool isLeaf = node.childClusterID0 == geometry::kInvalidClusterID;
            if (!isLeaf) {
                continue;
            }

            EntityGapStats& stats = statsByEntity[node.entityID];
            ++stats.leafCount;

            // Walk this leaf's own node and every ancestor up to the root(s), looking for a single
            // DRAW decision anywhere reachable. A node can now have up to 2 parents (one of a
            // group's up to kMaxGroupOutputClusters output siblings, see geometry::ClusterDAGGroup)
            // so this is a small bounded BFS over both parent branches at every step, not a single
            // linear walk -- `visited` keeps it from re-exploring a node reachable through more
            // than one path (e.g. two siblings that later share a common further ancestor) and
            // bounds total work by m_TotalNodeCount regardless of branching, exactly like the old
            // single-path walk's own step cap (the DAG is acyclic by construction/
            // ValidateClusterDAG, so this should never actually be exhausted) rather than trusting
            // an unbounded traversal on data this debug tool itself must remain robust against if
            // the DAG were ever corrupt.
            bool foundDraw = false;
            std::vector<bool> visited(m_TotalNodeCount, false);
            std::vector<uint32_t> frontier{ i };
            visited[i] = true;
            for (uint32_t step = 0; step < m_TotalNodeCount && !frontier.empty() && !foundDraw; ++step) {
                std::vector<uint32_t> nextFrontier;
                for (uint32_t currentIndex : frontier) {
                    if (decisions[currentIndex] == kDagDecisionDraw) {
                        foundDraw = true;
                        break;
                    }
                    const DAGNodePayload& current = m_DebugDagNodesCopy[currentIndex];
                    for (uint32_t parentClusterID : { current.parentClusterID0, current.parentClusterID1 }) {
                        if (parentClusterID != geometry::kInvalidClusterID && !visited[parentClusterID]) {
                            visited[parentClusterID] = true;
                            nextFrontier.push_back(parentClusterID);
                        }
                    }
                }
                frontier = std::move(nextFrontier);
            }

            if (!foundDraw) {
                ++stats.gapCount;
                if (stats.exampleGapClusterIDs.size() < 5) {
                    stats.exampleGapClusterIDs.push_back(node.clusterID);
                }
            }
        }

        uint32_t totalLeaves = 0;
        uint32_t totalGaps = 0;
        for (const auto& [entityID, stats] : statsByEntity) {
            totalLeaves += stats.leafCount;
            totalGaps += stats.gapCount;
            if (stats.gapCount == 0) {
                continue;
            }

            std::string exampleList;
            for (uint32_t clusterID : stats.exampleGapClusterIDs) {
                const DAGNodePayload& node = m_DebugDagNodesCopy[clusterID];
                exampleList += std::format("  clusterID={} level={} sphereCenter=({:.3f},{:.3f},{:.3f}) sphereRadius={:.3f} clusterError={:.6f} parentError={:.6f}\n",
                    clusterID, node.level, node.sphereCenter.x, node.sphereCenter.y, node.sphereCenter.z, node.sphereRadius,
                    node.clusterError, node.parentError);
            }

            // Full leaf-to-root decision chain for the FIRST example only (kept to one to bound
            // log size) -- decision values: 0=DRAW, 1=SUBDIVIDE, 2=SKIP (DAG_DECISION_* in
            // ClusterDAGScreenError.comp). Directly shows WHY no node on the path ever satisfies
            // the DRAW condition (e.g. every node stuck at SUBDIVIDE, or a monotonicity break).
            // Follows parentClusterID0 only (a single illustrative path is enough for this dump;
            // the existence check above already explored both parent branches) -- logged
            // explicitly so parentClusterID1, when present, isn't mistaken for having been walked.
            std::string chainDump;
            if (!stats.exampleGapClusterIDs.empty()) {
                uint32_t chainIndex = stats.exampleGapClusterIDs[0];
                for (uint32_t step = 0; step < m_TotalNodeCount; ++step) {
                    const DAGNodePayload& chainNode = m_DebugDagNodesCopy[chainIndex];
                    chainDump += std::format("    [{}] clusterID={} level={} decision={} clusterError={:.6f} parentError={:.6f} parentClusterID0={}\n",
                        step, chainIndex, chainNode.level, decisions[chainIndex],
                        chainNode.clusterError, chainNode.parentError,
                        chainNode.parentClusterID0 == geometry::kInvalidClusterID ? -1 : static_cast<int32_t>(chainNode.parentClusterID0));
                    if (chainNode.parentClusterID0 == geometry::kInvalidClusterID) {
                        break;
                    }
                    chainIndex = chainNode.parentClusterID0;
                }
            }

            LOG_WARNING(std::format(
                "[ClusterLODSelectionPass][DebugDAGCutGaps] entityID={}: {}/{} leaves have NO DRAW decision anywhere on their leaf-to-root path (a genuine DAG-cut gap -- RequestClusterResidency() is never called for these, so they never even enter the streaming queue). Examples:\n{}\nFull chain for first example:\n{}",
                entityID, stats.gapCount, stats.leafCount, exampleList, chainDump));
        }

        LOG_INFO(std::format(
            "[ClusterLODSelectionPass][DebugDAGCutGaps] Analyzed {} entities, {} total leaves, {} total gap leaves ({:.2f}%).",
            statsByEntity.size(), totalLeaves, totalGaps,
            totalLeaves > 0 ? (100.0 * static_cast<double>(totalGaps) / static_cast<double>(totalLeaves)) : 0.0));

        // ---------------------------------------------------------------------------------------
        // Floor position-sanity scan (2026-07-16 "floating disconnected fragments" investigation):
        // the floor (entityID 12) is a perfectly flat, non-rotating plane at Y=-0.8 -- EVERY node's
        // sphereCenter, at every DAG level, must sit at that exact height and within the floor's
        // 300x300m footprint ([-150, 150] on X/Z). GI (radiosity/SSRT/world probes) was already
        // ruled out as the source of the floating fragments seen in screenshots (identical with all
        // three off) -- this checks whether the DAG DATA ITSELF already contains out-of-place
        // positions for the floor, which would point at cache-build/simplification corruption
        // rather than a runtime rendering/streaming bug.
        constexpr uint32_t kFloorEntityID = 12;
        constexpr float kFloorY = -0.8f;
        constexpr float kFloorHalfExtent = 150.0f;
        constexpr float kYTolerance = 0.05f;
        constexpr float kXZTolerance = 1.0f; // A cluster's own sphereRadius can push its center slightly past the exact edge.

        uint32_t floorNodesChecked = 0;
        uint32_t floorOutliers = 0;
        std::string outlierExamples;
        for (uint32_t i = 0; i < m_TotalNodeCount; ++i) {
            const DAGNodePayload& node = m_DebugDagNodesCopy[i];
            if (node.entityID != kFloorEntityID) {
                continue;
            }
            ++floorNodesChecked;

            bool yBad = std::abs(node.sphereCenter.y - kFloorY) > kYTolerance;
            bool xBad = std::abs(node.sphereCenter.x) > (kFloorHalfExtent + kXZTolerance);
            bool zBad = std::abs(node.sphereCenter.z) > (kFloorHalfExtent + kXZTolerance);
            if (yBad || xBad || zBad) {
                ++floorOutliers;
                if (floorOutliers <= 10) {
                    outlierExamples += std::format("  clusterID={} level={} sphereCenter=({:.3f},{:.3f},{:.3f}) sphereRadius={:.3f} parentClusterID0={}\n",
                        i, node.level, node.sphereCenter.x, node.sphereCenter.y, node.sphereCenter.z, node.sphereRadius,
                        node.parentClusterID0 == geometry::kInvalidClusterID ? -1 : static_cast<int32_t>(node.parentClusterID0));
                }
            }
        }

        if (floorOutliers > 0) {
            LOG_WARNING(std::format(
                "[ClusterLODSelectionPass][DebugFloorPositionScan] {}/{} floor DAG nodes have a sphereCenter outside the expected flat Y={:.1f} / [-{:.0f},{:.0f}] XZ footprint -- the cache-build data itself is corrupt for these, not just a runtime culling/streaming symptom. Examples:\n{}",
                floorOutliers, floorNodesChecked, kFloorY, kFloorHalfExtent, kFloorHalfExtent, outlierExamples));
        } else {
            LOG_INFO(std::format(
                "[ClusterLODSelectionPass][DebugFloorPositionScan] Checked {} floor DAG nodes: all sphereCenters are within the expected flat Y={:.1f} / [-{:.0f},{:.0f}] XZ footprint. Floor cache data is NOT corrupt -- the floating fragments must come from elsewhere (rendering/decode/vis-buffer, not the DAG itself).",
                floorNodesChecked, kFloorY, kFloorHalfExtent, kFloorHalfExtent));
        }

        // ---------------------------------------------------------------------------------------
        // Same sanity idea for the 12 non-floor (sculpted, grid-arranged) primitives: each entity's
        // whole grid slot + own half-extent puts every one of its clusters, at every DAG level,
        // well within a generous +/-10m box on X/Z and +/-3m on Y (see VulkanContext::GridSlot's
        // 3m spacing / 3-wide grid and each primitive's own ~1-2m size). A node outside that would
        // mean THAT entity's own cache data is corrupt -- independent of the floor investigation
        // above, e.g. relevant to the box primitive's own persistent crack.
        // ---------------------------------------------------------------------------------------
        constexpr float kPrimitiveXZBound = 10.0f;
        constexpr float kPrimitiveYBound = 3.0f;
        std::unordered_map<uint32_t, uint32_t> primitiveOutlierCountByEntity;
        std::unordered_map<uint32_t, std::string> primitiveOutlierExamplesByEntity;
        for (uint32_t i = 0; i < m_TotalNodeCount; ++i) {
            const DAGNodePayload& node = m_DebugDagNodesCopy[i];
            if (node.entityID == kFloorEntityID) {
                continue;
            }
            bool outOfBounds = std::abs(node.sphereCenter.x) > kPrimitiveXZBound
                || std::abs(node.sphereCenter.z) > kPrimitiveXZBound
                || std::abs(node.sphereCenter.y) > kPrimitiveYBound;
            if (outOfBounds) {
                uint32_t& count = primitiveOutlierCountByEntity[node.entityID];
                ++count;
                std::string& examples = primitiveOutlierExamplesByEntity[node.entityID];
                if (count <= 5) {
                    examples += std::format("  clusterID={} level={} sphereCenter=({:.3f},{:.3f},{:.3f}) sphereRadius={:.3f}\n",
                        i, node.level, node.sphereCenter.x, node.sphereCenter.y, node.sphereCenter.z, node.sphereRadius);
                }
            }
        }

        if (primitiveOutlierCountByEntity.empty()) {
            LOG_INFO("[ClusterLODSelectionPass][DebugPrimitivePositionScan] All 12 non-floor entities' DAG nodes are within the expected grid-local bounds. No corrupt cache data found there either.");
        } else {
            for (const auto& [entityID, count] : primitiveOutlierCountByEntity) {
                LOG_WARNING(std::format(
                    "[ClusterLODSelectionPass][DebugPrimitivePositionScan] entityID={}: {} DAG node(s) have a sphereCenter outside the expected +/-{:.0f}m XZ / +/-{:.0f}m Y grid-local bounds -- this entity's own cache data is corrupt. Examples:\n{}",
                    entityID, count, kPrimitiveXZBound, kPrimitiveYBound, primitiveOutlierExamplesByEntity[entityID]));
            }
        }
    }
#endif

}
