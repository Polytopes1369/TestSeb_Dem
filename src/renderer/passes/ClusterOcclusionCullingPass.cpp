#include "renderer/passes/ClusterOcclusionCullingPass.h"

#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {


    void ClusterOcclusionCullingPass::Init(VkDevice device, VmaAllocator allocator, uint32_t maxClusters, uint32_t totalClusterCount,
        VkBuffer candidateMetadataBuffer, VkBuffer candidateCountBuffer, VkBuffer entityTransformBuffer,
        VkImageView hzbFullView, VkExtent2D hzbMip0Extent, uint32_t hzbMipLevelCount) {
        Shutdown();

        m_Device = device;
        m_MaxClusters = maxClusters;
        m_CandidateMetadataBuffer = candidateMetadataBuffer;
        m_CandidateCountBuffer = candidateCountBuffer;
        m_EntityTransformBuffer = entityTransformBuffer;
        m_HZBView = hzbFullView;
        m_HZBMip0Width = static_cast<float>(hzbMip0Extent.width);
        m_HZBMip0Height = static_cast<float>(hzbMip0Extent.height);
        m_HZBMipCount = static_cast<float>(hzbMipLevelCount);

        LOG_INFO(std::format("[ClusterOcclusionCullingPass] Initializing pass: maxClusters={}, totalClusterCount={}, HZB={}x{}",
            maxClusters, totalClusterCount, hzbMip0Extent.width, hzbMip0Extent.height));

        // --- Buffers ---
        m_ViewParamsBuffer.Create(
            allocator,
            sizeof(ClusterCullViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_HZBParamsBuffer.Create(
            allocator,
            sizeof(HZBOcclusionViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Persisted across frames -- only ever cleared by the caller-invoked
        // ClearPersistedVisibility(), never by RecordClearFrame(). Sized to totalClusterCount (all
        // DAG levels), NOT maxClusters: indexed by each cluster's persistent, stable clusterID, so
        // every possible clusterID needs a slot, not just this frame's candidate count.
        m_VisibleLastFrameBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(uint32_t)) * totalClusterCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // { uint count; uint clusterIndex[maxClusters]; } -- one extra leading word for the count.
        m_PendingListBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(uint32_t)) * (1u + maxClusters),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // { uint count; uint clusterIndex[maxClusters]; }, same shape as m_PendingListBuffer --
        // clusters routed to renderer::ClusterSoftwareRasterPass instead of a hardware indirect
        // draw (see ClusterHZBOcclusionCull.comp's ShouldUseSoftwareRaster/EmitSoftwareCluster).
        m_SoftwareClusterListBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(uint32_t)) * (1u + maxClusters),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_EarlyIndirectCommandBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_EarlyDrawCountBuffer.Create(
            allocator,
            sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_LateIndirectCommandBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_LateDrawCountBuffer.Create(
            allocator,
            sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // --- Opaque-list counterparts (bindings 12-16), same shapes/usages as the five masked-list
        // buffers just above -- see ClusterHZBOcclusionCull.comp's "Opaque / masked classification"
        // class comment. ---
        m_EarlyIndirectCommandOpaqueBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_EarlyDrawCountOpaqueBuffer.Create(
            allocator,
            sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_LateIndirectCommandOpaqueBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_LateDrawCountOpaqueBuffer.Create(
            allocator,
            sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_SoftwareClusterListOpaqueBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(uint32_t)) * (1u + maxClusters),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // VkDispatchIndirectCommand (3x uint32) -- only ever written by BuildDispatchIndirectArgs.comp.
        m_LateDispatchArgsBuffer.Create(
            allocator,
            sizeof(VkDispatchIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // --- HZB sampler: nearest filtering AND nearest mipmap mode. Mip selection is done
        // explicitly in-shader (textureLod), never automatically -- but more importantly, linear
        // filtering would blend two texels' (or two mips') max-depth values together, which is
        // not itself a valid max-depth bound for the blended footprint and would silently break
        // IsClusterOccluded's conservativeness (see hzb_occlusion.glsl). ---
        m_HZBSampler = VulkanUtils::CreateNearestSampler(m_Device, m_HZBMipCount);

        // --- Main descriptor set layout: 17 bindings, all compute-visible, matching
        // ClusterHZBOcclusionCull.comp's set = 0 bindings 0..16 exactly (0..11 the original set,
        // 12..16 the opaque-list counterparts -- see that shader's "Opaque / masked classification"
        // class comment). Shared by both the early and late pipelines. ---
        VkDescriptorSetLayoutBinding bindings[18]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // ClusterCullMetadataSSBO
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // CullingViewParamsUBO
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // HZBOcclusionViewParamsUBO
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_HZBTexture
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // VisibleLastFrameSSBO
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // PendingListSSBO
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // EarlyIndirectCommandsSSBO
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // EarlyDrawCountSSBO
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // LateIndirectCommandsSSBO
        bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // LateDrawCountSSBO
        bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // SoftwareClusterListSSBO
        bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // CandidateCountSSBO (borrowed, read-only)
        bindings[12] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // EarlyIndirectCommandsOpaqueSSBO
        bindings[13] = { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // EarlyDrawCountOpaqueSSBO
        bindings[14] = { 14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // LateIndirectCommandsOpaqueSSBO
        bindings[15] = { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // LateDrawCountOpaqueSSBO
        bindings[16] = { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // SoftwareClusterListOpaqueSSBO
        bindings[17] = { 17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // EntityTransformBuffer (borrowed)

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 18;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        // --- BuildDispatchIndirectArgs set layout: 2 bindings ---
        m_BuildArgsSetLayout = VulkanPipeline::CreateBuildDispatchIndirectArgsSetLayout(m_Device);

        // --- One shared descriptor pool for both sets ---
        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 15 + 2 };      // Main set's 15 SSBOs (incl. the 3 borrowed ones) + BuildArgs set's 2 SSBOs.
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 };
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAlloc.descriptorPool = m_DescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_DescriptorSet));

        VkDescriptorSetAllocateInfo buildArgsSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        buildArgsSetAlloc.descriptorPool = m_DescriptorPool;
        buildArgsSetAlloc.descriptorSetCount = 1;
        buildArgsSetAlloc.pSetLayouts = &m_BuildArgsSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &buildArgsSetAlloc, &m_BuildArgsDescriptorSet));

        // --- Main set descriptor writes ---
        VkDescriptorBufferInfo clusterInfo{ m_CandidateMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorBufferInfo hzbParamsInfo{ m_HZBParamsBuffer.Handle(), 0, m_HZBParamsBuffer.Size() };
        VkDescriptorBufferInfo visibleLastFrameInfo{ m_VisibleLastFrameBuffer.Handle(), 0, m_VisibleLastFrameBuffer.Size() };
        VkDescriptorBufferInfo pendingListInfo{ m_PendingListBuffer.Handle(), 0, m_PendingListBuffer.Size() };
        VkDescriptorBufferInfo earlyIndirectInfo{ m_EarlyIndirectCommandBuffer.Handle(), 0, m_EarlyIndirectCommandBuffer.Size() };
        VkDescriptorBufferInfo earlyDrawCountInfo{ m_EarlyDrawCountBuffer.Handle(), 0, m_EarlyDrawCountBuffer.Size() };
        VkDescriptorBufferInfo lateIndirectInfo{ m_LateIndirectCommandBuffer.Handle(), 0, m_LateIndirectCommandBuffer.Size() };
        VkDescriptorBufferInfo lateDrawCountInfo{ m_LateDrawCountBuffer.Handle(), 0, m_LateDrawCountBuffer.Size() };
        VkDescriptorBufferInfo softwareClusterListInfo{ m_SoftwareClusterListBuffer.Handle(), 0, m_SoftwareClusterListBuffer.Size() };
        VkDescriptorBufferInfo candidateCountInfo{ m_CandidateCountBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo earlyIndirectOpaqueInfo{ m_EarlyIndirectCommandOpaqueBuffer.Handle(), 0, m_EarlyIndirectCommandOpaqueBuffer.Size() };
        VkDescriptorBufferInfo earlyDrawCountOpaqueInfo{ m_EarlyDrawCountOpaqueBuffer.Handle(), 0, m_EarlyDrawCountOpaqueBuffer.Size() };
        VkDescriptorBufferInfo lateIndirectOpaqueInfo{ m_LateIndirectCommandOpaqueBuffer.Handle(), 0, m_LateIndirectCommandOpaqueBuffer.Size() };
        VkDescriptorBufferInfo lateDrawCountOpaqueInfo{ m_LateDrawCountOpaqueBuffer.Handle(), 0, m_LateDrawCountOpaqueBuffer.Size() };
        VkDescriptorBufferInfo softwareClusterListOpaqueInfo{ m_SoftwareClusterListOpaqueBuffer.Handle(), 0, m_SoftwareClusterListOpaqueBuffer.Size() };
        VkDescriptorBufferInfo entityTransformInfo{ m_EntityTransformBuffer, 0, VK_WHOLE_SIZE };

        VkDescriptorImageInfo hzbImageInfo{};
        hzbImageInfo.sampler = m_HZBSampler;
        hzbImageInfo.imageView = m_HZBView;
        // HZBPass keeps its image permanently in GENERAL layout (compute-only reads/writes, never
        // an attachment or a sampled-only resource) -- see HZBPass.h's class comment.
        hzbImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[18]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &hzbParamsInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hzbImageInfo, nullptr, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibleLastFrameInfo, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pendingListInfo, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyIndirectInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyDrawCountInfo, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateIndirectInfo, nullptr };
        writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateDrawCountInfo, nullptr };
        writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &softwareClusterListInfo, nullptr };
        writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &candidateCountInfo, nullptr };
        writes[12] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyIndirectOpaqueInfo, nullptr };
        writes[13] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &earlyDrawCountOpaqueInfo, nullptr };
        writes[14] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateIndirectOpaqueInfo, nullptr };
        writes[15] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lateDrawCountOpaqueInfo, nullptr };
        writes[16] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 16, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &softwareClusterListOpaqueInfo, nullptr };
        writes[17] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 17, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 18, writes, 0, nullptr);

        // --- BuildDispatchIndirectArgs set descriptor writes -- binding 0 aliases the pending
        // list buffer, reading only its leading count word (see BuildDispatchIndirectArgs.comp's
        // SourceCountSSBO), binding 1 is the dispatch-args output buffer. ---
        VkDescriptorBufferInfo sourceCountInfo{ m_PendingListBuffer.Handle(), 0, sizeof(uint32_t) };
        VkDescriptorBufferInfo dispatchArgsInfo{ m_LateDispatchArgsBuffer.Handle(), 0, m_LateDispatchArgsBuffer.Size() };

        VkWriteDescriptorSet buildArgsWrites[2]{};
        buildArgsWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sourceCountInfo, nullptr };
        buildArgsWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dispatchArgsInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 2, buildArgsWrites, 0, nullptr);

        // --- Main pipeline layout: shared by both the early and late specializations ---
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
#ifndef NDEBUG
        pushConstantRange.size = 2 * sizeof(uint32_t); // Matches ClusterOcclusionPushConstants { softwareRasterThresholdPixels; disableOcclusionCulling; } -- clusterCount is now read from CandidateCountSSBO (binding 11).
#else
        pushConstantRange.size = 1 * sizeof(uint32_t); // Matches ClusterOcclusionPushConstants { softwareRasterThresholdPixels; }
#endif

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        // --- Early/late pipelines: one shader module, two specializations of the LATE_PASS spec
        // constant (constant_id = 0 in ClusterHZBOcclusionCull.comp), built in a single
        // vkCreateComputePipelines call. ---
        VkShaderModule occlusionShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterHZBOcclusionCull.comp.spv");

        VkSpecializationMapEntry specEntry{};
        specEntry.constantID = 0;
        specEntry.offset = 0;
        specEntry.size = sizeof(VkBool32);

        VkBool32 latePassFalse = VK_FALSE;
        VkBool32 latePassTrue = VK_TRUE;

        VkSpecializationInfo earlySpecInfo{};
        earlySpecInfo.mapEntryCount = 1;
        earlySpecInfo.pMapEntries = &specEntry;
        earlySpecInfo.dataSize = sizeof(VkBool32);
        earlySpecInfo.pData = &latePassFalse;

        VkSpecializationInfo lateSpecInfo{};
        lateSpecInfo.mapEntryCount = 1;
        lateSpecInfo.pMapEntries = &specEntry;
        lateSpecInfo.dataSize = sizeof(VkBool32);
        lateSpecInfo.pData = &latePassTrue;

        VkComputePipelineCreateInfo occlusionPipelineInfos[2]{};
        occlusionPipelineInfos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        occlusionPipelineInfos[0].layout = m_PipelineLayout;
        occlusionPipelineInfos[0].stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        occlusionPipelineInfos[0].stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        occlusionPipelineInfos[0].stage.module = occlusionShaderModule;
        occlusionPipelineInfos[0].stage.pName = "main";
        occlusionPipelineInfos[0].stage.pSpecializationInfo = &earlySpecInfo;

        occlusionPipelineInfos[1] = occlusionPipelineInfos[0];
        occlusionPipelineInfos[1].stage.pSpecializationInfo = &lateSpecInfo;

        VkPipeline occlusionPipelines[2]{};
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 2, occlusionPipelineInfos, nullptr, occlusionPipelines));
        m_EarlyPipeline = occlusionPipelines[0];
        m_LatePipeline = occlusionPipelines[1];

        vkDestroyShaderModule(m_Device, occlusionShaderModule, nullptr);

        // --- BuildDispatchIndirectArgs pipeline ---
        VulkanPipeline::CreateBuildDispatchIndirectArgsPipeline(
            m_Device, m_BuildArgsSetLayout, m_BuildArgsPipelineLayout, m_BuildArgsPipeline);
    }

    void ClusterOcclusionCullingPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[ClusterOcclusionCullingPass] Shutting down pass...");
            if (m_EarlyPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_EarlyPipeline, nullptr);
            }
            if (m_LatePipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_LatePipeline, nullptr);
            }
            if (m_BuildArgsPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_BuildArgsPipeline, nullptr);
            }
            if (m_PipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            }
            if (m_BuildArgsPipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_Device, m_BuildArgsPipelineLayout, nullptr);
            }
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_DescriptorSet / m_BuildArgsDescriptorSet.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_SetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            }
            if (m_BuildArgsSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_Device, m_BuildArgsSetLayout, nullptr);
            }
            if (m_HZBSampler != VK_NULL_HANDLE) {
                vkDestroySampler(m_Device, m_HZBSampler, nullptr);
            }
        }

        m_EarlyPipeline = VK_NULL_HANDLE;
        m_LatePipeline = VK_NULL_HANDLE;
        m_BuildArgsPipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_BuildArgsPipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_BuildArgsSetLayout = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_BuildArgsDescriptorSet = VK_NULL_HANDLE;
        m_HZBSampler = VK_NULL_HANDLE;
        m_HZBView = VK_NULL_HANDLE; // Not owned -- just forgotten.

        // GpuBuffer::Destroy() is null-safe (no-op on an already-empty instance). The candidate
        // metadata/count buffers are borrowed (plain VkBuffer, not GpuBuffer) -- just forgotten.
        m_CandidateMetadataBuffer = VK_NULL_HANDLE;
        m_CandidateCountBuffer = VK_NULL_HANDLE;
        m_ViewParamsBuffer.Destroy();
        m_HZBParamsBuffer.Destroy();
        m_VisibleLastFrameBuffer.Destroy();
        m_PendingListBuffer.Destroy();
        m_EarlyIndirectCommandBuffer.Destroy();
        m_EarlyDrawCountBuffer.Destroy();
        m_LateIndirectCommandBuffer.Destroy();
        m_LateDrawCountBuffer.Destroy();
        m_SoftwareClusterListBuffer.Destroy();
        m_EarlyIndirectCommandOpaqueBuffer.Destroy();
        m_EarlyDrawCountOpaqueBuffer.Destroy();
        m_LateIndirectCommandOpaqueBuffer.Destroy();
        m_LateDrawCountOpaqueBuffer.Destroy();
        m_SoftwareClusterListOpaqueBuffer.Destroy();
        m_LateDispatchArgsBuffer.Destroy();

        m_MaxClusters = 0;
        m_HZBMip0Width = 0.0f;
        m_HZBMip0Height = 0.0f;
        m_HZBMipCount = 0.0f;
        m_LastSoftwareRasterThresholdPixels = 0.0f;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterOcclusionCullingPass::ClearPersistedVisibility(VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_VisibleLastFrameBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    void ClusterOcclusionCullingPass::RecordClearFrame(VkCommandBuffer cmd) {
        // Only the leading count words need clearing every frame -- the shaders never read past
        // their respective counts, so stale entries beyond that from a previous, larger frame are
        // never observed. m_VisibleLastFrameBuffer is deliberately NOT touched here.
        vkCmdFillBuffer(cmd, m_PendingListBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_EarlyDrawCountBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_LateDrawCountBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_SoftwareClusterListBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_EarlyDrawCountOpaqueBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_LateDrawCountOpaqueBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_SoftwareClusterListOpaqueBuffer.Handle(), 0, sizeof(uint32_t), 0u);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    void ClusterOcclusionCullingPass::RecordEarlyPass(VkCommandBuffer cmd, const ClusterCullViewParams& viewParams, const maths::mat4& viewProj,
        float projScaleY, VkBuffer earlyDispatchArgsBuffer, float softwareRasterThresholdPixels, uint32_t disableOcclusionCulling) {

        m_LastSoftwareRasterThresholdPixels = softwareRasterThresholdPixels;

        // --- Upload this frame's frustum/camera params and HZB projection params ---
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ClusterCullViewParams), &viewParams);

        HZBOcclusionViewParams hzbParams{};
        hzbParams.viewProj = viewProj;
        hzbParams.hzbMip0Width = m_HZBMip0Width;
        hzbParams.hzbMip0Height = m_HZBMip0Height;
        hzbParams.hzbMipCount = m_HZBMipCount;
        hzbParams.projScaleY = projScaleY;
        vkCmdUpdateBuffer(cmd, m_HZBParamsBuffer.Handle(), 0, sizeof(HZBOcclusionViewParams), &hzbParams);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        // --- Dispatch: one invocation per this frame's GPU-compacted candidate (count only ever
        // known on the GPU -- see CandidateCountSSBO, binding 11 -- hence the indirect dispatch),
        // early (LATE_PASS = false) specialization ---
#ifndef NDEBUG
        struct { float softwareRasterThresholdPixels; uint32_t disableOcclusionCulling; } pushConstants{ softwareRasterThresholdPixels, disableOcclusionCulling };
#else
        struct { float softwareRasterThresholdPixels; } pushConstants{ softwareRasterThresholdPixels };
#endif

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_EarlyPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatchIndirect(cmd, earlyDispatchArgsBuffer, 0);

        // --- Make the early indirect-draw list + its count, the software cluster list, and the
        // pending list all visible to a later vkCmdDrawIndexedIndirectCount /
        // RecordBuildLateDispatchArgs() / RecordLatePass() / renderer::ClusterSoftwareRasterPass
        // compute reads. ---
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void ClusterOcclusionCullingPass::RecordBuildLateDispatchArgs(VkCommandBuffer cmd) {
        // One thread per pending-list entry -- perElementMultiplier = 1, unlike
        // renderer::ClusterSoftwareRasterPass's own use of this same shader (see
        // BuildDispatchIndirectArgs.comp's class comment).
        uint32_t pushConstants[2] = { kWorkgroupSize, 1u };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipelineLayout, 0, 1, &m_BuildArgsDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_BuildArgsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
        vkCmdDispatch(cmd, 1, 1, 1);

        // The dispatch args must be visible to RecordLatePass()'s vkCmdDispatchIndirect --
        // classified the same as vkCmdDrawIndirect's indirect-buffer read
        // (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT / VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

    void ClusterOcclusionCullingPass::RecordLatePass(VkCommandBuffer cmd, uint32_t disableOcclusionCulling) {
        // --- Indirect dispatch: group count comes from GetLateDispatchArgsBuffer(), sized by
        // RecordBuildLateDispatchArgs() from the pending list's own atomic count -- no CPU
        // round-trip. The late (LATE_PASS = true) specialization reads g_PendingList, the (by now
        // freshly rebuilt, by the caller) HZB texture, and -- via ShouldUseSoftwareRaster() --
        // pc.softwareRasterThresholdPixels. The whole push-constant block must still be re-pushed
        // here: RecordBuildLateDispatchArgs() bound a different, incompatible pipeline layout in
        // between, which the Vulkan spec allows to invalidate push constant values pushed under
        // m_PipelineLayout earlier in this same command buffer (see
        // renderer::ClusterOcclusionCullingPass.h's RecordLatePass doc). ---
#ifndef NDEBUG
        struct { float softwareRasterThresholdPixels; uint32_t disableOcclusionCulling; } pushConstants{ m_LastSoftwareRasterThresholdPixels, disableOcclusionCulling };
#else
        struct { float softwareRasterThresholdPixels; } pushConstants{ m_LastSoftwareRasterThresholdPixels };
#endif

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LatePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatchIndirect(cmd, m_LateDispatchArgsBuffer.Handle(), 0);

        // --- Make the late indirect-draw list + its count visible to a later
        // vkCmdDrawIndexedIndirectCount. ---
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
    }

}
