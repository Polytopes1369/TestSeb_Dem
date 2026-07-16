#include "renderer/ClusterOcclusionCullingPass.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "core/Logger.h"

namespace renderer {

    namespace {

        // Mirrors HZBPass::ReadShaderFile / ClusterCullingPass's own copy -- duplicated rather
        // than shared because this class is deliberately self-contained (no VulkanContext
        // dependency), matching this codebase's existing per-pass convention.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("ClusterOcclusionCullingPass: failed to open SPIR-V file: " + filename);
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

    } // namespace

    void ClusterOcclusionCullingPass::Init(VkDevice device, VmaAllocator allocator, uint32_t maxClusters,
        VkImageView hzbFullView, VkExtent2D hzbMip0Extent, uint32_t hzbMipLevelCount) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_MaxClusters = maxClusters;
        m_HZBView = hzbFullView;
        m_HZBMip0Width = static_cast<float>(hzbMip0Extent.width);
        m_HZBMip0Height = static_cast<float>(hzbMip0Extent.height);
        m_HZBMipCount = static_cast<float>(hzbMipLevelCount);

        // --- Buffers ---
        m_ClusterMetadataBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(ClusterCullMetadata)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

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
        // ClearPersistedVisibility(), never by RecordClearFrame().
        m_VisibleLastFrameBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(uint32_t)) * maxClusters,
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

        // DEBUG (temporary): one outcome code per cluster slot, overwritten every frame by whichever
        // decision point in ClusterHZBOcclusionCull.comp last touched that cluster -- needs
        // TRANSFER_SRC_BIT so ClusterRenderPipeline can read it back for the diagnostic histogram.
        m_DebugOutcomeBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(sizeof(uint32_t)) * maxClusters,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = m_HZBMipCount;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_HZBSampler));

        // --- Main descriptor set layout: 12 bindings, all compute-visible, matching
        // ClusterHZBOcclusionCull.comp's set = 0 bindings 0..11 exactly. Shared by both the early
        // and late pipelines. ---
        VkDescriptorSetLayoutBinding bindings[12]{};
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
        bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // DEBUG: DebugOutcomeSSBO

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 12;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        // --- BuildDispatchIndirectArgs set layout: 2 bindings ---
        VkDescriptorSetLayoutBinding buildArgsBindings[2]{};
        buildArgsBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SourceCountSSBO
        buildArgsBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DispatchArgsSSBO

        VkDescriptorSetLayoutCreateInfo buildArgsLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        buildArgsLayoutInfo.bindingCount = 2;
        buildArgsLayoutInfo.pBindings = buildArgsBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &buildArgsLayoutInfo, nullptr, &m_BuildArgsSetLayout));

        // --- One shared descriptor pool for both sets ---
        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9 + 2 };       // Main set's 8 SSBOs + DEBUG DebugOutcomeSSBO + BuildArgs set's 2 SSBOs.
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
        VkDescriptorBufferInfo clusterInfo{ m_ClusterMetadataBuffer.Handle(), 0, m_ClusterMetadataBuffer.Size() };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorBufferInfo hzbParamsInfo{ m_HZBParamsBuffer.Handle(), 0, m_HZBParamsBuffer.Size() };
        VkDescriptorBufferInfo visibleLastFrameInfo{ m_VisibleLastFrameBuffer.Handle(), 0, m_VisibleLastFrameBuffer.Size() };
        VkDescriptorBufferInfo pendingListInfo{ m_PendingListBuffer.Handle(), 0, m_PendingListBuffer.Size() };
        VkDescriptorBufferInfo earlyIndirectInfo{ m_EarlyIndirectCommandBuffer.Handle(), 0, m_EarlyIndirectCommandBuffer.Size() };
        VkDescriptorBufferInfo earlyDrawCountInfo{ m_EarlyDrawCountBuffer.Handle(), 0, m_EarlyDrawCountBuffer.Size() };
        VkDescriptorBufferInfo lateIndirectInfo{ m_LateIndirectCommandBuffer.Handle(), 0, m_LateIndirectCommandBuffer.Size() };
        VkDescriptorBufferInfo lateDrawCountInfo{ m_LateDrawCountBuffer.Handle(), 0, m_LateDrawCountBuffer.Size() };
        VkDescriptorBufferInfo softwareClusterListInfo{ m_SoftwareClusterListBuffer.Handle(), 0, m_SoftwareClusterListBuffer.Size() };
        VkDescriptorBufferInfo debugOutcomeInfo{ m_DebugOutcomeBuffer.Handle(), 0, m_DebugOutcomeBuffer.Size() };

        VkDescriptorImageInfo hzbImageInfo{};
        hzbImageInfo.sampler = m_HZBSampler;
        hzbImageInfo.imageView = m_HZBView;
        // HZBPass keeps its image permanently in GENERAL layout (compute-only reads/writes, never
        // an attachment or a sampled-only resource) -- see HZBPass.h's class comment.
        hzbImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[12]{};
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
        writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &debugOutcomeInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 12, writes, 0, nullptr);

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
        pushConstantRange.size = 2 * sizeof(uint32_t); // Matches ClusterOcclusionPushConstants { clusterCount; softwareRasterThresholdPixels; } (float is 4 bytes, same as uint32_t).

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        // --- Early/late pipelines: one shader module, two specializations of the LATE_PASS spec
        // constant (constant_id = 0 in ClusterHZBOcclusionCull.comp), built in a single
        // vkCreateComputePipelines call. ---
        std::vector<char> occlusionShaderCode = ReadShaderFile("shaders/ClusterHZBOcclusionCull.comp.spv");
        VkShaderModule occlusionShaderModule = CreateShaderModule(m_Device, occlusionShaderCode);

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
        VkPushConstantRange buildArgsPushConstantRange{};
        buildArgsPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        buildArgsPushConstantRange.offset = 0;
        buildArgsPushConstantRange.size = 2 * sizeof(uint32_t); // Matches BuildDispatchArgsPushConstants { workgroupSize; perElementMultiplier; }.

        VkPipelineLayoutCreateInfo buildArgsPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        buildArgsPipelineLayoutInfo.setLayoutCount = 1;
        buildArgsPipelineLayoutInfo.pSetLayouts = &m_BuildArgsSetLayout;
        buildArgsPipelineLayoutInfo.pushConstantRangeCount = 1;
        buildArgsPipelineLayoutInfo.pPushConstantRanges = &buildArgsPushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &buildArgsPipelineLayoutInfo, nullptr, &m_BuildArgsPipelineLayout));

        std::vector<char> buildArgsShaderCode = ReadShaderFile("shaders/BuildDispatchIndirectArgs.comp.spv");
        VkShaderModule buildArgsShaderModule = CreateShaderModule(m_Device, buildArgsShaderCode);

        VkComputePipelineCreateInfo buildArgsPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        buildArgsPipelineInfo.layout = m_BuildArgsPipelineLayout;
        buildArgsPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        buildArgsPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        buildArgsPipelineInfo.stage.module = buildArgsShaderModule;
        buildArgsPipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &buildArgsPipelineInfo, nullptr, &m_BuildArgsPipeline));
        vkDestroyShaderModule(m_Device, buildArgsShaderModule, nullptr);
    }

    void ClusterOcclusionCullingPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
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

        // GpuBuffer::Destroy() is null-safe (no-op on an already-empty instance).
        m_ClusterMetadataBuffer.Destroy();
        m_ViewParamsBuffer.Destroy();
        m_HZBParamsBuffer.Destroy();
        m_VisibleLastFrameBuffer.Destroy();
        m_PendingListBuffer.Destroy();
        m_EarlyIndirectCommandBuffer.Destroy();
        m_EarlyDrawCountBuffer.Destroy();
        m_LateIndirectCommandBuffer.Destroy();
        m_LateDrawCountBuffer.Destroy();
        m_SoftwareClusterListBuffer.Destroy();
        m_DebugOutcomeBuffer.Destroy();
        m_LateDispatchArgsBuffer.Destroy();

        m_MaxClusters = 0;
        m_HZBMip0Width = 0.0f;
        m_HZBMip0Height = 0.0f;
        m_HZBMipCount = 0.0f;
        m_LastSoftwareRasterThresholdPixels = 0.0f;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterOcclusionCullingPass::ClearPersistedVisibility(VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, m_VisibleLastFrameBuffer.Handle(), 0, VK_WHOLE_SIZE, 0u);

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

    void ClusterOcclusionCullingPass::UploadClusterMetadata(VkCommandPool commandPool, VkQueue queue, const std::vector<ClusterCullMetadata>& clusters) {
        assert(clusters.size() <= m_MaxClusters && "ClusterOcclusionCullingPass::UploadClusterMetadata: candidate list exceeds maxClusters");
        if (clusters.empty()) {
            return;
        }

        // m_ClusterMetadataBuffer is VMA_MEMORY_USAGE_GPU_ONLY (not host-visible), so uploading
        // the CPU-authored candidate list requires a temporary host-visible staging buffer plus an
        // explicit GPU-side copy, mirroring VulkanContext::UploadEntityData's one-time-submit
        // staging pattern (identical to renderer::ClusterCullingPass::UploadClusterMetadata).
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

        VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd;
        VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = uploadSize;
        vkCmdCopyBuffer(cmd, stagingBuffer, m_ClusterMetadataBuffer.Handle(), 1, &copyRegion);

        VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));

        vkFreeCommandBuffers(m_Device, commandPool, 1, &cmd);
        vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
    }

    void ClusterOcclusionCullingPass::RecordClearFrame(VkCommandBuffer cmd) {
        // Only the leading count words need clearing every frame -- the shaders never read past
        // their respective counts, so stale entries beyond that from a previous, larger frame are
        // never observed. m_VisibleLastFrameBuffer is deliberately NOT touched here.
        vkCmdFillBuffer(cmd, m_PendingListBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_EarlyDrawCountBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_LateDrawCountBuffer.Handle(), 0, sizeof(uint32_t), 0u);
        vkCmdFillBuffer(cmd, m_SoftwareClusterListBuffer.Handle(), 0, sizeof(uint32_t), 0u);

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

    void ClusterOcclusionCullingPass::RecordEarlyPass(VkCommandBuffer cmd, const ClusterCullViewParams& viewParams, const maths::mat4& viewProj,
        float projScaleY, uint32_t clusterCount, float softwareRasterThresholdPixels) {
        assert(clusterCount <= m_MaxClusters && "ClusterOcclusionCullingPass::RecordEarlyPass: clusterCount exceeds maxClusters");

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

        VkMemoryBarrier2 uboBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uboBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uboBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uboBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        uboBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;

        VkDependencyInfo uboDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        uboDependency.memoryBarrierCount = 1;
        uboDependency.pMemoryBarriers = &uboBarrier;
        vkCmdPipelineBarrier2(cmd, &uboDependency);

        // --- Dispatch: one invocation per candidate cluster, early (LATE_PASS = false) specialization ---
        // Matches ClusterOcclusionPushConstants { uint clusterCount; float softwareRasterThresholdPixels; }.
        struct { uint32_t clusterCount; float softwareRasterThresholdPixels; } pushConstants{ clusterCount, softwareRasterThresholdPixels };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_EarlyPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

        uint32_t groupCount = (clusterCount + kWorkgroupSize - 1) / kWorkgroupSize;
        if (groupCount > 0) {
            vkCmdDispatch(cmd, groupCount, 1, 1);
        }

        // --- Make the early indirect-draw list + its count, the software cluster list, and the
        // pending list all visible to a later vkCmdDrawIndexedIndirectCount /
        // RecordBuildLateDispatchArgs() / RecordLatePass() / renderer::ClusterSoftwareRasterPass
        // compute reads. ---
        VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo outputDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        outputDependency.memoryBarrierCount = 1;
        outputDependency.pMemoryBarriers = &outputBarrier;
        vkCmdPipelineBarrier2(cmd, &outputDependency);
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
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void ClusterOcclusionCullingPass::RecordLatePass(VkCommandBuffer cmd) {
        // --- Indirect dispatch: group count comes from GetLateDispatchArgsBuffer(), sized by
        // RecordBuildLateDispatchArgs() from the pending list's own atomic count -- no CPU
        // round-trip. The late (LATE_PASS = true) specialization reads g_PendingList, the (by now
        // freshly rebuilt, by the caller) HZB texture, and -- via ShouldUseSoftwareRaster() --
        // pc.softwareRasterThresholdPixels; the push-constant clusterCount itself is unused by this
        // specialization's code path (dead-code-eliminated at pipeline specialization time), but
        // the whole push-constant block must still be re-pushed here: RecordBuildLateDispatchArgs()
        // bound a different, incompatible pipeline layout in between, which the Vulkan spec allows
        // to invalidate push constant values pushed under m_PipelineLayout earlier in this same
        // command buffer (see renderer::ClusterOcclusionCullingPass.h's RecordLatePass doc). ---
        struct { uint32_t clusterCount; float softwareRasterThresholdPixels; } pushConstants{ 0u, m_LastSoftwareRasterThresholdPixels };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LatePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatchIndirect(cmd, m_LateDispatchArgsBuffer.Handle(), 0);

        // --- Make the late indirect-draw list + its count visible to a later
        // vkCmdDrawIndexedIndirectCount. ---
        VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        outputBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

        VkDependencyInfo outputDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        outputDependency.memoryBarrierCount = 1;
        outputDependency.pMemoryBarriers = &outputBarrier;
        vkCmdPipelineBarrier2(cmd, &outputDependency);
    }

}
