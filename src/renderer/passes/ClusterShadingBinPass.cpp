#include "renderer/passes/ClusterShadingBinPass.h"

#include <array>
#include <format>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/Logger.h"
#include "renderer/MaterialParameterTable.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of PixelScratchEntry/SortedPixelEntry in shading_bin_common.glsl
        // (both 12 bytes: 2x uint32 + 1x float, std430 needs no padding since 12 is already a
        // multiple of the 4-byte scalar alignment) -- used only for sizeof() below, never
        // constructed/copied on the CPU (both buffers are written and read entirely on the GPU).
        struct PixelScratchEntryGpu { uint32_t materialSlot; uint32_t packedVisID; float winningDepth; };
        struct SortedPixelEntryGpu { uint32_t pixelLinearIndex; uint32_t packedVisID; float winningDepth; };
        static_assert(sizeof(PixelScratchEntryGpu) == 12, "Must match PixelScratchEntry in shading_bin_common.glsl exactly (std430 layout)");
        static_assert(sizeof(SortedPixelEntryGpu) == 12, "Must match SortedPixelEntry in shading_bin_common.glsl exactly (std430 layout)");

        // Byte-for-byte mirror of the ViewportSizePC push constant block declared identically in
        // ClusterShadingBinClassify.comp and ClusterShadingBinScatter.comp.
        struct ViewportSizePC { uint32_t width; uint32_t height; };
        static_assert(sizeof(ViewportSizePC) == 8, "ViewportSizePC must match the GLSL push constant block exactly");

    } // namespace

    void ClusterShadingBinPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, VkBuffer clusterMetadataBuffer,
        VkImageView hwClusterIDView, VkImageView hwTriangleIDView, VkImageView hwDepthView,
        VkImageView swVisBufferAtomicView, VkImageView outputColorView, VkImageView outputNormalView,
        VkImageView outputDepthView, VkImageView outputAlbedoView, VkImageView outputRoughnessMetallicView) {
        Shutdown();

        m_Device = device;
        m_RenderExtent = renderExtent;
        const uint32_t pixelCount = renderExtent.width * renderExtent.height;

        // --- Buffers ---
        m_BinHistogramBuffer.Create(allocator, sizeof(uint32_t) * kMaxMaterials,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_PixelScratchBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(PixelScratchEntryGpu)) * pixelCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_BinOffsetsBuffer.Create(allocator, sizeof(uint32_t) * kMaxMaterials,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_BinCursorBuffer.Create(allocator, sizeof(uint32_t) * kMaxMaterials,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        // INDIRECT_BUFFER_BIT: this buffer is later read by vkCmdDispatchIndirect (in
        // renderer::ClusterResolvePass::RecordResolveBinned) -- see BuildDispatchIndirectArgs.comp's
        // own C++ counterpart (ClusterOcclusionCullingPass.cpp) for the identical usage-flag pattern.
        m_BinDispatchArgsBuffer.Create(allocator, sizeof(VkDispatchIndirectCommand) * kMaxMaterials,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_SortedPixelListBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(SortedPixelEntryGpu)) * pixelCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // --- Depth sampler for Stage A's g_HWDepthTexture (visbuffer_arbitration.glsl) -- nearest
        // filtering, matching HZBPass/ClusterResolvePass's own depth-sampling convention (the
        // shader always reads exact integer texels via texelFetch, no filtering ever happens).
        // Not exposed by ClusterResolvePass for borrowing (private, no getter), so this class owns
        // its own small, cheap copy rather than plumbing a new accessor through that class.
        VkSampler depthSampler = VulkanUtils::CreateNearestSampler(m_Device);
        m_DepthSampler = depthSampler;

        // --- Descriptor pool: sized for the union of all 3 sets' bindings (one pool, one alloc). ---
        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 }; // Classify:3 + PrefixSum:4 + Scatter:3
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 };   // Classify only: 3 VisBuffer/atomic + 5 output images
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }; // Classify only: HW depth

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 3;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        // =====================================================================================
        // Stage A: Classify
        // =====================================================================================
        {
            std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // ClusterCullMetadataSSBO
            bindings[1] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };   // g_HWClusterIDImage
            bindings[2] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };   // g_HWTriangleIDImage
            bindings[3] = { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_HWDepthTexture
            bindings[4] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };   // g_SWVisBufferAtomic
            bindings[5] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_BinHistogram
            bindings[6] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_PixelScratch
            bindings[7] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };   // g_OutputColor
            bindings[8] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };   // g_OutputNormal
            bindings[9] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_OutputDepth
            bindings[10] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_OutputAlbedo
            bindings[11] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_OutputRoughnessMetallic

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ClassifySetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_ClassifySetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_ClassifySet));

            VkDescriptorBufferInfo clusterMetaInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo hwClusterInfo{ VK_NULL_HANDLE, hwClusterIDView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo hwTriangleInfo{ VK_NULL_HANDLE, hwTriangleIDView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo hwDepthInfo{ depthSampler, hwDepthView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo swAtomicInfo{ VK_NULL_HANDLE, swVisBufferAtomicView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo histogramInfo{ m_BinHistogramBuffer.Handle(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo scratchInfo{ m_PixelScratchBuffer.Handle(), 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, outputColorView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputNormalInfo{ VK_NULL_HANDLE, outputNormalView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputDepthInfo{ VK_NULL_HANDLE, outputDepthView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputAlbedoInfo{ VK_NULL_HANDLE, outputAlbedoView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputRMInfo{ VK_NULL_HANDLE, outputRoughnessMetallicView, VK_IMAGE_LAYOUT_GENERAL };

            std::array<VkWriteDescriptorSet, 12> writes{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetaInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hwClusterInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hwTriangleInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hwDepthInfo, nullptr, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &swAtomicInfo, nullptr, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &histogramInfo, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &scratchInfo, nullptr };
            writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputColorInfo, nullptr, nullptr };
            writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputNormalInfo, nullptr, nullptr };
            writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputDepthInfo, nullptr, nullptr };
            writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputAlbedoInfo, nullptr, nullptr };
            writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClassifySet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputRMInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ViewportSizePC) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ClassifySetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ClassifyPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterShadingBinClassify.comp.spv");
            m_ClassifyPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ClassifyPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // Stage B: Prefix-sum
        // =====================================================================================
        {
            std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_BinHistogram
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_BinOffsets
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_BinCursor
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_BinDispatchArgs

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_PrefixSumSetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_PrefixSumSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_PrefixSumSet));

            VkDescriptorBufferInfo histogramInfo{ m_BinHistogramBuffer.Handle(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo offsetsInfo{ m_BinOffsetsBuffer.Handle(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo cursorInfo{ m_BinCursorBuffer.Handle(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo dispatchArgsInfo{ m_BinDispatchArgsBuffer.Handle(), 0, VK_WHOLE_SIZE };

            std::array<VkWriteDescriptorSet, 4> writes{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_PrefixSumSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &histogramInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_PrefixSumSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &offsetsInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_PrefixSumSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &cursorInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_PrefixSumSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dispatchArgsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_PrefixSumSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PrefixSumPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterShadingBinPrefixSum.comp.spv");
            m_PrefixSumPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PrefixSumPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // Stage C: Scatter
        // =====================================================================================
        {
            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_PixelScratch
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_BinCursor
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_SortedPixelList

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ScatterSetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_ScatterSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_ScatterSet));

            VkDescriptorBufferInfo scratchInfo{ m_PixelScratchBuffer.Handle(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo cursorInfo{ m_BinCursorBuffer.Handle(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo sortedListInfo{ m_SortedPixelListBuffer.Handle(), 0, VK_WHOLE_SIZE };

            std::array<VkWriteDescriptorSet, 3> writes{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScatterSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &scratchInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScatterSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &cursorInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ScatterSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sortedListInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ViewportSizePC) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ScatterSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ScatterPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterShadingBinScatter.comp.spv");
            m_ScatterPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ScatterPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        LOG_INFO(std::format("[ClusterShadingBinPass] Initialized: {} materials, {}x{} pixel buffers.",
            kMaxMaterials, renderExtent.width, renderExtent.height));

        (void)commandPool;
        (void)queue; // Reserved for parity with every other pass's Init() signature; no one-time submit needed here (no image layout transitions, no CPU-authored upload).
    }

    void ClusterShadingBinPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_ClassifyPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ClassifyPipeline, nullptr);
            if (m_PrefixSumPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_PrefixSumPipeline, nullptr);
            if (m_ScatterPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ScatterPipeline, nullptr);
            if (m_ClassifyPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ClassifyPipelineLayout, nullptr);
            if (m_PrefixSumPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PrefixSumPipelineLayout, nullptr);
            if (m_ScatterPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ScatterPipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees all 3 sets -- not freed individually.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_ClassifySetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ClassifySetLayout, nullptr);
            if (m_PrefixSumSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_PrefixSumSetLayout, nullptr);
            if (m_ScatterSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ScatterSetLayout, nullptr);
            if (m_DepthSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_DepthSampler, nullptr);
        }

        m_ClassifyPipeline = VK_NULL_HANDLE;
        m_PrefixSumPipeline = VK_NULL_HANDLE;
        m_ScatterPipeline = VK_NULL_HANDLE;
        m_ClassifyPipelineLayout = VK_NULL_HANDLE;
        m_PrefixSumPipelineLayout = VK_NULL_HANDLE;
        m_ScatterPipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_ClassifySetLayout = VK_NULL_HANDLE;
        m_PrefixSumSetLayout = VK_NULL_HANDLE;
        m_ScatterSetLayout = VK_NULL_HANDLE;
        m_ClassifySet = VK_NULL_HANDLE;
        m_PrefixSumSet = VK_NULL_HANDLE;
        m_ScatterSet = VK_NULL_HANDLE;
        m_DepthSampler = VK_NULL_HANDLE;

        m_BinHistogramBuffer.Destroy();
        m_PixelScratchBuffer.Destroy();
        m_BinOffsetsBuffer.Destroy();
        m_BinCursorBuffer.Destroy();
        m_BinDispatchArgsBuffer.Destroy();
        m_SortedPixelListBuffer.Destroy();

        m_RenderExtent = { 0, 0 };
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterShadingBinPass::RecordClassifyAndSort(VkCommandBuffer cmd, VkExtent2D renderExtent) {
        ViewportSizePC viewportPC{ renderExtent.width, renderExtent.height };

        // --- Reset histogram to zero (Classify's atomicAdd needs a clean starting point every
        // frame; g_BinOffsets/g_BinCursor/g_BinDispatchArgs need no reset -- PrefixSum fully
        // overwrites all kMaxMaterials entries of each, deterministically, every invocation). ---
        vkCmdFillBuffer(cmd, m_BinHistogramBuffer.Handle(), 0, sizeof(uint32_t) * kMaxMaterials, 0u);
        {
            // Plain (non-buffer-scoped) memory barrier, matching this codebase's own established
            // convention for a post-vkCmdFillBuffer reset barrier -- see e.g.
            // ClusterOcclusionCullingPass::RecordClearFrame's identical stage/access pair.
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // =====================================================================================
        // Stage A: Classify
        // =====================================================================================
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ClassifyPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ClassifyPipelineLayout, 0, 1, &m_ClassifySet, 0, nullptr);
        vkCmdPushConstants(cmd, m_ClassifyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ViewportSizePC), &viewportPC);
        {
            uint32_t groupCountX = (renderExtent.width + kClassifyScatterWorkgroupSize - 1) / kClassifyScatterWorkgroupSize;
            uint32_t groupCountY = (renderExtent.height + kClassifyScatterWorkgroupSize - 1) / kClassifyScatterWorkgroupSize;
            vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
        }

        // Makes the histogram (read by Stage B) and the pixel-scratch buffer (read by Stage C)
        // visible -- both are plain COMPUTE_SHADER -> COMPUTE_SHADER dependencies since every
        // consumer is itself a compute dispatch.
        {
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }

        // =====================================================================================
        // Stage B: Prefix-sum (single invocation)
        // =====================================================================================
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PrefixSumPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PrefixSumPipelineLayout, 0, 1, &m_PrefixSumSet, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);

        // Makes g_BinOffsets/g_BinCursor visible to Stage C's compute reads (g_BinCursor needs
        // both READ and WRITE access -- Stage C's atomicAdd is a read-modify-write) and
        // g_BinDispatchArgs visible to the LATER vkCmdDispatchIndirect calls
        // renderer::ClusterResolvePass::RecordResolveBinned issues after this function returns --
        // covering that future access here (rather than deferring it to that other call) is safe:
        // memory made visible to a given dstStage/dstAccess pair stays visible to every
        // subsequently-submitted command using that same pair, with no intervening write.
        {
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
        }

        // =====================================================================================
        // Stage C: Scatter
        // =====================================================================================
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScatterPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScatterPipelineLayout, 0, 1, &m_ScatterSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_ScatterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ViewportSizePC), &viewportPC);
        {
            uint32_t groupCountX = (renderExtent.width + kClassifyScatterWorkgroupSize - 1) / kClassifyScatterWorkgroupSize;
            uint32_t groupCountY = (renderExtent.height + kClassifyScatterWorkgroupSize - 1) / kClassifyScatterWorkgroupSize;
            vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
        }

        // Trailing barrier: makes g_SortedPixelList visible to renderer::ClusterResolvePass::
        // RecordResolveBinned's own compute reads (Stage D) -- g_BinOffsets/g_BinHistogram/
        // g_BinDispatchArgs are already visible to that same dstStage/dstAccess combination from
        // the barrier after Stage B above (no repeat needed, see that barrier's own comment).
        {
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }
    }

}
