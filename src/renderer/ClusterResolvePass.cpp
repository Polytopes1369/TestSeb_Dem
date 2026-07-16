#include "renderer/ClusterResolvePass.h"

#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/Logger.h"
#include "renderer/VulkanPipeline.h"

namespace renderer {

    namespace {

        // Mirrors HZBPass::ReadShaderFile / every other pass's own copy -- duplicated rather than
        // shared because this class is deliberately self-contained (no VulkanContext dependency),
        // matching this codebase's existing per-pass convention.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("ClusterResolvePass: failed to open SPIR-V file: " + filename);
            }
            size_t fileSize = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            file.close();
            return buffer;
        }

        // Byte-for-byte mirror of ResolveViewParamsUBO in ClusterResolve.comp (std140): mat4 (64
        // bytes) + vec2 (8 bytes) + 2 pad floats rounding up to the struct's own 16-byte base
        // alignment (80 bytes total) -- same shape as ClusterSoftwareRasterPass.cpp's
        // SoftwareRasterViewParams.
        struct ResolveViewParams {
            maths::mat4 viewProj;
            float viewportWidth = 0.0f;
            float viewportHeight = 0.0f;
            float _pad0 = 0.0f;
            float _pad1 = 0.0f;
        };
        static_assert(sizeof(ResolveViewParams) == 80,
            "ResolveViewParams must match ResolveViewParamsUBO in ClusterResolve.comp exactly (std140 layout)");

    } // namespace

    void ClusterResolvePass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
        VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
        VkImageView hwClusterIDView, VkImageView hwTriangleIDView, VkImageView hwDepthView,
        VkImageView swVisBufferAtomicView, const std::vector<VkDescriptorImageInfo>& maskImageInfos) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;
        uint32_t maskTextureCount = static_cast<uint32_t>(maskImageInfos.size());

        // --- Output color image: RGBA8 storage image, sized to the render target. ---
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kOutputColorFormat;
        imageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // SAMPLED_BIT/TRANSFER_SRC_BIT: a future consumer may sample this directly or blit it to
        // the swapchain -- neither is wired up by this change (see the class comment), but the
        // usage flags are pre-positioned exactly like VulkanContext's swapchain TRANSFER_DST_BIT
        // was pre-positioned ahead of the VisBuffer conversion that later needed it.
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo imageAllocInfo{};
        imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &imageAllocInfo, &m_OutputColorImage, &m_OutputColorAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_OutputColorImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kOutputColorFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputColorView));

        // --- One-time UNDEFINED -> GENERAL transition (mirrors HZBPass::Init /
        // ClusterSoftwareRasterPass::Init's own one-shot pattern) -- stays in GENERAL for its
        // entire lifetime, touched only by this class's own compute shader. ---
        {
            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;

            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.srcAccessMask = 0;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_OutputColorImage;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));

            vkFreeCommandBuffers(m_Device, commandPool, 1, &cmd);
        }

        // --- Depth sampler: nearest filtering, matching HZBPass's own depth-sampling convention
        // (the shader always reads exact integer texels via texelFetch, no filtering ever actually
        // happens). ---
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_DepthSampler));

        // --- View-params UBO ---
        m_ViewParamsBuffer.Create(
            allocator,
            sizeof(ResolveViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // --- Descriptor set layout: 9 bindings, matching ClusterResolve.comp's set = 0 bindings
        // 0..8 exactly. ---
        VkDescriptorSetLayoutBinding bindings[9]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // ClusterCullMetadataSSBO
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // CompressedClusterPoolSSBO
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_HWClusterIDImage (r32ui)
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_HWTriangleIDImage (r32ui)
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_HWDepthTexture
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_SWVisBufferAtomic (r64ui)
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_OutputColor (rgba8)
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // ResolveViewParamsUBO
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_MaskTextures[]

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 9;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 };
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 + maskTextureCount };
        poolSizes[3] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAlloc.descriptorPool = m_DescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_DescriptorSet));

        VkDescriptorBufferInfo clusterMetadataInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo compressedPoolInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

        VkDescriptorImageInfo hwClusterIDInfo{};
        hwClusterIDInfo.imageView = hwClusterIDView;
        hwClusterIDInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // See the class/Init() doc comment: caller must guarantee this.

        VkDescriptorImageInfo hwTriangleIDInfo{};
        hwTriangleIDInfo.imageView = hwTriangleIDView;
        hwTriangleIDInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo hwDepthInfo{};
        hwDepthInfo.sampler = m_DepthSampler;
        hwDepthInfo.imageView = hwDepthView;
        hwDepthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo swAtomicInfo{};
        swAtomicInfo.imageView = swVisBufferAtomicView;
        swAtomicInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // renderer::ClusterSoftwareRasterPass keeps this permanently GENERAL.

        VkDescriptorImageInfo outputColorInfo{};
        outputColorInfo.imageView = m_OutputColorView;
        outputColorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[9]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetadataInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hwClusterIDInfo, nullptr, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hwTriangleIDInfo, nullptr, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hwDepthInfo, nullptr, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &swAtomicInfo, nullptr, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputColorInfo, nullptr, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 8, 0, maskTextureCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskImageInfos.data(), nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 9, writes, 0, nullptr);

        // --- Pipeline layout + pipeline ---
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
#ifndef NDEBUG
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t); // viewMode
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
#else
        pipelineLayoutInfo.pushConstantRangeCount = 0;
#endif
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        std::vector<char> shaderCode = ReadShaderFile("shaders/ClusterResolve.comp.spv");
        VkShaderModule shaderModule = VulkanPipeline::CreateShaderModule(m_Device, shaderCode);
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shaderModule);
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);
    }

    void ClusterResolvePass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_DescriptorSet -- not freed individually.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_DepthSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_DepthSampler, nullptr);
            if (m_OutputColorView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputColorView, nullptr);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_DepthSampler = VK_NULL_HANDLE;
        m_OutputColorView = VK_NULL_HANDLE;

        m_ViewParamsBuffer.Destroy();

        // vmaDestroyImage tolerates VK_NULL_HANDLE for both handle arguments (no-op), matching
        // GpuBuffer::Destroy()'s own null-safe convention.
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_OutputColorImage, m_OutputColorAllocation);
        }
        m_OutputColorImage = VK_NULL_HANDLE;
        m_OutputColorAllocation = VK_NULL_HANDLE;

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterResolvePass::RecordResolve(VkCommandBuffer cmd, const maths::mat4& viewProj, uint32_t debugViewMode) {
        ResolveViewParams viewParams{};
        viewParams.viewProj = viewProj;
        viewParams.viewportWidth = static_cast<float>(m_RenderExtent.width);
        viewParams.viewportHeight = static_cast<float>(m_RenderExtent.height);
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ResolveViewParams), &viewParams);

        VkMemoryBarrier2 uboBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uboBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uboBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uboBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        uboBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;

        VkDependencyInfo uboDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        uboDependency.memoryBarrierCount = 1;
        uboDependency.pMemoryBarriers = &uboBarrier;
        vkCmdPipelineBarrier2(cmd, &uboDependency);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

#ifndef NDEBUG
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &debugViewMode);
#endif

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // GetOutputColorImage() visible to a later sampled read (a future preview/blit pass) or a
        // transfer-based blit to the swapchain -- whichever a caller ends up using; both dst
        // stage/access pairs are included since this class does not itself know which.
        VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        outputBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;

        VkDependencyInfo outputDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        outputDependency.memoryBarrierCount = 1;
        outputDependency.pMemoryBarriers = &outputBarrier;
        vkCmdPipelineBarrier2(cmd, &outputDependency);
    }

}
