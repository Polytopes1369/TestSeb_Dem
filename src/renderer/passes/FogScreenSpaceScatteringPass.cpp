#include "renderer/passes/FogScreenSpaceScatteringPass.h"

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        void CreateOwnedImage(VmaAllocator allocator, VkDevice device, VkExtent2D extent, VkFormat format,
            VkImage& outImage, VmaAllocation& outAllocation, VkImageView& outView) {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = { extent.width, extent.height, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            // STORAGE (this pass' own imageStore output) + SAMPLED (mode 1 texelFetches the scratch
            // image; renderer::PostProcessPass samples the final output image).
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage, &outAllocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = outImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outView));
        }

    } // namespace

    void FogScreenSpaceScatteringPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D displayExtent, VkImageView depthView, VkImageView volumetricFogView, VkSampler fogSampler) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_DisplayExtent = displayExtent;

        CreateOwnedImage(allocator, device, displayExtent, kFormat, m_ScratchImage, m_ScratchAllocation, m_ScratchView);
        CreateOwnedImage(allocator, device, displayExtent, kFormat, m_OutputImage, m_OutputAllocation, m_OutputView);

        // One-time UNDEFINED -> GENERAL transition for BOTH owned images (mirrors renderer::
        // SubsurfaceScatteringPass::Init's own one-shot pattern) -- GENERAL is valid for both this
        // pass' own storage-write and sampled-read uses, so neither image ever transitions again.
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 barriers[2]{};
            VkImage images[2] = { m_ScratchImage, m_OutputImage };
            for (uint32_t i = 0; i < 2; ++i) {
                barriers[i] = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                barriers[i].srcAccessMask = 0;
                barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].image = images[i];
                barriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 2;
            depInfo.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // LINEAR, not NEAREST: g_Depth needs the same bilinear render-res -> display-res upsample
        // renderer::PostProcessPass's own m_LinearSampler already provides for its identically-bound
        // depth view, and g_VolumetricFog needs real trilinear filtering across the froxel grid's own
        // coarse 160x90x64 resolution (matches renderer::AtmosVolumetricFogPass's own m_FogSampler --
        // this pass creates its OWN sampler rather than borrowing that one purely so this class stays
        // self-contained/independently shutdown-able, exactly like every other leaf compute pass in
        // this codebase owns its own sampler(s) rather than sharing).
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_LinearSampler));

        // --- Descriptor set layout: 4 bindings, shared by both per-mode sets (only the g_ScatterInput
        // and g_Output views differ between them -- see the header's own m_Sets comment). ---
        VkDescriptorSetLayoutBinding bindings[4]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_Depth
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_VolumetricFog
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_ScatterInput
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_Output

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 4;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[2] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 2 }, // depth + fog3D + scatterInput, x2 sets.
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 * 2 }           // output, x2 sets.
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout setLayouts[2] = { m_SetLayout, m_SetLayout };
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 2;
        setAllocInfo.pSetLayouts = setLayouts;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_Sets));

        // Per-set (scatter input, storage output) views: [0] mode 0 (g_ScatterInput unused by the
        // shader but still bound to a valid view -- the scratch image itself, matching this
        // codebase's "declared but unused" convention rather than leaving a descriptor slot
        // unwritten), writes m_ScratchImage. [1] mode 1 reads m_ScratchImage as g_ScatterInput,
        // writes m_OutputImage.
        VkImageView setScatterInputs[2] = { m_ScratchView, m_ScratchView };
        VkImageView setStorageOutputs[2] = { m_ScratchView, m_OutputView };

        for (uint32_t i = 0; i < 2; ++i) {
            VkDescriptorImageInfo depthInfo{ m_LinearSampler, depthView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo fog3DInfo{ fogSampler, volumetricFogView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo scatterInputInfo{ m_LinearSampler, setScatterInputs[i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, setStorageOutputs[i], VK_IMAGE_LAYOUT_GENERAL };

            VkWriteDescriptorSet writes[4]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &fog3DInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &scatterInputInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(FogScatterPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_SetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/FogScreenSpaceScattering.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shaderModule);
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[FogScreenSpaceScatteringPass] Initialized separable screen-space fog scattering pass.");
    }

    void FogScreenSpaceScatteringPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_LinearSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_LinearSampler, nullptr);
            if (m_ScratchView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_ScratchView, nullptr);
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_ScratchImage, m_ScratchAllocation);
            vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Sets[0] = VK_NULL_HANDLE;
        m_Sets[1] = VK_NULL_HANDLE;
        m_LinearSampler = VK_NULL_HANDLE;
        m_ScratchImage = VK_NULL_HANDLE;
        m_ScratchAllocation = VK_NULL_HANDLE;
        m_ScratchView = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE;
        m_OutputAllocation = VK_NULL_HANDLE;
        m_OutputView = VK_NULL_HANDLE;

        m_DisplayExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void FogScreenSpaceScatteringPass::RecordOnePass(VkCommandBuffer cmd, VkDescriptorSet set, const FogScatterPushConstants& pc) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FogScatterPushConstants), &pc);
        uint32_t groupCountX = (m_DisplayExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        uint32_t groupCountY = (m_DisplayExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
    }

    void FogScreenSpaceScatteringPass::RecordScatter(VkCommandBuffer cmd, const maths::mat4& invViewProj,
        const maths::vec3& cameraPositionWorld, const maths::vec3& cameraForward, float blurRadiusPixels) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);

        FogScatterPushConstants pc{};
        pc.invViewProj = invViewProj;
        pc.cameraPosX = cameraPositionWorld.x;
        pc.cameraPosY = cameraPositionWorld.y;
        pc.cameraPosZ = cameraPositionWorld.z;
        pc.blurRadiusPixels = blurRadiusPixels;
        pc.cameraForwardX = cameraForward.x;
        pc.cameraForwardY = cameraForward.y;
        pc.cameraForwardZ = cameraForward.z;
        pc.viewportWidth = static_cast<float>(m_DisplayExtent.width);
        pc.viewportHeight = static_cast<float>(m_DisplayExtent.height);

        // --- Mode 0: extract (3D froxel lookup) + horizontal blur (-> m_ScratchImage) ---
        pc.mode = 0u;
        RecordOnePass(cmd, m_Sets[0], pc);

        // Barrier between the two separable passes -- mode 0's STORAGE_WRITE of m_ScratchImage must
        // be visible to mode 1's SHADER_SAMPLED_READ (texelFetch) of that same image. Same convention
        // as renderer::SubsurfaceScatteringPass::RecordUpdate's own inter-pass barrier.
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // --- Mode 1: vertical blur (m_ScratchImage -> m_OutputImage) ---
        pc.mode = 1u;
        RecordOnePass(cmd, m_Sets[1], pc);

        // Final barrier: makes GetOutputView() (m_OutputImage, written by mode 1) visible to
        // renderer::PostProcessPass::RecordComposite's own COMPUTE_SHADER sampled read right after.
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
    }

}
