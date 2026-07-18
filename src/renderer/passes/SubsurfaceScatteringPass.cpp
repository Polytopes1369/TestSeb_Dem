#include "renderer/passes/SubsurfaceScatteringPass.h"

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    void SubsurfaceScatteringPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, VkImageView sceneColorView, VkImageView depthView,
        VkImageView normalView, VkImageView materialIDView, VkBuffer materialParamsBuffer) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        // --- The single horizontal-blur scratch image (rgba16f, same extent as the scene color) ---
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kScratchFormat;
        imageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // STORAGE (pass 0's imageStore output) + SAMPLED (pass 1's texelFetch input). No transfer/
        // color-attachment usage: this image is purely an internal ping-pong buffer, never blitted or
        // drawn onto (unlike GICompositePass's own output image, which this pass writes back into).
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_ScratchImage, &m_ScratchAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_ScratchImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kScratchFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_ScratchView));

        // One-time UNDEFINED -> GENERAL transition (mirrors ATrousDenoisePass::Init's own one-shot
        // pattern) -- GENERAL is valid for both this image's storage-write and sampled-read uses, so
        // it stays GENERAL for its entire lifetime with no further transitions.
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.srcAccessMask = 0;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_ScratchImage;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        m_NearestSampler = VulkanUtils::CreateNearestSampler(m_Device);

        // --- Descriptor set layout: 6 bindings, shared by both per-direction sets (only the scene-
        // color input view and the storage-image output view differ between them). ---
        VkDescriptorSetLayoutBinding bindings[6]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_SceneColor
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_Output
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_Depth
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_Normal
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_MaterialID (r16ui)
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // MaterialParamsSSBO

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 6;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[3] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 2 }, // sceneColor + depth + normal, x2 sets.
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * 2 },          // output + materialID, x2 sets.
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 * 2 }          // material params, x2 sets.
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout setLayouts[2] = { m_SetLayout, m_SetLayout };
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 2;
        setAllocInfo.pSetLayouts = setLayouts;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_Sets));

        // Per-set (input scene color, storage output) views -- see the header's own m_Sets comment:
        // [0] horizontal (scene color -> scratch), [1] vertical (scratch -> scene color, in place).
        VkImageView setSceneColorInputs[2] = { sceneColorView, m_ScratchView };
        VkImageView setStorageOutputs[2] = { m_ScratchView, sceneColorView };

        VkDescriptorBufferInfo materialParamsInfo{ materialParamsBuffer, 0, VK_WHOLE_SIZE };

        for (uint32_t i = 0; i < 2; ++i) {
            VkDescriptorImageInfo sceneColorInfo{ m_NearestSampler, setSceneColorInputs[i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, setStorageOutputs[i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo depthInfo{ m_NearestSampler, depthView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo normalInfo{ m_NearestSampler, normalView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo materialIDInfo{ VK_NULL_HANDLE, materialIDView, VK_IMAGE_LAYOUT_GENERAL };

            VkWriteDescriptorSet writes[6]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneColorInfo, nullptr, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normalInfo, nullptr, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &materialIDInfo, nullptr, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Sets[i], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 6, writes, 0, nullptr);
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SSSPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_SetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/SubsurfaceScattering.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shaderModule);
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[SubsurfaceScatteringPass] Initialized separable screen-space SSS diffusion pass.");
    }

    void SubsurfaceScatteringPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_NearestSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_NearestSampler, nullptr);
            if (m_ScratchView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_ScratchView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_ScratchImage, m_ScratchAllocation);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Sets[0] = VK_NULL_HANDLE;
        m_Sets[1] = VK_NULL_HANDLE;
        m_NearestSampler = VK_NULL_HANDLE;
        m_ScratchImage = VK_NULL_HANDLE;
        m_ScratchAllocation = VK_NULL_HANDLE;
        m_ScratchView = VK_NULL_HANDLE;

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void SubsurfaceScatteringPass::RecordOnePass(VkCommandBuffer cmd, VkDescriptorSet set, const SSSPushConstants& pc) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SSSPushConstants), &pc);
        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
    }

    void SubsurfaceScatteringPass::RecordUpdate(VkCommandBuffer cmd, const maths::mat4& invViewProj,
        const maths::vec3& cameraPositionWorld, float projScaleY, float radiusScale, bool enabled) {
        // Debug-only A/B toggle: when disabled, record nothing at all -- GICompositePass's output is
        // left exactly as it was (no SSS applied, zero GPU cost). Release always passes enabled=true.
        if (!enabled) {
            return;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);

        SSSPushConstants pc{};
        pc.invViewProj = invViewProj;
        pc.cameraPosX = cameraPositionWorld.x;
        pc.cameraPosY = cameraPositionWorld.y;
        pc.cameraPosZ = cameraPositionWorld.z;
        pc.radiusScale = radiusScale;
        pc.viewportWidth = static_cast<float>(m_RenderExtent.width);
        pc.viewportHeight = static_cast<float>(m_RenderExtent.height);
        pc.projScaleY = projScaleY;

        // --- Pass 0: horizontal blur (scene color -> scratch) ---
        pc.direction = 0u;
        RecordOnePass(cmd, m_Sets[0], pc);

        // Barrier between the two separable passes. Covers BOTH hazards at once:
        //  (a) RAW on the scratch image  -- pass 0's STORAGE_WRITE must be visible to pass 1's
        //      SHADER_SAMPLED_READ of the same image.
        //  (b) WAR on the scene-color image -- pass 0 SAMPLED it as input; pass 1 STORAGE_WRITEs it
        //      back in place, so pass 0's read must complete first. A COMPUTE->COMPUTE execution
        //      dependency (which this barrier is) suffices for a WAR (no cache flush needed for it);
        //      the STORAGE_WRITE->SAMPLED_READ memory dependency for (a) rides the same barrier.
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // --- Pass 1: vertical blur (scratch -> scene color, in place) ---
        pc.direction = 1u;
        RecordOnePass(cmd, m_Sets[1], pc);

        // Final barrier: make the now-diffused scene-color image (written by pass 1) visible to every
        // downstream consumer renderer::ClusterRenderPipeline runs after this. Deliberately broad on
        // the destination side -- it must cover BOTH the forward-transparent passes that draw onto this
        // same image as a color attachment (COLOR_ATTACHMENT_OUTPUT, LOAD+blend => read+write) AND
        // TAA's later compute sampled read (COMPUTE_SHADER, SHADER_SAMPLED_READ). A superset dst scope
        // is always safe; it exactly re-establishes the visibility GICompositePass's own trailing
        // barrier used to provide before this pass was inserted.
        {
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
                | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT
                | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }
    }

}
