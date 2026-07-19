#include "renderer/passes/GICompositePass.h"

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"
#include "renderer/passes/WorldProbeGridPass.h"

namespace renderer {

    namespace {

#ifndef NDEBUG
        // Byte-for-byte mirror of GICompositeViewParamsUBO in GIComposite.comp (std140, debug-only).
        struct GICompositeViewParams {
            maths::mat4 invViewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f;
            float _pad0 = 0.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(GICompositeViewParams) == 96,
            "GICompositeViewParams must match GIComposite.comp's GICompositeViewParamsUBO exactly (std140 layout, debug-only)");
#endif

    } // namespace

    void GICompositePass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, VkImageView directColorView, VkImageView denoisedGIView,
        VkImageView aoView, VkImageView depthView, const WorldProbeGridPass& worldProbes) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kOutputFormat;
        imageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // COLOR_ATTACHMENT_BIT: unconditional, NOT Debug-only -- renderer::TransparentForwardPass::
        // RecordDraw() draws directly onto this image via vkCmdBeginRendering in BOTH Debug and
        // Release (renderer::ClusterRenderPipeline::RecordFrame's own transparent-forward call site
        // has no `#ifndef NDEBUG` guard around it). The comment this replaced only accounted for
        // renderer::debug::DebugTextOverlay::RecordDraw's own Debug-only use of this same flag,
        // which is real but not the only one -- gating the flag behind NDEBUG left Release builds
        // creating this image without the usage its own always-on transparent draw needs.
        // SAMPLED_BIT: unconditional -- renderer::TAATSRPass::UpdateDescriptorSets() binds this
        // image as a VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER (its own `g_LowResColor` binding 0),
        // also unconditional. Missing this produced VUID-VkWriteDescriptorSet-descriptorType-00337
        // every frame.
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_OutputImage, &m_OutputAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_OutputImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kOutputFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputView));

        // One-time UNDEFINED -> GENERAL transition (mirrors ClusterResolvePass::Init's own
        // one-shot pattern) -- GENERAL is both a valid storage-image layout (this pass' own
        // imageStore) and a valid transfer-source layout (renderer::ClusterRenderPipeline's final
        // blit), so it stays GENERAL for this image's entire lifetime with no further transitions.
        VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_OutputImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

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
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_NearestSampler));

#ifndef NDEBUG
        constexpr uint32_t kBindingCount = 8;
        m_ViewParamsBuffer.Create(allocator, sizeof(GICompositeViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_WorldProbeGridParamsBuffer.Create(allocator, sizeof(WorldProbeGridParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
#else
        constexpr uint32_t kBindingCount = 4;
        (void)depthView;
        (void)worldProbes;
#endif

        std::vector<VkDescriptorSetLayoutBinding> bindings(kBindingCount);
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_Output
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_DirectColor
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_DenoisedGI
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_AO (Phase PP4, always-on)
#ifndef NDEBUG
        bindings[4] = { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_Depth
        bindings[5] = { 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // GICompositeViewParamsUBO
        bindings[6] = { 9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_WorldProbeGrid
        bindings[7] = { 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };        // WorldProbeGridParamsUBO
#endif

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = kBindingCount;
        setLayoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

#ifndef NDEBUG
        VkDescriptorPoolSize poolSizes[3] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 }
        };
        uint32_t poolSizeCount = 3;
#else
        VkDescriptorPoolSize poolSizes[2] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 }
        };
        uint32_t poolSizeCount = 2;
#endif
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = poolSizeCount;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_Set));

        VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, m_OutputView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo directColorInfo{ m_NearestSampler, directColorView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo denoisedGIInfo{ m_NearestSampler, denoisedGIView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo aoInfo{ m_NearestSampler, aoView, VK_IMAGE_LAYOUT_GENERAL };

        std::vector<VkWriteDescriptorSet> writes(kBindingCount);
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &directColorInfo, nullptr, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &denoisedGIInfo, nullptr, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &aoInfo, nullptr, nullptr };

#ifndef NDEBUG
        // GENERAL, not DEPTH_STENCIL_READ_ONLY_OPTIMAL: `depthView` is renderer::
        // ClusterResolvePass::GetOutputDepthView(), a plain COLOR-aspect R32_SFLOAT GBuffer image
        // (the winning hw-vs-sw arbitrated NDC depth, not a real depth-attachment image), kept in
        // VK_IMAGE_LAYOUT_GENERAL for its entire lifetime -- same convention every other consumer
        // of this exact image already follows (renderer::ATrousDenoisePass, renderer::
        // ReflectionPass, renderer::ScreenTracePass, renderer::MegaLightsPass). Binding it with a
        // real depth-attachment layout here mismatched the image's actual COLOR aspectMask and
        // produced a VUID-VkDescriptorImageInfo-imageLayout-09426 validation error (plus a cascade
        // of downstream errors once the shader actually sampled it) on every Debug run.
        VkDescriptorImageInfo depthInfo{ m_NearestSampler, depthView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorImageInfo worldProbeGridInfo{ worldProbes.GetGridSampler(), worldProbes.GetGridView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo worldProbeGridParamsInfo{ m_WorldProbeGridParamsBuffer.Handle(), 0, m_WorldProbeGridParamsBuffer.Size() };

        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 8, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 9, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &worldProbeGridInfo, nullptr, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &worldProbeGridParamsInfo, nullptr };
#endif
        vkUpdateDescriptorSets(m_Device, kBindingCount, writes.data(), 0, nullptr);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_SetLayout;
#ifndef NDEBUG
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t); // viewMode
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
#else
        layoutInfo.pushConstantRangeCount = 0;
#endif
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/GIComposite.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[GICompositePass] Initialized final GI composite.");
    }

    void GICompositePass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_NearestSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_NearestSampler, nullptr);
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Set = VK_NULL_HANDLE;
        m_NearestSampler = VK_NULL_HANDLE;
        m_OutputView = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE;
        m_OutputAllocation = VK_NULL_HANDLE;

#ifndef NDEBUG
        m_ViewParamsBuffer.Destroy();
        m_WorldProbeGridParamsBuffer.Destroy();
#endif

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void GICompositePass::RecordComposite(VkCommandBuffer cmd
#ifndef NDEBUG
        , const CameraPushConstants& camera, const maths::vec3& cameraPositionWorld,
        const maths::vec3& worldProbeGridOrigin
#endif
    ) {
#ifndef NDEBUG
        GICompositeViewParams viewParams{};
        viewParams.invViewProj = (camera.proj * camera.view).Inverse();
        viewParams.cameraPosX = cameraPositionWorld.x;
        viewParams.cameraPosY = cameraPositionWorld.y;
        viewParams.cameraPosZ = cameraPositionWorld.z;
        viewParams.viewportWidth = static_cast<float>(m_RenderExtent.width);
        viewParams.viewportHeight = static_cast<float>(m_RenderExtent.height);
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(GICompositeViewParams), &viewParams);

        WorldProbeGridParams gridParams{};
        gridParams.gridOriginX = worldProbeGridOrigin.x;
        gridParams.gridOriginY = worldProbeGridOrigin.y;
        gridParams.gridOriginZ = worldProbeGridOrigin.z;
        gridParams.probeSpacing = WorldProbeGridPass::kProbeSpacing;
        gridParams.gridResolution = static_cast<float>(WorldProbeGridPass::kGridResolution);
        vkCmdUpdateBuffer(cmd, m_WorldProbeGridParamsBuffer.Handle(), 0, sizeof(WorldProbeGridParams), &gridParams);

        VkMemoryBarrier2 uboBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uboBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uboBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uboBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        uboBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;

        VkDependencyInfo uboDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        uboDependency.memoryBarrierCount = 1;
        uboDependency.pMemoryBarriers = &uboBarrier;
        vkCmdPipelineBarrier2(cmd, &uboDependency);
#endif

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

#ifndef NDEBUG
        uint32_t viewMode = camera.debugViewMode;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &viewMode);
#endif

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        outputBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

        VkDependencyInfo outputDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        outputDependency.memoryBarrierCount = 1;
        outputDependency.pMemoryBarriers = &outputBarrier;
        vkCmdPipelineBarrier2(cmd, &outputDependency);
    }

}
