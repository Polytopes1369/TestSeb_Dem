#include "renderer/passes/PostProcessPass.h"

#include <array>
#include <cmath>
#include <cstring>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        struct ExposureState {
            float currentEV100 = 0.0f;
            float currentAvgLuminance = 0.0f;
            float _pad0 = 0.0f;
            float _pad1 = 0.0f;
        };

        // Standard photographic EV100 formula (Saturation-based speed, ISO 100 calibration):
        // EV100 = log2(N^2 / t * 100 / S), N = f-stop, t = shutter seconds, S = ISO. Matches
        // Unreal Engine's own FMath::Log2(Aperture*Aperture / ShutterSpeed * 100 / ISO).
        float ComputeManualEV100(float aperture, float shutterSpeedSeconds, float isoSensitivity) {
            return std::log2((aperture * aperture) / std::max(shutterSpeedSeconds, 1e-6f)
                * 100.0f / std::max(isoSensitivity, 1.0f));
        }

    } // namespace

    void PostProcessPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D displayExtent, VkImageView hdrColorView, VkImageView bloomView,
        VkImageView depthView, VkImageView refractionOffsetView, VkImageView skyViewLUTView,
        VkImageView volumetricFogView) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_DisplayExtent = displayExtent;

        // --- Output image (RGBA8_UNORM, GENERAL for its whole lifetime -- same convention every
        // other compute-written intermediate in this codebase already follows, e.g.
        // renderer::GICompositePass's own output image) ---
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kOutputFormat;
        imageInfo.extent = { displayExtent.width, displayExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // STORAGE_BIT: this pass' own imageStore. TRANSFER_SRC_BIT: renderer::ClusterRenderPipeline's
        // final blit now sources from this image instead of renderer::TAATSRPass's raw HDR output.
        // COLOR_ATTACHMENT_BIT (Debug only): renderer::debug::DebugTextOverlay draws the stat
        // overlay directly onto this image in the normal (non-debug-visualization) view path now
        // that it is display-resolution RGBA8, matching every other overlay target candidate's own
        // usage flags (see renderer::GICompositePass's own imageInfo.usage comment for why this
        // flag is needed at all).
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
#ifndef NDEBUG
        imageInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
#endif
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

        VulkanUtils::TransitionImageLayoutOneShot(m_Device, commandPool, queue, m_OutputImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        // LINEAR, not NEAREST: Phase PP1's own g_HDRColor reads were all texelFetch (filter-mode-
        // agnostic), but Phase PP2's Chromatic Aberration (fractional per-channel UV offsets) and
        // Bloom (a half-res-and-smaller mip sampled back up) both need real bilinear filtering to
        // look smooth instead of blocky -- see PostProcessComposite.comp's own texture() calls.
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

        // --- GPU-owned buffers ---
        m_HistogramBuffer.Create(allocator, kHistogramBinCount * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ExposureStateBuffer.Create(allocator, sizeof(ExposureState),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ParamsBuffer.Create(allocator, sizeof(PostProcessParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // One-shot clear of the histogram (bins start life referenced by AutoExposureHistogram.comp
        // before AutoExposureAdapt.comp has ever run once to clear them itself) and a sane initial
        // exposure state (the manual EV100 for this pass' own Settings{} defaults -- aperture 4,
        // 1/60s, ISO 100 -- so the very first frame isn't a jarring black-to-correct pop even before
        // Auto Exposure's own histogram has converged).
        ExposureState initialState{};
        initialState.currentEV100 = ComputeManualEV100(4.0f, 1.0f / 60.0f, 100.0f);
        initialState.currentAvgLuminance = 0.18f;

        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, m_HistogramBuffer.Handle(), 0, VK_WHOLE_SIZE, 0);
            vkCmdUpdateBuffer(cmd, m_ExposureStateBuffer.Handle(), 0, sizeof(ExposureState), &initialState);

            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // --- Descriptor pool (shared by all 3 stages' sets) ---
        // STORAGE_BUFFER count of 4: Histogram set uses 1 (HistogramSSBO), Adapt set uses 2
        // (HistogramSSBO + ExposureStateSSBO), Composite set uses 1 (ExposureStateSSBO read-only).
        std::array<VkDescriptorPoolSize, 4> poolSizes{ {
            // Histogram HDR input + Composite HDR input + Composite g_Bloom (PP2) + Composite
            // g_Depth + Composite g_RefractionOffset (both PP3) + Composite g_SkyViewLUT (Atmos
            // Subtask 2) + Composite g_VolumetricFog (Atmos Subtask 3).
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },          // Composite output.
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },         // Composite params.
        } };

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 3;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        // --- Stage 1: AutoExposureHistogram.comp ---
        {
            std::array<VkDescriptorSetLayoutBinding, 2> bindings{ {
                { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_HDRColor
                { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // HistogramSSBO
            } };
            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_HistogramSetLayout));

            VkDescriptorSetAllocateInfo allocSet{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocSet.descriptorPool = m_DescriptorPool;
            allocSet.descriptorSetCount = 1;
            allocSet.pSetLayouts = &m_HistogramSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocSet, &m_HistogramSet));

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HistogramPushConstants) };
            VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            plInfo.setLayoutCount = 1;
            plInfo.pSetLayouts = &m_HistogramSetLayout;
            plInfo.pushConstantRangeCount = 1;
            plInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_HistogramPipelineLayout));

            VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/AutoExposureHistogram.comp.spv");
            m_HistogramPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_HistogramPipelineLayout, shader);
            vkDestroyShaderModule(m_Device, shader, nullptr);
        }

        // --- Stage 2: AutoExposureAdapt.comp ---
        {
            std::array<VkDescriptorSetLayoutBinding, 2> bindings{ {
                { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // HistogramSSBO
                { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // ExposureStateSSBO
            } };
            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_AdaptSetLayout));

            VkDescriptorSetAllocateInfo allocSet{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocSet.descriptorPool = m_DescriptorPool;
            allocSet.descriptorSetCount = 1;
            allocSet.pSetLayouts = &m_AdaptSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocSet, &m_AdaptSet));

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AdaptPushConstants) };
            VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            plInfo.setLayoutCount = 1;
            plInfo.pSetLayouts = &m_AdaptSetLayout;
            plInfo.pushConstantRangeCount = 1;
            plInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_AdaptPipelineLayout));

            VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/AutoExposureAdapt.comp.spv");
            m_AdaptPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_AdaptPipelineLayout, shader);
            vkDestroyShaderModule(m_Device, shader, nullptr);
        }

        // --- Stage 3: PostProcessComposite.comp ---
        {
            std::array<VkDescriptorSetLayoutBinding, 9> bindings{ {
                { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_HDRColor
                { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_Output
                { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // ExposureStateSSBO
                { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // PostProcessParamsUBO
                { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_Bloom (Phase PP2)
                { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_Depth (Phase PP3)
                { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_RefractionOffset (Phase PP3)
                { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_SkyViewLUT (Atmos Subtask 2)
                { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_VolumetricFog (Atmos Subtask 3)
            } };
            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_CompositeSetLayout));

            VkDescriptorSetAllocateInfo allocSet{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocSet.descriptorPool = m_DescriptorPool;
            allocSet.descriptorSetCount = 1;
            allocSet.pSetLayouts = &m_CompositeSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocSet, &m_CompositeSet));

            VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            plInfo.setLayoutCount = 1;
            plInfo.pSetLayouts = &m_CompositeSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_CompositePipelineLayout));

            VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PostProcessComposite.comp.spv");
            m_CompositePipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_CompositePipelineLayout, shader);
            vkDestroyShaderModule(m_Device, shader, nullptr);
        }

        UpdateDescriptorSets(hdrColorView, bloomView);

        // --- Phase PP3: g_Depth / g_RefractionOffset, written ONCE here (unlike hdrColorView/
        // bloomView above) -- both source images keep a fixed identity for this pipeline's entire
        // lifetime (renderer::ClusterResolvePass's depth is never ping-ponged; renderer::
        // TransparentForwardPass's own refraction image is a single owned GpuImage -- see Init()'s
        // own header comment). ---
        {
            VkDescriptorImageInfo depthInfo{ m_LinearSampler, depthView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo refractionInfo{ m_LinearSampler, refractionOffsetView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo skyViewInfo{ m_LinearSampler, skyViewLUTView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo volumetricFogInfo{ m_LinearSampler, volumetricFogView, VK_IMAGE_LAYOUT_GENERAL };
            std::array<VkWriteDescriptorSet, 4> writes{ {
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &refractionInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &skyViewInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &volumetricFogInfo, nullptr, nullptr },
            } };
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        LOG_INFO("[PostProcessPass] Initialized (Phase PP1+PP2+PP3: exposure/color-grading + bloom/CA/vignette + motion-blur/fog/heat-distortion).");
    }

    void PostProcessPass::UpdateDescriptorSets(VkImageView hdrColorView, VkImageView bloomView) {
        VkDescriptorImageInfo hdrInfo{ m_LinearSampler, hdrColorView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo bloomInfo{ m_LinearSampler, bloomView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo histogramInfo{ m_HistogramBuffer.Handle(), 0, m_HistogramBuffer.Size() };
        VkDescriptorBufferInfo exposureInfo{ m_ExposureStateBuffer.Handle(), 0, m_ExposureStateBuffer.Size() };
        VkDescriptorImageInfo outputInfo{ VK_NULL_HANDLE, m_OutputView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo paramsInfo{ m_ParamsBuffer.Handle(), 0, m_ParamsBuffer.Size() };

        std::array<VkWriteDescriptorSet, 2> histogramWrites{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_HistogramSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hdrInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_HistogramSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &histogramInfo, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(histogramWrites.size()), histogramWrites.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 2> adaptWrites{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_AdaptSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &histogramInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_AdaptSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &exposureInfo, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(adaptWrites.size()), adaptWrites.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 5> compositeWrites{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hdrInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &exposureInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &paramsInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompositeSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &bloomInfo, nullptr, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(compositeWrites.size()), compositeWrites.data(), 0, nullptr);
    }

    void PostProcessPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_HistogramPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_HistogramPipeline, nullptr);
            if (m_HistogramPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_HistogramPipelineLayout, nullptr);
            if (m_HistogramSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_HistogramSetLayout, nullptr);

            if (m_AdaptPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_AdaptPipeline, nullptr);
            if (m_AdaptPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_AdaptPipelineLayout, nullptr);
            if (m_AdaptSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_AdaptSetLayout, nullptr);

            if (m_CompositePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_CompositePipeline, nullptr);
            if (m_CompositePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_CompositePipelineLayout, nullptr);
            if (m_CompositeSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_CompositeSetLayout, nullptr);

            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_LinearSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_LinearSampler, nullptr);
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE && m_OutputImage != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
        }

        m_HistogramBuffer.Destroy();
        m_ExposureStateBuffer.Destroy();
        m_ParamsBuffer.Destroy();

        m_HistogramPipeline = VK_NULL_HANDLE; m_HistogramPipelineLayout = VK_NULL_HANDLE; m_HistogramSetLayout = VK_NULL_HANDLE; m_HistogramSet = VK_NULL_HANDLE;
        m_AdaptPipeline = VK_NULL_HANDLE; m_AdaptPipelineLayout = VK_NULL_HANDLE; m_AdaptSetLayout = VK_NULL_HANDLE; m_AdaptSet = VK_NULL_HANDLE;
        m_CompositePipeline = VK_NULL_HANDLE; m_CompositePipelineLayout = VK_NULL_HANDLE; m_CompositeSetLayout = VK_NULL_HANDLE; m_CompositeSet = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_LinearSampler = VK_NULL_HANDLE;
        m_OutputView = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE;
        m_OutputAllocation = VK_NULL_HANDLE;

        m_DisplayExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void PostProcessPass::RecordComposite(VkCommandBuffer cmd, float deltaTimeSeconds, const Settings& settings,
        const maths::mat4& invViewProj, const maths::mat4& prevViewProj, const maths::vec3& cameraPositionWorld,
        const maths::mat4& viewProj, const maths::vec3& sunDirection, const maths::vec3& cameraForward,
        float fovYRadians, float aspectRatio, uint32_t frameIndex) {
        // --- Upload this frame's params UBO ---
        PostProcessParamsUBO params{};
        params.aperture = settings.aperture;
        params.shutterSpeed = settings.shutterSpeedSeconds;
        params.isoSensitivity = settings.isoSensitivity;
        params.useAutoExposure = settings.useAutoExposure ? 1u : 0u;
        params.exposureCompensationEV = settings.exposureCompensationEV;
        params.adaptationSpeedUp = settings.adaptationSpeedUpEVPerSec;
        params.adaptationSpeedDown = settings.adaptationSpeedDownEVPerSec;
        params.whiteBalanceTempKelvin = settings.whiteBalanceTempKelvin;
        params.whiteBalanceTint = settings.whiteBalanceTint;
        params.lift[0] = settings.liftR; params.lift[1] = settings.liftG; params.lift[2] = settings.liftB;
        params.gamma[0] = settings.gammaR; params.gamma[1] = settings.gammaG; params.gamma[2] = settings.gammaB;
        params.gain[0] = settings.gainR; params.gain[1] = settings.gainG; params.gain[2] = settings.gainB;
        params.saturation = settings.saturation;
        params.contrast = settings.contrast;
        params.displayGamma = settings.displayGamma;
        params.bloomIntensity = settings.bloomIntensity;
        params.chromaticAberrationIntensity = settings.chromaticAberrationIntensity;
        params.vignetteIntensity = settings.vignetteIntensity;
        params.vignetteSmoothness = settings.vignetteSmoothness;
        params.vignetteColorBleed = settings.vignetteColorBleed;

        params.invViewProj = invViewProj;
        params.prevViewProj = prevViewProj;
        params.cameraPosX = cameraPositionWorld.x;
        params.cameraPosY = cameraPositionWorld.y;
        params.cameraPosZ = cameraPositionWorld.z;
        params.heatDistortionIntensity = settings.heatDistortionIntensity;
        params.motionBlurIntensity = settings.motionBlurIntensity;
        params.motionBlurMaxVelocityUV = settings.motionBlurMaxVelocityUV;
        params.fogColor[0] = settings.fogColorR; params.fogColor[1] = settings.fogColorG; params.fogColor[2] = settings.fogColorB;
        params.fogDensity = settings.fogDensity;
        params.fogHeightFalloff = settings.fogHeightFalloff;
        params.fogHeightOffset = settings.fogHeightOffset;
        params.fogStartDistance = settings.fogStartDistance;
        params.fogMaxOpacity = settings.fogMaxOpacity;

        // Phase PP4: God Rays -- project a point far along -sunDirection (the sun itself, since
        // sunDirection points FROM the light TOWARD the scene) through this frame's `viewProj` to
        // find its screen UV; `sunScreenValid` is 0 whenever that point sits behind the camera
        // (clip.w <= 0) -- an off-screen-but-in-front sun is still valid (the classic technique's
        // radial streaks read correctly even with the sun origin outside the viewport). maths::mat4
        // has no vec4 type/operator* to lean on (see maths::mat4's own column-major `m` array
        // layout, matched by ClusterRenderPipeline.cpp's own manual unprojections nowhere -- this is
        // the first CPU-side forward projection in this codebase), so this multiplies manually.
        float sunWorldX = cameraPositionWorld.x - sunDirection.x * 10000.0f;
        float sunWorldY = cameraPositionWorld.y - sunDirection.y * 10000.0f;
        float sunWorldZ = cameraPositionWorld.z - sunDirection.z * 10000.0f;
        const auto& vp = viewProj.m;
        float clipX = vp[0] * sunWorldX + vp[4] * sunWorldY + vp[8] * sunWorldZ + vp[12];
        float clipY = vp[1] * sunWorldX + vp[5] * sunWorldY + vp[9] * sunWorldZ + vp[13];
        float clipW = vp[3] * sunWorldX + vp[7] * sunWorldY + vp[11] * sunWorldZ + vp[15];
        if (clipW > 0.0001f) {
            params.sunScreenU = (clipX / clipW) * 0.5f + 0.5f;
            params.sunScreenV = (clipY / clipW) * 0.5f + 0.5f;
            params.sunScreenValid = 1.0f;
        } else {
            params.sunScreenValid = 0.0f;
        }
        params.godRaysIntensity = settings.godRaysIntensity;
        params.godRaysDecay = settings.godRaysDecay;
        params.godRaysDensity = settings.godRaysDensity;
        params.godRaysWeight = settings.godRaysWeight;

        // Atmos weather system, Subtask 2: raw sun direction (unlike sunScreenU/V above, NOT
        // projected -- PostProcessComposite.comp's own SkyViewLUTUVFromDirection() needs the real
        // 3D direction to reproduce AtmosSkyLUTs.comp's own sun-relative azimuth mapping).
        params.sunDirWorldX = sunDirection.x;
        params.sunDirWorldY = sunDirection.y;
        params.sunDirWorldZ = sunDirection.z;

        // Atmos weather system, Subtask 3.
        params.cameraForwardX = cameraForward.x;
        params.cameraForwardY = cameraForward.y;
        params.cameraForwardZ = cameraForward.z;

        // Phase PP5: Panini Projection -- half-FOV tangents drive the tangent-space UV remap (see
        // PostProcessComposite.comp's own ApplyPaniniProjection comment). Matches maths::mat4::
        // PerspectiveVulkan's own g=1/tan(fovY/2), m[0]=g/aspect relationship: tan(halfFovY) =
        // 1/g, tan(halfFovX) = aspect * tan(halfFovY).
        float halfFovTanY = std::tan(fovYRadians * 0.5f);
        params.paniniD = settings.paniniD;
        params.paniniS = settings.paniniS;
        params.halfFovTanX = halfFovTanY * aspectRatio;
        params.halfFovTanY = halfFovTanY;

        params.sharpenIntensity = settings.sharpenIntensity;
        params.sharpenRadiusPixels = settings.sharpenRadiusPixels;
        params.filmGrainIntensity = settings.filmGrainIntensity;
        params.filmGrainResponseMidpoint = settings.filmGrainResponseMidpoint;
        params.frameIndex = frameIndex;

        vkCmdUpdateBuffer(cmd, m_ParamsBuffer.Handle(), 0, sizeof(PostProcessParamsUBO), &params);

        VkMemoryBarrier2 uploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        uploadBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo uploadDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        uploadDep.memoryBarrierCount = 1;
        uploadDep.pMemoryBarriers = &uploadBarrier;
        vkCmdPipelineBarrier2(cmd, &uploadDep);

        // --- Stage 1: histogram build ---
        HistogramPushConstants histPC{};
        histPC.imageWidth = static_cast<float>(m_DisplayExtent.width);
        histPC.imageHeight = static_cast<float>(m_DisplayExtent.height);
        histPC.minLogLuminance = kMinLogLuminance;
        histPC.logLuminanceRange = kLogLuminanceRange;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_HistogramPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_HistogramPipelineLayout, 0, 1, &m_HistogramSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_HistogramPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HistogramPushConstants), &histPC);
        uint32_t groupsX = (m_DisplayExtent.width + 15u) / 16u;
        uint32_t groupsY = (m_DisplayExtent.height + 15u) / 16u;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        VkMemoryBarrier2 histToAdaptBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        histToAdaptBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        histToAdaptBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        histToAdaptBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        histToAdaptBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo histToAdaptDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        histToAdaptDep.memoryBarrierCount = 1;
        histToAdaptDep.pMemoryBarriers = &histToAdaptBarrier;
        vkCmdPipelineBarrier2(cmd, &histToAdaptDep);

        // --- Stage 2: reduce + adapt ---
        AdaptPushConstants adaptPC{};
        adaptPC.minLogLuminance = kMinLogLuminance;
        adaptPC.logLuminanceRange = kLogLuminanceRange;
        adaptPC.pixelCount = static_cast<float>(m_DisplayExtent.width) * static_cast<float>(m_DisplayExtent.height);
        adaptPC.deltaTimeSeconds = deltaTimeSeconds;
        adaptPC.adaptationSpeedUp = settings.adaptationSpeedUpEVPerSec;
        adaptPC.adaptationSpeedDown = settings.adaptationSpeedDownEVPerSec;
        adaptPC.exposureCompensationEV = settings.exposureCompensationEV;
        adaptPC.manualEV100 = ComputeManualEV100(settings.aperture, settings.shutterSpeedSeconds, settings.isoSensitivity);
        adaptPC.useAutoExposure = settings.useAutoExposure ? 1u : 0u;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_AdaptPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_AdaptPipelineLayout, 0, 1, &m_AdaptSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_AdaptPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AdaptPushConstants), &adaptPC);
        vkCmdDispatch(cmd, 1, 1, 1);

        VkMemoryBarrier2 adaptToCompositeBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        adaptToCompositeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        adaptToCompositeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        adaptToCompositeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        adaptToCompositeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo adaptToCompositeDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        adaptToCompositeDep.memoryBarrierCount = 1;
        adaptToCompositeDep.pMemoryBarriers = &adaptToCompositeBarrier;
        vkCmdPipelineBarrier2(cmd, &adaptToCompositeDep);

        // --- Stage 3: final composite ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompositePipelineLayout, 0, 1, &m_CompositeSet, 0, nullptr);
        uint32_t compGroupsX = (m_DisplayExtent.width + 7u) / 8u;
        uint32_t compGroupsY = (m_DisplayExtent.height + 7u) / 8u;
        vkCmdDispatch(cmd, compGroupsX, compGroupsY, 1);

        VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        outputBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        VkDependencyInfo outputDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        outputDep.memoryBarrierCount = 1;
        outputDep.pMemoryBarriers = &outputBarrier;
        vkCmdPipelineBarrier2(cmd, &outputDep);
    }

}
