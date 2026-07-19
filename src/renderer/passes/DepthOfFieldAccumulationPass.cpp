#include "renderer/passes/DepthOfFieldAccumulationPass.h"

#include <array>
#include <cstring>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {
        constexpr uint32_t kWorkgroupSize = 8;
    }

    void DepthOfFieldAccumulationPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D displayExtent, VkImageView hdrColorView, VkImageView depthView) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_DisplayExtent = displayExtent;

        // 1. Ping-pong history images (display resolution). RGB = accumulated color, A = accumulated
        //    sample weight -- see the header comment for why both live in one image.
        for (uint32_t i = 0; i < 2; ++i) {
            VulkanUtils::CreateStorageSampledImage2D(allocator, device, kHistoryFormat, displayExtent,
                m_HistoryImages[i], m_HistoryAllocations[i], m_HistoryImageViews[i]);
        }

        // One-time UNDEFINED -> GENERAL transition + neutral (RGB=0, A=0 -- "zero accumulated
        // samples") clear for both slots. Without this, a caller reading GetOutputView() before this
        // pass has ever written to that particular slot (the very first frame -- see
        // GetOutputView()'s own header comment on the one-call read/write timing) would sample
        // genuinely undefined GPU memory instead of a well-defined "no history yet" value.
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkClearColorValue zeroClear{};
            zeroClear.float32[0] = 0.0f; zeroClear.float32[1] = 0.0f; zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 0.0f;
            for (uint32_t i = 0; i < 2; ++i) {
                VulkanUtils::ClearComputeImageToGeneral(cmd, m_HistoryImages[i], zeroClear);
            }
        });

        // 2. Sampler (linear, matches DepthOfFieldPass's own g_HDRColor/g_Depth sampler convention).
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

        // 3. UBO -- mapped/host-visible, single-frame-in-flight (same convention TAATSRPass's own
        //    m_ViewParamsBuffer uses: a plain memcpy every frame is safe, no transfer barrier needed).
        m_ParamsBuffer.Create(allocator, sizeof(ParamsUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, true);

        // 4. Descriptor set layout: 0=HDRColor (rewritten every frame via UpdateSourceDescriptor),
        //    1=Depth (fixed), 2=HistoryInput / 3=HistoryOutput (ping-ponged per set index -- same
        //    convention as TAATSRPass::UpdateDescriptorSets), 4=UBO (shared, not ping-ponged).
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{ {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_HDRColor
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_Depth
            { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_HistoryInput
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_HistoryOutput
            { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // DepthOfFieldAccumulationParamsUBO
        } };
        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        std::array<VkDescriptorPoolSize, 3> poolSizes{ {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 2 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 * 2 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 },
        } };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout setLayouts[2] = { m_SetLayout, m_SetLayout };
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 2;
        setAllocInfo.pSetLayouts = setLayouts;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_DescriptorSets));

        // g_Depth (binding 1) and the UBO (binding 4) are fixed/shared across both sets -- written
        // once here. g_HistoryInput/g_HistoryOutput (bindings 2/3) ping-pong per set index: set i
        // writes slot i, reads slot 1-i (same convention as TAATSRPass::UpdateDescriptorSets).
        // g_HDRColor (binding 0) is bound by UpdateSourceDescriptor() below instead (TAATSRPass's own
        // output view identity changes every frame).
        for (uint32_t i = 0; i < 2; ++i) {
            VkDescriptorImageInfo depthInfo{ m_LinearSampler, depthView, VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo historyInputInfo{ m_LinearSampler, m_HistoryImageViews[1 - i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo historyOutputInfo{ VK_NULL_HANDLE, m_HistoryImageViews[i], VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo paramsInfo{ m_ParamsBuffer.Handle(), 0, m_ParamsBuffer.Size() };

            std::array<VkWriteDescriptorSet, 4> writes{ {
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &depthInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &historyInputInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &historyOutputInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &paramsInfo, nullptr },
            } };
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/DepthOfFieldAccumulation.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shader);
        vkDestroyShaderModule(m_Device, shader, nullptr);

        UpdateSourceDescriptor(hdrColorView);

        m_FramesSinceReset = 0;

        LOG_INFO("[DepthOfFieldAccumulationPass] Initialized (UE5.8-parity Accumulation DOF: per-frame jittered lens sample + temporal reprojection).");
    }

    void DepthOfFieldAccumulationPass::UpdateSourceDescriptor(VkImageView hdrColorView) {
        // Both descriptor sets need binding 0 rewritten -- either one could be the active WRITE
        // target (bound via vkCmdBindDescriptorSets in RecordGenerate) on any given frame.
        for (uint32_t i = 0; i < 2; ++i) {
            VkDescriptorImageInfo sourceInfo{ m_LinearSampler, hdrColorView, VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSets[i], 0, 0, 1,
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sourceInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
        }
    }

    void DepthOfFieldAccumulationPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_LinearSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_LinearSampler, nullptr);
            for (uint32_t i = 0; i < 2; ++i) {
                if (m_HistoryImageViews[i] != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_HistoryImageViews[i], nullptr);
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            for (uint32_t i = 0; i < 2; ++i) {
                if (m_HistoryImages[i] != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_HistoryImages[i], m_HistoryAllocations[i]);
            }
        }

        m_ParamsBuffer.Destroy();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_DescriptorSets[0] = VK_NULL_HANDLE;
        m_DescriptorSets[1] = VK_NULL_HANDLE;
        m_LinearSampler = VK_NULL_HANDLE;
        for (uint32_t i = 0; i < 2; ++i) {
            m_HistoryImageViews[i] = VK_NULL_HANDLE;
            m_HistoryImages[i] = VK_NULL_HANDLE;
            m_HistoryAllocations[i] = VK_NULL_HANDLE;
        }
        m_CurrentHistoryIndex = 0;
        m_FramesSinceReset = 0;

        m_DisplayExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void DepthOfFieldAccumulationPass::RecordGenerate(VkCommandBuffer cmd, const maths::mat4& invViewProj, const maths::mat4& prevViewProj,
        const maths::vec3& cameraPositionWorld, float aperture, uint32_t frameIndex, bool resetHistory,
        const Settings& settings) {
        ParamsUBO params{};
        params.invViewProj = invViewProj;
        params.prevViewProj = prevViewProj;
        params.cameraPosX = cameraPositionWorld.x;
        params.cameraPosY = cameraPositionWorld.y;
        params.cameraPosZ = cameraPositionWorld.z;
        params.aperture = aperture;
        params.focalLengthMM = settings.focalLengthMM;
        params.focusDistanceWorldUnits = settings.focusDistanceWorldUnits;
        params.maxCoCRadiusPixels = settings.maxCoCRadiusPixels;
        params.frameIndex = frameIndex;
        params.resetHistory = resetHistory ? 1u : 0u;
        params.maxAccumulationSamples = settings.maxAccumulationSamples;
        // Mapped/host-visible UBO, single-frame-in-flight (see Init's own comment) -- a direct memcpy
        // is safe, no vkCmdUpdateBuffer + transfer barrier needed (unlike DepthOfFieldPass's own
        // GPU_ONLY ParamsBuffer).
        std::memcpy(m_ParamsBuffer.MappedData(), &params, sizeof(ParamsUBO));

        m_FramesSinceReset = resetHistory ? 0u : (m_FramesSinceReset + 1u);

        // Pre-dispatch barrier: last frame's history-input slot (written by last frame's own
        // dispatch) must be visible to this frame's sampled read, and this frame's write-target slot
        // must be safe to storage-write. Both history images stay in VK_IMAGE_LAYOUT_GENERAL for
        // their entire lifetime (set once in Init), so a plain global memory barrier is sufficient --
        // no layout transition needed here, exactly like TAATSRPass::RecordPass' own equivalent
        // pre-dispatch barrier.
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

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        // m_CurrentHistoryIndex is the slot WRITTEN this dispatch (its matching descriptor set reads
        // the OTHER slot as history input -- see Init's own binding-write loop comment).
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSets[m_CurrentHistoryIndex], 0, nullptr);

        uint32_t groupsX = (m_DisplayExtent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupsY = (m_DisplayExtent.height + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        // Post-dispatch barrier: this frame's freshly written history slot must be visible to
        // whatever samples GetOutputView() next this frame (BloomPass/PostProcessPass, both a
        // COMPUTE_SHADER sampled read).
        {
            VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            outputBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo outputDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            outputDep.memoryBarrierCount = 1;
            outputDep.pMemoryBarriers = &outputBarrier;
            vkCmdPipelineBarrier2(cmd, &outputDep);
        }

        // Ping-pong swap for next frame -- same convention/timing as TAATSRPass::RecordPass' own
        // step 6 (see GetOutputView()'s own header comment for the resulting read/write call timing).
        m_CurrentHistoryIndex = 1 - m_CurrentHistoryIndex;
    }

}
