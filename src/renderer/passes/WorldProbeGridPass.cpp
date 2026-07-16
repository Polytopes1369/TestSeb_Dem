#include "renderer/passes/WorldProbeGridPass.h"

#include <cmath>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte layout match for WorldProbeInjectPushConstants in WorldProbeInject.comp.
        struct WorldProbeInjectPushConstants {
            float gridOriginX = 0.0f, gridOriginY = 0.0f, gridOriginZ = 0.0f;
            float probeSpacing = 0.0f;
            uint32_t entityCount = 0;
            uint32_t frameIndex = 0;
            uint32_t traceMode = 0;
        };
        static_assert(sizeof(WorldProbeInjectPushConstants) == 28,
            "WorldProbeInjectPushConstants must match WorldProbeInject.comp's push_constant block exactly");

    } // namespace

    bool WorldProbeGridPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
        const SurfaceCacheRayTracingPass& rtPass) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- the grid image itself: kGridResolution^3, rgba16f, storage + sampled (the
        // former for this pass' own imageStore, the latter for world_probe_sampling.glsl's
        // consumers), kept VK_IMAGE_LAYOUT_GENERAL for its entire lifetime.
        // =====================================================================================
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.format = kGridFormat;
        imageInfo.extent = { kGridResolution, kGridResolution, kGridResolution };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_GridImage, &m_GridAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_GridImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = kGridFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_GridView));

        // Linear + CLAMP_TO_EDGE: world_probe_sampling.glsl's SampleWorldProbeGrid relies on
        // hardware trilinear filtering for smooth probe-to-probe interpolation, and CLAMP_TO_EDGE
        // means a dynamic object slightly outside the grid's own bounds still samples the nearest
        // valid probe layer instead of wrapping or reading black.
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
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_GridSampler));

        // One-time UNDEFINED -> GENERAL transition (mirrors ClusterResolvePass::Init's own one-shot
        // pattern) -- stays GENERAL for this image's entire lifetime.
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
            barrier.image = m_GridImage;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // =====================================================================================
        // STEP 2 -- set 0's layout: grid output + the shared TLAS/vertex/index/draw-range
        // resources (a smaller set-0 than SurfaceCacheGIInjectPass's -- see the class comment on
        // why probes don't read the 4 per-texel Surface Cache atlases).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 5;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[3] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_Set));

        VkDescriptorImageInfo gridStorageInfo{ VK_NULL_HANDLE, m_GridView, VK_IMAGE_LAYOUT_GENERAL };

        VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        VkAccelerationStructureKHR tlasHandle = rtPass.GetTLASHandle();
        accelWrite.accelerationStructureCount = 1;
        accelWrite.pAccelerationStructures = &tlasHandle;

        VkDescriptorBufferInfo vertexBufferInfo{ surfaceCache.GetVertexBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo indexBufferInfo{ surfaceCache.GetIndexBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo drawRangeBufferInfo{ rtPass.GetDrawRangeBuffer(), 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[5]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m_Set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].pImageInfo = &gridStorageInfo;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].pNext = &accelWrite;
        writes[1].dstSet = m_Set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[2].dstSet = m_Set; writes[2].dstBinding = 2; writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[2].pBufferInfo = &vertexBufferInfo;

        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[3].dstSet = m_Set; writes[3].dstBinding = 3; writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[3].pBufferInfo = &indexBufferInfo;

        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[4].dstSet = m_Set; writes[4].dstBinding = 4; writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[4].pBufferInfo = &drawRangeBufferInfo;

        vkUpdateDescriptorSets(m_Device, 5, writes, 0, nullptr);

        // =====================================================================================
        // STEP 3 -- pipeline layout (set 0 above + set 1 mesh SDF trace scene + set 2 surface
        // cache sampling, both shared unmodified from traceContext, exactly like
        // SurfaceCacheGIInjectPass's own 3-set layout) + compute pipeline.
        // =====================================================================================
        VkDescriptorSetLayout setLayouts[3] = {
            m_SetLayout,
            traceContext.GetMeshSdfTraceSetLayout(),
            traceContext.GetSurfaceCacheSamplingSetLayout()
        };

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(WorldProbeInjectPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 3;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/WorldProbeInject.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        m_GridOriginWorld = maths::vec3{ 0.0f, 0.0f, 0.0f };
        m_FrameIndex = 0;

        LOG_INFO("[WorldProbeGridPass] Initialized: 32^3 grid, 2.0 world-unit spacing.");
        return true;
    }

    void WorldProbeGridPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_GridSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_GridSampler, nullptr);
            if (m_GridView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_GridView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_GridImage, m_GridAllocation);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Set = VK_NULL_HANDLE;
        m_GridSampler = VK_NULL_HANDLE;
        m_GridView = VK_NULL_HANDLE;
        m_GridImage = VK_NULL_HANDLE;
        m_GridAllocation = VK_NULL_HANDLE;
        m_GridOriginWorld = maths::vec3{};
        m_FrameIndex = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void WorldProbeGridPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld,
        const SurfaceCacheTraceContext& traceContext, uint32_t traceMode) {
        // Snap the grid's own origin (texel (0,0,0) corner, kGridResolution/2 probes below/behind
        // the camera on each axis) to whole kProbeSpacing steps -- see the class/header comment on
        // why this matters for temporal coherence, mirroring GlobalSDFPass::kSnapChunkVoxels's own
        // snapping rationale (just without that class' incremental-streaming machinery, since this
        // grid is fully rebuilt every call regardless of how far the snapped origin moved).
        float halfExtent = (static_cast<float>(kGridResolution) * 0.5f) * kProbeSpacing;
        maths::vec3 unsnappedOrigin = cameraPositionWorld - maths::vec3{ halfExtent, halfExtent, halfExtent };
        m_GridOriginWorld = maths::vec3{
            std::floor(unsnappedOrigin.x / kProbeSpacing) * kProbeSpacing,
            std::floor(unsnappedOrigin.y / kProbeSpacing) * kProbeSpacing,
            std::floor(unsnappedOrigin.z / kProbeSpacing) * kProbeSpacing
        };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        VkDescriptorSet sets[3] = { m_Set, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 3, sets, 0, nullptr);

        WorldProbeInjectPushConstants pc{};
        pc.gridOriginX = m_GridOriginWorld.x;
        pc.gridOriginY = m_GridOriginWorld.y;
        pc.gridOriginZ = m_GridOriginWorld.z;
        pc.probeSpacing = kProbeSpacing;
        pc.entityCount = traceContext.GetEntityCount();
        pc.frameIndex = m_FrameIndex;
        pc.traceMode = traceMode;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupCount = kGridResolution / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCount, groupCount, groupCount);

        ++m_FrameIndex;
    }

}
