// UE5.8 rendering-parity gap G10b -- reference Path Tracer pass implementation. The ENTIRE file is
// wrapped in #ifndef NDEBUG (CLAUDE.md rule 8, "modes de visualisation" excluded from Release): in a
// Release build this compiles to an empty translation unit -- zero code / symbols survive. See
// PathTracerPass.h's own header comment for the full rationale.
#ifndef NDEBUG

#include "renderer/passes/PathTracerPass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/EngineConfig.h"
#include "core/EntityData.h"
#include "core/Logger.h"
#include "geometry/ClusterFormat.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/AccelerationStructure.h"
#include "renderer/vulkan/RayTracingFunctions.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        // std140-exact mirror of PathTracerViewParamsUBO in PathTracer.rgen (176 bytes).
        struct PathTracerViewParamsUBO {
            maths::mat4 invViewProj;          // 64
            float cameraPosWorld[4];          // 16
            float sunDirection[4];            // 16
            float sunColorIntensity[4];       // 16
            float skyHorizon[4];              // 16
            float skyZenith[4];               // 16
            float viewportSize[2];            // 8
            uint32_t sampleIndex;             // 4
            uint32_t frameIndex;              // 4
            uint32_t maxBounces;              // 4
            uint32_t lightSamples;            // 4
            uint32_t pad0;                    // 4
            uint32_t pad1;                    // 4
        };
        static_assert(sizeof(PathTracerViewParamsUBO) == 176,
            "PathTracerViewParamsUBO must match PathTracer.rgen's UBO exactly (std140 layout)");

        // Byte-for-byte layout match for ResolvePushConstants in PathTracerResolve.comp.
        struct ResolvePushConstants {
            uint32_t sampleCount;
            float exposure;
            float invGamma;
            uint32_t width;
            uint32_t height;
        };
        static_assert(sizeof(ResolvePushConstants) == 20, "ResolvePushConstants must match PathTracerResolve.comp exactly");

        // config-driven integration budget -- kept modest so this Debug-only reference tool stays
        // interactive (and never risks a driver TDR) while still converging to an unbiased result.
        constexpr uint32_t kMaxBounces = 4u;
        constexpr uint32_t kPointLightSamplesPerVertex = 4u;

        // Physical-camera exposure divisor, IDENTICAL to PostProcessComposite.comp's
        // 1.0 / (1.2 * 2^EV100) (and PostProcessPass::ComputeManualEV100) so the reference is exposed
        // exactly like the real-time view it validates.
        float ComputeExposureMultiplier() {
            const float aperture = config::postprocess::EXPOSURE_APERTURE;
            const float shutter = std::max(config::postprocess::EXPOSURE_SHUTTER_SPEED_SECONDS, 1.0e-6f);
            const float iso = config::postprocess::EXPOSURE_ISO;
            float ev100 = std::log2((aperture * aperture) / shutter * 100.0f / iso);
            ev100 -= config::postprocess::EXPOSURE_COMPENSATION_EV; // Positive compensation brightens -> lowers EV100.
            return 1.0f / (1.2f * std::exp2(ev100));
        }

        // Creates a single-mip 2D GPU image in GENERAL-ready state with STORAGE + the caller's extra
        // usage (e.g. TRANSFER_SRC for the display image the swapchain blit reads). Left in UNDEFINED
        // layout -- the caller transitions it to GENERAL once via TransitionImageLayoutOneShot.
        bool CreateStorageImage(VmaAllocator allocator, VkDevice device, VkFormat format, VkExtent2D extent,
            VkImageUsageFlags extraUsage, VkImage& outImage, VmaAllocation& outAlloc, VkImageView& outView) {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = { extent.width, extent.height, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | extraUsage;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage, &outAlloc, nullptr) != VK_SUCCESS) {
                return false;
            }

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = outImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outView));
            return true;
        }

    } // namespace

    bool PathTracerPass::Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
        VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
        const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
        const SurfaceCacheRayTracingPass& rtPass,
        const core::EntityData* entityDataCPU,
        VkBuffer materialParamsBuffer, VkDeviceSize materialParamsSize,
        VkBuffer megaLightsBuffer, VkDeviceSize megaLightsSize) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        // =====================================================================================
        // STEP 1 -- Accumulation (rgba32f, STORAGE) + display (rgba8, STORAGE + TRANSFER_SRC for the
        // swapchain blit) images, both transitioned once to GENERAL (kept there for their lifetime).
        // =====================================================================================
        if (!CreateStorageImage(allocator, device, kAccumFormat, renderExtent, 0,
                m_AccumImage, m_AccumAllocation, m_AccumView)) {
            LOG_ERROR("[PathTracerPass] Failed to create accumulation image.");
            return false;
        }
        if (!CreateStorageImage(allocator, device, kDisplayFormat, renderExtent, VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                m_DisplayImage, m_DisplayAllocation, m_DisplayView)) {
            LOG_ERROR("[PathTracerPass] Failed to create display image.");
            return false;
        }
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VulkanUtils::TransitionImageLayout(cmd, m_AccumImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            VulkanUtils::TransitionImageLayout(cmd, m_DisplayImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            });

        // =====================================================================================
        // STEP 2 -- Per-traced-entity materialID buffer (dense traced index -> materialID), built
        // from core::EntityData: index-aligned with SurfaceCacheTraceContext::GetTracedEntities()
        // exactly like renderer::SurfaceCacheRayTracingPass's own draw-range buffer, so
        // gl_InstanceCustomIndexEXT (== the dense traced index) resolves it directly in PathTracer.rchit.
        // =====================================================================================
        {
            const std::vector<SurfaceCacheTraceContext::TracedEntity>& tracedEntities = traceContext.GetTracedEntities();
            std::vector<uint32_t> materialIDs(std::max<size_t>(tracedEntities.size(), 1), 0u);
            for (size_t i = 0; i < tracedEntities.size(); ++i) {
                materialIDs[i] = entityDataCPU ? entityDataCPU[tracedEntities[i].entityID].materialID : 0u;
            }
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(materialIDs.size()) * sizeof(uint32_t);
            m_TracedMaterialIDBuffer.Create(allocator, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
            std::memcpy(m_TracedMaterialIDBuffer.MappedData(), materialIDs.data(), static_cast<size_t>(bytes));
        }

        // Per-frame view params UBO -- host-visible, persistently mapped (single-frame-in-flight, so a
        // plain memcpy each RecordFrame is safe -- see that method).
        m_ViewParamsBuffer.Create(allocator, sizeof(PathTracerViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        // =====================================================================================
        // STEP 3 -- RT descriptor set layout (set 0, 9 bindings).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[9]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, nullptr };
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 9;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_TraceSetLayout));

        VkDescriptorPoolSize poolSizes[4] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 }, // bindings 2,3,4,5,6,7
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_TraceDescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_TraceDescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_TraceSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_TraceSet));

        // Bindings 1..4 (TLAS + Fallback Mesh vertex/index/draw-range) via the shared helper -- same
        // resources every other GI trace pass binds. instanceCustomIndex/draw-range alignment matches
        // renderer::SurfaceCacheRayTracingPass exactly.
        VulkanUtils::WriteSharedGeometryBindings(m_Device, m_TraceSet, /*baseBinding=*/1,
            rtPass.GetTLASHandle(), surfaceCache.GetVertexBuffer(), surfaceCache.GetIndexBuffer(),
            rtPass.GetDrawRangeBuffer());

        // Remaining bindings (0 accum image, 5 materialID, 6 material table, 7 lights, 8 view params).
        VkDescriptorImageInfo accumInfo{ VK_NULL_HANDLE, m_AccumView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo matIdInfo{ m_TracedMaterialIDBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo matParamsInfo{ materialParamsBuffer, 0, materialParamsSize };
        VkDescriptorBufferInfo lightsInfo{ megaLightsBuffer, 0, megaLightsSize };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

        VkWriteDescriptorSet writes[5]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m_TraceSet; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].pImageInfo = &accumInfo;
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = m_TraceSet; writes[1].dstBinding = 5; writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[1].pBufferInfo = &matIdInfo;
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[2].dstSet = m_TraceSet; writes[2].dstBinding = 6; writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[2].pBufferInfo = &matParamsInfo;
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[3].dstSet = m_TraceSet; writes[3].dstBinding = 7; writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[3].pBufferInfo = &lightsInfo;
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[4].dstSet = m_TraceSet; writes[4].dstBinding = 8; writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[4].pBufferInfo = &viewParamsInfo;
        vkUpdateDescriptorSets(m_Device, 5, writes, 0, nullptr);

        // =====================================================================================
        // STEP 4 -- RT pipeline layout + the 3-stage ray tracing pipeline (rgen/rmiss/rchit).
        // =====================================================================================
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_TraceSetLayout;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_TracePipelineLayout));

        VkShaderModule rgenModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PathTracer.rgen.spv");
        VkShaderModule missModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PathTracer.rmiss.spv");
        VkShaderModule chitModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PathTracer.rchit.spv");

        VkPipelineShaderStageCreateInfo stages[3]{};
        stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR; stages[0].module = rgenModule; stages[0].pName = "main";
        stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR; stages[1].module = missModule; stages[1].pName = "main";
        stages[2] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR; stages[2].module = chitModule; stages[2].pName = "main";

        VkRayTracingShaderGroupCreateInfoKHR groups[3]{};
        groups[0] = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
        groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[0].generalShader = 0; groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[0].anyHitShader = VK_SHADER_UNUSED_KHR; groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
        groups[1] = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
        groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[1].generalShader = 1; groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[1].anyHitShader = VK_SHADER_UNUSED_KHR; groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;
        groups[2] = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
        groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[2].generalShader = VK_SHADER_UNUSED_KHR; groups[2].closestHitShader = 2;
        groups[2].anyHitShader = VK_SHADER_UNUSED_KHR; groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

        VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
        rtPipelineInfo.stageCount = 3;
        rtPipelineInfo.pStages = stages;
        rtPipelineInfo.groupCount = 3;
        rtPipelineInfo.pGroups = groups;
        // The ray-gen shader calls traceRayEXT iteratively (a bounce loop), never recursively -- the
        // closest-hit shader does not itself trace -- so recursion depth 1 (the minimum) suffices.
        // NEE shadow rays use inline rayQueryEXT, which does not count toward pipeline recursion.
        rtPipelineInfo.maxPipelineRayRecursionDepth = 1;
        rtPipelineInfo.layout = m_TracePipelineLayout;
        VK_CHECK(g_RTFunctions.vkCreateRayTracingPipelinesKHR(m_Device, VK_NULL_HANDLE, VulkanPipeline::GetPipelineCache(), 1, &rtPipelineInfo, nullptr, &m_TracePipeline));

        vkDestroyShaderModule(m_Device, rgenModule, nullptr);
        vkDestroyShaderModule(m_Device, missModule, nullptr);
        vkDestroyShaderModule(m_Device, chitModule, nullptr);

        // =====================================================================================
        // STEP 5 -- Shader Binding Table (one shaderGroupBaseAlignment-aligned buffer per region),
        // identical construction to renderer::SurfaceCacheRayTracingPass::Init.
        // =====================================================================================
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
        VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props2.pNext = &rtProps;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        const VkDeviceSize handleSize = rtProps.shaderGroupHandleSize;
        const VkDeviceSize handleSizeAligned = AlignUp(handleSize, rtProps.shaderGroupHandleAlignment);
        const VkDeviceSize regionSize = AlignUp(handleSizeAligned, rtProps.shaderGroupBaseAlignment);

        std::vector<uint8_t> allHandles(3 * static_cast<size_t>(handleSize));
        VK_CHECK(g_RTFunctions.vkGetRayTracingShaderGroupHandlesKHR(m_Device, m_TracePipeline, 0, 3,
            allHandles.size(), allHandles.data()));

        auto makeSbtRegion = [&](GpuBuffer& buffer, const uint8_t* handleData, VkDeviceSize count) -> VkStridedDeviceAddressRegionKHR {
            buffer.Create(allocator, regionSize,
                VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
            std::memset(buffer.MappedData(), 0, static_cast<size_t>(regionSize));
            std::memcpy(buffer.MappedData(), handleData, static_cast<size_t>(handleSize));
            VkStridedDeviceAddressRegionKHR region{};
            region.deviceAddress = GetBufferDeviceAddress(m_Device, buffer.Handle());
            region.stride = handleSizeAligned;
            region.size = handleSizeAligned * count;
            return region;
            };

        m_RaygenRegion = makeSbtRegion(m_RaygenSBT, allHandles.data() + 0 * handleSize, 1);
        m_MissRegion = makeSbtRegion(m_MissSBT, allHandles.data() + 1 * handleSize, 1);
        m_HitRegion = makeSbtRegion(m_HitSBT, allHandles.data() + 2 * handleSize, 1);
        // Raygen region VUID: size == stride (exactly one raygen record per vkCmdTraceRaysKHR).
        m_RaygenRegion.size = m_RaygenRegion.stride;

        // =====================================================================================
        // STEP 6 -- Resolve/tonemap compute pipeline (set 0: accum readonly + display writeonly).
        // =====================================================================================
        VkDescriptorSetLayoutBinding resolveBindings[2]{};
        resolveBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        resolveBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo resolveLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        resolveLayoutInfo.bindingCount = 2;
        resolveLayoutInfo.pBindings = resolveBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &resolveLayoutInfo, nullptr, &m_ResolveSetLayout));

        VkDescriptorPoolSize resolvePoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 };
        VkDescriptorPoolCreateInfo resolvePoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        resolvePoolInfo.maxSets = 1;
        resolvePoolInfo.poolSizeCount = 1;
        resolvePoolInfo.pPoolSizes = &resolvePoolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &resolvePoolInfo, nullptr, &m_ResolveDescriptorPool));

        VkDescriptorSetAllocateInfo resolveSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        resolveSetAlloc.descriptorPool = m_ResolveDescriptorPool;
        resolveSetAlloc.descriptorSetCount = 1;
        resolveSetAlloc.pSetLayouts = &m_ResolveSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &resolveSetAlloc, &m_ResolveSet));

        VkDescriptorImageInfo resolveAccumInfo{ VK_NULL_HANDLE, m_AccumView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo resolveDisplayInfo{ VK_NULL_HANDLE, m_DisplayView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet resolveWrites[2]{};
        resolveWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        resolveWrites[0].dstSet = m_ResolveSet; resolveWrites[0].dstBinding = 0; resolveWrites[0].descriptorCount = 1;
        resolveWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; resolveWrites[0].pImageInfo = &resolveAccumInfo;
        resolveWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        resolveWrites[1].dstSet = m_ResolveSet; resolveWrites[1].dstBinding = 1; resolveWrites[1].descriptorCount = 1;
        resolveWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; resolveWrites[1].pImageInfo = &resolveDisplayInfo;
        vkUpdateDescriptorSets(m_Device, 2, resolveWrites, 0, nullptr);

        VkPushConstantRange resolvePushRange{};
        resolvePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        resolvePushRange.offset = 0;
        resolvePushRange.size = sizeof(ResolvePushConstants);
        VkPipelineLayoutCreateInfo resolvePLInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        resolvePLInfo.setLayoutCount = 1;
        resolvePLInfo.pSetLayouts = &m_ResolveSetLayout;
        resolvePLInfo.pushConstantRangeCount = 1;
        resolvePLInfo.pPushConstantRanges = &resolvePushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &resolvePLInfo, nullptr, &m_ResolvePipelineLayout));

        VkShaderModule resolveModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PathTracerResolve.comp.spv");
        m_ResolvePipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ResolvePipelineLayout, resolveModule);
        vkDestroyShaderModule(m_Device, resolveModule, nullptr);

        m_AccumulatedSamples = 0;
        m_HasPrevCameraView = false;

        LOG_INFO("[PathTracerPass] Initialized reference path tracer (RT pipeline + SBT + resolve).");
        return true;
    }

    void PathTracerPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_TracePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_TracePipeline, nullptr);
            if (m_TracePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_TracePipelineLayout, nullptr);
            if (m_TraceDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_TraceDescriptorPool, nullptr);
            if (m_TraceSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_TraceSetLayout, nullptr);

            if (m_ResolvePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ResolvePipeline, nullptr);
            if (m_ResolvePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ResolvePipelineLayout, nullptr);
            if (m_ResolveDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_ResolveDescriptorPool, nullptr);
            if (m_ResolveSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ResolveSetLayout, nullptr);

            if (m_AccumView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_AccumView, nullptr);
            if (m_AccumImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_AccumImage, m_AccumAllocation);
            if (m_DisplayView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_DisplayView, nullptr);
            if (m_DisplayImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_DisplayImage, m_DisplayAllocation);
        }

        m_TracedMaterialIDBuffer.Destroy();
        m_ViewParamsBuffer.Destroy();
        m_RaygenSBT.Destroy();
        m_MissSBT.Destroy();
        m_HitSBT.Destroy();

        m_TracePipeline = VK_NULL_HANDLE;
        m_TracePipelineLayout = VK_NULL_HANDLE;
        m_TraceDescriptorPool = VK_NULL_HANDLE;
        m_TraceSetLayout = VK_NULL_HANDLE;
        m_TraceSet = VK_NULL_HANDLE;
        m_ResolvePipeline = VK_NULL_HANDLE;
        m_ResolvePipelineLayout = VK_NULL_HANDLE;
        m_ResolveDescriptorPool = VK_NULL_HANDLE;
        m_ResolveSetLayout = VK_NULL_HANDLE;
        m_ResolveSet = VK_NULL_HANDLE;
        m_AccumImage = VK_NULL_HANDLE; m_AccumAllocation = VK_NULL_HANDLE; m_AccumView = VK_NULL_HANDLE;
        m_DisplayImage = VK_NULL_HANDLE; m_DisplayAllocation = VK_NULL_HANDLE; m_DisplayView = VK_NULL_HANDLE;
        m_RaygenRegion = {}; m_MissRegion = {}; m_HitRegion = {}; m_CallableRegion = {};
        m_AccumulatedSamples = 0;
        m_HasPrevCameraView = false;
        m_Device = VK_NULL_HANDLE;
    }

    void PathTracerPass::RecordFrame(VkCommandBuffer cmd, const maths::mat4& invViewProj, const maths::mat4& cameraView,
        const maths::vec3& cameraPositionWorld, const DirectionalLight& sun, uint32_t frameIndex) {
        if (m_TracePipeline == VK_NULL_HANDLE) {
            return;
        }

        // --- Camera-movement reset: compare this frame's (jitter-free) view matrix against the last.
        // Any change zeroes the accumulation so a moved camera starts a fresh, converging integration
        // -- the standard reference-path-tracer behavior. ---
        bool reset = !m_HasPrevCameraView;
        if (m_HasPrevCameraView) {
            for (int i = 0; i < 16; ++i) {
                if (std::fabs(cameraView.m[i] - m_PrevCameraView.m[i]) > 1.0e-6f) {
                    reset = true;
                    break;
                }
            }
        }
        if (reset) {
            m_AccumulatedSamples = 0;
        }
        m_PrevCameraView = cameraView;
        m_HasPrevCameraView = true;

        // --- Upload this frame's view params (single-frame-in-flight -> a direct memcpy into the
        // persistently-mapped host UBO is safe; queue submission makes the write visible to the GPU
        // read with no explicit barrier). sampleIndex == m_AccumulatedSamples: 0 the frame after a
        // reset (the ray-gen shader overwrites the sum), otherwise the running index. ---
        PathTracerViewParamsUBO params{};
        params.invViewProj = invViewProj;
        params.cameraPosWorld[0] = cameraPositionWorld.x;
        params.cameraPosWorld[1] = cameraPositionWorld.y;
        params.cameraPosWorld[2] = cameraPositionWorld.z;
        params.sunDirection[0] = sun.direction.x;
        params.sunDirection[1] = sun.direction.y;
        params.sunDirection[2] = sun.direction.z;
        params.sunColorIntensity[0] = sun.color.x;
        params.sunColorIntensity[1] = sun.color.y;
        params.sunColorIntensity[2] = sun.color.z;
        params.sunColorIntensity[3] = sun.intensity;
        // Plausible HDR sky-dome radiance (pre-scaled), balanced against the sun's ~10k lux so the
        // shared exposure/tonemap lands sky + sunlit surfaces in a sensible display range.
        const float kHorizon[3] = { 0.90f, 0.95f, 1.00f };
        const float kZenith[3] = { 0.35f, 0.55f, 0.95f };
        const float kHorizonLum = 1400.0f;
        const float kZenithLum = 1000.0f;
        for (int c = 0; c < 3; ++c) {
            params.skyHorizon[c] = kHorizon[c] * kHorizonLum;
            params.skyZenith[c] = kZenith[c] * kZenithLum;
        }
        params.viewportSize[0] = static_cast<float>(m_RenderExtent.width);
        params.viewportSize[1] = static_cast<float>(m_RenderExtent.height);
        params.sampleIndex = m_AccumulatedSamples;
        params.frameIndex = frameIndex;
        params.maxBounces = kMaxBounces;
        params.lightSamples = kPointLightSamplesPerVertex;
        std::memcpy(m_ViewParamsBuffer.MappedData(), &params, sizeof(params));

        // --- Path-trace dispatch: one ray-gen invocation per pixel accumulates this frame's sample
        // into the rgba32f sum image (which stays in GENERAL). ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_TracePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_TracePipelineLayout, 0, 1, &m_TraceSet, 0, nullptr);
        g_RTFunctions.vkCmdTraceRaysKHR(cmd, &m_RaygenRegion, &m_MissRegion, &m_HitRegion, &m_CallableRegion,
            m_RenderExtent.width, m_RenderExtent.height, 1);

        // Barrier: ray-gen's accumulation writes -> resolve compute reads (same image, still GENERAL).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        // --- Resolve/tonemap dispatch: average the sum and tonemap into the display image. The
        // display image's write (COMPUTE_SHADER SHADER_STORAGE_WRITE) is made visible to the final
        // swapchain blit by renderer::ClusterRenderPipeline's own existing resolve->blit barrier. ---
        ResolvePushConstants pc{};
        pc.sampleCount = m_AccumulatedSamples + 1u; // this frame contributes one more sample.
        pc.exposure = ComputeExposureMultiplier();
        pc.invGamma = 1.0f / std::max(config::postprocess::DISPLAY_GAMMA, 1.0e-3f);
        pc.width = m_RenderExtent.width;
        pc.height = m_RenderExtent.height;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ResolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ResolvePipelineLayout, 0, 1, &m_ResolveSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_ResolvePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        const uint32_t gx = (m_RenderExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        const uint32_t gy = (m_RenderExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, gx, gy, 1);

        // This frame's sample is now recorded.
        m_AccumulatedSamples += 1u;
    }

}

#endif // NDEBUG
