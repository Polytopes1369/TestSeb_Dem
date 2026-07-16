#include "renderer/passes/SurfaceCacheRayTracingPass.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h"
#include "renderer/vulkan/RayTracingFunctions.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        // std430-exact mirror of SurfaceCacheHWRT.rchit's EntityDrawRangeGpu -- see that file's
        // own comment on why this is 16 bytes (padded) rather than 12.
        struct EntityDrawRangeGpu {
            int32_t vertexOffset = 0;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
            uint32_t _pad = 0;
        };
        static_assert(sizeof(EntityDrawRangeGpu) == 16, "EntityDrawRangeGpu must match SurfaceCacheHWRT.rchit's layout exactly");

        // Byte-for-byte layout match for HWRTPushConstants in SurfaceCacheHWRT.rgen.
        struct HWRTPushConstants {
            uint32_t rayCount = 0;
        };

    } // namespace

    bool SurfaceCacheRayTracingPass::Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
        VkCommandPool commandPool, VkQueue queue,
        const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache) {
        Shutdown();
        m_Device = device;

        const std::vector<SurfaceCacheTraceContext::TracedEntity>& tracedEntities = traceContext.GetTracedEntities();
        VkBuffer vertexBuffer = surfaceCache.GetVertexBuffer();
        VkBuffer indexBuffer = surfaceCache.GetIndexBuffer();
        const uint32_t totalVertexCount = static_cast<uint32_t>(surfaceCache.GetVertexBufferSize() / sizeof(geometry::FallbackVertex));

        // =====================================================================================
        // STEP 1 -- One BLAS per traced entity with non-empty Fallback Mesh geometry, built
        // directly against surfaceCache's own combined vertex/index buffers (no geometry
        // duplication -- see AccelerationStructure::BuildBLAS's own comment). Entities with no
        // geometry (should not happen in practice -- see SurfaceCacheTraceContext's own comment
        // on why GlobalSDFPass and SurfaceCachePass read the same cache file -- but handled
        // defensively) simply get no TLAS instance: instanceCustomIndex is an explicit per-
        // instance field, NOT tied to the instance array's own position, so skipping an entity
        // here leaves every OTHER entity's gl_InstanceCustomIndexEXT resolution unaffected.
        // =====================================================================================
        m_BLASList.reserve(tracedEntities.size());
        std::vector<VkAccelerationStructureInstanceKHR> instances;
        instances.reserve(tracedEntities.size());
        std::vector<EntityDrawRangeGpu> hostDrawRanges(tracedEntities.size());

        VkTransformMatrixKHR identityTransform{};
        identityTransform.matrix[0][0] = 1.0f;
        identityTransform.matrix[1][1] = 1.0f;
        identityTransform.matrix[2][2] = 1.0f;

        for (uint32_t i = 0; i < tracedEntities.size(); ++i) {
            const SurfaceCachePass::EntityDrawRange& range = tracedEntities[i].drawRange;

            EntityDrawRangeGpu gpuRange{};
            gpuRange.vertexOffset = range.vertexOffset;
            gpuRange.firstIndex = range.firstIndex;
            gpuRange.indexCount = range.indexCount;
            hostDrawRanges[i] = gpuRange;

            if (range.indexCount == 0) {
                LOG_WARNING(std::format(
                    "[SurfaceCacheRayTracingPass] Traced entityID={} (index {}) has no Fallback Mesh geometry; "
                    "no BLAS/TLAS instance built for it.", tracedEntities[i].entityID, i));
                continue;
            }

            const uint32_t triangleCount = range.indexCount / 3u;
            // Conservative (not necessarily tight) upper bound on any vertex index this entity's
            // own index sub-range can reference -- safe per the VK_KHR_acceleration_structure spec
            // (maxVertex only needs to bound the actually-read range, an inflated bound costs
            // nothing but a slightly larger internal BLAS scratch estimate). Computing the EXACT
            // per-entity vertex count would require SurfaceCachePass to expose each entity's
            // vertex-array length individually, which nothing else in this codebase needs.
            const uint32_t maxVertex = totalVertexCount > static_cast<uint32_t>(range.vertexOffset)
                ? (totalVertexCount - static_cast<uint32_t>(range.vertexOffset) - 1u) : 0u;

            AccelerationStructure blas = BuildBLAS(physicalDevice, device, allocator, commandPool, queue,
                vertexBuffer, sizeof(geometry::FallbackVertex), maxVertex,
                static_cast<VkDeviceSize>(range.vertexOffset) * sizeof(geometry::FallbackVertex),
                indexBuffer, static_cast<VkDeviceSize>(range.firstIndex) * sizeof(uint32_t),
                triangleCount);

            VkAccelerationStructureInstanceKHR instance{};
            instance.transform = identityTransform;
            instance.instanceCustomIndex = i; // The dense traced-entity index -- see SurfaceCacheHWRT.rchit's own comment.
            instance.mask = 0xFF;
            instance.instanceShaderBindingTableRecordOffset = 0; // Single hit group (index 0).
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference = blas.DeviceAddress();
            instances.push_back(instance);

            m_BLASList.push_back(std::move(blas));
        }

        LOG_INFO(std::format("[SurfaceCacheRayTracingPass] Built {} BLAS / {} TLAS instance(s).",
            m_BLASList.size(), instances.size()));

        // =====================================================================================
        // STEP 2 -- TLAS.
        // =====================================================================================
        m_TLAS = BuildTLAS(physicalDevice, device, allocator, commandPool, queue, instances);

        // =====================================================================================
        // STEP 3 -- set 3's draw-range SSBO (index-aligned with EVERY traced entity, including
        // the geometry-less ones skipped above -- see this function's own STEP 1 comment on why
        // that is always safe: no ray can ever produce an instanceCustomIndex for a skipped slot).
        // =====================================================================================
        const VkDeviceSize drawRangeBytes = static_cast<VkDeviceSize>(std::max<size_t>(hostDrawRanges.size(), 1)) * sizeof(EntityDrawRangeGpu);
        m_DrawRangeBuffer.Create(allocator, drawRangeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        if (!hostDrawRanges.empty()) {
            std::memcpy(m_DrawRangeBuffer.MappedData(), hostDrawRanges.data(), hostDrawRanges.size() * sizeof(EntityDrawRangeGpu));
        } else {
            std::memset(m_DrawRangeBuffer.MappedData(), 0, static_cast<size_t>(drawRangeBytes));
        }

        // =====================================================================================
        // STEP 4 -- descriptor set layouts: set 0 (ray I/O + TLAS) and set 3 (Fallback Mesh
        // vertex/index/draw-range buffers). Sets 1/2 come from `traceContext` unmodified.
        // =====================================================================================
        VkDescriptorSetLayoutBinding rayBindings[3]{};
        rayBindings[0].binding = 0;
        rayBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        rayBindings[0].descriptorCount = 1;
        rayBindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        rayBindings[1].binding = 1;
        rayBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        rayBindings[1].descriptorCount = 1;
        rayBindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        rayBindings[2].binding = 2;
        rayBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        rayBindings[2].descriptorCount = 1;
        rayBindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo raySetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        raySetLayoutInfo.bindingCount = 3;
        raySetLayoutInfo.pBindings = rayBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &raySetLayoutInfo, nullptr, &m_RaySetLayout));

        VkDescriptorSetLayoutBinding geometryBindings[3]{};
        for (uint32_t b = 0; b < 3; ++b) {
            geometryBindings[b].binding = b;
            geometryBindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            geometryBindings[b].descriptorCount = 1;
            geometryBindings[b].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        }
        VkDescriptorSetLayoutCreateInfo geometrySetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        geometrySetLayoutInfo.bindingCount = 3;
        geometrySetLayoutInfo.pBindings = geometryBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &geometrySetLayoutInfo, nullptr, &m_GeometrySetLayout));

        VkDescriptorPoolSize poolSizes[2] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 },
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout layoutsToAllocate[2] = { m_RaySetLayout, m_GeometrySetLayout };
        VkDescriptorSet allocatedSets[2] = {};
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 2;
        setAllocInfo.pSetLayouts = layoutsToAllocate;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, allocatedSets));
        m_RaySet = allocatedSets[0];
        m_GeometrySet = allocatedSets[1];

        VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        VkAccelerationStructureKHR tlasHandle = m_TLAS.Handle();
        accelWrite.accelerationStructureCount = 1;
        accelWrite.pAccelerationStructures = &tlasHandle;

        VkDescriptorBufferInfo vertexBufferInfo{ vertexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo indexBufferInfo{ indexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo drawRangeBufferInfo{ m_DrawRangeBuffer.Handle(), 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[4]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].pNext = &accelWrite;
        writes[0].dstSet = m_RaySet;
        writes[0].dstBinding = 2;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = m_GeometrySet;
        writes[1].dstBinding = 0;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &vertexBufferInfo;

        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[2].dstSet = m_GeometrySet;
        writes[2].dstBinding = 1;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &indexBufferInfo;

        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[3].dstSet = m_GeometrySet;
        writes[3].dstBinding = 2;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &drawRangeBufferInfo;

        vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);
        // Note: set 0's bindings 0/1 (ray request/result) are written by SetRayBuffers(), not
        // here -- they are caller-supplied and may change across calls (see that function's own
        // comment).

        // =====================================================================================
        // STEP 5 -- pipeline layout (sets 0..3) + the 3-stage ray tracing pipeline.
        // =====================================================================================
        VkDescriptorSetLayout pipelineSetLayouts[4] = {
            m_RaySetLayout,
            traceContext.GetMeshSdfTraceSetLayout(),
            traceContext.GetSurfaceCacheSamplingSetLayout(),
            m_GeometrySetLayout
        };

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(HWRTPushConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 4;
        pipelineLayoutInfo.pSetLayouts = pipelineSetLayouts;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule rgenModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/SurfaceCacheHWRT.rgen.spv");
        VkShaderModule missModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/SurfaceCacheHWRT.rmiss.spv");
        VkShaderModule chitModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/SurfaceCacheHWRT.rchit.spv");

        VkPipelineShaderStageCreateInfo stages[3]{};
        stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        stages[0].module = rgenModule;
        stages[0].pName = "main";
        stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        stages[1].module = missModule;
        stages[1].pName = "main";
        stages[2] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        stages[2].module = chitModule;
        stages[2].pName = "main";

        VkRayTracingShaderGroupCreateInfoKHR groups[3]{};
        groups[0] = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
        groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[0].generalShader = 0; // rgen
        groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
        groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
        groups[1] = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
        groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        groups[1].generalShader = 1; // rmiss
        groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
        groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;
        groups[2] = { VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
        groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        groups[2].generalShader = VK_SHADER_UNUSED_KHR;
        groups[2].closestHitShader = 2; // rchit
        groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
        groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

        VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
        rtPipelineInfo.stageCount = 3;
        rtPipelineInfo.pStages = stages;
        rtPipelineInfo.groupCount = 3;
        rtPipelineInfo.pGroups = groups;
        // No recursive traceRayEXT calls anywhere in this pipeline (SurfaceCacheHWRT.rchit
        // returns its result via the payload with no further trace) -- depth 1 is the minimum
        // and cheapest legal value.
        rtPipelineInfo.maxPipelineRayRecursionDepth = 1;
        rtPipelineInfo.layout = m_PipelineLayout;
        VK_CHECK(g_RTFunctions.vkCreateRayTracingPipelinesKHR(m_Device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtPipelineInfo, nullptr, &m_Pipeline));

        vkDestroyShaderModule(m_Device, rgenModule, nullptr);
        vkDestroyShaderModule(m_Device, missModule, nullptr);
        vkDestroyShaderModule(m_Device, chitModule, nullptr);

        // =====================================================================================
        // STEP 6 -- Shader Binding Table: one dedicated, shaderGroupBaseAlignment-aligned buffer
        // per region (raygen/miss/hit -- no callable shaders in this pipeline). A single shared
        // SBT buffer with computed per-region offsets would also be legal, but 3 separate
        // allocations sidestep having to hand-verify each region's offset independently satisfies
        // shaderGroupBaseAlignment (only each buffer's own base address needs to, which VMA's
        // dedicated-allocation guarantee already covers here).
        // =====================================================================================
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
        VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
        props2.pNext = &rtProps;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        const VkDeviceSize handleSize = rtProps.shaderGroupHandleSize;
        const VkDeviceSize handleSizeAligned = AlignUp(handleSize, rtProps.shaderGroupHandleAlignment);
        const VkDeviceSize regionSize = AlignUp(handleSizeAligned, rtProps.shaderGroupBaseAlignment);

        std::vector<uint8_t> allHandles(3 * static_cast<size_t>(handleSize));
        VK_CHECK(g_RTFunctions.vkGetRayTracingShaderGroupHandlesKHR(m_Device, m_Pipeline, 0, 3,
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
        // Raygen region has the additional VUID requirement that size == stride (exactly one
        // raygen shader record is ever addressed per vkCmdTraceRaysKHR call).
        m_RaygenRegion.size = m_RaygenRegion.stride;

        LOG_INFO("[SurfaceCacheRayTracingPass] Initialized HWRT pipeline + SBT.");
        return true;
    }

    void SurfaceCacheRayTracingPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_RaySetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RaySetLayout, nullptr);
            if (m_GeometrySetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_GeometrySetLayout, nullptr);
        }
        m_BLASList.clear();
        m_TLAS.Destroy();
        m_DrawRangeBuffer.Destroy();
        m_RaygenSBT.Destroy();
        m_MissSBT.Destroy();
        m_HitSBT.Destroy();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_RaySetLayout = VK_NULL_HANDLE;
        m_GeometrySetLayout = VK_NULL_HANDLE;
        m_RaySet = VK_NULL_HANDLE;
        m_GeometrySet = VK_NULL_HANDLE;
        m_RaygenRegion = {};
        m_MissRegion = {};
        m_HitRegion = {};
        m_CallableRegion = {};
        m_Device = VK_NULL_HANDLE;
    }

    void SurfaceCacheRayTracingPass::SetRayBuffers(VkBuffer rayBuffer, VkDeviceSize rayBufferSize, VkBuffer resultBuffer, VkDeviceSize resultBufferSize) {
        VulkanUtils::WriteRayBuffersDescriptorSet(m_Device, m_RaySet, rayBuffer, rayBufferSize, resultBuffer, resultBufferSize);
    }

    void SurfaceCacheRayTracingPass::RecordTrace(VkCommandBuffer cmd, uint32_t rayCount, const SurfaceCacheTraceContext& traceContext) {
        if (rayCount == 0) {
            return;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_Pipeline);

        VkDescriptorSet sets[4] = { m_RaySet, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet(), m_GeometrySet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_PipelineLayout, 0, 4, sets, 0, nullptr);

        HWRTPushConstants pc{};
        pc.rayCount = rayCount;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, sizeof(pc), &pc);

        g_RTFunctions.vkCmdTraceRaysKHR(cmd, &m_RaygenRegion, &m_MissRegion, &m_HitRegion, &m_CallableRegion, rayCount, 1, 1);
    }

}
