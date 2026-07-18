#include "renderer/passes/SurfaceCacheRayTracingPass.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>

#include "core/EntityData.h" // core::GetFlag/EntityFlags::IsSkeletallyAnimated -- creature detection, see Init()'s own comment.
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

        // Skeletal-animation feature: byte-for-byte layout match for CreatureSkinningConstants in
        // CreatureBlasSkinning.comp.
        struct CreatureSkinningConstants {
            uint32_t srcVertexOffset = 0;
            uint32_t vertexCount = 0;
            float centerX = 0.0f;
            float centerY = 0.0f;
            float centerZ = 0.0f;
        };

    } // namespace

    bool SurfaceCacheRayTracingPass::Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
        VkCommandPool commandPool, VkQueue queue,
        const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
        VkBuffer boneMatricesBuffer, const core::EntityData* entityDataCPU) {
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

            // Skeletal-animation feature (ray-traced GI/reflections fix): the ONE traced entity
            // with core::EntityFlags::IsSkeletallyAnimated set gets ALLOW_UPDATE build flags instead
            // of every other entity's default PREFER_FAST_TRACE -- see this method's own header
            // comment and AccelerationStructure.h's own BuildBLAS comment for why. `entityDataCPU`
            // is indexed by tracedEntities[i].entityID (a meshID, the same index every OTHER
            // entityDataCPU consumer in this codebase uses), which is always in-bounds for a real
            // traced entity by construction (see SurfaceCacheTraceContext::TracedEntity's own
            // comment) -- guarded by entityDataCPU != nullptr regardless, per this method's own
            // tolerance for a null array (see its own header comment).
            const bool isCreature = entityDataCPU != nullptr &&
                core::GetFlag(entityDataCPU[tracedEntities[i].entityID].flags, core::EntityFlags::IsSkeletallyAnimated);

            AccelerationStructure blas = BuildBLAS(physicalDevice, device, allocator, commandPool, queue,
                vertexBuffer, sizeof(geometry::FallbackVertex), maxVertex,
                static_cast<VkDeviceSize>(range.vertexOffset) * sizeof(geometry::FallbackVertex),
                indexBuffer, static_cast<VkDeviceSize>(range.firstIndex) * sizeof(uint32_t),
                triangleCount, /*allowUpdate=*/isCreature);

            VkAccelerationStructureInstanceKHR instance{};
            instance.transform = identityTransform;
            instance.instanceCustomIndex = i; // The dense traced-entity index -- see SurfaceCacheHWRT.rchit's own comment.
            instance.mask = 0xFF;
            instance.instanceShaderBindingTableRecordOffset = 0; // Single hit group (index 0).
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference = blas.DeviceAddress();
            instances.push_back(instance);

            if (isCreature) {
                // Recorded BEFORE the push_back below, so this index lands exactly on the entry
                // that push_back is about to append -- mirrors m_RefitInstanceInfos' own "record
                // the about-to-be-appended entry's index" pattern one line below.
                m_CreatureBlasListIndex = static_cast<uint32_t>(m_BLASList.size());
                m_CreatureEntityID = tracedEntities[i].entityID;
                m_CreatureSrcVertexOffset = static_cast<uint32_t>(range.vertexOffset);
                m_CreatureVertexCount = maxVertex + 1u; // == totalVertexCount - vertexOffset (see maxVertex's own comment) -- from this entity's own span to the end of the shared buffer, the same conservative bound maxVertex itself already uses.
                m_CreatureMaxVertex = maxVertex;
                m_CreatureIndexBuffer = indexBuffer;
                m_CreatureIndexOffsetBytes = static_cast<VkDeviceSize>(range.firstIndex) * sizeof(uint32_t);
                m_CreatureTriangleCount = triangleCount;
            }

            m_BLASList.push_back(std::move(blas));
            // Phase 4 integration: records exactly which tracedEntities index/entityID this BLAS
            // (now m_BLASList's newest entry) corresponds to -- see m_RefitInstanceInfos' own
            // header comment on why this is NOT assumed to be an implicit 1:1 index alignment.
            m_RefitInstanceInfos.push_back(RefitInstanceInfo{ i, tracedEntities[i].entityID });
        }

        LOG_INFO(std::format("[SurfaceCacheRayTracingPass] Built {} BLAS / {} TLAS instance(s).",
            m_BLASList.size(), instances.size()));

        // =====================================================================================
        // STEP 1.5 -- Skeletal-animation feature (ray-traced GI/reflections fix): if STEP 1 found a
        // skeletally-animated entity, allocate its dedicated per-frame-skinned vertex buffer + BLAS
        // UPDATE-mode scratch resources + CreatureBlasSkinning.comp's own compute pipeline/
        // descriptor set. Entirely skipped (every m_Creature*/m_Skinning* member stays at its
        // default/empty state) if no such entity was traced -- RecordCreatureBlasUpdate() checks
        // HasCreatureBlas() and is a no-op in that case.
        // =====================================================================================
        if (HasCreatureBlas()) {
            const VkDeviceSize creatureVertexBytes = static_cast<VkDeviceSize>(m_CreatureVertexCount) * sizeof(geometry::FallbackVertex);
            m_CreatureSkinnedVertexBuffer.Create(allocator, creatureVertexBytes,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            m_CreatureBlasUpdateResources = CreateBlasUpdateResources(physicalDevice, device, allocator,
                sizeof(geometry::FallbackVertex), m_CreatureMaxVertex, m_CreatureTriangleCount);

            // --- CreatureBlasSkinning.comp's own compute pipeline: 3-binding descriptor set (source
            // vertex buffer / bone matrices SSBO / dest skinned vertex buffer), all COMPUTE-stage
            // STORAGE_BUFFER -- kept separate from m_RaySetLayout/m_GeometrySetLayout above (a
            // different pipeline bind point entirely). ---
            VkDescriptorSetLayoutBinding skinningBindings[3]{};
            for (uint32_t b = 0; b < 3; ++b) {
                skinningBindings[b].binding = b;
                skinningBindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                skinningBindings[b].descriptorCount = 1;
                skinningBindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo skinningSetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            skinningSetLayoutInfo.bindingCount = 3;
            skinningSetLayoutInfo.pBindings = skinningBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &skinningSetLayoutInfo, nullptr, &m_SkinningSetLayout));

            VkDescriptorPoolSize skinningPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
            VkDescriptorPoolCreateInfo skinningPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            skinningPoolInfo.maxSets = 1;
            skinningPoolInfo.poolSizeCount = 1;
            skinningPoolInfo.pPoolSizes = &skinningPoolSize;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &skinningPoolInfo, nullptr, &m_SkinningDescriptorPool));

            VkDescriptorSetAllocateInfo skinningSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            skinningSetAllocInfo.descriptorPool = m_SkinningDescriptorPool;
            skinningSetAllocInfo.descriptorSetCount = 1;
            skinningSetAllocInfo.pSetLayouts = &m_SkinningSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &skinningSetAllocInfo, &m_SkinningSet));

            // Binding 0 is the SHARED surfaceCache vertex buffer (read-only source -- every other
            // entity's BLAS/SurfaceCacheHWRT.rchit continue to read it unmodified, see
            // CreatureBlasSkinning.comp's own header comment), NOT m_CreatureSkinnedVertexBuffer.
            VkDescriptorBufferInfo skinningSrcInfo{ vertexBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo skinningBoneInfo{ boneMatricesBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo skinningDstInfo{ m_CreatureSkinnedVertexBuffer.Handle(), 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet skinningWrites[3]{};
            skinningWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            skinningWrites[0].dstSet = m_SkinningSet;
            skinningWrites[0].dstBinding = 0;
            skinningWrites[0].descriptorCount = 1;
            skinningWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            skinningWrites[0].pBufferInfo = &skinningSrcInfo;
            skinningWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            skinningWrites[1].dstSet = m_SkinningSet;
            skinningWrites[1].dstBinding = 1;
            skinningWrites[1].descriptorCount = 1;
            skinningWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            skinningWrites[1].pBufferInfo = &skinningBoneInfo;
            skinningWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            skinningWrites[2].dstSet = m_SkinningSet;
            skinningWrites[2].dstBinding = 2;
            skinningWrites[2].descriptorCount = 1;
            skinningWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            skinningWrites[2].pBufferInfo = &skinningDstInfo;
            vkUpdateDescriptorSets(m_Device, 3, skinningWrites, 0, nullptr);

            VkPushConstantRange skinningPushConstantRange{};
            skinningPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            skinningPushConstantRange.offset = 0;
            skinningPushConstantRange.size = sizeof(CreatureSkinningConstants);

            VkPipelineLayoutCreateInfo skinningLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            skinningLayoutInfo.setLayoutCount = 1;
            skinningLayoutInfo.pSetLayouts = &m_SkinningSetLayout;
            skinningLayoutInfo.pushConstantRangeCount = 1;
            skinningLayoutInfo.pPushConstantRanges = &skinningPushConstantRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &skinningLayoutInfo, nullptr, &m_SkinningPipelineLayout));

            VkShaderModule skinningModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/CreatureBlasSkinning.comp.spv");
            VkComputePipelineCreateInfo skinningPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            skinningPipelineInfo.layout = m_SkinningPipelineLayout;
            skinningPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            skinningPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            skinningPipelineInfo.stage.module = skinningModule;
            skinningPipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &skinningPipelineInfo, nullptr, &m_SkinningPipeline));
            vkDestroyShaderModule(m_Device, skinningModule, nullptr);

            LOG_INFO(std::format("[SurfaceCacheRayTracingPass] Skeletal-animation feature: creature entityID={} BLAS is per-frame UPDATE-refit ({} vertices, {} triangles).",
                m_CreatureEntityID, m_CreatureVertexCount, m_CreatureTriangleCount));
        }

        // =====================================================================================
        // STEP 2 -- TLAS.
        // =====================================================================================
        m_TLAS = BuildTLAS(physicalDevice, device, allocator, commandPool, queue, instances);

        // Phase 4 integration: persistent per-frame refit buffers, sized for the SAME instance
        // count this initial build just used -- see RecordRefreshTLAS's own comment.
        m_TlasRefitResources = CreateTlasRefitResources(physicalDevice, device, allocator, static_cast<uint32_t>(instances.size()));

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
            // Skeletal-animation feature: CreatureBlasSkinning.comp's own pipeline/layout/descriptor
            // pool+layout -- mirrors the ray tracing pipeline's own teardown just above.
            if (m_SkinningPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_SkinningPipeline, nullptr);
            if (m_SkinningPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_SkinningPipelineLayout, nullptr);
            if (m_SkinningDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_SkinningDescriptorPool, nullptr);
            if (m_SkinningSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SkinningSetLayout, nullptr);
        }
        m_BLASList.clear();
        m_RefitInstanceInfos.clear();
        m_TlasRefitResources.instanceBuffer.Destroy();
        m_TlasRefitResources.scratchBuffer.Destroy();
        m_TlasRefitResources.scratchAddress = 0;
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

        // Skeletal-animation feature: creature BLAS-refit state, reset to "no skeletally-animated
        // entity traced" (see m_CreatureBlasListIndex's own header comment).
        m_CreatureBlasListIndex = kInvalidBlasListIndex;
        m_CreatureEntityID = 0;
        m_CreatureSrcVertexOffset = 0;
        m_CreatureVertexCount = 0;
        m_CreatureMaxVertex = 0;
        m_CreatureIndexBuffer = VK_NULL_HANDLE;
        m_CreatureIndexOffsetBytes = 0;
        m_CreatureTriangleCount = 0;
        m_CreatureSkinnedVertexBuffer.Destroy();
        m_CreatureBlasUpdateResources.scratchBuffer.Destroy();
        m_CreatureBlasUpdateResources.scratchAddress = 0;
        m_SkinningPipeline = VK_NULL_HANDLE;
        m_SkinningPipelineLayout = VK_NULL_HANDLE;
        m_SkinningDescriptorPool = VK_NULL_HANDLE;
        m_SkinningSetLayout = VK_NULL_HANDLE;
        m_SkinningSet = VK_NULL_HANDLE;

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

    void SurfaceCacheRayTracingPass::RecordRefreshTLAS(VkCommandBuffer cmd, const core::EntityTransformCPU* entityTransformsCPU) {
        // Rebuilds the instance array fresh every frame from this frame's rotation data -- BLAS
        // device addresses/instanceCustomIndex/mask/SBT-record-offset/flags are identical to
        // Init()'s own build (geometry never changes), only .transform differs.
        std::vector<VkAccelerationStructureInstanceKHR> instances;
        instances.reserve(m_RefitInstanceInfos.size());

        for (size_t k = 0; k < m_RefitInstanceInfos.size(); ++k) {
            const RefitInstanceInfo& info = m_RefitInstanceInfos[k];
            const core::EntityTransformCPU& xform = entityTransformsCPU[info.entityID];

            // VkTransformMatrixKHR = [R | center - R*center] -- the BLAS stores geometry already
            // positioned at its rest-pose world location, so the correct instance transform
            // rotates around the entity's OWN pivot (xform.center), not the world origin: a point
            // p transforms as p' = center + R*(p - center) = R*p + (center - R*center), giving the
            // 3x4 affine form Vulkan's row-major VkTransformMatrixKHR expects directly.
            const maths::mat4& rot = xform.rotation;
            maths::vec3 translation = xform.center - maths::vec3{
                rot.m[0] * xform.center.x + rot.m[4] * xform.center.y + rot.m[8]  * xform.center.z,
                rot.m[1] * xform.center.x + rot.m[5] * xform.center.y + rot.m[9]  * xform.center.z,
                rot.m[2] * xform.center.x + rot.m[6] * xform.center.y + rot.m[10] * xform.center.z};

            VkTransformMatrixKHR transform{};
            transform.matrix[0][0] = rot.m[0]; transform.matrix[0][1] = rot.m[4]; transform.matrix[0][2] = rot.m[8];  transform.matrix[0][3] = translation.x;
            transform.matrix[1][0] = rot.m[1]; transform.matrix[1][1] = rot.m[5]; transform.matrix[1][2] = rot.m[9];  transform.matrix[1][3] = translation.y;
            transform.matrix[2][0] = rot.m[2]; transform.matrix[2][1] = rot.m[6]; transform.matrix[2][2] = rot.m[10]; transform.matrix[2][3] = translation.z;

            VkAccelerationStructureInstanceKHR instance{};
            instance.transform = transform;
            instance.instanceCustomIndex = info.instanceCustomIndex;
            instance.mask = 0xFF;
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference = m_BLASList[k].DeviceAddress();
            instances.push_back(instance);
        }

        RecordRefitTLAS(cmd, m_Device, m_TLAS.Handle(), m_TlasRefitResources, instances);
    }

    void SurfaceCacheRayTracingPass::RecordCreatureBlasUpdate(VkCommandBuffer cmd, const core::EntityTransformCPU* entityTransformsCPU) {
        if (!HasCreatureBlas()) {
            return; // Init() found no skeletally-animated entity -- see this method's own declaration comment.
        }

        // =========================================================================================
        // Step 1 -- CreatureBlasSkinning.comp: writes this frame's skinned vertex positions into
        // m_CreatureSkinnedVertexBuffer from entityTransformsCPU[m_CreatureEntityID].center + the
        // bone-matrices SSBO this pipeline's descriptor set was bound to at Init() (the CALLER's
        // responsibility to have already refreshed THIS frame -- see this method's own declaration
        // comment).
        // =========================================================================================
        const core::EntityTransformCPU& xform = entityTransformsCPU[m_CreatureEntityID];

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SkinningPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SkinningPipelineLayout, 0, 1, &m_SkinningSet, 0, nullptr);

        CreatureSkinningConstants pc{};
        pc.srcVertexOffset = m_CreatureSrcVertexOffset;
        pc.vertexCount = m_CreatureVertexCount;
        pc.centerX = xform.center.x;
        pc.centerY = xform.center.y;
        pc.centerZ = xform.center.z;
        vkCmdPushConstants(cmd, m_SkinningPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        constexpr uint32_t kSkinningLocalSizeX = 64; // Matches CreatureBlasSkinning.comp's own layout(local_size_x = 64).
        const uint32_t groupCount = (m_CreatureVertexCount + kSkinningLocalSizeX - 1u) / kSkinningLocalSizeX;
        vkCmdDispatch(cmd, groupCount, 1, 1);

        // =========================================================================================
        // Step 2 -- compute-write -> acceleration-structure-build-read barrier. Per the
        // VK_KHR_acceleration_structure spec's own synchronization guidance (and this file's own
        // BuildAccelerationStructure/RecordRefitTLAS precedent just above, both of which already use
        // VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR for "this stage reads X as acceleration-
        // structure build input", never the WRITE bit, which describes writing the acceleration
        // structure's OWN memory, not reading a geometry input buffer feeding into that write): the
        // destination access for "an UPDATE-mode build reading m_CreatureSkinnedVertexBuffer as its
        // vertex geometry input" is ACCELERATION_STRUCTURE_READ_BIT_KHR at the
        // ACCELERATION_STRUCTURE_BUILD_BIT_KHR stage -- the buffer is consumed as build INPUT, it is
        // never written by the build itself (only the acceleration structure's own backing buffer,
        // `blas`'s m_Buffer, is written -- RecordUpdateBLAS below issues its own separate barrier for
        // that write, scoped to the AS object itself, not this vertex buffer).
        // =========================================================================================
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        // =========================================================================================
        // Step 3 -- BLAS UPDATE refit, reading m_CreatureSkinnedVertexBuffer (just written above)
        // instead of the shared, static surfaceCache vertex buffer Init()'s own one-shot build read
        // -- the index buffer/offset/triangleCount stay the SAME as Init()'s own build (this BLAS's
        // topology never changes, only vertex positions do). RecordUpdateBLAS records its own
        // internal pre/post ACCELERATION_STRUCTURE_BUILD barriers around the actual
        // vkCmdBuildAccelerationStructuresKHR call -- see that function's own comment
        // (AccelerationStructure.h/.cpp). Caller (renderer::ClusterRenderPipeline) is responsible for
        // sequencing this method's own call before whichever RecordRefreshTLAS call executes this
        // same frame, on the SAME command buffer/queue -- see this method's own declaration comment.
        // =========================================================================================
        RecordUpdateBLAS(cmd, m_Device, m_BLASList[m_CreatureBlasListIndex].Handle(), m_CreatureBlasUpdateResources,
            m_CreatureSkinnedVertexBuffer.Handle(), sizeof(geometry::FallbackVertex), m_CreatureMaxVertex, /*vertexOffsetBytes=*/0,
            m_CreatureIndexBuffer, m_CreatureIndexOffsetBytes, m_CreatureTriangleCount);
    }

}
