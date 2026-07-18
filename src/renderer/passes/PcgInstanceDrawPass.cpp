#include "renderer/passes/PcgInstanceDrawPass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // std140 mirror of PcgInstanceDraw.frag's own PcgLightingParamsUBO -- 48 bytes (3 x 16-byte
        // vec3+scalar blocks). See that shader's own header comment for why this is a fixed,
        // Init()-time-only upload rather than a per-frame one (Phase 0.2 scope).
        struct PcgLightingParamsUBO {
            float sunDirX = 0.0f, sunDirY = 1.0f, sunDirZ = 0.0f;
            float sunIntensity = 1.0f;
            float sunColorR = 1.0f, sunColorG = 1.0f, sunColorB = 1.0f;
            float _pad0 = 0.0f;
            float ambientR = 0.1f, ambientG = 0.1f, ambientB = 0.12f;
            float _pad1 = 0.0f;
        };
        static_assert(sizeof(PcgLightingParamsUBO) == 48,
            "PcgLightingParamsUBO must match PcgInstanceDraw.frag's own UBO exactly (std140 layout)");

        // Transforms every one of a LOCAL-space AABB's 8 corners by `localToWorld` and returns the
        // resulting WORLD-space AABB -- the standard "transform every corner, take the new min/max"
        // technique (a rotated AABB is not itself an AABB, so no cheaper closed form exists without
        // assuming axis-aligned rotation).
        void TransformAABB(const maths::mat4& localToWorld, const maths::vec3& localMin, const maths::vec3& localMax,
            maths::vec3& outWorldMin, maths::vec3& outWorldMax) {
            maths::ResetAABB(outWorldMin, outWorldMax);
            for (int i = 0; i < 8; ++i) {
                maths::vec3 corner(
                    (i & 1) ? localMax.x : localMin.x,
                    (i & 2) ? localMax.y : localMin.y,
                    (i & 4) ? localMax.z : localMin.z);
                // mat4 * vec4(corner, 1.0), homogeneous transform (see maths::mat4's own column-major
                // multiply convention) -- w is always 1 for a point (not a direction), and this
                // engine's local-to-world matrices are always affine (no projective row), so the
                // result's w is always 1 too and can be safely dropped.
                float x = localToWorld.m[0] * corner.x + localToWorld.m[4] * corner.y + localToWorld.m[8] * corner.z + localToWorld.m[12];
                float y = localToWorld.m[1] * corner.x + localToWorld.m[5] * corner.y + localToWorld.m[9] * corner.z + localToWorld.m[13];
                float z = localToWorld.m[2] * corner.x + localToWorld.m[6] * corner.y + localToWorld.m[10] * corner.z + localToWorld.m[14];
                maths::ExpandAABB(outWorldMin, outWorldMax, maths::vec3(x, y, z));
            }
        }

        maths::vec3 TransformPoint(const maths::mat4& localToWorld, const maths::vec3& p) {
            return maths::vec3(
                localToWorld.m[0] * p.x + localToWorld.m[4] * p.y + localToWorld.m[8] * p.z + localToWorld.m[12],
                localToWorld.m[1] * p.x + localToWorld.m[5] * p.y + localToWorld.m[9] * p.z + localToWorld.m[13],
                localToWorld.m[2] * p.x + localToWorld.m[6] * p.y + localToWorld.m[10] * p.z + localToWorld.m[14]);
        }

    } // namespace

    void PcgInstanceDrawPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkBuffer compressedPhysicalPoolBuffer, const MaterialTable& materialTable,
        const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
        const maths::vec3& ambientColor,
        VkFormat colorFormat, VkFormat depthFormat,
        uint32_t maxInstances, uint32_t maxCandidateClusters) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_MaxInstances = maxInstances;
        m_MaxCandidateClusters = maxCandidateClusters;

        // =====================================================================================
        // STEP 1 -- CPU-side instance pool bookkeeping (LIFO free list, see this class' own header
        // comment) and every GPU buffer this pass owns.
        // =====================================================================================
        m_Slots.assign(maxInstances, InstanceSlotCPU{});
        m_Occupied.assign(maxInstances, false);
        m_FreeList.clear();
        m_FreeList.reserve(maxInstances);
        m_HighWaterMark = 0;
        m_LiveCount = 0;

        m_InstanceBuffer.Create(allocator, static_cast<VkDeviceSize>(maxInstances) * sizeof(GpuInstanceData),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_LocalBoundsBuffer.Create(allocator, static_cast<VkDeviceSize>(maxCandidateClusters) * sizeof(LocalBoundsGpu),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_MaterialParamsBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxMaterials) * sizeof(MaterialParameters),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_LightingParamsBuffer.Create(allocator, sizeof(PcgLightingParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        m_Culling.Init(device, allocator, maxCandidateClusters);

        // =====================================================================================
        // STEP 2 -- One-shot upload: this pass' own private copy of the material table + the fixed
        // lighting UBO. Neither is ever re-uploaded after this (see both buffers' own declaration
        // comments for why).
        // =====================================================================================
        {
            PcgLightingParamsUBO lighting{};
            maths::vec3 sunDirNorm = sunDirectionWorld.Normalize();
            lighting.sunDirX = sunDirNorm.x; lighting.sunDirY = sunDirNorm.y; lighting.sunDirZ = sunDirNorm.z;
            lighting.sunIntensity = sunIntensity;
            lighting.sunColorR = sunColor.x; lighting.sunColorG = sunColor.y; lighting.sunColorB = sunColor.z;
            lighting.ambientR = ambientColor.x; lighting.ambientG = ambientColor.y; lighting.ambientB = ambientColor.z;

            VkDeviceSize materialBytes = static_cast<VkDeviceSize>(kMaxMaterials) * sizeof(MaterialParameters);
            GpuBuffer staging;
            staging.Create(allocator, materialBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
            std::memcpy(staging.MappedData(), materialTable.params.data(), static_cast<size_t>(materialBytes));

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkBufferCopy materialCopy{ 0, 0, materialBytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_MaterialParamsBuffer.Handle(), 1, &materialCopy);
                vkCmdUpdateBuffer(cmd, m_LightingParamsBuffer.Handle(), 0, sizeof(lighting), &lighting);

                VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                depInfo.memoryBarrierCount = 1;
                depInfo.pMemoryBarriers = &memBarrier;
                vkCmdPipelineBarrier2(cmd, &depInfo);
                });

            staging.Destroy();
        }

        // =====================================================================================
        // STEP 3 -- Descriptor set layout (6 bindings, see PcgInstanceDraw.vert/.frag's own binding
        // comments): 0 = ClusterCullMetadataSSBO, 1 = CompressedClusterPoolSSBO (vertex-stage only,
        // both), 2 = PcgClusterLocalBoundsSSBO, 3 = PcgInstanceDataSSBO (vertex-stage only, both),
        // 4 = MaterialParamsSSBO, 5 = PcgLightingParamsUBO (fragment-stage only, both).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[6]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 6;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

            VkDescriptorPoolSize poolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 2;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_DescriptorPool;
            setAllocInfo.descriptorSetCount = 1;
            setAllocInfo.pSetLayouts = &m_SetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_DescriptorSet));

            VkDescriptorBufferInfo clusterMetadataInfo{ m_Culling.GetClusterMetadataBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo compressedPoolInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo localBoundsInfo{ m_LocalBoundsBuffer.Handle(), 0, m_LocalBoundsBuffer.Size() };
            VkDescriptorBufferInfo instanceInfo{ m_InstanceBuffer.Handle(), 0, m_InstanceBuffer.Size() };
            VkDescriptorBufferInfo materialInfo{ m_MaterialParamsBuffer.Handle(), 0, m_MaterialParamsBuffer.Size() };
            VkDescriptorBufferInfo lightingInfo{ m_LightingParamsBuffer.Handle(), 0, m_LightingParamsBuffer.Size() };

            VkWriteDescriptorSet writes[6]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetadataInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &localBoundsInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instanceInfo, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialInfo, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lightingInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 6, writes, 0, nullptr);
        }

        // =====================================================================================
        // STEP 4 -- graphics pipeline: no vertex-attribute input (PcgInstanceDraw.vert pulls every
        // attribute from SSBOs via gl_VertexIndex/gl_InstanceIndex, exactly like ClusterRaster.vert),
        // opaque (depth test+write ON, VK_COMPARE_OP_GREATER reversed-Z, matching this codebase's
        // site-wide convention -- see maths::mat4::PerspectiveVulkan's own comment), no blending.
        // =====================================================================================
        {
            VkPushConstantRange pushRange{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(CameraPushConstants::view) + sizeof(CameraPushConstants::proj) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

            VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PcgInstanceDraw.vert.spv");
            VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PcgInstanceDraw.frag.spv");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr };

            VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.blendEnable = VK_FALSE;
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.stencilTestEnable = VK_FALSE;

            VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            pipelineRendering.colorAttachmentCount = 1;
            pipelineRendering.pColorAttachmentFormats = &colorFormat;
            pipelineRendering.depthAttachmentFormat = depthFormat;

            VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = stages;
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = m_PipelineLayout;
            pipelineInfo.pNext = &pipelineRendering;

            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));

            vkDestroyShaderModule(m_Device, vertModule, nullptr);
            vkDestroyShaderModule(m_Device, fragModule, nullptr);
        }

        LOG_INFO(std::format("[PcgInstanceDrawPass] Initialized (maxInstances={}, maxCandidateClusters={}).",
            maxInstances, maxCandidateClusters));
    }

    void PcgInstanceDrawPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); m_Pipeline = VK_NULL_HANDLE; }
            if (m_PipelineLayout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); m_PipelineLayout = VK_NULL_HANDLE; }
            if (m_DescriptorPool != VK_NULL_HANDLE) { vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr); m_DescriptorPool = VK_NULL_HANDLE; }
            if (m_SetLayout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr); m_SetLayout = VK_NULL_HANDLE; }
        }
        m_DescriptorSet = VK_NULL_HANDLE;

        m_Culling.Shutdown();
        m_InstanceBuffer.Destroy();
        m_LocalBoundsBuffer.Destroy();
        m_MaterialParamsBuffer.Destroy();
        m_LightingParamsBuffer.Destroy();

        m_Slots.clear();
        m_Occupied.clear();
        m_FreeList.clear();
        m_HighWaterMark = 0;
        m_LiveCount = 0;
        m_MaxInstances = 0;
        m_MaxCandidateClusters = 0;
        m_LastCandidateClusterCount = 0;

        m_Device = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
    }

    uint32_t PcgInstanceDrawPass::AcquireInstance(uint32_t meshID, uint32_t materialID,
        const maths::vec3& position, const maths::quat& rotation, const maths::vec3& scale) {
        uint32_t index;
        if (!m_FreeList.empty()) {
            index = m_FreeList.back();
            m_FreeList.pop_back();
        } else if (m_HighWaterMark < m_MaxInstances) {
            index = m_HighWaterMark++;
        } else {
            LOG_ERROR("[PcgInstanceDrawPass] AcquireInstance() FAILED: pool exhausted.");
            return kInvalidInstance;
        }

        m_Occupied[index] = true;
        m_Slots[index] = InstanceSlotCPU{ meshID, materialID, position, rotation, scale };
        ++m_LiveCount;
        return index;
    }

    void PcgInstanceDrawPass::ReleaseInstance(uint32_t instanceSlot) {
        if (instanceSlot >= m_HighWaterMark || !m_Occupied[instanceSlot]) {
            LOG_ERROR(std::format("[PcgInstanceDrawPass] ReleaseInstance({}) FAILED: not currently acquired.", instanceSlot));
            return;
        }
        m_Occupied[instanceSlot] = false;
        m_Slots[instanceSlot] = InstanceSlotCPU{};
        m_FreeList.push_back(instanceSlot);
        --m_LiveCount;
    }

    uint32_t PcgInstanceDrawPass::UploadInstances(VkCommandPool commandPool, VkQueue queue,
        const std::vector<geometry::ClusterIndexEntry>& indexEntries,
        const std::vector<geometry::DAGNodeEntry>& dagEntries,
        const GpuGeometryPagePool& pagePool) {
        std::vector<ClusterCullMetadata> candidateMetadata;
        std::vector<LocalBoundsGpu> localBounds;
        std::vector<GpuInstanceData> instanceData(m_MaxInstances);

        const size_t clusterCount = std::min(indexEntries.size(), dagEntries.size());

        for (uint32_t slot = 0; slot < m_HighWaterMark; ++slot) {
            if (!m_Occupied[slot]) {
                continue;
            }
            const InstanceSlotCPU& inst = m_Slots[slot];

            // This meshID's own baked local-space pivot -- geometry::ClusterIndexEntry::boundsMin/
            // boundsMax are NOT necessarily authored relative to (0,0,0): World Partition streaming
            // archetypes in particular are baked at their own widely-separated "parking position"
            // (see renderer::VulkanContext::StreamingSlotParkPosition's own comment -- e.g. around
            // (-400,-400), so geom_autosmooth.comp's cross-mesh normal weld never fires between two
            // different parked archetypes sharing the origin). This pass has no knowledge of that
            // (or any other entity-specific) baking convention, so instead of assuming a clean
            // local origin, it derives the correct pivot directly from the baked geometry's own
            // aggregate AABB across every leaf cluster of this meshID -- a single scan, cheap at
            // this pass' own "a handful of instances, rebuilt on set-change, not every frame" scale.
            maths::vec3 meshLocalMin, meshLocalMax;
            maths::ResetAABB(meshLocalMin, meshLocalMax);
            for (size_t p = 0; p < clusterCount; ++p) {
                const geometry::ClusterIndexEntry& probeEntry = indexEntries[p];
                const geometry::DAGNodeEntry& probeNode = dagEntries[p];
                if (probeEntry.entityID != inst.meshID || probeNode.level != 0) {
                    continue;
                }
                maths::ExpandAABB(meshLocalMin, meshLocalMax, maths::vec3(probeEntry.boundsMin[0], probeEntry.boundsMin[1], probeEntry.boundsMin[2]));
                maths::ExpandAABB(meshLocalMin, meshLocalMax, maths::vec3(probeEntry.boundsMax[0], probeEntry.boundsMax[1], probeEntry.boundsMax[2]));
            }
            maths::vec3 meshLocalPivot = maths::AABBCenter(meshLocalMin, meshLocalMax);

            // Compose: translate the DECODED-AS-AUTHORED local position back to be relative to its
            // own mesh's pivot FIRST (Translate(-meshLocalPivot)), THEN apply this instance's own
            // scale/rotation/position -- exactly mirroring struct_custo.glsl's established
            // "rotation*(restPos-center)" convention every other entity in this engine already uses
            // for the identical reason (see ClusterRaster.vert's own EntityTransform application).
            maths::mat4 localToWorld = maths::mat4::Translate(inst.position) * maths::mat4::FromQuat(inst.rotation)
                * maths::mat4::Scale(inst.scale) * maths::mat4::Translate(meshLocalPivot * -1.0f);
            instanceData[slot].localToWorld = localToWorld;
            instanceData[slot].materialID = inst.materialID;

            float maxScaleComponent = std::max({ std::abs(inst.scale.x), std::abs(inst.scale.y), std::abs(inst.scale.z) });

            for (size_t k = 0; k < clusterCount; ++k) {
                const geometry::ClusterIndexEntry& entry = indexEntries[k];
                const geometry::DAGNodeEntry& node = dagEntries[k];
                if (entry.entityID != inst.meshID || node.level != 0) {
                    continue; // Only this instance's own LEAF (full-detail) clusters -- see this class' own header comment on LOD scope.
                }

                uint32_t physicalPageIndex = pagePool.GetPhysicalPageIndex(entry.virtualAddress);
                if (physicalPageIndex == geometry::kInvalidPhysicalPage) {
                    LOG_ERROR(std::format(
                        "[PcgInstanceDrawPass] UploadInstances(): cluster {} (meshID {}) is not resident -- skipped. "
                        "This should never happen for content streamed in at ClusterRenderPipeline::Init() time.",
                        entry.clusterID, inst.meshID));
                    continue;
                }

                if (candidateMetadata.size() >= m_MaxCandidateClusters) {
                    LOG_ERROR(std::format(
                        "[PcgInstanceDrawPass] UploadInstances(): candidate cluster count exceeds maxCandidateClusters ({}) -- "
                        "remaining clusters dropped.", m_MaxCandidateClusters));
                    break;
                }

                maths::vec3 localMin(entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2]);
                maths::vec3 localMax(entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2]);

                ClusterCullMetadata meta{};
                TransformAABB(localToWorld, localMin, localMax, meta.boundsMin, meta.boundsMax);
                meta.sphereCenter = TransformPoint(localToWorld, maths::vec3(entry.sphereCenter[0], entry.sphereCenter[1], entry.sphereCenter[2]));
                meta.sphereRadius = entry.sphereRadius * maxScaleComponent;

                // Cone axis: rotation-only transform (a direction, not a point) -- see
                // renderer::ClusterCullingPass::ClusterCullMetadata's own comment for the int8 ->
                // normalized-float widening (divide by 127), the exact inverse of
                // geometry::VirtualGeometryCacheTest.cpp's BuildIndexEntry quantization.
                maths::vec3 localConeAxis(entry.coneAxisX / 127.0f, entry.coneAxisY / 127.0f, entry.coneAxisZ / 127.0f);
                meta.coneAxis = inst.rotation.RotateVector(localConeAxis).Normalize();
                meta.coneCutoff = entry.coneCutoff / 127.0f;

                meta.indexCount = entry.indexCount;
                meta.firstIndex = physicalPageIndex * geometry::kMaxClusterIndices;
                meta.vertexOffset = physicalPageIndex * geometry::kMaxClusterVertices;
                meta.clusterID = entry.clusterID;
                meta.maxWPOAmplitude = 0.0f; // PCG instances do not sway -- see PcgInstanceDraw.vert's own header comment.
                meta.maskTextureIndex = geometry::kInvalidMaskTextureIndex; // Opaque-only, Phase 0.2 scope.
                // Repurposed to carry this instance's OWN slot index, NOT the real scene meshID --
                // see PcgInstanceDraw.vert's own header comment for why this is safe.
                meta.entityID = slot;
                meta.materialID = inst.materialID;

                candidateMetadata.push_back(meta);
                localBounds.push_back(LocalBoundsGpu{ localMin, 0.0f, localMax, 0.0f });
            }
        }

        m_LastCandidateClusterCount = static_cast<uint32_t>(candidateMetadata.size());

        // --- Upload the per-instance transform/material buffer (full maxInstances-length array;
        // unoccupied slots' entries are never referenced by any candidate cluster, see this
        // method's own comment, so their content is irrelevant). ---
        {
            VkDeviceSize bytes = static_cast<VkDeviceSize>(m_MaxInstances) * sizeof(GpuInstanceData);
            GpuBuffer staging;
            staging.Create(m_Allocator, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
            std::memcpy(staging.MappedData(), instanceData.data(), static_cast<size_t>(bytes));

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkBufferCopy copyRegion{ 0, 0, bytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_InstanceBuffer.Handle(), 1, &copyRegion);

                VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
                memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                depInfo.memoryBarrierCount = 1;
                depInfo.pMemoryBarriers = &memBarrier;
                vkCmdPipelineBarrier2(cmd, &depInfo);
                });

            staging.Destroy();
        }

        // --- Upload the LOCAL-space bounds buffer, index-aligned with candidateMetadata. ---
        if (!localBounds.empty()) {
            VkDeviceSize bytes = static_cast<VkDeviceSize>(localBounds.size()) * sizeof(LocalBoundsGpu);
            GpuBuffer staging;
            staging.Create(m_Allocator, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
            std::memcpy(staging.MappedData(), localBounds.data(), static_cast<size_t>(bytes));

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkBufferCopy copyRegion{ 0, 0, bytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_LocalBoundsBuffer.Handle(), 1, &copyRegion);

                VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
                memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                depInfo.memoryBarrierCount = 1;
                depInfo.pMemoryBarriers = &memBarrier;
                vkCmdPipelineBarrier2(cmd, &depInfo);
                });

            staging.Destroy();
        }

        // --- Publish the candidate list itself to renderer::ClusterCullingPass (its own staging
        // upload, see that class' own UploadClusterMetadata() comment). ---
        m_Culling.UploadClusterMetadata(commandPool, queue, candidateMetadata);

        LOG_INFO(std::format("[PcgInstanceDrawPass] UploadInstances(): {} live instance(s), {} candidate cluster(s) uploaded.",
            m_LiveCount, m_LastCandidateClusterCount));
        return m_LastCandidateClusterCount;
    }

    void PcgInstanceDrawPass::RecordClear(VkCommandBuffer cmd) {
        m_Culling.RecordClear(cmd);
    }

    void PcgInstanceDrawPass::RecordCull(VkCommandBuffer cmd, const ClusterCullViewParams& viewParams) {
        m_Culling.RecordCull(cmd, viewParams, m_LastCandidateClusterCount);
    }

    void PcgInstanceDrawPass::RecordDraw(VkCommandBuffer cmd, const CameraPushConstants& camera, VkExtent2D renderExtent,
        VkBuffer decompressedIndexPoolBuffer) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

        // Only {view, proj} -- see this pass' own Init() comment on why the push-constant range is
        // exactly 128 bytes (the Vulkan-guaranteed minimum), not the full CameraPushConstants struct
        // (whose Debug-only trailing fields would overflow that budget).
        struct { maths::mat4 view; maths::mat4 proj; } pushData{ camera.view, camera.proj };
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), &pushData);

        vkCmdBindIndexBuffer(cmd, decompressedIndexPoolBuffer, 0, VK_INDEX_TYPE_UINT32);

        VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(renderExtent.width), static_cast<float>(renderExtent.height), 0.0f, 1.0f };
        VkRect2D scissor{ { 0, 0 }, renderExtent };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDrawIndexedIndirectCount(cmd, m_Culling.GetIndirectCommandBuffer(), 0,
            m_Culling.GetDrawCountBuffer(), 0, m_MaxCandidateClusters, sizeof(VkDrawIndexedIndirectCommand));
    }

}
