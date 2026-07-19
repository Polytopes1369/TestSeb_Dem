#include "renderer/passes/VegetationScatterPass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <vector>

#include "core/Logger.h"
#include "core/EngineConfig.h"
#include "renderer/passes/ClusterCullingPass.h"      // ExtractFrustumPlanes
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/passes/WorldProbeGridPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // --- Archetype base-mesh tuning (world units at instance scale == 1). ---
        constexpr uint32_t kGrassBladeCount = 5u;
        constexpr float    kGrassBladeHeight = 0.9f;
        constexpr float    kGrassBladeHalfWidth = 0.11f;
        constexpr float    kGrassBendAmount = 0.12f;

        constexpr uint32_t kBushRings = 6u, kBushSectors = 8u;
        constexpr float    kBushRadius = 0.55f, kBushNoiseAmp = 0.35f, kBushNoiseFreq = 2.5f, kBushFlatten = 0.25f;

        constexpr uint32_t kRockRings = 5u, kRockSectors = 7u;
        constexpr float    kRockRadius = 0.45f, kRockNoiseAmp = 0.55f, kRockNoiseFreq = 3.5f, kRockFlatten = 0.5f;

        // Per-archetype bounding-sphere radius (at scale == 1), sphere centered one radius above the
        // ground-anchored instance position -- generous enough to never pop a valid instance out at a
        // frustum/HZB edge. Order: grass, bush, rock.
        constexpr float kArchetypeBoundingRadius[VegetationScatterPass::kArchetypeCount] = { 0.8f, 0.9f, 0.75f };

        // Mirror of geom_scatter_grass.comp's push constant.
        struct GrassGenParams {
            uint32_t bladeCount;
            float bladeHeight, bladeHalfWidth, bendAmount;
            uint32_t seed, meshID;
            float materialID;
            uint32_t vertexOffset, indexOffset;
        };

        // Mirror of geom_scatter_blob.comp's push constant.
        struct BlobGenParams {
            float radius, noiseAmplitude, noiseFrequency;
            uint32_t rings, sectors, seed, meshID;
            float materialID;
            uint32_t vertexOffset, indexOffset;
            float flattenBottom;
        };

        // Mirror of VegetationScatterGen.comp's push constant.
        struct ScatterGenParams {
            float originX, originZ, cellSize;
            uint32_t gridDim;
            float grassDensity, bushDensity, rockDensity;
            uint32_t maxInstances, seed;
            float _pad0, _pad1;
        };

        // Mirror of VegetationInstanceCull.comp's CullParams UBO (std140, 224 bytes).
        struct CullParamsUBO {
            float frustumPlanes[24]; // 6 * vec4
            float cameraPos[4];
            maths::mat4 viewProj;    // 64 bytes
            float hzbMip0Size[2];
            float hzbMipCount;
            uint32_t enableOcclusion;
            float archetypeRadius[4];
            uint32_t maxInstances;
            uint32_t _pad0, _pad1, _pad2;
        };
        static_assert(sizeof(CullParamsUBO) == 224, "CullParamsUBO must match VegetationInstanceCull.comp's CullParams (std140).");

        // Mirror of VegetationInstanced.vert/.frag's RenderParamsUBO (std140, 112 bytes).
        struct RenderParamsUBO {
            maths::mat4 viewProj;
            float cameraPos[3]; float _pad0;
            float sunDirection[3]; float sunIntensity;
            float sunColor[3]; float _pad1;
        };
        static_assert(sizeof(RenderParamsUBO) == 112, "RenderParamsUBO must match VegetationInstanced.vert/.frag (std140).");

        // Mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO (std140, 64 bytes -- F1,
        // "Lumen Lite": one (origin, spacing) pair per renderer::WorldProbeGridPass::kLevelCount
        // clipmap level) -- identical to ParticleSystemPass's own copy.
        struct WorldProbeGridParamsUBO {
            struct Level {
                float originX = 0.0f, originY = 0.0f, originZ = 0.0f;
                float spacing = 0.0f;
            };
            Level levels[3];
            float gridResolution = 0.0f;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(WorldProbeGridParamsUBO) == 64, "WorldProbeGridParamsUBO must match world_probe_sampling.glsl (std140).");

        // Mirror of VegetationInstanced.vert/.frag's push constant (16 bytes).
        struct VegRenderPushConstants {
            uint32_t archetypeSegmentBase;
            uint32_t archetype;
            uint32_t _pad0, _pad1;
        };

    } // namespace

    bool VegetationScatterPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
        VkImageView hzbView, VkExtent2D hzbMip0Extent, uint32_t hzbMipCount,
        VkFormat colorFormat, VkFormat depthFormat) {
        // Self-reinit (see ShadowMapPass's own migration comment for the identical pattern):
        // Shutdown() clears m_Device/m_Allocator, which RenderPass<VegetationScatterPass>::Init()
        // already set to this call's values just before invoking this function -- restore them.
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_CommandPool = commandPool;
        m_Queue = queue;
        m_HZBMip0Extent = hzbMip0Extent;
        m_HZBMipCount = hzbMipCount;

        // =====================================================================================
        // STEP 1 -- Compute the 3 archetypes' vertex/index footprints and lay them out back-to-back
        // in the combined vertex/index buffers (order: grass, bush, rock). vertexOffset for the draw
        // is always 0 -- the geom generators bake GLOBAL vertex indices (like every geom_*.comp), so
        // the index buffer already carries absolute indices.
        // =====================================================================================
        const uint32_t grassVerts = kGrassBladeCount * 4u;
        const uint32_t grassIndices = kGrassBladeCount * 6u;
        const uint32_t bushVerts = (kBushRings + 1u) * (kBushSectors + 1u);
        const uint32_t bushIndices = kBushRings * kBushSectors * 6u;
        const uint32_t rockVerts = (kRockRings + 1u) * (kRockSectors + 1u);
        const uint32_t rockIndices = kRockRings * kRockSectors * 6u;

        const uint32_t grassVertexBase = 0u;
        const uint32_t bushVertexBase = grassVertexBase + grassVerts;
        const uint32_t rockVertexBase = bushVertexBase + bushVerts;
        const uint32_t totalVerts = rockVertexBase + rockVerts;

        m_Ranges[0] = { 0u, grassIndices };                                  // grass
        m_Ranges[1] = { grassIndices, bushIndices };                         // bush
        m_Ranges[2] = { grassIndices + bushIndices, rockIndices };           // rock
        const uint32_t totalIndices = grassIndices + bushIndices + rockIndices;

        // =====================================================================================
        // STEP 2 -- Buffer allocation. Every buffer is GPU_ONLY (host-touched only via one-shot
        // staged compute at bake time / vkCmdUpdateBuffer per-frame), same convention as
        // ParticleSystemPass's own GPU_ONLY buffers.
        // =====================================================================================
        m_ArchetypeVertexBuffer.Create(allocator, static_cast<VkDeviceSize>(totalVerts) * sizeof(float) * 12u, // struct_custo Vertex == 48 bytes == 12 floats
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ArchetypeIndexBuffer.Create(allocator, static_cast<VkDeviceSize>(totalIndices) * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_InstanceBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxInstances) * sizeof(GpuVegetationInstance),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_CounterBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        m_VisibleIndexBuffer.Create(allocator, static_cast<VkDeviceSize>(kArchetypeCount) * kMaxInstances * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_IndirectCommandBuffer.Create(allocator, static_cast<VkDeviceSize>(kArchetypeCount) * sizeof(VkDrawIndexedIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        m_CullParamsBuffer.Create(allocator, sizeof(CullParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_RenderParamsBuffer.Create(allocator, sizeof(RenderParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_WorldProbeGridParamsBuffer.Create(allocator, sizeof(WorldProbeGridParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_CountReadbackBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        RegisterResource([this] {
            m_ArchetypeVertexBuffer.Destroy();
            m_ArchetypeIndexBuffer.Destroy();
            m_InstanceBuffer.Destroy();
            m_CounterBuffer.Destroy();
            m_VisibleIndexBuffer.Destroy();
            m_IndirectCommandBuffer.Destroy();
            m_CullParamsBuffer.Destroy();
            m_RenderParamsBuffer.Destroy();
            m_WorldProbeGridParamsBuffer.Destroy();
            m_CountReadbackBuffer.Destroy();
        });

        // Initialize the indirect commands once (instanceCount 0) so a frame that never culls (e.g.
        // zero instances) still issues a well-defined zero-instance draw rather than reading garbage.
        {
            std::array<VkDrawIndexedIndirectCommand, kArchetypeCount> initialCmds{};
            for (uint32_t a = 0; a < kArchetypeCount; ++a) {
                initialCmds[a] = { m_Ranges[a].indexCount, 0u, m_Ranges[a].firstIndex, 0u, 0u };
            }
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdUpdateBuffer(cmd, m_IndirectCommandBuffer.Handle(), 0,
                    sizeof(VkDrawIndexedIndirectCommand) * kArchetypeCount, initialCmds.data());
            });
        }

        // =====================================================================================
        // STEP 3 -- Descriptor set layouts + one shared pool + every set.
        // =====================================================================================
        auto makeLayout = [&](std::vector<VkDescriptorSetLayoutBinding> bindings) {
            VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            info.bindingCount = static_cast<uint32_t>(bindings.size());
            info.pBindings = bindings.data();
            VkDescriptorSetLayout layout = VK_NULL_HANDLE;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &info, nullptr, &layout));
            return layout;
        };

        m_GeomSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr } });
        m_ScatterGenSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr } });
        m_CullSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr } });
        m_RenderInstanceSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr } });
        m_RenderParamsSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr } });
        m_LightingSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },          // shadow page table
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },          // shadow feedback
            { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },  // shadow atlas
            { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },          // shadow sun levels
            { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, WorldProbeGridPass::kLevelCount, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },  // world probe grid (F1: multi-level)
            { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr } });       // world probe grid params

        VkDescriptorPoolSize poolSizes[3] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
            // F1 ("Lumen Lite"): +2 vs. pre-F1 (was 4: HZB + shadow atlas + 1 world probe grid sampler,
            // now WorldProbeGridPass::kLevelCount=3 world probe grid samplers).
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 + WorldProbeGridPass::kLevelCount },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 6;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        auto allocSet = [&](VkDescriptorSetLayout layout) {
            VkDescriptorSetAllocateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            info.descriptorPool = m_DescriptorPool;
            info.descriptorSetCount = 1;
            info.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &info, &set));
            return set;
        };
        m_GeomSet = allocSet(m_GeomSetLayout);
        m_ScatterGenSet = allocSet(m_ScatterGenSetLayout);
        m_CullSet = allocSet(m_CullSetLayout);
        m_RenderInstanceSet = allocSet(m_RenderInstanceSetLayout);
        m_RenderParamsSet = allocSet(m_RenderParamsSetLayout);
        m_LightingSet = allocSet(m_LightingSetLayout);
        RegisterResource([this] {
            // Destroying the pool implicitly frees all 6 sets allocated from it.
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_GeomSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_ScatterGenSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_CullSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_RenderInstanceSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_RenderParamsSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_LightingSetLayout, nullptr);
            m_DescriptorPool = VK_NULL_HANDLE;
            m_GeomSetLayout = VK_NULL_HANDLE; m_ScatterGenSetLayout = VK_NULL_HANDLE; m_CullSetLayout = VK_NULL_HANDLE;
            m_RenderInstanceSetLayout = VK_NULL_HANDLE; m_RenderParamsSetLayout = VK_NULL_HANDLE; m_LightingSetLayout = VK_NULL_HANDLE;
            m_GeomSet = VK_NULL_HANDLE; m_ScatterGenSet = VK_NULL_HANDLE; m_CullSet = VK_NULL_HANDLE;
            m_RenderInstanceSet = VK_NULL_HANDLE; m_RenderParamsSet = VK_NULL_HANDLE; m_LightingSet = VK_NULL_HANDLE;
        });

        // HZB sampler: nearest / nearest-mip across the whole pyramid (see hzb_occlusion.glsl).
        m_HZBSampler = VulkanUtils::CreateNearestSampler(m_Device, static_cast<float>(m_HZBMipCount > 0 ? m_HZBMipCount - 1u : 0u));
        RegisterResource([this] { vkDestroySampler(m_Device, m_HZBSampler, nullptr); m_HZBSampler = VK_NULL_HANDLE; });

        // --- Descriptor writes (every binding here is stable for this pass' lifetime). ---
        VkDescriptorBufferInfo vtxInfo{ m_ArchetypeVertexBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo idxInfo{ m_ArchetypeIndexBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo instInfo{ m_InstanceBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo counterInfo{ m_CounterBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo visibleInfo{ m_VisibleIndexBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo indirectInfo{ m_IndirectCommandBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo cullParamsInfo{ m_CullParamsBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo renderParamsInfo{ m_RenderParamsBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo hzbInfo{ m_HZBSampler, hzbView, VK_IMAGE_LAYOUT_GENERAL };

        VkDescriptorBufferInfo pageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo feedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo atlasInfo{ vsm.GetPhysicalAtlasSampler(), vsm.GetPhysicalAtlasView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo sunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };
        // F1 ("Lumen Lite"): one VkDescriptorImageInfo per renderer::WorldProbeGridPass::kLevelCount
        // clipmap level.
        VkDescriptorImageInfo probeGridInfos[WorldProbeGridPass::kLevelCount]{};
        for (uint32_t level = 0; level < WorldProbeGridPass::kLevelCount; ++level) {
            probeGridInfos[level] = { worldProbes.GetGridSampler(), worldProbes.GetGridView(level), VK_IMAGE_LAYOUT_GENERAL };
        }
        VkDescriptorBufferInfo gridParamsInfo{ m_WorldProbeGridParamsBuffer.Handle(), 0, VK_WHOLE_SIZE };

        auto bufWrite = [](VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* bi) {
            return VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, type, nullptr, bi, nullptr };
        };
        auto imgWrite = [](VkDescriptorSet set, uint32_t binding, const VkDescriptorImageInfo* ii) {
            return VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ii, nullptr, nullptr };
        };
        auto imgArrayWrite = [](VkDescriptorSet set, uint32_t binding, uint32_t count, const VkDescriptorImageInfo* ii) {
            return VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, count, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ii, nullptr, nullptr };
        };

        std::vector<VkWriteDescriptorSet> writes = {
            bufWrite(m_GeomSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &vtxInfo),
            bufWrite(m_GeomSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &idxInfo),

            bufWrite(m_ScatterGenSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &instInfo),
            bufWrite(m_ScatterGenSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &counterInfo),

            bufWrite(m_CullSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &instInfo),
            bufWrite(m_CullSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &visibleInfo),
            bufWrite(m_CullSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &indirectInfo),
            imgWrite(m_CullSet, 3, &hzbInfo),
            bufWrite(m_CullSet, 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &cullParamsInfo),
            bufWrite(m_CullSet, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &counterInfo),

            bufWrite(m_RenderInstanceSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &vtxInfo),
            bufWrite(m_RenderInstanceSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &instInfo),
            bufWrite(m_RenderInstanceSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &visibleInfo),

            bufWrite(m_RenderParamsSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &renderParamsInfo),

            bufWrite(m_LightingSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &pageTableInfo),
            bufWrite(m_LightingSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &feedbackInfo),
            imgWrite(m_LightingSet, 2, &atlasInfo),
            bufWrite(m_LightingSet, 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &sunLevelsInfo),
            imgArrayWrite(m_LightingSet, 4, WorldProbeGridPass::kLevelCount, probeGridInfos),
            bufWrite(m_LightingSet, 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &gridParamsInfo),
        };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // One-time World Probe grid params upload (gridOrigin is unused by SampleWorldProbeGrid's own
        // absolute-index addressing -- see world_probe_sampling.glsl's comment -- so a static upload
        // is correct here, same simplification ParticleSystemPass uses).
        {
            WorldProbeGridParamsUBO gridParams{};
            for (uint32_t level = 0; level < WorldProbeGridPass::kLevelCount; ++level) {
                const maths::vec3& origin = worldProbes.GetGridOriginWorld(level);
                gridParams.levels[level].originX = origin.x;
                gridParams.levels[level].originY = origin.y;
                gridParams.levels[level].originZ = origin.z;
                gridParams.levels[level].spacing = WorldProbeGridPass::GetLevelSpacing(level);
            }
            gridParams.gridResolution = static_cast<float>(WorldProbeGridPass::kGridResolution);
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdUpdateBuffer(cmd, m_WorldProbeGridParamsBuffer.Handle(), 0, sizeof(gridParams), &gridParams);
            });
        }

        // =====================================================================================
        // STEP 4 -- Pipelines.
        // =====================================================================================
        // Geometry generators (grass + blob) share one layout: set 0 = vtx/idx SSBOs, a 64-byte
        // compute push range covering both generators' push structs.
        {
            VkPushConstantRange geomPush{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 64 };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 1;
            li.pSetLayouts = &m_GeomSetLayout;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &geomPush;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &li, nullptr, &m_GeomPipelineLayout));

            VkShaderModule grassModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/geom_scatter_grass.comp.spv");
            m_GrassGenPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_GeomPipelineLayout, grassModule);
            vkDestroyShaderModule(m_Device, grassModule, nullptr);

            VkShaderModule blobModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/geom_scatter_blob.comp.spv");
            m_BlobGenPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_GeomPipelineLayout, blobModule);
            vkDestroyShaderModule(m_Device, blobModule, nullptr);
            RegisterResource([this] {
                vkDestroyPipeline(m_Device, m_GrassGenPipeline, nullptr);
                vkDestroyPipeline(m_Device, m_BlobGenPipeline, nullptr);
                vkDestroyPipelineLayout(m_Device, m_GeomPipelineLayout, nullptr);
                m_GrassGenPipeline = VK_NULL_HANDLE; m_BlobGenPipeline = VK_NULL_HANDLE; m_GeomPipelineLayout = VK_NULL_HANDLE;
            });
        }
        // Scatter generator.
        {
            VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ScatterGenParams) };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 1;
            li.pSetLayouts = &m_ScatterGenSetLayout;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &li, nullptr, &m_ScatterGenPipelineLayout));
            VkShaderModule module = VulkanPipeline::LoadShaderModule(m_Device, "shaders/VegetationScatterGen.comp.spv");
            m_ScatterGenPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ScatterGenPipelineLayout, module);
            vkDestroyShaderModule(m_Device, module, nullptr);
            RegisterResource([this] {
                vkDestroyPipeline(m_Device, m_ScatterGenPipeline, nullptr);
                vkDestroyPipelineLayout(m_Device, m_ScatterGenPipelineLayout, nullptr);
                m_ScatterGenPipeline = VK_NULL_HANDLE; m_ScatterGenPipelineLayout = VK_NULL_HANDLE;
            });
        }
        // Per-instance cull.
        {
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 1;
            li.pSetLayouts = &m_CullSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &li, nullptr, &m_CullPipelineLayout));
            VkShaderModule module = VulkanPipeline::LoadShaderModule(m_Device, "shaders/VegetationInstanceCull.comp.spv");
            m_CullPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_CullPipelineLayout, module);
            vkDestroyShaderModule(m_Device, module, nullptr);
            RegisterResource([this] {
                vkDestroyPipeline(m_Device, m_CullPipeline, nullptr);
                vkDestroyPipelineLayout(m_Device, m_CullPipelineLayout, nullptr);
                m_CullPipeline = VK_NULL_HANDLE; m_CullPipelineLayout = VK_NULL_HANDLE;
            });
        }
        // Instanced forward render pipeline (+ Debug wireframe variant).
        {
            VkDescriptorSetLayout setLayouts[3] = { m_RenderInstanceSetLayout, m_RenderParamsSetLayout, m_LightingSetLayout };
            VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(VegRenderPushConstants) };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 3;
            li.pSetLayouts = setLayouts;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &li, nullptr, &m_RenderPipelineLayout));

            VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/VegetationInstanced.vert.spv");
            VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/VegetationInstanced.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr };

            // No bound vertex attribute buffer -- vertices are fetched from an SSBO via gl_VertexIndex.
            VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
            VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            // No backface culling: grass cards are viewed from both sides (double-sided). The minor
            // bush/rock backface overdraw is hidden by the depth test anyway.
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Opaque: no blend.
            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;
            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            pipelineRendering.colorAttachmentCount = 1;
            pipelineRendering.pColorAttachmentFormats = &colorFormat;
            pipelineRendering.depthAttachmentFormat = depthFormat;

            // Opaque, depth-tested (reversed-Z) AND depth-writing (unlike glass/particles) -- mirrors
            // TessellationPass's own depthStencil state.
            VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;

            VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            pipelineInfo.pNext = &pipelineRendering;
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
            pipelineInfo.layout = m_RenderPipelineLayout;
            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_RenderPipeline));

#ifndef NDEBUG
            // Debug-only wireframe/bounds visualization: same pipeline, VK_POLYGON_MODE_LINE, and
            // depth-write OFF so it overlays cleanly without polluting depth for later forward passes.
            VkPipelineRasterizationStateCreateInfo wireRaster = rasterizer;
            wireRaster.polygonMode = VK_POLYGON_MODE_LINE;
            VkPipelineDepthStencilStateCreateInfo wireDepth = depthStencil;
            wireDepth.depthWriteEnable = VK_FALSE;
            VkGraphicsPipelineCreateInfo wirePipelineInfo = pipelineInfo;
            wirePipelineInfo.pRasterizationState = &wireRaster;
            wirePipelineInfo.pDepthStencilState = &wireDepth;
            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &wirePipelineInfo, nullptr, &m_WireframePipeline));
#endif

            vkDestroyShaderModule(m_Device, vertModule, nullptr);
            vkDestroyShaderModule(m_Device, fragModule, nullptr);

            RegisterResource([this] {
#ifndef NDEBUG
                vkDestroyPipeline(m_Device, m_WireframePipeline, nullptr);
                m_WireframePipeline = VK_NULL_HANDLE;
#endif
                vkDestroyPipeline(m_Device, m_RenderPipeline, nullptr);
                vkDestroyPipelineLayout(m_Device, m_RenderPipelineLayout, nullptr);
                m_RenderPipeline = VK_NULL_HANDLE; m_RenderPipelineLayout = VK_NULL_HANDLE;
            });
        }

        // =====================================================================================
        // STEP 5 -- Bake the 3 archetype base meshes, then generate the initial scatter.
        // =====================================================================================
        BakeArchetypeGeometry();
        GenerateScatter();
        // GetInstanceCount() is a public getter -- register its reset so a Shutdown() not
        // immediately followed by Init() doesn't leave a caller reading a stale count.
        RegisterResource([this] { m_InstanceCount = 0; });

        LOG_INFO(std::format("[VegetationScatterPass] Initialized: {} archetype verts / {} indices, {} instances scattered (cap {}).",
            totalVerts, totalIndices, m_InstanceCount, kMaxInstances));
        return true;
    }

    void VegetationScatterPass::BakeArchetypeGeometry() {
        const uint32_t grassVerts = kGrassBladeCount * 4u;
        const uint32_t bushVertexBase = grassVerts;
        const uint32_t bushVerts = (kBushRings + 1u) * (kBushSectors + 1u);
        const uint32_t rockVertexBase = bushVertexBase + bushVerts;

        VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_Queue, [&](VkCommandBuffer cmd) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GeomPipelineLayout, 0, 1, &m_GeomSet, 0, nullptr);

            // Grass tuft (region: vertex base 0, index base 0).
            GrassGenParams gp{};
            gp.bladeCount = kGrassBladeCount;
            gp.bladeHeight = kGrassBladeHeight;
            gp.bladeHalfWidth = kGrassBladeHalfWidth;
            gp.bendAmount = kGrassBendAmount;
            gp.seed = 101u;
            gp.meshID = 0u;
            gp.materialID = 0.0f;
            gp.vertexOffset = 0u;
            gp.indexOffset = m_Ranges[0].firstIndex;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GrassGenPipeline);
            vkCmdPushConstants(cmd, m_GeomPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gp), &gp);
            vkCmdDispatch(cmd, (kGrassBladeCount + 63u) / 64u, 1, 1);

            // Bush blob.
            BlobGenParams bp{};
            bp.radius = kBushRadius; bp.noiseAmplitude = kBushNoiseAmp; bp.noiseFrequency = kBushNoiseFreq;
            bp.rings = kBushRings; bp.sectors = kBushSectors; bp.seed = 202u; bp.meshID = 1u; bp.materialID = 0.0f;
            bp.vertexOffset = bushVertexBase; bp.indexOffset = m_Ranges[1].firstIndex; bp.flattenBottom = kBushFlatten;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BlobGenPipeline);
            vkCmdPushConstants(cmd, m_GeomPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bp), &bp);
            vkCmdDispatch(cmd, (kBushRings + 1u + 7u) / 8u, (kBushSectors + 1u + 7u) / 8u, 1);

            // Rock blob (reuses the blob pipeline, distinct params/seed for variety).
            BlobGenParams rp{};
            rp.radius = kRockRadius; rp.noiseAmplitude = kRockNoiseAmp; rp.noiseFrequency = kRockNoiseFreq;
            rp.rings = kRockRings; rp.sectors = kRockSectors; rp.seed = 303u; rp.meshID = 2u; rp.materialID = 0.0f;
            rp.vertexOffset = rockVertexBase; rp.indexOffset = m_Ranges[2].firstIndex; rp.flattenBottom = kRockFlatten;
            vkCmdPushConstants(cmd, m_GeomPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rp), &rp);
            vkCmdDispatch(cmd, (kRockRings + 1u + 7u) / 8u, (kRockSectors + 1u + 7u) / 8u, 1);
            // The three dispatches write disjoint regions of the shared vertex/index buffers, so no
            // inter-dispatch barrier is needed; ExecuteOneShotCommands' own fence makes all writes
            // visible before any later (render) use.
        });
    }

    void VegetationScatterPass::GenerateScatter() {
        const float halfExtent = config::vegetation::REGION_HALF_EXTENT;
        const float cellSize = std::max(0.05f, config::vegetation::CELL_SIZE);
        const uint32_t gridDim = static_cast<uint32_t>(std::ceil((2.0f * halfExtent) / cellSize));

        ScatterGenParams params{};
        params.originX = -halfExtent;
        params.originZ = -halfExtent;
        params.cellSize = cellSize;
        params.gridDim = gridDim;
        params.grassDensity = config::vegetation::GRASS_DENSITY;
        params.bushDensity = config::vegetation::BUSH_DENSITY;
        params.rockDensity = config::vegetation::ROCK_DENSITY;
        params.maxInstances = kMaxInstances;
        params.seed = config::vegetation::SEED;

        VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_Queue, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, m_CounterBuffer.Handle(), 0, sizeof(uint32_t), 0u);
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScatterGenPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ScatterGenPipelineLayout, 0, 1, &m_ScatterGenSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_ScatterGenPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
            vkCmdDispatch(cmd, (gridDim + 7u) / 8u, (gridDim + 7u) / 8u, 1);

            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            VkBufferCopy copy{ 0, 0, sizeof(uint32_t) };
            vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_CountReadbackBuffer.Handle(), 1, &copy);
        });

        uint32_t count = 0;
        if (m_CountReadbackBuffer.MappedData() != nullptr) {
            std::memcpy(&count, m_CountReadbackBuffer.MappedData(), sizeof(uint32_t));
        }
        m_InstanceCount = std::min(count, kMaxInstances);
        LOG_INFO(std::format("[VegetationScatterPass] Scatter generated: {} instances (grid {}x{}, cell {:.2f}m, region +/-{:.0f}m).",
            m_InstanceCount, gridDim, gridDim, cellSize, halfExtent));
    }

    void VegetationScatterPass::RecordCull(VkCommandBuffer cmd, const maths::mat4& viewProj,
        const maths::vec3& cameraPositionWorld, bool occlusionCullEnabled) {
        if (m_InstanceCount == 0) {
            return;
        }

        // --- Upload this frame's cull parameters. ---
        CullParamsUBO ubo{};
        std::array<std::array<float, 4>, 6> planes = ExtractFrustumPlanes(viewProj);
        for (uint32_t i = 0; i < 6; ++i) {
            for (uint32_t c = 0; c < 4; ++c) {
                ubo.frustumPlanes[i * 4u + c] = planes[i][c];
            }
        }
        ubo.cameraPos[0] = cameraPositionWorld.x;
        ubo.cameraPos[1] = cameraPositionWorld.y;
        ubo.cameraPos[2] = cameraPositionWorld.z;
        ubo.cameraPos[3] = 0.0f;
        ubo.viewProj = viewProj;
        ubo.hzbMip0Size[0] = static_cast<float>(m_HZBMip0Extent.width);
        ubo.hzbMip0Size[1] = static_cast<float>(m_HZBMip0Extent.height);
        ubo.hzbMipCount = static_cast<float>(m_HZBMipCount);
        ubo.enableOcclusion = occlusionCullEnabled ? 1u : 0u;
        for (uint32_t a = 0; a < kArchetypeCount; ++a) {
            ubo.archetypeRadius[a] = kArchetypeBoundingRadius[a];
        }
        ubo.archetypeRadius[3] = 0.0f;
        ubo.maxInstances = kMaxInstances;
        vkCmdUpdateBuffer(cmd, m_CullParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        // Reset each archetype's indirect command instanceCount to 0 (fixed fields re-stamped too).
        std::array<VkDrawIndexedIndirectCommand, kArchetypeCount> cmds{};
        for (uint32_t a = 0; a < kArchetypeCount; ++a) {
            cmds[a] = { m_Ranges[a].indexCount, 0u, m_Ranges[a].firstIndex, 0u, 0u };
        }
        vkCmdUpdateBuffer(cmd, m_IndirectCommandBuffer.Handle(), 0,
            sizeof(VkDrawIndexedIndirectCommand) * kArchetypeCount, cmds.data());

        // Make the UBO/indirect updates AND this frame's freshly-rebuilt HZB (compute imageStore in
        // RecordFrameLate's own second HZB rebuild, earlier in this same command buffer) visible to
        // the cull dispatch's uniform/storage/sampled reads.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipelineLayout, 0, 1, &m_CullSet, 0, nullptr);
        vkCmdDispatch(cmd, (m_InstanceCount + 63u) / 64u, 1, 1);

        // Cull output -> indirect draw + vertex-stage instance/vertex reads.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void VegetationScatterPass::RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView,
        VkImage depthImage, VkImageView depthView, VkExtent2D renderExtent,
        const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
        const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
        bool debugWireframe) {
        if (m_InstanceCount == 0) {
            return;
        }

        // --- Per-frame render params. ---
        RenderParamsUBO ubo{};
        ubo.viewProj = viewProj;
        ubo.cameraPos[0] = cameraPositionWorld.x; ubo.cameraPos[1] = cameraPositionWorld.y; ubo.cameraPos[2] = cameraPositionWorld.z;
        ubo.sunDirection[0] = sunDirectionWorld.x; ubo.sunDirection[1] = sunDirectionWorld.y; ubo.sunDirection[2] = sunDirectionWorld.z;
        ubo.sunIntensity = sunIntensity;
        ubo.sunColor[0] = sunColor.x; ubo.sunColor[1] = sunColor.y; ubo.sunColor[2] = sunColor.z;
        vkCmdUpdateBuffer(cmd, m_RenderParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        // --- Transition color (GENERAL -> COLOR_ATTACHMENT) and depth (READ_ONLY -> ATTACHMENT,
        // test AND write) -- exact mirror of TessellationPass::RecordDraw's own entry barriers, since
        // this pass is likewise an opaque, depth-writing forward pass drawn right after it. ---
        VkImageMemoryBarrier2 toAttachment[2]{};
        toAttachment[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachment[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toAttachment[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachment[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[0].image = colorImage;
        toAttachment[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        toAttachment[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachment[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toAttachment[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        toAttachment[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        toAttachment[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        toAttachment[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        toAttachment[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        toAttachment[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[1].image = depthImage;
        toAttachment[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        VkDependencyInfo toAttachmentDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDep.imageMemoryBarrierCount = 2;
        toAttachmentDep.pImageMemoryBarriers = toAttachment;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDep);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, renderExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        VkPipeline pipeline = m_RenderPipeline;
#ifndef NDEBUG
        if (debugWireframe && m_WireframePipeline != VK_NULL_HANDLE) {
            pipeline = m_WireframePipeline;
        }
#else
        (void)debugWireframe;
#endif
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDescriptorSet sets[3] = { m_RenderInstanceSet, m_RenderParamsSet, m_LightingSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RenderPipelineLayout, 0, 3, sets, 0, nullptr);

        VkViewport viewport{};
        viewport.width = static_cast<float>(renderExtent.width);
        viewport.height = static_cast<float>(renderExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.extent = renderExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindIndexBuffer(cmd, m_ArchetypeIndexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

        // One indirect draw per archetype -- the GPU cull wrote each command's instanceCount. The
        // push constant selects this archetype's compacted visible-index segment + base color.
        for (uint32_t a = 0; a < kArchetypeCount; ++a) {
            VegRenderPushConstants pc{};
            pc.archetypeSegmentBase = a * kMaxInstances;
            pc.archetype = a;
            vkCmdPushConstants(cmd, m_RenderPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
            vkCmdDrawIndexedIndirect(cmd, m_IndirectCommandBuffer.Handle(),
                static_cast<VkDeviceSize>(a) * sizeof(VkDrawIndexedIndirectCommand), 1, sizeof(VkDrawIndexedIndirectCommand));
        }

        vkCmdEndRendering(cmd);

        // --- Restore color -> GENERAL and depth -> READ_ONLY for the subsequent forward passes'
        // depth test + the next frame's HZB rebuild -- exact mirror of TessellationPass' own restore. ---
        VkImageMemoryBarrier2 restore[2]{};
        restore[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restore[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        restore[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        restore[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        restore[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        restore[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        restore[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        restore[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].image = colorImage;
        restore[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        restore[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restore[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        restore[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        restore[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        restore[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        restore[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        restore[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        restore[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].image = depthImage;
        restore[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        VkDependencyInfo restoreDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        restoreDep.imageMemoryBarrierCount = 2;
        restoreDep.pImageMemoryBarriers = restore;
        vkCmdPipelineBarrier2(cmd, &restoreDep);
    }

    // Shutdown() is inherited from RenderPass<VegetationScatterPass>: runs the RegisterResource()
    // cleanups above in reverse (instance-count reset -> render/wireframe pipeline -> cull pipeline
    // -> scatter-gen pipeline -> geom pipelines -> HZB sampler -> descriptor pool + 6 set layouts
    // -> the 10 owned buffers), the same dependency-safe order the hand-written Shutdown() used.

}
