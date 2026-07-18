#include "renderer/passes/ParticleSystemPass.h"

#include <cstring>
#include <format>
#include <vector>

#include "core/Logger.h"
#include "renderer/passes/AtmosClimatePass.h"
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/GlobalSDFPass.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/passes/WorldProbeGridPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Matches VkDrawIndirectCommand's own field order/size exactly (16 bytes) -- used only to
        // build the one-time initial content this class uploads into m_IndirectDrawBuffer; the real
        // struct is used directly everywhere else (vkCmdDrawIndirect, Subtask 4).
        static_assert(sizeof(VkDrawIndirectCommand) == 16, "VkDrawIndirectCommand must be 16 bytes for this one-shot upload to be correct");

        // Byte-for-byte mirror of ParticleSimulation.comp's own ParticleSimulationPC push-constant
        // block -- flat float/int arrays throughout (no vec3), matching this codebase's own
        // established push-constant convention (see e.g. SDFRayMarchPC's own comment) of avoiding
        // vec3's implicit 16-byte alignment padding so the C++ and GLSL byte layouts are trivially
        // identical without needing manual padding fields.
        struct ParticleSimulationPC {
            float dt = 0.0f;
            float time = 0.0f;
            float emitterPosition[3] = { 0.0f, 0.0f, 0.0f };
            float bounceElasticity = 0.0f;
            float friction = 0.0f;
            float dragCoefficient = 0.0f;
            float gravityY = 0.0f;
            float levelVoxelSize[4] = {};
            int32_t levelCenterVoxel[12] = {};
            int32_t clipmapResolution = 0;
            uint32_t spawnCount = 0;
            uint32_t randomSeedBase = 0;
            int32_t mode = 0; // 0 = update, 1 = spawn (embers or, per spawnMode below, a secondary recipe), 2 = spawn precipitation.
            // Precipitation feature (mode == 2 only) -- see ParticleSimulation.comp's own header comment.
            uint32_t precipSpawnCount = 0;
            uint32_t precipKind = 0; // kParticleKindRain or kParticleKindSnow (ParticleCommon.glsl).
            // Rivers/waterfalls feature (mode == 1 only) -- selects which spawn recipe this dispatch
            // uses (0 = default ember recipe, see ParticleSimulation.comp's own field comment for the
            // rest, e.g. 1 = waterfall mist).
            uint32_t spawnMode = 0;
        };
        static_assert(sizeof(ParticleSimulationPC) == 128, "ParticleSimulationPC must match ParticleSimulation.comp's own push-constant block exactly");
        static_assert(sizeof(ParticleSimulationPC) <= 128, "ParticleSimulationPC must stay within the Vulkan-guaranteed minimum maxPushConstantsSize (128 bytes) -- move any new fields into PrecipitationParamsUBO instead of growing this struct further");

        // Precipitation feature -- std140 mirror of ParticleSimulation.comp's own PrecipitationParamsUBO
        // (environment set, binding 2). Deliberately a UBO rather than more push-constant fields: see
        // ParticleSimulationPC's own static_assert above for why this codebase keeps that struct under
        // the guaranteed-minimum push-constant budget.
        struct PrecipitationParamsUBO {
            float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f, spawnRadius = 0.0f;
            float spawnHeightAboveCenter = 0.0f, spawnBandThickness = 0.0f, floorBelowCenter = 0.0f, rainFallSpeed = 0.0f;
            float snowFallSpeed = 0.0f, snowWobbleStrength = 0.0f, _pad0 = 0.0f, _pad1 = 0.0f;
        };
        static_assert(sizeof(PrecipitationParamsUBO) == 48, "PrecipitationParamsUBO must match ParticleSimulation.comp's own UBO exactly (std140 layout)");

        // Byte-for-byte mirror of ParticleSort.comp's own SortedPair struct -- 8 bytes, std430
        // (two 4-byte scalars, no padding needed).
        struct SortedPair {
            uint32_t index = 0;
            float key = 0.0f;
        };
        static_assert(sizeof(SortedPair) == 8, "SortedPair must match ParticleSort.comp's own struct exactly (std430 layout)");

        // Byte-for-byte mirror of ParticleSort.comp's own ParticleSortPC push-constant block.
        struct ParticleSortPC {
            float cameraPosition[3] = { 0.0f, 0.0f, 0.0f };
            float cameraForward[3] = { 0.0f, 0.0f, 0.0f };
            uint32_t stageSize = 0;
            uint32_t passSize = 0;
            int32_t mode = 0;
        };
        static_assert(sizeof(ParticleSortPC) == 36, "ParticleSortPC must match ParticleSort.comp's own push-constant block exactly");

        // Byte-for-byte mirror of ParticleRender.vert/.frag's own ParticleRenderParamsUBO (std140).
        // maths::mat4 is used directly (not decomposed into a flat float array like the push-constant
        // structs above) since a UBO's std140 mat4 member is what this codebase's OTHER UBO-mirror
        // structs already do (see e.g. ScreenSpaceEffectsPass.cpp's own GTAOParamsUBO::invViewProj).
        struct ParticleRenderParamsUBO {
            maths::mat4 viewProj{};
            maths::mat4 invViewProj{};
            float cameraPositionX = 0.0f, cameraPositionY = 0.0f, cameraPositionZ = 0.0f, _pad0 = 0.0f;
            float cameraRightX = 0.0f, cameraRightY = 0.0f, cameraRightZ = 0.0f, _pad1 = 0.0f;
            float cameraUpX = 0.0f, cameraUpY = 0.0f, cameraUpZ = 0.0f, _pad2 = 0.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f, softFadeDistance = 0.0f, globalTime = 0.0f;
            // Subtask 5.
            float sunDirectionX = 0.0f, sunDirectionY = 0.0f, sunDirectionZ = 0.0f, sunIntensity = 0.0f;
            float sunColorR = 0.0f, sunColorG = 0.0f, sunColorB = 0.0f, _pad3 = 0.0f;
            float heatShimmerStrength = 0.0f, _pad4 = 0.0f, _pad5 = 0.0f, _pad6 = 0.0f;
        };
        static_assert(sizeof(ParticleRenderParamsUBO) == 240, "ParticleRenderParamsUBO must match ParticleRender.vert/.frag's own UBO exactly (std140 layout)");

        // Byte-for-byte mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO (std140) --
        // identical to renderer::TessellationPass's own copy (see that class' own comment).
        struct WorldProbeGridParamsUBO {
            float gridOriginX = 0.0f, gridOriginY = 0.0f, gridOriginZ = 0.0f;
            float probeSpacing = 0.0f;
            float gridResolution = 0.0f;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(WorldProbeGridParamsUBO) == 32, "WorldProbeGridParamsUBO must match world_probe_sampling.glsl's own UBO exactly (std140 layout)");

    } // namespace

    bool ParticleSystemPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const AtmosClimatePass& atmosClimate, const GlobalSDFPass& globalSDF, const ClusterResolvePass& resolvePass,
        const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
        VkFormat colorFormat, VkFormat depthFormat) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Buffer allocation. Every buffer here is GPU_ONLY: none of them are written from
        // the host on a per-frame basis (Subtask 2's compute dispatches own all steady-state writes),
        // only once here at Init() via a staging upload (see STEP 2 below), matching the convention
        // renderer::GpuGeometryPagePool's own physical-pool/page-table buffers already establish for
        // "large GPU-only buffer, host-populated exactly once."
        // =====================================================================================
        constexpr VkDeviceSize kParticleBufferBytes = static_cast<VkDeviceSize>(kMaxParticles) * sizeof(GpuParticle);
        constexpr VkDeviceSize kIndexListBytes = static_cast<VkDeviceSize>(kMaxParticles) * sizeof(uint32_t);
        constexpr VkDeviceSize kCounterBufferBytes = 16; // {deadCount, aliveCount, spawnQueue, _pad0}, matches ParticleCommon.glsl's CounterBuffer.
        constexpr VkDeviceSize kIndirectDrawBufferBytes = sizeof(VkDrawIndirectCommand);

        for (uint32_t i = 0; i < 2; ++i) {
            m_ParticleBuffer[i].Create(allocator, kParticleBufferBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }
        m_DeadListBuffer.Create(allocator, kIndexListBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_AliveListBuffer.Create(allocator, kIndexListBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        // TRANSFER_SRC_BIT (on top of the DST_BIT every other buffer here also needs, for the
        // Init()-time seed upload): RecordSort()'s own instanceCountCopy reads aliveCount OUT of
        // this buffer via vkCmdCopyBuffer (see that method's own comment), and so does Subtask 6's
        // Debug-only alive-count readback -- both are genuine copy SOURCES, not just destinations,
        // unlike the other three buffers below.
        m_CounterBuffer.Create(allocator, kCounterBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_IndirectDrawBuffer.Create(allocator, kIndirectDrawBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        // Subtask 3: always kMaxParticles entries long (a power of two, required for bitonic sort --
        // see ParticleSort.comp's own header comment for why this is NOT sized to the frame's actual
        // aliveCount instead). Never host-written -- ParticleSort.comp's own InitKeys pass fully
        // overwrites it every single frame it runs, so no seed upload is needed for this buffer.
        m_SortedPairsBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxParticles) * sizeof(SortedPair),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

#ifndef NDEBUG
        // Subtask 6, Debug-only: see GetLastAliveCountApprox()'s own comment.
        m_AliveCountReadbackBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
#endif

        // =====================================================================================
        // STEP 2 -- One-shot host -> device seed upload: the dead-list starts holding every slot
        // index 0..kMaxParticles-1 (every particle begins dead/available -- Subtask 2's spawn step
        // is the only thing that ever moves an index out of this list), the counter block starts at
        // {deadCount=kMaxParticles, aliveCount=0, spawnQueue=0}, and the indirect-draw buffer starts
        // at {vertexCount=6 (one unindexed billboard quad, two triangles -- Subtask 4), instanceCount=0,
        // firstVertex=0, firstInstance=0} so an early RecordDraw() call before any particle has ever
        // spawned is still a well-defined (zero-instance) no-op draw rather than reading uninitialized
        // GPU memory. A single host-visible staging buffer holds all three payloads back-to-back and
        // is copied into the three GPU_ONLY destinations via vkCmdCopyBuffer inside one one-shot
        // command buffer (VulkanUtils::ExecuteOneShotCommands), then destroyed -- exactly the
        // "temporary CPU_TO_GPU staging buffer, one-shot copy, discard" idiom this codebase's own
        // asset-upload paths (e.g. renderer::SurfaceCacheRayTracingPass's BLAS vertex/index uploads)
        // already use.
        // =====================================================================================
        {
            std::vector<uint32_t> deadListInitial(kMaxParticles);
            for (uint32_t i = 0; i < kMaxParticles; ++i) {
                deadListInitial[i] = i;
            }
            uint32_t counterInitial[4] = { kMaxParticles, 0u, 0u, 0u };
            VkDrawIndirectCommand indirectInitial{ 6u, 0u, 0u, 0u };

            VkDeviceSize stagingBytes = kIndexListBytes + kCounterBufferBytes + kIndirectDrawBufferBytes;
            GpuBuffer staging;
            staging.Create(allocator, stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

            uint8_t* dst = static_cast<uint8_t*>(staging.MappedData());
            VkDeviceSize deadListOffset = 0;
            VkDeviceSize counterOffset = kIndexListBytes;
            VkDeviceSize indirectOffset = kIndexListBytes + kCounterBufferBytes;
            std::memcpy(dst + deadListOffset, deadListInitial.data(), kIndexListBytes);
            std::memcpy(dst + counterOffset, counterInitial, kCounterBufferBytes);
            std::memcpy(dst + indirectOffset, &indirectInitial, kIndirectDrawBufferBytes);

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkBufferCopy deadListCopy{ deadListOffset, 0, kIndexListBytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_DeadListBuffer.Handle(), 1, &deadListCopy);

                VkBufferCopy counterCopy{ counterOffset, 0, kCounterBufferBytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_CounterBuffer.Handle(), 1, &counterCopy);

                VkBufferCopy indirectCopy{ indirectOffset, 0, kIndirectDrawBufferBytes };
                vkCmdCopyBuffer(cmd, staging.Handle(), m_IndirectDrawBuffer.Handle(), 1, &indirectCopy);

                // Every later reader (Subtask 2's simulation compute dispatch, Subtask 4's indirect
                // draw) issues its own COMPUTE_SHADER/DRAW_INDIRECT-stage barrier before touching
                // these buffers for the first time next frame -- this one-shot command buffer's own
                // vkQueueWaitIdle-equivalent completion (ExecuteOneShotCommands blocks until done, see
                // its own header comment) is already a full execution + memory barrier by construction,
                // so no additional VkMemoryBarrier2 is needed here.
                });

            staging.Destroy();
        }

        // =====================================================================================
        // STEP 3 -- Single VkDescriptorSetLayout every particle shader (Subtasks 2-4) binds
        // unmodified, matching src/shaders/include/ParticleCommon.glsl's 4 fixed bindings exactly:
        // 0 = ParticleBuffer, 1 = DeadListBuffer, 2 = AliveListBuffer, 3 = CounterBuffer. Two
        // VkDescriptorSet instances are allocated against it (m_ParticleSet[2]) -- one per physical
        // m_ParticleBuffer[i], everything else (dead/alive/counter) shared and written identically
        // into both sets, since only binding 0 ever differs between the two ping-pong sets (see this
        // class' own header comment on why the free-lists are NOT ping-ponged).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[4]{};
            for (uint32_t b = 0; b < 4; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            }

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 4;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

            VkDescriptorPoolSize poolSizes[1] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * 2 } // 4 bindings x 2 ping-pong sets.
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_SetLayout, m_SetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_DescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_ParticleSet));

            for (uint32_t i = 0; i < 2; ++i) {
                VkDescriptorBufferInfo particleInfo{ m_ParticleBuffer[i].Handle(), 0, m_ParticleBuffer[i].Size() };
                VkDescriptorBufferInfo deadListInfo{ m_DeadListBuffer.Handle(), 0, m_DeadListBuffer.Size() };
                VkDescriptorBufferInfo aliveListInfo{ m_AliveListBuffer.Handle(), 0, m_AliveListBuffer.Size() };
                VkDescriptorBufferInfo counterInfo{ m_CounterBuffer.Handle(), 0, m_CounterBuffer.Size() };

                VkWriteDescriptorSet writes[4]{};
                writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &particleInfo, nullptr };
                writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &deadListInfo, nullptr };
                writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &aliveListInfo, nullptr };
                writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ParticleSet[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);
            }
        }

        m_CurrentIndex = 0;

        // =====================================================================================
        // STEP 4 (Subtask 2) -- ParticleSimulation.comp's environment set (set 1): AtmosGlobalsUBO
        // (wind, borrowed unmodified from `atmosClimate`) + the 4 Global SDF clipmap levels
        // (collision, borrowed unmodified from `globalSDF`, sampled with this pass' own dedicated
        // NEAREST sampler -- see m_ClipmapSampler's own declaration comment for why). Both
        // dependencies already Init'd by the time ClusterRenderPipeline::Init() reaches this call
        // (see this method's own header comment), so written once here, no deferred setter needed.
        // =====================================================================================
        {
            VkSamplerCreateInfo clipmapSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            clipmapSamplerInfo.magFilter = VK_FILTER_NEAREST;
            clipmapSamplerInfo.minFilter = VK_FILTER_NEAREST;
            clipmapSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            clipmapSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            clipmapSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            clipmapSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            clipmapSamplerInfo.minLod = 0.0f;
            clipmapSamplerInfo.maxLod = 0.0f;
            clipmapSamplerInfo.unnormalizedCoordinates = VK_FALSE;
            VK_CHECK(vkCreateSampler(m_Device, &clipmapSamplerInfo, nullptr, &m_ClipmapSampler));

            // Precipitation feature: this pass' own PrecipitationParamsUBO, created here (before the
            // descriptor writes below need its handle) and updated every RecordSimulate() call --
            // see m_PrecipitationParamsBuffer's own declaration comment for why it lives in this
            // environment set rather than in a push constant.
            m_PrecipitationParamsBuffer.Create(allocator, sizeof(PrecipitationParamsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            VkDescriptorSetLayoutBinding envBindings[3]{};
            envBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            envBindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, GlobalSDFPass::kLevelCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            envBindings[2] = { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo envLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            envLayoutInfo.bindingCount = 3;
            envLayoutInfo.pBindings = envBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &envLayoutInfo, nullptr, &m_EnvironmentSetLayout));

            VkDescriptorPoolSize envPoolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 }, // AtmosGlobalsUBO (binding 0) + PrecipitationParamsUBO (binding 2).
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, GlobalSDFPass::kLevelCount }
            };
            VkDescriptorPoolCreateInfo envPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            envPoolInfo.maxSets = 1;
            envPoolInfo.poolSizeCount = 2;
            envPoolInfo.pPoolSizes = envPoolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &envPoolInfo, nullptr, &m_EnvironmentDescriptorPool));

            VkDescriptorSetAllocateInfo envSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            envSetAllocInfo.descriptorPool = m_EnvironmentDescriptorPool;
            envSetAllocInfo.descriptorSetCount = 1;
            envSetAllocInfo.pSetLayouts = &m_EnvironmentSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &envSetAllocInfo, &m_EnvironmentSet));

            VkDescriptorBufferInfo atmosGlobalsInfo{ atmosClimate.GetGlobalsBufferHandle(), 0, atmosClimate.GetGlobalsBufferSize() };
            VkDescriptorImageInfo clipmapInfos[GlobalSDFPass::kLevelCount]{};
            for (uint32_t level = 0; level < GlobalSDFPass::kLevelCount; ++level) {
                clipmapInfos[level] = { m_ClipmapSampler, globalSDF.GetClipmapView(level), VK_IMAGE_LAYOUT_GENERAL };
            }
            VkDescriptorBufferInfo precipParamsInfo{ m_PrecipitationParamsBuffer.Handle(), 0, m_PrecipitationParamsBuffer.Size() };

            VkWriteDescriptorSet envWrites[3]{};
            envWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &atmosGlobalsInfo, nullptr };
            envWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 1, 0, GlobalSDFPass::kLevelCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, clipmapInfos, nullptr, nullptr };
            envWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &precipParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 3, envWrites, 0, nullptr);

            VkDescriptorSetLayout simSetLayouts[2] = { m_SetLayout, m_EnvironmentSetLayout };
            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ParticleSimulationPC) };
            VkPipelineLayoutCreateInfo simPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            simPipelineLayoutInfo.setLayoutCount = 2;
            simPipelineLayoutInfo.pSetLayouts = simSetLayouts;
            simPipelineLayoutInfo.pushConstantRangeCount = 1;
            simPipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &simPipelineLayoutInfo, nullptr, &m_SimPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleSimulation.comp.spv");
            VkComputePipelineCreateInfo simPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            simPipelineInfo.layout = m_SimPipelineLayout;
            simPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            simPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            simPipelineInfo.stage.module = shaderModule;
            simPipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &simPipelineInfo, nullptr, &m_SimPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 5 (Subtask 3) -- ParticleSort.comp's own set 1 (a single SortedPairsBuffer binding --
        // see m_SortedPairsBuffer's own declaration comment for why this is a completely independent
        // set 1 from Subtask 2's environment set, not a shared/extended one).
        // =====================================================================================
        {
            // VERTEX_BIT included alongside COMPUTE_BIT: this same set/layout is reused unmodified
            // by Subtask 4's render pipeline (ParticleRender.vert reads sortedPairs[gl_InstanceIndex]
            // to find which particle each billboard instance draws) -- see renderer::
            // ParticleSystemPass::Init's own STEP 6 comment for why no separate set is built for it.
            VkDescriptorSetLayoutBinding sortBinding{ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo sortLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            sortLayoutInfo.bindingCount = 1;
            sortLayoutInfo.pBindings = &sortBinding;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &sortLayoutInfo, nullptr, &m_SortSetLayout));

            VkDescriptorPoolSize sortPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
            VkDescriptorPoolCreateInfo sortPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            sortPoolInfo.maxSets = 1;
            sortPoolInfo.poolSizeCount = 1;
            sortPoolInfo.pPoolSizes = &sortPoolSize;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &sortPoolInfo, nullptr, &m_SortDescriptorPool));

            VkDescriptorSetAllocateInfo sortSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            sortSetAllocInfo.descriptorPool = m_SortDescriptorPool;
            sortSetAllocInfo.descriptorSetCount = 1;
            sortSetAllocInfo.pSetLayouts = &m_SortSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &sortSetAllocInfo, &m_SortSet));

            VkDescriptorBufferInfo sortedPairsInfo{ m_SortedPairsBuffer.Handle(), 0, m_SortedPairsBuffer.Size() };
            VkWriteDescriptorSet sortWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_SortSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sortedPairsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 1, &sortWrite, 0, nullptr);

            VkDescriptorSetLayout sortSetLayouts[2] = { m_SetLayout, m_SortSetLayout };
            VkPushConstantRange sortPushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ParticleSortPC) };
            VkPipelineLayoutCreateInfo sortPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            sortPipelineLayoutInfo.setLayoutCount = 2;
            sortPipelineLayoutInfo.pSetLayouts = sortSetLayouts;
            sortPipelineLayoutInfo.pushConstantRangeCount = 1;
            sortPipelineLayoutInfo.pPushConstantRanges = &sortPushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &sortPipelineLayoutInfo, nullptr, &m_SortPipelineLayout));

            VkShaderModule sortShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleSort.comp.spv");
            VkComputePipelineCreateInfo sortPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            sortPipelineInfo.layout = m_SortPipelineLayout;
            sortPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            sortPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            sortPipelineInfo.stage.module = sortShaderModule;
            sortPipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &sortPipelineInfo, nullptr, &m_SortPipeline));
            vkDestroyShaderModule(m_Device, sortShaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 6 (Subtask 4) -- ParticleRender.vert/.frag's own set 2 (ParticleRenderParamsUBO +
        // `resolvePass`'s sampled GBuffer depth copy, borrowed unmodified and bound once here, same
        // one-time convention as STEP 4's environment set) plus the graphics pipeline itself. Sets 0
        // and 1 for this pipeline are m_SetLayout/m_SortSetLayout, REUSED unmodified -- no new
        // descriptor sets needed for the particle-state/sorted-order bindings this pipeline reads.
        // =====================================================================================
        {
            m_RenderParamsBuffer.Create(allocator, sizeof(ParticleRenderParamsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            VkSamplerCreateInfo depthSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            depthSamplerInfo.magFilter = VK_FILTER_NEAREST;
            depthSamplerInfo.minFilter = VK_FILTER_NEAREST;
            depthSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            depthSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            depthSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            depthSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            depthSamplerInfo.minLod = 0.0f;
            depthSamplerInfo.maxLod = 0.0f;
            depthSamplerInfo.unnormalizedCoordinates = VK_FALSE;
            VK_CHECK(vkCreateSampler(m_Device, &depthSamplerInfo, nullptr, &m_SceneDepthSampler));

            VkDescriptorSetLayoutBinding renderBindings[2]{};
            renderBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            renderBindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo renderLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            renderLayoutInfo.bindingCount = 2;
            renderLayoutInfo.pBindings = renderBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &renderLayoutInfo, nullptr, &m_RenderSetLayout));

            VkDescriptorPoolSize renderPoolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }
            };
            VkDescriptorPoolCreateInfo renderPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            renderPoolInfo.maxSets = 1;
            renderPoolInfo.poolSizeCount = 2;
            renderPoolInfo.pPoolSizes = renderPoolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &renderPoolInfo, nullptr, &m_RenderDescriptorPool));

            VkDescriptorSetAllocateInfo renderSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            renderSetAllocInfo.descriptorPool = m_RenderDescriptorPool;
            renderSetAllocInfo.descriptorSetCount = 1;
            renderSetAllocInfo.pSetLayouts = &m_RenderSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &renderSetAllocInfo, &m_RenderSet));

            VkDescriptorBufferInfo renderParamsInfo{ m_RenderParamsBuffer.Handle(), 0, m_RenderParamsBuffer.Size() };
            VkDescriptorImageInfo sceneDepthInfo{ m_SceneDepthSampler, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet renderWrites[2]{};
            renderWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RenderSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &renderParamsInfo, nullptr };
            renderWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RenderSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sceneDepthInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, renderWrites, 0, nullptr);
        }

        // =====================================================================================
        // STEP 6b (Subtask 5) -- ParticleRender.frag's own set 3 ("lighting"): `vsm`'s 4 Virtual
        // Shadow Map resources + `worldProbes`' grid + a one-time-uploaded WorldProbeGridParamsUBO
        // -- see Init()'s own comment for the full rationale. Binding indices mirror
        // ParticleRender.frag's own SHADOW_*/WORLD_PROBE_GRID_* macro definitions exactly (0-3 = VSM,
        // 4-5 = World Probe Grid).
        // =====================================================================================
        {
            m_WorldProbeGridParamsBuffer.Create(allocator, sizeof(WorldProbeGridParamsUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            WorldProbeGridParamsUBO gridParams{};
            gridParams.gridOriginX = worldProbes.GetGridOriginWorld().x;
            gridParams.gridOriginY = worldProbes.GetGridOriginWorld().y;
            gridParams.gridOriginZ = worldProbes.GetGridOriginWorld().z;
            gridParams.probeSpacing = WorldProbeGridPass::kProbeSpacing;
            gridParams.gridResolution = static_cast<float>(WorldProbeGridPass::kGridResolution);
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdUpdateBuffer(cmd, m_WorldProbeGridParamsBuffer.Handle(), 0, sizeof(gridParams), &gridParams);
                });

            VkDescriptorSetLayoutBinding lightingBindings[6]{};
            lightingBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };          // Shadow page table.
            lightingBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };          // Shadow feedback.
            lightingBindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };  // Shadow physical atlas.
            lightingBindings[3] = { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // Shadow sun clipmap levels.
            lightingBindings[4] = { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // World Probe Grid.
            lightingBindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // World Probe Grid params.

            VkDescriptorSetLayoutCreateInfo lightingLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            lightingLayoutInfo.bindingCount = 6;
            lightingLayoutInfo.pBindings = lightingBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &lightingLayoutInfo, nullptr, &m_LightingSetLayout));

            VkDescriptorPoolSize lightingPoolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 }
            };
            VkDescriptorPoolCreateInfo lightingPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            lightingPoolInfo.maxSets = 1;
            lightingPoolInfo.poolSizeCount = 3;
            lightingPoolInfo.pPoolSizes = lightingPoolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &lightingPoolInfo, nullptr, &m_LightingDescriptorPool));

            VkDescriptorSetAllocateInfo lightingSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            lightingSetAllocInfo.descriptorPool = m_LightingDescriptorPool;
            lightingSetAllocInfo.descriptorSetCount = 1;
            lightingSetAllocInfo.pSetLayouts = &m_LightingSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &lightingSetAllocInfo, &m_LightingSet));

            VkDescriptorBufferInfo pageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo feedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo atlasInfo{ vsm.GetPhysicalAtlasSampler(), vsm.GetPhysicalAtlasView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo sunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo worldProbeGridInfo{ worldProbes.GetGridSampler(), worldProbes.GetGridView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo worldProbeGridParamsInfo{ m_WorldProbeGridParamsBuffer.Handle(), 0, m_WorldProbeGridParamsBuffer.Size() };

            VkWriteDescriptorSet lightingWrites[6]{};
            lightingWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
            lightingWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
            lightingWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &atlasInfo, nullptr, nullptr };
            lightingWrites[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sunLevelsInfo, nullptr };
            lightingWrites[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &worldProbeGridInfo, nullptr, nullptr };
            lightingWrites[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_LightingSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &worldProbeGridParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 6, lightingWrites, 0, nullptr);
        }

        // =====================================================================================
        // STEP 6c (Subtask 4+5) -- ParticleRender.vert/.frag's own graphics pipeline: 4 sets
        // (m_SetLayout/m_SortSetLayout/m_RenderSetLayout/m_LightingSetLayout) and 2 color
        // attachments (particle color, alpha-blended; refraction offset, plain overwrite -- see
        // this pass' own colorBlendAttachments comment below).
        // =====================================================================================
        {
            VkDescriptorSetLayout renderPipelineSetLayouts[4] = { m_SetLayout, m_SortSetLayout, m_RenderSetLayout, m_LightingSetLayout };
            VkPipelineLayoutCreateInfo renderPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            renderPipelineLayoutInfo.setLayoutCount = 4;
            renderPipelineLayoutInfo.pSetLayouts = renderPipelineSetLayouts;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &renderPipelineLayoutInfo, nullptr, &m_RenderPipelineLayout));

            VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleRender.vert.spv");
            VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleRender.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr };

            // No bound vertex buffer -- ParticleRender.vert generates every corner from
            // gl_VertexIndex, see that shader's own header comment.
            VkPipelineVertexInputStateCreateInfo vertexInputInfo{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            // No culling -- a camera-facing billboard's "back face" should never be visible in
            // practice (it is always oriented toward the camera by construction), but disabling
            // culling costs nothing here and avoids a silent black quad if a future rotation/size
            // edge case ever flips winding order.
            VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Standard "over" alpha blend against the already-composited scene -- same state as
            // renderer::TransparentForwardPass's own colorBlendAttachment (see that class' own
            // comment for the rationale, identical here).
            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_TRUE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

            // Subtask 5: g_RefractionOffset (attachment 1) -- a plain overwrite (blendEnable=FALSE),
            // same rationale as renderer::TransparentForwardPass's own identical second-attachment
            // state: a distortion vector doesn't "compose" over a previous forward pass' contribution
            // the way color does, and ParticleRender.frag's own main() always writes SOME value here
            // (explicitly (0,0) when heatShimmerStrength == 0, see that shader's own comment), so a
            // plain overwrite from whichever forward pass drew this pixel LAST is exactly correct.
            VkPipelineColorBlendAttachmentState refractionBlendAttachment{};
            refractionBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            refractionBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState colorBlendAttachments[2] = { colorBlendAttachment, refractionBlendAttachment };
            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            colorBlending.attachmentCount = 2;
            colorBlending.pAttachments = colorBlendAttachments;

            VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            // Subtask 5: attachment 1 is renderer::TransparentForwardPass::kRefractionOffsetFormat
            // (RG16F) -- must match exactly, since this pass writes into that SAME shared image (see
            // RecordDraw's own comment), not a format renderer::TransparentForwardPass merely happens
            // to also use.
            VkFormat colorFormats[2] = { colorFormat, VK_FORMAT_R16G16_SFLOAT };
            VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            pipelineRendering.colorAttachmentCount = 2;
            pipelineRendering.pColorAttachmentFormats = colorFormats;
            pipelineRendering.depthAttachmentFormat = depthFormat;

            // Depth-tested (reversed-Z) but NOT written, same rationale as
            // renderer::TransparentForwardPass's own depthStencil state -- particles must be hidden
            // behind opaque geometry but never occlude each other via the real depth buffer (Subtask
            // 3's sort already gives them a correct relative draw order).
            VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_FALSE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.stencilTestEnable = VK_FALSE;

            VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            pipelineInfo.pNext = &pipelineRendering;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = stages;
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pDynamicState = &dynamicState;
            pipelineInfo.layout = m_RenderPipelineLayout;

            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_RenderPipeline));

            vkDestroyShaderModule(m_Device, vertModule, nullptr);
            vkDestroyShaderModule(m_Device, fragModule, nullptr);
        }

        LOG_INFO(std::format("[ParticleSystemPass] Initialized: {} max particles, {} KB particle buffer x2, simulation + sort + render pipelines ready.",
            kMaxParticles, static_cast<uint32_t>(kParticleBufferBytes / 1024)));
        return true;
    }

    void ParticleSystemPass::RecordSimulate(VkCommandBuffer cmd, const GlobalSDFPass& globalSDF, float dt, float time,
        const float emitterPositionWorld[3], uint32_t spawnCount,
        const float precipCenterWorld[3], uint32_t precipSpawnCount, uint32_t precipKind,
        float precipSpawnRadiusMeters, float precipSpawnHeightAboveCenterMeters,
        float precipSpawnBandThicknessMeters, float precipFloorBelowCenterMeters,
        float precipRainFallSpeedMps, float precipSnowFallSpeedMps, float precipSnowWobbleStrength,
        const float secondaryEmitterPositionWorld[3], uint32_t secondarySpawnCount, uint32_t secondarySpawnMode) {
        // Reset aliveCount to 0 (offset 4) and set spawnQueue to the TOTAL requested this call
        // (primary embers + secondary/mist, purely informational -- see ParticleSimulation.comp's own
        // header comment on why the shader itself reads pc.spawnCount, not this buffer field) -- leaves
        // deadCount (offset 0) and _pad0 (offset 12) untouched, since only the GPU itself tracks
        // deadCount's true current value (see this method's own header comment for why aliveCount's
        // reset-then-rebuild is correct here). spawnQueue only ever tracked the EMBERS spawn count
        // historically -- precipitation's own precipSpawnCount travels via the push constant instead
        // (see ParticleSimulationPC's own comment), so this pair of updates is unchanged.
        uint32_t zero = 0u;
        uint32_t totalSpawnCount = spawnCount + secondarySpawnCount;
        vkCmdUpdateBuffer(cmd, m_CounterBuffer.Handle(), 4, sizeof(uint32_t), &zero);
        vkCmdUpdateBuffer(cmd, m_CounterBuffer.Handle(), 8, sizeof(uint32_t), &totalSpawnCount);

        // Precipitation feature: this frame's camera-relative spawn-shell geometry + per-kind fall
        // speed/wobble constants, consumed by BOTH the mode == 2 spawn dispatch below (spawn-shell
        // geometry) and the mode == 0 update dispatch (fall speed/wobble/floor, for every
        // rain/snow particle currently alive, not just ones spawned this frame) -- see
        // m_PrecipitationParamsBuffer's own declaration comment for why this is a UBO instead of more
        // push-constant fields.
        PrecipitationParamsUBO precipUbo{};
        precipUbo.centerX = precipCenterWorld[0];
        precipUbo.centerY = precipCenterWorld[1];
        precipUbo.centerZ = precipCenterWorld[2];
        precipUbo.spawnRadius = precipSpawnRadiusMeters;
        precipUbo.spawnHeightAboveCenter = precipSpawnHeightAboveCenterMeters;
        precipUbo.spawnBandThickness = precipSpawnBandThicknessMeters;
        precipUbo.floorBelowCenter = precipFloorBelowCenterMeters;
        precipUbo.rainFallSpeed = precipRainFallSpeedMps;
        precipUbo.snowFallSpeed = precipSnowFallSpeedMps;
        precipUbo.snowWobbleStrength = precipSnowWobbleStrength;
        vkCmdUpdateBuffer(cmd, m_PrecipitationParamsBuffer.Handle(), 0, sizeof(precipUbo), &precipUbo);

        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_UNIFORM_READ_BIT);

        VkDescriptorSet sets[2] = { GetCurrentSet(), m_EnvironmentSet };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SimPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SimPipelineLayout, 0, 2, sets, 0, nullptr);

        ParticleSimulationPC pc{};
        pc.dt = dt;
        pc.time = time;
        pc.emitterPosition[0] = emitterPositionWorld[0];
        pc.emitterPosition[1] = emitterPositionWorld[1];
        pc.emitterPosition[2] = emitterPositionWorld[2];
        pc.bounceElasticity = 0.4f;
        pc.friction = 0.85f;
        pc.dragCoefficient = 0.5f;
        pc.gravityY = -9.8f;
        for (uint32_t level = 0; level < GlobalSDFPass::kLevelCount; ++level) {
            pc.levelVoxelSize[level] = globalSDF.GetLevelVoxelSize(level);
            int32_t centerVoxel[3];
            globalSDF.GetLevelSnappedCenterVoxel(level, centerVoxel);
            pc.levelCenterVoxel[level * 3 + 0] = centerVoxel[0];
            pc.levelCenterVoxel[level * 3 + 1] = centerVoxel[1];
            pc.levelCenterVoxel[level * 3 + 2] = centerVoxel[2];
        }
        pc.clipmapResolution = static_cast<int32_t>(GlobalSDFPass::kClipmapResolution);
        pc.spawnCount = spawnCount;
        pc.randomSeedBase = static_cast<uint32_t>(time * 1000.0f) * 2654435761u;
        pc.spawnMode = 0;
        pc.precipSpawnCount = precipSpawnCount;
        pc.precipKind = precipKind;

        if (spawnCount > 0) {
            pc.mode = 1;
            vkCmdPushConstants(cmd, m_SimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t spawnGroups = (spawnCount + 63u) / 64u;
            vkCmdDispatch(cmd, spawnGroups, 1, 1);

            // Spawn wrote fresh particles into slots the update dispatch below is about to read (and
            // mutated deadCount) -- both dispatches are COMPUTE_SHADER-stage, so a same-stage
            // execution + memory barrier is all that is needed between them.
            VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // Rivers/waterfalls feature: optional SECOND Spawn dispatch (e.g. waterfall mist) -- see
        // this method's own header comment for why this is a second Spawn dispatch rather than a
        // second whole RecordSimulate() call. Own randomSeedBase salt (XOR, not a fresh hash of a
        // different quantity) so this dispatch's PRNG stream never exactly repeats the primary
        // dispatch's, while staying just as deterministic-per-frame.
        if (secondarySpawnCount > 0 && secondaryEmitterPositionWorld != nullptr) {
            pc.emitterPosition[0] = secondaryEmitterPositionWorld[0];
            pc.emitterPosition[1] = secondaryEmitterPositionWorld[1];
            pc.emitterPosition[2] = secondaryEmitterPositionWorld[2];
            pc.spawnCount = secondarySpawnCount;
            pc.spawnMode = secondarySpawnMode;
            pc.randomSeedBase ^= 0x9E3779B9u;
            pc.mode = 1;
            vkCmdPushConstants(cmd, m_SimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t spawnGroups = (secondarySpawnCount + 63u) / 64u;
            vkCmdDispatch(cmd, spawnGroups, 1, 1);

            VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        // Precipitation feature -- a third, independent spawn dispatch (mode == 2), against the SAME
        // shared dead-list the embers/mist spawns above just drew from (see ParticleSimulation.comp's
        // own TryPopDeadListSlot comment for why this is the correct "share the pool, back off
        // gracefully" behavior rather than a starvation risk). Deliberately its own dispatch, not
        // folded into the mode == 1 one(s) above: embers/mist and precipitation have entirely
        // different spawn-volume/initial-velocity logic (SpawnParticle/SpawnMistParticle vs
        // SpawnPrecipitationParticle), and this codebase's own established "one shader, multiple modes
        // via a push-constant int + branch in main()" convention (see this file's own header comment)
        // keeps that branch at the dispatch level, not re-decided per-thread inside a single fused
        // kernel. pc.spawnMode/pc.emitterPosition are left at whatever the last spawn dispatch above
        // set them to -- irrelevant here, since mode == 2 reads precipSpawnCount/precipKind (and the
        // spawn-shell geometry from m_PrecipitationParamsBuffer) instead of either field.
        if (precipSpawnCount > 0) {
            pc.mode = 2;
            vkCmdPushConstants(cmd, m_SimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            uint32_t precipSpawnGroups = (precipSpawnCount + 63u) / 64u;
            vkCmdDispatch(cmd, precipSpawnGroups, 1, 1);

            VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }

        pc.mode = 0;
        vkCmdPushConstants(cmd, m_SimPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        uint32_t updateGroups = (kMaxParticles + 63u) / 64u;
        vkCmdDispatch(cmd, updateGroups, 1, 1);

        // Trailing barrier for the next COMPUTE_SHADER-stage consumer (Subtask 3's sort dispatch) --
        // see this method's own header comment for why a future render-stage consumer (Subtask 4)
        // will need its own additional barrier at that time.
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    void ParticleSystemPass::RecordSort(VkCommandBuffer cmd, const float cameraPositionWorld[3], const float cameraForwardWorld[3]) {
        VkDescriptorSet sets[2] = { GetCurrentSet(), m_SortSet };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_SortPipelineLayout, 0, 2, sets, 0, nullptr);

        ParticleSortPC pc{};
        pc.cameraPosition[0] = cameraPositionWorld[0];
        pc.cameraPosition[1] = cameraPositionWorld[1];
        pc.cameraPosition[2] = cameraPositionWorld[2];
        pc.cameraForward[0] = cameraForwardWorld[0];
        pc.cameraForward[1] = cameraForwardWorld[1];
        pc.cameraForward[2] = cameraForwardWorld[2];

        uint32_t groups = (kMaxParticles + 255u) / 256u;

        // --- InitKeys (mode 0) ---
        pc.mode = 0;
        vkCmdPushConstants(cmd, m_SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups, 1, 1);
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        // --- Bitonic compare-exchange network (mode 1) -- see ParticleSort.comp's own header
        // comment for the (stageSize, passSize) iteration this reproduces and why a full memory
        // barrier is required after EVERY single step, not just between stages. ---
        pc.mode = 1;
        for (uint32_t stageSize = 2; stageSize <= kMaxParticles; stageSize *= 2) {
            for (uint32_t passSize = stageSize / 2; passSize > 0; passSize /= 2) {
                pc.stageSize = stageSize;
                pc.passSize = passSize;
                vkCmdPushConstants(cmd, m_SortPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, groups, 1, 1);
                VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }
        }

        // Propagate this frame's real alive count into the indirect-draw buffer's own
        // `instanceCount` field (VkDrawIndirectCommand's second uint32_t, byte offset 4) -- a
        // GPU-side copy, no CPU readback, so a future indirect draw call (Subtask 4) always reflects
        // the count this exact RecordSort() call just finished sorting for.
        VkBufferCopy instanceCountCopy{ 4, 4, sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_IndirectDrawBuffer.Handle(), 1, &instanceCountCopy);

#ifndef NDEBUG
        // Subtask 6, Debug-only: see GetLastAliveCountApprox()'s own comment -- deliberately no
        // fence-wait/barrier around this specific copy, since the whole point is a cheap, stale-
        // tolerant observability readout, not a value any GPU work this frame depends on.
        VkBufferCopy aliveCountReadbackCopy{ 4, 0, sizeof(uint32_t) };
        vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_AliveCountReadbackBuffer.Handle(), 1, &aliveCountReadbackCopy);
#endif

        // Trailing barrier for the next stage -- covers both a future COMPUTE_SHADER consumer and
        // the TRANSFER write just issued above; a render-stage consumer (Subtask 4) will additionally
        // need its own INDIRECT_COMMAND_READ barrier on the indirect-draw buffer specifically at that
        // call site (this method does not know yet whether/when a draw call follows it).
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

#ifndef NDEBUG
    uint32_t ParticleSystemPass::GetLastAliveCountApprox() const {
        if (m_AliveCountReadbackBuffer.MappedData() == nullptr) {
            return 0;
        }
        return *static_cast<const uint32_t*>(m_AliveCountReadbackBuffer.MappedData());
    }
#endif

    void ParticleSystemPass::RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView, VkImageView depthView,
        VkImageView refractionOffsetView, VkExtent2D renderExtent,
        const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
        const maths::vec3& cameraRightWorld, const maths::vec3& cameraUpWorld,
        const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
        float softFadeDistanceWorld, float heatShimmerStrength, float globalTimeSeconds) {
        ParticleRenderParamsUBO ubo{};
        ubo.viewProj = viewProj;
        ubo.invViewProj = viewProj.Inverse();
        ubo.cameraPositionX = cameraPositionWorld.x; ubo.cameraPositionY = cameraPositionWorld.y; ubo.cameraPositionZ = cameraPositionWorld.z;
        ubo.cameraRightX = cameraRightWorld.x; ubo.cameraRightY = cameraRightWorld.y; ubo.cameraRightZ = cameraRightWorld.z;
        ubo.cameraUpX = cameraUpWorld.x; ubo.cameraUpY = cameraUpWorld.y; ubo.cameraUpZ = cameraUpWorld.z;
        ubo.viewportWidth = static_cast<float>(renderExtent.width);
        ubo.viewportHeight = static_cast<float>(renderExtent.height);
        ubo.softFadeDistance = softFadeDistanceWorld;
        ubo.globalTime = globalTimeSeconds;
        ubo.sunDirectionX = sunDirectionWorld.x; ubo.sunDirectionY = sunDirectionWorld.y; ubo.sunDirectionZ = sunDirectionWorld.z;
        ubo.sunIntensity = sunIntensity;
        ubo.sunColorR = sunColor.x; ubo.sunColorG = sunColor.y; ubo.sunColorB = sunColor.z;
        ubo.heatShimmerStrength = heatShimmerStrength;
        vkCmdUpdateBuffer(cmd, m_RenderParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        // Covers both this UBO update AND Subtask 3's own trailing barrier scope gap (RecordSort's
        // own trailing barrier only makes the sorted-pair/indirect-draw data visible to
        // COMPUTE_SHADER, not to this draw's VERTEX_SHADER/DRAW_INDIRECT reads -- see RecordSort's
        // own comment).
        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT);

        VkImageMemoryBarrier2 toAttachment{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toAttachment.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        toAttachment.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        toAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.image = colorImage;
        toAttachment.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo toAttachmentDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDependency.imageMemoryBarrierCount = 1;
        toAttachmentDependency.pImageMemoryBarriers = &toAttachment;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDependency);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Preserve the already-composited scene underneath.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Subtask 5: renderer::TransparentForwardPass's own shared heat-distortion image -- already
        // VK_IMAGE_LAYOUT_GENERAL by this point in the frame (that pass' own RecordDraw leaves it
        // there for renderer::PostProcessPass' later compute read, see ParticleRender.frag's own
        // header comment) and dynamic rendering legally accepts GENERAL for any attachment use, so
        // this pass binds it AT that layout directly -- no barrier/transition dance needed for this
        // specific image, unlike `colorImage`/`depthView` above. loadOp=LOAD preserves whatever
        // TransparentForwardPass/WaterForwardPass already wrote there this frame (see
        // ParticleRender.frag's own "why write (0,0) explicitly" comment for the other half of this
        // contract).
        VkRenderingAttachmentInfo refractionAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        refractionAttachment.imageView = refractionOffsetView;
        refractionAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        refractionAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        refractionAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Same "already read-only by this point in the frame" assumption as
        // renderer::TransparentForwardPass::RecordDraw's own depth attachment -- see that method's
        // own comment. No transition/barrier needed: depthWriteEnable=FALSE means this pass never
        // writes it either.
        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo colorAttachments[2] = { colorAttachment, refractionAttachment };
        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, renderExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 2;
        renderingInfo.pColorAttachments = colorAttachments;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RenderPipeline);
        VkDescriptorSet renderSets[4] = { GetCurrentSet(), m_SortSet, m_RenderSet, m_LightingSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RenderPipelineLayout, 0, 4, renderSets, 0, nullptr);

        VkViewport viewport{};
        viewport.width = static_cast<float>(renderExtent.width);
        viewport.height = static_cast<float>(renderExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = renderExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // This codebase's first vkCmdDrawIndirect call (every prior indirect-draw consumer uses the
        // INDEXED variant -- see m_IndirectDrawBuffer's own declaration comment). `vertexCount=6`/
        // `instanceCount=aliveCount` were both already written into this buffer -- the former once
        // at Init(), the latter every RecordSort() call (see that method's own trailing comment).
        vkCmdDrawIndirect(cmd, m_IndirectDrawBuffer.Handle(), 0, 1, sizeof(VkDrawIndirectCommand));

        vkCmdEndRendering(cmd);

        VkImageMemoryBarrier2 toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toGeneral.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = colorImage;
        toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo toGeneralDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toGeneralDependency.imageMemoryBarrierCount = 1;
        toGeneralDependency.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cmd, &toGeneralDependency);
    }

    void ParticleSystemPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_LightingDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_LightingDescriptorPool, nullptr);
            if (m_LightingSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_LightingSetLayout, nullptr);

            if (m_RenderPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_RenderPipeline, nullptr);
            if (m_RenderPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_RenderPipelineLayout, nullptr);
            if (m_RenderDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_RenderDescriptorPool, nullptr);
            if (m_RenderSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RenderSetLayout, nullptr);
            if (m_SceneDepthSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_SceneDepthSampler, nullptr);

            if (m_SortPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_SortPipeline, nullptr);
            if (m_SortPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_SortPipelineLayout, nullptr);
            if (m_SortDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_SortDescriptorPool, nullptr);
            if (m_SortSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SortSetLayout, nullptr);

            if (m_SimPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_SimPipeline, nullptr);
            if (m_SimPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_SimPipelineLayout, nullptr);
            if (m_EnvironmentDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_EnvironmentDescriptorPool, nullptr);
            if (m_EnvironmentSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_EnvironmentSetLayout, nullptr);
            if (m_ClipmapSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ClipmapSampler, nullptr);

            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }

        m_SortedPairsBuffer.Destroy();
#ifndef NDEBUG
        m_AliveCountReadbackBuffer.Destroy();
#endif
        m_IndirectDrawBuffer.Destroy();
        m_CounterBuffer.Destroy();
        m_AliveListBuffer.Destroy();
        m_DeadListBuffer.Destroy();
        for (uint32_t i = 0; i < 2; ++i) {
            m_ParticleBuffer[i].Destroy();
        }

        m_PrecipitationParamsBuffer.Destroy();
        m_RenderParamsBuffer.Destroy();
        m_WorldProbeGridParamsBuffer.Destroy();

        m_LightingDescriptorPool = VK_NULL_HANDLE;
        m_LightingSetLayout = VK_NULL_HANDLE;
        m_LightingSet = VK_NULL_HANDLE;

        m_RenderPipeline = VK_NULL_HANDLE;
        m_RenderPipelineLayout = VK_NULL_HANDLE;
        m_RenderDescriptorPool = VK_NULL_HANDLE;
        m_RenderSetLayout = VK_NULL_HANDLE;
        m_RenderSet = VK_NULL_HANDLE;
        m_SceneDepthSampler = VK_NULL_HANDLE;

        m_SortPipeline = VK_NULL_HANDLE;
        m_SortPipelineLayout = VK_NULL_HANDLE;
        m_SortDescriptorPool = VK_NULL_HANDLE;
        m_SortSetLayout = VK_NULL_HANDLE;
        m_SortSet = VK_NULL_HANDLE;

        m_SimPipeline = VK_NULL_HANDLE;
        m_SimPipelineLayout = VK_NULL_HANDLE;
        m_EnvironmentDescriptorPool = VK_NULL_HANDLE;
        m_EnvironmentSetLayout = VK_NULL_HANDLE;
        m_EnvironmentSet = VK_NULL_HANDLE;
        m_ClipmapSampler = VK_NULL_HANDLE;

        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_ParticleSet[0] = VK_NULL_HANDLE;
        m_ParticleSet[1] = VK_NULL_HANDLE;
        m_CurrentIndex = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

}
