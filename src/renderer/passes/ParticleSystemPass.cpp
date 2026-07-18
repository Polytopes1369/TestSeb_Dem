#include "renderer/passes/ParticleSystemPass.h"

#include <cstring>
#include <format>
#include <vector>

#include "core/Logger.h"
#include "renderer/passes/AtmosClimatePass.h"
#include "renderer/passes/GlobalSDFPass.h"
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
            int32_t mode = 0;
        };
        static_assert(sizeof(ParticleSimulationPC) == 116, "ParticleSimulationPC must match ParticleSimulation.comp's own push-constant block exactly");

    } // namespace

    bool ParticleSystemPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const AtmosClimatePass& atmosClimate, const GlobalSDFPass& globalSDF) {
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
        m_CounterBuffer.Create(allocator, kCounterBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_IndirectDrawBuffer.Create(allocator, kIndirectDrawBufferBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

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

            VkDescriptorSetLayoutBinding envBindings[2]{};
            envBindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            envBindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, GlobalSDFPass::kLevelCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo envLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            envLayoutInfo.bindingCount = 2;
            envLayoutInfo.pBindings = envBindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &envLayoutInfo, nullptr, &m_EnvironmentSetLayout));

            VkDescriptorPoolSize envPoolSizes[2] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
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

            VkWriteDescriptorSet envWrites[2]{};
            envWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &atmosGlobalsInfo, nullptr };
            envWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EnvironmentSet, 1, 0, GlobalSDFPass::kLevelCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, clipmapInfos, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 2, envWrites, 0, nullptr);

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

        LOG_INFO(std::format("[ParticleSystemPass] Initialized: {} max particles, {} KB particle buffer x2, simulation pipeline ready.",
            kMaxParticles, static_cast<uint32_t>(kParticleBufferBytes / 1024)));
        return true;
    }

    void ParticleSystemPass::RecordSimulate(VkCommandBuffer cmd, const GlobalSDFPass& globalSDF, float dt, float time,
        const float emitterPositionWorld[3], uint32_t spawnCount) {
        // Reset aliveCount to 0 (offset 4) and set spawnQueue to spawnCount (offset 8) -- leaves
        // deadCount (offset 0) and _pad0 (offset 12) untouched, since only the GPU itself tracks
        // deadCount's true current value (see this method's own header comment for why aliveCount's
        // reset-then-rebuild is correct here).
        uint32_t zero = 0u;
        vkCmdUpdateBuffer(cmd, m_CounterBuffer.Handle(), 4, sizeof(uint32_t), &zero);
        vkCmdUpdateBuffer(cmd, m_CounterBuffer.Handle(), 8, sizeof(uint32_t), &spawnCount);

        VulkanUtils::RecordMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

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

    void ParticleSystemPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_SimPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_SimPipeline, nullptr);
            if (m_SimPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_SimPipelineLayout, nullptr);
            if (m_EnvironmentDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_EnvironmentDescriptorPool, nullptr);
            if (m_EnvironmentSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_EnvironmentSetLayout, nullptr);
            if (m_ClipmapSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ClipmapSampler, nullptr);

            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }

        m_IndirectDrawBuffer.Destroy();
        m_CounterBuffer.Destroy();
        m_AliveListBuffer.Destroy();
        m_DeadListBuffer.Destroy();
        for (uint32_t i = 0; i < 2; ++i) {
            m_ParticleBuffer[i].Destroy();
        }

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
