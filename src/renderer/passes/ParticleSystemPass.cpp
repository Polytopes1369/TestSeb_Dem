#include "renderer/passes/ParticleSystemPass.h"

#include <cstring>
#include <format>
#include <vector>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Matches VkDrawIndirectCommand's own field order/size exactly (16 bytes) -- used only to
        // build the one-time initial content this class uploads into m_IndirectDrawBuffer; the real
        // struct is used directly everywhere else (vkCmdDrawIndirect, Subtask 4).
        static_assert(sizeof(VkDrawIndirectCommand) == 16, "VkDrawIndirectCommand must be 16 bytes for this one-shot upload to be correct");

    } // namespace

    bool ParticleSystemPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
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

        LOG_INFO(std::format("[ParticleSystemPass] Initialized: {} max particles, {} KB particle buffer x2, descriptor set layout ready for Subtask 2's simulation pipeline.",
            kMaxParticles, static_cast<uint32_t>(kParticleBufferBytes / 1024)));
        return true;
    }

    void ParticleSystemPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
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

        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_ParticleSet[0] = VK_NULL_HANDLE;
        m_ParticleSet[1] = VK_NULL_HANDLE;
        m_CurrentIndex = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

}
