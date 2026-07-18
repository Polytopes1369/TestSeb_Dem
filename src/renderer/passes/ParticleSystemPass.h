#pragma once
// GPU-driven particle system (Niagara-style), Subtask 1 of particle_system_integration_plan.md
// (project root): owns every buffer the later subtasks' compute/graphics dispatches operate on --
// a double-buffered particle SSBO, a dead-list/alive-list free-list pair, a single counter block,
// and an indirect-draw-arguments buffer -- plus the one VkDescriptorSetLayout every particle shader
// (ParticleSimulation.comp/ParticleSort.comp/ParticleRender.vert+.frag, Subtasks 2-5) binds
// unmodified. See src/shaders/include/ParticleCommon.glsl for the exact GLSL mirror of every buffer
// declared here -- the two files must be kept byte-for-byte in sync (std430).
//
// This subtask deliberately stops at "buffers + descriptor set layout" -- no compute pipeline, no
// simulation/sort/render logic yet (Subtasks 2-4 add those on top of this skeleton, each via their
// own RecordXxx() method). Nothing here is Debug-only: per CLAUDE.md's build-separation rule, only
// verbose diagnostic strings/overlays/the validation-layer/logger machinery itself are excluded from
// Release -- LOG_INFO/LOG_ERROR (core/Logger.h) already compile to a no-op in Release builds, so no
// additional #ifdef guarding is needed in this class for that rule; the particle system's actual
// simulate/sort/render logic must run in Release exactly as in Debug.
//
// Double buffering (m_ParticleBuffer[2]/m_ParticleSet[2]): Subtask 2's simulation dispatch reads the
// PREVIOUS frame's particle buffer and writes the buffer for THIS frame, then the two swap for next
// frame -- same ping-pong idiom renderer::ReflectionPass's own m_Slots[2]/m_TraceSet[2] already
// establish (see that class' own header comment). GetCurrentSetIndex()/Advance() below let a caller
// (this class itself, from RecordSimulate() once Subtask 2 lands) track and flip which of the two
// sets is "current" without exposing raw buffer handles.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    // Byte-for-byte mirror of ParticleCommon.glsl's `Particle` struct -- 64 bytes, std430 (vec3
    // members are 16-byte aligned in std430, so the trailing scalar after each vec3 packs into the
    // same 16-byte slot with no manual padding needed; see ParticleCommon.glsl's own comment for the
    // full layout rationale). Named `GpuParticle`, not `Particle`, to avoid colliding with any
    // future CPU-side particle/emitter-authoring type Subtask 6's ImGui panel might introduce.
    struct GpuParticle {
        float positionX = 0.0f, positionY = 0.0f, positionZ = 0.0f;
        float life = 0.0f;
        float velocityX = 0.0f, velocityY = 0.0f, velocityZ = 0.0f;
        float maxLife = 0.0f;
        float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f, colorA = 0.0f;
        float sizeX = 0.0f, sizeY = 0.0f;
        float rotation = 0.0f;
        uint32_t randomSeed = 0;
    };
    static_assert(sizeof(GpuParticle) == 64, "GpuParticle must match ParticleCommon.glsl's Particle struct exactly (std430 layout)");

    class ParticleSystemPass {
    public:
        ParticleSystemPass() = default;

        ParticleSystemPass(const ParticleSystemPass&) = delete;
        ParticleSystemPass& operator=(const ParticleSystemPass&) = delete;

        // Total particle-buffer capacity (both the alive and dead lists are sized to this, and the
        // dead-list is fully populated with indices 0..kMaxParticles-1 at Init() time -- see Init's
        // own "seed the dead-list" comment). A mid-range GPU-particle budget for a single demoscene
        // scene's worth of emitters (smoke/sparks/embers, Subtask 6) -- not tied to any hardware
        // limit, just the working set this system's buffers are sized for.
        static constexpr uint32_t kMaxParticles = 65536;

        // Allocates every buffer this pass owns (see this class' own header comment for the full
        // list) via `allocator`, seeds the dead-list with every index 0..kMaxParticles-1 and the
        // counter block with {deadCount=kMaxParticles, aliveCount=0, spawnQueue=0} via a one-shot
        // staging upload (`commandPool`/`queue`), and builds the single VkDescriptorSetLayout +
        // VkDescriptorPool + the 2 ping-pong VkDescriptorSet instances every particle shader binds.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // The descriptor set layout every particle-system compute/graphics pipeline (Subtasks 2-4)
        // must build its VkPipelineLayout against -- exposed so those later subtasks' Init() calls
        // don't need to reconstruct an identical layout themselves.
        VkDescriptorSetLayout GetSetLayout() const { return m_SetLayout; }

        // The ping-pong descriptor set currently holding the MOST RECENTLY WRITTEN particle state
        // (i.e. what a reader -- a future sort/render dispatch -- should bind this frame, before
        // Subtask 2's simulation dispatch, once it exists, advances the index for the NEXT frame's
        // write target). Index 0 both before the first simulate call and immediately after Init().
        VkDescriptorSet GetCurrentSet() const { return m_ParticleSet[m_CurrentIndex]; }
        uint32_t GetCurrentSetIndex() const { return m_CurrentIndex; }

        VkBuffer GetIndirectDrawBufferHandle() const { return m_IndirectDrawBuffer.Handle(); }
        VkBuffer GetCounterBufferHandle() const { return m_CounterBuffer.Handle(); }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        // Double-buffered particle state (see this class' own header comment on ping-ponging).
        GpuBuffer m_ParticleBuffer[2];
        // Free-list of currently-dead particle-buffer slot indices, shared (NOT ping-ponged) across
        // both particle buffers -- a slot index is a position in whichever GpuParticle array is
        // currently "live," not a value tied to one specific physical buffer.
        GpuBuffer m_DeadListBuffer;
        // Compacted list of currently-alive particle-buffer slot indices, rebuilt every frame by
        // Subtask 2's simulation dispatch -- shared (not ping-ponged) for the same reason as above.
        GpuBuffer m_AliveListBuffer;
        // Single 16-byte {deadCount, aliveCount, spawnQueue, _pad0} block -- see ParticleCommon.glsl's
        // own CounterBuffer comment.
        GpuBuffer m_CounterBuffer;
        // sizeof(VkDrawIndirectCommand), GPU_ONLY -- Subtask 3/4's sort/compact step writes
        // `instanceCount` from the current aliveCount each frame; `vertexCount` is fixed at 6 (one
        // unindexed billboard quad, two triangles) and never changes after Init().
        GpuBuffer m_IndirectDrawBuffer;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_ParticleSet[2]{};

        uint32_t m_CurrentIndex = 0;
    };

}
