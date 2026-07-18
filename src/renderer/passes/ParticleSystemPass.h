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
// Double buffering (m_ParticleBuffer[2]/m_ParticleSet[2]): allocated per the plan doc's explicit
// request, but Subtask 2's simulation dispatch (see RecordSimulate()'s own comment) deliberately
// does a plain in-place read-modify-write on whichever buffer GetCurrentSet() currently names --
// unlike renderer::ReflectionPass's own m_Slots[2] ping-pong (which genuinely needs last frame's
// separate history for temporal accumulation), no particle-system consumer reads a "previous frame"
// snapshot yet, and each simulation thread only ever touches its own particle slot (no cross-particle
// reads that would make in-place mutation unsafe within one dispatch). m_CurrentIndex therefore stays
// 0 unless/until a future subtask (e.g. async-compute overlap between this frame's sim and last
// frame's render, mirroring MegaLightsPass's own async-compute precedent) actually needs the second
// buffer -- no Advance()/flip method exists yet, since nothing would call it.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class AtmosClimatePass;
    class GlobalSDFPass;

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
        // staging upload (`commandPool`/`queue`), builds the single VkDescriptorSetLayout +
        // VkDescriptorPool + the 2 ping-pong VkDescriptorSet instances every particle shader binds,
        // and (Subtask 2) the ParticleSimulation.comp pipeline itself -- its own descriptor set 1
        // borrows `atmosClimate`'s AtmosGlobalsUBO (wind) and `globalSDF`'s 4 clipmap levels
        // (collision), so both must already be Init'd and must outlive this pass, same "borrowed,
        // unmodified" convention as e.g. renderer::AtmosVolumetricFogPass::Init's own parameters.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const AtmosClimatePass& atmosClimate, const GlobalSDFPass& globalSDF);

        void Shutdown();

        // The descriptor set layout every particle-system compute/graphics pipeline (Subtasks 3-4)
        // must build its VkPipelineLayout against -- exposed so those later subtasks' Init() calls
        // don't need to reconstruct an identical layout themselves.
        VkDescriptorSetLayout GetSetLayout() const { return m_SetLayout; }

        // The ping-pong descriptor set currently holding the MOST RECENTLY WRITTEN particle state
        // (i.e. what a reader -- a future sort/render dispatch -- should bind this frame). See this
        // class' own header comment on why m_CurrentIndex currently never changes.
        VkDescriptorSet GetCurrentSet() const { return m_ParticleSet[m_CurrentIndex]; }
        uint32_t GetCurrentSetIndex() const { return m_CurrentIndex; }

        VkBuffer GetIndirectDrawBufferHandle() const { return m_IndirectDrawBuffer.Handle(); }
        VkBuffer GetCounterBufferHandle() const { return m_CounterBuffer.Handle(); }

        // Dispatches ParticleSimulation.comp in two passes against GetCurrentSet() (see that
        // shader's own header comment for the full spawn/update contract):
        //   1. Spawn: resets CounterBuffer.aliveCount to 0 (full rebuild, see below) and
        //      CounterBuffer.spawnQueue to `spawnCount` via two small vkCmdUpdateBuffer calls, then
        //      dispatches ceil(spawnCount / 64) workgroups that each pop one dead-list slot and
        //      initialize a fresh particle there (position/velocity jittered around
        //      `emitterPositionWorld`, life = maxLife). Does NOT touch the alive-list.
        //   2. Update: dispatches ceil(kMaxParticles / 64) workgroups, one thread per particle SLOT
        //      (not per alive-list entry -- covers pre-existing AND just-spawned particles
        //      uniformly). A dead slot (life <= 0) is skipped outright. A live slot integrates
        //      gravity + wind (AtmosNoiseCommon.glsl's SampleWindVelocity, fed by `atmosClimate`'s
        //      AtmosGlobalsUBO) + drag, resolves Global SDF collisions (central-difference normal,
        //      reflect() split into elastic-normal/frictional-tangential components), then either
        //      returns its index to the dead-list (life just expired) or appends it to the
        //      alive-list (still alive) -- this is the ONLY place either list is written from an
        //      existing particle, so aliveCount's reset-to-0-then-rebuild above is exactly correct
        //      (never double-counts a particle spawn() also touched).
        // `time` feeds both wind domain-scrolling and is stored nowhere -- purely a per-call input.
        // Caller owns every barrier before (environment/GlobalSDF data visible to COMPUTE_SHADER)
        // and after (this call's own trailing VkMemoryBarrier2 only covers COMPUTE_SHADER-stage
        // consumers, e.g. a future Subtask 3 sort dispatch -- a render-stage consumer, Subtask 4,
        // will need its own additional barrier at that time).
        // `globalSDF` is passed fresh every call (NOT retained from Init(), unlike
        // AtmosGlobalsUBO/the clipmap image views which are bound once and never change) because its
        // 4 levels' currently-covered windows recenter every frame as the camera moves -- mirrors
        // renderer::SDFRayMarchPass::RecordRayMarch's own identical "views bound once, per-frame
        // window data passed fresh each call" split.
        void RecordSimulate(VkCommandBuffer cmd, const GlobalSDFPass& globalSDF, float dt, float time,
            const float emitterPositionWorld[3], uint32_t spawnCount);

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

        // Subtask 2 -- ParticleSimulation.comp's set 1 (environment: AtmosGlobalsUBO + Global SDF
        // clipmaps), never ping-ponged (both borrowed resources are single, stable buffers/images
        // for this pass' entire lifetime) and never re-written after Init() -- unlike
        // renderer::SurfaceCachePass's deferred SetAtmosCloudShadow()-style setters, both
        // dependencies are already Init'd by the time ClusterRenderPipeline::Init() reaches this
        // pass' own Init() call, so no separate deferred-binding call is needed.
        VkSampler m_ClipmapSampler = VK_NULL_HANDLE; // Nearest -- must not linearly filter across a toroidal clipmap's wrap seam, see SDFRayMarchPass's own identical sampler comment.
        VkDescriptorSetLayout m_EnvironmentSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_EnvironmentDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_EnvironmentSet = VK_NULL_HANDLE;

        VkPipelineLayout m_SimPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_SimPipeline = VK_NULL_HANDLE;
    };

}
