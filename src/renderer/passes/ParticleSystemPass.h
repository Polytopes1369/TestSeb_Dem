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

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class AtmosClimatePass;
    class GlobalSDFPass;
    class ClusterResolvePass;
    class VirtualShadowMapPass;
    class WorldProbeGridPass;

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
        // Precipitation feature: top 2 bits are a particle "kind" tag (see ParticleCommon.glsl's
        // PackParticleSeed/UnpackParticleKind) -- this field is otherwise dead after GPU-side spawn
        // (no CPU reader, no other GPU consumer treats it as a raw PRNG value again), so it is the
        // only place a per-particle kind can live without growing this already byte-exact struct.
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
        // own "seed the dead-list" comment). Must stay a power of two (ParticleSort.comp's bitonic
        // network requirement, see that shader's own header comment).
        //
        // Subtask 6 finding (real bug, found only once RecordSimulate/RecordSort/RecordDraw were
        // actually wired into RecordFrame and run every frame for the first time -- Init-time
        // validation alone never dispatches these): the original value here, 65536, made
        // RecordSort()'s 136 full-buffer bitonic compare-exchange dispatches (see that method's own
        // comment) expensive enough, EVERY single frame, to blow past the Windows driver's TDR
        // (Timeout Detection and Recovery) budget under Debug + validation-layer overhead, killing
        // the device (VK_ERROR_DEVICE_LOST) partway through --test-pipeline. At
        // config::particles::SPAWN_RATE_PER_SECOND's own default (200/s) and a ~2-4s particle
        // lifetime, steady-state alive count is only a few hundred -- 65536 slots was ~100x more
        // capacity than this demo's own single test emitter would ever actually use. Reduced to
        // 4096 (2^12): still generous headroom over realistic steady-state counts, and cuts
        // RecordSort()'s own dispatch count from 136 to 78 ((12*13)/2) with each dispatch covering
        // 16x fewer workgroups -- resolved the hang. If a future scene needs a much larger particle
        // budget, RecordSort's O(log2(N)^2) full-buffer network is the thing to replace first (a
        // shared-memory local-merge bitonic sort, or an indirect dispatch sized to the actual
        // aliveCount rounded up to the next power of two instead of always sorting the full fixed
        // capacity), not just raising this constant back up.
        static constexpr uint32_t kMaxParticles = 4096;

        // Allocates every buffer this pass owns (see this class' own header comment for the full
        // list) via `allocator`, seeds the dead-list with every index 0..kMaxParticles-1 and the
        // counter block with {deadCount=kMaxParticles, aliveCount=0, spawnQueue=0} via a one-shot
        // staging upload (`commandPool`/`queue`), builds the single VkDescriptorSetLayout +
        // VkDescriptorPool + the 2 ping-pong VkDescriptorSet instances every particle shader binds,
        // and (Subtask 2) the ParticleSimulation.comp pipeline itself -- its own descriptor set 1
        // borrows `atmosClimate`'s AtmosGlobalsUBO (wind) and `globalSDF`'s 4 clipmap levels
        // (collision), so both must already be Init'd and must outlive this pass, same "borrowed,
        // unmodified" convention as e.g. renderer::AtmosVolumetricFogPass::Init's own parameters.
        // (Subtask 4) Also builds the ParticleRender.vert/.frag graphics pipeline -- its own set 2
        // borrows `resolvePass`'s GBuffer depth copy (GetOutputDepthView(), soft-particle
        // reconstruction) with a dedicated NEAREST sampler, same one-time-bind convention as the
        // Subtask 2 environment set above. `colorFormat`/`depthFormat` are the render target formats
        // (matching e.g. renderer::TransparentForwardPass::Init's own plain-VkFormat parameters) --
        // this pass draws onto the SAME color image and real depth-stencil attachment every other
        // forward pass (Hero/Water/TransparentForward) targets, not `resolvePass`'s own GBuffer
        // images (those are read-only inputs here, see GetOutputDepthView()'s own comment above).
        // (Subtask 5) Also builds a THIRD render-pipeline set (set 3, "lighting"): `vsm`'s 4 Virtual
        // Shadow Map resources (physical atlas + sampler, page table, feedback, sun clipmap levels --
        // same 4-resource contract as renderer::TransparentForwardPass's own SetVirtualShadowMap, but
        // taken directly as an Init() parameter here rather than a deferred setter, since
        // ClusterRenderPipeline::Init() already has `vsm` fully ready by the time it reaches this
        // call) and `worldProbes`' grid + a ONE-TIME-uploaded WorldProbeGridParamsUBO (mirrors
        // TransparentForwardPass's own identical "static addressing" simplification -- the grid's
        // toroidal recentering is not re-uploaded per frame here either, see that class' own Init()
        // comment for why that limitation already exists elsewhere in this codebase).
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const AtmosClimatePass& atmosClimate, const GlobalSDFPass& globalSDF, const ClusterResolvePass& resolvePass,
            const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
            VkFormat colorFormat, VkFormat depthFormat);

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

#ifndef NDEBUG
        // Debug-only observability (Subtask 6's own "GPU Particles: alive/max" HUD/ImGui readout):
        // the alive-particle count as of the last completed GPU->CPU readback, 1-2 frames stale
        // (RecordSort() copies CounterBuffer.aliveCount into a small host-visible buffer every call,
        // with no fence-wait for that specific copy -- reading whatever bytes are currently there is
        // the same "necessarily stale, purely observability" convention this codebase's own World
        // Partition streaming overlay already accepts). Never compiled into Release -- per CLAUDE.md's
        // stats-overlay exclusion rule, this whole accessor and its backing buffer exist only in
        // Debug builds; RecordSort()'s own copy into that buffer is equally `#ifndef NDEBUG`-guarded.
        uint32_t GetLastAliveCountApprox() const;
#endif

        // Dispatches ParticleSimulation.comp in up to three passes against GetCurrentSet() (see that
        // shader's own header comment for the full spawn/update contract):
        //   1. Spawn embers: resets CounterBuffer.aliveCount to 0 (full rebuild, see below) and
        //      CounterBuffer.spawnQueue to `spawnCount` via two small vkCmdUpdateBuffer calls, then
        //      dispatches ceil(spawnCount / 64) workgroups that each pop one dead-list slot and
        //      initialize a fresh particle there (position/velocity jittered around
        //      `emitterPositionWorld`, life = maxLife). Does NOT touch the alive-list. Skipped
        //      entirely when spawnCount == 0.
        //   2. Spawn precipitation (rain/snow -- Atmos weather system precipitation feature): first
        //      uploads `precipCenterWorld`/the fall-speed and spawn-geometry constants into this
        //      pass' own PrecipitationParamsUBO (m_PrecipitationParamsBuffer, environment set binding
        //      2), then dispatches ceil(precipSpawnCount / 64) workgroups against ParticleSimulation.
        //      comp's mode == 2, each popping one dead-list slot (the SAME shared free-list step 1
        //      above draws from -- see ParticleSimulation.comp's own TryPopDeadListSlot comment for
        //      why this is exactly the "share the pool, back off gracefully under pressure" contract
        //      the particle-budget requirement asks for) and spawning a rain/snow particle inside a
        //      horizontal box "shell" centered on `precipCenterWorld`'s XZ (see ParticleSimulation.
        //      comp's own SpawnPrecipitationParticle comment). `precipKind` (kParticleKindRain/Snow,
        //      ParticleCommon.glsl) is resolved by the CALLER from the current Atmos temperature --
        //      this method does not read config::atmos:: itself, matching the existing convention
        //      that config values are read once at the call site (renderer::ClusterRenderPipeline)
        //      and passed down as plain parameters, same as `emitterPositionWorld`/`spawnCount` above
        //      already are for embers. Skipped entirely when precipSpawnCount == 0.
        //   3. Update: dispatches ceil(kMaxParticles / 64) workgroups, one thread per particle SLOT
        //      (not per alive-list entry -- covers pre-existing AND just-spawned particles
        //      uniformly, from EITHER spawn pass above). A dead slot (life <= 0) is skipped outright.
        //      A live slot's physics branches on its own stored kind (ParticleCommon.glsl's
        //      UnpackParticleKind): embers integrate gravity + wind (AtmosNoiseCommon.glsl's
        //      SampleWindVelocity, fed by `atmosClimate`'s AtmosGlobalsUBO) + drag and resolve Global
        //      SDF collisions with a bounce (central-difference normal, reflect() split into
        //      elastic-normal/frictional-tangential components); rain/snow instead relax toward a
        //      fixed fall speed (+ wind drift, + a sine wobble for snow) and are simply absorbed (no
        //      bounce) on any Global SDF contact or once they sink below the camera-relative recycle
        //      floor. Either way the particle then either returns its index to the dead-list (life
        //      just expired) or appends it to the alive-list (still alive) -- this is the ONLY place
        //      either list is written from an existing particle, so aliveCount's reset-to-0-then-
        //      rebuild above is exactly correct (never double-counts a particle either spawn pass
        //      also touched).
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
            const float emitterPositionWorld[3], uint32_t spawnCount,
            const float precipCenterWorld[3], uint32_t precipSpawnCount, uint32_t precipKind,
            float precipSpawnRadiusMeters, float precipSpawnHeightAboveCenterMeters,
            float precipSpawnBandThicknessMeters, float precipFloorBelowCenterMeters,
            float precipRainFallSpeedMps, float precipSnowFallSpeedMps, float precipSnowWobbleStrength);

        // Sorted {particleIndex, depthKey} pairs, back-to-front (farthest first) among the first
        // GetCounterBufferHandle()-reported aliveCount entries -- see ParticleSort.comp's own header
        // comment for why the buffer is always kMaxParticles long (a power of two, required for
        // bitonic sort) rather than sized to the frame's actual (non-power-of-two) aliveCount.
        // Exposed so Subtask 4's render pass can bind it directly, same "borrow a raw handle"
        // convention as GetIndirectDrawBufferHandle().
        VkBuffer GetSortedPairsBufferHandle() const { return m_SortedPairsBuffer.Handle(); }

        // Dispatches ParticleSort.comp (see that shader's own header comment for the full InitKeys/
        // CompareExchange contract) against GetCurrentSet()'s particle state: one InitKeys pass,
        // then the full O(log2(kMaxParticles)^2) bitonic compare-exchange network (78 dispatches for
        // kMaxParticles == 4096 -- see that constant's own comment on why it is NOT 65536), each
        // followed by its own VkMemoryBarrier2 (global-memory bitonic sort has a genuine
        // read-after-write dependency between every single step -- no shared-memory local-merge
        // optimization exists yet, see ParticleSort.comp's own comment).
        // Finishes by copying CounterBuffer.aliveCount into the indirect-draw buffer's own
        // `instanceCount` field (a GPU-side vkCmdCopyBuffer, no CPU readback) so a future indirect
        // draw call (Subtask 4) always reflects this frame's real alive count with no extra work at
        // that call site. `cameraPositionWorld`/`cameraForwardWorld` feed InitKeys' own depth-key
        // projection. Caller owns the barrier before this call (particle/alive-list state visible to
        // COMPUTE_SHADER, e.g. RecordSimulate()'s own trailing barrier already covers this) and
        // after it (this call's own trailing barrier only covers COMPUTE_SHADER/TRANSFER-stage
        // consumers -- a render-stage consumer, Subtask 4, will need its own additional barrier,
        // including one for INDIRECT_COMMAND_READ on the indirect-draw buffer specifically).
        void RecordSort(VkCommandBuffer cmd, const float cameraPositionWorld[3], const float cameraForwardWorld[3]);

        // Draws every alive particle (see ParticleRender.vert's own header comment) via
        // vkCmdDrawIndirect against `colorView`/`depthView` -- the SAME forward-pass color/depth
        // attachment pair renderer::TransparentForwardPass/TessellationPass/WaterForwardPass
        // already target (`colorImage` is needed for a trailing layout-transition barrier, same
        // convention as those passes' own RecordDraw). Depth is bound read-only (loadOp=LOAD,
        // depthWriteEnable=FALSE, VK_COMPARE_OP_GREATER reversed-Z) -- particles are correctly
        // hidden behind opaque geometry but never occlude each other via the depth buffer (their
        // relative order comes entirely from Subtask 3's back-to-front sort instead). `viewProj`
        // mirrors renderer::ScreenSpaceEffectsPass::RecordAmbientOcclusion's own convention (a single
        // combined matrix, `.Inverse()`'d internally for the fragment shader's soft-particle
        // reconstruction) rather than separate view/proj. `cameraRightWorld`/`cameraUpWorld` are the
        // camera's own right/up basis vectors (already computed once per frame by whatever call site
        // tracks the camera, e.g. CameraFrameInfo) -- billboarding needs them directly, not
        // re-derived from `viewProj` in-shader. `softFadeDistanceWorld` is the world-unit band a
        // particle fades over as it nears intersecting opaque geometry (Subtask 6's ImGui panel will
        // make this tunable; a fixed default is used until then). Caller owns the barrier before this
        // call (particle/sorted-pair/indirect-draw-buffer state visible to VERTEX_SHADER/
        // DRAW_INDIRECT -- Subtask 3's RecordSort does not fully cover this on its own, see that
        // method's own comment) and after it (this call ends with vkCmdEndRendering and its own
        // color-attachment-to-whatever-the-caller-needs-next transition, same pattern as
        // TransparentForwardPass::RecordDraw's own trailing barrier).
        // (Subtask 5) `sunDirectionWorld`/`sunColor`/`sunIntensity` feed the fragment shader's own
        // shadowed-sun + indirect-diffuse combination (no BRDF/phase term -- a billboard has no real
        // surface normal, see ParticleRender.frag's own header comment). `refractionOffsetView` is
        // renderer::TransparentForwardPass::GetRefractionOffsetView() -- this pass becomes that
        // image's first SECOND writer (bound with loadOp=LOAD at layout GENERAL the entire time, no
        // extra barrier dance needed: it is already GENERAL by the time TransparentForwardPass's own
        // RecordDraw finishes, see ParticleSystemPass.cpp's own comment, and dynamic rendering
        // legally accepts GENERAL for any attachment use). `heatShimmerStrength` is a PER-DRAW-CALL
        // toggle, not per-particle -- GpuParticle's already-merged 64-byte layout has no spare
        // "isRefractive" flag, and a demoscene emitter is realistically one thermal "kind" or
        // another (Subtask 6's ImGui panel will make this tunable per emitter).
        void RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView, VkImageView depthView,
            VkImageView refractionOffsetView, VkExtent2D renderExtent,
            const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
            const maths::vec3& cameraRightWorld, const maths::vec3& cameraUpWorld,
            const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
            float softFadeDistanceWorld, float heatShimmerStrength, float globalTimeSeconds);

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

        // Precipitation feature -- environment set binding 2 (PrecipitationParamsUBO). Unlike
        // AtmosGlobalsUBO/the clipmaps above (borrowed from other passes, written once at Init() and
        // never again by THIS class), this buffer is owned AND updated by ParticleSystemPass itself,
        // every single RecordSimulate() call (camera-relative spawn-shell center changes every frame)
        // -- see that method's own comment. 48 bytes, GPU_ONLY, same vkCmdUpdateBuffer-then-barrier
        // idiom as AtmosClimatePass::RecordUpdate's own AtmosGlobalsUBO upload.
        GpuBuffer m_PrecipitationParamsBuffer;

        VkPipelineLayout m_SimPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_SimPipeline = VK_NULL_HANDLE;

        // Subtask 3 -- ParticleSort.comp's own set 1 (a single SortedPairsBuffer binding, unrelated
        // to Subtask 2's environment set 1 above -- each compute pipeline has its OWN independent
        // set 1, they are never bound together in the same dispatch). kMaxParticles * 8 bytes
        // (uint index + float key per entry), GPU_ONLY, never host-written (ParticleSort.comp's own
        // InitKeys pass is this buffer's only writer, every single frame it runs).
        GpuBuffer m_SortedPairsBuffer;
        VkDescriptorSetLayout m_SortSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_SortDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_SortSet = VK_NULL_HANDLE;
        VkPipelineLayout m_SortPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_SortPipeline = VK_NULL_HANDLE;

#ifndef NDEBUG
        // Subtask 6, Debug-only: see GetLastAliveCountApprox()'s own comment. 4 bytes, CPU_TO_GPU,
        // persistently mapped -- RecordSort() vkCmdCopyBuffer's CounterBuffer.aliveCount into this
        // every call, no fence-wait (deliberately stale-tolerant, observability only).
        GpuBuffer m_AliveCountReadbackBuffer;
#endif

        // Subtask 4 -- ParticleRender.vert/.frag's own set 2 (ParticleRenderParamsUBO + the sampled
        // GBuffer depth copy borrowed from `resolvePass`, see Init()'s own comment) plus the graphics
        // pipeline itself. Set 0/1 for this pipeline are m_SetLayout/m_SortSetLayout, REUSED
        // unmodified from Subtasks 1/3 (the render pass reads the exact same particle state and
        // sorted-pair order those own, no reason to duplicate either descriptor set).
        GpuBuffer m_RenderParamsBuffer; // ParticleRenderParamsUBO, std140, GPU_ONLY, updated once per RecordDraw() call.
        VkSampler m_SceneDepthSampler = VK_NULL_HANDLE; // Nearest -- avoids blending genuinely different NDC depths across a geometry silhouette edge.
        VkDescriptorSetLayout m_RenderSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_RenderDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_RenderSet = VK_NULL_HANDLE;
        VkPipelineLayout m_RenderPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_RenderPipeline = VK_NULL_HANDLE;

        // Subtask 5 -- ParticleRender.frag's own set 3 ("lighting"): `vsm`'s 4 Virtual Shadow Map
        // resources (borrowed, bound once, never re-written) + `worldProbes`' grid + a one-time-
        // uploaded WorldProbeGridParamsUBO (see Init()'s own comment on why this is a static upload,
        // not per-frame). Same one-time-bind convention as the Subtask 2 environment set.
        GpuBuffer m_WorldProbeGridParamsBuffer; // WorldProbeGridParamsUBO, std140, GPU_ONLY, filled once at Init().
        VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_LightingDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_LightingSet = VK_NULL_HANDLE;
    };

}
