#pragma once
// World Probe grid (Lumen-style "Translucency Volume" / global illumination volume): a single,
// low-resolution, camera-centered 3D grid of ambient irradiance probes, fully rebuilt every frame
// from the Surface Cache (renderer::SurfaceCachePass's radiance atlas, sampled via the same
// mesh_sdf_trace.glsl + surface_cache_sampling.glsl trace-and-sample primitive
// renderer::SurfaceCacheGIInjectPass already uses -- see WorldProbeInject.comp's own class
// comment). This is INTENDED as what dynamic or unmapped objects -- particle systems, animated
// characters, anything without a Surface Cache Card of its own -- would sample for indirect
// light, via world_probe_sampling.glsl's SampleWorldProbeGrid(), instead of tracing their own rays.
//
// --- Live consumers (since the ScreenTracePass/GICompositePass integration, 2026-07-16) ---
// SampleWorldProbeGrid() is sampled every frame by renderer::ScreenTracePass (its own screen-space
// march's miss fallback -- see ScreenTrace.comp's own comment) and, Debug-only, by
// renderer::GICompositePass's DEBUG_VIEW_SPATIAL_PROBES visualization. This pass's own
// RecordUpdate() runs unconditionally in Release (renderer::ClusterRenderPipeline::RecordFrame()
// hardcodes `worldProbesEnabled = true` there) since the real GPU cost is now paying for a real
// visual contribution; Debug still gates it behind
// renderer::ClusterRenderPipeline::SetDebugWorldProbesEnabled() (main.cpp's 'H' key) purely for
// A/B cost comparison, not because the grid is unconsumed.
//
// --- Incremental toroidal streaming (Phase 6, UE5.8 parity roadmap: adaptive/importance-sampled
// probes) ---
// Pre-Phase-6, this grid was fully rebuilt every single RecordUpdate() call regardless of camera
// motion (~459K rays/frame, kGridResolution^3 probes x kProbeSampleDirections rays, unconditionally)
// -- affordable at this scale, but wasteful: a probe whose surroundings haven't changed does not
// need re-tracing every frame. This class now mirrors renderer::GlobalSDFPass's own incremental
// clipmap-streaming technique (see that class' own header comment for the general rationale): a
// single FIXED, infinite lattice of probe indices in world space, addressed physically via
// `(index mod kGridResolution)` per axis (a toroidal wrap) -- what moves is only which
// kGridResolution^3 WINDOW of that lattice is considered "covered" (recentered on the camera,
// snapped to whole-probe steps for temporal coherence, exactly like GlobalSDFPass's own snapping).
// When the window shifts, only the newly-revealed probes (a thin per-axis slab, mirroring
// GlobalSDFPass::EnqueueDirtyRegionsForLevel's own per-axis decomposition) are queued dirty and
// re-traced over the next few calls (kMaxDirtySlabsPerCall budget) -- probes still within the
// window that were NOT just revealed simply keep their last-traced irradiance, valid indefinitely
// for a probe grid this coarse/low-frequency (the same assumption GlobalSDFPass already relies on
// for its own SDF clipmap). Unlike GlobalSDFPass, there is only ONE level here (no multi-resolution
// clipmap hierarchy) and no per-entity composite-min accumulation (WorldProbeInject.comp computes
// each probe's irradiance in one self-contained dispatch, not a multi-source min-blend), so
// kMaxDirtySlabsPerCall only needs to cover "at most 3 axes shifted this frame" (3), not
// GlobalSDFPass's "4 levels x up to 3 dynamic entities" backlog (16) -- see that member's own
// comment. world_probe_sampling.glsl's SampleWorldProbeGrid() was updated in lockstep (manual
// 8-corner trilinear blend via texelFetch against the wrapped addressing, replacing hardware
// trilinear filtering, which would incorrectly blend across the wrap seam -- the same reasoning
// renderer::SDFRayMarchPass's own NEAREST-at-voxel-center sampling already relies on for
// GlobalSDFPass's own clipmaps).
//
// --- Why no per-texel hemisphere importance sampling (unlike SurfaceCacheGIInjectPass) ---
// A Surface Cache texel's re-injected bounce needs to look correct at close range, under a
// specific surface normal, so SurfaceCacheGIInject.comp fires a full cosine-weighted hemisphere
// (kSampleCountPerTexel Halton samples) oriented by that texel's own normal. A World Probe has no
// surface normal at all (it floats in open space) and is only ever sampled at ambient, low-
// frequency quality by a dynamic object far from any single texel -- a small FIXED set of
// directions (see kProbeSampleDirections), evenly spread over the full sphere rather than a
// hemisphere, is the appropriate (and far cheaper) sampling shape for an omnidirectional ambient
// probe.

#include <cstdint>
#include <deque>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "core/EngineConfig.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;

    // Byte-for-byte mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO (std140). Shared
    // by every pass that samples the world probe grid (GICompositePass, ScreenTracePass) so the
    // layout can never silently drift between two independently-hand-typed copies.
    struct WorldProbeGridParams {
        float gridOriginX = 0.0f, gridOriginY = 0.0f, gridOriginZ = 0.0f;
        float probeSpacing = 0.0f;
        float gridResolution = 0.0f;
        float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
    };
    static_assert(sizeof(WorldProbeGridParams) == 32,
        "WorldProbeGridParams must match world_probe_sampling.glsl's WorldProbeGridParamsUBO exactly (std140 layout)");

    class WorldProbeGridPass {
    public:
        WorldProbeGridPass() = default;

        WorldProbeGridPass(const WorldProbeGridPass&) = delete;
        WorldProbeGridPass& operator=(const WorldProbeGridPass&) = delete;

        // Cubic probe count per axis -- see the class comment's "why a full rebuild" note for why
        // this stays small enough that a full rebuild every frame is affordable.
        static inline uint32_t kGridResolution = 64u;
        // World-space distance between adjacent probes -- together with kGridResolution, this
        // grid covers a (kGridResolution * kProbeSpacing)^3 = 64^3 world-unit cube centered on the
        // camera, recentered every RecordUpdate() call (see that method's own comment).
        static inline float kProbeSpacing = 1.0f;
        // Matches WorldProbeInject.comp's local_size_x/y/z exactly.
        static constexpr uint32_t kWorkgroupSize = 4;
        // Fixed omnidirectional sample set per probe -- see the class comment's own "why no
        // hemisphere sampling" note. Matches WorldProbeInject.comp's kProbeSampleDirections array
        // length exactly.
        static inline uint32_t kProbeSampleDirections = 14u;

        static constexpr VkFormat kGridFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        // Allocates the kGridResolution^3 image3D (kept VK_IMAGE_LAYOUT_GENERAL for its entire
        // lifetime, matching every other long-lived storage image in this codebase) and builds
        // set 0 (grid output + the shared TLAS/vertex/index/draw-range resources, all borrowed
        // unmodified from `rtPass`/`surfaceCache` -- see the class comment on WorldProbeInject.comp's
        // smaller set-0 shape vs. SurfaceCacheGIInjectPass's own) plus the compute pipeline. `traceContext`,
        // `surfaceCache` and `rtPass` must all already be Init'd and must outlive this pass, exactly
        // like renderer::SurfaceCacheGIInjectPass::Init's own contract.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
            const SurfaceCacheRayTracingPass& rtPass);

        void Shutdown();

        // Atmos weather system, Subtask 5: binds renderer::AtmosSkyPass's Sky-View LUT into set 0's
        // binding 5 -- must be called exactly once after both Init() and AtmosSkyPass's own Init(),
        // before the first RecordUpdate() call.
        void SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler);

        // Phase 6 (UE5.8 parity roadmap): recenters the grid's covered WINDOW (snapped to whole-
        // probe steps, so a probe that hasn't left the window keeps its own stable physical texel
        // address across frames -- see the class comment's toroidal-streaming rationale) around
        // `cameraPositionWorld`, enqueues any newly-revealed probes as dirty slabs, then drains up
        // to kMaxDirtySlabsPerCall of them (bounded per-call GPU cost -- a large camera jump may
        // take a few extra calls to fully catch up, exactly like GlobalSDFPass's own backlog
        // draining). Each drained slab's probes are traced (SWRT or HWRT per `traceMode`, matching
        // renderer::SurfaceCacheGIInjectPass::RecordInject's own traceMode convention) and written
        // into their wrapped texel address. Caller owns every synchronization barrier before
        // (Surface Cache radiance atlas + TLAS visible) and after (grid writes visible to a later
        // sampled read, e.g. renderer::ScreenTracePass) this call -- same discipline as
        // SurfaceCacheGIInjectPass::RecordInject.
        // `sunDirectionWorld` (Atmos Subtask 5): fed to a probe-ray miss's own Sky-View LUT sample,
        // points FROM the light TOWARD the scene.
        void RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld,
            const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld);

        VkImageView GetGridView() const { return m_GridView; }
        VkSampler GetGridSampler() const { return m_GridSampler; }
        // Phase 6: the world-space MINIMUM corner of the grid's CURRENTLY COVERED WINDOW (i.e.
        // `snappedCenterProbe * kProbeSpacing - halfExtent`), as of the most recent RecordUpdate()
        // call -- NOT necessarily where texel (0,0,0) physically lives anymore (that texel's
        // CONTENT is whichever world-probe-index currently wraps to it, `index mod
        // kGridResolution`, which shifts as the window moves -- see the class comment). Still
        // exactly what world_probe_sampling.glsl's SampleWorldProbeGrid needs (plus
        // kProbeSpacing/kGridResolution) to convert a world position into the grid's own probe-
        // index space before wrapping.
        const maths::vec3& GetGridOriginWorld() const { return m_GridOriginWorld; }

    private:
        // One axis-aligned "newly revealed" region, in absolute world-probe-index space -- NOT yet
        // split into wrap-contiguous pieces (that split happens when the slab is actually drained,
        // exactly like GlobalSDFPass::DirtySlab's own comment explains). No `level` field (unlike
        // GlobalSDFPass::DirtySlab) -- this grid has only one.
        struct DirtySlab {
            int32_t probeMin[3] = { 0, 0, 0 };
            int32_t probeMax[3] = { 0, 0, 0 }; // Exclusive.
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        VkImage m_GridImage = VK_NULL_HANDLE;
        VmaAllocation m_GridAllocation = VK_NULL_HANDLE;
        VkImageView m_GridView = VK_NULL_HANDLE;
        VkSampler m_GridSampler = VK_NULL_HANDLE; // Linear + CLAMP_TO_EDGE -- see .cpp's sampler comment.

        maths::vec3 m_GridOriginWorld{};

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE; // set 0.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        uint32_t m_FrameIndex = 0; // Halton sequence temporal offset, advanced every RecordUpdate() call.

        // Phase 6: which world-probe-index currently sits at the window's center, per axis -- the
        // toroidal-streaming analog of GlobalSDFPass::ClipmapLevel::snappedCenterVoxel (one grid,
        // so no per-level array). `m_HasValidWindow` is false until the first RecordUpdate() call,
        // mirroring ClipmapLevel::hasValidWindow exactly (forces one full-volume dirty slab).
        int32_t m_SnappedCenterProbe[3] = { 0, 0, 0 };
        bool m_HasValidWindow = false;

        std::deque<DirtySlab> m_PendingSlabs;

        // Bounded per-RecordUpdate() drain budget. A single grid (not GlobalSDFPass's 4 clipmap
        // levels) with no per-entity composite step needs to cover at most 3 newly-dirtied slabs
        // per call (one per axis, the maximum a single camera-motion step can shift) -- so, unlike
        // GlobalSDFPass::kMaxDirtySlabsPerCall (16, sized for 4 levels x multiple dynamic entities'
        // worth of backlog), 3 drains everything in one call under ordinary continuous motion; only
        // an exceptionally large single-frame camera jump would ever leave a residual backlog for
        // a later call to finish.
        static constexpr uint32_t kMaxDirtySlabsPerCall = 3;

        void EnqueueDirtyRegionsForGrid(const maths::vec3& cameraPositionWorld);
        void DrainAndRecordSlabs(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld);
        void RecordSlab(VkCommandBuffer cmd, const DirtySlab& slab, const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld);
    };

}
