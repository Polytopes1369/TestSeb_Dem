#pragma once
// World Probe grid (Lumen-style "Translucency Volume" / global illumination volume): a
// camera-centered, multi-level 3D clipmap of ambient irradiance probes, built from the Surface
// Cache (renderer::SurfaceCachePass's radiance atlas, sampled via the same mesh_sdf_trace.glsl +
// surface_cache_sampling.glsl trace-and-sample primitive renderer::SurfaceCacheGIInjectPass already
// uses -- see WorldProbeInject.comp's own class comment). This is what dynamic or unmapped
// objects -- particle systems, animated characters, anything without a Surface Cache Card of its
// own -- sample for indirect light, via world_probe_sampling.glsl's SampleWorldProbeGrid(), instead
// of tracing their own rays.
//
// --- Live consumers ---
// SampleWorldProbeGrid() is sampled every frame by renderer::ScreenTracePass -- as its own
// screen-space march's miss fallback in DEFAULT (config::lumen::GIMode::HighQuality) mode, or as
// the PRIMARY (near-exclusive) irradiance source in F1's Lite mode, see ScreenTrace.comp's own
// giMode branch -- and, Debug-only, by renderer::GICompositePass's DEBUG_VIEW_SPATIAL_PROBES
// visualization. This pass' own RecordUpdate() runs unconditionally in Release (renderer::
// ClusterRenderPipeline::RecordFrame() hardcodes `worldProbesEnabled = true` there) regardless of
// GI mode -- Lite mode needs the grid as its PRIMARY term, and HighQuality mode still needs it kept
// live as ScreenTracePass's own fallback, so there is no GI mode in which this pass can be skipped.
// Debug still gates it behind renderer::ClusterRenderPipeline::SetDebugWorldProbesEnabled()
// (main.cpp's 'H' key) purely for A/B cost comparison, not because the grid is unconsumed.
//
// --- F1 ("Lumen Lite", UE5.8 parity roadmap): multi-level clipmap ---
// Pre-F1, this class held exactly ONE grid (a single kGridResolution^3 window, camera-centered,
// covering a comparatively small world-space cube). Real Lumen's "Radiance Field" (what UE5.8 calls
// its medium-quality "Lumen Lite" mode) needs coverage far beyond what one fine grid can afford at
// a useful probe density, so this class now mirrors renderer::GlobalSDFPass's own multi-level
// clipmap pattern exactly (see that class' own header comment for the general rationale): kLevelCount
// independent levels, level L's probe spacing = kProbeSpacing * 2^L (so level L covers 2^L times the
// world-space extent of level 0 at the SAME kGridResolution^3 probe count -- coarser, not bigger-
// but-denser), each an independent toroidally-streamed window with its own snapped center/dirty-slab
// backlog (see EnqueueDirtyRegionsForLevel/DrainAndRecordSlabs below, byte-for-byte the same
// per-level bookkeeping GlobalSDFPass::ClipmapLevel uses). world_probe_sampling.glsl's
// SampleWorldProbeGrid() selects the finest level whose covered window contains the query position
// (falling back coarser as needed) and cross-blends across the boundary between two adjacent levels
// so a query position crossing that boundary never pops -- see that function's own comment.
//
// --- F1: probe occlusion (DDGI-style Chebyshev visibility) ---
// A pure trilinear blend of 8 neighboring probes' irradiance leaks light through thin walls/corners
// whenever one or more of those 8 probes sits on the OTHER side of an occluder from the point being
// shaded (its irradiance is correct for ITS OWN location, not for a point on the far side of a
// wall). Real DDGI/Lumen Lite fixes this by additionally storing, per probe, statistics of how far
// each traced ray got before hitting something (mean hit distance + mean-SQUARED hit distance) and
// down-weighting a corner probe at sample time via a Chebyshev one-sided variance test whenever the
// query point is farther from that probe than the probe's own mean hit distance (a strong signal an
// occluder sits between them). This class stores that mean/mean-squared pair PER PROBE, aggregated
// over every one of WorldProbeInject.comp's kProbeSampleDirections rays (a deliberate scope
// reduction from full per-direction DDGI visibility, which would need a small octahedral tile per
// probe instead of one texel -- this engine's coarse, ambient-only probe grid does not warrant that
// extra storage/complexity; an omnidirectional aggregate still captures "this probe is embedded
// near/inside geometry" vs. "this probe sits in open air," which is what actually prevents most
// wall/corner leaking in practice). See world_probe_sampling.glsl's own ChebyshevVisibility().

#include <cstdint>
#include <deque>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "core/EngineConfig.h"
#include "renderer/vulkan/RenderPass.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;

    // Byte-for-byte mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO (std140). Shared
    // by every pass that samples the world probe grid (GICompositePass, ScreenTracePass) so the
    // layout can never silently drift between two independently-hand-typed copies. F1: now carries
    // one (origin, spacing) pair PER LEVEL (each packed as a vec3+float = one vec4 slot, matching
    // the shader-side WorldProbeGridLevelParams struct's own std140 array-of-struct layout) instead
    // of a single flat origin/spacing pair.
    struct WorldProbeGridParams {
        struct Level {
            float originX = 0.0f, originY = 0.0f, originZ = 0.0f;
            float spacing = 0.0f;
        };
        Level levels[3];
        float gridResolution = 0.0f;
        float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
    };
    static_assert(sizeof(WorldProbeGridParams) == 64,
        "WorldProbeGridParams must match world_probe_sampling.glsl's WorldProbeGridParamsUBO exactly (std140 layout)");

    // Migrated to RenderPass<Derived> (see renderer/vulkan/RenderPass.h): Init()/Shutdown() are
    // inherited. Note InitImpl() still calls the inherited Shutdown() as its own first statement
    // (preserved from the original Init(), which self-reinitializes -- see ShadowMapPass's own
    // migration comment for the exact same pattern and why m_Device/m_Allocator must be
    // re-assigned right after that call).
    class WorldProbeGridPass : public RenderPass<WorldProbeGridPass> {
        friend class RenderPass<WorldProbeGridPass>; // Lets Init() call our private InitImpl().

    public:
        WorldProbeGridPass() = default;

        WorldProbeGridPass(const WorldProbeGridPass&) = delete;
        WorldProbeGridPass& operator=(const WorldProbeGridPass&) = delete;

        // F1: number of clipmap levels -- mirrors GlobalSDFPass::kLevelCount's own fixed-cap
        // convention (compile-time, not runtime-tiered: every quality profile gets the same level
        // COUNT, only kGridResolution/kProbeSpacing are tier-scaled, exactly like GlobalSDFPass's
        // own kClipmapResolution vs. kLevelCount split).
        static constexpr uint32_t kLevelCount = 3;

        // Cubic probe count per axis, PER LEVEL -- see the class comment's "why a full rebuild" note
        // for why this stays small enough that a full rebuild every frame is affordable.
        static inline uint32_t kGridResolution = 64u;
        // Level 0's world-space distance between adjacent probes; level L's is kProbeSpacing * 2^L
        // (see the class comment's own multi-level note) -- together with kGridResolution, level L
        // covers a (kGridResolution * kProbeSpacing * 2^L)^3 world-unit cube centered on the camera,
        // recentered every RecordUpdate() call (see that method's own comment).
        static inline float kProbeSpacing = 1.0f;
        // Matches WorldProbeInject.comp's local_size_x/y/z exactly.
        static constexpr uint32_t kWorkgroupSize = 4;
        // Fixed omnidirectional sample set per probe -- see the class comment's own "why no
        // hemisphere sampling" note. Matches WorldProbeInject.comp's kProbeSampleDirections array
        // length exactly.
        static inline uint32_t kProbeSampleDirections = 14u;

        static constexpr VkFormat kGridFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        // F1: per-probe occlusion (mean, mean-squared hit distance) -- see the class comment's own
        // "probe occlusion" note. RG16F is plenty of precision for a world-unit distance pair at
        // this grid's scale.
        static constexpr VkFormat kOcclusionFormat = VK_FORMAT_R16G16_SFLOAT;

        // Level L's world-space distance between adjacent probes (kProbeSpacing * 2^L).
        static float GetLevelSpacing(uint32_t level) { return kProbeSpacing * static_cast<float>(1u << level); }

        // Allocates, for every level, the kGridResolution^3 image3D pair (irradiance + occlusion,
        // both kept VK_IMAGE_LAYOUT_GENERAL for their entire lifetime, matching every other
        // long-lived storage image in this codebase) and builds that level's own set 0 (grid +
        // occlusion output + the shared TLAS/vertex/index/draw-range resources, all borrowed
        // unmodified from `rtPass`/`surfaceCache` -- see the class comment on WorldProbeInject.comp's
        // smaller set-0 shape vs. SurfaceCacheGIInjectPass's own) plus ONE compute pipeline shared by
        // every level (only the bound descriptor set differs per level). `traceContext`,
        // `surfaceCache` and `rtPass` must all already be Init'd and must outlive this pass, exactly
        // like renderer::SurfaceCacheGIInjectPass::Init's own contract.
        // Init(VkDevice, VmaAllocator, VkCommandPool, VkQueue, const SurfaceCacheTraceContext&,
        // const SurfaceCachePass&, const SurfaceCacheRayTracingPass&) -> bool and Shutdown() are
        // inherited from RenderPass<WorldProbeGridPass>; see InitImpl() below.

        // Atmos weather system, Subtask 5: binds renderer::AtmosSkyPass's Sky-View LUT into every
        // level's own set 0 binding 6 -- must be called exactly once after both Init() and
        // AtmosSkyPass's own Init(), before the first RecordUpdate() call.
        void SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler);

        // Phase 6 (UE5.8 parity roadmap) / F1 (multi-level): for EVERY level, recenters that level's
        // covered WINDOW (snapped to whole-probe steps, so a probe that hasn't left the window keeps
        // its own stable physical texel address across frames -- see the class comment's toroidal-
        // streaming rationale) around `cameraPositionWorld`, enqueues any newly-revealed probes as
        // dirty slabs, then drains up to kMaxDirtySlabsPerCall of them (bounded per-call GPU cost --
        // a large camera jump may take a few extra calls to fully catch up, exactly like
        // GlobalSDFPass's own backlog draining). Each drained slab's probes are traced (SWRT or HWRT
        // per `traceMode`, matching renderer::SurfaceCacheGIInjectPass::RecordInject's own traceMode
        // convention) and written (irradiance + occlusion) into their wrapped texel address. Caller
        // owns every synchronization barrier before (Surface Cache radiance atlas + TLAS visible)
        // and after (grid writes visible to a later sampled read, e.g. renderer::ScreenTracePass)
        // this call -- same discipline as SurfaceCacheGIInjectPass::RecordInject.
        // `sunDirectionWorld` (Atmos Subtask 5): fed to a probe-ray miss's own Sky-View LUT sample,
        // points FROM the light TOWARD the scene.
        void RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld,
            const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld);

        // F1: level-indexed accessors -- one grid/occlusion view pair + origin per level, replacing
        // the pre-F1 single-level GetGridView()/GetGridOriginWorld(). GetGridSampler() stays a
        // single shared sampler (NEAREST + CLAMP_TO_EDGE, identical filtering need for every level --
        // see the .cpp's sampler comment), bound once per level's descriptor.
        VkImageView GetGridView(uint32_t level) const { return m_Levels[level].gridView; }
        VkImageView GetOcclusionView(uint32_t level) const { return m_Levels[level].occlusionView; }
        VkSampler GetGridSampler() const { return m_GridSampler; }

        // Forces the next RecordUpdate() call to re-enqueue EVERY level's entire grid volume as
        // dirty (exactly like each level's very first call ever, see ClipmapLevel::hasValidWindow's
        // own comment), re-tracing every probe against the CURRENT Surface Cache contents. Exists
        // because the incremental toroidal streaming above only ever re-traces probes newly
        // revealed by camera motion -- a probe traced on frame 1 against a still-cold, partially-
        // captured Surface Cache keeps that stale irradiance indefinitely under a static camera.
        // Driven by the ImGui "Rebuild World Probes" button (main.cpp's Lumen tab, Debug-only).
        void RequestFullRetrace() {
            for (ClipmapLevel& level : m_Levels) {
                level.hasValidWindow = false;
            }
        }

        // How many dirty slabs (across every level -- m_PendingSlabs is one shared queue) are
        // still queued after the most recent RecordUpdate() call (0 = every probe in every level's
        // covered window has been traced) -- feeds the Debug overlay's WORLDPROBES status line so
        // the on-screen text reflects real state instead of a hardcoded label.
        uint32_t GetPendingSlabCount() const { return static_cast<uint32_t>(m_PendingSlabs.size()); }

        // The world-space MINIMUM corner of `level`'s CURRENTLY COVERED WINDOW (i.e.
        // `snappedCenterProbe * GetLevelSpacing(level) - halfExtent`), as of the most recent
        // RecordUpdate() call -- NOT necessarily where texel (0,0,0) physically lives anymore (that
        // texel's CONTENT is whichever world-probe-index currently wraps to it, `index mod
        // kGridResolution`, which shifts as the window moves -- see the class comment). Still
        // exactly what world_probe_sampling.glsl needs (plus GetLevelSpacing(level)/
        // kGridResolution) to convert a world position into that level's own probe-index space
        // before wrapping. Level 0's origin (the finest/innermost level) is also what every non-
        // level-aware caller (e.g. ScreenTracePass's own miss-fallback reconstruction) historically
        // used, so it doubles as the "primary" grid origin for logging/debug purposes.
        const maths::vec3& GetGridOriginWorld(uint32_t level) const { return m_Levels[level].originWorld; }

    private:
        // One axis-aligned "newly revealed" region, in absolute world-probe-index space for its OWN
        // level -- NOT yet split into wrap-contiguous pieces (that split happens when the slab is
        // actually drained, exactly like GlobalSDFPass::DirtySlab's own comment explains).
        struct DirtySlab {
            uint32_t level = 0;
            int32_t probeMin[3] = { 0, 0, 0 };
            int32_t probeMax[3] = { 0, 0, 0 }; // Exclusive.
        };

        // F1: one level's full resource set -- byte-for-byte the same shape GlobalSDFPass::
        // ClipmapLevel uses, plus this pass' own occlusion image/view and per-level descriptor set
        // (GlobalSDFPass's own set 0 has no per-level TLAS/geometry bindings to duplicate; this
        // pass' set 0 does, so each level needs its own full descriptor set, not just its own image).
        struct ClipmapLevel {
            VkImage gridImage = VK_NULL_HANDLE;
            VmaAllocation gridAllocation = VK_NULL_HANDLE;
            VkImageView gridView = VK_NULL_HANDLE;
            VkImage occlusionImage = VK_NULL_HANDLE;
            VmaAllocation occlusionAllocation = VK_NULL_HANDLE;
            VkImageView occlusionView = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE; // set 0: this level's own grid+occlusion+shared-geometry bindings.

            maths::vec3 originWorld{};
            int32_t snappedCenterProbe[3] = { 0, 0, 0 };
            bool hasValidWindow = false; // False until this level's first RecordUpdate() call.
        };

        bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
            const SurfaceCacheRayTracingPass& rtPass);

        // m_Device / m_Allocator are inherited (protected) from RenderPass<WorldProbeGridPass>.

        ClipmapLevel m_Levels[kLevelCount];

        VkSampler m_GridSampler = VK_NULL_HANDLE; // Linear + CLAMP_TO_EDGE -- see .cpp's sampler comment. Shared by every level.

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE; // set 0, shared shape across every level.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE; // Sized for kLevelCount sets.

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        uint32_t m_FrameIndex = 0; // Halton sequence temporal offset, advanced every RecordUpdate() call.

        std::deque<DirtySlab> m_PendingSlabs;

        // Bounded per-RecordUpdate() drain budget. F1: now kLevelCount levels (was a single grid),
        // each still needing at most 3 newly-dirtied slabs per call (one per axis, the maximum a
        // single camera-motion step can shift) under ordinary continuous motion -- so
        // kLevelCount * 3 drains everything in one call in the common case, mirroring
        // GlobalSDFPass::kMaxDirtySlabsPerCall's own "levels x per-level worst case" sizing logic
        // (that class' own value additionally covers multiple dynamic entities' worth of backlog,
        // which this pass has no equivalent of). Only an exceptionally large single-frame camera
        // jump across every level at once would ever leave a residual backlog for a later call to
        // finish.
        static constexpr uint32_t kMaxDirtySlabsPerCall = kLevelCount * 3;

        void EnqueueDirtyRegionsForLevel(uint32_t level, const maths::vec3& cameraPositionWorld);
        void DrainAndRecordSlabs(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld);
        void RecordSlab(VkCommandBuffer cmd, const DirtySlab& slab, const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, const maths::vec3& sunDirectionWorld);
    };

}
