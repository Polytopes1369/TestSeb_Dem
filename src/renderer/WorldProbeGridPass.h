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
// --- Current status (2026-07-16 UE5.8-parity audit) ---
// SampleWorldProbeGrid() has no live caller today -- it is referenced only by the dead
// ScreenTracePass/GICompositePass (both exist as compilable files but neither is instantiated by
// renderer::ClusterRenderPipeline). This pass's own RecordUpdate() runs correctly and writes real
// data into GetGridView() every frame it's called, but nothing downstream reads that view yet.
// renderer::ClusterRenderPipeline::SetDebugWorldProbesEnabled() gates the RecordUpdate() call
// itself for exactly this reason (Release skips it rather than paying for an unconsumed result) --
// see that method's own comment for the plan to re-enable it once a real consumer is wired in.
//
// --- Why a full rebuild every frame, not incremental/toroidal streaming ---
// renderer::GlobalSDFPass's clipmap levels stream incrementally (only newly-revealed slabs are
// recomposited each frame) because a level can be kClipmapResolution=32 voxels of a MUCH larger,
// multi-level clipmap hierarchy, expensive enough per voxel (compositing every overlapping
// entity's Mesh SDF) that a full rebuild every frame would be wasteful. This grid is a single
// level, kGridResolution^3 = 32768 probes, each one cheap (kProbeSampleDirections fixed-direction
// traces, not per-entity SDF compositing) -- a full rebuild every frame is both affordable at this
// scale and literally what was asked ("Propage l'éclairage du Surface Cache directement dans
// cette grille 3D à chaque frame"), so this class does not reproduce GlobalSDFPass's incremental-
// streaming machinery at all.
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
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "core/EngineConfig.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;

    class WorldProbeGridPass {
    public:
        WorldProbeGridPass() = default;

        WorldProbeGridPass(const WorldProbeGridPass&) = delete;
        WorldProbeGridPass& operator=(const WorldProbeGridPass&) = delete;

        // Cubic probe count per axis -- see the class comment's "why a full rebuild" note for why
        // this stays small enough that a full rebuild every frame is affordable.
        static constexpr uint32_t kGridResolution = config::lumen::PROBE_GRID_RESOLUTION;
        // World-space distance between adjacent probes -- together with kGridResolution, this
        // grid covers a (kGridResolution * kProbeSpacing)^3 = 64^3 world-unit cube centered on the
        // camera, recentered every RecordUpdate() call (see that method's own comment).
        static constexpr float kProbeSpacing = config::lumen::PROBE_SPACING;
        // Matches WorldProbeInject.comp's local_size_x/y/z exactly.
        static constexpr uint32_t kWorkgroupSize = 4;
        // Fixed omnidirectional sample set per probe -- see the class comment's own "why no
        // hemisphere sampling" note. Matches WorldProbeInject.comp's kProbeSampleDirections array
        // length exactly.
        static constexpr uint32_t kProbeSampleDirections = config::lumen::PROBE_SAMPLE_DIRECTIONS;

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

        // Recenters the grid (snapped to kProbeSpacing-sized steps, so it does not re-derive a
        // different origin -- and thus lose temporal coherence for a probe that didn't actually
        // move -- on every single frame of continuous camera motion) around `cameraPositionWorld`,
        // then dispatches ONE full-grid compute pass (kGridResolution/kWorkgroupSize groups per
        // axis) that traces every probe's kProbeSampleDirections rays (SWRT or HWRT per `traceMode`,
        // matching renderer::SurfaceCacheGIInjectPass::RecordInject's own traceMode convention) and
        // writes the averaged hit radiance into that probe's texel. Caller owns every
        // synchronization barrier before (Surface Cache radiance atlas + TLAS visible) and after
        // (grid writes visible to a later sampled read, e.g. renderer::ScreenTracePass) this call --
        // same discipline as SurfaceCacheGIInjectPass::RecordInject.
        void RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld,
            const SurfaceCacheTraceContext& traceContext, uint32_t traceMode);

        VkImageView GetGridView() const { return m_GridView; }
        VkSampler GetGridSampler() const { return m_GridSampler; }
        // World-space position of the grid's own texel (0,0,0) corner (NOT its center) as of the
        // MOST RECENT RecordUpdate() call -- world_probe_sampling.glsl's SampleWorldProbeGrid needs
        // this (plus kProbeSpacing/kGridResolution) to convert a world position into the grid's
        // own normalized [0,1]^3 sampling coordinate.
        const maths::vec3& GetGridOriginWorld() const { return m_GridOriginWorld; }

    private:
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
    };

}
