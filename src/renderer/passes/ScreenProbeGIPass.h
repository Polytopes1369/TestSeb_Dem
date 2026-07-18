#pragma once
// Screen Space Probe GI (Lumen-style "Screen Probe Gather"): one probe per kProbeTileSize^2 pixel
// tile, kProbeRayCount Fibonacci-sphere-distributed rays traced per probe through the same SWRT
// (mesh_sdf_trace.glsl) / HWRT (renderer::SurfaceCacheRayTracingPass's TLAS, inline rayQueryEXT)
// back-ends renderer::SurfaceCacheGIInjectPass already uses, sampling renderer::SurfaceCachePass's
// radiance atlas at each hit (surface_cache_sampling.glsl's SampleCardRadiance) -- i.e. this pass
// is a Surface-Cache FINAL GATHER, not a from-scratch light transport solver: the Surface Cache
// atlas is what actually accumulates (possibly multi-bounce, via SurfaceCacheGIInjectPass) direct +
// indirect radiance over time, and every probe ray's single hit-sample reads whatever that cache
// already knows about the point it lands on.
//
// --- Storage: L1 spherical harmonics, not a raw octahedral radiance atlas ---
// Each probe's 64 (direction, radiance) samples are projected into 4 real-SH-basis coefficients per
// color channel (include/sh_probe.glsl) instead of stored as a per-ray octahedral texel grid. This
// makes both the bilateral 4-probe reconstruction (RecordGather) and the exponential temporal blend
// (RecordTemporal) a trivial per-coefficient mix()/weighted-sum, with no resampling between
// mismatched octahedral grids -- at the cost of only reconstructing a first-order (cosine-lobe)
// directional response, which is what a single diffuse-irradiance gather needs anyway.
//
// --- Double-buffered ping-pong, no copy pass ---
// Every probe field (3 SH color channels + world position + octahedral normal) is allocated TWICE
// (m_Slot[0] / m_Slot[1]); m_CurrentSlotIndex flips once per RecordTrace() call. RecordTrace()
// writes this frame's raw new sample into the CURRENT slot; RecordTemporal() reads that raw sample
// back plus a bilinearly-reprojected sample from the OTHER (history) slot, and overwrites the
// CURRENT slot's SH channels in place with the blended result -- which is then exactly what next
// frame finds waiting in "the other slot" once the index flips again. No separate scratch buffer or
// explicit copy step is needed. Every probe image carries both STORAGE_BIT (RecordTrace's/
// RecordTemporal's imageLoad/imageStore) and SAMPLED_BIT (RecordTemporal's/RecordGather's linear-
// filtered history/bilateral reads via a shared sampler) usage, permanently in VK_IMAGE_LAYOUT_GENERAL
// (mirrors every other Lumen-style atlas in this codebase, e.g. renderer::SurfaceCachePass's own
// atlas images).
//
// --- Per-frame call order a caller must follow ---
//   1. RecordUpdateViewParams(cmd, viewProj, prevViewProj) -- uploads the shared UBO every other
//      call below reads; `prevViewProj` should be an identity matrix on the very first frame ever
//      recorded (see renderer::ClusterRenderPipeline's own m_HasPrevViewProj guard).
//   2. RecordTrace(cmd, traceContext, entityCount, traceMode, frameIndex) -- needs
//      renderer::ClusterResolvePass's GBuffer (normal/depth) already visible to COMPUTE_SHADER
//      reads, and renderer::SurfaceCachePass's radiance atlas + renderer::SurfaceCacheRayTracingPass's
//      TLAS already visible (both already true after renderer::SurfaceCacheGIInjectPass's own
//      trailing barrier, if that pass ran first this frame).
//   3. RecordTemporal(cmd) -- needs RecordTrace's own trailing barrier (this class inserts one).
//   4. RecordGather(cmd, debugViewMode) -- needs RecordTemporal's own trailing barrier (this class
//      inserts one), and renderer::ClusterResolvePass's output color image already in GENERAL
//      (true for that image's entire lifetime). Read-modify-writes that color image directly; the
//      caller's own barrier after this call (before any further read of the color image, e.g. the
//      swapchain blit) must cover this pass' own COMPUTE_SHADER / SHADER_STORAGE_WRITE access.
//
// --- Phase 6 (UE5.8 parity roadmap): two-level hierarchical adaptive placement ---
// Pre-Phase-6, RecordTrace()/RecordTemporal() unconditionally traced/updated EVERY probe of the
// fixed kProbeTileSize grid, every frame -- expensive regardless of whether a tile's underlying
// geometry was flat/static or full of detail. This class now additionally maintains an always-on
// COARSE grid (m_CoarseProbeTileSize, exactly 2x kProbeTileSize -- a quarter as many probes),
// which is fully traced/updated every frame (cheap at that density), while the original FINE grid
// is now only selectively traced/updated for tiles a new classify pass (ScreenProbeClassify.comp)
// flags as containing a depth/normal discontinuity -- flat/uniform regions fall back to the coarse
// grid's own bilateral tap instead of paying for a dense fine trace they don't need (see
// ScreenProbeGather.comp's now-multi-tier fallback chain). A tile not selected this frame keeps
// whatever fine-grid data it last had (a `vkCmdCopyImage` "report" at the top of every
// RecordTrace() call copies the history slot forward into the new current slot BEFORE the
// selective fine trace overwrites only the active tiles -- see RecordTrace()'s own comment for why
// this copy is necessary for correctness, not just an optimization). None of this changes the
// public call contract above: RecordTrace()/RecordTemporal()/RecordGather() keep their exact
// existing signatures -- every new buffer/pipeline/dispatch this requires is encapsulated entirely
// inside this class.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "core/EngineConfig.h"

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/RenderPass.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;
    class ClusterResolvePass;

    // Migrated to RenderPass<Derived> (see renderer/vulkan/RenderPass.h): Init()/Shutdown() are
    // inherited. Note InitImpl() still calls the inherited Shutdown() as its own first statement
    // (self-reinit, same pattern/reason as ShadowMapPass's own migration comment). 5 separate
    // descriptor pools (Trace/Temporal maxSets=4 each fine+coarse ping-pong; Gather maxSets=2;
    // BuildArgs maxSets=2; Classify maxSets=1) -- only Classify fits the single-set helper, the
    // other 4 stay raw Vulkan + RegisterResource(), same reasoning as ReflectionPass.
    class ScreenProbeGIPass : public RenderPass<ScreenProbeGIPass> {
        friend class RenderPass<ScreenProbeGIPass>; // Lets Init() call our private InitImpl().

    public:
        ScreenProbeGIPass() = default;

        ScreenProbeGIPass(const ScreenProbeGIPass&) = delete;
        ScreenProbeGIPass& operator=(const ScreenProbeGIPass&) = delete;

        static inline uint32_t kProbeTileSize = 8u;   // Screen pixels per FINE probe tile, both axes.
        // Phase 6: the COARSE grid's own tile size -- always exactly 2x kProbeTileSize (a quarter
        // as many probes per axis-pair) -- see the class comment's own hierarchy description. Set
        // alongside kProbeTileSize at Init() time (both runtime-configurable via config::lumen),
        // not a compile-time constant.
        static inline uint32_t kCoarseProbeTileSize = 16u;
        static inline uint32_t kProbeRayCount = 64u;  // Fibonacci-sphere rays traced per probe per frame.
        static constexpr uint32_t kGridWorkgroupSize = 8; // RecordTemporal's/RecordGather's local_size_x/y (also the COARSE grid's own Trace/Temporal 2D dispatch tiling).
        static inline float kTemporalAlpha = 0.05f;  // Exponential moving-average blend factor.

        static constexpr VkFormat kSHFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        static constexpr VkFormat kWorldPosFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // rgb = world pos, a = validity.
        static constexpr VkFormat kNormalFormat = VK_FORMAT_R16G16_SFLOAT;         // Octahedral-encoded.

        // `traceContext`/`surfaceCache`/`rtPass`/`resolvePass` must all already be Init'd and must
        // outlive this pass -- their resources are bound into this pass' own descriptor sets
        // unmodified (mirrors renderer::SurfaceCacheGIInjectPass's own borrowing convention).
        // Init(VkDevice, VmaAllocator, VkCommandPool, VkQueue, VkExtent2D,
        // const SurfaceCacheTraceContext&, const SurfaceCachePass&, const SurfaceCacheRayTracingPass&,
        // const ClusterResolvePass&) -> bool and Shutdown() are inherited from
        // RenderPass<ScreenProbeGIPass>; see InitImpl() below.

        // Uploads this frame's ScreenProbeViewParamsUBO (invViewProj, prevViewProj, viewport/probe
        // grid sizes) -- `viewProj` is this frame's combined matrix (its inverse is computed here,
        // CPU-side, via maths::mat4::Inverse()); `prevViewProj` is the previous frame's own combined
        // matrix (renderer::ClusterRenderPipeline's own m_PrevViewProj).
        void RecordUpdateViewParams(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj);

        // Flips m_CurrentSlotIndex FIRST (so this frame writes into whichever slot held the
        // PREVIOUS frame's history, and that previous frame's now-stale "current" slot becomes
        // this frame's reprojection source), then traces kProbeRayCount rays per probe (one
        // workgroup per probe, local_size_x = kProbeRayCount) via SWRT (traceMode = 0) or HWRT
        // (traceMode = 1) and projects the results into the new current slot's SH images.
        // RecordTemporal()/RecordGather() called right after this see that same slot as "current".
        //
        // Phase 6: internally, this now ALSO (a) copy-forwards the fine grid's history slot into
        // the new current slot (see class comment), (b) always fully traces the COARSE grid
        // (direct dispatch), and (c) runs ScreenProbeClassify.comp to build this frame's compacted
        // active-fine-tile list, then traces ONLY those fine tiles (indirect dispatch, via
        // BuildDispatchIndirectArgs.comp) -- none of this changes the signature or call contract
        // above.
        void RecordTrace(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
            uint32_t entityCount, uint32_t traceMode, uint32_t frameIndex);

        // Reprojects + exponentially blends the current slot's raw new sample against the history
        // slot, in place. Phase 6: internally also updates the COARSE grid (direct dispatch) and
        // the active fine tiles only (indirect dispatch) -- same no-signature-change note as
        // RecordTrace() above.
        void RecordTemporal(VkCommandBuffer cmd);

        // Full-resolution bilateral 4-probe gather, read-modify-writing `resolvePass`'s output color
        // image (the same instance passed to Init()). `debugViewMode` is forwarded to
        // ScreenProbeGather.comp's own DEBUG_VIEW_SPATIAL_PROBES (14) visualization branch; ignored
        // (and not even part of the pipeline layout) in a Release build.
        void RecordGather(VkCommandBuffer cmd
#ifndef NDEBUG
            , uint32_t debugViewMode
#endif
        );

    private:
        // One ping-pong slot's 5 probe-field images, all sized (m_ProbeCountX, m_ProbeCountY).
        struct ProbeSlot {
            VkImage shRImage = VK_NULL_HANDLE; VmaAllocation shRAllocation = VK_NULL_HANDLE; VkImageView shRView = VK_NULL_HANDLE;
            VkImage shGImage = VK_NULL_HANDLE; VmaAllocation shGAllocation = VK_NULL_HANDLE; VkImageView shGView = VK_NULL_HANDLE;
            VkImage shBImage = VK_NULL_HANDLE; VmaAllocation shBAllocation = VK_NULL_HANDLE; VkImageView shBView = VK_NULL_HANDLE;
            VkImage worldPosImage = VK_NULL_HANDLE; VmaAllocation worldPosAllocation = VK_NULL_HANDLE; VkImageView worldPosView = VK_NULL_HANDLE;
            VkImage normalImage = VK_NULL_HANDLE; VmaAllocation normalAllocation = VK_NULL_HANDLE; VkImageView normalView = VK_NULL_HANDLE;
        };

        bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, const SurfaceCacheTraceContext& traceContext,
            const SurfaceCachePass& surfaceCache, const SurfaceCacheRayTracingPass& rtPass,
            const ClusterResolvePass& resolvePass);

        // m_Device / m_Allocator are inherited (protected) from RenderPass<ScreenProbeGIPass>.

        uint32_t m_ProbeCountX = 0;
        uint32_t m_ProbeCountY = 0;
        VkExtent2D m_RenderExtent{ 0, 0 };

        ProbeSlot m_Slots[2];
        uint32_t m_CurrentSlotIndex = 0;

        VkSampler m_ProbeSampler = VK_NULL_HANDLE; // Linear, clamp-to-edge -- history reprojection + bilateral gather taps.

        // Shared view-params UBO (ScreenProbeViewParamsUBO, std140) -- bound identically into every
        // pass' own set 0.
        GpuBuffer m_ViewParamsBuffer;

        // Trace: 2 variants (indexed by m_CurrentSlotIndex at record time) since which physical slot
        // is "current" (write target) flips every frame.
        VkDescriptorSetLayout m_TraceSetLayout = VK_NULL_HANDLE; // set 0.
        VkDescriptorPool m_TraceDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_TraceSet[2]{};
        VkPipelineLayout m_TracePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_TracePipeline = VK_NULL_HANDLE;

        // Temporal: 2 variants, same reason.
        VkDescriptorSetLayout m_TemporalSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_TemporalDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_TemporalSet[2]{};
        VkPipelineLayout m_TemporalPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_TemporalPipeline = VK_NULL_HANDLE;

        // Gather: 2 variants, same reason -- now reads BOTH the fine (bindings 0-4) and coarse
        // (bindings 5-9) current-slot images, so no separate "coarse gather set" is needed (see
        // the class comment's own fallback-chain description).
        VkDescriptorSetLayout m_GatherSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_GatherDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_GatherSet[2]{};
        VkPipelineLayout m_GatherPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_GatherPipeline = VK_NULL_HANDLE;

        // Phase 6: the always-on COARSE grid -- same ProbeSlot shape, own ping-pong pair, sized
        // (m_ProbeCoarseCountX, m_ProbeCoarseCountY) -- see the class comment's hierarchy
        // description. m_CurrentSlotIndex is shared with the fine grid (both flip in lockstep every
        // RecordTrace() call).
        uint32_t m_ProbeCoarseCountX = 0;
        uint32_t m_ProbeCoarseCountY = 0;
        ProbeSlot m_CoarseSlots[2];

        // Trace/Temporal: the COARSE grid's own ping-pong descriptor-set pair, reusing
        // m_TraceSetLayout/m_TemporalSetLayout unmodified (identical bindings, only the bound
        // images differ) but allocated as a SEPARATE pair of sets from the correspondingly
        // enlarged pool (pointing at m_CoarseSlots instead of m_Slots).
        VkDescriptorSet m_CoarseTraceSet[2]{};
        VkDescriptorSet m_CoarseTemporalSet[2]{};

        // Phase 6: per-tile depth/normal-discontinuity classification -- one dispatch per COARSE
        // tile, direct (never indirect: this always covers the whole coarse grid). Writes into
        // m_ActiveFineTileListBuffer below. Set 0 only (GBuffer normal/depth readonly + view-params
        // UBO + the active-list SSBO) -- no mesh-SDF/surface-cache trace sets needed, this pass
        // never traces anything.
        VkDescriptorSetLayout m_ClassifySetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_ClassifyDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_ClassifySet = VK_NULL_HANDLE;
        VkPipelineLayout m_ClassifyPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ClassifyPipeline = VK_NULL_HANDLE;

        // Phase 6: `{uint count; uint coords[fineProbeCountX*fineProbeCountY];}` -- the compacted
        // list of fine tiles ScreenProbeClassify.comp flagged this frame, sized for the worst case
        // (every fine tile active at once) so it never needs runtime resizing. Reset (`count` word
        // only) via vkCmdFillBuffer at the top of every RecordTrace() call.
        GpuBuffer m_ActiveFineTileListBuffer;

        // Phase 6: renderer::BuildDispatchIndirectArgs.comp instance, reused twice per RecordTrace()/
        // RecordTemporal() pair (once for the fine Trace dispatch, once for the fine Temporal
        // dispatch -- different workgroupSize/perElementMultiplier each time, see RecordTrace()'s
        // own body) -- mirrors renderer::ClusterOcclusionCullingPass's own reuse of this exact
        // shader for its own late-pass dispatch.
        VkDescriptorSetLayout m_BuildArgsSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_BuildArgsDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_BuildArgsSet[2]{}; // 0 = Trace's own dispatch-args buffer, 1 = Temporal's.
        VkPipelineLayout m_BuildArgsPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_BuildArgsPipeline = VK_NULL_HANDLE;
        GpuBuffer m_FineTraceDispatchArgsBuffer;    // VkDispatchIndirectCommand (12 bytes).
        GpuBuffer m_FineTemporalDispatchArgsBuffer; // VkDispatchIndirectCommand (12 bytes).
    };

}
