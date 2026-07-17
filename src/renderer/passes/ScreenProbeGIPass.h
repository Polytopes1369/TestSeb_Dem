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

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "core/EngineConfig.h"

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;
    class ClusterResolvePass;

    class ScreenProbeGIPass {
    public:
        ScreenProbeGIPass() = default;

        ScreenProbeGIPass(const ScreenProbeGIPass&) = delete;
        ScreenProbeGIPass& operator=(const ScreenProbeGIPass&) = delete;

        static inline uint32_t kProbeTileSize = 8u;   // Screen pixels per probe tile, both axes.
        static inline uint32_t kProbeRayCount = 64u;  // Fibonacci-sphere rays traced per probe per frame.
        static constexpr uint32_t kGridWorkgroupSize = 8; // RecordTemporal's/RecordGather's local_size_x/y.
        static inline float kTemporalAlpha = 0.05f;  // Exponential moving-average blend factor.

        static constexpr VkFormat kSHFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        static constexpr VkFormat kWorldPosFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // rgb = world pos, a = validity.
        static constexpr VkFormat kNormalFormat = VK_FORMAT_R16G16_SFLOAT;         // Octahedral-encoded.

        // `traceContext`/`surfaceCache`/`rtPass`/`resolvePass` must all already be Init'd and must
        // outlive this pass -- their resources are bound into this pass' own descriptor sets
        // unmodified (mirrors renderer::SurfaceCacheGIInjectPass's own borrowing convention).
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, const SurfaceCacheTraceContext& traceContext,
            const SurfaceCachePass& surfaceCache, const SurfaceCacheRayTracingPass& rtPass,
            const ClusterResolvePass& resolvePass);

        void Shutdown();

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
        void RecordTrace(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
            uint32_t entityCount, uint32_t traceMode, uint32_t frameIndex);

        // Reprojects + exponentially blends the current slot's raw new sample against the history
        // slot, in place.
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

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

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

        // Gather: 2 variants, same reason.
        VkDescriptorSetLayout m_GatherSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_GatherDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_GatherSet[2]{};
        VkPipelineLayout m_GatherPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_GatherPipeline = VK_NULL_HANDLE;
    };

}
