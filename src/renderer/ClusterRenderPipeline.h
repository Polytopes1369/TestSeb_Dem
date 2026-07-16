#pragma once
// Final integration orchestrator for the whole Nanite-style virtual geometry pipeline: connects
// the previously self-contained building blocks into one working chain --
//
//   procedural generation (VulkanContext::GenerateGeometry, already run before Init())
//     -> consolidated 4 KB-paged .cache file (written at startup by
//        geometry::RunVirtualGeometryCacheTest via geometry::CacheFileManager)
//     -> streaming: cache file -> host staging -> renderer::GpuGeometryPagePool physical pages
//        (BindPage) -> renderer::GeometryDecompressionPass (vertex decode + index expansion)
//     -> two-phase GPU occlusion culling (renderer::ClusterOcclusionCullingPass + renderer::HZBPass)
//        with hardware/software rasterization routing per cluster
//     -> hybrid rasterization into the Visibility Buffer (renderer::ClusterHardwareRasterPass via
//        vkCmdDrawIndexedIndirectCount + renderer::ClusterSoftwareRasterPass via 64-bit atomics)
//     -> deferred material resolve (renderer::ClusterResolvePass)
//     -> blit to the acquired swapchain image, ready for present.
//
// --- Streaming scope in this integration ---
// The current scene's entire cluster set fits comfortably in the physical page pool, so Init()
// streams every cluster page in once, at startup, through the real streaming path (cache file ->
// staging -> BindPage -> DecompressPage) and everything stays resident -- there is no per-frame
// disk traffic, which is also the strongest possible anti-stutter guarantee for a demoscene-sized
// scene. The per-frame async residency loop (FeedbackBuffer misses -> StreamingRequestQueue ->
// AsyncFileStreamer -> BindPageEvictingIfFull) stays future work: it additionally requires
// residency-aware culling (a cluster whose page is evicted must never reach a raster worklist),
// which no culling shader implements yet.
//
// --- LOD scope ---
// The candidate cluster set uploaded to the culling pass is the DAG's LEAF level (full detail,
// geometry::DAGNodeEntry::level == 0) -- exactly the geometry the flat pre-cluster path drew. The
// GPU LOD cut (ClusterDAGScreenError.comp) exists but has no driver yet; when it lands, it slots
// in as a pass that rewrites this candidate set per frame, upstream of the occlusion cull, with no
// change to anything downstream. Entity self-rotation (EntityTransform) is likewise not applied by
// the clustered path yet -- clusters render the static geometry as captured into the cache.
//
// --- Per-frame GPU work (all recorded by RecordFrame() into ONE command buffer, submitted once:
// no mid-frame vkQueueSubmit/vkQueueWaitIdle anywhere, the other half of the anti-stutter
// guarantee) ---
//   1. Clear the frame's atomic worklists (pending/early/late/software counts) + the software
//      rasterizer's atomic VisBuffer.
//   2. EARLY cull: every leaf cluster vs frustum/backface + last frame's HZB; survivors routed to
//      the early hardware draw list or the software cluster list; uncertain ones to the pending list.
//   3. Early hardware raster into the VisBuffer (2x R32_UINT) + depth.
//   4. HZB rebuild from the early depth; LATE cull re-tests exactly the pending list against it.
//   5. Late hardware raster (loadOp = LOAD on top of the early pass's output).
//   6. Software raster of every micro-triangle cluster listed this frame (early + late routed).
//   7. Resolve: per-pixel hardware-vs-software depth arbitration, barycentric reconstruction,
//      material evaluation into the output color image.
//   8. Second HZB rebuild from the now-complete depth, so next frame's EARLY pass tests against
//      the full previous-frame occlusion state (the standard two-phase scheme).
//   9. Blit the resolved image to the swapchain image and transition it to PRESENT_SRC_KHR.
// Every inter-pass hazard is covered by an explicit VkMemoryBarrier2/VkImageMemoryBarrier2 inside
// RecordFrame() or inside the passes' own Record*() functions -- see the .cpp for the exact
// stage/access/layout reasoning at each step.

#include <cstdint>
#include <filesystem>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/Camera.h"
#include "core/maths/Maths.h"
#include "renderer/ClusterHardwareRasterPass.h"
#include "renderer/ClusterLODSelectionPass.h"
#include "renderer/ClusterOcclusionCullingPass.h"
#include "renderer/ClusterResolvePass.h"
#include "renderer/ClusterSoftwareRasterPass.h"
#include "renderer/GeometryDecompressionPass.h"
#include "renderer/GeometryStreamingCoordinator.h"
#include "renderer/GlobalSDFPass.h"
#include "renderer/GpuBuffer.h"
#include "renderer/GpuGeometryPagePool.h"
#include "renderer/HZBPass.h"
#include "renderer/LightingTypes.h"
#include "renderer/ProceduralMaskGenerator.h"
#include "renderer/ScreenProbeGIPass.h"
#include "renderer/ShadowMapPass.h"
#include "renderer/SurfaceCacheGIInjectPass.h"
#include "renderer/SurfaceCachePass.h"
#include "renderer/SurfaceCacheRayTracingPass.h"
#include "renderer/SurfaceCacheTraceContext.h"
#ifndef NDEBUG
#include "renderer/debug/ClusterTriangleStatsPass.h"
#include "renderer/debug/DebugTextOverlay.h"
#include "renderer/SDFRayMarchPass.h"
#endif

namespace renderer {

    // Everything Init() needs from the outside world, in one struct -- all handles are borrowed
    // (owned by VulkanContext), only the passes/buffers this class creates itself are owned.
    struct ClusterRenderPipelineCreateInfo {
        // Needed only by renderer::SurfaceCacheRayTracingPass::Init (VkPhysicalDeviceRayTracingPipelinePropertiesKHR
        // query -- shaderGroupBaseAlignment/shaderGroupHandleSize for the Shader Binding Table).
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkQueue queue = VK_NULL_HANDLE;
        VkExtent2D renderExtent{ 0, 0 };

        // Hardware-path Visibility Buffer attachments + depth (VulkanContext's images).
        VkImage visBufferClusterIDImage = VK_NULL_HANDLE;
        VkImageView visBufferClusterIDView = VK_NULL_HANDLE;
        VkImage visBufferTriangleIDImage = VK_NULL_HANDLE;
        VkImageView visBufferTriangleIDView = VK_NULL_HANDLE;
        VkFormat visBufferFormat = VK_FORMAT_UNDEFINED;
        VkImage depthImage = VK_NULL_HANDLE;
        VkImageView depthImageView = VK_NULL_HANDLE;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;

        // The consolidated scene cache written at startup (geometry::RunVirtualGeometryCacheTest).
        std::filesystem::path cacheFilePath = "scene.cache";
    };

    class ClusterRenderPipeline {
    public:
        ClusterRenderPipeline() = default;

        ClusterRenderPipeline(const ClusterRenderPipeline&) = delete;
        ClusterRenderPipeline& operator=(const ClusterRenderPipeline&) = delete;

        // Average estimated triangle screen size (pixels) below which a cluster is routed to the
        // software rasterizer -- see ClusterHZBOcclusionCull.comp's ShouldUseSoftwareRaster().
        static constexpr float kSoftwareRasterThresholdPixels = 6.0f;

        // Target on-screen geometric error (pixels) for the LOD DAG cut -- see
        // ClusterDAGScreenError.comp's pixelThreshold. A node draws once its own projected error
        // drops below this value while its parent's still exceeds it; typical Nanite-style engines
        // target roughly 1 pixel.
        static constexpr float kLODPixelErrorThreshold = 1.0f;

        // Reads the .cache file's header/tables, streams every cluster's 4 KB geometry page into
        // the physical pool (one staging buffer, one setup command buffer, one blocking submit --
        // startup-only, never repeated per frame), decompresses vertices and expands indices for
        // every page, builds the leaf-cluster culling metadata, and initializes every render pass.
        // Returns false (with the failure logged) if the cache file cannot be read -- the clustered
        // pipeline cannot run without it.
        bool Init(const ClusterRenderPipelineCreateInfo& createInfo);

        void Shutdown();

        // Records the entire frame's GPU work (culling -> hybrid raster -> resolve -> blit +
        // present transition of `swapchainImage`) into `cmd` -- see the class comment's step list.
        // `camera` supplies view/proj (must be the frame's final matrices: every stage this frame
        // uses the same viewProj, which is what makes the resolve pass's barycentric
        // reconstruction exact) and `cameraPositionWorld` the eye point for the backface cone
        // test. `globalTimeSeconds` (main.cpp's glfwGetTime(), monotonically increasing) drives the
        // World Position Offset sway function (wpo_deformation.glsl's ApplyWPODeformation, called
        // identically from ClusterRaster.vert and ClusterSoftwareRaster.comp) -- uploaded once per
        // frame into m_WPOGlobalsBuffer before either raster pass runs. The caller only
        // begins/ends/submits the command buffer and presents.
        void RecordFrame(VkCommandBuffer cmd, const CameraPushConstants& camera,
            const maths::vec3& cameraPositionWorld, const CameraFrameInfo& cameraFrameInfo,
            float globalTimeSeconds, VkImage swapchainImage);

        // Upper bound on this frame's actual candidate count (the DAG's total leaf count) -- NOT
        // this frame's real candidate count, which only ever exists on the GPU now that
        // m_LODSelection computes a dynamic per-frame cut (see ClusterLODSelectionPass).
        uint32_t GetClusterCount() const { return m_ClusterCount; }

#ifndef NDEBUG
        // SWRT/HWRT back-end toggle shared by m_GIInject and m_ScreenProbeGI (0 = SWRT mesh-SDF
        // sphere tracing, 1 = HWRT inline rayQueryEXT against m_SurfaceCacheRT's TLAS) -- debug-only
        // (main.cpp numpad/T key) so both back-ends stay exercised; Release always uses HWRT (see
        // RecordFrame()'s own use of this member).
        void SetDebugTraceMode(uint32_t traceMode) { m_DebugTraceMode = traceMode; }
#endif

    private:
        // Records the vkCmdBeginRendering block shared by the early and late hardware raster
        // passes: `clearAttachments` selects between the early pass's CLEAR (VisBuffer sentinel
        // 0xFFFFFFFF, depth 1.0) and the late pass's LOAD (draw on top of the early output).
        void BeginVisBufferRendering(VkCommandBuffer cmd, bool clearAttachments) const;

        VkDevice m_Device = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        // Borrowed attachment handles (owned by VulkanContext).
        VkImage m_VisBufferClusterIDImage = VK_NULL_HANDLE;
        VkImageView m_VisBufferClusterIDView = VK_NULL_HANDLE;
        VkImage m_VisBufferTriangleIDImage = VK_NULL_HANDLE;
        VkImageView m_VisBufferTriangleIDView = VK_NULL_HANDLE;
        VkImage m_DepthImage = VK_NULL_HANDLE;
        VkImageView m_DepthImageView = VK_NULL_HANDLE;

        // Upper bound on this frame's actual candidate count (the DAG's total leaf count) -- see
        // GetClusterCount()'s own comment.
        uint32_t m_ClusterCount = 0;

        // WPOGlobalsUBO (16 bytes, std140: float globalTime + 3 pad floats) -- uploaded once per
        // frame (vkCmdUpdateBuffer) in RecordFrame() before either raster pass runs; bound into
        // both ClusterHardwareRasterPass and ClusterSoftwareRasterPass at Init() time. See
        // wpo_deformation.glsl's ApplyWPODeformation for how the raster shaders consume it.
        GpuBuffer m_WPOGlobalsBuffer;

        // Generates the bindless procedural cutout mask array (mask_sampling.glsl) once at Init()
        // time, before any raster/resolve pass is initialized -- see ProceduralMaskGenerator's own
        // class comment. GetMaskImageInfos() is threaded into all three passes below.
        ProceduralMaskGenerator m_MaskGenerator;

        // Owned pipeline stages, in rough execution order.
        GpuGeometryPagePool m_PagePool;
        GeometryDecompressionPass m_Decompression;
        HZBPass m_HZB;
        // Per-frame GPU-driven LOD DAG cut (ClusterDAGScreenError.comp -> ClusterLODResidencyFallback
        // .comp -> ClusterLODCompact.comp), replacing the previous static "always DAG level 0"
        // candidate upload -- see ClusterLODSelectionPass's own class comment. Its output
        // (GetCandidateMetadataBuffer()/GetCandidateCountBuffer()) feeds m_OcclusionCulling below.
        ClusterLODSelectionPass m_LODSelection;
        // Drives the previously-dormant async streaming stack (AsyncFileStreamer/
        // StreamingRequestQueue/FeedbackBuffer) for real, once per frame -- see
        // GeometryStreamingCoordinator's own class comment for the exact sequencing contract with
        // m_LODSelection's feedback buffer.
        GeometryStreamingCoordinator m_Streaming;
        ClusterOcclusionCullingPass m_OcclusionCulling;
        ClusterHardwareRasterPass m_HardwareRaster;
        ClusterSoftwareRasterPass m_SoftwareRaster;
        ClusterResolvePass m_Resolve;

        // Lumen-style GI infrastructure -- unlike the debug-only stats/overlay block below, these
        // are real (if not yet light-transport-consuming) systems, not visualization tools, so
        // they compile in Release too (matching how the actual Nanite cluster pipeline above is
        // likewise unconditional; only the NUMPAD-DRIVEN DEBUG VIEW SWITCHING is Debug-only, per
        // CLAUDE.md's build-separation rule). Recorded every frame in RecordFrame() regardless of
        // camera.debugViewMode.
        ShadowMapPass m_ShadowMap;
        SurfaceCachePass m_SurfaceCache;
        GlobalSDFPass m_GlobalSDF;
        // Fixed sun-only lighting for now (default-constructed SceneLights: no point lights) --
        // see renderer::LightingTypes.h's own comment; a future scene-authoring system would
        // populate this instead of leaving it default.
        SceneLights m_SceneLights;

        // Shared trace-scene descriptor sets (mesh SDF trace + Surface Cache sampling, see
        // SurfaceCacheTraceContext's own class comment) built once from m_GlobalSDF + m_SurfaceCache,
        // reused unmodified by m_SurfaceCacheRT, m_GIInject, and m_ScreenProbeGI's own trace pass.
        SurfaceCacheTraceContext m_TraceContext;
        // HWRT back-end: one BLAS per traced entity + one TLAS, built once (static scene) against
        // m_SurfaceCache's own Fallback Mesh vertex/index buffers.
        SurfaceCacheRayTracingPass m_SurfaceCacheRT;
        // Per-card hemisphere-sampled secondary-bounce injection into m_SurfaceCache's own radiance
        // atlas -- makes that atlas genuinely multi-bounce over time, which is what
        // m_ScreenProbeGI's single-hit-sample final gather then benefits from.
        SurfaceCacheGIInjectPass m_GIInject;

        // Screen Space Probe GI (Lumen "Screen Probe Gather"): traces Fibonacci-sphere rays per
        // 8x8-pixel probe (sampling m_SurfaceCache's radiance atlas at each hit via m_TraceContext/
        // m_SurfaceCacheRT), temporally accumulates, and gathers the result back into m_Resolve's
        // output color image -- see ScreenProbeGIPass's own class comment.
        ScreenProbeGIPass m_ScreenProbeGI;

        // Previous frame's combined view-projection matrix -- ScreenProbeGIPass's temporal pass
        // (and ClusterResolve.comp's DEBUG_VIEW_MOTION_VECTORS) reprojects a probe's/pixel's
        // (static, see this class' own "Entity self-rotation" scope note) world position with this
        // matrix to find its screen position last frame. Updated at the very end of RecordFrame(),
        // after every consumer of "the previous frame's" value has run.
        maths::mat4 m_PrevViewProj{};
        bool m_HasPrevViewProj = false;

        // Advances once per RecordFrame() call -- ScreenProbeTrace.comp's own per-frame Fibonacci-
        // sphere jitter rotation (include/sh_probe.glsl's JitterDirection).
        uint32_t m_FrameIndex = 0;

#ifndef NDEBUG
        uint32_t m_DebugTraceMode = 0;
#endif

#ifndef NDEBUG
        // Real-time stat overlay (GPU memory used, pending SSD page loads, disk read throughput,
        // HW-vs-software triangle split) -- see debug::DebugTextOverlay's own class comment. Whole
        // block compiled out in Release (CLAUDE.md's debug/Release separation rule): zero code,
        // zero strings, zero extra buffers survive into the production executable.
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Borrowed, stored only for RecordFrame()'s vmaGetHeapBudgets() query.
        debug::ClusterTriangleStatsPass m_TriangleStats;
        debug::DebugTextOverlay m_DebugOverlay;
        // Bytes/sec is a delta over wall-clock time, not an instantaneous GPU counter -- sampled
        // once per frame against `globalTimeSeconds` (RecordFrame()'s own parameter, glfwGetTime()-
        // derived) and GeometryStreamingCoordinator::GetTotalBytesCompleted()'s running total.
        float m_LastStatsSampleTime = 0.0f;
        uint64_t m_LastStatsSampleBytes = 0;

        // Two-tier SDF ray march DEBUG VISUALIZATION (see renderer::SDFRayMarchPass's own class
        // comment: its whole output is "a full-screen debug-visualization image", never consumed
        // by production lighting) -- Debug-only per CLAUDE.md's build-separation rule, exactly
        // like m_TriangleStats/m_DebugOverlay above. Only recorded (and only ever blitted to the
        // swapchain in place of m_Resolve's output) when camera.debugViewMode == DEBUG_VIEW_LUMEN.
        SDFRayMarchPass m_SDFRayMarch;
#endif
    };

}
