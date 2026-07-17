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
// Init() streams every cluster page in once at startup through the real streaming path (cache file
// -> staging -> BindPage -> DecompressPage), so everything the LOD cut could ever select is already
// resident before the first frame -- the strongest possible anti-stutter guarantee for a demoscene-
// sized scene. On top of that baseline, m_Streaming (GeometryStreamingCoordinator) drives the async
// residency loop for real, every frame (FeedbackBuffer misses from m_LODSelection's own residency
// check -> StreamingRequestQueue -> AsyncFileStreamer -> BindPageEvictingIfFull), so this class is
// exercised exactly as a scene that DOESN'T fit entirely resident would be -- see that class's own
// class comment for the exact per-frame sequencing contract with m_LODSelection's feedback buffer.
//
// --- LOD scope ---
// The candidate cluster set fed to the occlusion cull is NOT a fixed DAG level anymore -- m_LODSelection
// (ClusterLODSelectionPass) rewrites it every frame from a real GPU-driven DAG cut (ClusterDAGScreenError
// .comp's per-node projected-error test -> ClusterLODResidencyFallback.comp's non-resident-node
// ancestor walk, so a cut node whose page hasn't streamed in yet never reaches the raster worklist ->
// ClusterLODCompact.comp's final candidate emission), upstream of m_OcclusionCulling with no change to
// anything downstream -- see ClusterLODSelectionPass's own class comment for the full 3-dispatch
// sequence. Entity self-rotation (EntityTransform, cluster_entity_transform.glsl) IS applied by the
// clustered path (culling/LOD/raster/resolve), gated by config::ENTITY_SELF_ROTATION_ENABLED --
// VulkanContext::UpdateEntityRotations() is called every frame (main.cpp), before RecordFrame().
// Phase 4 integration (dynamic scenes onto main) additionally threads that same per-frame rotation
// into the ray-traced/GI side of this pipeline (TLAS refit, Surface Cache re-capture, Global SDF
// object-space compositing -- see m_SurfaceCacheRT/m_SurfaceCache/m_GlobalSDF's own comments below).
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
#include "core/EngineConfig.h"
#include "core/EntityData.h" // core::EntityTransformCPU
#include "core/LoadingManager.h"
#include "core/maths/Maths.h"
#include "renderer/passes/ATrousDenoisePass.h"
#include "renderer/passes/ClusterHardwareRasterPass.h"
#include "renderer/passes/ClusterLODSelectionPass.h"
#include "renderer/passes/ClusterOcclusionCullingPass.h"
#include "renderer/passes/ClusterResolvePass.h"
#include "renderer/passes/ClusterShadingBinPass.h"
#include "renderer/passes/ClusterSoftwareRasterPass.h"
#include "renderer/passes/GeometryDecompressionPass.h"
#include "renderer/streaming/GeometryStreamingCoordinator.h"
#include "renderer/passes/GlobalSDFPass.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/streaming/GpuGeometryPagePool.h"
#include "renderer/passes/HeroTessellationPass.h"
#include "renderer/passes/HZBPass.h"
#include "renderer/LightingTypes.h"
#include "renderer/passes/ProceduralMaskGenerator.h"
#include "renderer/passes/ReflectionPass.h"
#include "renderer/passes/ScreenSpaceEffectsPass.h"
#include "renderer/passes/MegaLightsPass.h"
#include "renderer/passes/ScreenProbeGIPass.h"
#include "renderer/passes/SurfaceCacheGIInjectPass.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/passes/TransparentForwardPass.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/passes/VirtualTextureRenderPass.h"
#include "renderer/streaming/VirtualTextureManager.h"
#include "renderer/streaming/VirtualTextureStreamingCoordinator.h"
#include "renderer/passes/WorldProbeGridPass.h"
#include "renderer/passes/TAATSRPass.h"
#include "renderer/passes/DepthOfFieldPass.h"
#include "renderer/passes/BloomPass.h"
#include "renderer/passes/PostProcessPass.h"
#include "renderer/passes/ScreenTracePass.h"
#include "renderer/passes/GICompositePass.h"
#ifndef NDEBUG
#include "renderer/debug/ClusterTriangleStatsPass.h"
#include "renderer/debug/DebugTextOverlay.h"
#include "renderer/passes/SDFRayMarchPass.h"
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

        // Dedicated transfer queue (UE 5.8 RHI parity) -- see VulkanContext::GetTransferQueue()'s
        // own comment. Forwarded to m_PagePool.Init() so it knows whether the geometry page pool's
        // upload path needs queue-family-ownership-transfer barriers (distinct family) or none
        // (fallback: same family as `queue` above).
        uint32_t graphicsQueueFamilyIndex = 0;
        uint32_t transferQueueFamilyIndex = 0;

        VkExtent2D renderExtent{ 0, 0 };
        VkExtent2D displayExtent{ 0, 0 };

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

        // Entity transform and data buffers for dynamic primitive rotations
        VkBuffer entityTransformBuffer = VK_NULL_HANDLE;
        VkBuffer entityDataBuffer = VK_NULL_HANDLE;

        // Randomly-generated PBR materials (renderer::VulkanContext::GetMaterialTable(), itself
        // built by renderer::GenerateRandomMaterialTable) -- uploaded once into ClusterResolvePass's
        // GPU SSBO and TransparentForwardPass's own descriptor set.
        MaterialTable materialTable{};
    };

    class ClusterRenderPipeline {
    public:
        ClusterRenderPipeline() = default;

        ClusterRenderPipeline(const ClusterRenderPipeline&) = delete;
        ClusterRenderPipeline& operator=(const ClusterRenderPipeline&) = delete;

        // Average estimated triangle screen size (pixels) below which a cluster is routed to the
        // software rasterizer -- see ClusterHZBOcclusionCull.comp's ShouldUseSoftwareRaster().
        static inline float kSoftwareRasterThresholdPixels = 8.0f;

        // Target on-screen geometric error (pixels) for the LOD DAG cut -- see
        // ClusterDAGScreenError.comp's pixelThreshold. A node draws once its own projected error
        // drops below this value while its parent's still exceeds it; typical Nanite-style engines
        // target roughly 1 pixel.
        static inline float kLODPixelErrorThreshold = 0.5f;

        // Number of consecutive SurfaceCacheGIInjectPass::RecordInject calls made THIS frame (see
        // RecordFrame()'s own [1z] block). Each call advances m_GIInject's own round-robin cursor by
        // its fixed per-call card budget (SurfaceCacheGIInjectPass::kCardsPerFrameBudget) and is
        // barrier-isolated from the next, so bounce N+1's freshly-processed cards sample the Surface
        // Cache atlas INCLUDING bounce N's own same-frame writes wherever their sampled footprints
        // overlap -- a genuine intra-frame multi-bounce chain, layered on top of (not a replacement
        // for) the pass's existing cross-frame convergence. 3 is enough for indirect light to
        // visibly wrap a corner within a single frame without tripling GI cost for a barely-visible
        // 4th+ bounce (each bounce's marginal energy contribution falls off quickly against typical
        // scene albedo).
        static inline uint32_t kRadiosityBounceCount = 4u;

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
        // begins/ends/submits the command buffer and presents. `entityTransformsCPU` (Phase 4
        // integration, UE5.8 parity roadmap, dynamic scenes onto main -- renderer::VulkanContext::
        // GetEntityTransformsCPU(), index == meshID) drives this frame's TLAS refit
        // (m_SurfaceCacheRT.RecordRefreshTLAS) and Global SDF object-space compositing
        // (m_GlobalSDF.RecordUpdate) -- see those two call sites' own comments.
        // `transferCmd` (VulkanContext::GetTransferCommandBuffer(), already begun/ended and
        // submitted to the transfer queue by the caller with the ordering main.cpp's per-frame
        // sequence establishes) is where this frame's geometry page uploads are recorded -- see
        // GeometryStreamingCoordinator::ProcessFeedbackAndDrainCompletions's own comment.
        void RecordFrame(VkCommandBuffer cmd, VkCommandBuffer transferCmd, const CameraPushConstants& camera,
            const maths::vec3& cameraPositionWorld, const CameraFrameInfo& cameraFrameInfo,
            float globalTimeSeconds, VkImage swapchainImage, VkImageView swapchainImageView,
            const core::EntityTransformCPU* entityTransformsCPU);

        // Upper bound on this frame's actual candidate count (the DAG's total leaf count) -- NOT
        // this frame's real candidate count, which only ever exists on the GPU now that
        // m_LODSelection computes a dynamic per-frame cut (see ClusterLODSelectionPass).
        uint32_t GetClusterCount() const { return m_ClusterCount; }

#ifndef NDEBUG
        // SWRT/HWRT back-end toggle shared by m_GIInject and m_WorldProbes' own trace pass (0 = SWRT
        // mesh-SDF sphere tracing, 1 = HWRT inline rayQueryEXT against m_SurfaceCacheRT's TLAS) --
        // debug-only (main.cpp's 'T'/'Y' explicit-set keys) so both back-ends stay exercised; Release
        // always uses HWRT (see RecordFrame()'s own use of this member). m_ScreenTrace does not use
        // this toggle at all -- its own march is a plain screen-space depth march, not an SWRT/HWRT
        // trace (see ScreenTracePass's own class comment).
        void SetDebugTraceMode(uint32_t traceMode) { m_DebugTraceMode = traceMode; }

        // Independently gates RecordFrame()'s [1z] intra-frame radiosity bounce loop
        // (kRadiosityBounceCount calls to SurfaceCacheGIInjectPass::RecordInject) so its cost and
        // visual contribution can be A/B'd via the FPS counter and the resolved image -- see
        // main.cpp's 'G' key. Defaults to true (Release has no toggle and always runs the loop).
        void SetDebugRadiosityEnabled(bool enabled) { m_DebugRadiosityEnabled = enabled; }

        // Independently gates RecordFrame()'s [12b] m_ScreenTrace.RecordTrace call (Lumen-style
        // Screen Trace: linear screen-space depth march, falling back to m_WorldProbes on a miss --
        // see ScreenTracePass's own class comment) entirely -- skipped means m_ScreenTrace's output
        // image is cleared to zero instead of traced this frame, so its cost/contribution is
        // directly A/B-able -- see main.cpp's 'F' key. Defaults to true (Release has no toggle and
        // always runs the trace).
        void SetDebugSSRTEnabled(bool enabled) { m_DebugSSRTEnabled = enabled; }

        // Gates RecordFrame()'s [12c] m_WorldProbes.RecordUpdate() dispatch -- see main.cpp's 'H'
        // key. This grid now has real live consumers every frame: m_ScreenTrace's own miss fallback
        // and GICompositePass's DEBUG_VIEW_SPATIAL_PROBES visualization (both wired in by the
        // ScreenTracePass/GICompositePass integration -- see m_WorldProbes' own member comment), so
        // Release hardcodes this ON (RecordFrame() always dispatches the update), matching
        // SetDebugRadiosityEnabled/SetDebugSSRTEnabled's own Release-always-on convention. Debug
        // defaults to true too, purely for A/B cost comparison via this toggle.
        void SetDebugWorldProbesEnabled(bool enabled) { m_DebugWorldProbesEnabled = enabled; }

        // Independently gates RecordFrame()'s [12b2] m_Reflection RecordTrace/RecordTemporal/
        // RecordGather trio -- see main.cpp's 'R' key. Like SetDebugWorldProbesEnabled above, this
        // pass has a real live consumer from its very first frame (m_Resolve's own output color
        // image, via a direct-RMW convention), so it defaults to true here AND Release hardcodes it
        // permanently on (no toggle exists in Release, matching SetDebugSSRTEnabled/
        // SetDebugRadiosityEnabled/SetDebugWorldProbesEnabled's own Release-always-on convention).
        void SetDebugReflectionsEnabled(bool enabled) { m_DebugReflectionsEnabled = enabled; }
        void SetDebugTAATSREnabled(bool enabled) { m_DebugTAATSREnabled = enabled; }

        // Phase A of the MegaLights native-port roadmap: gates RecordFrame()'s [12b3]
        // m_MegaLights.RecordShade() call -- see main.cpp's 'X' key. Same Release-always-on
        // convention as SetDebugReflectionsEnabled above (a real live consumer from frame one, via
        // the same additive-RMW-into-m_Resolve's-color-image convention reflections already use).
        void SetDebugMegaLightsEnabled(bool enabled) { m_DebugMegaLightsEnabled = enabled; }

        // Investigating the 2026-07-16 "persistent holes" bug (see project memory
        // project_persistent_cluster_holes_open_bug.md / ClusterLODSelectionPass::
        // DebugLogDAGCutGaps()'s own doc comment for the full analysis this triggers). Call once
        // from main.cpp's 'K' key handler: arms a one-shot dump that RecordFrame() records THIS
        // frame (m_LODSelection.RecordDebugReadback()), then main.cpp calls
        // PumpDebugDAGCutGapsDump() at the top of the NEXT frame (after that frame's fence has
        // been waited on, guaranteeing the readback landed) to actually log it.
        void RequestDebugDAGCutGapsDump() { m_DebugDAGCutGapsDumpState = 1; }

        // See RequestDebugDAGCutGapsDump()'s doc comment -- must be called once per frame, after
        // this frame's own frameFence wait, before RecordFrame(). A no-op unless a dump is
        // actually pending.
        void PumpDebugDAGCutGapsDump() {
            if (m_DebugDAGCutGapsDumpState == 2) {
                m_LODSelection.DebugLogDAGCutGaps();
                m_DebugDAGCutGapsDumpState = 0;
            }
        }

        // Last HW/SW rasterizer triangle split read by RecordFrame()'s own [13b] stat-overlay
        // block (m_TriangleStats.ReadStats(), always one frame behind -- see that block's own
        // comment). Exposed for DebugTestPipeline's feature tests, which have no other way to
        // observe m_TriangleStats (private, overlay-only otherwise).
        void GetDebugTriangleStats(uint32_t& outHwTriangleCount, uint32_t& outSwTriangleCount) const {
            outHwTriangleCount = m_LastHWTriangleCount;
            outSwTriangleCount = m_LastSWTriangleCount;
        }
#endif

    private:
        // Records the vkCmdBeginRendering block shared by the early and late hardware raster
        // passes: `clearAttachments` selects between the early pass's CLEAR (VisBuffer sentinel
        // 0xFFFFFFFF, depth 1.0) and the late pass's LOAD (draw on top of the early output).
        void BeginVisBufferRendering(VkCommandBuffer cmd, bool clearAttachments) const;

        VkDevice m_Device = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };
        VkExtent2D m_DisplayExtent{ 0, 0 };

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

        // Generic multithreaded background-job pool (hardware_concurrency worker threads) -- see
        // core::LoadingManager's own class comment. Currently consumed by m_GlobalSDF::Init() to
        // fan its per-entity Mesh SDF bake out across every core instead of a single-threaded loop
        // (see GlobalSDFPass::Init's own comment for why that stays a blocking WaitIdle() rather
        // than a frame-spread arrival); owned at this pipeline-wide scope, not inside GlobalSDFPass
        // itself, so any future CPU-bound generator (procedural terrain/tree/texture systems -- see
        // CLAUDE.md's stated scope) can share the same worker pool instead of each spinning up its
        // own threads. Deliberately NOT torn down by Shutdown() (which also runs defensively at the
        // top of every Init() call, unlike this pool's intended whole-pipeline-lifetime scope) --
        // see Shutdown()'s own comment in the .cpp for why. Only core::LoadingManager's own
        // destructor ever joins its worker threads, exactly once, when this object is destructed.
        core::LoadingManager m_LoadingManager;

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
        // Phase 1b (UE5.8 parity roadmap): the 3-stage GPU counting sort that buckets VisBuffer
        // pixels by materialID, feeding m_Resolve's own RecordResolveBinned() -- see
        // renderer::ClusterShadingBinPass's own class comment for the full pipeline. Recorded
        // between [11] and [12] below (after the VisBuffer/depth hand-off, before shading), unlike
        // m_Resolve which stays unconditional/always-initialized -- this pass is likewise always
        // initialized (not Debug-only), matching CLAUDE.md's build-separation rule: only the
        // NUMPAD-DRIVEN DEBUG VIEW SWITCHING that picks between this path and m_Resolve's original
        // RecordResolve() is Debug-only, not the pass itself (Release always uses this path).
        ClusterShadingBinPass m_ShadingBin;
        // Forward-rendered translucent/transparent materials -- see TransparentForwardPass's own
        // class comment for why this is a completely separate path from the opaque VisBuffer
        // pipeline above. Recorded once GI/reflections have fully composited the opaque scene (so
        // transparency draws on top of the final lit result), always initialized (not Debug-only),
        // same build-separation rule as m_ShadingBin above.
        TransparentForwardPass m_TransparentForward;

        // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): forward-rendered, screen-space-
        // adaptively-tessellated, procedurally-displaced hero Icosphere (materialID
        // kHeroMaterialID, culled out of the opaque Nanite path entirely via core::EntityFlags::
        // IsTransparent -- see VulkanContext::BuildEntityData()'s own comment) -- lit exactly like
        // m_TransparentForward above (direct+shadowed sun, MegaLights RIS point lights,
        // m_WorldProbes' indirect diffuse, an optional single-sample front-layer specular
        // reflection), but fully OPAQUE and depth-WRITING (unlike glass). Recorded right BEFORE
        // m_TransparentForward's own draw -- specifically so that pass' own read-only depth test
        // correctly occludes glass/translucent entities against this entity's real (displaced)
        // surface, see HeroTessellationPass's own class comment for the resulting barrier-scope
        // consequence.
        HeroTessellationPass m_HeroTessellation;

        // Lumen-style GI infrastructure -- unlike the debug-only stats/overlay block below, these
        // are real (if not yet light-transport-consuming) systems, not visualization tools, so
        // they compile in Release too (matching how the actual Nanite cluster pipeline above is
        // likewise unconditional; only the NUMPAD-DRIVEN DEBUG VIEW SWITCHING is Debug-only, per
        // CLAUDE.md's build-separation rule). Recorded every frame in RecordFrame() regardless of
        // camera.debugViewMode.
        // Phase 3 (UE5.8 parity roadmap): replaces the pre-Phase-3 ShadowMapPass (a single
        // whole-scene-fit 2048x2048 orthographic map, fully redrawn every frame, sun only) with a
        // genuine page-table-virtualized system -- a 3-level clipmap of Virtual Shadow Maps for the
        // sun plus a per-point-light 6-face cube, sharing one physical page pool, incrementally
        // updated (bounded pages rendered per frame) instead of a full redraw -- see
        // VirtualShadowMapPass's own class comment for the full per-frame contract.
        // ShadowMapPass.h/.cpp remain in the repo (file deletion blocked, see memory
        // feedback_file_deletion_blocked) but are no longer instantiated here.
        VirtualShadowMapPass m_VirtualShadowMap;
        // Step 4 (Virtual Texturing / RVT-SVT, UE 5.8 parity): renderer::VirtualTextureManager owns
        // the page table + physical atlas (own single Albedo pool, see Init()'s own comment on the
        // format chosen), renderer::VirtualTextureRenderPass renders terrain/decal pages on demand
        // (not yet driven by any real per-frame page-visibility source -- see Init()'s own comment),
        // and renderer::VirtualTextureStreamingCoordinator streams previously-baked tiles back off
        // disk (.vtcache, io::VirtualTextureCacheFormat.h) via GPU feedback misses -- mirrors the
        // exact three-way split VirtualShadowMapPass/m_Streaming already establish for shadow pages/
        // geometry clusters respectively. Gated by config::lumen::BUILD_VIRTUAL_TEXTURES at the
        // per-frame streaming step only (see VirtualTextureStreamingCoordinator::RecordBeginFrame's
        // own comment) -- Init() and the descriptor wiring below always run unconditionally, exactly
        // like VirtualShadowMapPass's own config::lumen::BUILD_SHADOWS gating.
        VirtualTextureManager m_VTManager;
        VirtualTextureRenderPass m_VTRenderPass;
        VirtualTextureStreamingCoordinator m_VTStreaming;
        // The world-space volume m_VTRenderPass/m_Resolve's VirtualTextureVolumeUBO both address --
        // retained so RecordFrame() (or a future terrain system) can call m_VTRenderPass::RequestPage
        // using the SAME bounds Init() wired into the consuming shader.
        VirtualTextureVolumeBounds m_VTBounds;
        SurfaceCachePass m_SurfaceCache;
        GlobalSDFPass m_GlobalSDF;
        // Sun direction is fixed by default; one point light is authored in Init() specifically to
        // exercise/verify Phase 3's point-light Virtual Shadow Maps (see Init()'s own comment) --
        // see renderer::LightingTypes.h's own comment for the full field-by-field default.
        SceneLights m_SceneLights;

        // Shared trace-scene descriptor sets (mesh SDF trace + Surface Cache sampling, see
        // SurfaceCacheTraceContext's own class comment) built once from m_GlobalSDF + m_SurfaceCache,
        // reused unmodified by m_SurfaceCacheRT, m_GIInject, and m_WorldProbes' own trace pass.
        SurfaceCacheTraceContext m_TraceContext;
        // HWRT back-end: one BLAS per traced entity + one TLAS, built once (static scene) against
        // m_SurfaceCache's own Fallback Mesh vertex/index buffers.
        SurfaceCacheRayTracingPass m_SurfaceCacheRT;
        // Per-card hemisphere-sampled secondary-bounce injection into m_SurfaceCache's own radiance
        // atlas -- makes that atlas genuinely multi-bounce over time, which every downstream GI
        // consumer (m_WorldProbes' own trace pass, m_ScreenTrace's near-field hit samples) then
        // benefits from.
        SurfaceCacheGIInjectPass m_GIInject;

        // NOTE: this class previously also owned a ScreenProbeGIPass ("Screen Space Probe GI" /
        // Lumen "Screen Probe Gather") here -- a per-8x8-tile SH probe grid, temporally accumulated,
        // gathered back into m_Resolve's output color image via read-modify-write. The
        // ScreenTracePass/GICompositePass integration below replaced it as this codebase's near-
        // field screen-space GI term (a plain per-pixel screen-space march instead of a per-tile
        // probe grid); the instantiation was removed here to stop paying for its ~10 full-resolution
        // images/3 pipelines every run. ScreenProbeGIPass.h/.cpp and its 3 shaders (ScreenProbeTrace/
        // Temporal/Gather.comp) remain in the repo (file deletion blocked, see memory
        // feedback_file_deletion_blocked) but are no longer instantiated here -- same convention as
        // ShadowMapPass.h/.cpp above.

        // Phase 2 (UE5.8 parity roadmap): specular reflections / GI -- traces ONE GGX-VNDF-
        // importance-sampled ray per pixel per frame (full resolution), sampling the same
        // m_SurfaceCache radiance atlas at each hit, and Fresnel-weighted read-modify-writes the
        // result into m_Resolve's own output color image -- see ReflectionPass's own class comment
        // for the full pipeline and why its storage is raw RGBA16F radiance, not spherical harmonics
        // (physically incapable of a narrow specular lobe).
        ReflectionPass m_Reflection;

        // Phase PP4 (post-process stack roadmap): GTAO (feeds m_GIComposite below) + Screen-Space
        // Contact Shadows (multiplicative RMW on m_Resolve's output color, called right after [12]
        // Resolve) + SSR Fallback (additive RMW on the same image, called after m_Reflection's own
        // Trace/Temporal/Gather trio) -- see ScreenSpaceEffectsPass's own class comment for exactly
        // why 3 independent Record*() call sites are needed instead of one.
        ScreenSpaceEffectsPass m_ScreenSpaceEffects;

        // Phase A of the MegaLights native-port roadmap: RIS-weighted stochastic multi-point-light
        // direct lighting (up to kMaxMegaLights procedurally-authored point lights), one ray-traced
        // shadow-visibility ray per pixel, additively read-modify-writing m_Resolve's own output
        // color image -- see MegaLightsPass's own class comment for the full pipeline (Shade ->
        // owned À-Trous denoise -> Composite) and why it owns a DEDICATED ATrousDenoisePass instance
        // rather than inheriting m_Denoiser below.
        MegaLightsPass m_MegaLights;

        // World Probe grid (Lumen "Translucency Volume" / global illumination volume): a low-
        // resolution, camera-centered 3D grid of ambient irradiance probes, fully rebuilt every
        // frame from m_SurfaceCache's radiance atlas, sampled via world_probe_sampling.glsl's
        // SampleWorldProbeGrid(). Live consumers: m_ScreenTrace's own miss fallback (every frame)
        // and GICompositePass's DEBUG_VIEW_SPATIAL_PROBES visualization (Debug only) -- see
        // WorldProbeGridPass's own class comment.
        WorldProbeGridPass m_WorldProbes;
        // Screen Trace GI (Lumen-style "Screen Trace"): per-pixel linear screen-space depth march,
        // near-field hit samples m_Resolve's own direct-lit color, miss falls back to m_WorldProbes
        // -- see ScreenTracePass's own class comment.
        ScreenTracePass m_ScreenTrace;
        // Final GI composite: direct-lit color + m_Denoiser's denoised indirect term, and (Debug
        // only) the DEBUG_VIEW_LUMEN/DEBUG_VIEW_SPATIAL_PROBES visualizations -- see
        // GICompositePass's own class comment.
        GICompositePass m_GIComposite;

        // Final spatial denoiser (À-Trous wavelet, edge-guided by m_Resolve's own G-buffer normal/
        // depth): a spatial cleanup pass over m_ScreenTrace's own noisy combined (near-field +
        // World Probe fallback) indirect term, applied only in the normal (non-debug-visualization)
        // view path -- see RecordFrame()'s own comment for exactly why.
        ATrousDenoisePass m_Denoiser;
        TAATSRPass m_TAATSR;
        // Phase PP3 (post-process stack roadmap): physically-derived Depth of Field -- reads
        // m_TAATSR's own HDR output directly, runs BEFORE m_Bloom (below) so an out-of-focus
        // highlight is already a soft disc by the time Bloom's own bright-pass threshold sees it --
        // see DepthOfFieldPass's own class comment. m_Bloom and m_PostProcess both read ITS
        // GetOutputView() now, not m_TAATSR's directly.
        DepthOfFieldPass m_DepthOfField;
        // Phase PP2 (post-process stack roadmap): Bloom / Lens Flare / Anamorphic Lens Flare / Lens
        // Dirt, all one dual-filter mip chain reading m_DepthOfField's own output -- see BloomPass's
        // own class comment. Recorded before m_PostProcess (below), whose composite shader samples
        // its GetOutputView() and adds it into the scene color.
        BloomPass m_Bloom;
        // Phase PP1 (post-process stack roadmap): Physical Camera / Auto Exposure / White Balance /
        // Color Correction / ACES Tone Mapping / Gamma Correction -- the normal-view-path blit
        // source instead of m_TAATSR's own raw HDR output directly (see PostProcessPass's own class
        // comment for the full 3-stage pipeline and why Gamma Correction here is the pipeline's
        // only display encode, not decorative).
        PostProcessPass m_PostProcess;

        // Previous frame's combined view-projection matrix -- ClusterResolve.comp's own
        // DEBUG_VIEW_MOTION_VECTORS reprojects a pixel's (static, see this class' own "Entity self-
        // rotation" scope note) world position with this matrix to find its screen position last
        // frame. Updated at the very end of RecordFrame(), after every consumer of "the previous
        // frame's" value has run.
        maths::mat4 m_PrevViewProj{};
        bool m_HasPrevViewProj = false;

        // Advances once per RecordFrame() call -- ScreenProbeTrace.comp's own per-frame Fibonacci-
        // sphere jitter rotation (include/sh_probe.glsl's JitterDirection).
        uint32_t m_FrameIndex = 0;

        // globalTimeSeconds (RecordFrame()'s own parameter) from the previous call -- differenced
        // against this frame's own value to get a real per-frame delta time for m_PostProcess's
        // Auto Exposure eye-adaptation (PostProcessPass::RecordComposite's own deltaTimeSeconds
        // parameter). Not Debug-only (unlike the stat overlay's own m_LastStatsSampleTime below):
        // exposure adaptation is a real Release-time effect, not debug tooling.
        float m_LastFrameTimeSeconds = 0.0f;
        bool m_HasLastFrameTime = false;

#ifndef NDEBUG
        uint32_t m_DebugTraceMode = 0;
        bool m_DebugRadiosityEnabled = true;
        bool m_DebugSSRTEnabled = true;
        // See SetDebugWorldProbesEnabled()'s own comment for why this defaults to true in Debug
        // while Release hardcodes the equivalent local to false (opposite of the two toggles above).
        bool m_DebugWorldProbesEnabled = true;
        // See SetDebugReflectionsEnabled()'s own comment: unlike m_DebugWorldProbesEnabled above,
        // Release hardcodes the equivalent local to true (matching m_DebugSSRTEnabled's own
        // Release-always-on convention), since this pass has a real live consumer already.
        bool m_DebugReflectionsEnabled = true;
        // See SetDebugMegaLightsEnabled()'s own comment: same Release-always-on convention as
        // m_DebugReflectionsEnabled above.
        bool m_DebugMegaLightsEnabled = true;
        bool m_DebugTAATSREnabled = config::temporal::ENABLED_BY_DEFAULT;

        // See RequestDebugDAGCutGapsDump()'s own comment: 0 = idle, 1 = "record the readback this
        // frame" (set by main.cpp's 'K' key), 2 = "readback landed, safe to log now" (set by
        // RecordFrame() right after it records the copy, consumed by PumpDebugDAGCutGapsDump()).
        uint32_t m_DebugDAGCutGapsDumpState = 0;
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
        // See GetDebugTriangleStats()'s own comment.
        uint32_t m_LastHWTriangleCount = 0;
        uint32_t m_LastSWTriangleCount = 0;

        // Two-tier SDF ray march DEBUG VISUALIZATION (see renderer::SDFRayMarchPass's own class
        // comment: its whole output is "a full-screen debug-visualization image", never consumed
        // by production lighting) -- Debug-only per CLAUDE.md's build-separation rule, exactly
        // like m_TriangleStats/m_DebugOverlay above. Only recorded (and only ever blitted to the
        // swapchain in place of m_Resolve's output) when camera.debugViewMode == DEBUG_VIEW_LUMEN.
        SDFRayMarchPass m_SDFRayMarch;
#endif
    };

}
