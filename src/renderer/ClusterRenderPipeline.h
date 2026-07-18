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
// --- Per-frame GPU work ---
// Phase 2 (Lumen advanced roadmap) fix, 2026-07-17: this class used to record the ENTIRE frame
// into ONE graphics command buffer, submitted once. That single-submission design turned out to
// make the async-compute queue's Surface Cache work (TLAS refit + radiosity injection) NOT
// actually concurrent with anything: per the Vulkan spec, a submission's signal semaphores only
// fire once EVERY command buffer in that submission has finished executing on the GPU, so the
// async-compute queue could never start before the entire graphics frame (including post-process
// and the final blit) had already finished -- and the next frame's own top-of-loop fence wait for
// that async work then serialized everything AGAIN. Net effect: graphics(N) -> async(N) ->
// graphics(N+1) -> async(N+1) -> ..., zero real overlap, despite the barrier infrastructure being
// otherwise correct. Fixed by splitting the frame into THREE graphics command buffers/submissions
// (still zero mid-frame vkQueueWaitIdle/vkDeviceWaitIdle -- the anti-stutter guarantee is
// preserved, "one submit per phase" replaces "one submit per frame"), letting the async-compute
// queue's work run genuinely concurrently with the middle one:
//   - RecordFrameEarly() -> cmdEarly: Virtual Shadow Map/Virtual Texture "begin frame", Surface
//     Cache capture, Global SDF update, and (async compute active) the RELEASE half of the Surface
//     Cache atlas/TLAS ownership transfer to the async-compute queue family. Signals
//     VulkanContext::GetAsyncComputeCanStartSemaphore().
//   - RecordAsyncCompute() -> asyncComputeCmd (separate queue): ACQUIRE + SurfaceCacheRayTracingPass
//     ::RecordRefreshTLAS + the radiosity bounce loop + RELEASE back to graphics. Waits on
//     can-start, signals VulkanContext::GetAsyncComputeFinishedSemaphore(). Runs CONCURRENTLY with
//     cmdMid below on the GPU (different queue, no wait between them).
//   - RecordFrameMid() -> cmdMid: everything from this section's old steps 1-7 below (worklist
//      clears -> two-phase cull/raster -> resolve) PLUS this frame's geometry-streaming triage --
//      confirmed (by reading every pass' own descriptor bindings, not assumed) to never sample the
//      Surface Cache atlas, the TLAS, or GI-related Global SDF state, so it needs no wait on the
//      async-compute queue at all. Same-queue submission order (not a semaphore) is what orders it
//      after cmdEarly.
//   - RecordFrameLate() -> cmdLate: (async compute active) the ACQUIRE half of the ownership
//     transfer back from the async-compute queue family, then every pass that DOES read the
//     Surface Cache atlas/TLAS this same frame (Screen Trace/World Probe fallback, Reflection,
//     MegaLights' shadow-visibility rays, World Probe grid rebuild, the 3 forward passes), the
//     final [8]/[9] HZB rebuild + blit below, and post-process. Waits on
//     GetAsyncComputeFinishedSemaphore() (plus the swapchain-image-acquire and transfer-queue
//     semaphores) since this is the submission that actually needs the async work's THIS-FRAME
//     output -- a genuine same-frame hand-off now, not the old design's one-frame pipeline lag.
// See main.cpp's own per-frame submit-sequence comment for the exact semaphore/fence graph, and
// VulkanContext::GetCommandBufferEarly()'s own comment for why 3 command buffers from the SAME
// pool need no semaphore between themselves (same-queue submission order suffices, exactly like
// this class' existing intra-frame barriers already relied on for e.g. the [8] HZB rebuild feeding
// next frame's [2] early cull).
//
// The steps below (now split across cmdMid/cmdLate as described above) are otherwise unchanged:
//   1. Clear the frame's atomic worklists (pending/early/late/software counts) + the software
//      rasterizer's atomic VisBuffer.                                                    [cmdMid]
//   2. EARLY cull: every leaf cluster vs frustum/backface + last frame's HZB; survivors routed to
//      the early hardware draw list or the software cluster list; uncertain ones to the pending
//      list.                                                                              [cmdMid]
//   3. Early hardware raster into the VisBuffer (2x R32_UINT) + depth.                    [cmdMid]
//   4. HZB rebuild from the early depth; LATE cull re-tests exactly the pending list against it.
//                                                                                          [cmdMid]
//   5. Late hardware raster (loadOp = LOAD on top of the early pass's output).            [cmdMid]
//   6. Software raster of every micro-triangle cluster listed this frame (early + late routed).
//                                                                                          [cmdMid]
//   7. Resolve: per-pixel hardware-vs-software depth arbitration, barycentric reconstruction,
//      material evaluation into the output color image.                                  [cmdMid]
//   8. Second HZB rebuild from the now-complete depth, so next frame's EARLY pass tests against
//      the full previous-frame occlusion state (the standard two-phase scheme).          [cmdLate]
//   9. Blit the resolved image to the swapchain image and transition it to PRESENT_SRC_KHR.
//                                                                                         [cmdLate]
// Every inter-pass hazard is covered by an explicit VkMemoryBarrier2/VkImageMemoryBarrier2 inside
// the Record*() methods above or inside the passes' own Record*() functions -- see the .cpp for
// the exact stage/access/layout reasoning at each step.

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
#include "renderer/passes/AtmosClimatePass.h"
#include "renderer/passes/AtmosCloudsPass.h"
#include "renderer/passes/AtmosSkyPass.h"
#include "renderer/passes/AtmosVolumetricFogPass.h"
#include "renderer/passes/GlobalSDFPass.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/streaming/GpuGeometryPagePool.h"
#include "renderer/passes/TessellationPass.h"
#include "renderer/passes/WaterForwardPass.h"
#include "renderer/passes/ParticleSystemPass.h"
#include "renderer/passes/VegetationScatterPass.h"
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
#include "renderer/debug/DebugBufferViewPass.h"
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

        // Phase 2 (Lumen advanced roadmap): dedicated async-compute queue (UE 5.8 RHI parity) --
        // see VulkanContext::GetAsyncComputeQueue()'s own comment. `asyncComputeQueue` is the
        // actual queue submissions target (VulkanContext::GetAsyncComputeQueue(), already aliased
        // to the graphics queue when no dedicated family exists); `asyncComputeQueueFamilyIndex`/
        // `hasDedicatedAsyncComputeQueue` decide whether the acquire/release barriers this class
        // records around the Surface Cache atlas images + the TLAS actually need to transfer queue
        // family ownership (same family == none needed, mirrors m_PagePool's own
        // transferQueueFamilyIndex convention).
        VkQueue asyncComputeQueue = VK_NULL_HANDLE;
        uint32_t asyncComputeQueueFamilyIndex = 0;
        bool hasDedicatedAsyncComputeQueue = false;

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
        // Feature 2 (Lumen advanced roadmap): CPU-side mirror of entityDataBuffer's own contents
        // (renderer::VulkanContext::GetEntityData(), index == meshID), needed by
        // SurfaceCachePass::Init() to read each card's owning entity's real materialID BEFORE any
        // GPU buffer is bound/read -- see that method's own comment for why this must be a CPU-side
        // array, not a GPU readback, at pass-init time.
        const core::EntityData* entityDataCPU = nullptr;

        // Randomly-generated PBR materials (renderer::VulkanContext::GetMaterialTable(), itself
        // built by renderer::GenerateShowcaseMaterialTable) -- uploaded once into ClusterResolvePass's
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

        // Exposes the pipeline-wide worker pool (see m_LoadingManager's own comment: owned here,
        // not per-pass, specifically so future CPU-bound systems can share it) to runtime World
        // Partition streaming (world::StreamingManager, io::AsyncDecompressingLoader) constructed
        // in main.cpp -- previously only used internally by GlobalSDFPass::Init(). Must outlive
        // anything handed this reference, exactly as m_LoadingManager's own comment already states.
        core::LoadingManager& GetLoadingManager() { return m_LoadingManager; }

        // Phase 2 (Lumen advanced roadmap) fix: the frame is now recorded across 4 calls instead of
        // one RecordFrame() -- see this class' own header comment ("Per-frame GPU work") for the
        // full root-cause/redesign explanation and main.cpp for the exact submit sequence each call
        // sits between. Every call must happen in this order, every frame:
        //   RecordFrameEarly() -> [main.cpp ends/submits cmdEarly] ->
        //   RecordAsyncCompute() -> [main.cpp ends/submits asyncComputeCmd] ->
        //   RecordFrameMid() -> [main.cpp ends/submits transferCmd, then cmdMid] ->
        //   RecordFrameLate() -> [main.cpp ends/submits cmdLate, then presents]
        // `camera`/`cameraPositionWorld`/`cameraFrameInfo`/`globalTimeSeconds`/`entityTransformsCPU`
        // are only ever passed to RecordFrameEarly() (the first call each frame) -- every later call
        // reads the same values back from this frame's own scratch state (m_FrameScratch),
        // populated once by RecordFrameEarly() and valid for the rest of that same frame's 4 calls,
        // exactly as if this were still one function with function-scope locals.

        // [cmdEarly] Virtual Shadow Map/Virtual Texture "begin frame", Surface Cache capture,
        // Global SDF update, and -- only when this frame routes through the async-compute queue --
        // the RELEASE half of the Surface Cache atlas/TLAS ownership transfer to the async-compute
        // queue family (see RecordSurfaceCacheOwnershipTransfer's own comment for exactly which 5
        // images + 4 buffers). `camera` supplies view/proj (must be the frame's final matrices:
        // every later stage this frame uses the same viewProj, which is what makes the resolve
        // pass's barycentric reconstruction exact) and `cameraPositionWorld` the eye point for the
        // backface cone test. `globalTimeSeconds` (main.cpp's glfwGetTime(), monotonically
        // increasing) drives the World Position Offset sway function (wpo_deformation.glsl's
        // ApplyWPODeformation) -- computed here, actually uploaded in RecordFrameMid() before either
        // raster pass runs. `entityTransformsCPU` (Phase 4 integration, UE5.8 parity roadmap --
        // renderer::VulkanContext::GetEntityTransformsCPU(), index == meshID) drives this frame's
        // TLAS refit (m_SurfaceCacheRT.RecordRefreshTLAS, in RecordAsyncCompute()/here depending on
        // routing) and Global SDF object-space compositing (m_GlobalSDF.RecordUpdate, here).
        void RecordFrameEarly(VkCommandBuffer cmdEarly, const CameraPushConstants& camera,
            const maths::vec3& cameraPositionWorld, const CameraFrameInfo& cameraFrameInfo,
            float globalTimeSeconds, const core::EntityTransformCPU* entityTransformsCPU);

        // [asyncComputeCmd] ACQUIRE (from graphics) + SurfaceCacheRayTracingPass::RecordRefreshTLAS
        // + the radiosity bounce loop + RELEASE (back to graphics) -- a no-op (records nothing) if
        // this frame's routing decision (computed by RecordFrameEarly(), stored in m_FrameScratch)
        // was the fully-graphics-queue-serialized fallback path instead. `asyncComputeCmd`
        // (VulkanContext::GetAsyncComputeCommandBuffer(), already begun by the caller AFTER cmdEarly
        // has been submitted -- see main.cpp's own comment on why the async-compute fence wait is
        // deliberately deferred to right before this) is ended/submitted by the caller exactly like
        // cmdEarly.
        void RecordAsyncCompute(VkCommandBuffer asyncComputeCmd);

        // [cmdMid] The Nanite VisBuffer pipeline (worklist clears -> two-phase cull/raster -> HZB ->
        // resolve) plus this frame's geometry-streaming triage -- confirmed by reading every pass'
        // own descriptor bindings (not assumed) to never sample the Surface Cache atlas, the TLAS,
        // or GI-related Global SDF state, so this command buffer needs no wait on the async-compute
        // queue's own submission at all (only on the transfer queue's, for the geometry page pool's
        // own ownership-transfer acquire + decompression this method records). `transferCmd`
        // (VulkanContext::GetTransferCommandBuffer(), already begun by the caller) is where this
        // frame's geometry page uploads are recorded -- see
        // GeometryStreamingCoordinator::ProcessFeedbackAndDrainCompletions's own comment. The caller
        // ends/submits both `transferCmd` (first) and `cmdMid` (waiting on the former).
        void RecordFrameMid(VkCommandBuffer cmdMid, VkCommandBuffer transferCmd);

        // [cmdLate] -- only when this frame routed through the async-compute queue -- the ACQUIRE
        // half of the Surface Cache ownership transfer back from the async-compute queue family
        // (this command buffer's very first GPU work, since every pass below reads what it
        // acquires), then every pass that samples the Surface Cache atlas/TLAS THIS SAME FRAME
        // (Screen Trace's World Probe fallback, Reflection, MegaLights' shadow-visibility rays,
        // World Probe grid rebuild, the 3 forward passes), the second HZB rebuild, post-process, and
        // finally the blit of the resolved/post-processed image into `swapchainImage`
        // (`swapchainImageView` for the Debug-only ImGui overlay) + the PRESENT_SRC_KHR transition.
        // The caller ends/submits `cmdLate` (waiting on the swapchain-acquire semaphore and
        // VulkanContext::GetAsyncComputeFinishedSemaphore(), guarded by this frame's frameFence) and
        // presents.
        void RecordFrameLate(VkCommandBuffer cmdLate, VkImage swapchainImage, VkImageView swapchainImageView);

        // Upper bound on this frame's actual candidate count (the DAG's total leaf count) -- NOT
        // this frame's real candidate count, which only ever exists on the GPU now that
        // m_LODSelection computes a dynamic per-frame cut (see ClusterLODSelectionPass).
        uint32_t GetClusterCount() const { return m_ClusterCount; }

        // Exposes AtmosClimatePass's last computed Dew Point / LCL Height for main.cpp's Volumetric
        // ImGui tab (atmos_integration_plan.md Subtask 1, objective #4) -- same "borrow a const ref"
        // convention as e.g. GetTracedEntityInfos() elsewhere in this class.
        const AtmosClimatePass& GetAtmosClimate() const { return m_AtmosClimate; }

        // Exposes ParticleSystemPass for main.cpp's own Particles ImGui tab (Subtask 6's own
        // Debug-only GetLastAliveCountApprox() readout) -- same "borrow a const ref" convention as
        // GetAtmosClimate() above.
        const ParticleSystemPass& GetParticleSystem() const { return m_ParticleSystem; }

        // Exposes the vegetation scatter pass for main.cpp's own Debug "Vegetation" ImGui tab
        // (instance-count readout) -- same "borrow a const ref" convention as GetParticleSystem().
        const VegetationScatterPass& GetVegetationScatter() const { return m_VegetationScatter; }

#ifndef NDEBUG
        // Debug-only: re-runs the vegetation scatter generator from the current config::vegetation::
        // density/region/seed knobs. Waits for the device to go idle first (the generation is a
        // blocking one-shot submit on the graphics queue, so no in-flight frame may still reference
        // the instance buffer) -- backs the Debug "Vegetation" tab's Regenerate button.
        void RegenerateVegetationScatter();
#endif

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

        // Phase 2 (Lumen advanced roadmap): gates whether RecordFrame()'s [1z] block moves
        // SurfaceCacheRayTracingPass::RecordRefreshTLAS + the radiosity bounce loop onto the async-
        // compute queue (true, mirrors SetDebugRadiosityEnabled's own toggle shape) or keeps them
        // fully graphics-queue-serialized exactly as before this phase (false) -- see main.cpp's
        // debug key binding. Bringing the feature up with this forced false first validates the CPU/
        // struct plumbing alone (asyncComputeCmd is still begun/ended/submitted every frame either
        // way, just left empty); flipping it true then isolates any remaining issue to the actual
        // cross-queue barrier/semaphore design specifically, per this phase's own approved plan.
        // Forcing SWRT (traceMode == 0) also falls back to this same fully-serialized path
        // regardless of this toggle's own value -- see RecordFrame()'s own [1z] comment for why.
        // Defaults to true (Release has no toggle and always attempts the async-queue path,
        // matching SetDebugRadiosityEnabled/SetDebugSSRTEnabled's own Release-always-on convention;
        // when GetAsyncComputeAvailableThisBuild() is false -- no dedicated queue on this GPU -- the
        // request still "succeeds" from the caller's point of view, just with every barrier
        // degenerating to a no-op and both queue handles aliased to the same graphics queue, see
        // m_AsyncComputeAvailableThisBuild's own comment).
        void SetDebugAsyncComputeEnabled(bool enabled) { m_DebugAsyncComputeEnabled = enabled; }

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

        // Phase 1 (Nanite advanced): gates RecordFrame()'s per-frame WPOGlobalsUBO upload of
        // `enhancedDisplacementDebugMultiplier` -- 1.0 when enabled (full effect, the Release-
        // always-on value, matching SetDebugReflectionsEnabled's own convention -- Release hardcodes
        // this true, no toggle exists there), 0.0 when disabled (ApplyEnhancedDisplacement's output
        // mixed all the way back to the undisplaced position, see enhanced_displacement.glsl's own
        // call-site comment in e.g. ClusterRaster.vert). Main.cpp's 'B' key.
        void SetDebugEnhancedDisplacementEnabled(bool enabled) { m_DebugEnhancedDisplacementEnabled = enabled; }

        // Phase 1 (Nanite advanced): gates RecordFrame()'s per-frame WPOGlobalsUBO upload of
        // `splineDeformationDebugMultiplier` -- same 1.0/0.0 convention as
        // SetDebugEnhancedDisplacementEnabled above, mixing ApplySplineDeformation's bent local
        // position back toward the undeformed rest pose when disabled. Main.cpp's 'U' key.
        void SetDebugSplineDeformationEnabled(bool enabled) { m_DebugSplineDeformationEnabled = enabled; }

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

        // Phase 2 (Lumen advanced roadmap): records ONE HALF (release if `isRelease`, acquire
        // otherwise) of a queue-family-ownership transfer covering every resource
        // SurfaceCacheRayTracingPass::RecordRefreshTLAS + SurfaceCacheGIInjectPass::RecordInject
        // touch when moved onto the async-compute queue this frame -- the 5 Surface Cache atlas
        // images GIInject reads/writes (Albedo/Normal/Emissive/Radiance/WorldPos -- confirmed by
        // reading SurfaceCacheGIInjectPass::Init's own descriptor writes; NOT DirectLighting, which
        // GIInject never binds) plus 4 buffers RecordRefreshTLAS/GIInject's own HWRT trace touch
        // (m_SurfaceCache's Fallback Mesh vertex/index buffers, m_SurfaceCacheRT's draw-range
        // buffer, and the TLAS's own backing buffer -- VkAccelerationStructureKHR itself is not a
        // barrier-typed resource, see SurfaceCacheRayTracingPass::GetTLASBufferHandle()'s own
        // comment). Mirrors GpuGeometryPagePool::ReleasePhysicalPoolOwnership/
        // AcquirePhysicalPoolOwnership's existing idiom exactly: per the Vulkan spec, only
        // srcStageMask/srcAccessMask are meaningful on the release side (the acquiring queue's own
        // barrier is what actually makes data visible to it) and only dstStageMask/dstAccessMask
        // are meaningful on the acquire side -- `activeStageMask`/`activeImageAccessMask`/
        // `activeBufferAccessMask` are therefore applied to src* when `isRelease` and to dst*
        // otherwise, with the other triple left NONE. Whole-resource scope (not sub-ranged), same
        // "trades a small amount of granularity for a much simpler, easier-to-verify-correct
        // synchronization story" reasoning GpuGeometryPagePool's own class comment documents for
        // its own whole-buffer release/acquire. No-op (records nothing) when
        // m_AsyncComputeAvailableThisBuild is false (same queue family already, no transfer needed
        // -- mirrors GpuGeometryPagePool::m_NeedsOwnershipTransfer's own skip pattern).
        //
        // KNOWN, DOCUMENTED SIMPLIFICATION: SurfaceCacheRayTracingPass's own internal
        // TlasRefitResources (instanceBuffer/scratchBuffer -- private scratch state consumed ENTIRELY
        // within RecordRefreshTLAS's own call, never read by any other pass on either queue) are
        // deliberately NOT included in this transfer. A CPU-mapped instance buffer's host writes are
        // not queue-family-scoped, so the only real risk is the GPU-side scratch buffer's ownership
        // history if SetDebugAsyncComputeEnabled() is toggled mid-session (a rare, Debug-only
        // scenario) -- accepted here rather than adding a second whole-resource-group transfer for
        // two purely-internal scratch buffers with no cross-pass consumer.
        void RecordSurfaceCacheOwnershipTransfer(VkCommandBuffer cmd, bool isRelease,
            uint32_t srcFamily, uint32_t dstFamily, VkPipelineStageFlags2 activeStageMask,
            VkAccessFlags2 activeImageAccessMask, VkAccessFlags2 activeBufferAccessMask);

#ifndef NDEBUG
        // Phase 1 (Nanite advanced): C++ port of HermiteEvaluate/BuildBendFrame/ApplySplineDeformation
        // (spline_deformation.glsl), called once at the end of Init() -- densely samples the
        // authored curve's parameter range AND, at each step, several points around the Tube's own
        // cross-section radius (see geom_tube.comp's outer radius constant), computing the TRUE
        // worst-case displacement between a straight-pipe vertex and its bent counterpart. LOG_ERRORs
        // if that true maximum ever exceeds SPLINE_MAX_DEVIATION -- the exact invariant this engine's
        // bounds-inflation contract (ClusterDAGScreenError.comp/ClusterLODCompact.comp) depends on
        // never being violated, mirrors geometry::ClusterDAG.cpp's own ValidateClusterDAG convention
        // (log-and-continue, not a hard crash -- this is a diagnostic sanity check, not a recoverable-
        // error path). Debug-only per CLAUDE.md's build-separation rule: this function's own file-
        // scope existence is guarded out of Release entirely, not merely skipped at runtime.
        void ValidateSplineBounds() const;

        // Maps config::debugview::SELECTED_BUFFER_INDEX to one of this pipeline's own GBuffer/GI
        // intermediate views (see the big index->buffer table in this method's own .cpp
        // definition) and dispatches m_DebugBufferView with it -- backs the ImGui "Buffer Viewer"
        // dropdown. Called at most once per frame, only when a buffer is actually selected (index
        // != 0/"Off") -- see RecordFrame()'s own call site for exactly where.
        void RecordDebugBufferView(VkCommandBuffer cmd);
#endif

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

        // Phase 1 (Nanite advanced): this scene's one authored Hermite bend curve (4x
        // SplineControlPoint, 128 bytes, std430) -- uploaded once at Init() (one-shot staged upload,
        // see Init()'s own comment, mirroring VulkanContext::UploadEntityData's pattern) into a
        // GPU_ONLY SSBO bound read-only by all 5 downstream vertex/compute shaders that decode
        // entity 6 (Tube)'s clusters (see spline_deformation.glsl's own header comment for why this
        // is applied in local space before rotation).
        GpuBuffer m_SplineControlPointsBuffer;

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

        // Phase 2 (Lumen advanced roadmap): the graphics queue family index (createInfo.
        // graphicsQueueFamilyIndex) -- needed as the src/dstQueueFamilyIndex counterpart in every
        // RecordSurfaceCacheOwnershipTransfer() call. Not previously stored by this class (every
        // other pre-Phase-2 consumer only ever needed createInfo.queue itself, not its family
        // index) -- see m_AsyncComputeQueueFamilyIndex's own sibling comment below.
        uint32_t m_GraphicsQueueFamilyIndex = 0;

        // Phase 2 (Lumen advanced roadmap): async-compute queue handles, copied from
        // ClusterRenderPipelineCreateInfo at Init() time (borrowed, owned by VulkanContext, same
        // convention as every other queue/family field this class stores). `queue`==VK_NULL_HANDLE
        // until Init() runs.
        VkQueue m_AsyncComputeQueue = VK_NULL_HANDLE;
        uint32_t m_AsyncComputeQueueFamilyIndex = 0;
        // Pure HARDWARE capability, copied once from createInfo.hasDedicatedAsyncComputeQueue at
        // Init() time (VulkanContext::HasDedicatedAsyncComputeQueue()) -- true only when this GPU
        // exposed a genuinely distinct async-compute-capable queue family. RecordFrame()'s [1z]
        // block ALWAYS records SurfaceCacheRayTracingPass::RecordRefreshTLAS + the radiosity bounce
        // loop into the separate asyncComputeCmd (never into `cmd` itself) whenever the debug toggle
        // is on and traceMode == HWRT, REGARDLESS of this flag's value -- what this flag actually
        // gates is whether the acquire/release barriers around that hand-off are real queue-family-
        // ownership-transfer barriers or degenerate to a no-op (mirrors GpuGeometryPagePool::
        // m_NeedsOwnershipTransfer's own skip pattern: same-family data needs no ownership transfer,
        // only the barrier's usual execution/memory dependency, which still fires either way).
        bool m_AsyncComputeAvailableThisBuild = false;

        // Phase 2 (Lumen advanced roadmap) fix, 2026-07-17: this class used to own a persistent
        // `m_AtlasOwnedByAsync` bool tracking Surface Cache atlas/TLAS ownership ACROSS the frame
        // boundary (release late in frame N, matching acquire at the top of frame N+1) -- a
        // deliberate one-frame pipeline lag, forced by the old single-submission design (see this
        // class' own header comment for the full root-cause explanation). Now that RecordFrameEarly
        // releases and RecordFrameLate re-acquires WITHIN THE SAME FRAME (the async-compute queue's
        // work runs concurrently with RecordFrameMid's own GPU work in between, so the hand-off no
        // longer needs to span a frame boundary to get real overlap), the release/acquire pair is
        // fully determined by that SAME frame's own m_FrameScratch.useAsyncCompute -- there is
        // nothing left to track across frames, so that member was removed entirely.
        //
        // Per-frame scratch state, written ONCE by RecordFrameEarly() and read-only for the rest of
        // that same frame's RecordAsyncCompute()/RecordFrameMid()/RecordFrameLate() calls -- exists
        // only because splitting one function into 4 calls (see this class' own header comment) lost
        // the ability to share function-scope locals directly; every field here is exactly a local
        // variable the old single RecordFrame() would have declared at its own top. Never read
        // before RecordFrameEarly() has run this frame (asserted there).
        struct FrameScratch {
            CameraPushConstants camera{};              // Original, UNjittered (RecordFrameMid's own HW raster draws use this one, not cameraCopy).
            CameraPushConstants cameraCopy{};           // Jittered (TAA/TSR) -- every other consumer.
            maths::vec3 cameraPositionWorld{};
            CameraFrameInfo cameraFrameInfo{};
            float globalTimeSeconds = 0.0f;
            const core::EntityTransformCPU* entityTransformsCPU = nullptr;

            maths::mat4 viewProj{};
            maths::mat4 invViewProj{};
            ClusterCullViewParams viewParams{};
            float projScaleY = 0.0f;
            float jitterX = 0.0f;
            float jitterY = 0.0f;
            bool taatsrEnabled = true;

            uint32_t traceMode = 1u;
            // This frame's async-compute routing decision -- see the old RecordFrame()'s own
            // `useAsyncCompute` local for the full derivation (traceMode==HWRT AND the debug toggle).
            // Read by RecordAsyncCompute() (skip everything if false) and RecordFrameLate() (skip the
            // ACQUIRE if false, since RecordFrameEarly never RELEASEd in that case either).
            bool useAsyncCompute = false;
            bool radiosityEnabled = true;
        };
        FrameScratch m_FrameScratch;

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

        // Generalized Nanite Tessellation (renderer::TessellationPass, generalized from the earlier
        // Phase 7a single-hardcoded-entity "HeroTessellationPass" -- see that class' own class
        // comment): forward-rendered, screen-space-adaptively-tessellated, procedurally-displaced
        // pass for every core::EntityFlags::IsTessellated entity (VulkanContext::
        // kTessellatedEntityIndices, culled out of the opaque Nanite path entirely via
        // core::EntityFlags::IsTransparent -- see VulkanContext::BuildEntityData()'s own comment)
        // -- lit exactly like m_TransparentForward above (direct+shadowed sun, MegaLights RIS point
        // lights, m_WorldProbes' indirect diffuse, an optional single-sample front-layer specular
        // reflection), but fully OPAQUE and depth-WRITING (unlike glass). Recorded right BEFORE
        // m_TransparentForward's own draw -- specifically so that pass' own read-only depth test
        // correctly occludes glass/translucent entities against every tessellated entity's real
        // (displaced) surface, see TessellationPass's own class comment for the resulting
        // barrier-scope consequence.
        TessellationPass m_Tessellation;

        // Phase 7c (UE5.8 parity roadmap, water/erosion): forward-rendered water plane (materialID
        // kWaterMaterialID, culled out of the opaque Nanite path via core::EntityFlags::
        // IsTransparent, same mechanism as m_Tessellation above -- see VulkanContext::
        // BuildEntityData()'s own comment). Recorded LAST among the forward passes (after
        // m_TransparentForward) -- see WaterForwardPass's own class comment for why: it blits the
        // ALREADY fully-composited frame (opaque + GI + glass + tessellated entities) into its own
        // private background-snapshot image for its refraction term, so anything drawn after it
        // would be invisible to that refraction (and anything drawn before it that skipped this
        // ordering would show through unrealistically, since water is meant to be the top-most
        // surface).
        WaterForwardPass m_WaterForward;

        // GPU-driven particle system (Niagara-style), particle_system_integration_plan.md (project
        // root), all 6 subtasks now integrated -- see renderer::ParticleSystemPass's own class
        // comment. RecordSimulate/RecordSort run in RecordFrameEarly (right after m_GlobalSDF's own
        // update, see that call site's own comment); RecordDraw runs in RecordFrameLate's [13c]
        // forward block, right after m_WaterForward (matching the plan doc's own "after opaque
        // Nanite + transparent, before post-process" ordering requirement). Declared after
        // m_WaterForward (not interleaved with the GI infrastructure below) for the same reason.
        // Always initialized (not Debug-only), same build-separation rule as m_TransparentForward
        // above.
        ParticleSystemPass m_ParticleSystem;

        // GPU-instanced procedural vegetation scatter (UE5.8 rendering-parity gap G2): grass/shrub/
        // rock instances scattered across the terrain -- see renderer::VegetationScatterPass's own
        // class comment for why this follows the particle-system instancing template rather than the
        // per-entity Nanite pattern the just-merged ProceduralTreePass uses. RecordCull/RecordDraw run
        // in RecordFrameLate's [13c] forward block, right after m_Tessellation (both are opaque,
        // depth-writing forward passes) and before m_TransparentForward, so glass depth-tests against
        // the scatter and water snapshots a frame that includes it. Always initialized (not Debug-
        // only), same build-separation rule as m_ParticleSystem above.
        VegetationScatterPass m_VegetationScatter;
        // Subtask 6: this pass' own frame-to-frame delta-time tracking, computed independently from
        // RecordFrameLate's own `deltaTimeSeconds` (that one isn't computed yet by the time
        // RecordFrameEarly reaches m_ParticleSystem.RecordSimulate() -- see that call site's own
        // comment) but using the identical clamp-against-stalls formula.
        float m_LastParticleFrameTimeSeconds = 0.0f;
        bool m_HasLastParticleFrameTime = false;
        // Multi-emitter roadmap (subtask A1): one fractional spawn-count carry-over per emitter slot
        // (was a single float pre-A1) so each emitter's own config::particles::EMITTERS[i].spawnRate
        // stays exact over time regardless of framerate, independently of every other emitter (e.g.
        // one emitter at 200/s and another at 40/s each round correctly on their own schedule, never
        // silently rounding a fractional request down to 0). The rivers/waterfalls feature's mist
        // emitter (EMITTERS[3]) rides this same per-slot array, not a separate accumulator.
        float m_ParticleSpawnAccumulator[ParticleSystemPass::kMaxEmitters] = {};
        // Precipitation feature (rain/snow tied to the Atmos climate simulation) -- identical
        // fractional-carry-over role as m_ParticleSpawnAccumulator above, just against
        // config::atmos::PRECIPITATION_INTENSITY * PRECIPITATION_MAX_SPAWN_RATE_PER_SECOND instead
        // of any embers emitter's own fixed spawnRate. Kept as a separate accumulator (not folded
        // into the m_ParticleSpawnAccumulator array above) since precipitation is not one of the
        // config::particles::EMITTERS[] slots -- it has its own dedicated camera-relative spawn-shell
        // and kind-resolution logic (see RecordSimulate's own precip* parameters).
        float m_PrecipSpawnAccumulator = 0.0f;

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
        // Atmos weather system, Subtask 1 (atmos_integration_plan.md): climate/wind state producer
        // -- see AtmosClimatePass's own class comment. Declared here (not consumed by anything yet)
        // so its RecordUpdate() call can run right at the top of RecordFrame()'s own [1z] GI-
        // infrastructure block, ahead of every future consumer (Froxel Fog / Volumetric Clouds,
        // Subtasks 3-4) that will eventually read its AtmosGlobalsUBO the same frame.
        AtmosClimatePass m_AtmosClimate;
        // Atmos weather system, Subtask 2: Physically Based Sky (Hillaire LUTs) -- see AtmosSkyPass's
        // own class comment. RecordUpdate() runs right after m_AtmosClimate's own, before anything
        // that samples GetSkyViewLUTView() this same frame (m_PostProcess, and Debug-only
        // m_SDFRayMarch).
        AtmosSkyPass m_AtmosSky;
        // Sun direction is fixed by default; one point light is authored in Init() specifically to
        // exercise/verify Phase 3's point-light Virtual Shadow Maps (see Init()'s own comment) --
        // see renderer::LightingTypes.h's own comment for the full field-by-field default.
        SceneLights m_SceneLights;
        // Dynamic Weather Simulation's seasonal cycle (AtmosClimatePass, see its own
        // GetSeasonalSunElevationOffsetRadians() comment): the FIXED base sun elevation/azimuth
        // Init() authors into m_SceneLights.sun.direction, above, kept separately so each frame can
        // recompute the seasonally-offset direction from this same unchanging base instead of
        // compounding an offset onto an already-offset value.
        maths::vec3 m_BaseSunDirection;

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
        // Atmos weather system, Subtask 3: Froxel Volumetric Fog -- see AtmosVolumetricFogPass's own
        // class comment. Init'd after m_MegaLights (its own last dependency to become ready: also
        // needs m_AtmosClimate and m_VirtualShadowMap, both already Init'd much earlier).
        AtmosVolumetricFogPass m_AtmosFog;
        // Atmos weather system, Subtask 4: Procedural Volumetric Clouds -- see AtmosCloudsPass's own
        // class comment. Only needs m_AtmosClimate (already Init'd much earlier); Init'd here purely
        // for proximity to the other Atmos passes, not because of any real dependency ordering.
        AtmosCloudsPass m_AtmosClouds;

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
        // See SetDebugAsyncComputeEnabled()'s own comment: defaults to true (Release-always-on
        // equivalent local, no toggle exists there -- matches m_DebugReflectionsEnabled's own
        // convention), so the async-compute path is attempted by default and must be explicitly
        // disabled to fall back to the fully-serialized graphics-queue path.
        bool m_DebugAsyncComputeEnabled = true;
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
        // Phase 1 (Nanite advanced): see SetDebugEnhancedDisplacementEnabled/
        // SetDebugSplineDeformationEnabled's own comments -- both default to true (Release-always-on
        // equivalent, no toggle exists there, matching m_DebugReflectionsEnabled's own convention).
        bool m_DebugEnhancedDisplacementEnabled = true;
        bool m_DebugSplineDeformationEnabled = true;

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

        // Backs the ImGui "Buffer Viewer" dropdown -- see debug::DebugBufferViewPass's own class
        // comment. Only dispatched (and only ever blitted to the swapchain in place of
        // m_PostProcess's output) when config::debugview::SELECTED_BUFFER_INDEX != 0.
        debug::DebugBufferViewPass m_DebugBufferView;
#endif
    };

}
