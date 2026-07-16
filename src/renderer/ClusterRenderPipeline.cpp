#include "renderer/ClusterRenderPipeline.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h"
#include "io/CacheFileManager.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of the WPOGlobalsUBO block declared identically in ClusterRaster.vert
        // and ClusterSoftwareRaster.comp (std140): a single float naturally needs no padding to reach
        // a 16-byte size, but the trailing pad floats are declared explicitly anyway (matching this
        // codebase's own convention, e.g. ClusterSoftwareRasterPass.cpp's SoftwareRasterViewParams)
        // so the CPU-side struct's sizeof() stays an honest, self-documenting match for the UBO's
        // actual GPU size.
        struct WPOGlobalsUBO {
            float globalTime = 0.0f;
            float _pad0 = 0.0f;
            float _pad1 = 0.0f;
            float _pad2 = 0.0f;
        };
        static_assert(sizeof(WPOGlobalsUBO) == 16,
            "WPOGlobalsUBO must match the WPOGlobalsUBO block in ClusterRaster.vert / ClusterSoftwareRaster.comp exactly (std140 layout)");

    } // namespace

bool ClusterRenderPipeline::Init(
    const ClusterRenderPipelineCreateInfo &createInfo) {
  Shutdown();

  m_Device = createInfo.device;
  m_RenderExtent = createInfo.renderExtent;
  m_VisBufferClusterIDImage = createInfo.visBufferClusterIDImage;
  m_VisBufferClusterIDView = createInfo.visBufferClusterIDView;
  m_VisBufferTriangleIDImage = createInfo.visBufferTriangleIDImage;
  m_VisBufferTriangleIDView = createInfo.visBufferTriangleIDView;
  m_DepthImage = createInfo.depthImage;
  m_DepthImageView = createInfo.depthImageView;

  // =========================================================================================
  // STEP 1 -- Read the consolidated .cache file's header and both tables (CPU
  // side).
  // =========================================================================================
  geometry::CacheFileManager cacheManager;

  geometry::CacheFileHeader header{};
  if (!cacheManager.ReadHeader(createInfo.cacheFilePath, header)) {
    LOG_ERROR(
        std::format("[ClusterRenderPipeline] Failed to read cache header '{}'.",
                    createInfo.cacheFilePath.string()));
    return false;
  }

  std::vector<geometry::ClusterIndexEntry> indexEntries;
  std::vector<geometry::DAGNodeEntry> dagEntries;
  if (!cacheManager.ReadClusterIndexTable(createInfo.cacheFilePath, header,
                                          indexEntries) ||
      !cacheManager.ReadDAGTable(createInfo.cacheFilePath, header,
                                 dagEntries) ||
      indexEntries.empty() || indexEntries.size() != dagEntries.size()) {
    LOG_ERROR(
        "[ClusterRenderPipeline] Failed to read cache cluster/DAG tables.");
    return false;
  }

  uint32_t totalClusterCount = static_cast<uint32_t>(indexEntries.size());
  LOG_INFO(
      std::format("[ClusterRenderPipeline] Cache tables read: {} clusters.",
                  totalClusterCount));

  // =========================================================================================
  // STEP 2 -- Streaming: read the whole geometry section into one host-visible
  // staging buffer. Startup-only bulk load through the exact same page-granular
  // path a per-frame streamer would use (BindPage per 4 KB page below) -- see
  // the class comment's streaming scope note for why the per-frame async
  // residency loop is not active yet.
  // =========================================================================================
  VkDeviceSize geometrySectionSize =
      header.totalFileSizeBytes - header.geometryDataBaseOffset;

  GpuBuffer stagingBuffer;
  stagingBuffer.Create(createInfo.allocator, geometrySectionSize,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);

  {
    std::ifstream file(createInfo.cacheFilePath, std::ios::binary);
    if (!file.is_open()) {
      LOG_ERROR("[ClusterRenderPipeline] Failed to open cache file for the "
                "geometry section read.");
      return false;
    }
    file.seekg(static_cast<std::streamoff>(header.geometryDataBaseOffset));
    file.read(static_cast<char *>(stagingBuffer.MappedData()),
              static_cast<std::streamsize>(geometrySectionSize));
    if (!file) {
      LOG_ERROR(
          "[ClusterRenderPipeline] Short read on the cache geometry section.");
      return false;
    }
  }

  // =========================================================================================
  // STEP 3 -- Initialize the GPU-side stages that the setup command buffer
  // below records into.
  // =========================================================================================
  // Logical address space spans the whole file (virtualAddress is a file
  // offset); physical capacity is exactly one page per cluster -- the whole
  // scene stays resident (see the class comment), so no eviction can ever
  // occur.
  uint32_t maxLogicalPages = static_cast<uint32_t>(header.totalFileSizeBytes /
                                                   geometry::kPageSizeBytes);
  m_PagePool.Init(createInfo.allocator, maxLogicalPages, totalClusterCount);
  m_Decompression.Init(createInfo.device, createInfo.allocator,
                       totalClusterCount, m_PagePool.GetPhysicalPoolBuffer());

  m_HZB.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
             createInfo.queue, createInfo.depthImageView,
             createInfo.renderExtent);

  // Candidate set = DAG leaves only (full detail) -- see the class comment's
  // LOD scope note.
  uint32_t leafCount = 0;
  for (const geometry::DAGNodeEntry &node : dagEntries) {
    if (node.level == 0) {
      ++leafCount;
    }
  }
  if (leafCount == 0) {
    LOG_ERROR("[ClusterRenderPipeline] Cache contains no leaf clusters.");
    return false;
  }

  // Per-frame GPU-driven LOD DAG cut, replacing the old CPU-baked "always DAG
  // level 0" candidate list -- see ClusterLODSelectionPass's own class
  // comment. Needs only the page table's buffer HANDLE (not yet-populated
  // content -- STEP 4 below binds every page after this call) and the full
  // index/DAG tables read in STEP 1.
  m_LODSelection.Init(createInfo.device, createInfo.allocator,
                      createInfo.commandPool, createInfo.queue,
                      m_PagePool.GetPageTableBuffer(), leafCount, indexEntries,
                      dagEntries);

  // Wires the async streaming stack for real -- see GeometryStreamingCoordinator's own class
  // comment. Needs only the cache file path (re-opened for unbuffered/overlapped reads,
  // independent of the STEP 2/4 bulk-load path above, which already closed its own plain
  // std::ifstream by this point) and its own copy of indexEntries (clusterID -> virtualAddress).
  m_Streaming.Init(createInfo.device, createInfo.allocator,
                   createInfo.cacheFilePath, indexEntries);

  m_OcclusionCulling.Init(createInfo.device, createInfo.allocator, leafCount,
                          totalClusterCount,
                          m_LODSelection.GetCandidateMetadataBuffer(),
                          m_LODSelection.GetCandidateCountBuffer(),
                          m_HZB.GetFullView(), m_HZB.GetMipExtent(0),
                          m_HZB.GetMipLevelCount());

  // =========================================================================================
  // STEP 4 -- One setup command buffer: page-table clear, then for EVERY
  // cluster (all DAG levels, so a future GPU LOD cut finds its coarser levels
  // already resident) one 4 KB page bind (staging -> physical pool) + vertex
  // decompression + index expansion, then the persisted-visibility clear. One
  // blocking submit, at startup only -- the per-frame path never submits or
  // waits mid-frame.
  // =========================================================================================
  {
    VkCommandBufferAllocateInfo cmdAllocInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAllocInfo.commandPool = createInfo.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    m_PagePool.ClearPageTable(cmd);

    for (const geometry::ClusterIndexEntry &entry : indexEntries) {
      bool bound = m_PagePool.BindPage(
          cmd, entry.virtualAddress, stagingBuffer.Handle(),
          entry.virtualAddress - header.geometryDataBaseOffset,
          geometry::kPageSizeBytes);
      if (!bound) {
        LOG_ERROR(std::format(
            "[ClusterRenderPipeline] BindPage failed for cluster {}.",
            entry.clusterID));
        vkEndCommandBuffer(cmd);
        vkFreeCommandBuffers(m_Device, createInfo.commandPool, 1, &cmd);
        return false;
      }

      // The logical->physical mapping is CPU-side bookkeeping, valid the moment
      // BindPage() records -- so the matching decompression dispatch can target
      // the exact physical slot immediately, in the same command buffer.
      uint32_t physicalPage =
          m_PagePool.GetPhysicalPageIndex(entry.virtualAddress);
      m_Decompression.DecompressPage(
          cmd, physicalPage,
          maths::vec3{entry.boundsMin[0], entry.boundsMin[1],
                      entry.boundsMin[2]},
          maths::vec3{entry.boundsMax[0], entry.boundsMax[1],
                      entry.boundsMax[2]});
    }

    m_OcclusionCulling.ClearPersistedVisibility(cmd);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(createInfo.queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(createInfo.queue));
    vkFreeCommandBuffers(m_Device, createInfo.commandPool, 1, &cmd);
  }

  stagingBuffer.Destroy();
  LOG_INFO(
      "[ClusterRenderPipeline] All cluster pages streamed + decompressed.");

  // Candidate metadata is now a per-frame GPU-computed LOD cut (m_LODSelection,
  // wired above) instead of a CPU-baked upload -- m_ClusterCount is repurposed
  // as an upper bound (the DAG's total leaf count), see GetClusterCount()'s doc.
  m_ClusterCount = leafCount;

  // =========================================================================================
  // STEP 6 -- Initialize the hybrid rasterization + resolve stages against the
  // buffers the culling pass owns.
  // =========================================================================================
  // WPOGlobalsUBO: 16 bytes, std140 (float globalTime + 3 pad floats) -- uploaded once per frame
  // in RecordFrame(), read by both raster passes' vertex/compute shaders (wpo_deformation.glsl).
  m_WPOGlobalsBuffer.Create(createInfo.allocator, sizeof(WPOGlobalsUBO),
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  // Procedurally generates the bindless cutout mask array once, before any consumer pass below
  // is initialized (each one binds GetMaskImageInfos() into its own descriptor set).
  m_MaskGenerator.Init(createInfo.device, createInfo.allocator,
                       createInfo.commandPool, createInfo.queue);

  std::array<VkFormat, 2> visBufferFormats{createInfo.visBufferFormat,
                                           createInfo.visBufferFormat};
  m_HardwareRaster.Init(createInfo.device,
                        m_OcclusionCulling.GetClusterMetadataBuffer(),
                        m_PagePool.GetPhysicalPoolBuffer(),
                        m_WPOGlobalsBuffer.Handle(),
                        m_MaskGenerator.GetMaskImageInfos(), visBufferFormats,
                        createInfo.depthFormat);

  m_SoftwareRaster.Init(createInfo.device, createInfo.allocator,
                        createInfo.commandPool, createInfo.queue,
                        createInfo.renderExtent,
                        m_OcclusionCulling.GetClusterMetadataBuffer(),
                        m_PagePool.GetPhysicalPoolBuffer(),
                        m_OcclusionCulling.GetSoftwareClusterListBuffer(),
                        m_OcclusionCulling.GetSoftwareClusterListOpaqueBuffer(),
                        m_WPOGlobalsBuffer.Handle(),
                        m_MaskGenerator.GetMaskImageInfos());

  m_Resolve.Init(
      createInfo.device, createInfo.allocator, createInfo.commandPool,
      createInfo.queue, createInfo.renderExtent,
      m_OcclusionCulling.GetClusterMetadataBuffer(),
      m_PagePool.GetPhysicalPoolBuffer(), createInfo.visBufferClusterIDView,
      createInfo.visBufferTriangleIDView, createInfo.depthImageView,
      m_SoftwareRaster.GetVisBufferAtomicView(),
      m_MaskGenerator.GetMaskImageInfos());

  // =========================================================================================
  // STEP 7 -- Lumen-style GI infrastructure: sun shadow map, Surface Cache, Global SDF clipmap
  // (see ClusterRenderPipeline.h's own class-comment addendum on why these are unconditional,
  // not Debug-only, unlike the stats/overlay block and m_SDFRayMarch below). Each pass re-reads
  // the same consolidated .cache file independently (this codebase's established "self-contained
  // pass" convention -- see e.g. renderer::ShadowMapPass's own class comment).
  // =========================================================================================
  if (!m_ShadowMap.Init(createInfo.device, createInfo.allocator,
                        createInfo.commandPool, createInfo.queue,
                        createInfo.cacheFilePath)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize ShadowMapPass.");
    return false;
  }
  if (!m_SurfaceCache.Init(createInfo.device, createInfo.allocator,
                           createInfo.commandPool, createInfo.queue,
                           createInfo.cacheFilePath)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize SurfaceCachePass.");
    return false;
  }
  // One-time wiring (see SurfaceCachePass::SetShadowMap's own comment): the shadow map's view/
  // sampler never change again after ShadowMapPass::Init(), so this binding is never refreshed.
  m_SurfaceCache.SetShadowMap(m_ShadowMap.GetShadowMapView(), m_ShadowMap.GetShadowMapSampler());

  if (!m_GlobalSDF.Init(createInfo.device, createInfo.allocator,
                        createInfo.commandPool, createInfo.queue,
                        createInfo.cacheFilePath)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize GlobalSDFPass.");
    return false;
  }

#ifndef NDEBUG
  // Two-tier SDF ray march DEBUG VISUALIZATION (see ClusterRenderPipeline.h's own comment on
  // m_SDFRayMarch) -- output sized to match the render extent so its debug image is a 1:1
  // resolution match for whatever m_Resolve would otherwise have produced.
  if (!m_SDFRayMarch.Init(createInfo.device, createInfo.allocator,
                          createInfo.commandPool, createInfo.queue,
                          createInfo.cacheFilePath, createInfo.renderExtent.width,
                          createInfo.renderExtent.height)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize SDFRayMarchPass.");
    return false;
  }
  // One-time wiring (see SDFRayMarchPass::SetGlobalSDFViews's own comment): the 4 clipmap level
  // views never change again after GlobalSDFPass::Init().
  m_SDFRayMarch.SetGlobalSDFViews(m_GlobalSDF);
#endif

#ifndef NDEBUG
  // Debug-only stat overlay: triangle-count compute pass (reads ONLY m_OcclusionCulling's already-
  // public, stable accessors -- both the masked and opaque list variants, see
  // ClusterTriangleStatsPass's own class comment) + the bitmap-font text renderer drawing directly
  // onto m_Resolve's own output color image.
  m_Allocator = createInfo.allocator;
  m_TriangleStats.Init(createInfo.device, createInfo.allocator,
                       m_OcclusionCulling.GetMaxClusters(),
                       m_OcclusionCulling.GetClusterMetadataBuffer(),
                       m_OcclusionCulling.GetEarlyIndirectCommandBuffer(),
                       m_OcclusionCulling.GetEarlyDrawCountBuffer(),
                       m_OcclusionCulling.GetLateIndirectCommandBuffer(),
                       m_OcclusionCulling.GetLateDrawCountBuffer(),
                       m_OcclusionCulling.GetSoftwareClusterListBuffer(),
                       m_OcclusionCulling.GetEarlyIndirectCommandOpaqueBuffer(),
                       m_OcclusionCulling.GetEarlyDrawCountOpaqueBuffer(),
                       m_OcclusionCulling.GetLateIndirectCommandOpaqueBuffer(),
                       m_OcclusionCulling.GetLateDrawCountOpaqueBuffer(),
                       m_OcclusionCulling.GetSoftwareClusterListOpaqueBuffer());
  m_DebugOverlay.Init(createInfo.device, createInfo.allocator,
                      createInfo.commandPool, createInfo.queue,
                      ClusterResolvePass::kOutputColorFormat);
#endif

  LOG_INFO(
      std::format("[ClusterRenderPipeline] Initialized: {} clusters streamed "
                  "from cache ({} leaf candidates, {} logical pages).",
                  totalClusterCount, m_ClusterCount, maxLogicalPages));
  return true;
}

void ClusterRenderPipeline::Shutdown() {
  // Reverse initialization order; each Shutdown() is null-safe on a
  // never-initialized pass.
#ifndef NDEBUG
  m_DebugOverlay.Shutdown();
  m_TriangleStats.Shutdown();
  m_Allocator = VK_NULL_HANDLE;
  m_LastStatsSampleTime = 0.0f;
  m_LastStatsSampleBytes = 0;
  m_SDFRayMarch.Shutdown();
#endif
  m_GlobalSDF.Shutdown();
  m_SurfaceCache.Shutdown();
  m_ShadowMap.Shutdown();
  m_Resolve.Shutdown();
  m_SoftwareRaster.Shutdown();
  m_HardwareRaster.Shutdown();
  m_MaskGenerator.Shutdown();
  m_OcclusionCulling.Shutdown();
  m_Streaming.Shutdown();
  m_LODSelection.Shutdown();
  m_HZB.Shutdown();
  m_Decompression.Shutdown();
  m_PagePool.Shutdown();
  m_WPOGlobalsBuffer.Destroy();

  m_ClusterCount = 0;
  m_VisBufferClusterIDImage = VK_NULL_HANDLE;
  m_VisBufferClusterIDView = VK_NULL_HANDLE;
  m_VisBufferTriangleIDImage = VK_NULL_HANDLE;
  m_VisBufferTriangleIDView = VK_NULL_HANDLE;
  m_DepthImage = VK_NULL_HANDLE;
  m_DepthImageView = VK_NULL_HANDLE;
  m_RenderExtent = {0, 0};
  m_Device = VK_NULL_HANDLE;
}

void ClusterRenderPipeline::BeginVisBufferRendering(
    VkCommandBuffer cmd, bool clearAttachments) const {
  VkAttachmentLoadOp loadOp = clearAttachments ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                               : VK_ATTACHMENT_LOAD_OP_LOAD;

  VkRenderingAttachmentInfo colorAttachments[2]{};
  colorAttachments[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  colorAttachments[0].imageView = m_VisBufferClusterIDView;
  colorAttachments[0].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachments[0].loadOp = loadOp;
  colorAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  // 0xFFFFFFFF sentinel: matches geometry::kInvalidClusterID and
  // ClusterResolve.comp's "nothing rasterized here" hardware-path check.
  colorAttachments[0].clearValue.color.uint32[0] = 0xFFFFFFFFu;

  colorAttachments[1] = colorAttachments[0];
  colorAttachments[1].imageView = m_VisBufferTriangleIDView;

  VkRenderingAttachmentInfo depthAttachment{
      VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAttachment.imageView = m_DepthImageView;
  depthAttachment.imageLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAttachment.loadOp = loadOp;
  // STORE (unlike the old flat path's DONT_CARE): this depth is consumed twice
  // after each raster pass -- by the HZB rebuilds and by the resolve pass's
  // per-pixel depth arbitration.
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.clearValue.depthStencil = {1.0f, 0};

  VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
  renderingInfo.renderArea = {{0, 0}, m_RenderExtent};
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = 2;
  renderingInfo.pColorAttachments = colorAttachments;
  renderingInfo.pDepthAttachment = &depthAttachment;

  vkCmdBeginRendering(cmd, &renderingInfo);
}

void ClusterRenderPipeline::RecordFrame(VkCommandBuffer cmd,
                                        const CameraPushConstants &camera,
                                        const maths::vec3 &cameraPositionWorld,
                                        const CameraFrameInfo &cameraFrameInfo,
                                        float globalTimeSeconds,
                                        VkImage swapchainImage) {
  assert(m_ClusterCount > 0 && "RecordFrame called before a successful Init");

  // Every stage of this frame consumes the SAME combined matrix -- this is what
  // makes the resolve pass's screen-space triangle re-projection bit-identical
  // to the rasterizers'.
  maths::mat4 viewProj = camera.proj * camera.view;

  ClusterCullViewParams viewParams{};
  viewParams.frustumPlanes = ExtractFrustumPlanes(viewProj);
  viewParams.cameraPositionWorld = cameraPositionWorld;

  // abs(proj.m[5]): PerspectiveVulkan stores -1/tan(fovY/2) there (Y flip), and
  // the screen size classification needs the positive scale.
  float projScaleY = std::abs(camera.proj.m[5]);

  // =========================================================================================
  // [1] Per-frame worklist clears. Each Record*() carries its own
  // CLEAR->COMPUTE barrier, so the culling/raster dispatches below can never
  // read a stale counter.
  // =========================================================================================
  m_OcclusionCulling.RecordClearFrame(cmd);
  m_SoftwareRaster.RecordClear(cmd);
  m_LODSelection.RecordClear(cmd);

  // =========================================================================================
  // [1z] Lumen-style GI infrastructure: sun shadow map -> Surface Cache capture -> Global SDF
  // clipmap streaming -> (Debug-only) SDF ray march debug visualization. Independent of the
  // Nanite VisBuffer/HZB machinery above and below (no shared images, no fence contention -- see
  // ClusterRenderPipeline.h's own class-comment addendum), so it can run anywhere in the frame;
  // placed early, right after the worklist clears, so it overlaps with the GPU's own internal
  // scheduling of the culling/streaming work that follows rather than sitting at the tail end.
  // =========================================================================================
  {
    // Sun direction is fixed for now (m_SceneLights' own default, see LightingTypes.h) -- a
    // future day/night system would rotate it per frame instead.
    const maths::vec3 sunDirection = m_SceneLights.sun.direction;

    // 1. Shadow map first: SurfaceCachePass's shadow lookup (below) needs THIS frame's light
    // view-projection, and ShadowMapPass::RecordCapture's own trailing barrier already makes its
    // depth writes visible to a fragment-shader sampled read (see that method's own comment).
    m_ShadowMap.RecordCapture(cmd, sunDirection);

    // 2. Surface Cache: feed this frame's light data (the light view-proj THIS frame's shadow map
    // was just rendered with, not a stale one) before the visibility-driven capture draws.
    m_SurfaceCache.UpdateLighting(m_SceneLights, m_ShadowMap.GetLightViewProj());
    m_SurfaceCache.UpdateVisibility(cameraFrameInfo.position, cameraFrameInfo.forward,
                                    maths::vec3{0.0f, 1.0f, 0.0f}, cameraFrameInfo.fovYRadians,
                                    cameraFrameInfo.aspectRatio, cameraFrameInfo.nearZ,
                                    cameraFrameInfo.farZ);
    m_SurfaceCache.RecordCapture(cmd);

    // 3. Global SDF clipmap streaming, from this frame's camera position.
    m_GlobalSDF.RecordUpdate(cmd, cameraFrameInfo.position);

#ifndef NDEBUG
    // GlobalSDFPass::RecordUpdate's own trailing barrier only extends visibility to
    // SHADER_STORAGE_READ/WRITE (its own compositing dispatches read/write the clipmap via
    // imageLoad/imageStore) -- SDFRayMarchPass instead samples it through a COMBINED IMAGE
    // SAMPLER (SetGlobalSDFViews), which needs SHADER_SAMPLED_READ, a distinct access flag the
    // barrier above does not cover. Same "STORAGE_WRITE -> SAMPLED_READ" extension this class's
    // own HZB rebuilds already need (see the barriers around m_HZB.Generate() below).
    VkMemoryBarrier2 globalSDFToSampledBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    globalSDFToSampledBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    globalSDFToSampledBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    globalSDFToSampledBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    globalSDFToSampledBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo globalSDFToSampledDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    globalSDFToSampledDep.memoryBarrierCount = 1;
    globalSDFToSampledDep.pMemoryBarriers = &globalSDFToSampledBarrier;
    vkCmdPipelineBarrier2(cmd, &globalSDFToSampledDep);

    // 4. Two-tier SDF ray march DEBUG VISUALIZATION -- only its OUTPUT IMAGE is ever displayed
    // (via the DEBUG_VIEW_LUMEN blit-swap below), but it is always recorded so the debug view
    // shows this frame's state the instant it's selected, with no one-frame lag.
    m_SDFRayMarch.RecordRayMarch(cmd, m_GlobalSDF, cameraFrameInfo.position, cameraFrameInfo.forward,
                                 maths::vec3{0.0f, 1.0f, 0.0f}, cameraFrameInfo.fovYRadians,
                                 cameraFrameInfo.aspectRatio, cameraFrameInfo.nearZ, cameraFrameInfo.farZ);
#endif
  }

  // =========================================================================================
  // [1a] Async streaming triage: read back LAST frame's residency misses (captured by this
  // frame's own [1b] readback below, one frame ago), submit new disk reads, and bind/decompress
  // whatever completed since last frame -- see GeometryStreamingCoordinator's own class comment.
  // Recorded before the LOD cut below so any page bound this frame is already resident by the
  // time this frame's own residency checks (ClusterLODResidencyFallback.comp/ClusterLODCompact
  // .comp) run.
  // =========================================================================================
  m_Streaming.ProcessFeedbackAndDrainCompletions(cmd, m_LODSelection.GetFeedbackBuffer(),
                                                 m_PagePool, m_Decompression);

  // =========================================================================================
  // [1b] Per-frame GPU-driven LOD DAG cut: evaluate every DAG node's screen-space error against
  // the current view, apply the parent-fallback residency policy, and stream-compact the result
  // into this frame's candidate ClusterCullMetadata list -- see ClusterLODSelectionPass's own
  // class comment. Must run before m_OcclusionCulling's early pass, which now consumes its
  // GPU-only candidate count/buffer instead of a CPU-known cluster list.
  // =========================================================================================
  {
    ClusterLODSelectionPass::ViewParams lodViewParams{};
    lodViewParams.view = camera.view;
    lodViewParams.proj = camera.proj;
    lodViewParams.pixelErrorThreshold = kLODPixelErrorThreshold;
    // 2*atan(1/projScaleY) recovers fovYRadians from the projection matrix's own Y-scale term
    // (see projScaleY's own comment below) -- computed here, ahead of projScaleY's declaration,
    // since ExtractFrustumPlanes/viewProj aren't needed for this derivation.
    float earlyProjScaleY = std::abs(camera.proj.m[5]);
    lodViewParams.fovYRadians = 2.0f * std::atan(1.0f / earlyProjScaleY);
    lodViewParams.viewportHeight = static_cast<float>(m_RenderExtent.height);
    lodViewParams.aspectRatio = static_cast<float>(m_RenderExtent.width) / static_cast<float>(m_RenderExtent.height);

    m_LODSelection.RecordEvaluateAndCompact(cmd, lodViewParams);

    // Captures THIS frame's residency-miss reports (ClusterLODResidencyFallback.comp, just
    // dispatched above) into the feedback buffer's host-visible readback half, for [1a]'s
    // ProcessFeedbackAndDrainCompletions() to consume next frame -- see FeedbackBuffer::
    // RecordReadback()'s own doc comment for why "next frame" (not this one) is the earliest safe
    // point to read it back on the CPU.
    m_LODSelection.GetFeedbackBuffer().RecordReadback(cmd);

    m_LODSelection.RecordBuildEarlyDispatchArgs(cmd);
  }

  // =========================================================================================
  // [1b] Upload this frame's WPOGlobalsUBO (globalTime + padding) -- read by both raster passes'
  // WPO sway function (wpo_deformation.glsl). Uploaded once here, well before either raster pass
  // records its draw/dispatch, so a single barrier below covers both consumers.
  // =========================================================================================
  {
    WPOGlobalsUBO wpoGlobals{};
    wpoGlobals.globalTime = globalTimeSeconds;
    vkCmdUpdateBuffer(cmd, m_WPOGlobalsBuffer.Handle(), 0, sizeof(WPOGlobalsUBO), &wpoGlobals);

    VkMemoryBarrier2 wpoBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    wpoBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    wpoBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    wpoBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    wpoBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;

    VkDependencyInfo wpoDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    wpoDepInfo.memoryBarrierCount = 1;
    wpoDepInfo.pMemoryBarriers = &wpoBarrier;
    vkCmdPipelineBarrier2(cmd, &wpoDepInfo);
  }

  // =========================================================================================
  // [2] EARLY cull: every leaf candidate vs frustum/backface + LAST frame's HZB
  // (rebuilt at the end of the previous RecordFrame from that frame's complete
  // depth). Its trailing barrier makes the early draw list/count visible to
  // DRAW_INDIRECT and the pending + software lists visible to later COMPUTE.
  // =========================================================================================
  m_OcclusionCulling.RecordEarlyPass(cmd, viewParams, viewProj, projScaleY,
                                     m_LODSelection.GetEarlyDispatchArgsBuffer(),
                                     kSoftwareRasterThresholdPixels);

  // =========================================================================================
  // [3] Attachment layout acquisition. All three images are re-acquired with
  // oldLayout = UNDEFINED (content discarded): both raster passes fully
  // re-render every frame, and cross-frame GPU ordering is already guaranteed
  // by the frame fence, so no previous-frame content ever needs preserving
  // here. srcStageMask still names the stages that touched these images at the
  // end of the previous frame (resolve's compute reads and the second HZB
  // rebuild's depth sampling) purely as an execution dependency, keeping the
  // transition correct even if the frame pacing later moves to multiple frames
  // in flight.
  // =========================================================================================
  {
    VkImageMemoryBarrier2 barriers[3]{};

    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].srcAccessMask =
        0; // UNDEFINED discard: nothing to make available.
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = m_VisBufferClusterIDImage;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[1] = barriers[0];
    barriers[1].image = m_VisBufferTriangleIDImage;

    barriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[2].srcAccessMask = 0;
    barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[2].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    barriers[2].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[2].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].image = m_DepthImage;
    barriers[2].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 3;
    depInfo.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

  // =========================================================================================
  // [4] EARLY hardware raster: draws exactly the early cull's survivors via
  // vkCmdDrawIndexedIndirectCount into the VisBuffer + depth (CLEAR load ops). Opaque first (no
  // discard, real hardware early-Z eligible), then masked (mask-sampling discard) -- opaque-first
  // is a perf-only ordering choice (lets the masked pipeline's depth test reject more fragments
  // before running its non-early-Z frag shader); final depth/VisBuffer content is order-independent.
  // =========================================================================================
  BeginVisBufferRendering(cmd, /*clearAttachments=*/true);
  m_HardwareRaster.RecordDraw(
      cmd, camera, m_RenderExtent,
      m_Decompression.GetDecompressedIndexPoolBuffer(),
      m_OcclusionCulling.GetEarlyIndirectCommandOpaqueBuffer(),
      m_OcclusionCulling.GetEarlyDrawCountOpaqueBuffer(), m_ClusterCount, /*opaque=*/true);
  m_HardwareRaster.RecordDraw(
      cmd, camera, m_RenderExtent,
      m_Decompression.GetDecompressedIndexPoolBuffer(),
      m_OcclusionCulling.GetEarlyIndirectCommandBuffer(),
      m_OcclusionCulling.GetEarlyDrawCountBuffer(), m_ClusterCount, /*opaque=*/false);
  vkCmdEndRendering(cmd);

  // =========================================================================================
  // [5] Depth -> sampled-readable for the HZB rebuild: the early pass's depth
  // writes (late fragment tests) must complete and be visible before
  // HZBBuildInit.comp samples the image.
  // =========================================================================================
  {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_DepthImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

  // =========================================================================================
  // [6] Mid-frame HZB rebuild from the early-pass depth. The early cull's own
  // HZB sampling (of last frame's pyramid) is ordered before Generate()'s first
  // imageStore by the execution dependency RecordEarlyPass's trailing barrier
  // already created (srcStage COMPUTE covers all prior compute, dst includes
  // COMPUTE) -- no WAR hazard. Generate()'s internal per-mip barriers end at
  // STORAGE_READ; the extra memory barrier below extends visibility to
  // SAMPLED_READ, which is how the late cull actually reads the pyramid
  // (textureLod through a combined image sampler) -- the exact contract
  // documented on ClusterOcclusionCullingPass step 5.
  // =========================================================================================
  m_HZB.Generate(cmd);
  {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

  // =========================================================================================
  // [7] LATE cull: GPU-sized indirect dispatch over exactly the pending list,
  // against the fresh HZB. RecordLatePass's trailing barrier covers the late
  // draw list for DRAW_INDIRECT; the extra barrier below additionally makes its
  // software-cluster-list writes visible to COMPUTE (the software raster's
  // dispatch-args build + raster reads), which that trailing barrier does not
  // cover.
  // =========================================================================================
  m_OcclusionCulling.RecordBuildLateDispatchArgs(cmd);
  m_OcclusionCulling.RecordLatePass(cmd);
  {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

  // =========================================================================================
  // [8] Re-arm the attachments for the LATE raster: depth returns to attachment
  // layout (contents PRESERVED -- the late draws must depth-test against the
  // early geometry), and both VisBuffer images get a write-after-write
  // dependency between the two rendering passes (no layout change; without it
  // the two passes' color writes are unordered).
  // =========================================================================================
  {
    VkImageMemoryBarrier2 barriers[3]{};

    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = m_VisBufferClusterIDImage;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[1] = barriers[0];
    barriers[1].image = m_VisBufferTriangleIDImage;

    // WAR on the depth image (HZB sampled it in [6]; the late raster writes
    // it): a read- before-write hazard needs only the execution dependency plus
    // the layout transition
    // -- there are no prior writes to make available, hence srcAccessMask = 0.
    barriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[2].srcAccessMask = 0;
    barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[2].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    barriers[2].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barriers[2].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].image = m_DepthImage;
    barriers[2].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 3;
    depInfo.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

  // =========================================================================================
  // [9] LATE hardware raster (loadOp = LOAD) -- draws the disocclusions the
  // early pass could not confirm, on top of the early output. Opaque first, same rationale as
  // the early pass above.
  // =========================================================================================
  BeginVisBufferRendering(cmd, /*clearAttachments=*/false);
  m_HardwareRaster.RecordDraw(cmd, camera, m_RenderExtent,
                              m_Decompression.GetDecompressedIndexPoolBuffer(),
                              m_OcclusionCulling.GetLateIndirectCommandOpaqueBuffer(),
                              m_OcclusionCulling.GetLateDrawCountOpaqueBuffer(),
                              m_ClusterCount, /*opaque=*/true);
  m_HardwareRaster.RecordDraw(cmd, camera, m_RenderExtent,
                              m_Decompression.GetDecompressedIndexPoolBuffer(),
                              m_OcclusionCulling.GetLateIndirectCommandBuffer(),
                              m_OcclusionCulling.GetLateDrawCountBuffer(),
                              m_ClusterCount, /*opaque=*/false);
  vkCmdEndRendering(cmd);

  // =========================================================================================
  // [10] Software raster of every micro-triangle cluster (early- and
  // late-routed entries are both in the list by now). Writes only its own R64
  // atomic image -- fully independent of the hardware attachments, so no
  // ordering against [9] is required beyond what its own internal barriers
  // already record.
  // =========================================================================================
  m_SoftwareRaster.RecordRaster(cmd, viewProj);

  // =========================================================================================
  // [11] Hand the hardware VisBuffer + depth to the resolve pass: color
  // attachments become GENERAL storage images (the layout the resolve
  // descriptors were written with), depth becomes sampled-readable again.
  // =========================================================================================
  {
    VkImageMemoryBarrier2 barriers[3]{};

    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = m_VisBufferClusterIDImage;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[1] = barriers[0];
    barriers[1].image = m_VisBufferTriangleIDImage;

    barriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[2].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[2].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barriers[2].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barriers[2].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barriers[2].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[2].image = m_DepthImage;
    barriers[2].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 3;
    depInfo.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

  // =========================================================================================
  // [12] Resolve: per-pixel hardware-vs-software arbitration + barycentric
  // reconstruction + material evaluation into the resolve pass's own RGBA8
  // output image. The software rasterizer's atomic writes are already visible
  // (RecordRaster's trailing COMPUTE -> COMPUTE barrier).
  // =========================================================================================
  m_Resolve.RecordResolve(cmd, viewProj
#ifndef NDEBUG
    , camera.debugViewMode
#endif
  );

  // =========================================================================================
  // [13] Second HZB rebuild, from the now-complete (early + late) depth: this
  // is the pyramid NEXT frame's early pass tests against. WAR ordering vs the
  // late cull's sampling of the previous pyramid content is transitively
  // guaranteed (multiple COMPUTE -> COMPUTE execution dependencies sit in
  // between), but is restated here explicitly so the dependency does not
  // silently vanish if an intermediate pass's barriers change. Depth is already
  // in DEPTH_STENCIL_READ_ONLY_OPTIMAL from [11], which Generate() accepts.
  // =========================================================================================
  {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask =
        0; // WAR only: the prior accesses were reads, nothing to flush.
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }
  m_HZB.Generate(cmd);
  {
    // Next frame's early cull samples the pyramid; same STORAGE_WRITE ->
    // SAMPLED_READ extension as [6]. Barriers order against subsequent commands
    // in submission order on the same queue, so this correctly covers the next
    // command buffer too.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

#ifndef NDEBUG
  // =========================================================================================
  // [13b] Debug-only stat overlay (whole block compiled out in Release). Timing contract: read
  // back the PREVIOUS frame's triangle-count result first (safe because main.cpp's frame fence
  // already guarantees that frame's GPU work is complete before this RecordFrame() call started
  // recording), THEN clear/dispatch/readback THIS frame's counters for next frame's read -- the
  // same "always one frame behind" contract renderer::FeedbackBuffer already uses.
  // =========================================================================================
  {
    uint32_t hwTriangleCount = 0;
    uint32_t swTriangleCount = 0;
    m_TriangleStats.ReadStats(hwTriangleCount, swTriangleCount);

    m_TriangleStats.RecordClear(cmd);
    m_TriangleStats.RecordCompute(cmd);
    m_TriangleStats.RecordReadback(cmd);

    const VkPhysicalDeviceMemoryProperties* memProps = nullptr;
    vmaGetMemoryProperties(m_Allocator, &memProps);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> heapBudgets{};
    vmaGetHeapBudgets(m_Allocator, heapBudgets.data());
    VkDeviceSize gpuMemUsedBytes = 0;
    for (uint32_t heap = 0; heap < memProps->memoryHeapCount; ++heap) {
      gpuMemUsedBytes += heapBudgets[heap].usage;
    }
    float gpuMemUsedMB = static_cast<float>(gpuMemUsedBytes) / (1024.0f * 1024.0f);

    uint32_t pendingPageLoads = m_Streaming.GetPendingRequestCount() + m_Streaming.GetInFlightReadCount();

    // Bytes/sec: a wall-clock delta against the streamer's running total, not an instantaneous GPU
    // counter -- globalTimeSeconds is main.cpp's glfwGetTime(), monotonically increasing, already
    // threaded through this function for WPO.
    uint64_t totalBytesCompleted = m_Streaming.GetTotalBytesCompleted();
    float deltaTime = globalTimeSeconds - m_LastStatsSampleTime;
    float bytesPerSecond = (deltaTime > 0.0f)
        ? static_cast<float>(totalBytesCompleted - m_LastStatsSampleBytes) / deltaTime
        : 0.0f;
    m_LastStatsSampleTime = globalTimeSeconds;
    m_LastStatsSampleBytes = totalBytesCompleted;

    m_DebugOverlay.BuildFrameText(gpuMemUsedMB, pendingPageLoads, bytesPerSecond, hwTriangleCount, swTriangleCount);
    m_DebugOverlay.RecordDraw(cmd, m_Resolve.GetOutputColorImage(), m_Resolve.GetOutputColorView(), m_RenderExtent);
  }
#endif

  // =========================================================================================
  // [14] Blit the resolved image to the swapchain image and hand it to present.
  // The resolve output stays in GENERAL (valid blit source layout); its own
  // trailing barrier targeted the COPY stage, but vkCmdBlitImage executes in
  // the BLIT stage, so visibility is re-extended here explicitly.
  // =========================================================================================
  // DEBUG_VIEW_LUMEN swaps the blit SOURCE to m_SDFRayMarch's own debug-visualization image
  // instead of m_Resolve's normal lit output -- see ClusterRenderPipeline.h's own comment on why
  // this (not sampling it from within ClusterResolve.comp) is the chosen mechanism. Both images
  // share the exact same format (VK_FORMAT_R8G8B8A8_UNORM) and permanent layout (GENERAL), and
  // both are last written by a compute shader's imageStore, so the SAME barrier below (targeting
  // COMPUTE_SHADER_BIT/SHADER_STORAGE_WRITE_BIT -> BLIT_BIT/TRANSFER_READ_BIT) covers either
  // choice with no changes needed -- only which VkImage is actually blitted differs.
  VkImage blitSourceImage = m_Resolve.GetOutputColorImage();
#ifndef NDEBUG
  if (camera.debugViewMode == DEBUG_VIEW_LUMEN) {
    blitSourceImage = m_SDFRayMarch.GetOutputImage();
  }
#endif
  {
    VkMemoryBarrier2 resolveToBlitBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    resolveToBlitBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    resolveToBlitBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    resolveToBlitBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    resolveToBlitBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;

    VkImageMemoryBarrier2 swapchainToBlitBarrier{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    swapchainToBlitBarrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    swapchainToBlitBarrier.srcAccessMask = 0;
    swapchainToBlitBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    swapchainToBlitBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    swapchainToBlitBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapchainToBlitBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapchainToBlitBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainToBlitBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapchainToBlitBarrier.image = swapchainImage;
    swapchainToBlitBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1,
                                               0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &resolveToBlitBarrier;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &swapchainToBlitBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }

  VkImageBlit blitRegion{};
  blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blitRegion.srcOffsets[1] = {static_cast<int32_t>(m_RenderExtent.width),
                              static_cast<int32_t>(m_RenderExtent.height), 1};
  blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blitRegion.dstOffsets[1] = {static_cast<int32_t>(m_RenderExtent.width),
                              static_cast<int32_t>(m_RenderExtent.height), 1};
  // Same extent both sides -- the "blit" is a 1:1 copy that also performs the
  // RGBA8 -> B8G8R8A8 component reordering the swapchain format requires.
  vkCmdBlitImage(cmd, blitSourceImage, VK_IMAGE_LAYOUT_GENERAL,
                 swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &blitRegion, VK_FILTER_NEAREST);

  {
    VkImageMemoryBarrier2 presentBarrier{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    presentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    presentBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    presentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    presentBarrier.dstAccessMask = 0;
    presentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    presentBarrier.image = swapchainImage;
    presentBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &presentBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  }
}

} // namespace renderer
