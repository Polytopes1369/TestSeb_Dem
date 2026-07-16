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
#include "renderer/VulkanUtils.h"

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
                      m_PagePool.GetPageTableBuffer(),
                      createInfo.entityTransformBuffer, leafCount,
                      indexEntries, dagEntries, createInfo.entityDataBuffer);

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
                          createInfo.entityTransformBuffer,
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
  VulkanUtils::ExecuteOneShotCommands(m_Device, createInfo.commandPool, createInfo.queue, [&](VkCommandBuffer cmd) {
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
        throw std::runtime_error("ClusterRenderPipeline: BindPage failed during initialization");
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
  });

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
                        createInfo.depthFormat,
                        createInfo.entityTransformBuffer,
                        createInfo.entityDataBuffer);

  m_SoftwareRaster.Init(createInfo.device, createInfo.allocator,
                        createInfo.commandPool, createInfo.queue,
                        createInfo.renderExtent,
                        m_OcclusionCulling.GetClusterMetadataBuffer(),
                        m_PagePool.GetPhysicalPoolBuffer(),
                        m_OcclusionCulling.GetSoftwareClusterListBuffer(),
                        m_OcclusionCulling.GetSoftwareClusterListOpaqueBuffer(),
                        m_WPOGlobalsBuffer.Handle(),
                        m_MaskGenerator.GetMaskImageInfos(),
                        createInfo.entityTransformBuffer,
                        createInfo.entityDataBuffer);

  m_Resolve.Init(
      createInfo.device, createInfo.allocator, createInfo.commandPool,
      createInfo.queue, createInfo.renderExtent,
      m_OcclusionCulling.GetClusterMetadataBuffer(),
      m_PagePool.GetPhysicalPoolBuffer(), createInfo.visBufferClusterIDView,
      createInfo.visBufferTriangleIDView, createInfo.depthImageView,
      m_SoftwareRaster.GetVisBufferAtomicView(),
      m_MaskGenerator.GetMaskImageInfos(),
      m_WPOGlobalsBuffer.Handle(),
      createInfo.entityTransformBuffer,
      createInfo.entityDataBuffer,
      createInfo.materialTable.params);

  // Phase 1b: the shading-bin sort pass needs m_Resolve's own 5 output image views (its Classify
  // stage writes background pixels directly into them, see ClusterShadingBinPass's own class
  // comment) -- must run after m_Resolve.Init() above for exactly that reason. m_Resolve's own
  // InitBinnedResolve() then runs AFTER m_ShadingBin.Init(), for the opposite reason (its
  // descriptor set needs m_ShadingBin's sorted-pixel-list/bin-offsets/bin-histogram buffers) --
  // see that method's own comment for why these two calls cannot be collapsed into one ordering.
  m_ShadingBin.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                    createInfo.queue, createInfo.renderExtent,
                    m_OcclusionCulling.GetClusterMetadataBuffer(),
                    createInfo.visBufferClusterIDView, createInfo.visBufferTriangleIDView,
                    createInfo.depthImageView, m_SoftwareRaster.GetVisBufferAtomicView(),
                    m_Resolve.GetOutputColorView(), m_Resolve.GetOutputNormalView(),
                    m_Resolve.GetOutputDepthView(), m_Resolve.GetOutputAlbedoView(),
                    m_Resolve.GetOutputRoughnessMetallicView());
  m_Resolve.InitBinnedResolve(createInfo.device, createInfo.commandPool, createInfo.queue, m_ShadingBin);

  // =========================================================================================
  // STEP 7 -- Lumen-style GI infrastructure: sun+point-light Virtual Shadow Maps (Phase 3), Surface
  // Cache, Global SDF clipmap (see ClusterRenderPipeline.h's own class-comment addendum on why
  // these are unconditional, not Debug-only, unlike the stats/overlay block and m_SDFRayMarch
  // below). Each pass re-reads the same consolidated .cache file independently (this codebase's
  // established "self-contained pass" convention -- see e.g. renderer::VirtualShadowMapPass's own
  // class comment).
  // =========================================================================================
  if (!m_VirtualShadowMap.Init(createInfo.device, createInfo.allocator,
                               createInfo.commandPool, createInfo.queue,
                               createInfo.cacheFilePath)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize VirtualShadowMapPass.");
    return false;
  }
  if (!m_SurfaceCache.Init(createInfo.device, createInfo.allocator,
                           createInfo.commandPool, createInfo.queue,
                           createInfo.cacheFilePath)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize SurfaceCachePass.");
    return false;
  }
  // One-time wiring (see SurfaceCachePass::SetVirtualShadowMap's own comment): none of
  // VirtualShadowMapPass's exposed buffer/image handles are ever recreated after Init(), so this
  // binding is never refreshed again. m_Resolve's own equivalent binding is deferred until after
  // m_Resolve.InitBinnedResolve() above has already run (SetVirtualShadowMap() needs BOTH of
  // m_Resolve's descriptor sets to already exist) -- see the call further below.
  m_SurfaceCache.SetVirtualShadowMap(m_VirtualShadowMap);
  m_Resolve.SetVirtualShadowMap(m_VirtualShadowMap);

  // Forward-rendered translucent/transparent materials (see TransparentForwardPass's own class
  // comment) -- reuses the SAME indexEntries/dagEntries this function loaded above for
  // m_LODSelection.Init(), and the SAME page pool/compressed pool/entity/WPO buffer handles every
  // opaque pass already borrows.
  m_TransparentForward.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                            m_PagePool.GetPageTableBuffer(), m_PagePool.GetPhysicalPoolBuffer(),
                            createInfo.entityTransformBuffer, createInfo.entityDataBuffer, m_WPOGlobalsBuffer.Handle(),
                            createInfo.materialTable.params, indexEntries, dagEntries,
                            ClusterResolvePass::kOutputColorFormat, createInfo.depthFormat);
  m_TransparentForward.SetVirtualShadowMap(m_VirtualShadowMap);

  // Sun orientation: Toronto (lat 43.6532N, lon 79.3832W), July 16, 16:30 local (EDT, UTC-4) --
  // a standard NOAA solar-position computation (equation of time + hour angle + declination for
  // day-of-year 197) gives solar elevation ~45.5 degrees and azimuth ~255.3 degrees (measured
  // clockwise from North -- i.e. WSW, past due-south/solar-noon since 16:30 EDT is mid-afternoon).
  // Axis convention chosen for this scene (no real-world map exists, Y is up): +X = East, -Z =
  // North, so azimuth Az and elevation El convert to the unit vector FROM the scene TOWARD the sun
  // as (cos(El)*sin(Az), sin(El), -cos(El)*cos(Az)); DirectionalLight::direction is the reverse of
  // that (points FROM the light TOWARD the scene, see LightingTypes.h's own convention comment).
  // The lower afternoon elevation (vs. a near-overhead default) is what actually produces long,
  // clearly readable cast shadows instead of short near-vertical ones.
  m_SceneLights.sun.direction = maths::vec3(0.678f, -0.713f, -0.178f).Normalize();
  // Mild warm tint for late-afternoon sunlight (intensity left at its existing default -- a 45
  // degree summer sun is still strong, not golden-hour-grazing).
  m_SceneLights.sun.color = maths::vec3(1.0f, 0.88f, 0.72f);

  // Phase 3 (UE5.8 parity roadmap) verification: SceneLights::pointLights defaults to an empty
  // array (see LightingTypes.h) -- with zero point lights ever authored, Phase 3's point-light
  // Virtual Shadow Maps would be entirely unexercised and unverifiable (the same difficulty this
  // roadmap's own Phase 2 ran into for its reflections). One point light is authored here,
  // positioned above and offset from the "box" entity (materialID 0, world position (-3,0,-3),
  // see VulkanContext::GridSlot()/BuildEntityData()) so its shadow has a genuine chance of falling
  // visibly toward a neighboring grid cell.
  m_SceneLights.pointLights[0].position = maths::vec3{ -3.0f, 2.5f, -1.5f };
  m_SceneLights.pointLights[0].color = maths::vec3{ 1.0f, 0.85f, 0.6f };
  m_SceneLights.pointLights[0].intensity = 4.0f;
  m_SceneLights.pointLights[0].radius = 8.0f;
  m_SceneLights.pointLightCount = 1;

  if (!m_GlobalSDF.Init(createInfo.device, createInfo.allocator,
                        createInfo.commandPool, createInfo.queue,
                        createInfo.cacheFilePath)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize GlobalSDFPass.");
    return false;
  }

  // Shared trace-scene descriptor sets (mesh SDF trace scene + Surface Cache sampling), built
  // once from m_GlobalSDF's per-entity SDF images and m_SurfaceCache's card table/atlases (both
  // already Init'd above) -- reused unmodified by m_SurfaceCacheRT/m_GIInject/m_ScreenProbeGI.
  if (!m_TraceContext.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                           createInfo.queue, m_GlobalSDF, m_SurfaceCache)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize SurfaceCacheTraceContext.");
    return false;
  }
  // HWRT back-end: one BLAS per traced entity + one TLAS, built once against m_SurfaceCache's own
  // Fallback Mesh vertex/index buffers (the scene is static -- see this class' own "Entity self-
  // rotation" scope note -- so no per-frame rebuild is needed).
  if (!m_SurfaceCacheRT.Init(createInfo.physicalDevice, createInfo.device, createInfo.allocator,
                             createInfo.commandPool, createInfo.queue, m_TraceContext, m_SurfaceCache)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize SurfaceCacheRayTracingPass.");
    return false;
  }
  // Per-card secondary-bounce injection into m_SurfaceCache's own radiance atlas -- makes that
  // atlas genuinely multi-bounce over time, which m_ScreenProbeGI's single-hit-sample final
  // gather then benefits from directly.
  if (!m_GIInject.Init(createInfo.device, createInfo.allocator, m_TraceContext, m_SurfaceCache, m_SurfaceCacheRT)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize SurfaceCacheGIInjectPass.");
    return false;
  }
  // Screen Space Probe GI -- needs m_Resolve's GBuffer (already Init'd, STEP 6 above) for the
  // probe placement/gather passes, and m_TraceContext/m_SurfaceCacheRT for its own trace pass.
  if (!m_ScreenProbeGI.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                            createInfo.queue, createInfo.renderExtent, m_TraceContext, m_SurfaceCache,
                            m_SurfaceCacheRT, m_Resolve)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize ScreenProbeGIPass.");
    return false;
  }
  // Phase 2 (UE5.8 parity roadmap): specular reflections -- same dependencies as m_ScreenProbeGI
  // above (needs m_Resolve's GBuffer, including its Phase 1a roughness/metallic channel, plus
  // m_TraceContext/m_SurfaceCacheRT for its own trace pass).
  if (!m_Reflection.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                         createInfo.queue, createInfo.renderExtent, m_TraceContext, m_SurfaceCache,
                         m_SurfaceCacheRT, m_Resolve)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize ReflectionPass.");
    return false;
  }
  // World Probe grid (Lumen "Translucency Volume") -- reuses the same shared trace-scene sets and
  // HWRT/BLAS/TLAS as every other GI consumer above; see ClusterRenderPipeline.h's own comment on
  // why this is a separate system from m_ScreenProbeGI, not a duplicate of it.
  if (!m_WorldProbes.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                          createInfo.queue, m_TraceContext, m_SurfaceCache, m_SurfaceCacheRT)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize WorldProbeGridPass.");
    return false;
  }
  // Final spatial denoiser: denoises m_Resolve's own output color image (direct light +, after
  // RecordFrame()'s own [12b] step, m_ScreenProbeGI's gathered indirect term), guided by
  // m_Resolve's own G-buffer normal/depth -- both already Init'd (STEP 6 above).
  m_Denoiser.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
      createInfo.renderExtent, m_Resolve.GetOutputColorView(), m_Resolve.GetOutputDepthView(),
      m_Resolve.GetOutputNormalView());

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
  if (m_Device != VK_NULL_HANDLE) {
    LOG_INFO("[ClusterRenderPipeline] Shutting down cluster render pipeline...");
  }
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
  m_Reflection.Shutdown();
  m_ScreenProbeGI.Shutdown();
  m_Denoiser.Shutdown();
  m_WorldProbes.Shutdown();
  m_GIInject.Shutdown();
  m_SurfaceCacheRT.Shutdown();
  m_TraceContext.Shutdown();
  m_GlobalSDF.Shutdown();
  m_SurfaceCache.Shutdown();
  m_VirtualShadowMap.Shutdown();
  m_PrevViewProj = maths::mat4{};
  m_HasPrevViewProj = false;
  m_FrameIndex = 0;
#ifndef NDEBUG
  m_DebugTraceMode = 0;
#endif

  m_TransparentForward.Shutdown();
  m_ShadingBin.Shutdown();
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
  // Reversed-Z: 0.0 is now the "nothing drawn here yet" far sentinel (see
  // maths::mat4::PerspectiveVulkan's own comment) -- VK_COMPARE_OP_GREATER
  // (VulkanPipeline::CreateGraphicsPipeline) then lets any real, nearer (larger) depth win.
  depthAttachment.clearValue.depthStencil = {0.0f, 0};

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

  // SWRT/HWRT back-end shared by m_GIInject ([1z] below) and m_ScreenProbeGI ([12b] below) --
  // debug-only toggle (main.cpp's 'T' key via SetDebugTraceMode), Release always uses HWRT (no
  // debug toggle to switch back to SWRT).
#ifndef NDEBUG
  uint32_t traceMode = m_DebugTraceMode;
#else
  uint32_t traceMode = 1u;
#endif

  // =========================================================================================
  // [1] Per-frame worklist clears. Each Record*() carries its own
  // CLEAR->COMPUTE barrier, so the culling/raster dispatches below can never
  // read a stale counter.
  // =========================================================================================
  m_OcclusionCulling.RecordClearFrame(cmd);
  m_SoftwareRaster.RecordClear(cmd);
  m_LODSelection.RecordClear(cmd);

  // =========================================================================================
  // [1z] Lumen-style GI infrastructure: Virtual Shadow Map page requests/renders (Phase 3) ->
  // Surface Cache capture -> Global SDF clipmap streaming -> (Debug-only) SDF ray march debug
  // visualization. Independent of the Nanite VisBuffer/HZB machinery above and below (no shared
  // images, no fence contention -- see ClusterRenderPipeline.h's own class-comment addendum), so
  // it can run anywhere in the frame; placed early, right after the worklist clears, so it
  // overlaps with the GPU's own internal scheduling of the culling/streaming work that follows
  // rather than sitting at the tail end.
  // =========================================================================================
  {
    // Sun direction is fixed for now (m_SceneLights' own default, see LightingTypes.h) -- a
    // future day/night system would rotate it per frame instead.
    const maths::vec3 sunDirection = m_SceneLights.sun.direction;

    // 1. Virtual Shadow Maps first: SurfaceCachePass's/m_Resolve's shadow lookups (below and at
    // [12]) need THIS frame's VSM view-projection matrices + any pages rendered this frame already
    // visible -- RecordBeginFrame()'s own trailing barrier (when it renders any page) covers that.
    // See VirtualShadowMapPass's own class comment for the full one-frame-lag feedback contract.
    m_VirtualShadowMap.RecordBeginFrame(cmd, sunDirection, m_SceneLights, cameraFrameInfo.position);

    // 2. Surface Cache: feed this frame's light data before the visibility-driven capture draws --
    // shadow lookups now read renderer::VirtualShadowMapPass's own UBOs directly (bound once via
    // SetVirtualShadowMap()), no per-frame light-view-proj parameter needed here anymore.
    m_SurfaceCache.UpdateLighting(m_SceneLights);
    m_SurfaceCache.UpdateVisibility(cameraFrameInfo.position, cameraFrameInfo.forward,
                                    maths::vec3{0.0f, 1.0f, 0.0f}, cameraFrameInfo.fovYRadians,
                                    cameraFrameInfo.aspectRatio, cameraFrameInfo.nearZ,
                                    cameraFrameInfo.farZ);
    m_SurfaceCache.RecordCapture(cmd);

    // 3. Global SDF clipmap streaming, from this frame's camera position.
    m_GlobalSDF.RecordUpdate(cmd, cameraFrameInfo.position);

    // 4. Secondary-bounce injection into m_SurfaceCache's own radiance atlas (budgeted, a handful
    // of cards per call -- see SurfaceCacheGIInjectPass::kCardsPerFrameBudget) via m_TraceContext's
    // shared trace-scene sets against m_SurfaceCacheRT's static TLAS or the per-entity mesh SDFs,
    // depending on `traceMode`. Called kRadiosityBounceCount times in a row, each isolated from the
    // next by the exact same barrier RecordInject itself doesn't carry (see the comment below) --
    // this turns the pass's single budgeted round into a genuine INTRA-FRAME multi-bounce chain:
    // m_GIInject's own round-robin cursor advances across every call in the loop, so bounce N+1's
    // newly-processed cards sample the atlas including bounce N's own fresh writes wherever their
    // footprints overlap, on top of (not instead of) the pass's existing cross-frame convergence.
    // `radiosityEnabled` (debug-only toggle, main.cpp's 'G' key) lets this whole loop be switched
    // off to isolate its cost/visual contribution from the rest of the GI stack; Release always
    // runs it.
#ifndef NDEBUG
    bool radiosityEnabled = m_DebugRadiosityEnabled;
#else
    bool radiosityEnabled = true;
#endif
    if (radiosityEnabled) {
      for (uint32_t bounce = 0; bounce < kRadiosityBounceCount; ++bounce) {
        m_GIInject.RecordInject(cmd, m_TraceContext, m_SurfaceCache, traceMode);

        // Unlike SurfaceCachePass::RecordCapture/VirtualShadowMapPass::RecordBeginFrame/
        // GlobalSDFPass::RecordUpdate, SurfaceCacheGIInjectPass::RecordInject does NOT end with its own trailing
        // barrier -- its read-modify-write of the radiance atlas (imageLoad/imageStore, STORAGE
        // access) must be made visible here, explicitly, before the NEXT bounce's own RecordInject
        // call samples that same atlas (surface_cache_sampling.glsl's g_SurfaceCacheRadiance) --
        // and, after the loop's final iteration, before m_ScreenProbeGI's own trace pass does the
        // same later this frame.
        VkMemoryBarrier2 giInjectBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        giInjectBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        giInjectBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        giInjectBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        giInjectBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo giInjectDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        giInjectDepInfo.memoryBarrierCount = 1;
        giInjectDepInfo.pMemoryBarriers = &giInjectBarrier;
        vkCmdPipelineBarrier2(cmd, &giInjectDepInfo);
      }
    }

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
    // (via the DEBUG_VIEW_LUMEN/DEBUG_VIEW_GLOBAL_SDF blit-swap below), but it is always recorded
    // so either debug view shows this frame's state the instant it's selected, with no one-frame
    // lag. `coarseOnly` picks which of the two views this frame's single recording serves: DEBUG_
    // VIEW_GLOBAL_SDF wants ONLY the coarse clipmap trace (see SDFRayMarch.comp's own comment on
    // why that's a distinct, useful signal from DEBUG_VIEW_LUMEN's full two-tier result); any other
    // view mode (including DEBUG_VIEW_LUMEN itself) gets the normal full march, since only one of
    // the two blit-swaps can be selected at a time anyway.
    m_SDFRayMarch.RecordRayMarch(cmd, m_GlobalSDF, cameraFrameInfo.position, cameraFrameInfo.forward,
                                 maths::vec3{0.0f, 1.0f, 0.0f}, cameraFrameInfo.fovYRadians,
                                 cameraFrameInfo.aspectRatio, cameraFrameInfo.nearZ, cameraFrameInfo.farZ,
                                 camera.debugViewMode == DEBUG_VIEW_GLOBAL_SDF);
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

#ifndef NDEBUG
    // See RequestDebugDAGCutGapsDump()'s own comment: state 1 means main.cpp's 'K' key armed a
    // dump this frame -- record the DAGDecisionSSBO readback now (right after the dispatch that
    // just wrote this frame's decisions) and advance to state 2 so PumpDebugDAGCutGapsDump() logs
    // it once this frame's fence confirms the copy has landed.
    if (m_DebugDAGCutGapsDumpState == 1) {
        m_LODSelection.RecordDebugReadback(cmd);
        m_DebugDAGCutGapsDumpState = 2;
    }
#endif

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
                                     kSoftwareRasterThresholdPixels
#ifndef NDEBUG
                                     , camera.disableOcclusionCulling
#endif
                                     );

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
  m_OcclusionCulling.RecordLatePass(cmd
#ifndef NDEBUG
    , camera.disableOcclusionCulling
#endif
  );
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
  // [11b] Phase 1b: shading-bin classify + sort (renderer::ClusterShadingBinPass), only for the
  // normal (non-debug-visualization) view path -- every debug view (DEBUG_VIEW_NANITE_TRIANGLES,
  // DEBUG_VIEW_LUMEN, etc.) instead falls back to [12] Resolve's original full-screen dispatch
  // unchanged, since duplicating 15 debug-view branches into the leaner binned shader
  // (ClusterResolveBinned.comp) would only maintain two copies of the same visualization code for
  // no benefit -- see renderer::ClusterResolvePass::RecordResolveBinned's own comment. Release has
  // no debug view switch at all (camera.debugViewMode does not exist as a field, see
  // core/Camera.h), so Release always takes the binned path -- zero runtime branch, matching
  // CLAUDE.md's build-separation rule.
  // [12] Resolve: per-pixel hardware-vs-software arbitration + barycentric reconstruction +
  // material evaluation into the resolve pass's own RGBA8 output image (either path). The
  // software rasterizer's atomic writes are already visible (RecordRaster's trailing COMPUTE ->
  // COMPUTE barrier).
  // =========================================================================================
#ifndef NDEBUG
  if (camera.debugViewMode == DEBUG_VIEW_NORMAL) {
    m_ShadingBin.RecordClassifyAndSort(cmd, m_RenderExtent);
    m_Resolve.RecordResolveBinned(cmd, viewProj, m_SceneLights.sun.direction, m_ShadingBin);
  } else {
    maths::mat4 prevViewProjForResolve = m_HasPrevViewProj ? m_PrevViewProj : viewProj;
    m_Resolve.RecordResolve(cmd, viewProj, prevViewProjForResolve, m_SceneLights.sun.direction, camera.debugViewMode);
  }
#else
  m_ShadingBin.RecordClassifyAndSort(cmd, m_RenderExtent);
  m_Resolve.RecordResolveBinned(cmd, viewProj, m_SceneLights.sun.direction, m_ShadingBin);
#endif

  // Captures THIS frame's shadow-page miss reports (written by SurfaceCacheCapture.frag at [1z]
  // above and by whichever ClusterResolve.comp/ClusterResolveBinned.comp path just ran) for
  // VirtualShadowMapPass::RecordBeginFrame() to consume next frame -- see that class' own
  // one-frame-lag contract. Placed here, right after every pass that can call
  // RequestShadowPageResidency() this frame has run.
  m_VirtualShadowMap.RecordEndFrame(cmd);

  // [12b] Screen Space Probe GI: trace -> temporal reprojection/accumulation -> bilateral
  // gather, read-modify-writing m_Resolve's own output color image directly (see
  // renderer::ScreenProbeGIPass's own class comment for the exact per-frame call contract). Runs
  // after [12] Resolve (needs its GBuffer) and after [1z]'s GI injection (this frame's Surface
  // Cache radiance atlas already reflects this frame's own lighting/capture/injection updates).
  // =========================================================================================
  {
    maths::mat4 prevViewProjForProbes = m_HasPrevViewProj ? m_PrevViewProj : maths::mat4{};
    m_ScreenProbeGI.RecordUpdateViewParams(cmd, viewProj, prevViewProjForProbes);

    // `ssrtEnabled` (debug-only toggle, main.cpp's 'F' key) gates the trace/temporal/gather trio
    // entirely -- skipped means zero screen-space probe indirect contribution this frame at all,
    // isolating this stage's own cost/visual contribution from the rest of the GI stack. Release
    // always runs the trio.
#ifndef NDEBUG
    bool ssrtEnabled = m_DebugSSRTEnabled;
#else
    bool ssrtEnabled = true;
#endif
    if (ssrtEnabled) {
      m_ScreenProbeGI.RecordTrace(cmd, m_TraceContext, m_TraceContext.GetEntityCount(), traceMode, m_FrameIndex);
      m_ScreenProbeGI.RecordTemporal(cmd);
      m_ScreenProbeGI.RecordGather(cmd
#ifndef NDEBUG
        , camera.debugViewMode
#endif
      );
    }
  }

  // =========================================================================================
  // [12b2] Phase 2 (UE5.8 parity roadmap): specular reflections -- trace -> temporal
  // reprojection/accumulation -> Fresnel-weighted gather, read-modify-writing m_Resolve's own
  // output color image directly, the exact same convention as [12b] above (see
  // renderer::ReflectionPass's own class comment for the full per-frame call contract). Grouped
  // right after [12b] since both read the same GBuffer and compose into the same color image;
  // runs BEFORE [12d]'s À-Trous denoiser deliberately -- reflections have no dedicated spatial
  // filter of their own, they inherit whatever spatial smoothing the existing denoiser already
  // applies to the rest of the composited image (see this phase's approved plan's own "Hors
  // scope" note).
  //
  // `reflectionsEnabled` (debug-only toggle, main.cpp's 'R' key) gates the trio entirely, same
  // A/B-ability as `ssrtEnabled` above. Release always runs the trio (see
  // SetDebugReflectionsEnabled()'s own comment for why this differs from worldProbesEnabled).
  // =========================================================================================
  {
    maths::mat4 prevViewProjForReflection = m_HasPrevViewProj ? m_PrevViewProj : maths::mat4{};
    m_Reflection.RecordUpdateViewParams(cmd, viewProj, prevViewProjForReflection, cameraPositionWorld);

#ifndef NDEBUG
    bool reflectionsEnabled = m_DebugReflectionsEnabled;
#else
    bool reflectionsEnabled = true;
#endif
    if (reflectionsEnabled) {
      m_Reflection.RecordTrace(cmd, m_TraceContext, m_TraceContext.GetEntityCount(), traceMode, m_FrameIndex);
      m_Reflection.RecordTemporal(cmd);
      m_Reflection.RecordGather(cmd);
    }
  }

  // =========================================================================================
  // [12c] World Probe grid: fully rebuilt every frame from the Surface Cache radiance atlas
  // [1z] already re-injected into this frame ("Propage l'éclairage du Surface Cache directement
  // dans cette grille 3D à chaque frame") -- INTENDED as what dynamic/off-screen objects would
  // sample for indirect light (world_probe_sampling.glsl's SampleWorldProbeGrid), since
  // m_ScreenProbeGI's screen-space probes only exist for on-screen pixels. Independent GPU work
  // from [12b] above (no data dependency either way), just recorded after it for locality with
  // the rest of this frame's GI additions.
  //
  // `worldProbesEnabled` (debug-only toggle, main.cpp's 'H' key) gates this dispatch entirely --
  // see SetDebugWorldProbesEnabled()'s own comment. UNLIKE radiosityEnabled/ssrtEnabled above,
  // this system has no live consumer yet (SampleWorldProbeGrid() is called only by the dead
  // ScreenTracePass/GICompositePass, per the 2026-07-16 UE5.8-parity audit) -- so Release
  // hardcodes this OFF instead of ON, skipping the dispatch (and its trailing barrier, since
  // nothing this frame reads the grid either way) rather than paying its GPU cost for zero visual
  // effect. Flip Release's hardcoded default once a real consumer samples this grid.
  // =========================================================================================
#ifndef NDEBUG
  bool worldProbesEnabled = m_DebugWorldProbesEnabled;
#else
  bool worldProbesEnabled = false;
#endif
  if (worldProbesEnabled) {
    m_WorldProbes.RecordUpdate(cmd, cameraFrameInfo.position, m_TraceContext, traceMode);
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
  }

  // =========================================================================================
  // [12d] Final spatial denoiser (À-Trous wavelet): only in the normal (non-debug-visualization)
  // view path -- DEBUG_VIEW_LUMEN substitutes a completely different image at the blit ([14]
  // below) and DEBUG_VIEW_SPATIAL_PROBES's own gridline overlay (drawn directly into m_Resolve's
  // color image by [12b] above) would be blurred by an edge-aware filter whose depth/normal
  // guides know nothing about that artificial overlay -- so denoising either would only degrade
  // an existing debug tool for no benefit. m_Resolve's own G-buffer normal/depth (already visible
  // from [11]/[12]'s own trailing barriers) guide the filter; ends with its own trailing barrier
  // (COMPUTE_SHADER/STORAGE_WRITE -> COMPUTE_SHADER/SHADER_SAMPLED_READ) for [14]'s blit read.
  // =========================================================================================
#ifndef NDEBUG
  bool applyDenoise = (camera.debugViewMode == DEBUG_VIEW_NORMAL);
#else
  bool applyDenoise = true;
#endif
  if (applyDenoise) {
    m_Denoiser.RecordDenoise(cmd);
  }

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

  // =========================================================================================
  // [13c] Forward-rendered translucent/transparent materials -- drawn onto whichever image [14]'s
  // blit will actually read (m_Denoiser's output when [12d] applied it, m_Resolve's own color
  // image otherwise -- same `applyDenoise` condition the debug overlay below and the blit itself
  // both use), so transparency composites on top of the fully-lit (GI/reflections included) opaque
  // scene. No-op internally if TransparentForwardPass::Init() found zero transparent leaf clusters
  // this run. Must run before the debug overlay below so the HUD stays on top of everything.
  // =========================================================================================
  {
    VkImage transparentTargetImage = applyDenoise ? m_Denoiser.GetOutputImage() : m_Resolve.GetOutputColorImage();
    VkImageView transparentTargetView = applyDenoise ? m_Denoiser.GetOutputView() : m_Resolve.GetOutputColorView();
    m_TransparentForward.RecordDraw(cmd, transparentTargetImage, transparentTargetView, m_DepthImageView,
        m_RenderExtent, camera.view, camera.proj, m_Decompression.GetDecompressedIndexPoolBuffer(),
        m_SceneLights.sun.direction);
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
    // Same wall-clock delta this block already samples once per frame for bytesPerSecond above --
    // reused directly rather than re-derived, since it already IS this frame's frame-to-frame time.
    float fps = (deltaTime > 0.0f) ? (1.0f / deltaTime) : 0.0f;
    m_LastStatsSampleTime = globalTimeSeconds;
    m_LastStatsSampleBytes = totalBytesCompleted;

    m_DebugOverlay.BuildFrameText(gpuMemUsedMB, pendingPageLoads, bytesPerSecond, hwTriangleCount, swTriangleCount,
        fps, static_cast<float>(m_RenderExtent.width), m_DebugRadiosityEnabled, m_DebugSSRTEnabled, traceMode,
        m_DebugWorldProbesEnabled);
    // Drawn onto whichever image [14]'s blit will actually read below (m_Denoiser's output when
    // [12d] applied it, m_Resolve's own color image otherwise -- same `applyDenoise` condition) --
    // the overlay text must land on the real final image, not on a buffer already bypassed.
    VkImage overlayTargetImage = applyDenoise ? m_Denoiser.GetOutputImage() : m_Resolve.GetOutputColorImage();
    VkImageView overlayTargetView = applyDenoise ? m_Denoiser.GetOutputView() : m_Resolve.GetOutputColorView();
    m_DebugOverlay.RecordDraw(cmd, overlayTargetImage, overlayTargetView, m_RenderExtent);
  }
#endif

  // =========================================================================================
  // [14] Blit the final image (m_Denoiser's own denoised output, when [12d] applied it; otherwise
  // m_Resolve's own color image directly -- see `applyDenoise`/`blitSourceImage` above) to the
  // swapchain image and hand it to present. Every candidate image stays in GENERAL (valid blit
  // source layout); each one's own trailing barrier already targeted the BLIT stage, but is
  // re-stated here explicitly for the same reason every other cross-pass barrier in this function
  // is: the dependency must not silently vanish if an intermediate pass's barriers change.
  // =========================================================================================
  // DEBUG_VIEW_LUMEN and DEBUG_VIEW_GLOBAL_SDF both swap the blit SOURCE to m_SDFRayMarch's own
  // debug-visualization image instead of m_Resolve's normal lit output -- see ClusterRenderPipeline
  // .h's own comment on why this (not sampling it from within ClusterResolve.comp) is the chosen
  // mechanism. Both share the exact same output image (this frame's single RecordRayMarch() call
  // above already picked which of the two variants -- full two-tier vs. coarse-only -- filled it,
  // via `coarseOnly`); the image's format (VK_FORMAT_R8G8B8A8_UNORM) and permanent layout (GENERAL)
  // match m_Resolve's own output either way, so the SAME barrier below (targeting COMPUTE_SHADER_
  // BIT/SHADER_STORAGE_WRITE_BIT -> BLIT_BIT/TRANSFER_READ_BIT) covers either choice with no changes
  // needed -- only which VkImage is actually blitted differs.
  // Denoised (see [12d] above) unless a debug view substitutes a different image entirely
  // (DEBUG_VIEW_LUMEN/DEBUG_VIEW_GLOBAL_SDF) or would have its own overlay blurred by the filter
  // (DEBUG_VIEW_SPATIAL_PROBES) -- applyDenoise (computed at [12d]) tracks exactly the same condition.
  VkImage blitSourceImage = applyDenoise ? m_Denoiser.GetOutputImage() : m_Resolve.GetOutputColorImage();
#ifndef NDEBUG
  if (camera.debugViewMode == DEBUG_VIEW_LUMEN || camera.debugViewMode == DEBUG_VIEW_GLOBAL_SDF) {
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

  // Recorded last, after every consumer of "the previous frame's" viewProj above (m_Resolve's own
  // motion-vector debug view, m_ScreenProbeGI's temporal reprojection) has already read it --
  // this frame's own matrix becomes "the previous frame's" for the NEXT RecordFrame() call.
  m_PrevViewProj = viewProj;
  m_HasPrevViewProj = true;
  ++m_FrameIndex;
}

} // namespace renderer
