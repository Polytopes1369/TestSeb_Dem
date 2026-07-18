#include "renderer/ClusterRenderPipeline.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>

#include "core/Logger.h"
#ifndef NDEBUG
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#endif
#include "geometry/ClusterFormat.h"
#include "io/CacheFileManager.h"
#include "renderer/MegaLightsTypes.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of the WPOGlobalsUBO block declared identically in ClusterRaster.vert
        // and ClusterSoftwareRaster.comp (std140): a single float naturally needs no padding to reach
        // a 16-byte size, but the trailing pad floats are declared explicitly anyway (matching this
        // codebase's own convention, e.g. ClusterSoftwareRasterPass.cpp's SoftwareRasterViewParams)
        // so the CPU-side struct's sizeof() stays an honest, self-documenting match for the UBO's
        // actual GPU size.
        //
        // Phase 1 (Nanite advanced): the two previously-dead pad floats are repurposed as debug
        // toggle multipliers (enhancedDisplacementDebugMultiplier/splineDeformationDebugMultiplier,
        // see SetDebugEnhancedDisplacementEnabled/SetDebugSplineDeformationEnabled's own comments) --
        // zero struct size change, sizeof() stays exactly 16 bytes (see the static_assert below).
        struct WPOGlobalsUBO {
            float globalTime = 0.0f;
            float enhancedDisplacementDebugMultiplier = 1.0f;
            float splineDeformationDebugMultiplier = 1.0f;
            float _pad2 = 0.0f;
        };
        static_assert(sizeof(WPOGlobalsUBO) == 16,
            "WPOGlobalsUBO must match the WPOGlobalsUBO block in ClusterRaster.vert / ClusterSoftwareRaster.comp exactly (std140 layout)");

        // Byte-for-byte mirror of SplineControlPoint (spline_deformation.glsl, std430): a position
        // and an (un-normalized, magnitude matters) tangent, each vec3 padded to 16 bytes by its own
        // implicit std430 alignment -- the explicit pad floats below add zero hidden slack, matching
        // this codebase's own "keep sizeof() honest" convention (see WPOGlobalsUBO's own comment).
        struct SplineControlPoint {
            maths::vec3 position;
            float _pad0 = 0.0f;
            maths::vec3 tangent;
            float _pad1 = 0.0f;
        };
        static_assert(sizeof(SplineControlPoint) == 32,
            "SplineControlPoint must match SplineControlPoint in spline_deformation.glsl exactly (std430 layout)");

        // The authored demo curve: a gentle S-curve spanning entity 6 (Tube)'s rest-pose local Y
        // range [-SPLINE_REST_POSE_HALF_HEIGHT, +SPLINE_REST_POSE_HALF_HEIGHT] (see
        // spline_deformation.glsl's own header comment), with a lateral offset up to ~0.5 in X.
        // Tangents point predominantly along +Y (magnitude 1.4, matching the segment's own Y span)
        // so the bend reads as one continuous curve, never a kink, across all 3 Hermite segments.
        constexpr std::array<SplineControlPoint, 4> kSplineControlPoints = { {
            { maths::vec3{0.00f, -0.70f, 0.0f}, 0.0f, maths::vec3{0.00f, 1.4f, 0.0f}, 0.0f },
            { maths::vec3{0.25f, -0.233f, 0.0f}, 0.0f, maths::vec3{0.30f, 1.4f, 0.0f}, 0.0f },
            { maths::vec3{0.45f,  0.233f, 0.0f}, 0.0f, maths::vec3{0.10f, 1.4f, 0.0f}, 0.0f },
            { maths::vec3{0.50f,  0.70f, 0.0f}, 0.0f, maths::vec3{0.00f, 1.4f, 0.0f}, 0.0f },
        } };

        float RadicalInverse(uint32_t index, uint32_t base) {
            float result = 0.0f;
            float f = 1.0f / static_cast<float>(base);
            uint32_t i = index;
            while (i > 0) {
                result += f * static_cast<float>(i % base);
                i /= base;
                f /= static_cast<float>(base);
            }
            return result;
        }

        maths::vec2 Halton23(uint32_t index) {
            return { RadicalInverse(index, 2), RadicalInverse(index, 3) };
        }

    } // namespace


bool ClusterRenderPipeline::Init(
    const ClusterRenderPipelineCreateInfo &createInfo) {
  kSoftwareRasterThresholdPixels = config::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
  kLODPixelErrorThreshold = config::nanite::LOD_PIXEL_ERROR_THRESHOLD;
  kRadiosityBounceCount = config::lumen::RADIOSITY_BOUNCE_COUNT;

  Shutdown();

  m_Device = createInfo.device;
  m_GraphicsQueueFamilyIndex = createInfo.graphicsQueueFamilyIndex;
  // Phase 2 (Lumen advanced roadmap): async-compute queue handles -- see this class' own
  // m_AsyncComputeAvailableThisBuild member comment for exactly what m_AsyncComputeAvailableThisBuild
  // does and does not gate.
  m_AsyncComputeQueue = createInfo.asyncComputeQueue;
  m_AsyncComputeQueueFamilyIndex = createInfo.asyncComputeQueueFamilyIndex;
  m_AsyncComputeAvailableThisBuild = createInfo.hasDedicatedAsyncComputeQueue;
  m_RenderExtent = createInfo.renderExtent;
  m_DisplayExtent = createInfo.displayExtent;
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
  m_PagePool.Init(createInfo.allocator, maxLogicalPages, totalClusterCount,
                  createInfo.transferQueueFamilyIndex, createInfo.graphicsQueueFamilyIndex);
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
                   createInfo.cacheFilePath, indexEntries, dagEntries);

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
      // Startup-only path: one blocking one-shot submit on a single queue (createInfo.queue), so
      // there is no cross-queue-family hop to synchronize here -- UploadPageData()'s copy and
      // FinalizeBoundPage()'s table write both simply record onto this same `cmd`, with no
      // Release/AcquirePhysicalPoolOwnership() call needed (those exist only for the per-frame
      // streaming path's transfer-queue -> graphics-queue handoff, see GeometryStreamingCoordinator).
      bool uploaded = m_PagePool.UploadPageData(
          cmd, entry.virtualAddress, stagingBuffer.Handle(),
          entry.virtualAddress - header.geometryDataBaseOffset,
          geometry::kPageSizeBytes);
      if (!uploaded) {
        LOG_ERROR(std::format(
            "[ClusterRenderPipeline] UploadPageData failed for cluster {}.",
            entry.clusterID));
        throw std::runtime_error("ClusterRenderPipeline: UploadPageData failed during initialization");
      }
      m_PagePool.FinalizeBoundPage(cmd, entry.virtualAddress);

      // The logical->physical mapping is CPU-side bookkeeping, valid the moment
      // UploadPageData() records -- so the matching decompression dispatch can target
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

  // Phase 1 (Nanite advanced): m_SplineControlPointsBuffer -- 128 bytes (4x SplineControlPoint,
  // std430), the one authored Hermite bend curve every spline-deformation call site reads. A
  // one-time CPU->GPU staged upload (not a per-frame vkCmdUpdateBuffer like m_WPOGlobalsBuffer
  // above -- this curve never changes after Init()), mirroring VulkanContext::UploadEntityData's
  // own staging-buffer + one-shot-command-buffer pattern exactly.
  {
    m_SplineControlPointsBuffer.Create(createInfo.allocator, sizeof(kSplineControlPoints),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingInfo.size = sizeof(kSplineControlPoints);
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo stagingAllocResultInfo{};
    VK_CHECK(vmaCreateBuffer(createInfo.allocator, &stagingInfo, &stagingAllocInfo,
        &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));
    std::memcpy(stagingAllocResultInfo.pMappedData, kSplineControlPoints.data(), sizeof(kSplineControlPoints));

    VulkanUtils::ExecuteOneShotCommands(m_Device, createInfo.commandPool, createInfo.queue, [&](VkCommandBuffer cmd) {
      VkBufferCopy copyRegion{};
      copyRegion.srcOffset = 0;
      copyRegion.dstOffset = 0;
      copyRegion.size = sizeof(kSplineControlPoints);
      vkCmdCopyBuffer(cmd, stagingBuffer, m_SplineControlPointsBuffer.Handle(), 1, &copyRegion);

      // Explicit barrier: the copy's writes must be visible to every vertex/compute shader stage
      // that reads this SSBO (all 5 spline-deformation call sites, see the class comment).
      VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

      VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      depInfo.memoryBarrierCount = 1;
      depInfo.pMemoryBarriers = &memBarrier;
      vkCmdPipelineBarrier2(cmd, &depInfo);
    });

    vmaDestroyBuffer(createInfo.allocator, stagingBuffer, stagingAllocation);
  }

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
                        createInfo.entityDataBuffer,
                        m_SplineControlPointsBuffer.Handle());

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
                        createInfo.entityDataBuffer,
                        m_SplineControlPointsBuffer.Handle());

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
      createInfo.materialTable.params,
      m_SplineControlPointsBuffer.Handle());

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
                    m_Resolve.GetOutputRoughnessMetallicView(), m_Resolve.GetOutputMaterialIDView());
  m_Resolve.InitBinnedResolve(createInfo.device, createInfo.commandPool, createInfo.queue, m_ShadingBin);

  // =========================================================================================
  // STEP 7 -- Lumen-style GI infrastructure: sun+point-light Virtual Shadow Maps (Phase 3), Surface
  // Cache, Global SDF clipmap (see ClusterRenderPipeline.h's own class-comment addendum on why
  // these are unconditional, not Debug-only, unlike the stats/overlay block and m_SDFRayMarch
  // below). Each pass re-reads the same consolidated .cache file independently (this codebase's
  // established "self-contained pass" convention -- see e.g. renderer::VirtualShadowMapPass's own
  // class comment).
  // =========================================================================================
  // VSM advanced roadmap, Feature 1/3: the 4 buffers below are the SAME ones
  // m_HardwareRaster.Init()/m_SoftwareRaster.Init()/m_Resolve.Init() already received above (all
  // already created/uploaded at this point in Init()), plus createInfo.entityDataCPU (Feature 1's
  // per-entity material lookup) and m_MaskGenerator.GetMaskImageInfos() (Feature 3, initialized
  // earlier at STEP 6 above, before any consumer pass that binds it).
  if (!m_VirtualShadowMap.Init(createInfo.physicalDevice, createInfo.device, createInfo.allocator,
                               createInfo.commandPool, createInfo.queue,
                               createInfo.cacheFilePath,
                               createInfo.entityTransformBuffer, createInfo.entityDataBuffer,
                               m_WPOGlobalsBuffer.Handle(), m_SplineControlPointsBuffer.Handle(),
                               createInfo.entityDataCPU, m_MaskGenerator.GetMaskImageInfos())) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize VirtualShadowMapPass.");
    return false;
  }
  if (!m_SurfaceCache.Init(createInfo.device, createInfo.allocator,
                           createInfo.commandPool, createInfo.queue,
                           createInfo.cacheFilePath, createInfo.entityDataBuffer,
                           m_Resolve.GetMaterialParamsBuffer(),
                           createInfo.entityDataCPU, createInfo.materialTable)) {
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

  // Step 4 (Virtual Texturing / RVT-SVT, UE 5.8 parity roadmap): a single Albedo-only physical pool
  // (RGBA8_UNORM, matching ClusterResolvePass::kOutputAlbedoFormat's own choice of format for the
  // exact same "simplest, most broadly supported color-attachment-and-storage-capable format"
  // reason) -- see renderer::VirtualTextureConfig's own struct defaults for the virtual space/tile/
  // border/physical-capacity sizing this demo uses unmodified. Always Init()'d and always wired into
  // m_Resolve's descriptor sets below regardless of config::lumen::BUILD_VIRTUAL_TEXTURES (that flag
  // only gates the expensive per-frame disk-streaming work, see VirtualTextureStreamingCoordinator::
  // RecordBeginFrame's own comment) -- exactly like m_VirtualShadowMap's own unconditional Init()
  // above, gated the same way by config::lumen::BUILD_SHADOWS instead.
  VirtualTextureConfig vtConfig{};
  std::vector<VkFormat> vtPoolFormats{ VK_FORMAT_R8G8B8A8_UNORM };
  if (!m_VTManager.Init(createInfo.physicalDevice, createInfo.device, createInfo.allocator,
                        createInfo.commandPool, createInfo.queue, vtConfig, vtPoolFormats)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize VirtualTextureManager.");
    return false;
  }

  // Default volume bounds (see VirtualTextureVolumeBounds' own struct defaults, [-8,8]^2 XZ / [-4,4]
  // Y) already cover this demo's scene bounding sphere (~6.4m radius, see VirtualShadowMapPass's own
  // scene-bounds read) with generous margin -- kept as-is rather than re-deriving from the geometry
  // cache file's own bounds.
  m_VTBounds = VirtualTextureVolumeBounds{};
  if (!m_VTRenderPass.Init(createInfo.device, &m_VTManager, m_VTBounds)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize VirtualTextureRenderPass.");
    return false;
  }

  // Streams previously-baked tiles back from disk (.vtcache, sibling to the geometry .cache file at
  // the same path with a different extension) -- see VirtualTextureStreamingCoordinator::Init's own
  // comment for why a missing file is not a hard failure (nothing has ever been baked/persisted yet
  // on a from-scratch run -- this demo has no offline VT bake step wired up yet, matching this
  // phase's own scope: the streaming PLUMBING is real and complete, the content SOURCE is future
  // work, exactly like VirtualTextureRenderPass itself is not yet driven by any real page-visibility
  // system either).
  std::filesystem::path vtCacheFilePath = createInfo.cacheFilePath;
  vtCacheFilePath.replace_extension(".vtcache");
  if (!m_VTStreaming.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                          createInfo.queue, vtCacheFilePath, &m_VTManager)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize VirtualTextureStreamingCoordinator.");
    return false;
  }

  m_Resolve.SetVirtualTexture(m_VTManager, m_VTBounds.worldMinXZ, m_VTBounds.worldMaxXZ,
                              m_VTStreaming.GetFeedbackDeviceBuffer());

  // NOTE: renderer::TransparentForwardPass::Init() is deliberately NOT called here anymore -- see
  // the call site further below (after m_SurfaceCacheRT/m_WorldProbes/m_MegaLights are all Init'd),
  // moved there specifically so its own World Probe Grid indirect-diffuse bindings (Phase 5) and
  // MegaLights Phase A follow-up bindings (shared TLAS + light SSBO) can all be bound immediately
  // at Init() time instead of needing extra deferred-binding calls alongside SetVirtualShadowMap().

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
  // Real photometric candela (renderer::PointLight's own comment, 2026-07-17 recalibration) --
  // left at this class's own default (see LightingTypes.h) rather than a distinct override.
  m_SceneLights.pointLights[0].intensity = renderer::PointLight{}.intensity;
  m_SceneLights.pointLights[0].radius = 8.0f;
  m_SceneLights.pointLightCount = 1;

  if (!m_GlobalSDF.Init(createInfo.device, createInfo.allocator,
                        createInfo.commandPool, createInfo.queue,
                        createInfo.cacheFilePath, m_LoadingManager)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize GlobalSDFPass.");
    return false;
  }

  // Atmos weather system, Subtask 1: climate/wind state producer -- no dependency on any pass
  // above/below (owns a single small UBO, no descriptor set -- see AtmosClimatePass.h's own class
  // comment), so its Init() placement here is arbitrary; kept next to m_GlobalSDF since both are
  // "self-contained globals producers" Init'd unconditionally in this same STEP 7 block.
  if (!m_AtmosClimate.Init(createInfo.device, createInfo.allocator,
                           createInfo.commandPool, createInfo.queue)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize AtmosClimatePass.");
    return false;
  }

  // Atmos weather system, Subtask 2: Physically Based Sky (Hillaire LUTs) -- also a self-contained
  // globals producer (no dependency on any other pass' output), Init'd unconditionally here so its
  // Sky-View LUT is available both to m_PostProcess below (Release-visible) and to m_SDFRayMarch's
  // deferred SetAtmosSkyView() wiring further down (Debug-only block).
  if (!m_AtmosSky.Init(createInfo.device, createInfo.allocator,
                       createInfo.commandPool, createInfo.queue)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize AtmosSkyPass.");
    return false;
  }

  // Shared trace-scene descriptor sets (mesh SDF trace scene + Surface Cache sampling), built
  // once from m_GlobalSDF's per-entity SDF images and m_SurfaceCache's card table/atlases (both
  // already Init'd above) -- reused unmodified by m_SurfaceCacheRT/m_GIInject/m_WorldProbes.
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
  // atlas genuinely multi-bounce over time, which every downstream GI consumer (m_WorldProbes'
  // trace pass, m_ScreenTrace's near-field hit samples below) then benefits from directly.
  if (!m_GIInject.Init(createInfo.device, createInfo.allocator, m_TraceContext, m_SurfaceCache, m_SurfaceCacheRT)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize SurfaceCacheGIInjectPass.");
    return false;
  }
  // Atmos weather system, Subtask 5: one-time wiring (see SurfaceCacheGIInjectPass::SetAtmosSkyView's
  // own comment) -- m_AtmosSky is already Init'd above and its Sky-View LUT view/sampler never change.
  m_GIInject.SetAtmosSkyView(m_AtmosSky.GetSkyViewLUTView(), m_AtmosSky.GetLUTSampler());
  // Phase 2 (UE5.8 parity roadmap): specular reflections -- needs m_Resolve's GBuffer (already
  // Init'd, STEP 6 above), including its Phase 1a roughness/metallic channel, plus
  // m_TraceContext/m_SurfaceCacheRT for its own trace pass.
  if (!m_Reflection.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                         createInfo.queue, createInfo.renderExtent, m_TraceContext, m_SurfaceCache,
                         m_SurfaceCacheRT, m_Resolve)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize ReflectionPass.");
    return false;
  }
  // Phase PP4 (post-process stack roadmap): GTAO / Screen-Space Contact Shadows / SSR Fallback --
  // needs m_Resolve's GBuffer (same as m_Reflection above) AND m_Reflection's own hit-mask (just
  // Init'd), so must Init after both.
  m_ScreenSpaceEffects.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                            createInfo.queue, createInfo.renderExtent, m_Resolve, m_Reflection);
  // Atmos weather system, Subtask 5: one-time wiring (see ScreenSpaceEffectsPass::SetAtmosSkyView's
  // own comment) -- m_AtmosSky is already Init'd above and its Sky-View LUT view/sampler never change.
  m_ScreenSpaceEffects.SetAtmosSkyView(m_AtmosSky.GetSkyViewLUTView(), m_AtmosSky.GetLUTSampler());
  // Phase A of the MegaLights native-port roadmap: procedurally scatter kMaxMegaLights point
  // lights around the demo's 13-entity grid (see MegaLightsTypes.h's own GenerateProceduralLights
  // comment), then Init the pass -- needs m_Resolve's GBuffer (same as m_Reflection above) and
  // m_SurfaceCacheRT's TLAS for its own shadow-visibility ray.
  {
    MegaLightsData megaLightsData = GenerateProceduralLights(4242u);
    if (!m_MegaLights.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                           createInfo.queue, createInfo.renderExtent, m_Resolve,
                           m_SurfaceCacheRT, megaLightsData)) {
      LOG_ERROR("[ClusterRenderPipeline] Failed to initialize MegaLightsPass.");
      return false;
    }
  }

  // Atmos weather system, Subtask 3: Froxel Volumetric Fog -- needs m_AtmosClimate (AtmosGlobalsUBO),
  // m_MegaLights (light SSBO), and m_VirtualShadowMap (shadow atlas/page-table/feedback/sun-levels),
  // all already Init'd above.
  if (!m_AtmosFog.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                       m_AtmosClimate, m_MegaLights, m_VirtualShadowMap)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize AtmosVolumetricFogPass.");
    return false;
  }

  // Atmos weather system, Subtask 4: Procedural Volumetric Clouds -- half-res output sized from
  // this frame's own render extent (see AtmosCloudsPass.h's own class comment for why).
  if (!m_AtmosClouds.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                          createInfo.renderExtent, m_AtmosClimate)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize AtmosCloudsPass.");
    return false;
  }
  // Atmos weather system, Subtask 5: one-time wiring, now that both producers (m_AtmosSky, just
  // above; m_AtmosClouds, just Init'd here) exist -- see SurfaceCachePass::SetAtmosCloudShadow's and
  // ClusterResolvePass::SetAtmosCloudLighting's own comments. Neither the Sky-View LUT nor the Cloud
  // Shadow Map's view/sampler are ever recreated after their owning pass' Init(), so these bindings
  // are never refreshed again.
  m_SurfaceCache.SetAtmosCloudShadow(m_AtmosClouds.GetCloudShadowMapView(), m_AtmosClouds.GetCloudShadowMapSampler());
  m_Resolve.SetAtmosCloudLighting(m_AtmosSky.GetLUTSampler(), m_AtmosSky.GetSkyViewLUTView(),
                                  m_AtmosClouds.GetCloudShadowMapSampler(), m_AtmosClouds.GetCloudShadowMapView());

  // World Probe grid (Lumen "Translucency Volume") -- reuses the same shared trace-scene sets and
  // HWRT/BLAS/TLAS as every other GI consumer above; see ClusterRenderPipeline.h's own comment on
  // its live consumers (m_ScreenTrace's fallback, GICompositePass's debug visualization). Must be
  // Init'd before m_TransparentForward below, which reads its grid view/sampler immediately.
  if (!m_WorldProbes.Init(createInfo.device, createInfo.allocator, createInfo.commandPool,
                          createInfo.queue, m_TraceContext, m_SurfaceCache, m_SurfaceCacheRT)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize WorldProbeGridPass.");
    return false;
  }
  // Atmos weather system, Subtask 5: one-time wiring (see WorldProbeGridPass::SetAtmosSkyView's own
  // comment) -- m_AtmosSky is already Init'd above and its Sky-View LUT view/sampler never change.
  m_WorldProbes.SetAtmosSkyView(m_AtmosSky.GetSkyViewLUTView(), m_AtmosSky.GetLUTSampler());

  // Forward-rendered translucent/transparent materials (see TransparentForwardPass's own class
  // comment) -- reuses the SAME indexEntries/dagEntries this function loaded above for
  // m_LODSelection.Init(), and the SAME page pool/compressed pool/entity/WPO buffer handles every
  // opaque pass already borrows. Placed here (after m_SurfaceCacheRT, m_MegaLights and m_WorldProbes,
  // not right after m_LODSelection where it used to sit) since its shading needs m_WorldProbes'
  // grid view/sampler and m_TraceContext's shared sets immediately at Init() time, plus MegaLights'
  // Phase A follow-up bindings (TLAS + light SSBO, bindings 11/12) -- see
  // TransparentForwardPass::Init()'s own parameter comment.
  // colorFormat = GICompositePass::kOutputFormat, NOT ClusterResolvePass::kOutputColorFormat: this
  // pass draws directly onto m_GIComposite's own output image (see its RecordDraw() call site
  // below), not m_Resolve's -- the two constants happen to hold the same value today, but this
  // call site must name the format of the image it actually targets, not a coincidentally-equal
  // one, so the two are free to diverge later without silently breaking this pipeline's format.
  m_TransparentForward.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                            m_RenderExtent,
                            m_PagePool.GetPageTableBuffer(), m_PagePool.GetPhysicalPoolBuffer(),
                            createInfo.entityTransformBuffer, createInfo.entityDataBuffer, m_WPOGlobalsBuffer.Handle(),
                            createInfo.materialTable.params, indexEntries, dagEntries,
                            GICompositePass::kOutputFormat, createInfo.depthFormat,
                            m_WorldProbes, m_TraceContext, m_SurfaceCacheRT.GetTLASHandle(),
                            m_MegaLights.GetLightBufferHandle(), m_MegaLights.GetLightBufferSize(),
                            m_SurfaceCache.GetVertexBuffer(), m_SurfaceCache.GetIndexBuffer(),
                            m_SurfaceCacheRT.GetDrawRangeBuffer(),
                            m_SplineControlPointsBuffer.Handle());
  m_TransparentForward.SetVirtualShadowMap(m_VirtualShadowMap);

  // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): forward-rendered, tessellated/
  // displaced hero Icosphere -- see ClusterRenderPipeline.h's own comment on m_HeroTessellation.
  // Same borrowed-resource contract as m_TransparentForward above, resolved for the hero entity
  // instead of every transparent entity; `heroMaterial` is the caller's own
  // materialTable.params[renderer::kHeroMaterialID] slot (populated by
  // GenerateShowcaseMaterialTable()'s own hero-recipe override, see that function's own comment).
  {
    // Matches VulkanContext::kHeroEntityIndex exactly (the Icosphere, generated first -- see that
    // class' own comment) -- a plain literal here rather than a cross-file constant reference,
    // since ClusterRenderPipeline.cpp does not otherwise depend on VulkanContext.h.
    constexpr uint32_t kHeroEntityID = 2u;
    const auto& entityRanges = m_SurfaceCache.GetEntityRanges();
    const auto heroRangeIt = entityRanges.find(kHeroEntityID);
    if (heroRangeIt == entityRanges.end()) {
      LOG_ERROR("[ClusterRenderPipeline] Hero entity (meshID=2) has no Fallback Mesh draw range -- cannot initialize HeroTessellationPass.");
      return false;
    }
    if (!m_HeroTessellation.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                            GICompositePass::kOutputFormat, createInfo.depthFormat,
                            createInfo.entityTransformBuffer,
                            createInfo.materialTable.params[kHeroMaterialID],
                            m_SurfaceCache.GetVertexBuffer(), m_SurfaceCache.GetIndexBuffer(),
                            heroRangeIt->second, kHeroEntityID,
                            m_SurfaceCacheRT.GetTLASHandle(), m_SurfaceCacheRT.GetDrawRangeBuffer(),
                            m_MegaLights.GetLightBufferHandle(), m_MegaLights.GetLightBufferSize(),
                            m_VirtualShadowMap, m_WorldProbes, m_TraceContext)) {
      LOG_ERROR("[ClusterRenderPipeline] Failed to initialize HeroTessellationPass.");
      return false;
    }
  }

  // Phase 7c (UE5.8 parity roadmap, water/erosion): forward-rendered water plane -- see
  // ClusterRenderPipeline.h's own comment on m_WaterForward. Same borrowed-resource contract as
  // m_HeroTessellation above, resolved for the water entity; `waterMaterial` is the caller's own
  // materialTable.params[renderer::kWaterMaterialID] slot (populated by
  // GenerateShowcaseMaterialTable()'s own water-recipe override, see that function's own comment).
  {
    // Matches VulkanContext::kWaterEntityIndex exactly (the water plane, generated last -- see
    // that class' own comment) -- a plain literal here rather than a cross-file constant
    // reference, same convention kHeroEntityID above already establishes.
    constexpr uint32_t kWaterEntityID = 15u;
    const auto& entityRanges = m_SurfaceCache.GetEntityRanges();
    const auto waterRangeIt = entityRanges.find(kWaterEntityID);
    if (waterRangeIt == entityRanges.end()) {
      LOG_ERROR("[ClusterRenderPipeline] Water entity (meshID=15) has no Fallback Mesh draw range -- cannot initialize WaterForwardPass.");
      return false;
    }
    if (!m_WaterForward.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                            GICompositePass::kOutputFormat, createInfo.depthFormat,
                            createInfo.entityTransformBuffer,
                            createInfo.materialTable.params[kWaterMaterialID],
                            m_SurfaceCache.GetVertexBuffer(), m_SurfaceCache.GetIndexBuffer(),
                            waterRangeIt->second, kWaterEntityID,
                            m_SurfaceCacheRT.GetTLASHandle(), m_SurfaceCacheRT.GetDrawRangeBuffer(),
                            m_TraceContext, createInfo.renderExtent)) {
      LOG_ERROR("[ClusterRenderPipeline] Failed to initialize WaterForwardPass.");
      return false;
    }
  }

  // GPU particle system (particle_system_integration_plan.md): buffer/descriptor-set skeleton
  // (Subtask 1) + simulation (Subtask 2) + sort (Subtask 3) + billboard render + Lumen/VSM lighting
  // (Subtasks 4-5) pipelines -- see renderer::ParticleSystemPass's own class comment. Depends on
  // m_AtmosClimate (wind), m_GlobalSDF (collision clipmaps, both Init'd above at STEP 7),
  // m_Resolve (GBuffer depth copy for soft particles, Init'd far earlier at STEP 6),
  // m_VirtualShadowMap (sun shadow, STEP 7) and m_WorldProbes (indirect diffuse, just above) -- all
  // five already ready by this point. colorFormat/depthFormat match TransparentForwardPass::Init's
  // own call site immediately below: this pass draws onto the SAME m_GIComposite output image and
  // real depth-stencil buffer every other forward pass targets. RecordSimulate/RecordSort/
  // RecordDraw are all implemented but NOT yet called from RecordFrame (Subtask 6 wires that up),
  // so this Init() has no RecordFrame ordering consequence yet.
  if (!m_ParticleSystem.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                             m_AtmosClimate, m_GlobalSDF, m_Resolve, m_VirtualShadowMap, m_WorldProbes,
                             GICompositePass::kOutputFormat, createInfo.depthFormat)) {
    LOG_ERROR("[ClusterRenderPipeline] Failed to initialize ParticleSystemPass.");
    return false;
  }

  // Screen Trace GI -- linear screen-space depth raymarching falling back to the World Probe grid.
  m_ScreenTrace.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
                     createInfo.renderExtent, m_Resolve.GetOutputDepthView(), m_Resolve.GetOutputNormalView(),
                     m_Resolve.GetOutputColorView(), m_WorldProbes);

  // Final spatial denoiser: denoises m_ScreenTrace's own output GI image, guided by
  // m_Resolve's own G-buffer normal/depth -- both already Init'd (STEP 6 above).
  m_Denoiser.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
      createInfo.renderExtent, m_ScreenTrace.GetOutputView(), m_Resolve.GetOutputDepthView(),
      m_Resolve.GetOutputNormalView());

  // GI Composite: blends m_Resolve's direct lighting / reflections with the denoised indirect GI.
  m_GIComposite.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
      createInfo.renderExtent, m_Resolve.GetOutputColorView(), m_Denoiser.GetOutputView(),
      m_ScreenSpaceEffects.GetAOView(), m_Resolve.GetOutputDepthView(), m_WorldProbes);

  m_TAATSR.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
      m_RenderExtent, m_DisplayExtent,
      m_GIComposite.GetOutputView(), m_Resolve.GetOutputDepthView());

  // Phase PP3 (post-process stack roadmap): physically-derived Depth of Field, reading m_TAATSR's
  // own HDR output directly -- must Init before m_Bloom/m_PostProcess so its own GetOutputView()
  // already exists for their own source bindings (see DepthOfFieldPass's own class comment for why
  // DOF runs before Bloom).
  m_DepthOfField.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
      m_DisplayExtent, m_TAATSR.GetOutputView(), m_Resolve.GetOutputDepthView());

  // Phase PP2 (post-process stack roadmap): Bloom/Lens Flare/Anamorphic Flare/Lens Dirt, reading
  // m_DepthOfField's own output -- must Init before m_PostProcess so its own GetOutputView()
  // already exists for m_PostProcess.Init's own g_Bloom binding.
  m_Bloom.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
      m_DisplayExtent, m_DepthOfField.GetOutputView());

  // Phase PP1 (post-process stack roadmap): exposure/white-balance/color-correction/ACES/gamma --
  // linear HDR (m_DepthOfField's own output, itself sourced from m_TAATSR's TAA-resolved
  // R16G16B16A16_SFLOAT history -- see [13e]'s own comment) -> LDR swapchain-ready color.
  // `hdrColorView` is rewritten every frame in RecordFrame via UpdateDescriptorSets() (m_TAATSR's
  // own output view alternates between 2 ping-pong images every frame, so m_DepthOfField's own
  // output view identity is likewise effectively "per-frame" even though DepthOfFieldPass itself
  // owns a single fixed image -- the same reason renderer::TAATSRPass's own UpdateDescriptorSets
  // exists), not just bound once here. `depthView`/`refractionOffsetView` (Phase PP3) ARE just
  // bound once here -- both keep a fixed identity for this pipeline's entire lifetime (see
  // PostProcessPass::Init's own comment).
  m_PostProcess.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue,
      m_DisplayExtent, m_DepthOfField.GetOutputView(), m_Bloom.GetOutputView(),
      m_Resolve.GetOutputDepthView(), m_TransparentForward.GetRefractionOffsetView(),
      m_AtmosSky.GetSkyViewLUTView(), m_AtmosFog.GetIntegratedFogView(), m_AtmosClouds.GetCloudView());

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
  // Atmos weather system, Subtask 2: same deferred-wiring convention as SetGlobalSDFViews above --
  // see SDFRayMarchPass::SetAtmosSkyView's own comment.
  m_SDFRayMarch.SetAtmosSkyView(m_AtmosSky.GetSkyViewLUTView(), m_AtmosSky.GetLUTSampler());

  // Backs the ImGui "Buffer Viewer" dropdown -- see debug::DebugBufferViewPass's own class
  // comment. Sized to m_DisplayExtent (it's blitted to the swapchain the same way
  // m_PostProcess's own output is, not sized to any one candidate buffer's own resolution).
  m_DebugBufferView.Init(createInfo.device, createInfo.allocator, createInfo.commandPool, createInfo.queue, m_DisplayExtent);
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
  // outputColorFormat = SDFRayMarchPass::kOutputFormat (RGBA8_UNORM): the only remaining candidate
  // RecordDraw() target that ISN'T renderer::TAATSRPass's own HDR history buffer -- m_Resolve and
  // m_GIComposite are both HDR (rgba16f) now (see ClusterResolvePass::kOutputColorFormat's own
  // comment), so their overlay draws route through kHdrTargetFormat's pipeline below instead (see
  // the overlayTargetFormat selection a few hundred lines down in RecordFrame).
  m_DebugOverlay.Init(createInfo.device, createInfo.allocator,
                      createInfo.commandPool, createInfo.queue,
                      SDFRayMarchPass::kOutputFormat);
#endif

#ifndef NDEBUG
  // Phase 1 (Nanite advanced): cross-checks SPLINE_MAX_DEVIATION against the authored curve's true
  // worst-case displacement -- see ValidateSplineBounds()'s own comment.
  ValidateSplineBounds();
#endif

  LOG_INFO(
      std::format("[ClusterRenderPipeline] Initialized: {} clusters streamed "
                  "from cache ({} leaf candidates, {} logical pages).",
                  totalClusterCount, m_ClusterCount, maxLogicalPages));
  return true;
}

#ifndef NDEBUG
namespace {

    // C++ mirror of spline_deformation.glsl's HermiteEvaluate -- see that function's own comment
    // for the basis-function derivation. `outTangent` is the analytic derivative w.r.t. u, not
    // renormalized to arc length (only its direction is used by BuildBendFrame below).
    void HermiteEvaluateCPU(const maths::vec3& p0, const maths::vec3& t0, const maths::vec3& p1, const maths::vec3& t1,
        float u, maths::vec3& outPosition, maths::vec3& outTangent) {
        float u2 = u * u;
        float u3 = u2 * u;

        float h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
        float h10 = u3 - 2.0f * u2 + u;
        float h01 = -2.0f * u3 + 3.0f * u2;
        float h11 = u3 - u2;
        outPosition = p0 * h00 + t0 * h10 + p1 * h01 + t1 * h11;

        float dh00 = 6.0f * u2 - 6.0f * u;
        float dh10 = 3.0f * u2 - 4.0f * u + 1.0f;
        float dh01 = -6.0f * u2 + 6.0f * u;
        float dh11 = 3.0f * u2 - 2.0f * u;
        outTangent = p0 * dh00 + t0 * dh10 + p1 * dh01 + t1 * dh11;
    }

    // C++ mirror of spline_deformation.glsl's BuildBendFrame -- see that function's own comment.
    void BuildBendFrameCPU(const maths::vec3& tangentDir, maths::vec3& bendRight, maths::vec3& bendForward) {
        maths::vec3 reference{1.0f, 0.0f, 0.0f};
        if (std::abs(tangentDir.Dot(reference)) > 0.98f) {
            reference = maths::vec3{0.0f, 0.0f, 1.0f};
        }
        bendRight = tangentDir.Cross(reference).Normalize();
        bendForward = bendRight.Cross(tangentDir);
    }

    // C++ mirror of spline_deformation.glsl's ApplySplineDeformation -- see that function's own
    // comment. `controlPoints` must have exactly 4 entries (SPLINE_CONTROL_POINT_COUNT).
    maths::vec3 ApplySplineDeformationCPU(const maths::vec3& localPos, const std::array<SplineControlPoint, 4>& controlPoints) {
        constexpr float kSplineRestPoseHalfHeight = 0.7f; // Must match SPLINE_REST_POSE_HALF_HEIGHT (spline_deformation.glsl).
        float tNormalized = std::clamp((localPos.y + kSplineRestPoseHalfHeight) / (2.0f * kSplineRestPoseHalfHeight), 0.0f, 1.0f);
        float globalT = tNormalized * 3.0f; // SPLINE_CONTROL_POINT_COUNT - 1 == 3.

        int segmentIndex = static_cast<int>(std::floor(globalT));
        segmentIndex = std::clamp(segmentIndex, 0, 2); // SPLINE_CONTROL_POINT_COUNT - 2 == 2.
        float localU = globalT - static_cast<float>(segmentIndex);

        const SplineControlPoint& c0 = controlPoints[static_cast<size_t>(segmentIndex)];
        const SplineControlPoint& c1 = controlPoints[static_cast<size_t>(segmentIndex) + 1];

        maths::vec3 curvePos, curveTangent;
        HermiteEvaluateCPU(c0.position, c0.tangent, c1.position, c1.tangent, localU, curvePos, curveTangent);

        maths::vec3 bendRight, bendForward;
        BuildBendFrameCPU(curveTangent.Normalize(), bendRight, bendForward);

        return curvePos + bendRight * localPos.x + bendForward * localPos.z;
    }

} // namespace

void ClusterRenderPipeline::ValidateSplineBounds() const {
    // Must match SPLINE_MAX_DEVIATION (spline_deformation.glsl) exactly -- this function's whole
    // purpose is cross-checking that constant against the true worst-case displacement the authored
    // curve (kSplineControlPoints) actually produces, so it cannot read the constant FROM that same
    // file (no C++/GLSL shared-header mechanism exists in this codebase for #define constants).
    constexpr float kSplineMaxDeviation = 1.6f;
    // Tube (entity 6)'s own outer radius (see VulkanContext::GenerateGeometry's Radius1 = 0.7f
    // passed to GenerateTube, and geom_tube.comp's "outer radius" field) -- the cross-section extent
    // ApplySplineDeformation's bendRight/bendForward terms actually displace.
    constexpr float kTubeOuterRadius = 0.7f;
    constexpr float kTubeInnerRadius = 0.5f;
    constexpr uint32_t kCurveSteps = 1000;
    constexpr uint32_t kAngleSteps = 16;

    float maxDisplacement = 0.0f;
    maths::vec3 worstLocalPos{};
    float worstT = 0.0f;

    for (uint32_t step = 0; step <= kCurveSteps; ++step) {
        float tNormalized = static_cast<float>(step) / static_cast<float>(kCurveSteps);
        float y = -0.7f + tNormalized * 1.4f; // SPLINE_REST_POSE_HALF_HEIGHT range [-0.7, 0.7].

        // Sample the pipe's actual cross-section: both radii (outer wall + inner wall, the hollow
        // pipe's two concentric rings -- see geom_tube.comp) at every angle, plus the degenerate
        // r=0 centerline point (a lower bound, included for completeness/symmetry, not expected to
        // ever be the worst case).
        for (uint32_t angleStep = 0; angleStep < kAngleSteps; ++angleStep) {
            float angle = (static_cast<float>(angleStep) / static_cast<float>(kAngleSteps)) * 6.28318530718f;
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);

            for (float radius : { 0.0f, kTubeInnerRadius, kTubeOuterRadius }) {
                maths::vec3 localPos{cosA * radius, y, sinA * radius};
                maths::vec3 bentPos = ApplySplineDeformationCPU(localPos, kSplineControlPoints);
                float displacement = (bentPos - localPos).Length();
                if (displacement > maxDisplacement) {
                    maxDisplacement = displacement;
                    worstLocalPos = localPos;
                    worstT = tNormalized;
                }
            }
        }
    }

    if (maxDisplacement > kSplineMaxDeviation) {
        LOG_ERROR(std::format(
            "[ClusterRenderPipeline] ValidateSplineBounds: true worst-case spline displacement {:.4f} "
            "EXCEEDS SPLINE_MAX_DEVIATION ({:.4f}) at t={:.4f}, localPos=({:.4f}, {:.4f}, {:.4f}) -- "
            "the bounds-inflation contract (ClusterDAGScreenError.comp/ClusterLODCompact.comp) is "
            "violated, bent geometry can pop through a bounding volume culling already decided was "
            "safe. Increase SPLINE_MAX_DEVIATION in spline_deformation.glsl to at least this value.",
            maxDisplacement, kSplineMaxDeviation, worstT, worstLocalPos.x, worstLocalPos.y, worstLocalPos.z));
    } else {
        LOG_INFO(std::format(
            "[ClusterRenderPipeline] ValidateSplineBounds: true worst-case spline displacement {:.4f} "
            "is within SPLINE_MAX_DEVIATION ({:.4f}). OK.", maxDisplacement, kSplineMaxDeviation));
    }
}
#endif

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
  m_DebugBufferView.Shutdown();
#endif
  m_AtmosClouds.Shutdown();
  m_AtmosFog.Shutdown();
  m_MegaLights.Shutdown();
  m_ScreenSpaceEffects.Shutdown();
  m_Reflection.Shutdown();
  m_ScreenTrace.Shutdown();
  m_GIComposite.Shutdown();
  m_Denoiser.Shutdown();
  // Phase PP1/PP2 (post-process stack roadmap): reverse Init() order (m_TAATSR -> m_Bloom ->
  // m_PostProcess), so shut down m_PostProcess and m_Bloom before m_TAATSR below. This was missing
  // for m_PostProcess ever since Phase PP1 (only self-shutdown via its own Init()'s defensive
  // top-of-function Shutdown() call, never explicitly on a real pipeline teardown) -- fixed here
  // alongside adding m_Bloom's own equivalent call.
  m_PostProcess.Shutdown();
  m_Bloom.Shutdown();
  m_DepthOfField.Shutdown();
  m_TAATSR.Shutdown();
  m_WorldProbes.Shutdown();
  m_GIInject.Shutdown();
  m_SurfaceCacheRT.Shutdown();
  m_TraceContext.Shutdown();
  m_GlobalSDF.Shutdown();
  m_AtmosClimate.Shutdown();
  m_AtmosSky.Shutdown();
  // Deliberately NOT calling m_LoadingManager.Shutdown() here: this Shutdown() method runs
  // defensively as the very first line of Init() too (see the top of this function), and
  // core::LoadingManager's worker threads are meant to live for this whole pipeline object's
  // lifetime, not be torn down and left dead on every Init() reset -- doing so once caused a real
  // hang (GlobalSDFPass::Init()'s WaitIdle() blocking forever on a pool whose workers had already
  // been joined and cleared before it ever submitted a job). core::LoadingManager's own destructor
  // joins its threads exactly once, when this pipeline object (and its m_LoadingManager member)
  // is itself destructed -- that is the only point its worker pool should ever die.
  m_SurfaceCache.Shutdown();
  m_VirtualShadowMap.Shutdown();
  // Step 4: shut down in the reverse of Init()'s own dependency order (m_VTStreaming borrows
  // &m_VTManager, m_VTRenderPass borrows it too -- neither is used again after this point this
  // frame/session, so plain declaration-adjacent ordering is safe here, unlike Init()'s own
  // pointer-validity requirement).
  m_VTStreaming.Shutdown();
  m_VTRenderPass.Shutdown();
  m_VTManager.Shutdown();
  m_PrevViewProj = maths::mat4{};
  m_HasPrevViewProj = false;
  m_FrameIndex = 0;
#ifndef NDEBUG
  m_DebugTraceMode = 0;
#endif

  m_HeroTessellation.Shutdown();
  m_WaterForward.Shutdown();
  m_ParticleSystem.Shutdown();
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
  m_SplineControlPointsBuffer.Destroy();

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

#ifndef NDEBUG
// Index->buffer table for the ImGui "Buffer Viewer" dropdown (config::debugview::
// SELECTED_BUFFER_INDEX). main.cpp's own ImGui::Combo item array MUST list these same 15 entries
// in this exact order -- there is no shared enum between the two translation units, just this
// comment as the single source of truth both sides are hand-kept in sync with.
//   0  = Off (normal final composite, this function is never called)
//   1  = Resolve: Direct Color (HDR)
//   2  = Resolve: World Normal
//   3  = Resolve: Depth
//   4  = Resolve: Albedo
//   5  = Resolve: Roughness/Metallic
//   6  = Reflection: Hit Mask
//   7  = Ambient Occlusion (GTAO)
//   8  = Bloom
//   9  = TAA/TSR Output
//   10 = Depth of Field Output
//   11 = Screen Trace GI
//   12 = Denoised GI (A-Trous)
//   13 = GI Composite
//   14 = Final Composite (Post-Process, pre-ImGui)
void ClusterRenderPipeline::RecordDebugBufferView(VkCommandBuffer cmd) {
    VkImageView sourceView = m_Resolve.GetOutputColorView();
    debug::DebugBufferViewPass::VisualizationMode mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap;

    switch (config::debugview::SELECTED_BUFFER_INDEX) {
        case 1: sourceView = m_Resolve.GetOutputColorView(); mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap; break;
        case 2: sourceView = m_Resolve.GetOutputNormalView(); mode = debug::DebugBufferViewPass::VisualizationMode::kOctNormal; break;
        case 3: sourceView = m_Resolve.GetOutputDepthView(); mode = debug::DebugBufferViewPass::VisualizationMode::kGrayscale; break;
        case 4: sourceView = m_Resolve.GetOutputAlbedoView(); mode = debug::DebugBufferViewPass::VisualizationMode::kPassthrough; break;
        case 5: sourceView = m_Resolve.GetOutputRoughnessMetallicView(); mode = debug::DebugBufferViewPass::VisualizationMode::kGrayscale; break;
        case 6: sourceView = m_Reflection.GetHitMaskView(); mode = debug::DebugBufferViewPass::VisualizationMode::kGrayscale; break;
        case 7: sourceView = m_ScreenSpaceEffects.GetAOView(); mode = debug::DebugBufferViewPass::VisualizationMode::kGrayscale; break;
        case 8: sourceView = m_Bloom.GetOutputView(); mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap; break;
        case 9: sourceView = m_TAATSR.GetOutputView(); mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap; break;
        case 10: sourceView = m_DepthOfField.GetOutputView(); mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap; break;
        case 11: sourceView = m_ScreenTrace.GetOutputView(); mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap; break;
        case 12: sourceView = m_Denoiser.GetOutputView(); mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap; break;
        case 13: sourceView = m_GIComposite.GetOutputView(); mode = debug::DebugBufferViewPass::VisualizationMode::kTonemap; break;
        case 14: sourceView = m_PostProcess.GetOutputView(); mode = debug::DebugBufferViewPass::VisualizationMode::kPassthrough; break;
        default: break; // Unknown index -- falls back to the Resolve color default set above.
    }

    m_DebugBufferView.RecordView(cmd, sourceView, mode);
}
#endif

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

void ClusterRenderPipeline::RecordSurfaceCacheOwnershipTransfer(VkCommandBuffer cmd, bool isRelease,
    uint32_t srcFamily, uint32_t dstFamily, VkPipelineStageFlags2 activeStageMask,
    VkAccessFlags2 activeImageAccessMask, VkAccessFlags2 activeBufferAccessMask) {
  if (!m_AsyncComputeAvailableThisBuild) {
    return; // Same queue family as the graphics/async-compute consumer -- nothing to transfer. See
             // this method's own header comment.
  }

  // Per the Vulkan spec, a queue family ownership transfer's RELEASE operation's dstAccessMask/
  // dstStageMask are ignored (the acquiring queue's own barrier is what actually makes the data
  // visible to it), and the matching ACQUIRE's srcAccessMask/srcStageMask are likewise ignored --
  // see GpuGeometryPagePool::ReleasePhysicalPoolOwnership/AcquirePhysicalPoolOwnership's own
  // comments for the same reasoning applied to a plain buffer. `activeStageMask`/
  // `activeImageAccessMask`/`activeBufferAccessMask` are therefore routed to src* on the release
  // side and dst* on the acquire side; the unused triple is left NONE/0 either way.
  VkPipelineStageFlags2 releaseStage = isRelease ? activeStageMask : VK_PIPELINE_STAGE_2_NONE;
  VkPipelineStageFlags2 acquireStage = isRelease ? VK_PIPELINE_STAGE_2_NONE : activeStageMask;
  VkAccessFlags2 releaseImageAccess = isRelease ? activeImageAccessMask : VK_ACCESS_2_NONE;
  VkAccessFlags2 acquireImageAccess = isRelease ? VK_ACCESS_2_NONE : activeImageAccessMask;
  VkAccessFlags2 releaseBufferAccess = isRelease ? activeBufferAccessMask : VK_ACCESS_2_NONE;
  VkAccessFlags2 acquireBufferAccess = isRelease ? VK_ACCESS_2_NONE : activeBufferAccessMask;

  // The 5 atlas images SurfaceCacheGIInjectPass::Init actually binds (confirmed by reading that
  // method's own descriptor writes) -- Albedo/Normal/Emissive/Radiance/WorldPos. NOT
  // DirectLighting: GIInject never binds it (only RecordCapture writes it, and only downstream
  // consumers other than GIInject ever sample it), so it stays on the graphics queue family the
  // whole time and needs no transfer at all.
  VkImage images[5] = {
      m_SurfaceCache.GetAlbedoImage(), m_SurfaceCache.GetNormalImage(), m_SurfaceCache.GetEmissiveImage(),
      m_SurfaceCache.GetRadianceImage(), m_SurfaceCache.GetWorldPosImage()
  };
  VkImageMemoryBarrier2 imageBarriers[5]{};
  for (uint32_t i = 0; i < 5; ++i) {
    imageBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarriers[i].srcStageMask = releaseStage;
    imageBarriers[i].srcAccessMask = releaseImageAccess;
    imageBarriers[i].dstStageMask = acquireStage;
    imageBarriers[i].dstAccessMask = acquireImageAccess;
    // Layout unchanged -- these atlas images live permanently in GENERAL for their entire lifetime
    // (see SurfaceCachePass' own class comment's "Atlas layout convention" section), exactly like
    // GpuGeometryPagePool's own buffers never change layout across their ownership transfer either.
    imageBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarriers[i].srcQueueFamilyIndex = srcFamily;
    imageBarriers[i].dstQueueFamilyIndex = dstFamily;
    imageBarriers[i].image = images[i];
    imageBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
  }

  // The 4 buffers RecordRefreshTLAS (rebuilds the TLAS) / GIInject's own TraceHWRT (reads the
  // Fallback Mesh vertex/index buffers + the draw-range table to resolve a ray hit) touch every
  // frame on whichever queue they now run on -- see this method's own header comment for why
  // SurfaceCacheRayTracingPass::TlasRefitResources' internal scratch/instance buffers are
  // deliberately excluded.
  VkBuffer buffers[4] = {
      m_SurfaceCache.GetVertexBuffer(), m_SurfaceCache.GetIndexBuffer(),
      m_SurfaceCacheRT.GetDrawRangeBuffer(), m_SurfaceCacheRT.GetTLASBufferHandle()
  };
  VkBufferMemoryBarrier2 bufferBarriers[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    bufferBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    bufferBarriers[i].srcStageMask = releaseStage;
    bufferBarriers[i].srcAccessMask = releaseBufferAccess;
    bufferBarriers[i].dstStageMask = acquireStage;
    bufferBarriers[i].dstAccessMask = acquireBufferAccess;
    bufferBarriers[i].srcQueueFamilyIndex = srcFamily;
    bufferBarriers[i].dstQueueFamilyIndex = dstFamily;
    bufferBarriers[i].buffer = buffers[i];
    bufferBarriers[i].offset = 0;
    bufferBarriers[i].size = VK_WHOLE_SIZE;
  }

  VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  depInfo.imageMemoryBarrierCount = 5;
  depInfo.pImageMemoryBarriers = imageBarriers;
  depInfo.bufferMemoryBarrierCount = 4;
  depInfo.pBufferMemoryBarriers = bufferBarriers;
  vkCmdPipelineBarrier2(cmd, &depInfo);
}

void ClusterRenderPipeline::RecordFrameEarly(VkCommandBuffer cmdEarly,
                                              const CameraPushConstants &camera,
                                              const maths::vec3 &cameraPositionWorld,
                                              const CameraFrameInfo &cameraFrameInfo,
                                              float globalTimeSeconds,
                                              const core::EntityTransformCPU *entityTransformsCPU) {
  assert(m_ClusterCount > 0 && "RecordFrameEarly called before a successful Init");

  CameraPushConstants cameraCopy = camera;
  float jitterX = 0.0f;
  float jitterY = 0.0f;

  bool taatsrEnabled = true;
#ifndef NDEBUG
  taatsrEnabled = m_DebugTAATSREnabled;
#endif

  // Only apply camera jitter if TAA/TSR is enabled
  if (taatsrEnabled) {
      // Halton jitter sequence centered around 0 in pixel space [-0.5, 0.5]
      maths::vec2 rawJitter = Halton23(m_FrameIndex % config::temporal::JITTER_FRAME_COUNT);
      jitterX = rawJitter.x - 0.5f;
      jitterY = rawJitter.y - 0.5f;


      // Apply subpixel jitter to projection matrix (row 0, col 2 and row 1, col 2 in column-major)
      float deltaX = (jitterX * 2.0f) / static_cast<float>(m_RenderExtent.width);
      float deltaY = (jitterY * 2.0f) / static_cast<float>(m_RenderExtent.height);
      cameraCopy.proj.m[8] += deltaX;
      cameraCopy.proj.m[9] += deltaY;
  }

  // Every stage of this frame consumes the SAME combined matrix -- this is what
  // makes the resolve pass's screen-space triangle re-projection bit-identical
  // to the rasterizers'.
  maths::mat4 viewProj = cameraCopy.proj * cameraCopy.view;
  maths::mat4 invViewProj = viewProj.Inverse();

  ClusterCullViewParams viewParams{};
  viewParams.frustumPlanes = ExtractFrustumPlanes(viewProj);
  viewParams.cameraPositionWorld = cameraPositionWorld;

  // abs(proj.m[5]): PerspectiveVulkan stores -1/tan(fovY/2) there (Y flip), and
  // the screen size classification needs the positive scale.
  float projScaleY = std::abs(cameraCopy.proj.m[5]);


  // SWRT/HWRT back-end shared by m_GIInject ([1z] below) and m_WorldProbes' own trace pass ([12c]
  // in RecordFrameLate) -- debug-only toggle (main.cpp's 'T' key via SetDebugTraceMode), Release
  // always uses HWRT (no debug toggle to switch back to SWRT).
#ifndef NDEBUG
  uint32_t traceMode = m_DebugTraceMode;
#else
  uint32_t traceMode = 1u;
#endif

  // Phase 2 (Lumen advanced roadmap): this frame's async-compute routing decision -- read by
  // RecordAsyncCompute() (whether to record anything at all) and RecordFrameLate() (whether to
  // ACQUIRE the atlas back), both later this same frame -- see m_FrameScratch's own comment for why
  // this is now stored there instead of an outer-function-scope local. Forced OFF whenever
  // traceMode is SWRT (0): moving GIInject's SWRT path to the async queue would additionally need
  // ownership-transfer barriers for every per-entity Global SDF image it samples
  // (TraceMeshSDFScene), which only the Debug SWRT toggle ever exercises -- not worth doubling the
  // barrier surface for a debug-only configuration when Release always uses HWRT anyway (see
  // SetDebugAsyncComputeEnabled's own header comment). The Debug toggle additionally gates it off
  // entirely for staged bring-up (false = prove the CPU/struct plumbing alone with everything still
  // graphics-queue-serialized; true = exercise the real cross-queue barrier design).
  bool useAsyncCompute = (traceMode == 1u);
#ifndef NDEBUG
  useAsyncCompute = useAsyncCompute && m_DebugAsyncComputeEnabled;
#endif

  bool radiosityEnabled = true;
#ifndef NDEBUG
  radiosityEnabled = m_DebugRadiosityEnabled;
#endif

  // Persist everything RecordAsyncCompute()/RecordFrameMid()/RecordFrameLate() need later THIS
  // SAME frame -- see FrameScratch's own comment. Written exactly once, here, at the top of the
  // frame's first call.
  m_FrameScratch.camera = camera;
  m_FrameScratch.cameraCopy = cameraCopy;
  m_FrameScratch.cameraPositionWorld = cameraPositionWorld;
  m_FrameScratch.cameraFrameInfo = cameraFrameInfo;
  m_FrameScratch.globalTimeSeconds = globalTimeSeconds;
  m_FrameScratch.entityTransformsCPU = entityTransformsCPU;
  m_FrameScratch.viewProj = viewProj;
  m_FrameScratch.invViewProj = invViewProj;
  m_FrameScratch.viewParams = viewParams;
  m_FrameScratch.projScaleY = projScaleY;
  m_FrameScratch.jitterX = jitterX;
  m_FrameScratch.jitterY = jitterY;
  m_FrameScratch.taatsrEnabled = taatsrEnabled;
  m_FrameScratch.traceMode = traceMode;
  m_FrameScratch.useAsyncCompute = useAsyncCompute;
  m_FrameScratch.radiosityEnabled = radiosityEnabled;

  // =========================================================================================
  // [1y] Atmos weather system, Subtask 1: refresh AtmosGlobalsUBO (wind, Magnus-Tetens dew point /
  // LCL height) from this frame's config::atmos::* knobs. No consumer yet (Phase 1 -- see
  // AtmosClimatePass.h's own class comment); placed immediately before [1z] so a future Fog/Cloud
  // consumer added inside that same block always sees an already-current buffer this frame.
  // =========================================================================================
  m_AtmosClimate.RecordUpdate(cmdEarly, globalTimeSeconds);
  // Atmos weather system, Subtask 2: Sky-View LUT refresh -- see AtmosSkyPass::RecordUpdate's own
  // comment for the Transmittance/Multi-Scattering dirty-tracking policy. Sun direction/intensity
  // sourced the same way m_Resolve's own sun uniform below is (m_SceneLights.sun).
  m_AtmosSky.RecordUpdate(cmdEarly, m_SceneLights.sun.direction, m_SceneLights.sun.intensity);

  // =========================================================================================
  // [1z] Lumen-style GI infrastructure: Virtual Shadow Map page requests/renders (Phase 3) ->
  // Surface Cache capture -> Global SDF clipmap streaming -> (Debug-only) SDF ray march debug
  // visualization. Independent of the Nanite VisBuffer/HZB machinery in RecordFrameMid (no shared
  // images, no fence contention -- see ClusterRenderPipeline.h's own class-comment addendum), so
  // it can run anywhere in the frame; placed in this class' own first per-frame call, cmdEarly, so
  // its release below can signal VulkanContext::GetAsyncComputeCanStartSemaphore() as early as
  // possible, letting the async-compute queue's work start overlapping with RecordFrameMid()'s own
  // GPU work as soon as the GPU actually reaches it.
  //
  // Phase 2 (Lumen advanced roadmap) fix, 2026-07-17: this frame's own RELEASE (below, when
  // useAsyncCompute) now happens HERE, immediately after RecordCapture/GlobalSDF.RecordUpdate --
  // NOT after every graphics-side consumer has read the atlas, the way the old single-command-
  // buffer design had to (see this class' own header comment for the full root-cause). The matching
  // ACQUIRE now happens at the very start of RecordFrameLate(), THIS SAME FRAME (not the top of
  // NEXT frame's own [1z] block the way the old m_AtlasOwnedByAsync cross-frame design worked) --
  // RecordFrameMid() (in between) never reads the Surface Cache atlas/TLAS at all (confirmed by
  // reading its own passes' descriptor bindings), so releasing this early strands no graphics-side
  // consumer.
  // =========================================================================================
  {
    // Sun direction is fixed for now (m_SceneLights' own default, see LightingTypes.h) -- a
    // future day/night system would rotate it per frame instead.
    const maths::vec3 sunDirection = m_SceneLights.sun.direction;

    // 1. Virtual Shadow Maps first: SurfaceCachePass's/m_Resolve's shadow lookups (below and at
    // [12]) need THIS frame's VSM view-projection matrices + any pages rendered this frame already
    // visible -- RecordBeginFrame()'s own trailing barrier (when it renders any page) covers that.
    // See VirtualShadowMapPass's own class comment for the full one-frame-lag feedback contract.
    m_VirtualShadowMap.RecordBeginFrame(cmdEarly, sunDirection, m_SceneLights, cameraFrameInfo.position, entityTransformsCPU);

    // 1b. Virtual Texture streaming: reads back LAST frame's page-miss feedback (m_Resolve's own
    // ClusterResolve.comp/ClusterResolveBinned.comp VT sampling call, see SetVirtualTexture()'s own
    // comment), issues new async tile reads, and drains/uploads completed ones -- must run before
    // m_Resolve's own dispatch in RecordFrameMid() (same one-frame-lag contract as
    // VirtualShadowMapPass's own RecordBeginFrame, see VirtualTextureStreamingCoordinator's own
    // class comment).
    m_VTStreaming.RecordBeginFrame(cmdEarly);

    // 2. Surface Cache: feed this frame's light data before the visibility-driven capture draws --
    // shadow lookups now read renderer::VirtualShadowMapPass's own UBOs directly (bound once via
    // SetVirtualShadowMap()), no per-frame light-view-proj parameter needed here anymore. STAYS on
    // the graphics queue (Feature 1's plan: only TLAS refit + GIInject move) -- this is a render
    // pass (vkCmdBeginRendering), not a compute dispatch, and async-compute queues on most hardware
    // cannot execute graphics-pipeline work at all.
    m_SurfaceCache.UpdateLighting(m_SceneLights);
    m_SurfaceCache.UpdateVisibility(cameraFrameInfo.position, cameraFrameInfo.forward,
                                    maths::vec3{0.0f, 1.0f, 0.0f}, cameraFrameInfo.fovYRadians,
                                    cameraFrameInfo.aspectRatio, cameraFrameInfo.nearZ,
                                    cameraFrameInfo.farZ);
    m_SurfaceCache.RecordCapture(cmdEarly, cameraFrameInfo.position, entityTransformsCPU);

    // 3. Global SDF clipmap streaming, from this frame's camera position. STAYS on the graphics
    // queue (Feature 1's plan explicitly excludes it -- GIInject's HWRT path, the only mode ever
    // moved to async, does not sample the Global SDF at all).
    m_GlobalSDF.RecordUpdate(cmdEarly, cameraFrameInfo.position, entityTransformsCPU);

    // 4. Secondary-bounce injection into m_SurfaceCache's own radiance atlas + the TLAS refit that
    // must precede it (Phase 4 integration) -- `useAsyncCompute` (m_FrameScratch, see its own
    // comment above) picks between two mutually exclusive paths:
    //   - false: the FALLBACK path -- TLAS refit + the radiosity bounce loop run right here, fully
    //     graphics-queue-serialized, exactly as before Phase 2 (the pre-Phase-2 behavior). Correct
    //     to do inline here specifically (not deferred to RecordAsyncCompute(), which will no-op)
    //     since nothing needs to hand anything off to a different queue in this case.
    //   - true: both are skipped here and instead recorded, once, by RecordAsyncCompute() into
    //     asyncComputeCmd -- see this class' own header comment for the full redesign. The RELEASE
    //     immediately below (this same `if`'s `else` branch) hands the atlas off for that.
    if (!useAsyncCompute) {
      // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): per-frame TLAS
      // refit so ray-traced GI/reflections see this frame's entity rotations -- must run before
      // anything below traces against m_SurfaceCacheRT's TLAS (radiosity injection right below,
      // and any later HWRT consumer this same frame). A no-op rebuild (identity transforms)
      // whenever config::ENTITY_SELF_ROTATION_ENABLED is off -- see RecordRefreshTLAS's own comment.
      m_SurfaceCacheRT.RecordRefreshTLAS(cmdEarly, entityTransformsCPU);

      if (radiosityEnabled) {
        for (uint32_t bounce = 0; bounce < kRadiosityBounceCount; ++bounce) {
          m_GIInject.RecordInject(cmdEarly, m_TraceContext, m_SurfaceCache, traceMode, m_SceneLights.sun.direction);

          // Unlike SurfaceCachePass::RecordCapture/VirtualShadowMapPass::RecordBeginFrame/
          // GlobalSDFPass::RecordUpdate, SurfaceCacheGIInjectPass::RecordInject does NOT end with
          // its own trailing barrier -- its read-modify-write of the radiance atlas (imageLoad/
          // imageStore, STORAGE access) must be made visible here, explicitly, before the NEXT
          // bounce's own RecordInject call samples that same atlas (surface_cache_sampling.glsl's
          // g_SurfaceCacheRadiance) -- and, after the loop's final iteration, before m_ScreenTrace's
          // own near-field hit samples and m_WorldProbes' own trace pass do the same later this
          // frame (RecordFrameLate()).
          VkMemoryBarrier2 giInjectBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
          giInjectBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          giInjectBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
          giInjectBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
          giInjectBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
          VkDependencyInfo giInjectDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
          giInjectDepInfo.memoryBarrierCount = 1;
          giInjectDepInfo.pMemoryBarriers = &giInjectBarrier;
          vkCmdPipelineBarrier2(cmdEarly, &giInjectDepInfo);
        }
      }
    } else {
      // RELEASE (graphics -> async-compute), moved HERE by the Phase 2 fix -- see this block's own
      // header comment above for why this timing is correct. Unlike the OLD design's LATE release
      // (which had to cover everything a full frame's worth of graphics-side consumers had already
      // done -- RecordCapture's writes AND every later COMPUTE_SHADER/RAY_TRACING_SHADER_BIT_KHR
      // read from Reflection/World Probes/the forward passes), this EARLY release only needs to
      // cover what has ACTUALLY happened to these 9 resources by THIS point in the frame:
      // RecordCapture's own COLOR_ATTACHMENT_OUTPUT write to the 5 atlas images + its VERTEX_INPUT
      // read of the Fallback Mesh vertex/index buffers (SurfaceCachePass::RecordCapture's own
      // vkCmdBindVertexBuffers/vkCmdBindIndexBuffer/vkCmdDrawIndexed) -- GlobalSDF.RecordUpdate
      // touches none of these 9 resources at all, and RefreshTLAS/GIInject haven't run yet this
      // frame (deferred to RecordAsyncCompute() below).
      RecordSurfaceCacheOwnershipTransfer(cmdEarly, /*isRelease=*/true,
          m_GraphicsQueueFamilyIndex, m_AsyncComputeQueueFamilyIndex,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
          VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT);
    }

#ifndef NDEBUG
    // GlobalSDFPass::RecordUpdate's own trailing barrier only extends visibility to
    // SHADER_STORAGE_READ/WRITE (its own compositing dispatches read/write the clipmap via
    // imageLoad/imageStore) -- SDFRayMarchPass instead samples it through a COMBINED IMAGE
    // SAMPLER (SetGlobalSDFViews), which needs SHADER_SAMPLED_READ, a distinct access flag the
    // barrier above does not cover. Same "STORAGE_WRITE -> SAMPLED_READ" extension this class's
    // own HZB rebuilds already need (see the barriers around m_HZB.Generate() in RecordFrameMid).
    VkMemoryBarrier2 globalSDFToSampledBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    globalSDFToSampledBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    globalSDFToSampledBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    globalSDFToSampledBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    globalSDFToSampledBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo globalSDFToSampledDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    globalSDFToSampledDep.memoryBarrierCount = 1;
    globalSDFToSampledDep.pMemoryBarriers = &globalSDFToSampledBarrier;
    vkCmdPipelineBarrier2(cmdEarly, &globalSDFToSampledDep);

    // 4. Two-tier SDF ray march DEBUG VISUALIZATION -- only its OUTPUT IMAGE is ever displayed
    // (via the DEBUG_VIEW_GLOBAL_SDF blit-swap in RecordFrameLate; DEBUG_VIEW_LUMEN sources
    // m_GIComposite's own output instead, see that blit-swap's own comment), but it is always
    // recorded so that view shows this frame's state the instant it's selected, with no one-frame
    // lag. `coarseOnly` selects DEBUG_VIEW_GLOBAL_SDF's own coarse-clipmap-only trace (see
    // SDFRayMarch.comp's own comment on why that's a distinct, useful signal from the full two-tier
    // march); any other view mode gets the normal full march, which costs nothing extra since this
    // dispatch runs either way.
    m_SDFRayMarch.RecordRayMarch(cmdEarly, m_GlobalSDF, cameraFrameInfo.position, cameraFrameInfo.forward,
                                 maths::vec3{0.0f, 1.0f, 0.0f}, cameraFrameInfo.fovYRadians,
                                 cameraFrameInfo.aspectRatio, cameraFrameInfo.nearZ, cameraFrameInfo.farZ,
                                 cameraCopy.debugViewMode == DEBUG_VIEW_GLOBAL_SDF, m_SceneLights.sun.direction);
#endif
  }
}

// =========================================================================================
// RecordAsyncCompute() -- see ClusterRenderPipeline.h's own declaration-site comment and this
// class' header comment for the full redesign. Content below is otherwise UNCHANGED from the old
// design's [1z2] block (just relocated into its own function and re-timed relative to the rest of
// the frame -- RecordSurfaceCacheOwnershipTransfer's own barrier logic is untouched per this fix's
// own constraints).
// =========================================================================================
void ClusterRenderPipeline::RecordAsyncCompute(VkCommandBuffer asyncComputeCmd) {
  if (!m_FrameScratch.useAsyncCompute) {
    return; // Fallback path already ran fully graphics-queue-serialized in RecordFrameEarly().
  }

  const uint32_t traceMode = m_FrameScratch.traceMode;
  const bool radiosityEnabled = m_FrameScratch.radiosityEnabled;
  const core::EntityTransformCPU* entityTransformsCPU = m_FrameScratch.entityTransformsCPU;

  // ACQUIRE (by async-compute): the graphics queue's own RecordFrameEarly() already released these
  // 9 resources (RecordSurfaceCacheOwnershipTransfer's own release half) into
  // VulkanContext::GetAsyncComputeCanStartSemaphore()'s signal, which main.cpp's per-frame
  // asyncComputeCmd submission waits on before this command buffer's own execution reaches this
  // barrier's dstStageMask -- see RecordSurfaceCacheOwnershipTransfer()'s own comment for exactly
  // which 5 images + 4 buffers this covers.
  RecordSurfaceCacheOwnershipTransfer(asyncComputeCmd, /*isRelease=*/false,
      m_GraphicsQueueFamilyIndex, m_AsyncComputeQueueFamilyIndex,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
          VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

  // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): per-frame TLAS refit,
  // forced HWRT-only here (traceMode == 1u whenever useAsyncCompute, see RecordFrameEarly()'s own
  // useAsyncCompute derivation) -- must run before the bounce loop below traces against it.
  m_SurfaceCacheRT.RecordRefreshTLAS(asyncComputeCmd, entityTransformsCPU);

  if (radiosityEnabled) {
    for (uint32_t bounce = 0; bounce < kRadiosityBounceCount; ++bounce) {
      m_GIInject.RecordInject(asyncComputeCmd, m_TraceContext, m_SurfaceCache, traceMode, m_SceneLights.sun.direction);

      // Intra-queue (asyncComputeCmd both sides), so purely the usual execution/memory
      // dependency -- no queue-family-ownership concern here. Same rationale as the fallback
      // path's own identical barrier in RecordFrameEarly().
      VkMemoryBarrier2 giInjectBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      giInjectBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      giInjectBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
      giInjectBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
      giInjectBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
      VkDependencyInfo giInjectDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      giInjectDepInfo.memoryBarrierCount = 1;
      giInjectDepInfo.pMemoryBarriers = &giInjectBarrier;
      vkCmdPipelineBarrier2(asyncComputeCmd, &giInjectDepInfo);
    }
  }

  // RELEASE (async-compute -> graphics): makes the bounce loop's Radiance writes + the TLAS
  // refit's rebuild available back to the graphics queue family -- consumed by THIS SAME frame's
  // own RecordFrameLate() ACQUIRE (a same-frame hand-off now, not the old design's one-frame
  // pipeline lag -- see this class' own header comment).
  RecordSurfaceCacheOwnershipTransfer(asyncComputeCmd, /*isRelease=*/true,
      m_AsyncComputeQueueFamilyIndex, m_GraphicsQueueFamilyIndex,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
          VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  // Signals m_AsyncComputeFinishedSemaphore as part of asyncComputeCmd's own submission completing
  // (main.cpp) -- waited on by THIS SAME frame's cmdLate submission before RecordFrameLate()'s own
  // ACQUIRE (above, mirrored) is allowed to execute.
}

void ClusterRenderPipeline::RecordFrameMid(VkCommandBuffer cmdMid, VkCommandBuffer transferCmd) {
  const CameraPushConstants& camera = m_FrameScratch.camera;
  const CameraPushConstants& cameraCopy = m_FrameScratch.cameraCopy;
  const maths::vec3& cameraPositionWorld = m_FrameScratch.cameraPositionWorld;
  const maths::mat4& viewProj = m_FrameScratch.viewProj;
  const ClusterCullViewParams& viewParams = m_FrameScratch.viewParams;
  const float projScaleY = m_FrameScratch.projScaleY;
  const float globalTimeSeconds = m_FrameScratch.globalTimeSeconds;

  // =========================================================================================
  // [1] Per-frame worklist clears. Each Record*() carries its own
  // CLEAR->COMPUTE barrier, so the culling/raster dispatches below can never
  // read a stale counter.
  // =========================================================================================
  m_OcclusionCulling.RecordClearFrame(cmdMid);
  m_SoftwareRaster.RecordClear(cmdMid);
  m_LODSelection.RecordClear(cmdMid);

  // =========================================================================================
  // [1a] Async streaming triage: read back LAST frame's residency misses (captured by this
  // frame's own [1b] readback below, one frame ago), submit new disk reads, and bind/decompress
  // whatever completed since last frame -- see GeometryStreamingCoordinator's own class comment.
  // Recorded before the LOD cut below so any page bound this frame is already resident by the
  // time this frame's own residency checks (ClusterLODResidencyFallback.comp/ClusterLODCompact
  // .comp) run. Confirmed (by reading ClusterOcclusionCullingPass/ClusterLODSelectionPass/
  // ClusterHardwareRasterPass/ClusterSoftwareRasterPass/ClusterResolvePass/ClusterShadingBinPass'
  // own descriptor bindings) that nothing in this whole cmdMid command buffer samples the Surface
  // Cache atlas images, the TLAS, or Global SDF -- this geometry-streaming/culling/raster/resolve
  // pipeline is entirely independent of the GI work RecordFrameEarly()/RecordAsyncCompute()
  // record, hence needs no wait on the async-compute queue's own submission. It DOES need
  // GetTransferFinishedSemaphore() (main.cpp) -- this method is what actually records the geometry
  // page pool's own ownership-transfer ACQUIRE (GpuGeometryPagePool::AcquirePhysicalPoolOwnership,
  // inside ProcessFeedbackAndDrainCompletions below) + the decompression + raster reads of
  // whatever the transfer queue just copied, so THIS command buffer -- not cmdEarly, which never
  // touches page-pool data at all -- is the one that must wait for the transfer queue's release +
  // copy to have completed first.
  // =========================================================================================
  m_Streaming.ProcessFeedbackAndDrainCompletions(cmdMid, transferCmd, m_LODSelection.GetFeedbackBuffer(),
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
    lodViewParams.view = cameraCopy.view;
    lodViewParams.proj = cameraCopy.proj;
    lodViewParams.pixelErrorThreshold = kLODPixelErrorThreshold;
    // 2*atan(1/projScaleY) recovers fovYRadians from the projection matrix's own Y-scale term
    // (see projScaleY's own comment below) -- computed here, ahead of projScaleY's declaration,
    // since ExtractFrustumPlanes/viewProj aren't needed for this derivation.
    float earlyProjScaleY = std::abs(cameraCopy.proj.m[5]);
    lodViewParams.fovYRadians = 2.0f * std::atan(1.0f / earlyProjScaleY);
    lodViewParams.viewportHeight = static_cast<float>(m_RenderExtent.height);
    lodViewParams.aspectRatio = static_cast<float>(m_RenderExtent.width) / static_cast<float>(m_RenderExtent.height);

    m_LODSelection.RecordEvaluateAndCompact(cmdMid, lodViewParams);

#ifndef NDEBUG
    // See RequestDebugDAGCutGapsDump()'s own comment: state 1 means main.cpp's 'K' key armed a
    // dump this frame -- record the DAGDecisionSSBO readback now (right after the dispatch that
    // just wrote this frame's decisions) and advance to state 2 so PumpDebugDAGCutGapsDump() logs
    // it once this frame's fence confirms the copy has landed.
    if (m_DebugDAGCutGapsDumpState == 1) {
        m_LODSelection.RecordDebugReadback(cmdMid);
        m_DebugDAGCutGapsDumpState = 2;
    }
#endif

    // Captures THIS frame's residency-miss reports (ClusterLODResidencyFallback.comp, just
    // dispatched above) into the feedback buffer's host-visible readback half, for [1a]'s
    // ProcessFeedbackAndDrainCompletions() to consume next frame -- see FeedbackBuffer::
    // RecordReadback()'s own doc comment for why "next frame" (not this one) is the earliest safe
    // point to read it back on the CPU.
    m_LODSelection.GetFeedbackBuffer().RecordReadback(cmdMid);

    m_LODSelection.RecordBuildEarlyDispatchArgs(cmdMid);
  }

  // =========================================================================================
  // [1b] Upload this frame's WPOGlobalsUBO (globalTime + padding) -- read by both raster passes'
  // WPO sway function (wpo_deformation.glsl). Uploaded once here, well before either raster pass
  // records its draw/dispatch, so a single barrier below covers both consumers.
  // =========================================================================================
  {
    WPOGlobalsUBO wpoGlobals{};
    wpoGlobals.globalTime = globalTimeSeconds;
    // Phase 1 (Nanite advanced) debug toggles ('B'/'U' in main.cpp) -- 1.0 = full effect, 0.0 =
    // fully off, same Release-always-on convention as every other SetDebug*Enabled toggle above
    // (Release hardcodes both multipliers to 1.0, no toggle exists there).
#ifndef NDEBUG
    wpoGlobals.enhancedDisplacementDebugMultiplier = m_DebugEnhancedDisplacementEnabled ? 1.0f : 0.0f;
    wpoGlobals.splineDeformationDebugMultiplier = m_DebugSplineDeformationEnabled ? 1.0f : 0.0f;
#else
    wpoGlobals.enhancedDisplacementDebugMultiplier = 1.0f;
    wpoGlobals.splineDeformationDebugMultiplier = 1.0f;
#endif
    vkCmdUpdateBuffer(cmdMid, m_WPOGlobalsBuffer.Handle(), 0, sizeof(WPOGlobalsUBO), &wpoGlobals);

    VkMemoryBarrier2 wpoBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    wpoBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    wpoBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    wpoBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    wpoBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;

    VkDependencyInfo wpoDepInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    wpoDepInfo.memoryBarrierCount = 1;
    wpoDepInfo.pMemoryBarriers = &wpoBarrier;
    vkCmdPipelineBarrier2(cmdMid, &wpoDepInfo);
  }

  // =========================================================================================
  // [2] EARLY cull: every leaf candidate vs frustum/backface + LAST frame's HZB
  // (rebuilt at the end of the previous frame's RecordFrameLate from that frame's complete
  // depth). Its trailing barrier makes the early draw list/count visible to
  // DRAW_INDIRECT and the pending + software lists visible to later COMPUTE.
  // =========================================================================================
  m_OcclusionCulling.RecordEarlyPass(cmdMid, viewParams, viewProj, projScaleY,
                                     m_LODSelection.GetEarlyDispatchArgsBuffer(),
                                     kSoftwareRasterThresholdPixels
#ifndef NDEBUG
                                     , cameraCopy.disableOcclusionCulling
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
    vkCmdPipelineBarrier2(cmdMid, &depInfo);
  }

  // =========================================================================================
  // [4] EARLY hardware raster: draws exactly the early cull's survivors via
  // vkCmdDrawIndexedIndirectCount into the VisBuffer + depth (CLEAR load ops). Opaque first (no
  // discard, real hardware early-Z eligible), then masked (mask-sampling discard) -- opaque-first
  // is a perf-only ordering choice (lets the masked pipeline's depth test reject more fragments
  // before running its non-early-Z frag shader); final depth/VisBuffer content is order-independent.
  // =========================================================================================
  BeginVisBufferRendering(cmdMid, /*clearAttachments=*/true);
  m_HardwareRaster.RecordDraw(
      cmdMid, camera, m_RenderExtent,
      m_Decompression.GetDecompressedIndexPoolBuffer(),
      m_OcclusionCulling.GetEarlyIndirectCommandOpaqueBuffer(),
      m_OcclusionCulling.GetEarlyDrawCountOpaqueBuffer(), m_ClusterCount, /*opaque=*/true);
  m_HardwareRaster.RecordDraw(
      cmdMid, camera, m_RenderExtent,
      m_Decompression.GetDecompressedIndexPoolBuffer(),
      m_OcclusionCulling.GetEarlyIndirectCommandBuffer(),
      m_OcclusionCulling.GetEarlyDrawCountBuffer(), m_ClusterCount, /*opaque=*/false);
  vkCmdEndRendering(cmdMid);

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
    vkCmdPipelineBarrier2(cmdMid, &depInfo);
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
  m_HZB.Generate(cmdMid);
  {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmdMid, &depInfo);
  }

  // =========================================================================================
  // [7] LATE cull: GPU-sized indirect dispatch over exactly the pending list,
  // against the fresh HZB. RecordLatePass's trailing barrier covers the late
  // draw list for DRAW_INDIRECT; the extra barrier below additionally makes its
  // software-cluster-list writes visible to COMPUTE (the software raster's
  // dispatch-args build + raster reads), which that trailing barrier does not
  // cover.
  // =========================================================================================
  m_OcclusionCulling.RecordBuildLateDispatchArgs(cmdMid);
  m_OcclusionCulling.RecordLatePass(cmdMid
#ifndef NDEBUG
    , cameraCopy.disableOcclusionCulling
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
    vkCmdPipelineBarrier2(cmdMid, &depInfo);
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
    vkCmdPipelineBarrier2(cmdMid, &depInfo);
  }

  // =========================================================================================
  // [9] LATE hardware raster (loadOp = LOAD) -- draws the disocclusions the
  // early pass could not confirm, on top of the early output. Opaque first, same rationale as
  // the early pass above.
  // =========================================================================================
  BeginVisBufferRendering(cmdMid, /*clearAttachments=*/false);
  m_HardwareRaster.RecordDraw(cmdMid, camera, m_RenderExtent,
                              m_Decompression.GetDecompressedIndexPoolBuffer(),
                              m_OcclusionCulling.GetLateIndirectCommandOpaqueBuffer(),
                              m_OcclusionCulling.GetLateDrawCountOpaqueBuffer(),
                              m_ClusterCount, /*opaque=*/true);
  m_HardwareRaster.RecordDraw(cmdMid, camera, m_RenderExtent,
                              m_Decompression.GetDecompressedIndexPoolBuffer(),
                              m_OcclusionCulling.GetLateIndirectCommandBuffer(),
                              m_OcclusionCulling.GetLateDrawCountBuffer(),
                              m_ClusterCount, /*opaque=*/false);
  vkCmdEndRendering(cmdMid);

  // =========================================================================================
  // [10] Software raster of every micro-triangle cluster (early- and
  // late-routed entries are both in the list by now). Writes only its own R64
  // atomic image -- fully independent of the hardware attachments, so no
  // ordering against [9] is required beyond what its own internal barriers
  // already record.
  // =========================================================================================
  m_SoftwareRaster.RecordRaster(cmdMid, viewProj);

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
    vkCmdPipelineBarrier2(cmdMid, &depInfo);
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
  // COMPUTE barrier). Confirmed (by reading ClusterResolvePass.h/.cpp/ClusterShadingBinPass.h) that
  // neither pass' own descriptor bindings reference the Surface Cache atlas, the TLAS, or any
  // GI-related Global SDF state -- only renderer::VirtualShadowMapPass's own shadow UBOs (direct
  // sun shadowing, not indirect GI) and renderer::VirtualTextureManager's page table.
  // =========================================================================================
#ifndef NDEBUG
  if (cameraCopy.debugViewMode == DEBUG_VIEW_NORMAL) {
    m_ShadingBin.RecordClassifyAndSort(cmdMid, m_RenderExtent);
    m_Resolve.RecordResolveBinned(cmdMid, viewProj, m_SceneLights.sun, cameraPositionWorld, m_ShadingBin);
  } else {
    maths::mat4 prevViewProjForResolve = m_HasPrevViewProj ? m_PrevViewProj : viewProj;
    m_Resolve.RecordResolve(cmdMid, viewProj, prevViewProjForResolve, m_SceneLights.sun, cameraPositionWorld, cameraCopy.debugViewMode);
  }
#else
  m_ShadingBin.RecordClassifyAndSort(cmdMid, m_RenderExtent);
  m_Resolve.RecordResolveBinned(cmdMid, viewProj, m_SceneLights.sun, cameraPositionWorld, m_ShadingBin);
#endif
}

void ClusterRenderPipeline::RecordFrameLate(VkCommandBuffer cmdLate, VkImage swapchainImage, VkImageView swapchainImageView) {
  const CameraPushConstants& cameraCopy = m_FrameScratch.cameraCopy;
  const maths::vec3& cameraPositionWorld = m_FrameScratch.cameraPositionWorld;
  const CameraFrameInfo& cameraFrameInfo = m_FrameScratch.cameraFrameInfo;
  const float globalTimeSeconds = m_FrameScratch.globalTimeSeconds;
  const maths::mat4& viewProj = m_FrameScratch.viewProj;
  const maths::mat4& invViewProj = m_FrameScratch.invViewProj;
  const float jitterX = m_FrameScratch.jitterX;
  const float jitterY = m_FrameScratch.jitterY;
  const bool taatsrEnabled = m_FrameScratch.taatsrEnabled;
  const uint32_t traceMode = m_FrameScratch.traceMode;
  const bool useAsyncCompute = m_FrameScratch.useAsyncCompute;

  // =========================================================================================
  // [1z-acquire] Phase 2 (Lumen advanced roadmap) fix: THIS SAME FRAME's own RecordFrameEarly()
  // released the Surface Cache atlas/TLAS to the async-compute queue (when useAsyncCompute); its
  // work (RecordAsyncCompute()) has, by the time this command buffer's execution actually reaches
  // this barrier, signaled VulkanContext::GetAsyncComputeFinishedSemaphore() -- this cmdLate
  // submission waits on it (main.cpp) before any GPU work below this point executes, so the
  // ACQUIRE below is safe the instant it's reached. Recorded as the very FIRST thing in this
  // command buffer since every pass below that reads the Surface Cache atlas/TLAS needs it.
  //
  // Narrower than the OLD design's top-of-frame acquire (which additionally covered
  // COLOR_ATTACHMENT_OUTPUT_BIT/VERTEX_INPUT_BIT/COLOR_ATTACHMENT_WRITE_BIT/
  // VERTEX_ATTRIBUTE_READ_BIT/INDEX_READ_BIT, because THAT acquire immediately preceded
  // RecordCapture's own raster draw into the atlas) -- RecordCapture already ran earlier THIS
  // frame, in RecordFrameEarly(), so nothing below this acquire writes to the atlas images or
  // reads the Fallback Mesh vertex/index buffers as actual raster vertex input. Verified (not
  // assumed) by reading VulkanUtils::WriteSharedGeometryBindings -- every one of Reflection/World
  // Probes/MegaLights/the 3 forward passes binds the Fallback Mesh vertex/index/draw-range buffers
  // as VK_DESCRIPTOR_TYPE_STORAGE_BUFFER (SHADER_STORAGE_READ_BIT) and the TLAS as
  // VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR (ACCELERATION_STRUCTURE_READ_BIT_KHR), never as
  // actual bound vertex/index buffers -- and none of them write to the 5 atlas images (only
  // GIInject/RecordCapture do, both already complete by this point).
  // =========================================================================================
  if (useAsyncCompute) {
    RecordSurfaceCacheOwnershipTransfer(cmdLate, /*isRelease=*/false,
        m_AsyncComputeQueueFamilyIndex, m_GraphicsQueueFamilyIndex,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  }

  // =========================================================================================
  // [12a] Phase PP4 (post-process stack roadmap): GTAO + Screen-Space Contact Shadows.
  // Contact Shadows MUST run here -- immediately after [12] Resolve, BEFORE [12b2]/[12b3]
  // (Reflections/MegaLights) add anything -- see ScreenSpaceEffectsPass::RecordContactShadows'
  // own comment for why that ordering is load-bearing. GTAO has no such ordering constraint (its
  // own owned AO image is only consumed later by [12e] GIComposite) -- grouped here purely for
  // locality, both read the exact same GBuffer this same point in the frame already has ready.
  // =========================================================================================
  {
    ScreenSpaceEffectsPass::Settings ssfxSettings{};
    ssfxSettings.aoRadiusWorld = config::postprocess::AO_RADIUS_WORLD;
    ssfxSettings.aoIntensity = config::postprocess::AO_ENABLED ? config::postprocess::AO_INTENSITY : 0.0f;
    ssfxSettings.aoPower = config::postprocess::AO_POWER;
    ssfxSettings.contactShadowLengthWorld = config::postprocess::CONTACT_SHADOW_LENGTH_WORLD;
    ssfxSettings.contactShadowIntensity = config::postprocess::CONTACT_SHADOW_ENABLED ? config::postprocess::CONTACT_SHADOW_INTENSITY : 0.0f;
    ssfxSettings.contactShadowThicknessWorld = config::postprocess::CONTACT_SHADOW_THICKNESS_WORLD;
    ssfxSettings.ssrFallbackMaxDistanceWorld = config::postprocess::SSR_FALLBACK_MAX_DISTANCE_WORLD;
    ssfxSettings.ssrFallbackThicknessWorld = config::postprocess::SSR_FALLBACK_THICKNESS_WORLD;
    ssfxSettings.ssrFallbackIntensity = config::postprocess::SSR_FALLBACK_INTENSITY;

    m_ScreenSpaceEffects.RecordAmbientOcclusion(cmdLate, viewProj, cameraPositionWorld, cameraFrameInfo.fovYRadians, ssfxSettings);
    m_ScreenSpaceEffects.RecordContactShadows(cmdLate, viewProj, cameraPositionWorld, m_SceneLights.sun.direction, ssfxSettings);
  }

  // Captures THIS frame's shadow-page miss reports (written by SurfaceCacheCapture.frag at [1z]
  // above and by whichever ClusterResolve.comp/ClusterResolveBinned.comp path just ran) for
  // VirtualShadowMapPass::RecordBeginFrame() to consume next frame -- see that class' own
  // one-frame-lag contract. Placed here, right after every pass that can call
  // RequestShadowPageResidency() this frame has run.
  m_VirtualShadowMap.RecordEndFrame(cmdLate);

  // Captures THIS frame's virtual texture page-miss reports (written by the same ClusterResolve
  // .comp/ClusterResolveBinned.comp path that just ran, see virtual_texture_lookup.glsl's own
  // miss-detection comment) for VirtualTextureStreamingCoordinator::RecordBeginFrame() to consume
  // next frame -- identical one-frame-lag placement to m_VirtualShadowMap.RecordEndFrame() above.
  m_VTStreaming.RecordEndFrame(cmdLate);

  // [12b] Screen Trace GI (Lumen Screen Trace + World Probe fallback): traces linear screen-space
  // rays against the GBuffer depth/normal, falling back to the 3D world probe grid on miss.
  // Writes to its own dedicated output image (m_ScreenTrace.GetOutputImage()).
  // =========================================================================================
  {
#ifndef NDEBUG
    bool ssrtEnabled = m_DebugSSRTEnabled;
#else
    bool ssrtEnabled = true;
#endif
    if (ssrtEnabled) {
      m_ScreenTrace.RecordTrace(cmdLate, cameraCopy, cameraPositionWorld, m_WorldProbes.GetGridOriginWorld(), m_FrameIndex);
    } else {
      VkClearColorValue blackClear{};
      blackClear.float32[0] = 0.0f;
      blackClear.float32[1] = 0.0f;
      blackClear.float32[2] = 0.0f;
      blackClear.float32[3] = 0.0f;
      VulkanUtils::ClearComputeImageToGeneral(cmdLate, m_ScreenTrace.GetOutputImage(), blackClear);
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
    m_Reflection.RecordUpdateViewParams(cmdLate, viewProj, prevViewProjForReflection, cameraPositionWorld);

#ifndef NDEBUG
    bool reflectionsEnabled = m_DebugReflectionsEnabled;
#else
    bool reflectionsEnabled = true;
#endif
    if (reflectionsEnabled) {
      m_Reflection.RecordTrace(cmdLate, m_TraceContext, m_TraceContext.GetEntityCount(), traceMode, m_FrameIndex);
      m_Reflection.RecordTemporal(cmdLate);
      m_Reflection.RecordGather(cmdLate);

      // Phase PP4: SSR Fallback -- needs m_Reflection's own hit-mask, just written by RecordTrace()
      // above this same frame, so gated under the same `reflectionsEnabled` toggle (a stale
      // hit-mask from a prior frame would otherwise misdirect this pass when reflections are
      // debug-disabled).
      ScreenSpaceEffectsPass::Settings ssrFallbackSettings{};
      ssrFallbackSettings.ssrFallbackMaxDistanceWorld = config::postprocess::SSR_FALLBACK_MAX_DISTANCE_WORLD;
      ssrFallbackSettings.ssrFallbackThicknessWorld = config::postprocess::SSR_FALLBACK_THICKNESS_WORLD;
      ssrFallbackSettings.ssrFallbackIntensity = config::postprocess::SSR_FALLBACK_ENABLED ? config::postprocess::SSR_FALLBACK_INTENSITY : 0.0f;
      m_ScreenSpaceEffects.RecordSSRFallback(cmdLate, viewProj, cameraPositionWorld, m_SceneLights.sun.direction, ssrFallbackSettings);
    }
  }

  // =========================================================================================
  // [12b3] Phase A of the MegaLights native-port roadmap: RIS-weighted stochastic multi-point-
  // light direct lighting + 1 ray-traced shadow-visibility ray per pixel, additively read-modify-
  // writing m_Resolve's own output color image -- same additive-RMW convention as [12b2] above.
  // Grouped right after [12b2] since both read the same GBuffer and compose into the same color
  // image. UNLIKE [12b2] reflections, this pass owns its OWN dedicated À-Trous denoiser instance
  // (m_MegaLights internally denoises its raw 1-sample-per-pixel Monte Carlo noise before
  // compositing) rather than relying on [12d]'s shared m_Denoiser -- that shared instance denoises
  // m_ScreenTrace's own Screen Trace GI output specifically (see GICompositePass's own header
  // comment), not the composited direct+reflections color the way it did before renderer::
  // ScreenTracePass/GICompositePass were wired up. See MegaLightsPass's own class comment for the
  // full "discovered mid-implementation" explanation.
  //
  // `megaLightsEnabled` (debug-only toggle, main.cpp's 'X' key) gates the call entirely, same
  // Release-always-on convention as `reflectionsEnabled` above (a real live consumer from frame
  // one).
  //
  // Phase 4 of the "Nanite advanced" roadmap (light BVH for RIS spatial bias, temporal ReSTIR with
  // per-frame revalidated visibility): RecordShade now also takes `prevViewProjForMegaLights` --
  // its own reservoir reprojection needs the previous frame's combined view-projection matrix, same
  // `m_HasPrevViewProj` frame-0-safety ternary already used for [12b2]'s own
  // prevViewProjForReflection above (identity matrix on the very first frame ever recorded; see
  // MegaLightsPass::RecordShade's own header comment for why an identity matrix is always safe here
  // even then, thanks to the reservoir's own sentinel-fill invalid-history guard).
  // =========================================================================================
  {
#ifndef NDEBUG
    bool megaLightsEnabled = m_DebugMegaLightsEnabled;
#else
    bool megaLightsEnabled = true;
#endif
    if (megaLightsEnabled) {
      maths::mat4 prevViewProjForMegaLights = m_HasPrevViewProj ? m_PrevViewProj : maths::mat4{};
      m_MegaLights.RecordShade(cmdLate, viewProj, prevViewProjForMegaLights, cameraPositionWorld, m_FrameIndex);
    }
  }

  // =========================================================================================
  // [12c] World Probe grid: fully rebuilt every frame from the Surface Cache radiance atlas
  // [1z] already re-injected into this frame ("Propagate Surface Cache lighting directly
  // into this 3D grid at each frame") -- INTENDED as what dynamic/off-screen objects would
  // sample for indirect light (world_probe_sampling.glsl's SampleWorldProbeGrid) -- also the
  // fallback m_ScreenTrace itself samples on a screen-space march miss, since a screen-space march
  // only ever sees on-screen pixels. Independent GPU work
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
  bool worldProbesEnabled = true;
#endif
  if (worldProbesEnabled) {
    m_WorldProbes.RecordUpdate(cmdLate, cameraFrameInfo.position, m_TraceContext, traceMode, m_SceneLights.sun.direction);
    {
      VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
      barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
      barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

      VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
      depInfo.memoryBarrierCount = 1;
      depInfo.pMemoryBarriers = &barrier;
      vkCmdPipelineBarrier2(cmdLate, &depInfo);
    }
  }

  // =========================================================================================
  // [12d] Final spatial denoiser (À-Trous wavelet): only in the normal (non-debug-visualization)
  // view path, plus DEBUG_VIEW_LUMEN (which wants to inspect the actual denoised GI term
  // GICompositePass substitutes for the whole final image, see [12e] below and the blit-swap's
  // own comment) -- DEBUG_VIEW_SPATIAL_PROBES is excluded because GICompositePass's own world-
  // probe visualization for that mode does not read m_Denoiser's output at all (see
  // GIComposite.comp's viewMode==14 branch), so denoising here would be pure wasted GPU work for
  // that view. m_Resolve's own G-buffer normal/depth (already visible from [11]/[12]'s own
  // trailing barriers) guide the filter; ends with its own trailing barrier (COMPUTE_SHADER/
  // STORAGE_WRITE -> COMPUTE_SHADER/SHADER_SAMPLED_READ) for [12e]'s read.
  // =========================================================================================
#ifndef NDEBUG
  bool applyDenoise = (cameraCopy.debugViewMode == DEBUG_VIEW_NORMAL || cameraCopy.debugViewMode == DEBUG_VIEW_LUMEN);
#else
  bool applyDenoise = true;
#endif
  if (applyDenoise) {
    m_Denoiser.RecordDenoise(cmdLate);
  }

  // =========================================================================================
  // [12e] GI Composite: blends direct lit color/reflections with denoised indirect GI.
  // =========================================================================================
  m_GIComposite.RecordComposite(cmdLate
#ifndef NDEBUG
      , cameraCopy, cameraPositionWorld, m_WorldProbes.GetGridOriginWorld()
#endif
  );

  // Barrier from GIComposite compute writes to TAA/TSR sampled reads
  {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmdLate, &depInfo);
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
    vkCmdPipelineBarrier2(cmdLate, &depInfo);
  }
  m_HZB.Generate(cmdLate);
  {
    // Next frame's early cull samples the pyramid; same STORAGE_WRITE ->
    // SAMPLED_READ extension as [6]. Barriers order against subsequent commands in submission
    // order on the same queue, so this correctly covers a LATER command buffer too -- and still
    // does, now that this frame's own graphics work spans 3 command buffers (cmdEarly/cmdMid/
    // cmdLate, this barrier is recorded in cmdLate) rather than 1: the graphics queue's per-frame
    // submission order is cmdEarly(N) -> cmdMid(N) -> cmdLate(N) -> cmdEarly(N+1) -> cmdMid(N+1) ->
    // ..., so cmdMid(N+1) (next frame's own [2] early cull, the actual consumer) is still strictly
    // LATER in this queue's submission order than THIS barrier (in cmdLate(N)), even though
    // cmdEarly(N+1) is sandwiched in between them -- cmdEarly(N+1) never touches the HZB pyramid at
    // all, so its presence in between changes nothing about this ordering guarantee.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmdLate, &depInfo);
  }

  // =========================================================================================
  // [13c] Forward-rendered translucent/transparent materials -- drawn directly onto
  // m_GIComposite's own output image (unconditionally the [14] blit's normal-view-mode source,
  // see the blit-swap's own comment below), so transparency composites on top of the fully-lit
  // (GI/reflections included) opaque scene. No-op internally if TransparentForwardPass::Init()
  // found zero transparent leaf clusters this run. Must run before the debug overlay below so the
  // HUD stays on top of everything.
  // =========================================================================================
  {
    VkImage transparentTargetImage = m_GIComposite.GetOutputImage();
    VkImageView transparentTargetView = m_GIComposite.GetOutputView();

    // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): recorded BEFORE the glass/
    // translucent draw below, deliberately -- this pass WRITES real depth (unlike glass, which
    // only reads it), so glass's own subsequent depth test correctly occludes against this
    // entity's real (displaced) surface. See HeroTessellationPass's own class comment for the
    // widened synchronization scope this write requires on exit.
    m_HeroTessellation.RecordDraw(cmdLate, viewProj, cameraFrameInfo.position,
        transparentTargetImage, transparentTargetView, m_DepthImage, m_DepthImageView,
        m_RenderExtent, traceMode, m_FrameIndex, m_TraceContext, m_WorldProbes, m_SceneLights);

    m_TransparentForward.RecordDraw(cmdLate, transparentTargetImage, transparentTargetView, m_DepthImageView,
        m_RenderExtent, cameraCopy.view, cameraCopy.proj, m_Decompression.GetDecompressedIndexPoolBuffer(),
        cameraFrameInfo.position, m_SceneLights, globalTimeSeconds, m_TraceContext, traceMode, m_FrameIndex);

    // Phase 7c (UE5.8 parity roadmap, water/erosion): recorded LAST among the forward passes --
    // see ClusterRenderPipeline.h's own comment on m_WaterForward for why (it snapshots the
    // already fully-composited frame, including the glass/translucent draw just above, for its
    // own refraction term).
    m_WaterForward.RecordDraw(cmdLate, viewProj, cameraFrameInfo.position,
        transparentTargetImage, transparentTargetView, m_DepthImage, m_DepthImageView,
        m_RenderExtent, traceMode, m_FrameIndex, globalTimeSeconds, m_TraceContext);
  }

  // Phase 2 (Lumen advanced roadmap) fix, 2026-07-17: the old [1z2] hand-off block that used to
  // live HERE (release the Surface Cache atlas/TLAS to the async-compute queue, late in the
  // frame, after every graphics-side consumer had read it) is GONE -- the release now happens in
  // RecordFrameEarly() (right after RecordCapture/GlobalSDF.RecordUpdate) and the matching acquire
  // + TLAS refit + radiosity bounce loop + release-back now live in RecordAsyncCompute(), both
  // executed earlier this same frame, before this method (RecordFrameLate) even began recording --
  // see this class' own header comment for the full redesign this fixed and why. Nothing here
  // needs to touch the Surface Cache atlas/TLAS again this frame (TAA/DoF/Bloom/PostProcess/blit
  // below are pure post-process, no GI sampling).

  // =========================================================================================
  // [13d] TAA & TSR pass (Temporal Anti-Aliasing and Temporal Super Resolution):
  // Takes the low-resolution color (with transparent materials already composited) and resolved
  // depth, performs subpixel reprojection, history clamping, and temporal blending, producing
  // a high-quality upscaled output at native display resolution.
  // =========================================================================================
  {
      maths::mat4 prevViewProjForTAA = m_HasPrevViewProj ? m_PrevViewProj : viewProj;
      bool resetHistory = !m_HasPrevViewProj || !taatsrEnabled;
      float passJitterX = taatsrEnabled ? jitterX : 0.0f;
      float passJitterY = taatsrEnabled ? jitterY : 0.0f;

      VkImageView currentLowResColorView = m_GIComposite.GetOutputView();
      m_TAATSR.UpdateDescriptorSets(currentLowResColorView, m_Resolve.GetOutputDepthView());

      m_TAATSR.RecordPass(cmdLate, viewProj, prevViewProjForTAA, invViewProj, passJitterX, passJitterY, m_FrameIndex, resetHistory);
  }

  // =========================================================================================
  // [13e] Phase PP3: physically-derived Depth of Field over m_TAATSR's own freshly-written HDR
  // output -- must run before [13f]'s Bloom (see DepthOfFieldPass's own class comment for why: an
  // out-of-focus highlight should itself bloom into a soft disc, which only happens if DOF's own
  // blur runs first). m_TAATSR.GetOutputView() ping-pongs between 2 images every RecordPass() call
  // (just above), so m_DepthOfField's own source descriptor must be re-written every frame too.
  // =========================================================================================
  {
      m_DepthOfField.UpdateSourceDescriptor(m_TAATSR.GetOutputView());

      // m_TAATSR's own output image's last writer is always its own compute dispatch -- re-stated
      // here explicitly rather than relying on m_TAATSR::RecordPass' own trailing barrier (which
      // only targets a BLIT read, not this compute-shader sampled read).
      VkMemoryBarrier2 taatsrToDofBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      taatsrToDofBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      taatsrToDofBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
      taatsrToDofBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      taatsrToDofBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
      VkDependencyInfo taatsrToDofDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      taatsrToDofDep.memoryBarrierCount = 1;
      taatsrToDofDep.pMemoryBarriers = &taatsrToDofBarrier;
      vkCmdPipelineBarrier2(cmdLate, &taatsrToDofDep);

      DepthOfFieldPass::Settings dofSettings{};
      dofSettings.focalLengthMM = config::postprocess::DOF_FOCAL_LENGTH_MM;
      dofSettings.focusDistanceWorldUnits = config::postprocess::DOF_FOCUS_DISTANCE_WORLD_UNITS;
      dofSettings.maxCoCRadiusPixels = config::postprocess::DOF_ENABLED ? config::postprocess::DOF_MAX_COC_RADIUS_PIXELS : 0.0f;
      m_DepthOfField.RecordGenerate(cmdLate, invViewProj, cameraPositionWorld, config::postprocess::EXPOSURE_APERTURE, dofSettings);
  }

  // =========================================================================================
  // [13f] Phase PP2 (post-process stack roadmap): Bloom/Lens Flare/Anamorphic Flare/Lens Dirt,
  // then Phase PP1's Physical Camera auto-exposure -> White Balance -> Color Correction -> ACES
  // Tone Mapping -> Gamma Correction composite (also folding in Phase PP3's Motion Blur / Height
  // Fog / Heat Distortion -- see PostProcessComposite.comp's own comment) -- both reading
  // m_DepthOfField's own freshly-written output -- see BloomPass's and PostProcessPass's own class
  // comments for why m_PostProcess must be the very last compute step before [14]'s blit (and
  // before [13b]'s Debug-only stat overlay below, which now draws onto ITS output instead of
  // m_TAATSR's raw HDR buffer). m_DepthOfField's own output view identity is effectively "per-
  // frame" too (it wraps m_TAATSR's own ping-ponged source), so both passes' own descriptor sets
  // pointing at it must be re-written every frame, exactly like m_TAATSR.UpdateDescriptorSets()
  // itself is called every frame above. Both run unconditionally every frame, exactly like [13d]'s
  // own TAA/TSR pass -- the debug views that substitute a different, deliberately-untonemapped
  // diagnostic image as the blit source instead (m_GIComposite/m_SDFRayMarch/m_Resolve, see [14]'s
  // own blitSourceImage swap below) simply never read either pass' output, same as how those views
  // already ignore m_TAATSR's own output today.
  // =========================================================================================
  {
      m_Bloom.UpdateSourceDescriptor(m_DepthOfField.GetOutputView());
      m_PostProcess.UpdateDescriptorSets(m_DepthOfField.GetOutputView(), m_Bloom.GetOutputView());

      // m_DepthOfField's own trailing barrier (inside RecordGenerate) already makes its output
      // visible to COMPUTE_SHADER/SHADER_SAMPLED_READ -- no further barrier needed before m_Bloom/
      // m_PostProcess read it here.

      BloomPass::Settings bloomSettings{};
      bloomSettings.threshold = config::postprocess::BLOOM_THRESHOLD;
      bloomSettings.softKnee = config::postprocess::BLOOM_SOFT_KNEE;
      bloomSettings.upsampleRadius = config::postprocess::BLOOM_UPSAMPLE_RADIUS;
      bloomSettings.ghostIntensity = config::postprocess::LENS_FLARE_GHOST_INTENSITY;
      bloomSettings.ghostCount = config::postprocess::LENS_FLARE_GHOST_COUNT;
      bloomSettings.ghostSpacing = config::postprocess::LENS_FLARE_GHOST_SPACING;
      bloomSettings.anamorphicIntensity = config::postprocess::ANAMORPHIC_FLARE_INTENSITY;
      bloomSettings.anamorphicStretch = config::postprocess::ANAMORPHIC_FLARE_STRETCH;
      bloomSettings.dirtIntensity = config::postprocess::LENS_DIRT_INTENSITY;
      bloomSettings.dirtScale = config::postprocess::LENS_DIRT_SCALE;
      m_Bloom.RecordGenerate(cmdLate, bloomSettings);

      float deltaTimeSeconds = m_HasLastFrameTime ? (globalTimeSeconds - m_LastFrameTimeSeconds) : (1.0f / 60.0f);
      deltaTimeSeconds = std::clamp(deltaTimeSeconds, 0.0f, 0.25f); // Guard against alt-tab/breakpoint stalls.
      m_LastFrameTimeSeconds = globalTimeSeconds;
      m_HasLastFrameTime = true;

      PostProcessPass::Settings ppSettings{};
      ppSettings.aperture = config::postprocess::EXPOSURE_APERTURE;
      ppSettings.shutterSpeedSeconds = config::postprocess::EXPOSURE_SHUTTER_SPEED_SECONDS;
      ppSettings.isoSensitivity = config::postprocess::EXPOSURE_ISO;
      ppSettings.useAutoExposure = config::postprocess::EXPOSURE_USE_AUTO;
      ppSettings.exposureCompensationEV = config::postprocess::EXPOSURE_COMPENSATION_EV;
      ppSettings.adaptationSpeedUpEVPerSec = config::postprocess::EXPOSURE_ADAPTATION_SPEED_UP_EV_PER_SEC;
      ppSettings.adaptationSpeedDownEVPerSec = config::postprocess::EXPOSURE_ADAPTATION_SPEED_DOWN_EV_PER_SEC;
      // Real-time "Post FX" toggles (ImGui, main.cpp): every effect below already has its own
      // zero-is-off strength knob (see config::postprocess's own comment on its *_ENABLED block)
      // -- White Balance/Color Correction are the two exceptions with no natural zero, so they
      // fall back to their own documented neutral/identity values instead.
      ppSettings.whiteBalanceTempKelvin = config::postprocess::WHITE_BALANCE_ENABLED ? config::postprocess::WHITE_BALANCE_TEMP_KELVIN : 6500.0f;
      ppSettings.whiteBalanceTint = config::postprocess::WHITE_BALANCE_ENABLED ? config::postprocess::WHITE_BALANCE_TINT : 0.0f;
      ppSettings.liftR = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_LIFT_R : 0.0f;
      ppSettings.liftG = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_LIFT_G : 0.0f;
      ppSettings.liftB = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_LIFT_B : 0.0f;
      ppSettings.gammaR = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_GAMMA_R : 1.0f;
      ppSettings.gammaG = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_GAMMA_G : 1.0f;
      ppSettings.gammaB = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_GAMMA_B : 1.0f;
      ppSettings.gainR = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_GAIN_R : 1.0f;
      ppSettings.gainG = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_GAIN_G : 1.0f;
      ppSettings.gainB = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_GAIN_B : 1.0f;
      ppSettings.saturation = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_SATURATION : 1.0f;
      ppSettings.contrast = config::postprocess::COLOR_CORRECTION_ENABLED ? config::postprocess::COLOR_CONTRAST : 1.0f;
      ppSettings.displayGamma = config::postprocess::DISPLAY_GAMMA;
      ppSettings.bloomIntensity = config::postprocess::BLOOM_ENABLED ? config::postprocess::BLOOM_INTENSITY : 0.0f;
      ppSettings.chromaticAberrationIntensity = config::postprocess::CHROMATIC_ABERRATION_ENABLED ? config::postprocess::CHROMATIC_ABERRATION_INTENSITY : 0.0f;
      ppSettings.vignetteIntensity = config::postprocess::VIGNETTE_ENABLED ? config::postprocess::VIGNETTE_INTENSITY : 0.0f;
      ppSettings.vignetteSmoothness = config::postprocess::VIGNETTE_SMOOTHNESS;
      ppSettings.vignetteColorBleed = config::postprocess::VIGNETTE_COLOR_BLEED;
      ppSettings.heatDistortionIntensity = config::postprocess::HEAT_DISTORTION_ENABLED ? config::postprocess::HEAT_DISTORTION_INTENSITY : 0.0f;
      ppSettings.motionBlurIntensity = config::postprocess::MOTION_BLUR_ENABLED ? config::postprocess::MOTION_BLUR_INTENSITY : 0.0f;
      ppSettings.motionBlurMaxVelocityUV = config::postprocess::MOTION_BLUR_MAX_VELOCITY_UV;
      ppSettings.fogColorR = config::postprocess::FOG_COLOR_R;
      ppSettings.fogColorG = config::postprocess::FOG_COLOR_G;
      ppSettings.fogColorB = config::postprocess::FOG_COLOR_B;
      ppSettings.fogDensity = config::postprocess::HEIGHT_FOG_ENABLED ? config::postprocess::FOG_DENSITY : 0.0f;
      ppSettings.fogHeightFalloff = config::postprocess::FOG_HEIGHT_FALLOFF;
      ppSettings.fogHeightOffset = config::postprocess::FOG_HEIGHT_OFFSET;
      ppSettings.fogStartDistance = config::postprocess::FOG_START_DISTANCE;
      ppSettings.fogMaxOpacity = config::postprocess::FOG_MAX_OPACITY;
      ppSettings.godRaysIntensity = config::postprocess::GOD_RAYS_ENABLED ? config::postprocess::GOD_RAYS_INTENSITY : 0.0f;
      ppSettings.godRaysDecay = config::postprocess::GOD_RAYS_DECAY;
      ppSettings.godRaysDensity = config::postprocess::GOD_RAYS_DENSITY;
      ppSettings.godRaysWeight = config::postprocess::GOD_RAYS_WEIGHT;
      ppSettings.paniniD = config::postprocess::PANINI_ENABLED ? config::postprocess::PANINI_D : 0.0f;
      ppSettings.paniniS = config::postprocess::PANINI_S;
      ppSettings.sharpenIntensity = config::postprocess::SHARPEN_ENABLED ? config::postprocess::SHARPEN_INTENSITY : 0.0f;
      ppSettings.sharpenRadiusPixels = config::postprocess::SHARPEN_RADIUS_PIXELS;
      ppSettings.filmGrainIntensity = config::postprocess::FILM_GRAIN_ENABLED ? config::postprocess::FILM_GRAIN_INTENSITY : 0.0f;
      ppSettings.filmGrainResponseMidpoint = config::postprocess::FILM_GRAIN_RESPONSE_MIDPOINT;

      // Same prevViewProj-or-current-frame-fallback expression [13d]'s own TAA pass already
      // resolved into its own block-scoped prevViewProjForTAA (out of scope here) -- Motion Blur's
      // own per-pixel reprojection needs the identical semantics, recomputed inline.
      maths::mat4 prevViewProjForPostProcess = m_HasPrevViewProj ? m_PrevViewProj : viewProj;

      // Atmos weather system, Subtask 3: refresh the froxel volumetric fog grid immediately before
      // its only consumer (m_PostProcess.RecordComposite below) -- keeps producer and consumer
      // adjacent in the frame graph, and guarantees renderer::VirtualShadowMapPass's shadow pages
      // (captured earlier, in [1z]) are this frame's freshest state by the time InjectLight samples
      // them.
      m_AtmosFog.RecordUpdate(cmdLate, cameraPositionWorld, cameraFrameInfo.forward, maths::vec3{0.0f, 1.0f, 0.0f},
          cameraFrameInfo.fovYRadians, cameraFrameInfo.aspectRatio,
          m_SceneLights.sun.direction, m_SceneLights.sun.color, m_SceneLights.sun.intensity, m_FrameIndex);

      // Atmos weather system, Subtask 4: refresh the half-res cloud raymarch immediately alongside
      // the fog update above -- same producer/consumer-adjacency reasoning.
      m_AtmosClouds.RecordUpdate(cmdLate, cameraPositionWorld, cameraFrameInfo.forward, maths::vec3{0.0f, 1.0f, 0.0f},
          cameraFrameInfo.fovYRadians, cameraFrameInfo.aspectRatio,
          m_SceneLights.sun.direction, m_SceneLights.sun.color, m_SceneLights.sun.intensity);

      m_PostProcess.RecordComposite(cmdLate, deltaTimeSeconds, ppSettings, invViewProj, prevViewProjForPostProcess, cameraPositionWorld,
          viewProj, m_SceneLights.sun.direction, cameraFrameInfo.forward,
          cameraFrameInfo.fovYRadians, cameraFrameInfo.aspectRatio, m_FrameIndex);
  }

#ifndef NDEBUG
  // ImGui "Buffer Viewer" dropdown: dispatched ONCE here (not inline in either blitSourceImage
  // selection block below, both of which only SELECT among already-produced images -- this is the
  // one exception that actually produces new GPU work this frame) so it's recorded before both the
  // HUD-targeting selection and the final blit's own selection reference its output image.
  if (config::debugview::SELECTED_BUFFER_INDEX != 0) {
    RecordDebugBufferView(cmdLate);
  }

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
    m_LastHWTriangleCount = hwTriangleCount;
    m_LastSWTriangleCount = swTriangleCount;

    m_TriangleStats.RecordClear(cmdLate);
    m_TriangleStats.RecordCompute(cmdLate);
    m_TriangleStats.RecordReadback(cmdLate);

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
        fps, static_cast<float>(m_RenderExtent.width), static_cast<float>(m_RenderExtent.height),
        m_DebugRadiosityEnabled, m_DebugSSRTEnabled, traceMode, m_DebugWorldProbesEnabled);

    // Determine blitSourceImage early to draw HUD directly onto it -- m_PostProcess's own output
    // (not m_TAATSR's raw HDR buffer) in the normal path now, so the HUD's own colors (e.g. plain
    // white text) pass through the SAME tone curve/gamma as the rest of the frame instead of being
    // drawn pre-tonemap and distorted by it (see renderer::PostProcessPass's own imageInfo.usage
    // comment on why its output image gained COLOR_ATTACHMENT_BIT, Debug only, for exactly this).
    VkImage blitSourceImage = m_PostProcess.GetOutputImage();
#ifndef NDEBUG
    if (cameraCopy.debugViewMode != DEBUG_VIEW_NORMAL && cameraCopy.debugViewMode != DEBUG_VIEW_MOTION_VECTORS) {
        if (cameraCopy.debugViewMode == DEBUG_VIEW_LUMEN || cameraCopy.debugViewMode == DEBUG_VIEW_SPATIAL_PROBES) {
            blitSourceImage = m_GIComposite.GetOutputImage();
        } else if (cameraCopy.debugViewMode == DEBUG_VIEW_GLOBAL_SDF) {
            blitSourceImage = m_SDFRayMarch.GetOutputImage();
        } else {
            blitSourceImage = m_Resolve.GetOutputColorImage();
        }
    }
    // ImGui "Buffer Viewer" dropdown -- takes priority over the Numpad debug-view-mode
    // substitutions above (an orthogonal selection mechanism; the actual dispatch already
    // happened just above this whole [13b] block, see that call site's own comment).
    if (config::debugview::SELECTED_BUFFER_INDEX != 0) {
        blitSourceImage = m_DebugBufferView.GetOutputImage();
    }
#endif

    VkImage overlayTargetImage = blitSourceImage;
    VkImageView overlayTargetView = VK_NULL_HANDLE;
    VkExtent2D overlayExtent = m_RenderExtent;
    // renderer::debug::DebugTextOverlay owns 2 pipeline variants (see its own Init()/RecordDraw()
    // comments) -- every branch below must report the ACTUAL format of whatever view it selects,
    // not assume one default. m_GIComposite/m_Resolve are rgba16f HDR (see ClusterResolvePass::
    // kOutputColorFormat's own comment), matching renderer::debug::DebugTextOverlay::
    // kHdrTargetFormat exactly -- m_PostProcess/m_SDFRayMarch are both RGBA8_UNORM (renderer::
    // PostProcessPass::kOutputFormat / SDFRayMarchPass::kOutputFormat), matching the OTHER pipeline
    // variant Init() built (see this class' own DebugTextOverlay::Init() call site) -- so the
    // default below covers m_PostProcess and only the two HDR branches need to override it.
    VkFormat overlayTargetFormat = SDFRayMarchPass::kOutputFormat;

    if (blitSourceImage == m_PostProcess.GetOutputImage()) {
        overlayTargetView = m_PostProcess.GetOutputView();
        overlayExtent = m_DisplayExtent;
    }
#ifndef NDEBUG
    else if (blitSourceImage == m_SDFRayMarch.GetOutputImage()) {
        overlayTargetView = m_SDFRayMarch.GetOutputView();
        overlayExtent = m_RenderExtent;
    }
    else if (blitSourceImage == m_DebugBufferView.GetOutputImage()) {
        // Also RGBA8_UNORM (DebugBufferViewPass::kOutputFormat), matching the default above --
        // only the view/extent need overriding.
        overlayTargetView = m_DebugBufferView.GetOutputView();
        overlayExtent = m_DisplayExtent;
    }
#endif
    else if (blitSourceImage == m_GIComposite.GetOutputImage()) {
        overlayTargetView = m_GIComposite.GetOutputView();
        overlayExtent = m_RenderExtent;
        overlayTargetFormat = debug::DebugTextOverlay::kHdrTargetFormat;
    }
    else {
        overlayTargetView = m_Resolve.GetOutputColorView();
        overlayExtent = m_RenderExtent;
        overlayTargetFormat = debug::DebugTextOverlay::kHdrTargetFormat;
    }

    m_DebugOverlay.RecordDraw(cmdLate, overlayTargetImage, overlayTargetView, overlayExtent, overlayTargetFormat);
  }
#endif

  // =========================================================================================
  // [14] Blit the final image (Phase PP1's m_PostProcess output in the normal view path -- itself
  // always sourced from m_TAATSR's own upscaled HDR output, see [13e] above -- or one of the
  // debug-view substitutes below) to the swapchain image and hand it to present. Every candidate
  // image stays in GENERAL (valid blit source layout); each one's own trailing barrier already
  // targeted the BLIT stage, but is re-stated here explicitly for the same reason every other
  // cross-pass barrier in this function is: the dependency must not silently vanish if an
  // intermediate pass's barriers change.
  // =========================================================================================
  // DEBUG_VIEW_LUMEN and DEBUG_VIEW_SPATIAL_PROBES both swap the blit SOURCE to m_GIComposite's
  // own output image instead of m_PostProcess's tonemapped one -- GIComposite.comp's own viewMode
  // 13/14 branches (see that shader's own comment) already picked which visualization filled it
  // this frame, so no separate per-mode image is needed here (and these are raw HDR-linear debug
  // visualizations, deliberately NOT run through Phase PP1's tonemap/gamma chain -- see [13e]'s own
  // comment on why those raw diagnostic visualizations are deliberately not tonemapped). DEBUG_VIEW_
  // GLOBAL_SDF instead swaps to m_SDFRayMarch's own debug-visualization image -- see
  // ClusterRenderPipeline.h's own comment on why this (not sampling it from within
  // ClusterResolve.comp) is the chosen mechanism. Every candidate image stays in GENERAL layout
  // (formats differ -- m_PostProcess/m_SDFRayMarch are RGBA8_UNORM, m_GIComposite/m_Resolve are
  // rgba16f HDR, see ClusterResolvePass::kOutputColorFormat's own comment -- but vkCmdBlitImage
  // performs the format conversion itself, so the SAME barrier below, targeting COMPUTE_SHADER_BIT/
  // SHADER_STORAGE_WRITE_BIT -> BLIT_BIT/TRANSFER_READ_BIT, still covers any choice with no changes
  // needed -- only which VkImage is actually blitted differs.
  // Denoised (see [12d] above) unless a debug view substitutes a different image entirely
  // (DEBUG_VIEW_LUMEN/DEBUG_VIEW_SPATIAL_PROBES/DEBUG_VIEW_GLOBAL_SDF all route through
  // m_GIComposite or m_SDFRayMarch instead of m_PostProcess's tonemapped output) -- applyDenoise
  // (computed at [12d]) tracks exactly the same DEBUG_VIEW_LUMEN condition for whether m_Denoiser
  // itself ran this frame.
  VkImage blitSourceImage = m_PostProcess.GetOutputImage();
#ifndef NDEBUG
  if (cameraCopy.debugViewMode != DEBUG_VIEW_NORMAL && cameraCopy.debugViewMode != DEBUG_VIEW_MOTION_VECTORS) {
      if (cameraCopy.debugViewMode == DEBUG_VIEW_LUMEN || cameraCopy.debugViewMode == DEBUG_VIEW_SPATIAL_PROBES) {
          blitSourceImage = m_GIComposite.GetOutputImage();
      } else if (cameraCopy.debugViewMode == DEBUG_VIEW_GLOBAL_SDF) {
          blitSourceImage = m_SDFRayMarch.GetOutputImage();
      } else {
          blitSourceImage = m_Resolve.GetOutputColorImage();
      }
  }
  // ImGui "Buffer Viewer" dropdown -- same override as the HUD-targeting selection above (the
  // dispatch itself already happened earlier at [13a.5], see that call site's own comment).
  if (config::debugview::SELECTED_BUFFER_INDEX != 0) {
      blitSourceImage = m_DebugBufferView.GetOutputImage();
  }
#endif

  VkExtent2D blitSrcExtent = m_RenderExtent;
  if (blitSourceImage == m_PostProcess.GetOutputImage()) {
      blitSrcExtent = m_DisplayExtent;
  }
#ifndef NDEBUG
  else if (blitSourceImage == m_DebugBufferView.GetOutputImage()) {
      blitSrcExtent = m_DisplayExtent;
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
    vkCmdPipelineBarrier2(cmdLate, &depInfo);
  }

  VkImageBlit blitRegion{};
  blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blitRegion.srcOffsets[1] = {static_cast<int32_t>(blitSrcExtent.width),
                              static_cast<int32_t>(blitSrcExtent.height), 1};
  blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  blitRegion.dstOffsets[1] = {static_cast<int32_t>(m_DisplayExtent.width),
                              static_cast<int32_t>(m_DisplayExtent.height), 1};
  // Same extent both sides -- the "blit" is a 1:1 copy that also performs the
  // RGBA8 -> B8G8R8A8 component reordering the swapchain format requires.
  vkCmdBlitImage(cmdLate, blitSourceImage, VK_IMAGE_LAYOUT_GENERAL,
                 swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &blitRegion, VK_FILTER_NEAREST);


#ifndef NDEBUG
  {
    VkImageMemoryBarrier2 attachmentBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    attachmentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
    attachmentBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    attachmentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    attachmentBarrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    attachmentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    attachmentBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    attachmentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    attachmentBarrier.image = swapchainImage;
    attachmentBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &attachmentBarrier;
    vkCmdPipelineBarrier2(cmdLate, &depInfo);
  }

  // Draw ImGui onto the swapchain image attachment
  VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  colorAttachment.imageView = swapchainImageView;
  colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Load the blitted content
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

  VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
  renderingInfo.renderArea = {{0, 0}, m_DisplayExtent};
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments = &colorAttachment;

  vkCmdBeginRendering(cmdLate, &renderingInfo);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdLate);
  vkCmdEndRendering(cmdLate);

  {
    VkImageMemoryBarrier2 presentBarrier{
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    presentBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    presentBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    presentBarrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    presentBarrier.dstAccessMask = 0;
    presentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    presentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    presentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    presentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    presentBarrier.image = swapchainImage;
    presentBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &presentBarrier;
    vkCmdPipelineBarrier2(cmdLate, &depInfo);
  }
#else
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
    vkCmdPipelineBarrier2(cmdLate, &depInfo);
  }
#endif

  // Recorded last, after every consumer of "the previous frame's" viewProj above (m_Resolve's own
  // motion-vector debug view) has already read it -- this frame's own matrix becomes "the previous
  // frame's" for NEXT frame's own RecordFrameMid()/RecordFrameLate() calls.
  m_PrevViewProj = viewProj;
  m_HasPrevViewProj = true;
  ++m_FrameIndex;
}

} // namespace renderer
