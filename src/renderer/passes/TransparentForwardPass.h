#pragma once
// Forward-rendered pass for translucent/transparent materials (core::EntityFlags::IsTransparent,
// renderer::MaterialParameters::alpha < 1.0) -- the counterpart, on this codebase's "why does this
// exist" axis, to the opaque Visibility Buffer pipeline (renderer::ClusterHardwareRasterPass/
// ClusterSoftwareRasterPass/ClusterResolvePass): a Visibility Buffer stores exactly one winning
// surface per pixel, so it can never represent "an opaque surface behind a translucent one" --
// matching real UE5.8, where Nanite only ever renders opaque/masked geometry and every translucent
// material goes through a completely separate forward renderer instead.
//
// --- Why this path deliberately skips LOD selection and occlusion culling ---
// The opaque pipeline's per-frame GPU LOD cut + two-phase HZB occlusion culling exists to keep
// a large, deep scene's draw cost bounded. The demo entities that can end up transparent (a random
// subset of renderer::VulkanContext's 12 small primitives -- never the floor, see
// VulkanContext::kFloorEntityIndex) are few, small, and always close to the camera in this sparse
// scene -- not worth a second copy of that whole machinery. Instead, Init() walks the SAME
// geometry::ClusterIndexEntry/DAGNodeEntry tables renderer::ClusterLODSelectionPass::Init() already
// receives, ONCE, and keeps a small STATIC list of every transparent entity's LEAF-level (level==0,
// i.e. full/finest detail -- no simplification) cluster. "Static" only covers cluster IDENTITY
// (clusterID/entityID/materialID/bounds/logicalPageID) -- a cluster's CURRENT physical page (and
// therefore its vertexOffset/firstIndex into the shared compressed pool) is NOT static, since the
// streaming system can page clusters in/out at any time; RecordDraw() re-resolves that, and checks
// residency, via a tiny per-frame compute dispatch (TransparentClusterCompact.comp) before the
// actual draw -- exactly the same physicalPageIndex/residency lookup ClusterLODCompact.comp already
// does per-frame for the opaque path, just without any LOD/frustum/occlusion decision attached.
//
// --- Rendering technique ---
// A real vertex+fragment graphics pipeline (not compute): fixed-function alpha blending is the
// correct tool for compositing a handful of overlapping transparent surfaces, and this codebase
// already has a proven precedent for "decode cluster vertices directly from the compressed pool
// inside a vertex shader, draw via a real index buffer" (renderer::ClusterHardwareRasterPass /
// ClusterRaster.vert) that TransparentForward.vert mirrors closely. Reuses the SAME decompressed
// index pool buffer (renderer::GeometryDecompressionPass) the opaque hardware path already
// maintains -- no separate index buffer of its own.
//
// Depth-tested (VK_COMPARE_OP_GREATER, reversed-Z, matching every other pass in this pipeline)
// against the shared hardware depth image -- already VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
// by the point RecordDraw() is called (see renderer::ClusterRenderPipeline::RecordFrame's own
// ordering) -- but never WRITES depth, so transparent surfaces never occlude each other via the
// depth buffer; ordering between different transparent entities instead relies on whatever order
// the tiny per-frame compute dispatch above happens to iterate them in (this codebase's simplest
// demo scene has at most a handful of transparent entities at a time -- true back-to-front sorting
// between them is a known, deliberately accepted simplification, not implemented here).
//
// Renders directly into whichever image renderer::ClusterRenderPipeline::RecordFrame is about to
// blit to the swapchain this frame (m_Resolve's own output color image, or m_Denoiser's own output
// when denoising ran) -- see RecordDraw()'s own parameter comment -- so transparency composites on
// top of the fully-lit (GI/reflections already applied) opaque scene, the same layering order a
// deferred renderer with a forward transparency pass always uses.
//
// Exactly like every other piece of this Nanite-style pipeline, this class is a self-contained
// building block -- Init()/Shutdown()/RecordDraw() only.

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/MaterialParameterTable.h"

namespace renderer {

    class VirtualShadowMapPass;

    // CPU/GPU-shared, std430-compatible layout for this pass's own static candidate list --
    // deliberately NOT geometry::ClusterCullMetadata (no LOD/occlusion culling ever runs against
    // it, see class comment): only the fields TransparentClusterCompact.comp / TransparentForward.
    // vert/.frag actually read. 64 bytes, four 16-byte blocks.
    struct TransparentClusterEntry {
        maths::vec3 boundsMin; float maxWPOAmplitude = 0.0f;
        maths::vec3 boundsMax; float _pad0 = 0.0f;
        uint32_t logicalPageID = 0;
        uint32_t indexCount = 0;
        uint32_t clusterID = 0;
        uint32_t entityID = 0;
        uint32_t materialID = 0;
        uint32_t maskTextureIndex = 0xFFFFFFFFu;
        float _pad1 = 0.0f;
        float _pad2 = 0.0f;
    };
    static_assert(sizeof(TransparentClusterEntry) == 64,
        "TransparentClusterEntry must match TransparentClusterEntry in TransparentClusterCompact.comp/TransparentForward.vert exactly (std430 layout)");

    class TransparentForwardPass {
    public:
        TransparentForwardPass() = default;

        TransparentForwardPass(const TransparentForwardPass&) = delete;
        TransparentForwardPass& operator=(const TransparentForwardPass&) = delete;

        // Filters `indexEntries`/`dagEntries` (the exact same arrays renderer::
        // ClusterLODSelectionPass::Init() receives, index-aligned 1:1 per ClusterFormat.h's
        // on-disk contract) down to leaf-level (dagEntries[i].level == 0) clusters whose
        // `materialTable.isTransparent[indexEntries[i].materialID]` is true, and uploads the
        // resulting static list once. `pageTableBuffer`/`compressedPhysicalPoolBuffer` are
        // renderer::GpuGeometryPagePool's own (same handles every opaque pass already borrows);
        // `entityTransformBuffer`/`entityDataBuffer`/`wpoGlobalsBuffer` mirror ClusterRaster.vert's
        // own identically-purposed bindings. `colorFormat`/`depthFormat` must match whatever image
        // RecordDraw() will later target (renderer::ClusterResolvePass::kOutputColorFormat and the
        // shared hardware depth image's format, respectively). `tlas`/`lightBuffer`/
        // `lightBufferSize` (MegaLights Phase A follow-up) are renderer::SurfaceCacheRayTracingPass
        // ::GetTLASHandle() and renderer::MegaLightsPass::GetLightBufferHandle()/GetLightBufferSize()
        // -- both must already exist by the time this is called (the caller must Init() those two
        // passes first, see renderer::ClusterRenderPipeline::Init()'s own ordering comment), bound
        // once here at bindings 11/12 for TransparentForward.frag's own inline RIS shadow-ray shading
        // (megalights_ris.glsl's SelectLightRIS, same algorithm renderer::MegaLightsPass's opaque
        // path uses -- see that shader's own header comment for why this is inlined rather than a
        // separate compute pass: translucent surfaces have no GBuffer entry for one to read from).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkBuffer pageTableBuffer, VkBuffer compressedPhysicalPoolBuffer,
            VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer wpoGlobalsBuffer,
            const std::array<MaterialParameters, kMaxMaterials>& materialTable,
            const std::vector<geometry::ClusterIndexEntry>& indexEntries,
            const std::vector<geometry::DAGNodeEntry>& dagEntries,
            VkFormat colorFormat, VkFormat depthFormat,
            VkAccelerationStructureKHR tlas, VkBuffer lightBuffer, VkDeviceSize lightBufferSize);

        void Shutdown();

        // Binds Phase 3's renderer::VirtualShadowMapPass resources (physical page atlas + sampler,
        // page table, feedback buffer, sun clipmap levels UBO) -- same 4-resource contract as
        // renderer::ClusterResolvePass::SetVirtualShadowMap()/renderer::SurfaceCachePass's own.
        // Must be called exactly once after Init(), before the first RecordDraw().
        void SetVirtualShadowMap(const VirtualShadowMapPass& vsm);

        // No-op (returns immediately, no barriers/draw recorded) if Init() found zero transparent
        // leaf clusters -- a fixed-seed run can legitimately roll none. Otherwise: dispatches the
        // tiny per-frame compute step (resolve each static entry's CURRENT physical page + residency,
        // write one VkDrawIndexedIndirectCommand per entry -- see class comment), then draws:
        // transitions `colorImage` GENERAL -> COLOR_ATTACHMENT_OPTIMAL, begins rendering against it
        // (loadOp=LOAD, preserving the opaque scene already there) and `depthView` (loadOp=LOAD,
        // depthWriteEnable=FALSE -- read-only, must already be VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_
        // ONLY_OPTIMAL), issues one vkCmdDrawIndexedIndirect covering every static entry (a
        // zero-indexCount command is a defined Vulkan no-op for any entry whose page isn't resident
        // this frame), ends rendering, and transitions `colorImage` back to GENERAL. `colorImage`/
        // `colorView` must be whichever image the caller is about to blit to the swapchain this
        // frame (see class comment) -- must have VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT (both of this
        // pipeline's candidate images already do, see renderer::ClusterResolvePass::Init()'s own
        // comment on why). `decompressedIndexPoolBuffer` is renderer::GeometryDecompressionPass::
        // GetDecompressedIndexPoolBuffer() -- rebound every call since which buffer is current can
        // change as pages stream in/out, matching ClusterHardwareRasterPass::RecordDraw()'s own
        // convention. `sunDirection` -- see renderer::ClusterResolvePass::RecordResolve's identical
        // parameter comment (points FROM the light TOWARD the scene). `frameIndex` feeds the same
        // Halton-indexed RIS candidate decorrelation renderer::MegaLightsPass's opaque path uses
        // (megalights_ris.glsl's SelectLightRIS).
        void RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView, VkImageView depthView,
            VkExtent2D renderExtent, const maths::mat4& view, const maths::mat4& proj,
            VkBuffer decompressedIndexPoolBuffer, const maths::vec3& sunDirection, uint32_t frameIndex);

        uint32_t GetTransparentClusterCount() const { return m_ClusterCount; }

    private:
        static constexpr uint32_t kCompactWorkgroupSize = 64; // Matches TransparentClusterCompact.comp's local_size_x.

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        uint32_t m_ClusterCount = 0;

        GpuBuffer m_ClusterEntriesBuffer;  // TransparentClusterEntry[m_ClusterCount], std430, GPU_ONLY. Written once at Init().
        GpuBuffer m_IndirectCommandsBuffer; // VkDrawIndexedIndirectCommand[m_ClusterCount], std430|INDIRECT, GPU_ONLY. Rewritten every frame.
        GpuBuffer m_ViewParamsBuffer;       // TransparentViewParamsUBO (view, proj, sunDirection), std140, GPU_ONLY.
        // MaterialParameters[kMaxMaterials], this pass's own copy of Init()'s `materialTable`
        // parameter -- see Init()'s own comment on why this isn't shared with ClusterResolvePass's
        // identical-content SSBO.
        GpuBuffer m_MaterialParamsBuffer;

        // --- Pipeline 1: TransparentClusterCompact.comp (per-frame physical-page resolve + residency). ---
        VkDescriptorSetLayout m_CompactSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_CompactDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_CompactPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CompactPipeline = VK_NULL_HANDLE;

        // --- Pipeline 2: TransparentForward.vert/.frag (the actual alpha-blended draw). ---
        VkDescriptorSetLayout m_ForwardSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_ForwardDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ForwardPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ForwardPipeline = VK_NULL_HANDLE;

        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE; // Shared by both descriptor sets above.
    };

}
