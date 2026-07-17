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
//
// --- Phase 5 integration (UE5.8 parity roadmap, translucency) ---
// Shading upgraded from flat sun-only Lambertian to: direct lighting (sun + point lights, both
// shadowed, matching SurfaceCacheCapture.frag's own ComputeDirectLighting), indirect diffuse from
// the World Probe Grid (applied to every transparent material -- cheap, one 3D texture sample),
// and a traced front-layer specular reflection (Lumen-style, HWRT/SWRT, GGX-VNDF single sample)
// gated PER-MATERIAL by renderer::MaterialParameters::hasReflections -- matching real UE5.8's
// "Output Reflections" material toggle, not applied to every transparent surface uniformly. See
// TransparentForward.frag's own header comment for the exact composite formula.

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/GpuImage.h"
#include "renderer/MaterialParameterTable.h"
#include "renderer/LightingTypes.h"

namespace renderer {

    class VirtualShadowMapPass;
    class WorldProbeGridPass;
    class SurfaceCacheTraceContext;

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
        // RecordDraw() will later target -- renderer::GICompositePass::kOutputFormat (the image this
        // pass actually draws onto, see RecordDraw()'s own call site, NOT renderer::ClusterResolvePass::
        // kOutputColorFormat) and the shared hardware depth image's format, respectively.
        //
        // Phase 5 additions, all consumed by TransparentForward.frag's new shading (see class
        // comment): `worldProbes` for indirect-diffuse sampling (GetGridView()/GetGridSampler()/
        // GetGridOriginWorld()); `tlas`/`fallbackVertexBuffer`/`fallbackIndexBuffer`/
        // `drawRangeBuffer` for the optional (per-material `hasReflections`) HWRT/SWRT reflection
        // trace -- same buffers renderer::ReflectionTrace.comp's own HWRT path borrows (renderer::
        // SurfaceCacheRayTracingPass::GetTLASHandle()/GetDrawRangeBuffer(), renderer::
        // SurfaceCachePass::GetVertexBuffer()/GetIndexBuffer()); `traceContext` for its two
        // ready-made SWRT descriptor-set LAYOUTS (mesh_sdf_trace set 1, surface_cache_sampling set
        // 2) -- fixed at pipeline-layout-creation time here, same convention as renderer::
        // ReflectionPass::Init()'s own identical 3-set pipeline layout; the actual VkDescriptorSets
        // are re-bound every RecordDraw() call, not stored.
        //
        // MegaLights Phase A follow-up (reconciled with the above -- see class comment): `tlas` is
        // the SAME renderer::SurfaceCacheRayTracingPass::GetTLASHandle() already listed above, bound
        // ONCE at binding 11 and used by BOTH the reflection trace's TraceHWRT and MegaLights' own
        // TraceShadowRay -- no second TLAS binding. `lightBuffer`/`lightBufferSize` are renderer::
        // MegaLightsPass::GetLightBufferHandle()/GetLightBufferSize(), bound at binding 12,
        // consumed by TransparentForward.frag's inlined RIS point-light selection (megalights_ris
        // .glsl's SelectLightRIS -- translucent surfaces have no GBuffer entry for MegaLightsPass's
        // own compute-pass composite to reach, so this mirrors that composite's algorithm inline
        // instead). Both must already exist by the time this is called -- the caller must Init()
        // renderer::SurfaceCacheRayTracingPass and renderer::MegaLightsPass first (see renderer::
        // ClusterRenderPipeline::Init()'s own ordering comment). Per-entity point-light shadowing
        // (VSM cube faces) is deliberately NOT bound here any more -- MegaLights' own traced shadow
        // ray supersedes it for this pass, see TransparentForward.frag's own header comment.
        // Phase PP3 (post-process stack roadmap): `renderExtent` sizes this pass' own newly-owned
        // g_RefractionOffset image (kRefractionOffsetFormat, RG16F, render resolution) -- created
        // and transitioned to VK_IMAGE_LAYOUT_GENERAL unconditionally, BEFORE the transparent-
        // leaf-cluster walk's own possible zero-cluster early-return, since renderer::PostProcessPass
        // needs a valid view to sample every frame regardless of whether any transparent geometry
        // exists this run (see GetRefractionOffsetView()'s own comment).
        //
        // `splineControlPointsBuffer` is renderer::ClusterRenderPipeline's own
        // m_SplineControlPointsBuffer (see ClusterHardwareRasterPass::Init's identical parameter
        // comment) -- bound read-only, vertex-stage-only, at the first free slot past this pass's
        // full pre-existing binding range (see m_ForwardSetLayout's own binding-layout comment in
        // the .cpp for the exact number, which may have shifted due to the Substrate/PP3 merge) so
        // TransparentForward.vert can evaluate ApplySplineDeformation() for entities with
        // core::EntityFlags::HasSplineDeformation set (Phase 1, Nanite advanced).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent,
            VkBuffer pageTableBuffer, VkBuffer compressedPhysicalPoolBuffer,
            VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer wpoGlobalsBuffer,
            const std::array<MaterialParameters, kMaxMaterials>& materialTable,
            const std::vector<geometry::ClusterIndexEntry>& indexEntries,
            const std::vector<geometry::DAGNodeEntry>& dagEntries,
            VkFormat colorFormat, VkFormat depthFormat,
            const WorldProbeGridPass& worldProbes, const SurfaceCacheTraceContext& traceContext,
            VkAccelerationStructureKHR tlas, VkBuffer lightBuffer, VkDeviceSize lightBufferSize,
            VkBuffer fallbackVertexBuffer, VkBuffer fallbackIndexBuffer, VkBuffer drawRangeBuffer,
            VkBuffer splineControlPointsBuffer);

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
        // ONLY_OPTIMAL), issues one vkCmdDrawIndexedIndirectCount covering every static entry (a
        // zero-indexCount command is a defined Vulkan no-op for any entry whose page isn't resident
        // this frame; *Count, not the plain indirect draw, since m_ClusterCount can exceed 1 and
        // this device does not enable multiDrawIndirect -- see m_DrawCountBuffer's own comment),
        // ends rendering, and transitions `colorImage` back to GENERAL. `colorImage`/`colorView`
        // must be whichever image the caller is about to blit to the swapchain this frame (see
        // class comment) -- must have VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT (both of this pipeline's
        // candidate images already do, see renderer::ClusterResolvePass::Init()'s own comment on
        // why). `decompressedIndexPoolBuffer` is renderer::GeometryDecompressionPass::
        // GetDecompressedIndexPoolBuffer() -- rebound every call since which buffer is current can
        // change as pages stream in/out, matching ClusterHardwareRasterPass::RecordDraw()'s own
        // convention. `cameraPositionWorld` -- feeds the optional reflection trace's view-direction
        // reconstruction (fragment stage; `view`/`proj` are vertex-stage only, see TransparentForward
        // .vert). `sceneLights` -- only `.sun` is read now (MegaLights' own traced shadow ray
        // supersedes this pass's own point-light shadowing, see class comment); packed into
        // TransparentViewParamsUBO every call, consumed by the fragment stage's ComputeDirectLighting
        // (see TransparentForward.frag's own comment). `traceContext` -- the SAME instance passed to
        // Init() (fixed set LAYOUTS there; here, its current VkDescriptorSets are bound, since the
        // two never change after that context's own one-time build). `traceMode` (0 = SWRT, 1 = HWRT)
        // / `frameIndex` -- pushed as push-constants, consumed by both the optional per-material
        // reflection trace (renderer::ReflectionPass::RecordTrace's identical convention) and
        // MegaLights' own RIS candidate decorrelation (megalights_ris.glsl's SelectLightRIS).
        // `globalTimeSeconds` (Phase PP3): feeds TransparentForward.frag's own procedural, animated
        // refraction-offset noise (see that shader's end-of-main() comment) -- packed into
        // TransparentViewParamsUBO's own `globalTime` field (repurposing what used to be pure std140
        // padding after cameraPositionWorld).
        void RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView, VkImageView depthView,
            VkExtent2D renderExtent, const maths::mat4& view, const maths::mat4& proj,
            VkBuffer decompressedIndexPoolBuffer, const maths::vec3& cameraPositionWorld,
            const SceneLights& sceneLights, float globalTimeSeconds,
            const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, uint32_t frameIndex);

        uint32_t GetTransparentClusterCount() const { return m_ClusterCount; }

        // Phase PP3: the second (RG16F) color attachment TransparentForward.frag writes a per-
        // material procedural refraction offset into (see MaterialParameters::heatDistortion's own
        // comment) -- always valid (created unconditionally in Init(), see that method's own
        // comment), cleared to (0,0) every RecordDraw() call before this pass' own draw, and stays
        // VK_IMAGE_LAYOUT_GENERAL between frames, ready for renderer::PostProcessPass's composite
        // shader to sample directly.
        VkImageView GetRefractionOffsetView() const { return m_RefractionOffsetImage.View(); }

    private:
        static constexpr uint32_t kCompactWorkgroupSize = 64; // Matches TransparentClusterCompact.comp's local_size_x.
        static constexpr VkFormat kRefractionOffsetFormat = VK_FORMAT_R16G16_SFLOAT;

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        uint32_t m_ClusterCount = 0;

        // Phase PP3: owned unconditionally (see Init()'s own comment), unlike every buffer below it
        // (which only exist when m_ClusterCount > 0).
        GpuImage m_RefractionOffsetImage;

        GpuBuffer m_ClusterEntriesBuffer;  // TransparentClusterEntry[m_ClusterCount], std430, GPU_ONLY. Written once at Init().
        GpuBuffer m_IndirectCommandsBuffer; // VkDrawIndexedIndirectCommand[m_ClusterCount], std430|INDIRECT, GPU_ONLY. Rewritten every frame.
        // Constant uint32_t = m_ClusterCount, written once at Init(). vkCmdDrawIndexedIndirectCount's
        // required count SOURCE buffer -- m_ClusterCount is fixed after Init() (no per-frame LOD/
        // culling ever changes it, see class comment), so this never needs rewriting, but the count
        // must still come from a GPU buffer (there is no vkCmdDrawIndexedIndirect overload that takes
        // a >1 host-known drawCount without the multiDrawIndirect feature, which this device does not
        // enable -- RecordDraw() previously hit exactly that cap, see its own updated comment).
        GpuBuffer m_DrawCountBuffer;
        GpuBuffer m_ViewParamsBuffer;       // TransparentViewParamsUBO (view, proj, full SceneLights), std140, GPU_ONLY.
        // MaterialParameters[kMaxMaterials], this pass's own copy of Init()'s `materialTable`
        // parameter -- see Init()'s own comment on why this isn't shared with ClusterResolvePass's
        // identical-content SSBO.
        GpuBuffer m_MaterialParamsBuffer;
        // World Probe Grid addressing (gridOrigin/probeSpacing/gridResolution), std140, GPU_ONLY --
        // written once at Init() (the grid's own addressing is static; only its texel CONTENTS
        // change frame to frame, sampled directly from renderer::WorldProbeGridPass's own image).
        GpuBuffer m_WorldProbeGridParamsBuffer;

        // --- Pipeline 1: TransparentClusterCompact.comp (per-frame physical-page resolve + residency). ---
        VkDescriptorSetLayout m_CompactSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_CompactDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_CompactPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CompactPipeline = VK_NULL_HANDLE;

        // --- Pipeline 2: TransparentForward.vert/.frag (the actual alpha-blended draw). Pipeline
        // layout has 3 sets: this pass's own m_ForwardSetLayout (set 0) + the SurfaceCacheTraceContext
        // instance passed to Init()'s own mesh_sdf_trace (set 1) / surface_cache_sampling (set 2)
        // layouts -- same 3-set shape as renderer::ReflectionPass::Init()'s own pipeline layout. ---
        VkDescriptorSetLayout m_ForwardSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_ForwardDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ForwardPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ForwardPipeline = VK_NULL_HANDLE;

        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE; // Shared by both descriptor sets above.
    };

}
