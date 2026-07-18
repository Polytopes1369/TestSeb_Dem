#pragma once
// Phase 7a (UE5.8 parity roadmap, hero asset tessellation) -- GENERALIZED: forward-rendered,
// screen-space-adaptively-tessellated, procedurally-displaced pass for ANY entity flagged
// core::EntityFlags::IsTessellated (originally shipped as "HeroTessellationPass", hardcoded to
// exactly one entity -- the Icosphere -- via renderer::kHeroMaterialID; renamed and generalized to
// a per-entity opt-in flag, matching real UE5.8 Nanite Tessellation (5.5+), which applies to any
// flagged mesh, not one hardcoded hero asset). This class mirrors renderer::TransparentForwardPass's
// own established template closely (same borrowed resources, same hand-rolled pipeline
// construction, same "never appears in ClusterResolvePass's own GBuffer, so this pass does its own
// inline direct+indirect+MegaLights+reflection lighting" constraint) -- it draws every entity
// VulkanContext::BuildEntityData() flagged core::EntityFlags::IsTessellated (and, same mechanism as
// before, forced core::EntityFlags::IsTransparent to exclude each of them from the opaque Nanite
// path entirely, see that function's own comment), directly against its existing Fallback Mesh
// geometry (renderer::SurfaceCachePass's combined vertex/index buffers -- no new geometry upload,
// shared read-only across every tessellated entity), lit by direct (shadowed sun) + MegaLights
// (RIS-selected point lights, same inlined algorithm as TransparentForward.frag) + indirect diffuse
// (World Probe Grid) + an optional (per-material hasReflections-gated) front-layer specular
// reflection (same GGX-VNDF single-sample technique as TransparentForwardPass), and
// tessellated/displaced via a real 4-stage graphics pipeline (vertex -> tessellation control ->
// tessellation evaluation -> fragment) instead of the usual 2-stage vertex/fragment pipeline every
// other rasterized pass in this codebase uses.
//
// --- Generalization shape: ONE pipeline, MANY entities, ONE draw call per entity ---
// Unlike TransparentForwardPass (which compacts many entities' clusters into a single indirect
// multi-draw), this pass keeps the original hero-entity design's simplicity: one direct
// vkCmdDrawIndexed per tessellated entity, all sharing the SAME bound pipeline/descriptor sets/
// vertex+index buffers (only the push constants' entityID/materialID and the draw's own
// firstIndex/vertexOffset/indexCount change between entities) -- appropriate for this feature's
// small entity count (a handful of showcase entities, never the whole scene), and it avoids
// needing a per-frame compact/indirect-count compute step TransparentForwardPass only needs because
// its OWN candidate set can include a variable number of resident streaming clusters; every
// tessellated entity here is one of the small, always-resident showcase gallery, so a fixed static
// list built once at Init() (mirroring TransparentForwardPass's own "walk once, keep a static list"
// philosophy, see that class' own header comment) is sufficient -- no per-frame residency check
// needed either, unlike TransparentForwardPass's own paged compressed-cluster-pool consumers.
//
// The material lookup generalizes the same way TransparentForwardPass's own does: instead of one
// single-element MaterialParams buffer (the original hero-only shape, since exactly one material was
// ever shaded), this pass now uploads the FULL kMaxMaterials-sized table once at Init() (same
// SSBO shape/convention as TransparentForwardPass's own m_MaterialParamsBuffer) and each entity's
// draw call selects its own slot via a per-draw materialID push-constant field -- entities keep
// whatever materialID VulkanContext::BuildEntityData() already assigned them (their own showcase
// recipe, or a reserved slot like kHeroMaterialID for the original hero), no single-material
// special-casing left in this pass at all.
//
// --- Difference from TransparentForwardPass: OPAQUE, writes real depth ---
// Unlike glass (blendEnable = true, depthWriteEnable = false, read-only against the opaque
// scene's depth), this pass is fully opaque (blendEnable = false) AND writes real depth
// (depthWriteEnable = true) -- necessary so renderer::TransparentForwardPass's own depth TEST,
// which runs immediately afterward in ClusterRenderPipeline::RecordFrame's own step, correctly
// occludes the glass/translucent entities against this pass' real (tessellated/displaced) surfaces
// instead of against nothing at all (these entities' clusters never reach any opaque raster list,
// so without this pass writing depth here, nothing would occupy their screen footprint in the depth
// buffer this frame). See RecordDraw()'s own comment for the resulting WIDER barrier scope this
// requires on exit (TransparentForwardPass's own restore barrier is NOT sufficient to copy, since
// it only ever READ depth, never wrote it).
//
// --- Render target handling ---
// Renders directly into whichever image the caller hands it (renderer::ClusterRenderPipeline's own
// call site selects between the denoiser's output and renderer::ClusterResolvePass's own output
// color image, matching TransparentForwardPass's own call site convention) via a LOAD-op
// vkCmdBeginRendering, and against renderer::VulkanContext's own real depth-stencil image (a real
// depth-testable attachment, not a plain storage image) -- that image sits in
// VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL on entry (same contract
// renderer::TransparentForwardPass's own header comment documents), and this pass transitions it
// to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for its own scope (covering every tessellated entity's draw)
// and back to READ_ONLY afterward.
//
// --- No runtime capability query ---
// patchControlPoints=3 and every tessellation level this pass' own .tesc shader emits stay
// comfortably under the Vulkan-guaranteed minimums (maxTessellationPatchSize >= 32,
// maxTessellationGenerationLevel >= 64) -- no VkPhysicalDeviceProperties query needed before
// relying on them, matching this codebase's existing "core Vulkan 1.0 guarantee, don't query"
// rigor for other fixed-function limits.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/LightingTypes.h"
#include "renderer/MaterialParameterTable.h" // MaterialParameters, kMaxMaterials
#include "renderer/passes/SurfaceCachePass.h" // EntityDrawRange

namespace renderer {

    class VirtualShadowMapPass;
    class WorldProbeGridPass;
    class SurfaceCacheTraceContext;

    // One tessellated entity's identity for this pass: which Fallback Mesh draw range to issue
    // (its own indexed sub-range inside the SHARED combined vertex/index buffers -- see class
    // comment), which entityID to resolve its current rigid transform with (EntityTransformBuffer
    // index, == core::EntityData::meshID, matching renderer::TessellationPass.vert's own direct
    // indexing convention), and which materialID to shade it with (its own
    // renderer::MaterialParameters slot in the full kMaxMaterials-sized table this pass now
    // uploads -- see class comment).
    struct TessellatedEntityDrawInfo {
        SurfaceCachePass::EntityDrawRange drawRange{};
        uint32_t entityID = 0;
        uint32_t materialID = 0;
    };

    class TessellationPass {
    public:
        TessellationPass() = default;

        TessellationPass(const TessellationPass&) = delete;
        TessellationPass& operator=(const TessellationPass&) = delete;

        // Same borrowed-resource contract as renderer::TransparentForwardPass::Init -- see that
        // class' own parameter-by-parameter comment; `entities` names every tessellated entity
        // (core::EntityFlags::IsTessellated, see VulkanContext::BuildEntityData()) instead of one
        // hardcoded hero. `materialTable` is the caller's FULL kMaxMaterials-sized table (mirrors
        // TransparentForwardPass's own `materialTable` parameter exactly -- each entity's own draw
        // selects its own slot at RecordDraw() time via a push-constant materialID, see class
        // comment), copied once into this pass' own GPU SSBO at Init() time. `entities` may be
        // empty (a valid, if degenerate, configuration -- RecordDraw() becomes a no-op, same
        // defensive convention TransparentForwardPass's own zero-cluster case already establishes)
        // but is expected to be non-empty in this codebase's own showcase scene (see
        // VulkanContext::BuildEntityData()'s own IsTessellated assignments). `lightBuffer`/
        // `lightBufferSize` are renderer::MegaLightsPass::GetLightBufferHandle()/
        // GetLightBufferSize() (same MegaLights RIS point-light shading as TransparentForward.frag,
        // see that shader's own header comment); `tlasHandle` is shared between MegaLights' own
        // TraceShadowRay and this pass' own optional reflection trace's TraceHWRT, same
        // one-TLAS-two-consumers convention TransparentForwardPass already established.
        // `vsm`/`worldProbes`/`traceContext` must all already be Init'd and must outlive this pass.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkFormat colorFormat, VkFormat depthFormat,
            VkBuffer entityTransformBuffer,
            const std::array<MaterialParameters, kMaxMaterials>& materialTable,
            VkBuffer fallbackVertexBuffer, VkBuffer fallbackIndexBuffer,
            const std::vector<TessellatedEntityDrawInfo>& entities,
            VkAccelerationStructureKHR tlasHandle, VkBuffer drawRangeBuffer,
            VkBuffer lightBuffer, VkDeviceSize lightBufferSize,
            const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
            const SurfaceCacheTraceContext& traceContext);

        void Shutdown();

        // No-op (returns immediately, no barriers/draw recorded) if Init() was given an empty
        // `entities` list -- see Init()'s own comment. Otherwise records one indexed draw
        // (reinterpreted as PATCH_LIST triangle patches, patchControlPoints=3 -- see this class'
        // own header comment) PER tessellated entity, all within a single vkCmdBeginRendering/
        // vkCmdEndRendering scope, into `colorImage`/`colorView` (LOAD op, opaque -- overwrites
        // whatever pixels each entity's silhouette covers), depth-testing AND WRITING against
        // `depthImage`/`depthView`. Both images must be in the layouts this method's own header
        // comment documents on entry; this call leaves them back in those same layouts on return
        // (color: GENERAL, depth: DEPTH_STENCIL_READ_ONLY_OPTIMAL), so the caller needs no
        // additional layout bookkeeping around this call. `traceMode`/`frameIndex`/`worldProbes` --
        // identical per-frame conventions to renderer::TransparentForwardPass::RecordDraw, see that
        // method's own comment. `sceneLights` -- only `.sun` is read (MegaLights' own traced shadow
        // ray covers point lights, see class comment), refreshed into this pass' own small
        // view-params UBO once per call (shared across every entity's draw this frame, not
        // per-entity).
        void RecordDraw(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
            VkImage colorImage, VkImageView colorView, VkImage depthImage, VkImageView depthView,
            VkExtent2D extent, uint32_t traceMode, uint32_t frameIndex,
            const SurfaceCacheTraceContext& traceContext, const WorldProbeGridPass& worldProbes,
            const SceneLights& sceneLights);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        // Every tessellated entity's identity -- see TessellatedEntityDrawInfo's own comment.
        // Fixed after Init() (this codebase's own scene is fully deterministic, see
        // GenerateShowcaseMaterialTable()'s own "no RNG at all" comment) -- RecordDraw() iterates
        // this unchanged every frame, one vkCmdDrawIndexed per entry.
        std::vector<TessellatedEntityDrawInfo> m_Entities;
        VkBuffer m_FallbackVertexBuffer = VK_NULL_HANDLE; // Borrowed, shared across every entity.
        VkBuffer m_FallbackIndexBuffer = VK_NULL_HANDLE;  // Borrowed, shared across every entity.

        // World-unit displacement amplitude -- shared by every tessellated entity (a single global
        // knob, not promoted to a per-entity MaterialParameters field, since this codebase's
        // showcase scene has no need for entities to displace by different amounts to read clearly
        // -- a future authoring-driven per-entity amplitude would extend TessellatedEntityDrawInfo
        // with its own field rather than touching this constant).
        static constexpr float kDisplacementScale = 0.06f;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE; // set 0: this pass' own 14 bindings.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        // Binding 0 -- this pass' own small view/sun-lighting UBO (view/proj unused here -- the
        // push constants already carry viewProj/cameraPos for the vertex/tessellation stages --
        // only sunDirectionAndIntensity/sunColor are actually read, by the fragment stage's
        // ComputeDirectLighting). Refreshed once per RecordDraw() call (shared across every
        // entity's draw that frame), mirrors renderer::TransparentForwardPass's own
        // m_ViewParamsBuffer convention.
        GpuBuffer m_ViewParamsBuffer;

        // Binding 7 -- this pass' own MaterialParameters[kMaxMaterials] SSBO (see Init()'s own
        // `materialTable` parameter comment -- generalized from the original single-element hero
        // buffer). Written once at Init() time; each entity's draw selects its own slot via its
        // push-constant materialID field, matching TransparentForwardPass's own
        // m_MaterialParamsBuffer shape/convention exactly.
        GpuBuffer m_MaterialParamsBuffer;

        // Binding 6 -- world_probe_sampling.glsl's WorldProbeGridParamsUBO. Persistently mapped,
        // refreshed once per RecordDraw() call (gridOrigin recenters every frame) -- see
        // RecordDraw's own doc comment, mirrors renderer::TransparentForwardPass's own identical
        // member.
        GpuBuffer m_WorldProbeGridParamsBuffer;

        // Pipeline layout spans 3 sets: set 0 (m_SetLayout, above), set 1 (mesh SDF trace scene)
        // and set 2 (surface cache sampling), both borrowed unmodified from the caller's
        // SurfaceCacheTraceContext -- same 3-set shape renderer::TransparentForwardPass/
        // renderer::ReflectionPass's own trace stage already establish.
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE; // 4-stage: vertex + tessellation control + tessellation evaluation + fragment.
    };

}
