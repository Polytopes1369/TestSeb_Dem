#pragma once
// Phase 7a (UE5.8 parity roadmap, hero asset tessellation): forward-rendered, screen-space-
// adaptively-tessellated, procedurally-displaced hero asset pass -- the first tessellation-stage
// pipeline in this codebase. Real UE5.8 Nanite Tessellation also renders through a dedicated,
// non-clustered path for tessellated objects (Nanite's own cluster/VisBuffer model has no
// representation for runtime-subdivided, displaced geometry). This class mirrors
// renderer::TransparentForwardPass's own established template closely (same borrowed resources,
// same hand-rolled pipeline construction, same "never appears in ClusterResolvePass's own
// GBuffer, so this pass does its own inline direct+indirect+MegaLights+reflection lighting"
// constraint) -- it draws exactly the entity VulkanContext::BuildEntityData() assigned
// renderer::kHeroMaterialID and forced core::EntityFlags::IsTransparent (excluding it from the
// opaque Nanite path entirely, see that function's own comment), directly against its existing
// Fallback Mesh geometry (renderer::SurfaceCachePass's combined vertex/index buffers -- no new
// geometry upload), lit by direct (shadowed sun) + MegaLights (RIS-selected point lights, same
// inlined algorithm as TransparentForward.frag) + indirect diffuse (World Probe Grid) + an
// optional (per-material hasReflections-gated) front-layer specular reflection (same GGX-VNDF
// single-sample technique as TransparentForwardPass), and tessellated/displaced via a real
// 4-stage graphics pipeline (vertex -> tessellation control -> tessellation evaluation ->
// fragment) instead of the usual 2-stage vertex/fragment pipeline every other rasterized pass in
// this codebase uses.
//
// --- Difference from TransparentForwardPass: OPAQUE, writes real depth ---
// Unlike glass (blendEnable = true, depthWriteEnable = false, read-only against the opaque
// scene's depth), this pass is fully opaque (blendEnable = false) AND writes real depth
// (depthWriteEnable = true) -- necessary so renderer::TransparentForwardPass's own depth TEST,
// which runs immediately afterward in ClusterRenderPipeline::RecordFrame's own step, correctly
// occludes the glass/translucent entities against this pass' real (tessellated/displaced) surface
// instead of against nothing at all (this entity's clusters never reach any opaque raster list,
// so without this pass writing depth here, nothing would occupy its screen footprint in the depth
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
// to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for its own scope and back to READ_ONLY afterward.
//
// --- No runtime capability query ---
// patchControlPoints=3 and every tessellation level this pass' own .tesc shader emits stay
// comfortably under the Vulkan-guaranteed minimums (maxTessellationPatchSize >= 32,
// maxTessellationGenerationLevel >= 64) -- no VkPhysicalDeviceProperties query needed before
// relying on them, matching this codebase's existing "core Vulkan 1.0 guarantee, don't query"
// rigor for other fixed-function limits.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/LightingTypes.h"
#include "renderer/MaterialParameterTable.h" // MaterialParameters
#include "renderer/passes/SurfaceCachePass.h" // EntityDrawRange

namespace renderer {

    class VirtualShadowMapPass;
    class WorldProbeGridPass;
    class SurfaceCacheTraceContext;

    class HeroTessellationPass {
    public:
        HeroTessellationPass() = default;

        HeroTessellationPass(const HeroTessellationPass&) = delete;
        HeroTessellationPass& operator=(const HeroTessellationPass&) = delete;

        // Same borrowed-resource contract as renderer::TransparentForwardPass::Init -- see that
        // class' own parameter-by-parameter comment; `heroEntityDrawRange`/`heroEntityID` name the
        // hero entity (VulkanContext::kHeroEntityIndex, the Icosphere) instead of every transparent
        // entity. Unlike TransparentForwardPass (which uploads the FULL kMaxMaterials-sized table,
        // since it draws many different materials), this pass only ever shades ONE fixed material
        // -- `heroMaterial` (the caller's own materialTable.params[renderer::kHeroMaterialID]) is
        // copied once into this pass' own single-element GPU buffer at Init() time, no per-draw
        // materialID lookup needed. `lightBuffer`/`lightBufferSize` are renderer::MegaLightsPass::
        // GetLightBufferHandle()/GetLightBufferSize() (same MegaLights RIS point-light shading as
        // TransparentForward.frag, see that shader's own header comment); `tlasHandle` is shared
        // between MegaLights' own TraceShadowRay and this pass' own optional reflection trace's
        // TraceHWRT, same one-TLAS-two-consumers convention TransparentForwardPass already
        // established. `vsm`/`worldProbes`/`traceContext` must all already be Init'd and must
        // outlive this pass.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkFormat colorFormat, VkFormat depthFormat,
            VkBuffer entityTransformBuffer,
            const MaterialParameters& heroMaterial,
            VkBuffer fallbackVertexBuffer, VkBuffer fallbackIndexBuffer,
            const SurfaceCachePass::EntityDrawRange& heroEntityDrawRange, uint32_t heroEntityID,
            VkAccelerationStructureKHR tlasHandle, VkBuffer drawRangeBuffer,
            VkBuffer lightBuffer, VkDeviceSize lightBufferSize,
            const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
            const SurfaceCacheTraceContext& traceContext);

        void Shutdown();

        // Records the hero entity's single indexed draw (reinterpreted as PATCH_LIST triangle
        // patches, patchControlPoints=3 -- see this class' own header comment) into
        // `colorImage`/`colorView` (LOAD op, opaque -- overwrites whatever pixels its silhouette
        // covers), depth-testing AND WRITING against `depthImage`/`depthView`. Both images must be
        // in the layouts this method's own header comment documents on entry; this call leaves
        // them back in those same layouts on return (color: GENERAL, depth:
        // DEPTH_STENCIL_READ_ONLY_OPTIMAL), so the caller needs no additional layout bookkeeping
        // around this call. `traceMode`/`frameIndex`/`worldProbes` -- identical per-frame
        // conventions to renderer::TransparentForwardPass::RecordDraw, see that method's own
        // comment. `sceneLights` -- only `.sun` is read (MegaLights' own traced shadow ray covers
        // point lights, see class comment), refreshed into this pass' own small view-params UBO
        // every call.
        void RecordDraw(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
            VkImage colorImage, VkImageView colorView, VkImage depthImage, VkImageView depthView,
            VkExtent2D extent, uint32_t traceMode, uint32_t frameIndex,
            const SurfaceCacheTraceContext& traceContext, const WorldProbeGridPass& worldProbes,
            const SceneLights& sceneLights);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        SurfaceCachePass::EntityDrawRange m_HeroEntityDrawRange{};
        uint32_t m_HeroEntityID = 0;
        VkBuffer m_FallbackVertexBuffer = VK_NULL_HANDLE; // Borrowed.
        VkBuffer m_FallbackIndexBuffer = VK_NULL_HANDLE;  // Borrowed.

        // World-unit displacement amplitude -- deliberately a private compile-time-ish constant
        // (set once in the .cpp, not exposed via MaterialParameters) since exactly one entity uses
        // it today; a future multi-hero-entity generalization would promote this to a per-entity
        // parameter (see this pass' own class comment / the phase's own "hors scope" note).
        static constexpr float kDisplacementScale = 0.06f;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE; // set 0: this pass' own 14 bindings.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        // Binding 0 -- this pass' own small view/sun-lighting UBO (view/proj unused here -- the
        // push constants already carry viewProj/cameraPos for the vertex/tessellation stages --
        // only sunDirectionAndIntensity/sunColor are actually read, by the fragment stage's
        // ComputeDirectLighting). Refreshed every RecordDraw() call, mirrors
        // renderer::TransparentForwardPass's own m_ViewParamsBuffer convention.
        GpuBuffer m_ViewParamsBuffer;

        // Binding 7 -- this pass' own single-element MaterialParams buffer (see Init()'s own
        // `heroMaterial` parameter comment). Written once at Init() time -- the hero material
        // never changes at runtime.
        GpuBuffer m_MaterialParamsBuffer;

        // Binding 6 -- world_probe_sampling.glsl's WorldProbeGridParamsUBO. Persistently mapped,
        // refreshed every RecordDraw() call (gridOrigin recenters every frame) -- see RecordDraw's
        // own doc comment, mirrors renderer::TransparentForwardPass's own identical member.
        GpuBuffer m_WorldProbeGridParamsBuffer;

        // Pipeline layout spans 3 sets: set 0 (m_SetLayout, above), set 1 (mesh SDF trace scene)
        // and set 2 (surface cache sampling), both borrowed unmodified from the caller's
        // SurfaceCacheTraceContext -- same 3-set shape renderer::TransparentForwardPass/
        // renderer::ReflectionPass's own trace stage already establish.
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE; // 4-stage: vertex + tessellation control + tessellation evaluation + fragment.
    };

}
