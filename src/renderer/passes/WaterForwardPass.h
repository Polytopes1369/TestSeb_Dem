#pragma once
// Phase 7c (UE5.8 parity roadmap, water/erosion): forward-rendered water plane -- mirrors
// renderer::TessellationPass's own STRUCTURE (Init/RecordDraw, hand-built graphics pipeline,
// GENERAL<->ATTACHMENT barrier dance, single-element MaterialParams buffer) without being a
// generalization of it. This pass composes MANUALLY in-shader against a frozen snapshot of the
// background (blendEnable=false, see RecordDraw's own comment), unlike TransparentForwardPass's
// fixed-function alpha blending -- an incompatible pipeline state within one VkPipeline object, so
// generalizing either existing forward pass would still need two pipelines (no gain) or a runtime
// shader branch adding divergence for nothing. This class also needs neither VirtualShadowMapPass
// nor WorldProbeGridPass nor MegaLights (water has no diffuse/shadowed term, only refraction + a
// front-layer specular reflection identical to hero/glass's own) -- set 0 is 7 bindings here versus
// TessellationPass's 14.
//
// --- Refraction mechanism ---
// This pipeline is forward-hybrid: the caller's own `colorImage` IS the final composited frame at
// the point this pass runs (recorded LAST, after TransparentForwardPass/TessellationPass --
// see renderer::ClusterRenderPipeline::RecordFrame's own call-site ordering), not an intermediate
// G-buffer, so there is no separate screen-space color buffer to sample for a classic deferred-
// style UV-offset refraction, and reading + writing the SAME image within one sub-pass is
// undefined behavior without framebuffer-fetch (unavailable here). Instead, RecordDraw() blits
// `colorImage` (still in VK_IMAGE_LAYOUT_GENERAL at that point) into `m_BackgroundSnapshotImage`, a
// private image owned by this pass, BEFORE beginning its own rendering -- mirrors the only other
// precedent in this codebase for reading `colorImage` as a transfer source while it sits in
// GENERAL (ClusterRenderPipeline's own final blit-to-swapchain step). The fragment shader then
// samples that frozen snapshot (never the live `colorImage`) for its refraction background, and
// composes reflection/refraction manually, writing the final result directly rather than relying
// on fixed-function blending.
//
// --- Render target handling ---
// Same as TransparentForwardPass/TessellationPass: renders into whichever color/depth images
// the caller hands it (LOAD-op vkCmdBeginRendering onto ClusterResolvePass's own output color
// image, real D32_SFLOAT depth), depth-tests READ-ONLY (depthWriteEnable=false, like glass, unlike
// TessellationPass -- water never needs to occlude anything drawn after it).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/RenderPass.h"
#include "renderer/MaterialParameterTable.h" // MaterialParameters
#include "renderer/passes/SurfaceCachePass.h" // EntityDrawRange

namespace renderer {

    class SurfaceCacheTraceContext;

    // Migrated to RenderPass<Derived> (see renderer/vulkan/RenderPass.h): Init()/Shutdown() are
    // inherited. Note InitImpl() still calls the inherited Shutdown() as its own first statement
    // (self-reinit, same pattern/reason as ShadowMapPass's own migration comment).
    class WaterForwardPass : public RenderPass<WaterForwardPass> {
        friend class RenderPass<WaterForwardPass>; // Lets Init() call our private InitImpl().

    public:
        WaterForwardPass() = default;

        WaterForwardPass(const WaterForwardPass&) = delete;
        WaterForwardPass& operator=(const WaterForwardPass&) = delete;

        // `commandPool`/`queue` are ACTUALLY used here (unlike TessellationPass::Init's own use
        // of them, both for one-shot uploads) -- for the one-shot UNDEFINED->SHADER_READ_ONLY_OPTIMAL
        // transition of m_BackgroundSnapshotImage, so RecordDraw() never needs a first-frame special
        // case (every frame sees the same precondition on entry). `colorFormat` must match
        // renderer::ClusterResolvePass::kOutputColorFormat exactly -- m_BackgroundSnapshotImage is
        // allocated with that same format/extent (`renderExtent`). `waterMaterial` (the caller's own
        // materialTable.params[renderer::kWaterMaterialID]) is copied once into this pass' own
        // single-element GPU buffer at Init() time, same convention renderer::TessellationPass
        // established. `fallbackVertexBuffer`/`fallbackIndexBuffer`, `waterEntityDrawRange`/
        // `waterEntityID`, `tlasHandle`/`drawRangeBuffer`, `traceContext` -- identical borrowed-
        // resource contract to TessellationPass::Init's own parameters of the same name.
        // Init(VkDevice, VmaAllocator, VkCommandPool, VkQueue, VkFormat, VkFormat, VkBuffer,
        // const MaterialParameters&, VkBuffer, VkBuffer, const SurfaceCachePass::EntityDrawRange&,
        // uint32_t, VkAccelerationStructureKHR, VkBuffer, const SurfaceCacheTraceContext&,
        // VkExtent2D) -> bool and Shutdown() are inherited from RenderPass<WaterForwardPass>; see
        // InitImpl() below.

        // Records the water entity's single indexed draw. Same layout contract as
        // TessellationPass::RecordDraw (color: GENERAL on entry/exit, depth:
        // DEPTH_STENCIL_READ_ONLY_OPTIMAL on entry/exit, never written -- see this class' own header
        // comment) PLUS an additional internal blit of `colorImage` into this pass' own background
        // snapshot before rendering -- entirely self-contained, no extra caller-side bookkeeping.
        // `timeSeconds` (ClusterRenderPipeline::RecordFrame's own `globalTimeSeconds` parameter)
        // drives the per-fragment wave normal perturbation (see WaterForward.frag).
        void RecordDraw(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
            VkImage colorImage, VkImageView colorView, VkImage depthImage, VkImageView depthView,
            VkExtent2D extent, uint32_t traceMode, uint32_t frameIndex, float timeSeconds,
            const SurfaceCacheTraceContext& traceContext);

    private:
        bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkFormat colorFormat, VkFormat depthFormat,
            VkBuffer entityTransformBuffer,
            const MaterialParameters& waterMaterial,
            VkBuffer fallbackVertexBuffer, VkBuffer fallbackIndexBuffer,
            const SurfaceCachePass::EntityDrawRange& waterEntityDrawRange, uint32_t waterEntityID,
            VkAccelerationStructureKHR tlasHandle, VkBuffer drawRangeBuffer,
            const SurfaceCacheTraceContext& traceContext, VkExtent2D renderExtent);

        // m_Device / m_Allocator are inherited (protected) from RenderPass<WaterForwardPass>.

        SurfaceCachePass::EntityDrawRange m_WaterEntityDrawRange{};
        uint32_t m_WaterEntityID = 0;
        VkBuffer m_FallbackVertexBuffer = VK_NULL_HANDLE; // Borrowed.
        VkBuffer m_FallbackIndexBuffer = VK_NULL_HANDLE;  // Borrowed.

        // Private snapshot of the composited scene color, refreshed every RecordDraw() call --
        // see this class' own header comment for the full refraction-mechanism rationale.
        VkImage m_BackgroundSnapshotImage = VK_NULL_HANDLE;
        VmaAllocation m_BackgroundSnapshotAllocation = VK_NULL_HANDLE;
        VkImageView m_BackgroundSnapshotView = VK_NULL_HANDLE;
        VkSampler m_BackgroundSnapshotSampler = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{};

        // Binding 0 -- this pass' own single-element MaterialParams buffer (see Init()'s own
        // `waterMaterial` parameter comment). Written once at Init() time -- the water material
        // never changes at runtime.
        GpuBuffer m_MaterialParamsBuffer;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE; // set 0: this pass' own 7 bindings.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        // Pipeline layout spans 3 sets: set 0 (m_SetLayout, above), set 1 (mesh SDF trace scene)
        // and set 2 (surface cache sampling), both borrowed unmodified from the caller's
        // SurfaceCacheTraceContext -- same 3-set shape TransparentForwardPass/TessellationPass
        // already establish.
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
