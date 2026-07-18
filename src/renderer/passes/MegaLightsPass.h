#pragma once
// Phase A of the MegaLights native-port roadmap (see the approved plan,
// C:\Users\Seb\.claude\plans\memoized-splashing-seal.md, for the full derivation): RIS-weighted
// stochastic multi-point-light direct lighting, one ray-traced shadow-visibility ray per pixel.
//
// Three-stage pipeline, all owned here -- Shade (MegaLightsShade.comp, writes RAW noisy radiance
// into its own image) -> denoise (an OWNED renderer::ATrousDenoisePass instance, reused as-is, zero
// new denoiser code written) -> Composite (MegaLightsComposite.comp, additively read-modify-writes
// renderer::ClusterResolvePass's own output color image, the same direct-RMW convention
// renderer::ReflectionPass's own RecordGather already uses).
//
// A DEDICATED denoiser instance is required here rather than reusing renderer::
// ClusterRenderPipeline's own shared [12d] ATrousDenoisePass instance the way Phase 2 reflections
// inherit it -- discovered mid-implementation, not part of the original plan text: `main` advanced
// under this branch (see this project's own "Concurrent main sessions" precedent) to wire up
// renderer::ScreenTracePass/renderer::GICompositePass, which redefined what [12d]'s shared instance
// actually denoises (Screen Trace GI specifically, per GICompositePass.h's own header comment) --
// no longer "the fully composited direct+indirect frame" as the plan's exploration had found on the
// pre-that-commit `main`. Reflections don't hit this problem because their own noise is cleaned by
// RecordTemporal's cross-frame accumulation instead, something Phase A deliberately had none of yet
// -- Phase 4 (this file's own current roadmap step) is what finally adds that: see this class' own
// "Phase 4" member comments below. MegaLights still keeps its dedicated À-Trous spatial filter on
// top of the new temporal reservoir reuse (the two are complementary, not redundant: temporal
// reuse fights single-frame variance across TIME, À-Trous fights whatever spatial noise remains in
// a single denoised frame).
//
// --- Phase 4 of the "Nanite advanced" roadmap (light BVH for RIS spatial bias, temporal ReSTIR
// with per-frame revalidated visibility -- see the approved plan) ---
// Feature 1: a CPU-built geometry::LightBVH (deliberately separate from geometry::EntityBVH, see
// that header's own comment) over this pass' own light population, built once right here in
// Init() after the light SSBO upload, biasing SelectLightRIS's candidate draw toward lights near
// the current shading point via 2 new Shade-set bindings (10/11) -- see megalights_bvh.glsl.
// Feature 2: temporal ReSTIR, folded directly into MegaLightsShade.comp (no new pass/dispatch --
// see that file's own header comment for why none is needed here, unlike ReflectionTemporal.comp).
// Adds a ping-ponged pair of GPU_ONLY MegaLightReservoir SSBOs (Shade-set bindings 12/13, current/
// history swapped per slot) -- mirrors renderer::ReflectionPass's own "two full descriptor-set
// variants, flip-first" ping-pong convention exactly (see m_ShadeSet's own member comment below).
// The mandatory shadow-visibility ray is still traced exactly once per pixel per frame, now toward
// the TEMPORALLY-COMBINED reservoir's winning light instead of the raw single-frame RIS winner.
//
// --- Spatial reuse follow-up (still Phase 4 of the "Nanite advanced" roadmap) ---
// RecordShade's single Shade dispatch above split into THREE compute dispatches, each separated by
// an explicit VkMemoryBarrier2 (see MegaLightsShade.comp/MegaLightsSpatialReuse.comp/
// MegaLightsFinalShade.comp's own header comments for the full per-stage derivation):
//   Stage 1 (MegaLightsShade.comp, pipeline members unchanged/reused as-is): BVH-biased RIS +
//     temporal ReSTIR combine only -- writes the TEMPORAL-only combined reservoir into
//     m_ReservoirBuffers[m_CurrentReservoirSlotIndex] (still the ping-pong pair, still what becomes
//     next frame's history), no shading, no shadow ray.
//   Stage 2 (MegaLightsSpatialReuse.comp, m_SpatialReuse* members below): reads that SAME buffer
//     (now barrier-visible) for this pixel and ~5 golden-angle-spaced screen-space neighbors,
//     streaming-resamples them into m_SpatialReservoirBuffer -- a single, non-ping-ponged, fully-
//     overwritten-every-frame buffer (deliberately never fed back into next frame's temporal
//     history, see MegaLightsShade.comp's own header comment on why).
//   Stage 3 (MegaLightsFinalShade.comp, m_FinalShade* members below): reads m_SpatialReservoirBuffer,
//     traces the one mandatory shadow-visibility ray toward its winning light, evaluates the
//     Substrate BSDF, and writes m_RawRadianceImage -- exactly what Stage 1 used to do at its own
//     tail before this split.
//
// --- Per-frame call order a caller must follow ---
//   1. RecordShade(cmd, viewProj, prevViewProj, cameraPositionWorld, frameIndex) -- needs
//      renderer::ClusterResolvePass's GBuffer (normal/depth/albedo/roughness-metallic) already
//      visible to COMPUTE_SHADER reads, and renderer::SurfaceCacheRayTracingPass's TLAS already
//      built. `prevViewProj` should be an identity matrix on the very first frame ever recorded
//      (see renderer::ClusterRenderPipeline's own m_HasPrevViewProj guard, same convention
//      ReflectionPass::RecordUpdateViewParams already documents) -- MegaLightsShade.comp's own
//      reservoir history fetch is additionally sentinel-gated (lightIndex == 0xFFFFFFFFu, the
//      zero-initialized fill pattern every reservoir slot starts at), so an identity prevViewProj
//      on frame 0 never actually risks reading garbage history, only ever a harmless miss. Runs
//      Stage 1 -> Stage 2 -> Stage 3 -> internal ATrous denoise -> Composite as one call (no
//      caller-visible per-stage state beyond the reservoir ping-pong flip, which this method
//      manages internally -- unlike ReflectionPass's own 3-call contract, there is nothing else
//      here for a caller to sequence).
//      Ends with a trailing VkMemoryBarrier2 making the composited color image visible to whatever
//      the caller records next (matches ReflectionPass::RecordGather's own contract).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/LightBVH.h"
#include "renderer/MegaLightsTypes.h"
#include "renderer/passes/ATrousDenoisePass.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class ClusterResolvePass;
    class SurfaceCacheRayTracingPass;

    class MegaLightsPass {
    public:
        MegaLightsPass() = default;

        MegaLightsPass(const MegaLightsPass&) = delete;
        MegaLightsPass& operator=(const MegaLightsPass&) = delete;

        static constexpr uint32_t kWorkgroupSize = 8; // Matches MegaLightsShade.comp/MegaLightsComposite.comp's local_size_x/y.

        // Must match renderer::ATrousDenoisePass::kFormat exactly (its ping-pong images and shader
        // are hardcoded to this format -- see MegaLightsShade.comp's own header comment for why
        // reusing that class forces this choice): R16G16B16A16_SFLOAT linear HDR, NOT rgba8 --
        // MegaLightsShade.comp's own radiance is a physically-based, inverse-square-falloff point-
        // light contribution with no upper bound near a light, so an 8-bit UNORM target would hard-
        // clip it to [0,1] before it ever reaches renderer::PostProcessPass, one of this codebase's own
        // "burned" overexposure bug's root causes (see ClusterResolvePass::kOutputColorFormat's own
        // comment, the image this pass' Composite stage additively read-modify-writes into).
        static constexpr VkFormat kRadianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        // `resolvePass`/`rtPass` must already be Init'd and outlive this pass -- borrowed,
        // unmodified, same convention as renderer::ReflectionPass::Init. `lightsData` is copied once
        // into an owned host-visible SSBO here, not retained by reference.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, const ClusterResolvePass& resolvePass,
            const SurfaceCacheRayTracingPass& rtPass, const MegaLightsData& lightsData);

        void Shutdown();

        // `cameraPositionWorld` (Substrate integration): feeds EvaluateSubstrateMaterial's specular/
        // Fresnel view-direction term in MegaLightsShade.comp -- the pre-Substrate shader had no
        // view-dependent term at all, so this is new; same value renderer::ClusterRenderPipeline
        // already threads into ReflectionPass/TransparentForwardPass/ClusterResolvePass's own
        // view-params UBOs.
        // `prevViewProj` (Phase 4, Feature 2): the PREVIOUS frame's own combined view-projection
        // matrix -- feeds MegaLightsShade.comp's reservoir reprojection, same
        // ReflectionPass::RecordUpdateViewParams convention (see this method's own header comment
        // for the frame-0 safety note).
        void RecordShade(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj,
            const maths::vec3& cameraPositionWorld, uint32_t frameIndex);

        // Exposes the light SSBO's raw handle/size so other passes needing the same light
        // population (e.g. renderer::TransparentForwardPass's own inline RIS shading, which has no
        // GBuffer entry to consume a MegaLightsPass composite from) can bind it directly without
        // duplicating the upload -- same "borrow a raw handle" convention renderer::
        // SurfaceCacheRayTracingPass::GetTLASHandle() already establishes.
        VkBuffer GetLightBufferHandle() const { return m_LightBuffer.Handle(); }
        VkDeviceSize GetLightBufferSize() const { return m_LightBuffer.Size(); }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        GpuBuffer m_LightBuffer; // Host-visible SSBO: 16-byte header (lightCount + pad) + kMaxMegaLights slots, filled once at Init.
        uint32_t m_LightCount = 0;

        // Phase 4, Feature 1: geometry::LightBVH, built once in Init() right after m_LightBuffer's
        // own upload above (lights are confirmed static for this pass' entire lifetime -- no
        // per-frame light-position update path exists) -- see LightBVH.h's own "why a separate,
        // duplicated implementation" comment. Host-visible/mapped like m_LightBuffer, same
        // "tiny data, no staging needed" rationale (~16KB total for 256 lights' worth of nodes).
        // Both sized to at least 1 node/index even when m_LightBVHNodeCount == 0 (a defensive,
        // never-expected-in-practice path -- this demo's light population is always non-empty, see
        // MegaLightsTypes.h's own GenerateProceduralLights comment) so buffer creation always gets
        // a valid non-zero size; that degenerate case is made inert at the shader side instead, by
        // forcing MegaLightsViewParamsUBO.spatialBiasRadius to 0 (see RecordShade's own comment).
        GpuBuffer m_LightBVHNodesBuffer;
        GpuBuffer m_LightBVHIndicesBuffer;
        uint32_t m_LightBVHNodeCount = 0;
        uint32_t m_LightBVHIndexCount = 0;

        VkImage m_RawRadianceImage = VK_NULL_HANDLE;
        VmaAllocation m_RawRadianceAllocation = VK_NULL_HANDLE;
        VkImageView m_RawRadianceView = VK_NULL_HANDLE;

        ATrousDenoisePass m_Denoiser; // Owned, dedicated instance -- see this class' own header comment.

        GpuBuffer m_ViewParamsBuffer; // MegaLightsViewParamsUBO, std140, GPU_ONLY.

        // Phase 4, Feature 2: ping-ponged MegaLightReservoir SSBOs (48 bytes/pixel, GPU_ONLY,
        // zero-initialized via vkCmdFillBuffer's 0xFFFFFFFFu sentinel-fill idiom -- same convention
        // renderer::VirtualShadowMapPool/renderer::GpuGeometryPagePool already use for their own
        // page-table sentinel clears). Sized renderExtent.width * height * sizeof(
        // MegaLightReservoir) each (~95 MiB at 1920x1080, consistent order-of-magnitude with
        // renderer::ReflectionPass's own ping-pong images). m_CurrentReservoirSlotIndex flips at
        // the TOP of every RecordShade() call, same "flip first" convention as
        // ReflectionPass::RecordTrace's own m_CurrentSlotIndex flip -- whichever slot held the
        // PREVIOUS frame's write becomes this frame's read-only history source.
        GpuBuffer m_ReservoirBuffers[2];
        uint32_t m_CurrentReservoirSlotIndex = 0;

        // Shade: 2 variants (indexed by m_CurrentReservoirSlotIndex at record time), since which
        // physical reservoir buffer is "current" (write target, binding 12) vs. "history" (read-only
        // source, binding 13) flips every frame -- every OTHER binding (0-11) is duplicated
        // identically across both variants, same convention as ReflectionPass::m_TraceSet's own
        // fixed-image bindings (see ReflectionPass.cpp's own STEP 2 comment).
        VkDescriptorSetLayout m_ShadeSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_ShadeDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_ShadeSet[2]{};
        VkPipelineLayout m_ShadePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ShadePipeline = VK_NULL_HANDLE;

        // Spatial reuse follow-up: this frame's Stage 2 output -- a SINGLE GPU_ONLY SSBO (NOT
        // ping-ponged, unlike m_ReservoirBuffers above), sized identically (renderExtent.width *
        // height * sizeof(MegaLightReservoir)). Fully overwritten every frame by
        // MegaLightsSpatialReuse.comp's own full-resolution dispatch before Stage 3 ever reads it,
        // so (unlike m_ReservoirBuffers) it needs no sentinel-fill at Init -- there is no "first
        // frame before the first write" read path for this buffer, RecordShade always writes it
        // before reading it within the same call. Deliberately never ping-ponged/persisted across
        // frames -- see this class' own header comment on why feeding a spatially-resampled
        // reservoir back into next frame's temporal history would compound bias.
        GpuBuffer m_SpatialReservoirBuffer;

        // Stage 2 (MegaLightsSpatialReuse.comp): 2 descriptor-set variants, same reason Stage 1's
        // m_ShadeSet needs 2 -- binding 0 (its TEMPORAL reservoir INPUT) must track whichever
        // physical buffer m_CurrentReservoirSlotIndex points at THIS frame (Stage 1's own write
        // target), so this pipeline's descriptor set is re-selected by the same index at record
        // time. Binding 1 (m_SpatialReservoirBuffer, its OUTPUT) is identical across both variants.
        VkDescriptorSetLayout m_SpatialReuseSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_SpatialReuseDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_SpatialReuseSet[2]{};
        VkPipelineLayout m_SpatialReusePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_SpatialReusePipeline = VK_NULL_HANDLE;

        // Stage 3 (MegaLightsFinalShade.comp): a SINGLE descriptor-set variant is sufficient here
        // (unlike Stage 1/Stage 2) -- every one of its inputs (m_RawRadianceImage, the GBuffer
        // views, the TLAS, m_LightBuffer, m_ViewParamsBuffer, the material bindings, and
        // m_SpatialReservoirBuffer) is a frame-invariant handle; nothing it reads flips per frame.
        VkDescriptorSetLayout m_FinalShadeSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_FinalShadeDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_FinalShadeSet = VK_NULL_HANDLE;
        VkPipelineLayout m_FinalShadePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_FinalShadePipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_CompositeSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_CompositeDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_CompositeSet = VK_NULL_HANDLE;
        VkPipelineLayout m_CompositePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CompositePipeline = VK_NULL_HANDLE;
    };

}
