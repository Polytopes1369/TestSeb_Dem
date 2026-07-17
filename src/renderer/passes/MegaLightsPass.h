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
// RecordTemporal's cross-frame accumulation instead, something Phase A deliberately has none of yet
// (temporal/spatial reservoir reuse is Phase B) -- so MegaLights genuinely needs SOME spatial filter
// of its own to avoid compositing raw 1-sample-per-pixel Monte Carlo noise into the final frame.
//
// --- Per-frame call order a caller must follow ---
//   1. RecordShade(cmd, viewProj, cameraPositionWorld, frameIndex) -- needs renderer::
//      ClusterResolvePass's GBuffer (normal/depth/albedo/roughness-metallic) already visible to
//      COMPUTE_SHADER reads, and renderer::SurfaceCacheRayTracingPass's TLAS already built. Runs
//      Shade -> internal ATrous denoise -> Composite as one call (Phase A has no caller-visible
//      per-stage state, unlike ReflectionPass's own 3-call contract -- no temporal ping-pong to
//      flip here). Ends with a trailing VkMemoryBarrier2 making the composited color image visible
//      to whatever the caller records next (matches ReflectionPass::RecordGather's own contract).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
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
        // clip it to [0,1] before it ever reaches renderer::TonemapPass, one of this codebase's own
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

        void RecordShade(VkCommandBuffer cmd, const maths::mat4& viewProj, uint32_t frameIndex);

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

        VkImage m_RawRadianceImage = VK_NULL_HANDLE;
        VmaAllocation m_RawRadianceAllocation = VK_NULL_HANDLE;
        VkImageView m_RawRadianceView = VK_NULL_HANDLE;

        ATrousDenoisePass m_Denoiser; // Owned, dedicated instance -- see this class' own header comment.

        GpuBuffer m_ViewParamsBuffer; // MegaLightsViewParamsUBO, std140, GPU_ONLY.

        VkDescriptorSetLayout m_ShadeSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_ShadeDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_ShadeSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ShadePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ShadePipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_CompositeSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_CompositeDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_CompositeSet = VK_NULL_HANDLE;
        VkPipelineLayout m_CompositePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CompositePipeline = VK_NULL_HANDLE;
    };

}
