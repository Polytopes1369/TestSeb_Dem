#pragma once
// F3 (UE5.8 rendering-parity gap: Fog Screen Space Scattering) -- a depth-aware bilateral spread of
// renderer::AtmosVolumetricFogPass's own per-pixel in-scattered radiance (Volumetric Fog AND, since
// Local Fog Volumes are already additively baked into that same froxel grid by
// AtmosVolumetricFog.comp's own InjectLight before this pass ever runs, Local Fog Volumes too),
// approximating multiple scattering: a bright, thin light shaft should visibly bloom/soften instead of
// staying crisply confined to the raw 160x90x64 froxel grid's own resolution. See
// FogScreenSpaceScattering.comp's own header comment for the full algorithm derivation (mirrors
// renderer::SubsurfaceScatteringPass's separable-2-pass-with-bilateral-edge-stops shape, but mode 0
// additionally EXTRACTS the per-pixel fog sample from the 3D froxel texture -- the two passes are NOT
// symmetric the way SubsurfaceScattering.comp's H/V passes are, see that shader's own comment).
//
// --- Where it runs, and what it replaces ---
// renderer::ClusterRenderPipeline records this immediately after renderer::AtmosVolumetricFogPass::
// RecordUpdate (this pass' only producer) and before renderer::PostProcessPass::RecordComposite (its
// only consumer) -- same "producer and consumer adjacent in the frame graph" convention
// AtmosVolumetricFogPass's own RecordUpdate call site already documents. PostProcessComposite.comp's
// own ApplyVolumetricFog function now samples this pass' 2D output (GetOutputView(), a plain
// texture() 2D fetch) INSTEAD OF directly re-sampling the raw 3D froxel texture it used to -- see that
// function's own updated comment. The original 3D texture binding stays present in
// PostProcessPass' own descriptor set (unused when the feature is on) so
// config::atmos::FOG_SCREEN_SPACE_SCATTERING_ENABLED can gate a runtime UBO branch back to the
// original unblurred lookup at zero extra pipeline/descriptor-set cost when off (Debug ImGui toggle),
// rather than needing this whole pass' own dispatch to be conditionally skipped (which would leave
// GetOutputView() stale/uninitialized on the very first disabled frame -- always running this pass
// and branching only in PostProcessComposite.comp avoids that class of bug entirely).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"

namespace renderer {

    class FogScreenSpaceScatteringPass {
    public:
        FogScreenSpaceScatteringPass() = default;

        FogScreenSpaceScatteringPass(const FogScreenSpaceScatteringPass&) = delete;
        FogScreenSpaceScatteringPass& operator=(const FogScreenSpaceScatteringPass&) = delete;

        // Matches FogScreenSpaceScattering.comp's local_size_x/y exactly.
        static constexpr uint32_t kWorkgroupSize = 8;
        // R16G16B16A16_SFLOAT (linear HDR): the extracted fog sample is unbounded in-scattered
        // radiance (same reasoning as every other HDR intermediate in this codebase -- see
        // renderer::ATrousDenoisePass::kFormat's own comment), so an 8-bit UNORM scratch/output would
        // hard-clip it before PostProcessComposite.comp ever gets to apply it.
        static constexpr VkFormat kFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        // `depthView` is renderer::ClusterResolvePass::GetOutputDepthView() (render-res, bilinear-
        // upsampled -- same convention renderer::PostProcessPass::Init's own depthView parameter
        // already documents). `volumetricFogView`/`fogSampler` are renderer::AtmosVolumetricFogPass::
        // GetIntegratedFogView()/GetFogSampler(). `displayExtent` matches renderer::PostProcessPass'
        // own extent (this pass' output is consumed at THAT resolution, not necessarily the render
        // resolution `depthView` was produced at -- both `g_Depth` here and PostProcessComposite.
        // comp's identically-named binding rely on the SAME bilinear-upsample-via-LINEAR-sampler
        // trick for that mismatch). All borrowed; must stay valid for this pass' entire lifetime
        // (bound once here, matching every other leaf compute pass in this codebase).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D displayExtent, VkImageView depthView, VkImageView volumetricFogView, VkSampler fogSampler);

        void Shutdown();

        // Records the two separable dispatches (mode 0: extract + blur horizontal; mode 1: blur
        // vertical) with a VkMemoryBarrier2 between them, and ends with a VkMemoryBarrier2 making
        // GetOutputView() visible to renderer::PostProcessPass::RecordComposite's own COMPUTE_SHADER
        // sampled read right after. `blurRadiusPixels` <= 0 still runs both dispatches (mode 0's
        // extraction is mandatory -- mode 1 depends on its output) but each degenerates to a pure
        // passthrough of the per-pixel extracted sample (see FogScreenSpaceScattering.comp's own
        // "sub-pixel radius" early-out) -- always-record, rather than a full skip, so
        // GetOutputView() is never stale on a frame the feature happens to be dialed to zero (see
        // this class' own header comment for why a full-skip alternative was rejected).
        void RecordScatter(VkCommandBuffer cmd, const maths::mat4& invViewProj,
            const maths::vec3& cameraPositionWorld, const maths::vec3& cameraForward, float blurRadiusPixels);

        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        // Byte-for-byte mirror of FogScreenSpaceScattering.comp's push_constant block.
        struct FogScatterPushConstants {
            maths::mat4 invViewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f, blurRadiusPixels = 0.0f;
            float cameraForwardX = 0.0f, cameraForwardY = 0.0f, cameraForwardZ = 0.0f, _pad0 = 0.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            uint32_t mode = 0u;
            float _pad1 = 0.0f;
        };
        static_assert(sizeof(FogScatterPushConstants) == 112,
            "FogScatterPushConstants must match FogScreenSpaceScattering.comp's push_constant block (std430) exactly");

        void RecordOnePass(VkCommandBuffer cmd, VkDescriptorSet set, const FogScatterPushConstants& pc);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_DisplayExtent{ 0, 0 };

        // Mode 0's own output (horizontal-blurred, extracted fog samples).
        VkImage m_ScratchImage = VK_NULL_HANDLE;
        VmaAllocation m_ScratchAllocation = VK_NULL_HANDLE;
        VkImageView m_ScratchView = VK_NULL_HANDLE;

        // Mode 1's own output -- this pass' final, externally-consumed result.
        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        VkSampler m_LinearSampler = VK_NULL_HANDLE; // g_Depth / g_VolumetricFog -- both need real bilinear filtering (see .cpp's own comment).

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        // [0] = mode 0 (depth + 3D fog -> scratch); [1] = mode 1 (depth + scratch -> output).
        VkDescriptorSet m_Sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
