#pragma once
// Final GI composite: blends renderer::ClusterResolvePass's direct-lit color with
// renderer::ATrousDenoisePass's denoised indirect-diffuse term into one owned rgba16f (linear HDR)
// output image -- the image renderer::ClusterRenderPipeline's own final blit now sources from
// (via renderer::TAATSRPass then renderer::TonemapPass) instead of ClusterResolvePass's output
// directly.
//
// --- Debug view modes 13 (LUMEN) / 14 (SPATIAL PROBES) ---
// core/Camera.h already reserves DEBUG_VIEW_LUMEN=13 / DEBUG_VIEW_SPATIAL_PROBES=14 (previously
// rendered as a magenta placeholder by ClusterResolve.comp's own unhandled-viewMode fallback).
// This pass replaces that placeholder for real: mode 13 shows the denoised GI term alone (no
// direct light added), mode 14 visualizes the World Probe grid directly at each pixel's
// reconstructed world position. Per CLAUDE.md's Debug/Release rule ("Lumen/Nanite visualization
// modes must not be compiled in Release"), every resource these two debug modes
// need beyond the always-present direct/denoised-GI blend (the depth image + an inverse-view-
// projection UBO, to reconstruct a world position; the World Probe grid itself) is bound ONLY in
// a `#ifndef NDEBUG` build, exactly like renderer::ClusterResolvePass's own viewMode push constant
// is -- a Release build's descriptor set layout, pipeline layout and shader binary all end up
// strictly smaller, with zero object code for either debug branch.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/Camera.h"
#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class WorldProbeGridPass;

    class GICompositePass {
    public:
        GICompositePass() = default;

        GICompositePass(const GICompositePass&) = delete;
        GICompositePass& operator=(const GICompositePass&) = delete;

        static constexpr uint32_t kWorkgroupSize = 8;
        // R16G16B16A16_SFLOAT (linear HDR): this pass sums two already-HDR terms (renderer::
        // ClusterResolvePass::kOutputColorFormat's direct light + renderer::ATrousDenoisePass::
        // kFormat's denoised indirect GI) -- an 8-bit UNORM output would hard-clip that sum to
        // [0,1] right at the point they converge, which is exactly what fed this codebase's
        // "burned" overexposure bug (see ClusterResolvePass::kOutputColorFormat's own comment for
        // the full chain). renderer::TransparentForwardPass draws directly onto this image (its own
        // pipeline's color attachment format must match -- see that pass' Init() call site) and
        // renderer::TAATSRPass samples it as its low-res HDR input; the actual display-referred
        // tonemap curve lives downstream of TAA, in renderer::TonemapPass.
        static constexpr VkFormat kOutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        // `directColorView` (renderer::ClusterResolvePass::GetOutputColorView()) and
        // `denoisedGIView` (renderer::ATrousDenoisePass::GetOutputView()) are always bound (set 0,
        // bindings 1/2). In a Debug build only, `depthView` and `worldProbes` are ALSO bound (see
        // the class comment) to support the two debug view modes; both parameters are ignored
        // entirely in a Release build (still accepted unconditionally so call sites do not need
        // their own `#ifndef NDEBUG` branch).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, VkImageView directColorView, VkImageView denoisedGIView,
            VkImageView depthView, const WorldProbeGridPass& worldProbes);

        void Shutdown();

        // Dispatches one invocation per output pixel: finalColor = directColor + denoisedGI. In a
        // Debug build, `camera`/`cameraPositionWorld`/`worldProbeGridOrigin` additionally drive the
        // viewMode 13/14 visualizations (unused, and not even present as parameters' worth of real
        // work, in Release -- see the class comment). Caller owns every synchronization barrier
        // before (direct color + denoised GI + [Debug] depth + World Probe grid all visible) and
        // after (GetOutputImage() visible to the final blit) this call.
        void RecordComposite(VkCommandBuffer cmd
#ifndef NDEBUG
            , const CameraPushConstants& camera, const maths::vec3& cameraPositionWorld,
            const maths::vec3& worldProbeGridOrigin
#endif
        );

        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        VkSampler m_NearestSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

#ifndef NDEBUG
        // Debug-only: bindings 3/4 (depth + this pass' own small invViewProj UBO) support viewMode
        // 14's world-position reconstruction; bindings 5/6 (World Probe grid sampler3D + params
        // UBO, matching world_probe_sampling.glsl's own contract) support viewMode 14's actual
        // sample. See the class comment -- this whole block, and every line of code that touches
        // it, compiles to nothing in a Release build.
        GpuBuffer m_ViewParamsBuffer;
        GpuBuffer m_WorldProbeGridParamsBuffer;
#endif
    };

}
