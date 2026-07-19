#pragma once
// Screen-Space Ray Tracing (SSRT), Lumen-style "Screen Trace": for every pixel, marches a short,
// fixed-step ray directly against the depth buffer (see the class comment's own scope note below
// for why this is a plain linear march, not a Hi-Z/HZB traversal) starting from a cosine-weighted
// hemisphere direction around that pixel's own surface normal. On a hit, samples
// renderer::ClusterResolvePass's own direct-lit color at the hit pixel as a cheap, correct-enough
// near-field indirect contribution ("use the color of this pixel as near-field light contribution").
// On a miss, falls back to renderer::WorldProbeGridPass's own multi-level ambient
// grid ("merge this result with the lighting from the global probes") -- exactly Lumen's own tiered
// Final Gather shape: cheap screen trace first, radiance-cache-style fallback second. The combined
// (still noisy -- one ray per pixel per frame) result is what renderer::ATrousDenoisePass denoises
// next.
//
// --- Scope: linear march against VulkanContext's own depth image, not a Hi-Z traversal ---
// A full Hierarchical-Z traversal (as renderer::HZBPass's own pyramid would enable) adds real
// complexity -- per-mip stepping, a cone/wedge test at each level -- for a ray this SHORT
// ("faible longueur," per the original spec) to meaningfully benefit from. A fixed-step linear
// march against the existing hardware depth image (the same one renderer::HZBPass itself reduces
// from, so this is consistent with this codebase's own existing "hardware depth only" scope
// choice) is the simpler, still-correct approach this class implements.
//
// --- F1 ("Lumen Lite", UE5.8 parity roadmap): GI mode switch ---
// In config::lumen::GIMode::HighQuality (the pre-F1 default and behavior), the march above always
// runs, exactly as before. In GIMode::Lite, ScreenTrace.comp skips the march ENTIRELY (see that
// shader's own giMode branch) and returns the World Probe grid's own multi-level, occlusion-weighted
// irradiance directly -- Lite mode treats the probe grid as the PRIMARY GI term rather than a rare
// fallback, matching real Lumen Lite's own "irradiance-field gather, minimal per-pixel work" design.
// RecordTrace() takes the live `giMode` every call (not cached at Init()) so the Debug ImGui GI-mode
// selector can flip it at runtime for A/B comparison.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/Camera.h"
#include "core/EngineConfig.h"
#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class WorldProbeGridPass;

    class ScreenTracePass {
    public:
        ScreenTracePass() = default;

        ScreenTracePass(const ScreenTracePass&) = delete;
        ScreenTracePass& operator=(const ScreenTracePass&) = delete;

        // Matches ScreenTrace.comp's local_size_x/y exactly.
        static constexpr uint32_t kWorkgroupSize = 8;
        // Fixed step count and world-space march distance -- a short, "faible longueur" trace, not
        // a long-range reflection ray (see the class comment's scope note).
        static constexpr uint32_t kScreenTraceSteps = 24;
        static constexpr float kScreenTraceMaxDistance = 3.0f;
        // Screen-space thickness epsilon (NDC depth units) a marched sample must fall within of
        // the depth buffer's own stored value to count as a hit -- guards against a ray that
        // passes safely IN FRONT of geometry (should keep marching) being mistaken for a hit
        // against whatever surface happens to project to the same pixel far behind it.
        static constexpr float kDepthThicknessNDC = 0.01f;

        static constexpr VkFormat kOutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        // Allocates the (renderExtent-sized) noisy GI output image and builds set 0: the output
        // image, the 3 sampled inputs (depth/normal/direct-color, all borrowed, unmodified for
        // this pass' entire lifetime -- see the class comment), this pass' own small per-frame view
        // -params UBO, and world_probe_sampling.glsl's sampler3D + params UBO bound against
        // `worldProbes` (already Init'd, must outlive this pass). `depthView` must be sampled while
        // in VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL -- the same layout
        // renderer::ClusterRenderPipeline's own step [11] barrier already leaves the depth image in
        // by the point this pass records (see RecordTrace()'s own comment).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, VkImageView depthView, VkImageView normalView, VkImageView directColorView,
            const WorldProbeGridPass& worldProbes);

        void Shutdown();

        // Uploads this frame's view params (inverse view-projection reconstructed from `camera`,
        // `cameraPositionWorld`, `frameIndex` for the Halton hemisphere jitter, `giMode` -- F1's own
        // live GI-mode switch, see the class comment) and every level's CURRENT origin/spacing (read
        // fresh from `worldProbes` every call, not cached from Init() -- recomputed every frame by
        // renderer::WorldProbeGridPass::RecordUpdate), then dispatches one invocation per output
        // pixel. The depth image must already be in VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
        // (this function does not transition it); the World Probe grid must already be visible to a
        // sampled read (the caller's barrier after renderer::WorldProbeGridPass::RecordUpdate).
        // Caller owns every synchronization barrier before/after this call. Ends with a
        // VkMemoryBarrier2 making GetOutputView() visible to renderer::ATrousDenoisePass's first
        // read.
        void RecordTrace(VkCommandBuffer cmd, const CameraPushConstants& camera,
            const maths::vec3& cameraPositionWorld, const WorldProbeGridPass& worldProbes,
            uint32_t frameIndex, config::lumen::GIMode giMode);

        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        VkSampler m_NearestSampler = VK_NULL_HANDLE; // Depth/normal/direct-color reads (exact-texel, no filtering intended).

        // ScreenTraceViewParamsUBO (set 0 binding 4) + world_probe_sampling.glsl's own
        // WorldProbeGridParamsUBO (set 0 binding 6) -- both rewritten every RecordTrace() call.
        GpuBuffer m_ViewParamsBuffer;
        GpuBuffer m_WorldProbeGridParamsBuffer;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
