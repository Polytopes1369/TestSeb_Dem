#pragma once
// À-Trous ("with holes") wavelet spatial denoiser: a generic, reusable `image in -> image out`
// filter with no coupling to any specific producer -- renderer::ScreenTracePass's noisy combined
// GI signal is its only consumer today, but this class knows nothing about SSRT, the World Probe
// grid, or Surface Cache sampling; it just denoises whatever `rgba16f` image it is given, guided
// by a depth image and an octahedral-encoded normal image (the same two G-buffer inputs
// renderer::ClusterResolvePass now produces).
//
// --- Why À-Trous instead of a plain box/Gaussian blur ---
// A dumb spatial blur would smear the denoised GI signal ACROSS a depth/material discontinuity
// (the edge of an object, a shadow boundary) exactly as much as across a flat, noisy-but-coherent
// surface -- destroying real geometric and lighting detail along with the noise. The À-Trous
// scheme instead runs several passes of a SPARSE (holes between taps, hence the name) 5x5 kernel
// with a dilating step size (1, 2, 4, 8, 16 texels), each tap weighted down whenever it crosses a
// depth discontinuity, a normal mismatch, or a large luminance jump relative to the CENTER texel
// -- the standard SVGF-style 3-way edge-stopping weight -- so noise gets averaged away only where
// the underlying signal is actually expected to be smooth, exactly the "n'élimine le bruit sans
// lisser les détails de la géométrie ou des textures du matériau final" requirement.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace renderer {

    class ATrousDenoisePass {
    public:
        ATrousDenoisePass() = default;

        ATrousDenoisePass(const ATrousDenoisePass&) = delete;
        ATrousDenoisePass& operator=(const ATrousDenoisePass&) = delete;

        // Matches ATrousDenoise.comp's local_size_x/y exactly.
        static constexpr uint32_t kWorkgroupSize = 8;
        // 5 iterations, step sizes {1,2,4,8,16} -- the standard À-Trous/SVGF dilation progression
        // (each pass effectively doubles the EFFECTIVE filter radius of the one before it, without
        // the O(radius^2) cost a single, equivalently-wide dense kernel would have).
        static constexpr uint32_t kIterations = 5;

        static constexpr VkFormat kFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        // Allocates the 2 ping-pong images (renderExtent-sized) and the 3 descriptor sets this
        // class alternates between across kIterations passes (external-input -> ping A, ping A ->
        // ping B, ping B -> ping A -- see the .cpp's own iteration-order comment), all sharing one
        // descriptor set LAYOUT. `noisyInputView` (the FIRST iteration's source -- a producer's
        // output image, e.g. renderer::ScreenTracePass::GetOutputView()), `depthView` and
        // `normalView` are borrowed and must stay valid for this pass' entire lifetime (bound once,
        // here, like every other static resource in this codebase's compute passes).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, VkImageView noisyInputView, VkImageView depthView, VkImageView normalView);

        void Shutdown();

        // Records kIterations dispatches, each with a VkMemoryBarrier2 (STORAGE_WRITE ->
        // SHADER_SAMPLED_READ) between it and the next iteration's read of its own output (the
        // first iteration instead depends on the caller's own barrier after whatever produced
        // `noisyInputView` -- not recorded here). The FIRST iteration's input is `noisyInputView`
        // itself, re-sampled fresh from GPU memory every call (not cached from Init()) since a new
        // producer dispatch is expected to have written new data into it between calls. Ends with
        // one more VkMemoryBarrier2 making GetOutputView() visible to the caller's next read.
        void RecordDenoise(VkCommandBuffer cmd);

        // The most recently completed iteration's output image (alternates between the two owned
        // ping-pong images every RecordDenoise() call in a fixed, deterministic pattern -- always
        // the SAME image after kIterations=5, an odd count, so this is a stable getter, not one
        // that changes identity between calls).
        VkImageView GetOutputView() const { return m_PingViews[0]; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_PingImages[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VmaAllocation m_PingAllocations[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkImageView m_PingViews[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        VkSampler m_NearestSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        // [0] external input -> ping[0]; [1] ping[0] -> ping[1]; [2] ping[1] -> ping[0] -- see the
        // .cpp's RecordDenoise() for the exact per-iteration selection.
        VkDescriptorSet m_Sets[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
