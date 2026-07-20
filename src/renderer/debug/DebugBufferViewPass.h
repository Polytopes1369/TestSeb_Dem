#pragma once
// Debug-only (whole file compiled out in Release -- see the #ifndef NDEBUG guard below): backs
// the ImGui "Buffer Viewer" dropdown (main.cpp's Engine Configuration Panel). Rather than one
// dedicated visualization pass per candidate buffer, this owns a SINGLE small compute pipeline
// (src/shaders/src/Debug/DebugBufferView.comp) whose one combined-image-sampler binding gets
// REWRITTEN every frame to whichever buffer the user currently has selected -- see RecordView()'s
// own comment for why this is safe (every candidate buffer in this pipeline is already kept in
// VK_IMAGE_LAYOUT_GENERAL for its entire lifetime, this codebase's own established convention for
// every intermediate GBuffer/GI image).
//
// --- Per-frame sequence a caller must follow, ONLY when a buffer view is actually selected
// (index != 0/"Off" in the ImGui dropdown -- otherwise this pass is skipped entirely, see
// renderer::ClusterRenderPipeline::RecordFrame's own call site) ---
//   1. RecordView(cmd, sourceView, mode) -- rewrites this pass' own descriptor to `sourceView`,
//      dispatches the visualization shader into this pass' own owned output image, and ends with
//      a trailing barrier making that output visible to a following blit.
//   2. Caller redirects its own final swapchain blit to read from GetOutputImage() instead of
//      renderer::PostProcessPass's own output for this frame.
#ifndef NDEBUG

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace renderer::debug {

    class DebugBufferViewPass {
    public:
        DebugBufferViewPass() = default;

        DebugBufferViewPass(const DebugBufferViewPass&) = delete;
        DebugBufferViewPass& operator=(const DebugBufferViewPass&) = delete;

        static constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr uint32_t kWorkgroupSize = 8;

        // Mirrors DebugBufferView.comp's own `visualizationMode` push constant values exactly --
        // see that shader's own header comment for what each mode does to the raw sampled texel.
        enum class VisualizationMode : uint32_t {
            kPassthrough = 0,    // Already display-ready (e.g. albedo, rgba8).
            kGrayscale = 1,      // Single-channel scalar (depth/AO/roughness-metallic/hit-mask).
            kOctNormal = 2,      // 2-channel octahedral-encoded world-space normal.
            kTonemap = 3,        // Linear HDR color -- needs ACES tonemap to stay visible/unclipped.
            kAlphaGrayscale = 4, // Scalar carried in the ALPHA channel of an otherwise-HDR image
                                 // (MegaLights raw radiance's Debug-only shadow-ray verdict -- see
                                 // MegaLightsFinalShade.comp's own debugShadowOcclusion comment).
        };

        // Allocates the owned rgba8 output image (sized to `outputExtent`, this pipeline's own
        // display extent -- the source buffer's own resolution is irrelevant, RecordView() always
        // samples it through a UV-based bilinear resize) plus the compute pipeline/descriptor set.
        // The descriptor's own combined-image-sampler binding is left pointing at a 1x1 dummy
        // image at Init() time (never actually sampled from before the first real RecordView()
        // call rewrites it) purely so the descriptor set starts in a valid, fully-bound state.
        // `exposureStateBuffer` is renderer::PostProcessPass::GetExposureStateBuffer() -- the SAME
        // persistent { float currentEV100; float currentAvgLuminance; } SSBO PostProcessComposite.
        // comp reads, bound ONCE here (never rewritten, unlike g_Source) since it's a fixed-identity
        // buffer for this pipeline's entire lifetime. DebugBufferView.comp's kModeTonemap path
        // reads it to apply the same exposure normalization PostProcessComposite.comp applies,
        // instead of feeding raw real-lux HDR radiance straight into ACES (which saturates to
        // white -- see this class' caller in ClusterRenderPipeline::Init for Init ordering: the
        // caller MUST call m_PostProcess.Init() before this, so the buffer already exists).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D outputExtent, VkBuffer exposureStateBuffer);

        void Shutdown();

        // Rewrites this pass' own g_Source binding to `sourceView` (a single
        // VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER write, cheap enough to do unconditionally every
        // frame this pass runs at all -- no ping-pong/double-buffering concern since this pass has
        // no persistent state of its own to protect), then dispatches DebugBufferView.comp.
        // `sourceView` must already be in VK_IMAGE_LAYOUT_GENERAL (true for every candidate buffer
        // in this pipeline's own established convention -- see this class' own header comment) and
        // visible to VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads (true for every candidate buffer,
        // each already written earlier this same frame by its own owning pass).
        void RecordView(VkCommandBuffer cmd, VkImageView sourceView, VisualizationMode mode);

        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_OutputExtent{ 0, 0 };

        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        // 1x1 dummy image the descriptor set's g_Source binding starts out pointing at (see Init()'s
        // own comment) -- never sampled from in practice, only exists so the descriptor set is
        // never left referencing a VK_NULL_HANDLE image view.
        VkImage m_DummyImage = VK_NULL_HANDLE;
        VmaAllocation m_DummyAllocation = VK_NULL_HANDLE;
        VkImageView m_DummyView = VK_NULL_HANDLE;

        VkSampler m_LinearSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
#endif
