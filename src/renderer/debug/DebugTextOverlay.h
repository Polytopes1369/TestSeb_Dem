#pragma once
// Debug-only (whole file compiled out in Release -- see the #ifndef NDEBUG guard below) real-time
// stat overlay: total GPU memory used, pending SSD page loads, disk read throughput, and the
// hardware-vs-software rasterizer triangle split. See src/shaders/src/Debug/DebugText.vert/.frag
// for the tiny quad-per-glyph pipeline this class drives, and BitmapFont8x8.h for the bitmap font
// data it uploads once at Init().
//
// --- Per-frame sequence a caller must record, in order (both Debug-only) ---
//   1. BuildFrameText(...) -- CPU-side only, builds this frame's glyph instance list from the
//      given stat values (no GPU work).
//   2. RecordDraw(cmd, ...) -- uploads the glyph instances and draws them directly on top of the
//      resolve pass's own output color image, which renderer::ClusterResolvePass keeps
//      permanently in VK_IMAGE_LAYOUT_GENERAL (compute-only reads/writes) between frames -- this
//      call transitions it to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL for the draw and back to
//      GENERAL before returning, so the caller's own subsequent blit-to-swapchain step sees
//      exactly the layout it already expects.
//
// Exactly like every other piece of this Nanite-style pipeline, this class is a self-contained
// building block -- Init()/Shutdown()/per-frame calls only.
#ifndef NDEBUG

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer::debug {

    class DebugTextOverlay {
    public:
        DebugTextOverlay() = default;

        DebugTextOverlay(const DebugTextOverlay&) = delete;
        DebugTextOverlay& operator=(const DebugTextOverlay&) = delete;

        // Upper bound on glyphs drawn in a single frame -- generously covers this overlay's fixed
        // ~7-line stat stack (see BuildFrameText()) with headroom to spare.
        static constexpr uint32_t kMaxGlyphs = 2048;

        // Fixed advance (8px glyph + 1px gap) between consecutive glyphs -- also used by
        // BuildFrameText() to estimate a line's pixel width for right-aligned placement (the FPS
        // counter), so it must stay a single shared constant rather than a value duplicated in
        // AppendLine()'s own body.
        static constexpr float kGlyphAdvanceX = 18.0f;

        // Builds the font bitmap SSBO (BitmapFont8x8.h, uploaded once via a blocking one-time
        // submit) and the glyph-instance SSBO/descriptor set/pipeline. `outputColorFormat` must
        // match renderer::ClusterResolvePass::kOutputColorFormat, so this pipeline is attachment-
        // compatible with the image RecordDraw() will target.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkFormat outputColorFormat);

        void Shutdown();

        // Formats a fixed top-left stack of short stat lines, a GI-component-state line, and a
        // top-right FPS counter into this frame's glyph instance list (CPU-side only -- no GPU
        // work happens here). `bytesPerSecond` is displayed in KB/s. `fps` and `viewportWidthPixels`
        // (the render extent's width, needed to right-align the FPS text) drive the top-right
        // counter; `radiosityEnabled`/`ssrtEnabled`/`traceMode` (0 = SWRT, 1 = HWRT)/
        // `worldProbesEnabled` mirror ClusterRenderPipeline's own current debug-toggle state exactly.
        void BuildFrameText(float gpuMemUsedMB, uint32_t pendingPageLoads, float bytesPerSecond,
            uint32_t hwTriangleCount, uint32_t swTriangleCount, float fps, float viewportWidthPixels, float viewportHeightPixels,
            bool radiosityEnabled, bool ssrtEnabled, uint32_t traceMode, bool worldProbesEnabled);

        void RecordDraw(VkCommandBuffer cmd, VkImage outputColorImage, VkImageView outputColorView, VkExtent2D extent);

    private:
        struct GlyphInstance {
            maths::vec2 screenPosPixels;
            uint32_t charCode = 0;
            uint32_t _pad = 0;
        };
        static_assert(sizeof(GlyphInstance) == 16,
            "GlyphInstance must match GlyphInstance in DebugText.vert exactly (std430 layout)");

        void AppendLine(const std::string& text, float x, float y);

        VkDevice m_Device = VK_NULL_HANDLE;

        GpuBuffer m_FontBuffer;          // 128 * 8 uint32 rows (BitmapFont8x8.h), GPU_ONLY, written once at Init.
        GpuBuffer m_GlyphInstanceBuffer; // GlyphInstance[kMaxGlyphs], GPU_ONLY, rewritten every frame.

        std::vector<GlyphInstance> m_PendingGlyphs;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}

#endif // NDEBUG
