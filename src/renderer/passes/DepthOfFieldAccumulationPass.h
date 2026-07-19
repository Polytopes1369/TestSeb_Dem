#pragma once
// UE5.8-parity "Accumulation Depth of Field": a cinematic, path-tracer-like bokeh mode that trades
// DepthOfFieldPass's own single-frame 16-tap Poisson gather for ONE stochastic thin-lens aperture
// sample per frame, temporally reprojected and accumulated into a running per-pixel mean -- see
// DepthOfFieldAccumulation.comp's own header comment for the full technique and why it converges
// toward the true thin-lens integral (the same one DepthOfField.comp's CoC math already uses) as the
// camera settles. Selected live via config::postprocess::DOF_MODE (0 = Gather/DepthOfFieldPass,
// 1 = Accumulation/this class) -- see renderer::ClusterRenderPipeline::RecordFrame's own [13e] DOF
// call site for the branch; both passes are always Init'd so switching modes live never needs a
// resize/recreate, only a one-frame history reset (see RecordGenerate's own `resetHistory` param).
//
// --- Per-frame call order a caller must follow (mirrors DepthOfFieldPass's own, plus TAATSRPass's
//     own reprojection inputs) ---
//   1. UpdateSourceDescriptor(hdrColorView) -- same reason DepthOfFieldPass::UpdateSourceDescriptor
//      exists: renderer::TAATSRPass's own output view ping-pongs between 2 images every frame, so
//      this pass' own binding to it must be re-written every frame too.
//   2. RecordGenerate(cmd, invViewProj, prevViewProj, cameraPositionWorld, aperture, frameIndex,
//      resetHistory, settings) -- single dispatch, ping-pongs its own internal history.
//
// GetOutputView() is read by the SAME downstream consumers DepthOfFieldPass::GetOutputView() is
// (renderer::BloomPass / renderer::PostProcessPass) -- renderer::ClusterRenderPipeline picks whichever
// DOF pass' own output view to bind based on config::postprocess::DOF_MODE every frame, exactly like
// it already re-binds DepthOfFieldPass's output every frame today.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class DepthOfFieldAccumulationPass {
    public:
        DepthOfFieldAccumulationPass() = default;
        ~DepthOfFieldAccumulationPass() { Shutdown(); }

        DepthOfFieldAccumulationPass(const DepthOfFieldAccumulationPass&) = delete;
        DepthOfFieldAccumulationPass& operator=(const DepthOfFieldAccumulationPass&) = delete;

        // RGB = accumulated running-mean color, A = accumulated sample weight (0..maxAccumulationSamples).
        // fp16 exactly represents integers well past any realistic cap, so packing both into one
        // image avoids a second ping-pong resource.
        static constexpr VkFormat kHistoryFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        struct Settings {
            float focalLengthMM = 50.0f;             // Same physical-camera meaning as DepthOfFieldPass::Settings.
            float focusDistanceWorldUnits = 10.0f;
            float maxCoCRadiusPixels = 24.0f;
            // How many frames a stationary-camera pixel accumulates before the running mean switches
            // from "add one more sample" (1/N weighting) to a fixed-window exponential moving average
            // (1/maxAccumulationSamples weighting) -- UE5.8's own Accumulation DOF exposes this as
            // "Temporal Blend Count". Higher = smoother converged bokeh but slower to converge/recover
            // from a fallback.
            float maxAccumulationSamples = 64.0f;
        };

        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D displayExtent, VkImageView hdrColorView, VkImageView depthView);

        void Shutdown();

        void UpdateSourceDescriptor(VkImageView hdrColorView);

        // `aperture` -- same PostProcessPass::Settings::aperture DepthOfFieldPass::RecordGenerate
        // takes (the SAME physical f-stop drives exposure, the Gather CoC, and this pass' own CoC).
        // `prevViewProj`/`resetHistory` -- same convention as TAATSRPass::RecordPass: `resetHistory`
        // forces a FULL-buffer reset (first frame ever, or the frame this mode is (re-)selected);
        // ordinary camera motion is instead handled PER-PIXEL by the reprojection/disocclusion test
        // inside DepthOfFieldAccumulation.comp, exactly like TAATSR.comp's own history clamp -- see
        // that shader's header comment.
        void RecordGenerate(VkCommandBuffer cmd, const maths::mat4& invViewProj, const maths::mat4& prevViewProj,
            const maths::vec3& cameraPositionWorld, float aperture, uint32_t frameIndex, bool resetHistory,
            const Settings& settings);

        VkImageView GetOutputView() const { return m_HistoryImageViews[m_CurrentHistoryIndex]; }

        // Frames elapsed since the last full-buffer reset (first frame / a mode switch) -- a rough
        // CPU-side convergence readout for the Debug ImGui panel. NOT a per-pixel sample count (a
        // disoccluded pixel converges slower than this number implies -- see
        // DepthOfFieldAccumulation.comp's own per-pixel alpha-channel weight for the real one).
        uint32_t GetFramesSinceReset() const { return m_FramesSinceReset; }

    private:
        // Byte-for-byte std140 mirror of DepthOfFieldAccumulationParamsUBO in DepthOfFieldAccumulation.comp.
        struct ParamsUBO {
            maths::mat4 invViewProj;
            maths::mat4 prevViewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f, _pad0 = 0.0f;
            float aperture = 4.0f;
            float focalLengthMM = 50.0f;
            float focusDistanceWorldUnits = 10.0f;
            float maxCoCRadiusPixels = 24.0f;
            uint32_t frameIndex = 0;
            uint32_t resetHistory = 0;
            float maxAccumulationSamples = 64.0f;
            float _pad1 = 0.0f;
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_DisplayExtent{ 0, 0 };

        // Ping-pong history images (display resolution) -- same convention as TAATSRPass's own
        // m_HistoryImages: slot i is the WRITE target on the frame m_CurrentHistoryIndex == i, and
        // becomes the READ (last frame's history) source on the following frame. See
        // GetOutputView()'s own comment for the resulting read/write call timing (identical to
        // TAATSRPass's own GetOutputView()).
        VkImage m_HistoryImages[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VmaAllocation m_HistoryAllocations[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkImageView m_HistoryImageViews[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        uint32_t m_CurrentHistoryIndex = 0;

        VkSampler m_LinearSampler = VK_NULL_HANDLE;

        GpuBuffer m_ParamsBuffer;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE }; // [0] writes slot 0 reading slot 1, [1] writes slot 1 reading slot 0.

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        uint32_t m_FramesSinceReset = 0;
    };

}
