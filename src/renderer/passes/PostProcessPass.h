#pragma once
// Phase PP1 (post-process stack roadmap -- see this repo's own project memory
// project_postprocess_stack_roadmap.md): the first real post-process pass this engine has ever
// had. Before this pass existed, renderer::ClusterRenderPipeline's final step blitted
// renderer::TAATSRPass's HDR (VK_FORMAT_R16G16B16A16_SFLOAT) output directly onto the swapchain
// image, relying on vkCmdBlitImage's implicit float->UNORM8 clamp as the only "tonemap" -- no
// exposure, no color grading, no display encode at all.
//
// --- Pipeline (matches Unreal Engine 5.8's own post-process evaluation order) ---
//   1. AutoExposureHistogram.comp: builds a 256-bin log2-luminance histogram of the HDR input
//      (workgroup-local reduction, see that shader's own comment).
//   2. AutoExposureAdapt.comp: single-workgroup reduction of the histogram into an average scene
//      luminance -> target EV100 (Auto Exposure Histogram metering) OR the Physical Camera's own
//      manual EV100 (computed here on the CPU from aperture/shutter/ISO, cheap scalar math) ->
//      exponentially eye-adapts (Auto) or snaps instantly (Manual) the persisted, GPU-owned
//      g_Exposure.currentEV100 -- see that shader's own comment for the Auto/Manual distinction.
//      Also clears the histogram for next frame.
//   3. PostProcessComposite.comp: exposure -> White Balance -> Color Correction (Contrast ->
//      Lift/Gamma/Gain -> Saturation) -> ACES Tone Mapping -> Gamma Correction (the pipeline's
//      only display encode -- the swapchain format is VK_FORMAT_B8G8R8A8_UNORM, not an _SRGB
//      format, so nothing else in the present path applies one), writing this pass' own owned
//      rgba8 output image.
//
// --- Per-frame call order a caller must follow ---
//   1. RecordComposite(cmd, hdrInputView-already-visible-to-COMPUTE_SHADER-reads, deltaTimeSeconds,
//      settings) -- internally records all 3 dispatches above with the barriers between them; the
//      caller only needs to barrier the HDR input into COMPUTE_SHADER-visible state beforehand and
//      this pass' own output into whatever consumes it (the final blit) afterward.
//
// The persisted exposure-state buffer (g_Exposure.currentEV100/currentAvgLuminance) is GPU-owned
// and never read back to the CPU -- cross-frame eye-adaptation continuity relies purely on the
// same single-command-buffer-per-frame, fence-waited submission model every other persistent GPU
// state in this codebase already relies on (renderer::TAATSRPass's own ping-pong history, for one).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class PostProcessPass {
    public:
        PostProcessPass() = default;
        ~PostProcessPass() { Shutdown(); }

        PostProcessPass(const PostProcessPass&) = delete;
        PostProcessPass& operator=(const PostProcessPass&) = delete;

        static constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr uint32_t kHistogramBinCount = 256u;
        // Log2-luminance range the histogram covers: roughly EV -10 (deep shadow) to EV +4 (bright
        // sky/sun highlight) in raw linear luminance terms -- wide enough for both indoor-dark and
        // outdoor-bright demoscene scenes without every sample collapsing into the two end bins.
        static constexpr float kMinLogLuminance = -10.0f;
        static constexpr float kLogLuminanceRange = 14.0f;

        // Physical Camera + White Balance + Color Correction, all user/scene tunable -- exposed as
        // one plain struct (not Debug-only: these are legitimate Release-time artistic controls,
        // not debug tooling, so nothing here is gated by CLAUDE.md's Debug/Release separation
        // rule). Defaults match Unreal Engine 5.8's own Post Process Volume defaults.
        struct Settings {
            // Physical Camera
            float aperture = 4.0f;           // f-stop.
            float shutterSpeedSeconds = 1.0f / 60.0f;
            float isoSensitivity = 100.0f;
            bool useAutoExposure = true;     // false = Manual metering (instant, no eye-adaptation).
            float exposureCompensationEV = 0.0f;
            float adaptationSpeedUpEVPerSec = 3.0f;   // Scene darkened -> exposure rising.
            float adaptationSpeedDownEVPerSec = 1.0f; // Scene brightened -> exposure falling.

            // White Balance
            float whiteBalanceTempKelvin = 6500.0f; // 6500 = neutral (D65).
            float whiteBalanceTint = 0.0f;

            // Color Correction (ASC CDL Lift/Gamma/Gain)
            float liftR = 0.0f, liftG = 0.0f, liftB = 0.0f;
            float gammaR = 1.0f, gammaG = 1.0f, gammaB = 1.0f;
            float gainR = 1.0f, gainG = 1.0f, gainB = 1.0f;
            float saturation = 1.0f;
            float contrast = 1.0f;

            // Gamma Correction (final display encode, see PostProcessComposite.comp's own comment)
            float displayGamma = 2.2f;

            // Phase PP2: Bloom (renderer::BloomPass's own owned mip chain -- see that pass' own
            // Settings for the threshold/ghost/anamorphic/dirt knobs that shape its content;
            // `bloomIntensity` here only scales how much of it gets added into the scene color).
            float bloomIntensity = 1.0f;

            // Chromatic Aberration: UV-space max per-channel radial offset at the screen corner.
            float chromaticAberrationIntensity = 0.0015f;

            // Vignette + Vignette Color Bleed
            float vignetteIntensity = 0.35f;
            float vignetteSmoothness = 0.55f;
            float vignetteColorBleed = 0.4f; // 0 = neutral gray falloff, 1 = strong chromatic falloff.

            // Phase PP3: Heat Distortion & Refraction -- global scale on renderer::
            // TransparentForwardPass's own per-material g_RefractionOffset (see MaterialParameters::
            // heatDistortion's own comment for the actual per-material intensity/source).
            float heatDistortionIntensity = 1.0f;

            // Motion Blur -- per-pixel velocity reconstructed from depth + invViewProj/prevViewProj,
            // no separate velocity buffer needed (see PostProcessComposite.comp's own comment).
            float motionBlurIntensity = 0.5f;
            float motionBlurMaxVelocityUV = 0.05f; // Caps smear length on a teleport/camera-cut frame.

            // Screen Space / Volumetric Height Fog (UE5.8's own analytic "Exponential Height Fog").
            float fogColorR = 0.55f, fogColorG = 0.60f, fogColorB = 0.68f;
            float fogDensity = 0.02f;
            float fogHeightFalloff = 0.15f;
            float fogHeightOffset = 0.0f;
            float fogStartDistance = 5.0f;
            float fogMaxOpacity = 0.85f;
        };

        // `bloomView` (renderer::BloomPass::GetOutputView(), its own upsample-chain mip 0) is
        // sampled by PostProcessComposite.comp's g_Bloom binding -- Phase PP2. `depthView`
        // (renderer::ClusterResolvePass::GetOutputDepthView(), render-resolution, fixed identity --
        // never ping-ponged) and `refractionOffsetView` (renderer::TransparentForwardPass::
        // GetRefractionOffsetView(), also fixed identity) are both Phase PP3: sampled through a
        // LINEAR sampler for their own implicit bilinear upsample to display resolution (this
        // pipeline has no display-resolution depth anywhere -- see DepthOfField.comp's own comment).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D displayExtent, VkImageView hdrColorView, VkImageView bloomView,
            VkImageView depthView, VkImageView refractionOffsetView);

        void Shutdown();

        // Recreates the descriptor bindings that reference `hdrColorView`/`bloomView` -- called if
        // renderer::TAATSRPass's own output view identity ever changes (it currently doesn't
        // across a pass's lifetime, ping-ponging between two fixed views instead, but this mirrors
        // renderer::TAATSRPass::UpdateDescriptorSets' own convention for symmetry/future-proofing).
        // `depthView`/`refractionOffsetView` are NOT re-bound here -- both source images keep a
        // fixed identity for this pipeline's entire lifetime (see Init()'s own comment).
        void UpdateDescriptorSets(VkImageView hdrColorView, VkImageView bloomView);

        // `invViewProj`/`prevViewProj`/`cameraPositionWorld` (Phase PP3): feed Motion Blur's own
        // per-pixel velocity reconstruction and Height Fog's own world-position/ray reconstruction
        // -- the SAME matrices renderer::ClusterRenderPipeline::RecordFrame already computes once
        // per frame for renderer::TAATSRPass and the opaque resolve's own motion-vector debug view.
        //
        // Records AutoExposureHistogram.comp -> AutoExposureAdapt.comp -> PostProcessComposite.comp,
        // with a VkMemoryBarrier2 between each (Histogram's global-atomic writes must land before
        // Adapt's reduction reads them; Adapt's ExposureStateSSBO write must land before Composite
        // reads it). Caller owns the barrier that makes the HDR input visible to COMPUTE_SHADER
        // reads beforehand, and the barrier that makes GetOutputImage() visible to its own next
        // consumer afterward (this pass' own trailing barrier only covers COMPUTE_SHADER writes).
        void RecordComposite(VkCommandBuffer cmd, float deltaTimeSeconds, const Settings& settings,
            const maths::mat4& invViewProj, const maths::mat4& prevViewProj, const maths::vec3& cameraPositionWorld);

        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        // Byte-for-byte mirror of PostProcessComposite.comp's PostProcessParamsUBO (std140).
        struct PostProcessParamsUBO {
            float aperture = 4.0f;
            float shutterSpeed = 1.0f / 60.0f;
            float isoSensitivity = 100.0f;
            uint32_t useAutoExposure = 1u;

            float exposureCompensationEV = 0.0f;
            float adaptationSpeedUp = 3.0f;
            float adaptationSpeedDown = 1.0f;
            float _pad0 = 0.0f;

            float whiteBalanceTempKelvin = 6500.0f;
            float whiteBalanceTint = 0.0f;
            float _pad1 = 0.0f, _pad2 = 0.0f;

            float lift[3] = { 0.0f, 0.0f, 0.0f };
            float _padLift = 0.0f;
            float gamma[3] = { 1.0f, 1.0f, 1.0f };
            float _padGamma = 0.0f;
            float gain[3] = { 1.0f, 1.0f, 1.0f };
            float _padGain = 0.0f;

            float saturation = 1.0f;
            float contrast = 1.0f;
            float displayGamma = 2.2f;
            float _pad3b = 0.0f;

            float bloomIntensity = 1.0f;
            float chromaticAberrationIntensity = 0.0015f;
            float vignetteIntensity = 0.35f;
            float vignetteSmoothness = 0.55f;

            float vignetteColorBleed = 0.4f;
            float _pad4 = 0.0f, _pad5 = 0.0f, _pad6 = 0.0f;

            // Phase PP3
            maths::mat4 invViewProj;
            maths::mat4 prevViewProj;

            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f;
            float heatDistortionIntensity = 1.0f;

            float motionBlurIntensity = 0.5f;
            float motionBlurMaxVelocityUV = 0.05f;
            float _pad7 = 0.0f, _pad8 = 0.0f;

            float fogColor[3] = { 0.55f, 0.60f, 0.68f };
            float fogDensity = 0.02f;

            float fogHeightFalloff = 0.15f;
            float fogHeightOffset = 0.0f;
            float fogStartDistance = 5.0f;
            float fogMaxOpacity = 0.85f;
        };
        static_assert(sizeof(PostProcessParamsUBO) == 336,
            "PostProcessParamsUBO must match PostProcessComposite.comp's PostProcessParamsUBO exactly (std140 layout)");

        // Byte-for-byte mirror of AutoExposureAdapt.comp's push_constant block.
        struct AdaptPushConstants {
            float minLogLuminance = 0.0f;
            float logLuminanceRange = 0.0f;
            float pixelCount = 0.0f;
            float deltaTimeSeconds = 0.0f;
            float adaptationSpeedUp = 0.0f;
            float adaptationSpeedDown = 0.0f;
            float exposureCompensationEV = 0.0f;
            float manualEV100 = 0.0f;
            uint32_t useAutoExposure = 1u;
        };

        // Byte-for-byte mirror of AutoExposureHistogram.comp's push_constant block.
        struct HistogramPushConstants {
            float imageWidth = 0.0f;
            float imageHeight = 0.0f;
            float minLogLuminance = 0.0f;
            float logLuminanceRange = 0.0f;
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_DisplayExtent{ 0, 0 };

        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        VkSampler m_LinearSampler = VK_NULL_HANDLE;

        GpuBuffer m_HistogramBuffer;      // 256 x uint, cleared by AutoExposureAdapt.comp every frame.
        GpuBuffer m_ExposureStateBuffer;  // { float currentEV100; float currentAvgLuminance; }, persistent.
        GpuBuffer m_ParamsBuffer;         // PostProcessParamsUBO, re-uploaded every frame.

        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;

        // Stage 1: histogram build.
        VkDescriptorSetLayout m_HistogramSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_HistogramSet = VK_NULL_HANDLE;
        VkPipelineLayout m_HistogramPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_HistogramPipeline = VK_NULL_HANDLE;

        // Stage 2: histogram reduction + eye adaptation.
        VkDescriptorSetLayout m_AdaptSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_AdaptSet = VK_NULL_HANDLE;
        VkPipelineLayout m_AdaptPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_AdaptPipeline = VK_NULL_HANDLE;

        // Stage 3: final composite.
        VkDescriptorSetLayout m_CompositeSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_CompositeSet = VK_NULL_HANDLE;
        VkPipelineLayout m_CompositePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CompositePipeline = VK_NULL_HANDLE;
    };

}
