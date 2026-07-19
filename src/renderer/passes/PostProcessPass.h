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
        // Log2-luminance range the histogram covers, in the SAME raw linear g_HDRColor units the
        // scene is actually lit in -- i.e. real photometric nits, not a normalized [0,1] "unitless"
        // scale. Re-tuned 2026-07-19 (Task F11, enabling Auto Exposure): the OLD range here
        // ([2^-10, 2^-10+2^14] = [~0.001, 16]) predates the 2026-07-17/18 lighting recalibration to
        // real UE5.8 lux/candela values (world::DirectionalLight::intensity = 10,000 lux -- see that
        // struct's own comment) and the matching manual Physical Camera re-tune (EXPOSURE_APERTURE's
        // own comment: EV100 ~17.0, MaxLuminance = 1.2*2^EV100 ~= 153,600). Under that real-lux
        // scale, a directly-sunlit mid-albedo diffuse surface alone already reads ~1,500-2,000 nits
        // (irradiance*albedo/pi), with GGX specular highlights and the sky/sun disk itself reaching
        // into the tens of thousands -- i.e. almost every non-black pixel in a real scene would have
        // clamped into the histogram's single top bin (256) under the old range, collapsing Auto
        // Exposure's weighted-mean luminance to a constant ~16 regardless of true scene content
        // (targetEV100 = log2(16*100/12.5) ~= 7.0, a ~10-stop under-exposure vs the correct ~17.0 --
        // a severe, constant blow-out, not oscillation, but just as wrong). New range covers
        // [2^-6, 2^-6+2^24] = [~0.0156, ~262,144]: comfortably brackets the manual metering's own
        // MaxLuminance ceiling with headroom for brighter specular fireflies (BloomUpsampleComposite
        // .comp/PostProcessComposite.comp's own SanitizeHDR guards separately cap those at 50,000
        // before they reach Bloom), while still resolving ~0.094 EV/bin (24 stops / 254 usable bins)
        // -- plenty fine for a stable weighted-mean read. Sub-0.005 luminance is still routed to the
        // dedicated bin 0 (see AutoExposureHistogram.comp's own ColorToBin()), unaffected by this.
        static constexpr float kMinLogLuminance = -6.0f;
        static constexpr float kLogLuminanceRange = 24.0f;

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
            // UE5.8's own Auto Exposure Histogram metering defaults: only the middle slice of the
            // histogram's CUMULATIVE population (by pixel count, not by bin index) feeds the
            // weighted-mean luminance -- the bottom `histogramLowPercent`% (deep shadow/near-black
            // pixels, which would otherwise pull exposure up whenever a scene has a lot of shadow
            // area on screen) and the top `100-histogramHighPercent`% (small, intense specular
            // highlights/fireflies -- including MegaLights' own residual per-pixel ReSTIR noise,
            // see EXPOSURE_USE_AUTO's own EngineConfig.h comment for why that mattered here
            // specifically) are trimmed before averaging. See AutoExposureAdapt.comp's own comment
            // for the exact trimmed-weighted-mean algorithm this drives.
            float histogramLowPercent = 80.0f;
            float histogramHighPercent = 98.3f;

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

            // Phase PP4: Volumetric Light Shafts / God Rays (Crepuscular Rays) -- radial screen-
            // space raymarch of this pass' own HDR input toward the sun's screen-projected position
            // (see RecordComposite's own `sunDirection`/`viewProj` parameters).
            float godRaysIntensity = 0.5f;
            float godRaysDecay = 0.95f;
            float godRaysDensity = 1.0f;
            float godRaysWeight = 0.25f;

            // Phase PP5: Panini Projection -- 0 = rectilinear (identity, off). See
            // PostProcessComposite.comp's own ApplyPaniniProjection comment for the algorithm.
            float paniniD = 0.0f;
            float paniniS = 0.0f;

            // Phase PP5: Local Contrast Enhancement / Sharpness (single-pass unsharp mask).
            float sharpenIntensity = 0.0f;
            float sharpenRadiusPixels = 1.0f;

            // Phase PP5: Film Grain -- animated, luminance-response-curved (more visible in
            // shadows/midtones, fades in highlights, matching real film stock).
            float filmGrainIntensity = 0.0f;
            float filmGrainResponseMidpoint = 0.5f;
        };

        // `bloomView` (renderer::BloomPass::GetOutputView(), its own upsample-chain mip 0) is
        // sampled by PostProcessComposite.comp's g_Bloom binding -- Phase PP2. `depthView`
        // (renderer::ClusterResolvePass::GetOutputDepthView(), render-resolution, fixed identity --
        // never ping-ponged) and `refractionOffsetView` (renderer::TransparentForwardPass::
        // GetRefractionOffsetView(), also fixed identity) are both Phase PP3: sampled through a
        // LINEAR sampler for their own implicit bilinear upsample to display resolution (this
        // pipeline has no display-resolution depth anywhere -- see DepthOfField.comp's own comment).
        // `skyViewLUTView` (Atmos weather system, Subtask 2): renderer::AtmosSkyPass's own Sky-View
        // LUT view. `volumetricFogView` (Atmos Subtask 3): renderer::AtmosVolumetricFogPass's own
        // integrated fog 3D texture. `cloudsView` (Atmos Subtask 4): renderer::AtmosCloudsPass's own
        // half-resolution cloud buffer. All sampled read-only through this pass' own m_LinearSampler
        // -- fixed identity for this pipeline's entire lifetime (no producer pass ever recreates its
        // own images after Init()), same convention as `depthView`/`refractionOffsetView` below.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D displayExtent, VkImageView hdrColorView, VkImageView bloomView,
            VkImageView depthView, VkImageView refractionOffsetView, VkImageView skyViewLUTView,
            VkImageView volumetricFogView, VkImageView cloudsView);

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
        // `viewProj`/`sunDirection` (Phase PP4): projects the sun's screen-space position for God
        // Rays -- `sunDirection` points FROM the light TOWARD the scene (same convention as
        // renderer::ClusterResolvePass::RecordResolve's own sunDirection parameter), so the sun
        // itself sits along -sunDirection from the camera.
        // `fovYRadians`/`aspectRatio` (Phase PP5, renderer::CameraFrameInfo's own fields): drive
        // Panini Projection's tangent-space conversion (PostProcessComposite.comp's own
        // `halfFovTanX`/`halfFovTanY`). `frameIndex` (Phase PP5): seeds Film Grain's own per-frame
        // animated noise hash (same pixel+frame hashing convention as ReflectionTrace.comp's own
        // GGX-VNDF sample decorrelation).
        // `cameraForward` (Atmos Subtask 3): see PostProcessParamsUBO's own cameraForwardX/Y/Z comment.
        void RecordComposite(VkCommandBuffer cmd, float deltaTimeSeconds, const Settings& settings,
            const maths::mat4& invViewProj, const maths::mat4& prevViewProj, const maths::vec3& cameraPositionWorld,
            const maths::mat4& viewProj, const maths::vec3& sunDirection, const maths::vec3& cameraForward,
            float fovYRadians, float aspectRatio, uint32_t frameIndex);

        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

        // Read-only access to the persistent { float currentEV100; float currentAvgLuminance; }
        // SSBO (see m_ExposureStateBuffer's own comment) for other passes that need the SAME
        // real, eye-adapted exposure this pass' own PostProcessComposite.comp reads -- currently
        // only renderer::debug::DebugBufferViewPass (Debug-only), so its HDR buffer-viewer entries
        // (Bloom, GI Composite, TAA/TSR output, etc.) tonemap with the same exposure normalization
        // instead of feeding raw real-lux radiance straight into ACES (which saturates to white).
        VkBuffer GetExposureStateBuffer() const { return m_ExposureStateBuffer.Handle(); }

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

            // Phase PP4: God Rays.
            float sunScreenU = 0.5f, sunScreenV = 0.5f;
            float sunScreenValid = 0.0f;
            float godRaysIntensity = 0.5f;
            float godRaysDecay = 0.95f;
            float godRaysDensity = 1.0f;
            float godRaysWeight = 0.25f;
            float _padGodRays = 0.0f;

            // Phase PP5
            float paniniD = 0.0f;
            float paniniS = 0.0f;
            float halfFovTanX = 1.0f;
            float halfFovTanY = 1.0f;

            float sharpenIntensity = 0.0f;
            float sharpenRadiusPixels = 1.0f;
            float filmGrainIntensity = 0.0f;
            float filmGrainResponseMidpoint = 0.5f;

            uint32_t frameIndex = 0u;
            float _padPP5a = 0.0f, _padPP5b = 0.0f, _padPP5c = 0.0f;

            // Atmos weather system, Subtask 2: current sun direction (points FROM the light TOWARD
            // the scene, same convention as `sunDirection` in RecordComposite's own parameter list),
            // fed to PostProcessComposite.comp's SkyViewLUTUVFromDirection() on a miss.
            float sunDirWorldX = 0.0f, sunDirWorldY = 0.0f, sunDirWorldZ = 0.0f, _padSky = 0.0f;

            // Atmos weather system, Subtask 3: camera forward axis (unit vector) -- lets
            // PostProcessComposite.comp recover each pixel's view-space depth (distance along this
            // axis, NOT Euclidean ray length) to look up renderer::AtmosVolumetricFogPass's own
            // froxel grid via atmos_volumetric_fog_mapping.glsl's ViewZToFroxelW().
            float cameraForwardX = 0.0f, cameraForwardY = 0.0f, cameraForwardZ = 0.0f, _padFog = 0.0f;
        };
        static_assert(sizeof(PostProcessParamsUBO) == 448,
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
            float histogramLowPercent = 80.0f;
            float histogramHighPercent = 98.3f;
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

#ifndef NDEBUG
        // Debug-only Auto Exposure telemetry (Task F11): a tiny persistently-mapped (CPU_ONLY,
        // mapped=true) 1-frame-latent copy of m_ExposureStateBuffer, recorded at the tail of every
        // RecordComposite() call and logged at the TOP of the next one -- same "record this frame,
        // consume next frame after the frame-fence wait already guarantees completion" pattern as
        // renderer::ClusterLODSelectionPass's own m_DebugDecisionReadbackBuffer/RecordDebugReadback
        // (see that class's own comment for the full fence-ordering argument, which applies
        // identically here since this pass also only ever runs on cmdLate). Exists purely so a
        // Debug build can log currentEV100/currentAvgLuminance to demo_log.txt every few frames for
        // empirical flicker validation (a raw scalar time series is far more reliable than diffing
        // screenshots) -- zero Release footprint, matching CLAUDE.md's Debug/Release separation rule.
        GpuBuffer m_DebugExposureReadbackBuffer;
        uint32_t m_DebugExposureLogFrameCounter = 0u;
#endif

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
