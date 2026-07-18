#pragma once
// Deferred material-resolve pass: the decoupled second half of the Visibility Buffer pipeline --
// see src/shaders/src/Renderer/ClusterResolve.comp for the compute shader this class drives, which
// reconciles renderer::ClusterHardwareRasterPass's (ClusterID/TriangleID + depth) output against
// renderer::ClusterSoftwareRasterPass's (atomic depth+visibility) output per pixel, reconstructs
// each winning triangle's barycentric coordinates from the pixel position and the triangle's own
// screen-space plane, and shades the result into an owned output color image.
//
// This class owns none of its three VisBuffer inputs (they belong to the two rasterization passes
// that filled them) -- only the output color image, the view-params UBO, and the depth sampler.
// Exactly like every other piece of this Nanite-style pipeline, this is a self-contained building
// block -- Init()/Shutdown()/RecordResolve() only -- not wired into VulkanContext/main.cpp by this
// change; in particular, GetOutputColorImage() is not blitted to the swapchain here (a future
// integration step's job, exactly like the gap already documented when the flat/legacy draw path
// was first converted to write a VisBuffer instead of shading directly).
//
// --- Per-frame sequence a caller must record, in order ---
//   1. Caller: transition the hardware ClusterID/TriangleID VisBuffer images (whatever
//      renderer::VulkanContext's kVisBufferFormat images -- or renderer::ClusterHardwareRasterPass
//      -- rendered into as color attachments) from VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL to
//      VK_IMAGE_LAYOUT_GENERAL (this class binds them as plain storage images, whose descriptor
//      writes are fixed at Init() time to imageLayout = GENERAL -- see Init()'s doc comment), and
//      the hardware depth image to VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL. Also make sure
//      renderer::ClusterSoftwareRasterPass::RecordRaster()'s own barrier has already been recorded
//      this frame -- this pass reads all three VisBuffer sources as of whatever they contain at
//      the point RecordResolve() is called.
//   2. RecordResolve(cmd, viewProj) -- must use the exact same view-projection matrix both
//      rasterization passes used this frame, so the re-projected triangle used for barycentric
//      reconstruction exactly matches the one that determined pixel coverage originally. Ends with
//      the barrier making GetOutputColorImage() visible to a later sampled read or blit.

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/LightingTypes.h"
#include "renderer/MaterialParameterTable.h"

namespace renderer {

    class ClusterShadingBinPass; // Phase 1b: see InitBinnedResolve()/RecordResolveBinned()'s own comments.
    class VirtualShadowMapPass;  // Phase 3: see SetVirtualShadowMap()'s own comment.
    class VirtualTextureManager; // Step 4: see SetVirtualTexture()'s own comment.

    class ClusterResolvePass {
    public:
        ClusterResolvePass() = default;

        ClusterResolvePass(const ClusterResolvePass&) = delete;
        ClusterResolvePass& operator=(const ClusterResolvePass&) = delete;

        // R16G16B16A16_SFLOAT (linear HDR): this pass' shaded output is a genuine linear radiance
        // value (sun diffuse term + renderer::MegaLightsPass's own additive RMW composite, whose
        // inverse-square point-light contributions are physically unbounded near a light), not a
        // display-ready [0,1] color -- an 8-bit UNORM target hard-clips (no soft rolloff) anything
        // above 1.0, which is exactly what produced this codebase's "burned"/blown-out highlights
        // bug once renderer::MegaLightsPass started compositing real point lights into this image
        // (mirrors why renderer::ScreenTracePass::kOutputFormat is already this same HDR format for
        // the analogous indirect-GI radiance). The actual display-referred tonemap curve now lives
        // in renderer::PostProcessPass, the final step after renderer::TAATSRPass's own temporal
        // resolve -- correct order for both: TAA's variance clipping wants linear HDR input, and
        // tonemapping HDR only once, at the very end, avoids compounding a non-linear curve through
        // every additive compositing step this value still passes through (renderer::MegaLightsPass,
        // renderer::GICompositePass, renderer::TransparentForwardPass) after this pass writes it.
        static constexpr VkFormat kOutputColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        // Minimal GBuffer, written alongside the shaded color above from the exact same per-pixel
        // normal/albedo/depth this shader already reconstructs in its Step 2/3 (previously computed
        // and discarded) -- consumed by renderer::ScreenProbeGIPass for probe placement/tracing,
        // the bilateral gather reconstruction, and the ClusterResolve.comp debug view modes 12
        // (motion vectors) / 14 (spatial probes).
        static constexpr VkFormat kOutputNormalFormat = VK_FORMAT_R16G16_SFLOAT;   // Octahedral-encoded world-space normal (include/octahedral.glsl).
        static constexpr VkFormat kOutputDepthFormat = VK_FORMAT_R32_SFLOAT;       // The winning (hw-vs-sw arbitrated) NDC depth -- not stored anywhere else.
        static constexpr VkFormat kOutputAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;  // The real per-material PBR base color (renderer::MaterialParameterTable), pre-lighting.
        // R=roughness, G=metallic, sampled from renderer::MaterialParameterTable (materialID looked
        // up via ClusterCullMetadata) -- kept as its OWN image rather than reusing the albedo
        // image's alpha channel (that channel is written but never read by any consumer today; a
        // future reader should not have to guess whether "1.0" there means "opaque" or "unset").
        // This is the channel Phase 2 (Lumen-style reflections) will read.
        static constexpr VkFormat kOutputRoughnessMetallicFormat = VK_FORMAT_R8G8_UNORM;
        // Substrate integration (Phase S3): this pixel's clamped materialID, 0xFFFF sentinel for
        // background -- see ClusterResolve.comp's own g_OutputMaterialID binding comment. R16_UINT
        // (not R8_UINT): kMaxMaterials is 32 today but this headroom costs nothing extra.
        static constexpr VkFormat kOutputMaterialIDFormat = VK_FORMAT_R16_UINT;

        // Allocates the output color image (sized to `renderExtent`, transitioned once to
        // VK_IMAGE_LAYOUT_GENERAL via a blocking one-time submit, mirroring HZBPass::Init /
        // renderer::ClusterSoftwareRasterPass::Init's own one-shot transition pattern), the
        // nearest-filter depth sampler, the view-params UBO, and the compute pipeline/descriptor
        // set. None of `clusterMetadataBuffer` / `compressedPhysicalPoolBuffer` /
        // `hwClusterIDView` / `hwTriangleIDView` / `hwDepthView` / `swVisBufferAtomicView` are
        // owned here -- they must all come from the same frame's
        // renderer::ClusterHardwareRasterPass / ClusterSoftwareRasterPass /
        // GpuGeometryPagePool / HZBPass-driven depth image instances. `hwClusterIDView` /
        // `hwTriangleIDView`'s descriptor writes are fixed here at Init() time to
        // imageLayout = VK_IMAGE_LAYOUT_GENERAL and `hwDepthView`'s to
        // VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL -- the caller must guarantee those images
        // are actually in those exact layouts every time RecordResolve() executes (see the class
        // comment's per-frame sequence), since a descriptor's declared imageLayout is fixed at
        // write time, not re-validated per use.
        // `maskImageInfos` is renderer::ProceduralMaskGenerator::GetMaskImageInfos(), bound as
        // binding 8 for ClusterResolve.comp's soft opacity-mask edge blending (mask_sampling.glsl).
        // `wpoGlobalsBuffer` is renderer::ClusterRenderPipeline's own WPOGlobalsUBO (bound read-only
        // here as binding 12) -- this shader must reapply the exact same World Position Offset sway
        // deformation (wpo_deformation.glsl's ApplyWPODeformation) that both rasterizers already
        // applied to the vertices they actually drew, so its own re-projection for barycentric
        // reconstruction operates on the same deformed triangle, not the rest-pose one.
        // `materialTable` is renderer::GenerateShowcaseMaterialTable()'s result (see
        // ClusterRenderPipelineCreateInfo::materialTable) -- copied once into the GPU SSBO here,
        // not retained by reference (the caller's own copy may not outlive this call).
        // `splineControlPointsBuffer` is renderer::ClusterRenderPipeline's own
        // m_SplineControlPointsBuffer (see ClusterHardwareRasterPass::Init's identical parameter
        // comment) -- bound read-only at binding 26 (the first free slot past this shader's full
        // 0-25 range, 0-24 original + 25 Substrate's g_OutputMaterialID) so ClusterResolve.comp can
        // re-derive the same spline-bent triangle both rasterizers already drew (Phase 1, Nanite
        // advanced). Retained (like entityDataBuffer) so InitBinnedResolve() below needs no
        // duplicate parameter for it.
        // `boneMatricesBuffer` (skeletal-animation feature) is animation::SkeletalAnimator::
        // GetBoneMatricesBuffer() -- bound read-only at binding 29 (the first free slot past this
        // shader's full 0-28 range, including the Atmos weather Sky-View LUT/Cloud Shadow Map
        // bindings 27/28) so ClusterResolve.comp/ClusterResolveBinned.comp can re-derive the same
        // skinned triangle both rasterizers already drew. Same "retained, no duplicate parameter"
        // convention as splineControlPointsBuffer above.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
            VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
            VkImageView hwClusterIDView, VkImageView hwTriangleIDView, VkImageView hwDepthView,
            VkImageView swVisBufferAtomicView, const std::vector<VkDescriptorImageInfo>& maskImageInfos,
            VkBuffer wpoGlobalsBuffer, VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer,
            const std::array<MaterialParameters, kMaxMaterials>& materialTable,
            VkBuffer splineControlPointsBuffer, VkBuffer boneMatricesBuffer);

        void Shutdown();

        // Uploads `viewProj` into the view-params UBO and dispatches one invocation per output
        // pixel (local_size_x/y = 8). Ends with the barrier making GetOutputColorImage() visible to
        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT / VK_ACCESS_2_SHADER_SAMPLED_READ_BIT (a future
        // sampled read) and VK_PIPELINE_STAGE_2_COPY_BIT / VK_ACCESS_2_TRANSFER_READ_BIT (a future
        // blit to the swapchain) -- whichever the caller ends up using.
        // `prevViewProj` is the previous frame's combined matrix (renderer::ClusterRenderPipeline's
        // own m_PrevViewProj) -- used only by DEBUG_VIEW_MOTION_VECTORS to reproject this frame's
        // reconstructed world position; pass an identity matrix on the very first frame (no
        // previous frame exists yet). `sun` (Phase 3 direction, points FROM the light TOWARD the
        // scene; real-photometric color+intensity in LUX, see renderer::DirectionalLight's own
        // comment) feeds the direct-lighting term's physically-based Lambertian evaluation AND its
        // shadow lookup (see ClusterResolve.comp's own Step 3 comment) -- `sun.direction` must be
        // the SAME direction renderer::VirtualShadowMapPass::RecordBeginFrame() was called with
        // this frame. `cameraPositionWorld` (Substrate integration): feeds EvaluateSubstrateMaterial's
        // specular/Fresnel view-direction term -- the pre-Substrate shader had no view-dependent term
        // at all, so this is new; same value renderer::ClusterRenderPipeline already threads into
        // ReflectionPass/TransparentForwardPass's own view-params UBOs. `surfaceWetness`/
        // `snowCoverage` (Atmos weather system, surface response extension): renderer::
        // AtmosClimatePass::GetSurfaceWetness()/GetSnowCoverage() -- [0,1] scalars integrated once
        // per frame from the live climate state, threaded straight into ResolveViewParamsUBO's own
        // (previously dead-padding) fields and consumed by substrate_bsdf.glsl's
        // ApplySurfaceWeather -- see that function's own comment for the exact wet/snow BSDF
        // modulation.
        // `glintDensityScale`/`glintIntensityScale` (UE5.8 rendering-parity gap G5, Substrate Glint/
        // sparkle): Debug-only tuning multipliers on every material's authored SubstrateSlab::
        // glintDensity/glintIntensity, threaded into ResolveViewParamsUBO and consumed by
        // substrate_bsdf.glsl's EvaluateSubstrateGlint. Both default to 1.0 (the authored value
        // unchanged) -- Release passes 1.0 (no toggle exists there), matching every other Debug-only
        // tuning knob's Release-always-on convention (see renderer::ClusterRenderPipeline's own
        // SetDebugGlint* setters); Debug drives them from the Post FX ImGui sliders.
        // `mixMaskSharpnessScale` (UE5.8 rendering-parity gap G6, Substrate horizontal mixing):
        // Debug-only tuning multiplier on every horizontally-mixed material's authored mixContrast
        // (the blend sharpness), threaded into ResolveViewParamsUBO and consumed by
        // substrate_bsdf.glsl's EvaluateSubstrateMixMask. Defaults to 1.0 (authored value unchanged);
        // Release passes 1.0 (no toggle exists there), driven in Debug by the Post FX "Mix Sharpness"
        // slider (renderer::ClusterRenderPipeline's own SetDebugMixMaskSharpnessScale setter).
        // `globalTimeSeconds` (Wave 2, UE5.8 caustics/light-function parity): real elapsed time,
        // threaded into ResolveViewParamsUBO.timeSeconds and consumed by ComputeUnderwaterCaustics's
        // scroll animation (procedural_light_modulation.glsl) -- EvaluateSunLightFunction does not
        // need it. Defaults to 0.0 (a static but still-correct caustics pattern) so every pre-Wave-2
        // caller compiles unchanged.
        void RecordResolve(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj,
            const DirectionalLight& sun, const maths::vec3& cameraPositionWorld, float surfaceWetness, float snowCoverage,
            float glintDensityScale = 1.0f, float glintIntensityScale = 1.0f, float mixMaskSharpnessScale = 1.0f,
            float globalTimeSeconds = 0.0f, uint32_t debugViewMode = 0);

        // --- Phase 1b: binned resolve path (renderer::ClusterShadingBinPass) ---
        // Second-phase init, called once after BOTH Init() above AND `shadingBinPass.Init()` have
        // already run -- a genuine ordering dependency, not an arbitrary split: this method's own
        // descriptor set needs `shadingBinPass`'s sorted-pixel-list/bin-offsets/bin-histogram
        // buffers, while `shadingBinPass.Init()` itself needs THIS class's 5 output image views
        // (its own Classify stage writes background pixels directly into them) -- the two classes'
        // Init() calls cannot be ordered as a single linear dependency chain, so this second phase
        // exists specifically to break the cycle. Reuses every resource the first Init() already
        // received/created (cluster metadata buffer, compressed pool buffer, mask image infos, WPO
        // globals buffer, view-params UBO, material params SSBO, output images) -- retained as
        // members after the first Init() call specifically so this method needs no duplicate
        // parameters for any of them.
        void InitBinnedResolve(VkDevice device, VkCommandPool commandPool, VkQueue queue,
            const ClusterShadingBinPass& shadingBinPass);

        // Uploads `viewProj` into the SAME ResolveViewParamsUBO RecordResolve() uses (prevViewProj
        // left at whatever RecordResolve() last wrote -- this path never reads it, since it never
        // serves DEBUG_VIEW_MOTION_VECTORS), then issues MaterialParameterTable::kMaxMaterials
        // indirect dispatches (one per material bin, `binIndex` a push constant) against
        // `shadingBinPass`'s own GetBinDispatchArgsBuffer(). Caller must have already recorded
        // `shadingBinPass.RecordClassifyAndSort()` earlier this frame (this method only reads that
        // work's output) -- see renderer::ClusterRenderPipeline::RecordFrame's own call site for
        // the exact per-frame ordering (this path replaces RecordResolve() entirely whenever
        // `camera.debugViewMode == 0`; Release always takes this path, see that field's own
        // Debug-only gating in core/Camera.h). Ends with the identical trailing barrier
        // RecordResolve() itself ends with. `sun`/`cameraPositionWorld`/`surfaceWetness`/
        // `snowCoverage` -- see RecordResolve()'s own comment (this Release-live path needs the
        // exact same weather modulation the Debug-only full-screen path gets, since this feature
        // must work correctly in Release, not just under a debug view mode).
        // `glintDensityScale`/`glintIntensityScale` (UE5.8 rendering-parity gap G5) and
        // `mixMaskSharpnessScale` (UE5.8 rendering-parity gap G6): see RecordResolve's own comment --
        // this Release-live path needs the same tuning inputs (all 1.0 in Release -> the material's
        // authored sparkle/mix-sharpness renders unchanged). No defaults here: `shadingBinPass` (a
        // non-defaulted trailing reference) follows them, so both callers pass them explicitly.
        void RecordResolveBinned(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const DirectionalLight& sun, const maths::vec3& cameraPositionWorld, float surfaceWetness, float snowCoverage,
            float glintDensityScale, float glintIntensityScale, float mixMaskSharpnessScale,
            float globalTimeSeconds, const ClusterShadingBinPass& shadingBinPass);

        // Binds Phase 3's renderer::VirtualShadowMapPass resources (physical page atlas + sampler,
        // page table, feedback buffer, sun clipmap levels UBO) into BOTH this pass's descriptor
        // sets (the Debug-only full-screen set AND the always-live binned-resolve set -- see
        // ClusterResolve.comp's / ClusterResolveBinned.comp's own binding-15-18 / 14-17 comments).
        // Must be called exactly once after Init() AND InitBinnedResolve() have BOTH already run
        // (needs both sets to already be allocated), before the first RecordResolve()/
        // RecordResolveBinned() call. Point light cube faces are NOT bound here -- unlike
        // renderer::SurfaceCachePass, this shader's direct-lighting term only shadows the sun (see
        // this phase's own plan for why point-light direct-visible shading was left unchanged).
        void SetVirtualShadowMap(const VirtualShadowMapPass& vsm);

        // Step 4 (Virtual Texturing): binds `vt`'s page table + physical pool atlas into BOTH this
        // pass's descriptor sets (mirrors SetVirtualShadowMap()'s own two-set-write pattern exactly
        // -- see ClusterResolve.comp's/ClusterResolveBinned.comp's own binding-2x comments). Only
        // `vt`'s pool 0 is wired (this demo's Albedo-only RVT layer) -- every one of the shader's
        // K_MAX_VT_PHYSICAL_POOLS array slots is written with THAT SAME view/sampler (rather than
        // left unwritten) because this engine does not enable
        // VkPhysicalDeviceVulkan12Features::descriptorBindingPartiallyBound, so every element of a
        // statically-referenced descriptor array binding must hold a valid descriptor (see
        // virtual_texture_limits.glsl's own comment on why this is a bounded, not runtime, array) --
        // harmless, since ClusterResolve.comp/ClusterResolveBinned.comp only ever index slot 0.
        // `worldMinXZ`/`worldMaxXZ` must be the SAME world-space volume bounds the caller's
        // renderer::VirtualTextureRenderPass instance was Init()'d with (see that class's own
        // VirtualTextureVolumeBounds), so a world position maps to the same virtual UV both a
        // page-render and this consuming sample agree on. `feedbackBuffer` is renderer::
        // VirtualTextureStreamingCoordinator::GetFeedbackDeviceBuffer(). Must be called exactly once
        // after Init() AND InitBinnedResolve() have BOTH already run, before the first
        // RecordResolve()/RecordResolveBinned() call -- same ordering contract as
        // SetVirtualShadowMap().
        void SetVirtualTexture(const VirtualTextureManager& vt, const maths::vec2& worldMinXZ,
            const maths::vec2& worldMaxXZ, VkBuffer feedbackBuffer);

        // Atmos weather system, Subtask 5: binds renderer::AtmosSkyPass's Sky-View LUT (ambient
        // replacement) and renderer::AtmosCloudsPass's Cloud Shadow Map (sun term modulation) into
        // BOTH descriptor sets -- mirrors SetVirtualShadowMap()'s own two-set-write pattern exactly.
        // Must be called after both producer passes' own Init(), before the first RecordResolve()/
        // RecordResolveBinned() call -- same ordering contract as SetVirtualShadowMap().
        void SetAtmosCloudLighting(VkSampler skyViewLUTSampler, VkImageView skyViewLUTView,
            VkSampler cloudShadowSampler, VkImageView cloudShadowView);

        VkImage GetOutputColorImage() const { return m_OutputColorImage; }
        VkImageView GetOutputColorView() const { return m_OutputColorView; }
        VkImageView GetOutputNormalView() const { return m_OutputNormalView; }
        VkImageView GetOutputDepthView() const { return m_OutputDepthView; }
        VkImageView GetOutputAlbedoView() const { return m_OutputAlbedoView; }
        VkImageView GetOutputRoughnessMetallicView() const { return m_OutputRoughnessMetallicView; }
        // Substrate integration (Phase S3): lets ReflectionPass/MegaLightsPass re-fetch the full
        // Slab data straight from g_MaterialParams instead of the GBuffer growing a field per
        // Substrate parameter -- see ClusterResolve.comp's own g_OutputMaterialID binding comment.
        VkImageView GetOutputMaterialIDView() const { return m_OutputMaterialIDView; }
        // Substrate integration (Phase S3): the SAME SSBO handle this pass already uploaded
        // Init()'s materialTable into -- ReflectionPass/MegaLightsPass bind this directly rather
        // than each owning a redundant duplicate upload (TransparentForwardPass's own separate copy
        // predates this getter and is left as-is, see that class's own m_MaterialParamsBuffer).
        VkBuffer GetMaterialParamsBuffer() const { return m_MaterialParamsBuffer.Handle(); }

    private:
        static constexpr uint32_t kWorkgroupSize = 8; // Matches ClusterResolve.comp's local_size_x/y.

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only to vmaDestroyImage() in Shutdown().
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_OutputColorImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputColorAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputColorView = VK_NULL_HANDLE;
        VkImage m_OutputNormalImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputNormalAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputNormalView = VK_NULL_HANDLE;
        VkImage m_OutputDepthImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputDepthAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputDepthView = VK_NULL_HANDLE;
        VkImage m_OutputAlbedoImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAlbedoAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputAlbedoView = VK_NULL_HANDLE;
        VkImage m_OutputRoughnessMetallicImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputRoughnessMetallicAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputRoughnessMetallicView = VK_NULL_HANDLE;
        // Substrate integration (Phase S3): r16ui, this pixel's clamped materialID (0xFFFF = no
        // geometry) -- see ClusterResolve.comp's own g_OutputMaterialID binding comment.
        VkImage m_OutputMaterialIDImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputMaterialIDAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputMaterialIDView = VK_NULL_HANDLE;

        VkSampler m_DepthSampler = VK_NULL_HANDLE; // Nearest filtering, matching HZBPass's own depth-sampling convention.

        GpuBuffer m_ViewParamsBuffer;     // ResolveViewParamsUBO, std140, GPU_ONLY.
        // Step 4: VirtualTextureVolumeUBO, std140, CPU_TO_GPU mapped -- filled once by
        // SetVirtualTexture() (the volume bounds never change per-frame, unlike m_ViewParamsBuffer
        // above), not recreated by InitBinnedResolve() (shared by both descriptor sets, same
        // convention as m_ViewParamsBuffer itself).
        GpuBuffer m_VTVolumeUBO;
        // renderer::MaterialParameters[kMaxMaterials], filled once at Init() time via
        // vkCmdUpdateBuffer from Init()'s own `materialTable` parameter -- not a per-frame upload
        // (unlike m_ViewParamsBuffer above).
        GpuBuffer m_MaterialParamsBuffer;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        // --- Phase 1b: resources retained from Init() purely so InitBinnedResolve() needs no
        // duplicate parameters for them (see that method's own comment). ---
        static constexpr uint32_t kResolveBinnedWorkgroupSize = 64; // Matches ClusterResolveBinned.comp's local_size_x.
        VkBuffer m_ClusterMetadataBuffer = VK_NULL_HANDLE; // Borrowed, same handle Init() received.
        VkBuffer m_CompressedPoolBuffer = VK_NULL_HANDLE;  // Borrowed, same handle Init() received.
        VkBuffer m_WPOGlobalsBuffer = VK_NULL_HANDLE;      // Borrowed, same handle Init() received.
        VkBuffer m_EntityTransformBuffer = VK_NULL_HANDLE; // Borrowed, same handle Init() received.
        VkBuffer m_EntityDataBuffer = VK_NULL_HANDLE;      // Borrowed, same handle Init() received.
        VkBuffer m_SplineControlPointsBuffer = VK_NULL_HANDLE; // Borrowed, same handle Init() received (Phase 1, Nanite advanced).
        VkBuffer m_BoneMatricesBuffer = VK_NULL_HANDLE;    // Borrowed, same handle Init() received (skeletal-animation feature).
        std::vector<VkDescriptorImageInfo> m_MaskImageInfos; // Copy of Init()'s own `maskImageInfos` parameter.

        VkDescriptorSetLayout m_ResolveBinnedSetLayout = VK_NULL_HANDLE;
        // Separate pool from m_DescriptorPool above (sized for exactly the 1 extra set below)
        // rather than recomputing m_DescriptorPool's own pool sizes to cover both sets -- keeps
        // this Phase 1b addition from touching Init()'s already-working, already-verified pool
        // sizing at all.
        VkDescriptorPool m_ResolveBinnedDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_ResolveBinnedSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ResolveBinnedPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ResolveBinnedPipeline = VK_NULL_HANDLE;
    };

}
