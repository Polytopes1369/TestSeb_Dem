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

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    class ClusterShadingBinPass; // Phase 1b: see InitBinnedResolve()/RecordResolveBinned()'s own comments.
    class VirtualShadowMapPass;  // Phase 3: see SetVirtualShadowMap()'s own comment.

    class ClusterResolvePass {
    public:
        ClusterResolvePass() = default;

        ClusterResolvePass(const ClusterResolvePass&) = delete;
        ClusterResolvePass& operator=(const ClusterResolvePass&) = delete;

        // RGBA8: this codebase's simplest, most broadly supported color-attachment-and-storage-
        // capable format (mirrors why VulkanContext::kVisBufferFormat picks a mandatory-support
        // format over an exotic one), sufficient for the procedural, non-HDR shading
        // ClusterResolve.comp currently performs (see that shader's class comment on why no real
        // material/texture system exists yet).
        static constexpr VkFormat kOutputColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
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
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
            VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
            VkImageView hwClusterIDView, VkImageView hwTriangleIDView, VkImageView hwDepthView,
            VkImageView swVisBufferAtomicView, const std::vector<VkDescriptorImageInfo>& maskImageInfos,
            VkBuffer wpoGlobalsBuffer, VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer);

        void Shutdown();

        // Uploads `viewProj` into the view-params UBO and dispatches one invocation per output
        // pixel (local_size_x/y = 8). Ends with the barrier making GetOutputColorImage() visible to
        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT / VK_ACCESS_2_SHADER_SAMPLED_READ_BIT (a future
        // sampled read) and VK_PIPELINE_STAGE_2_COPY_BIT / VK_ACCESS_2_TRANSFER_READ_BIT (a future
        // blit to the swapchain) -- whichever the caller ends up using.
        // `prevViewProj` is the previous frame's combined matrix (renderer::ClusterRenderPipeline's
        // own m_PrevViewProj) -- used only by DEBUG_VIEW_MOTION_VECTORS to reproject this frame's
        // reconstructed world position; pass an identity matrix on the very first frame (no
        // previous frame exists yet). `sunDirection` (Phase 3, points FROM the light TOWARD the
        // scene) feeds the direct-lighting term's light direction AND its shadow lookup (see
        // ClusterResolve.comp's own Step 3 comment) -- must be the SAME direction
        // renderer::VirtualShadowMapPass::RecordBeginFrame() was called with this frame.
        void RecordResolve(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj,
            const maths::vec3& sunDirection, uint32_t debugViewMode = 0);

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
        // RecordResolve() itself ends with. `sunDirection` -- see RecordResolve()'s own comment.
        void RecordResolveBinned(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const maths::vec3& sunDirection, const ClusterShadingBinPass& shadingBinPass);

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

        VkImage GetOutputColorImage() const { return m_OutputColorImage; }
        VkImageView GetOutputColorView() const { return m_OutputColorView; }
        VkImageView GetOutputNormalView() const { return m_OutputNormalView; }
        VkImageView GetOutputDepthView() const { return m_OutputDepthView; }
        VkImageView GetOutputAlbedoView() const { return m_OutputAlbedoView; }
        VkImageView GetOutputRoughnessMetallicView() const { return m_OutputRoughnessMetallicView; }

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

        VkSampler m_DepthSampler = VK_NULL_HANDLE; // Nearest filtering, matching HZBPass's own depth-sampling convention.

        GpuBuffer m_ViewParamsBuffer;     // ResolveViewParamsUBO, std140, GPU_ONLY.
        // renderer::MaterialParameters[kMaxMaterials] (renderer::kMaterialParameterTable), filled
        // once at Init() time via vkCmdUpdateBuffer -- a CPU-authored constexpr table, not a
        // per-frame upload (unlike m_ViewParamsBuffer above), so Init() needs no extra caller-
        // supplied parameter for it.
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
