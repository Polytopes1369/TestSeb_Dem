#pragma once
// Nanite-style two-phase GPU occlusion culling, built on top of renderer::HZBPass's depth pyramid
// and reusing renderer::ClusterCullMetadata / ClusterCullViewParams / ExtractFrustumPlanes from
// ClusterCullingPass.h. See src/shaders/src/Culling/ClusterHZBOcclusionCull.comp for the single
// compute shader (two specializations: early/late) this class drives, and
// src/shaders/include/hzb_occlusion.glsl for the HZB projection/sampling test both specializations
// share.
//
// --- Why two phases ---
// At the start of a frame, the only HZB available is the one built from the *previous* frame's
// final depth buffer -- this frame hasn't rasterized anything yet. Testing every candidate against
// that stale HZB and drawing whatever passes would still miss two cases: (a) something that was
// occluded last frame but is revealed this frame (the occluder moved/was itself culled), and (b) a
// brand-new candidate never seen before. The early pass handles the common case cheaply (most
// on-screen geometry is stable frame to frame) by drawing immediately anything that was visible
// last frame AND still passes the stale HZB; everything else is deferred to a pending list. Once
// the early pass's draws have updated the real depth buffer and the HZB has been rebuilt from it,
// the late pass re-tests exactly that pending list against the now-current HZB, catching every
// disocclusion and every new candidate with a single, cheap, GPU-driven indirect dispatch sized
// from the pending list's own atomic count (see BuildDispatchIndirectArgs.comp) -- no CPU
// round-trip anywhere in the pipeline.
//
// Exactly like renderer::HZBPass / renderer::FeedbackBuffer / renderer::ClusterCullingPass, this
// class is a self-contained building block -- Init()/Shutdown()/per-frame Record*() only -- not
// wired into VulkanContext/main.cpp by this change.
//
// --- Per-frame sequence a caller must record, in order ---
//   1. UploadClusterMetadata(...)     -- whenever this frame's candidate list changes.
//   2. RecordClearFrame(cmd)          -- resets the pending list and both draw counters to 0.
//      Never touches the persisted visibility buffer (see ClearPersistedVisibility below).
//   3. RecordEarlyPass(cmd, viewParams, viewProj, clusterCount, softwareRasterThresholdPixels) --
//      tests every candidate against the HZB as it stands right now (i.e. built from the
//      *previous* frame -- this class never rebuilds the HZB itself, see
//      renderer::HZBPass::Generate). Anything confirmed visible is routed to either the early
//      indirect-draw list (average estimated triangle size >= softwareRasterThresholdPixels) or
//      GetSoftwareClusterListBuffer() (smaller -- consumed by
//      renderer::ClusterSoftwareRasterPass instead of a hardware indirect draw); anything
//      uncertain is appended to the pending list, its own hardware/software routing decided later
//      by RecordLatePass() instead.
//   4. Caller: draw GetEarlyIndirectCommandBuffer() / GetEarlyDrawCountBuffer() with
//      vkCmdDrawIndexedIndirectCount into the frame's real depth/color attachments -- the "early"
//      pass. RecordEarlyPass()'s own final barrier already makes both buffers visible to
//      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT / VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT.
//   5. Caller: transition the depth image to a sampled-readable layout and call
//      renderer::HZBPass::Generate(cmd) again, rebuilding the pyramid from the depth the early
//      pass just produced (the same HZBPass object/image passed to this class's Init -- its
//      VkImageView handle never changes, only the pixel data it points at, so this class's
//      descriptor binding needs no update between phases). IMPORTANT: HZBPass::Generate()'s own
//      final barrier only grants VK_ACCESS_2_SHADER_STORAGE_READ_BIT (it was written for a future
//      imageLoad-based consumer); this class instead samples the pyramid through a
//      COMBINED_IMAGE_SAMPLER (g_HZBTexture / textureLod in hzb_occlusion.glsl), which requires
//      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT. The caller must therefore record one additional
//      VkMemoryBarrier2 immediately after every HZBPass::Generate() call whose result this class
//      will sample -- srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, srcAccessMask =
//      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
//      dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT -- before RecordBuildLateDispatchArgs()
//      / RecordLatePass() (and, symmetrically, before next frame's RecordEarlyPass() if the
//      optional step 9 rebuild below is used).
//   6. RecordBuildLateDispatchArgs(cmd) -- a 1x1x1 dispatch converting the pending list's atomic
//      count into GetLateDispatchArgsBuffer()'s VkDispatchIndirectCommand.
//   7. RecordLatePass(cmd) -- vkCmdDispatchIndirect-sized re-test of exactly the pending list
//      against the now-fresh HZB. Anything visible is appended to the late indirect-draw list, and
//      the persisted visibility flag for every pending cluster is written here (this is the last
//      word on that cluster's visibility for the frame).
//   8. Caller: draw GetLateIndirectCommandBuffer() / GetLateDrawCountBuffer() -- the "late" pass,
//      revealing anything the early pass missed. RecordLatePass()'s own final barrier already
//      makes both buffers visible to VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT /
//      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT.
//   9. Caller may optionally rebuild the HZB once more from the now-complete depth buffer, for
//      next frame's step 3 -- outside this class's scope, exactly like HZBPass's own per-frame
//      contract. If used, the same additional SHADER_SAMPLED_READ barrier described in step 5
//      above must follow it, before next frame's RecordEarlyPass().
//
// This sequence is self-correcting on the very first frame even though the HZB's initial content
// is undefined (HZBPass::Init only transitions its image's layout, it never clears it): since
// ClearPersistedVisibility() zeroes every entry, no candidate can take the early pass's
// "wasVisibleLastFrame" branch on frame 1, so every single candidate is deferred to the pending
// list regardless of what garbage the stale HZB contains. The early draw list is therefore empty,
// the depth buffer entering step 5 is exactly the frame's own clear value, and the late pass
// re-tests everything against an HZB honestly built from that (assumed cleared to 1.0, this
// engine's non-reversed [0, 1] depth convention -- see maths::mat4::PerspectiveVulkan) empty depth
// buffer, which is maximally permissive and correctly lets everything draw.

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/ClusterCullingPass.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    // GLSL-friendly, std140-compatible mirror of HZBOcclusionViewParams in hzb_occlusion.glsl:
    // the combined view-projection matrix plus the fixed HZB pyramid dimensions, used to project a
    // cluster's AABB into HZB screen space and pick a mip level.
    struct HZBOcclusionViewParams {
        maths::mat4 viewProj;
        float hzbMip0Width = 0.0f;
        float hzbMip0Height = 0.0f;
        float hzbMipCount = 0.0f;
        // 1 / tan(fovYRadians * 0.5) -- the camera projection matrix's own Y-scale term (see
        // maths::mat4::PerspectiveVulkan's `g`), i.e. abs(proj.m[5]). Used by
        // ClusterHZBOcclusionCull.comp's ShouldUseSoftwareRaster() to estimate a cluster's
        // projected screen size for hardware/software rasterization routing -- not derivable from
        // viewProj alone, since that is already the combined view*proj matrix.
        float projScaleY = 0.0f;
    };
    static_assert(sizeof(HZBOcclusionViewParams) == 80,
        "HZBOcclusionViewParams must match HZBOcclusionViewParams in hzb_occlusion.glsl exactly (std140 layout)");

    class ClusterOcclusionCullingPass {
    public:
        ClusterOcclusionCullingPass() = default;

        ClusterOcclusionCullingPass(const ClusterOcclusionCullingPass&) = delete;
        ClusterOcclusionCullingPass& operator=(const ClusterOcclusionCullingPass&) = delete;

        // Allocates every buffer sized from `maxClusters` (cluster metadata, persisted visibility,
        // pending list, both indirect-draw lists), the HZB sampler (nearest filtering/mipmapping --
        // required so mip selection via textureLod never blends two mips' max-depth values
        // together, which would silently break the occlusion test's conservativeness), and both
        // compute pipelines (the early/late specializations of ClusterHZBOcclusionCull.comp, plus
        // BuildDispatchIndirectArgs.comp). `hzbFullView`/`hzbMip0Extent`/`hzbMipLevelCount` must
        // come from an already-initialized renderer::HZBPass (GetFullView() / GetMipExtent(0) /
        // GetMipLevelCount()) -- that pyramid's image content does not need to be valid yet, only
        // its view/dimensions, since this class only samples it per-dispatch, never at Init time.
        //
        // Does NOT clear the persisted visibility buffer -- call ClearPersistedVisibility() once,
        // separately, before the first frame's RecordEarlyPass() (mirroring
        // renderer::GpuGeometryPagePool::Init()/ClearPageTable()'s split for the same reason: the
        // clear needs a command buffer the caller records and submits on their own schedule).
        void Init(VkDevice device, VmaAllocator allocator, uint32_t maxClusters,
            VkImageView hzbFullView, VkExtent2D hzbMip0Extent, uint32_t hzbMipLevelCount);

        void Shutdown();

        // Zeroes every entry of the persisted visibility buffer and inserts the barrier making
        // that clear visible to RecordEarlyPass()'s first read. Must be recorded exactly once,
        // after Init(), before the first frame's RecordEarlyPass() -- never again afterwards (the
        // buffer is meant to persist across every subsequent frame; see the class comment for why
        // an all-zero start is self-correcting even against an uninitialized HZB).
        void ClearPersistedVisibility(VkCommandBuffer cmd);

        // Uploads this frame's candidate cluster list (clusters.size() must be <= maxClusters)
        // into the metadata SSBO via a host-visible staging buffer + one-time command buffer copy,
        // mirroring VulkanContext::UploadEntityData's staging pattern (identical in spirit to
        // renderer::ClusterCullingPass::UploadClusterMetadata). A caller need not re-upload every
        // frame if the candidate list is unchanged from the previous frame.
        void UploadClusterMetadata(VkCommandPool commandPool, VkQueue queue, const std::vector<ClusterCullMetadata>& clusters);

        // Resets the pending list's count, both draw counters (early, late), and the software
        // cluster list's count to 0, with the barrier making that clear visible to
        // RecordEarlyPass()'s dispatch. Must be recorded once per frame, before RecordEarlyPass().
        // Never touches the persisted visibility buffer.
        void RecordClearFrame(VkCommandBuffer cmd);

        // Records the early specialization's dispatch: uploads `viewParams` (frustum planes +
        // camera position) and the HZB projection params (built from `viewProj` plus the fixed
        // pyramid dimensions captured at Init) into their UBOs, then dispatches one invocation per
        // candidate cluster in [0, clusterCount). Every cluster confirmed visible against the
        // *current* HZB (i.e. still holding last frame's data -- see the class comment) is routed,
        // by ClusterHZBOcclusionCull.comp's ShouldUseSoftwareRaster(), to either the early
        // indirect-draw list or GetSoftwareClusterListBuffer() based on its estimated average
        // triangle screen size versus `softwareRasterThresholdPixels` (typically 4-8 pixels);
        // every cluster whose visibility could not be confirmed is compacted into the pending list
        // instead, its routing decided later by RecordLatePass(). Ends with the barrier making the
        // early indirect-draw list, its draw count, the pending list, and the software cluster
        // list all visible to both a later vkCmdDrawIndexedIndirectCount
        // (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) and RecordBuildLateDispatchArgs()/
        // RecordLatePass()/renderer::ClusterSoftwareRasterPass's compute reads
        // (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT).
        // `projScaleY` is 1 / tan(fovYRadians * 0.5) -- abs(proj.m[5]) of the camera's own
        // (non-combined) projection matrix, see HZBOcclusionViewParams::projScaleY -- passed
        // separately from `viewProj` since the combined view*proj matrix cannot itself recover it.
        void RecordEarlyPass(VkCommandBuffer cmd, const ClusterCullViewParams& viewParams, const maths::mat4& viewProj,
            float projScaleY, uint32_t clusterCount, float softwareRasterThresholdPixels);

        // Records the 1x1x1 dispatch that converts the pending list's atomic count into
        // GetLateDispatchArgsBuffer()'s VkDispatchIndirectCommand (group count = ceil(pendingCount
        // / 64), matching ClusterHZBOcclusionCull.comp's local_size_x). Must be recorded after
        // RecordEarlyPass() (and, implicitly, after the caller has drawn the early list and
        // rebuilt the HZB -- see the class-level sequence -- since RecordLatePass() immediately
        // follows and depends on that fresh HZB being bound). Ends with the barrier making the
        // dispatch args visible to RecordLatePass()'s vkCmdDispatchIndirect
        // (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT / VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, the same
        // stage/access vkCmdDispatchIndirect's indirect-buffer read uses as vkCmdDrawIndirect's).
        void RecordBuildLateDispatchArgs(VkCommandBuffer cmd);

        // Records the late specialization's indirect dispatch (vkCmdDispatchIndirect over
        // GetLateDispatchArgsBuffer()): re-tests exactly the pending list's entries against the
        // HZB as it stands *now* (the caller must have rebuilt it, from the early pass's depth
        // output, before this call -- see the class-level sequence). Every cluster visible against
        // that fresh HZB is routed to either the late indirect-draw list or
        // GetSoftwareClusterListBuffer(), using the same softwareRasterThresholdPixels value
        // RecordEarlyPass() was called with this frame (cached internally and re-pushed here,
        // since RecordBuildLateDispatchArgs()'s intervening dispatch binds a different pipeline
        // layout and would otherwise invalidate it). The persisted visibility flag for every
        // pending cluster is written here too (its final value for this frame). Ends with the
        // barrier making the late indirect-draw list, its draw count, and the software cluster
        // list visible to a later vkCmdDrawIndexedIndirectCount /
        // renderer::ClusterSoftwareRasterPass compute read.
        void RecordLatePass(VkCommandBuffer cmd);

        VkBuffer GetEarlyIndirectCommandBuffer() const { return m_EarlyIndirectCommandBuffer.Handle(); }
        VkBuffer GetEarlyDrawCountBuffer() const { return m_EarlyDrawCountBuffer.Handle(); }
        VkBuffer GetLateIndirectCommandBuffer() const { return m_LateIndirectCommandBuffer.Handle(); }
        VkBuffer GetLateDrawCountBuffer() const { return m_LateDrawCountBuffer.Handle(); }
        VkBuffer GetLateDispatchArgsBuffer() const { return m_LateDispatchArgsBuffer.Handle(); }
        // Clusters routed to renderer::ClusterSoftwareRasterPass instead of a hardware indirect
        // draw this frame (both by the early and late specializations) -- { uint count; uint
        // clusterIndex[maxClusters]; }, index-aligned with GetClusterMetadataBuffer() exactly like
        // the indirect-draw lists' firstInstance values.
        VkBuffer GetSoftwareClusterListBuffer() const { return m_SoftwareClusterListBuffer.Handle(); }
        // Exposed so a downstream hardware raster pass (renderer::ClusterHardwareRasterPass) can
        // bind the exact same ClusterCullMetadataSSBO this pass populated -- both EARLY and LATE
        // draw commands' firstInstance (see ClusterHZBOcclusionCull.comp's EmitEarlyDraw/
        // EmitLateDraw) is this buffer's own array index for that cluster.
        VkBuffer GetClusterMetadataBuffer() const { return m_ClusterMetadataBuffer.Handle(); }
        uint32_t GetMaxClusters() const { return m_MaxClusters; }

    private:
        static constexpr uint32_t kWorkgroupSize = 64; // Matches ClusterHZBOcclusionCull.comp's local_size_x.

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only for UploadClusterMetadata()'s staging buffer.
        uint32_t m_MaxClusters = 0;

        // HZB pyramid dimensions captured at Init, baked into every frame's HZBOcclusionViewParams
        // upload alongside that frame's viewProj -- the pyramid itself never resizes at runtime
        // (see HZBPass's own "no runtime swapchain/depth resize" note), so these are fixed.
        float m_HZBMip0Width = 0.0f;
        float m_HZBMip0Height = 0.0f;
        float m_HZBMipCount = 0.0f;

        // Cached from RecordEarlyPass()'s parameter, re-pushed by RecordLatePass() -- see that
        // method's doc comment for why it cannot simply rely on RecordEarlyPass()'s own push
        // surviving RecordBuildLateDispatchArgs()'s intervening pipeline layout bind.
        float m_LastSoftwareRasterThresholdPixels = 0.0f;

        GpuBuffer m_ClusterMetadataBuffer;   // binding 0: ClusterCullMetadata[maxClusters], std430, GPU_ONLY.
        GpuBuffer m_ViewParamsBuffer;        // binding 1: CullingViewParams, std140 UBO, GPU_ONLY.
        GpuBuffer m_HZBParamsBuffer;         // binding 2: HZBOcclusionViewParams, std140 UBO, GPU_ONLY.
        // binding 3: HZB texture -- m_HZBView / m_HZBSampler below, not owned as a GpuBuffer.
        GpuBuffer m_VisibleLastFrameBuffer;  // binding 4: uint32[maxClusters], std430, GPU_ONLY, PERSISTENT across frames.
        GpuBuffer m_PendingListBuffer;       // binding 5: { uint count; uint clusterIndex[maxClusters]; }, std430, GPU_ONLY.
        GpuBuffer m_EarlyIndirectCommandBuffer; // binding 6: VkDrawIndexedIndirectCommand[maxClusters], std430, GPU_ONLY.
        GpuBuffer m_EarlyDrawCountBuffer;       // binding 7: single uint32 atomic counter, GPU_ONLY.
        GpuBuffer m_LateIndirectCommandBuffer;  // binding 8: VkDrawIndexedIndirectCommand[maxClusters], std430, GPU_ONLY.
        GpuBuffer m_LateDrawCountBuffer;        // binding 9: single uint32 atomic counter, GPU_ONLY.
        GpuBuffer m_SoftwareClusterListBuffer;  // binding 10: { uint count; uint clusterIndex[maxClusters]; }, std430, GPU_ONLY.

        GpuBuffer m_LateDispatchArgsBuffer; // VkDispatchIndirectCommand (3x uint32), GPU_ONLY, written by BuildDispatchIndirectArgs.comp.

        VkImageView m_HZBView = VK_NULL_HANDLE; // Not owned -- borrowed from the caller's renderer::HZBPass.
        VkSampler m_HZBSampler = VK_NULL_HANDLE; // Owned: nearest/nearest-mipmap, so textureLod never blends max-depth values across mips.

        // Main descriptor set/layout/pool, shared by both the early and late pipelines (both use
        // the exact same 11 bindings -- only the specialization constant differs).
        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE; // Shared with the BuildDispatchIndirectArgs set below.
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_EarlyPipeline = VK_NULL_HANDLE; // Specialized with LATE_PASS = false.
        VkPipeline m_LatePipeline = VK_NULL_HANDLE;  // Specialized with LATE_PASS = true.

        // Small, separate pipeline for BuildDispatchIndirectArgs.comp.
        VkDescriptorSetLayout m_BuildArgsSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_BuildArgsDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_BuildArgsPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_BuildArgsPipeline = VK_NULL_HANDLE;
    };

}
