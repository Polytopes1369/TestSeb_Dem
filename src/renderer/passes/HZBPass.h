#pragma once
// Hierarchical Z-Buffer (HZB) pyramid: the depth-based visibility oracle renderer::
// ClusterOcclusionCullingPass's GPU-driven occlusion cull (src/shaders/src/Culling/
// ClusterHZBOcclusionCull.comp, via include/hzb_occlusion.glsl's IsClusterOccluded) samples every
// frame, for both its early (previous frame's HZB) and late (this frame's own early-pass depth)
// passes, to reject clusters/instances whose screen-space bounds are fully behind already-
// rasterized geometry, without reading back anything to the CPU.
//
// --- What this builds ---
// A single VkImage with a full mip chain, format VK_FORMAT_R32G32_SFLOAT (min depth in the R
// channel, max depth in G), built from the depth attachment the graphics pass rendered in the
// previous frame (see VulkanContext::m_DepthImage / GetDepthImageView()). Mip 0 of the pyramid is
// HALF the source depth buffer's resolution: every texel of every mip level, including mip 0, is
// produced by reducing a non-overlapping 2x2 block of the level directly below it (the source depth
// buffer for mip 0, the previous HZB mip for every level after that) down to a single (min, max)
// depth pair. This makes the whole pyramid one uniform reduction rule top to bottom instead of
// treating mip 0 as a special full-resolution copy.
//
// Two compute shaders implement the two kinds of reduction step:
//   - src/shaders/src/Culling/HZBBuildInit.comp : source depth buffer -> HZB mip 0. Reads the
//     source depth through a plain nearest sampler (a depth image bound as a writable attachment
//     cannot also be read as a storage image, and depth formats are not guaranteed STORAGE_IMAGE-
//     capable -- sampling is the portable way to read it from a compute shader).
//   - src/shaders/src/Culling/HZBReduce.comp   : HZB mip N-1 -> HZB mip N, for every N in
//     [1, mipCount). Both mips are bound as storage images (rg32f), read-only / write-only
//     respectively, via single-mip VkImageViews (m_MipViews).
//
// --- Synchronization ---
// The whole HZB image is transitioned to VK_IMAGE_LAYOUT_GENERAL exactly once, in Init(), and never
// leaves that layout: it is only ever touched by compute shaders (imageLoad/imageStore), never used
// as an attachment or read through a sampler, so no further layout transitions are needed. Between
// every dispatch inside Generate() -- including the very first one -- an explicit
// VkImageMemoryBarrier2 scoped to the exact mip subresource just written makes that dispatch's
// imageStore visible before the next dispatch's imageLoad of the same mip runs; see
// BarrierAfterMipWrite(). This is a pure execution/memory dependency (oldLayout == newLayout ==
// GENERAL throughout), not a layout transition, but is expressed as a full VkImageMemoryBarrier2 --
// rather than a generic VkMemoryBarrier2 -- so the dependency stays scoped to the one mip
// subresource that actually changed, instead of serializing against the whole image.
//
// --- Integration contract (not wired into VulkanContext/main.cpp by this class) ---
// Exactly like renderer::GpuGeometryPagePool / renderer::GeometryDecompressionPass /
// renderer::FeedbackBuffer, this class is a self-contained building block: it knows how to build
// the pyramid from a given depth image view, and nothing about when in the frame that should
// happen or who consumes the result. A caller wiring this into the frame loop must, in order:
//   1. Call Init() once, after the depth image + its (sampled) VkImageView already exist.
//   2. Every frame, after the depth graphics pass's vkCmdEndRendering(), record a
//      VkImageMemoryBarrier2 transitioning the depth image from
//      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL to
//      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL (srcStage = LATE_FRAGMENT_TESTS,
//      srcAccess = DEPTH_STENCIL_ATTACHMENT_WRITE, dstStage = COMPUTE_SHADER,
//      dstAccess = SHADER_SAMPLED_READ) -- sampling a depth image while it is still bound as a
//      writable attachment is invalid.
//   3. Call Generate(cmd) once, recording the whole pyramid build into the same command buffer.
//   4. A later occlusion-culling compute dispatch in the same command buffer may then sample
//      GetFullView() (all mips) or imageLoad an individual GetMipView(level); Generate()'s final
//      barrier already makes every mip's data visible to VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT /
//      VK_ACCESS_2_SHADER_STORAGE_READ_BIT.
//
// The depth image passed to Init() must have been created with VK_IMAGE_USAGE_SAMPLED_BIT in
// addition to VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT (see VulkanContext::Init()'s
// depthImageInfo.usage).

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "renderer/vulkan/GpuImage.h"

namespace renderer {

    class HZBPass {
    public:
        HZBPass() = default;

        HZBPass(const HZBPass&) = delete;
        HZBPass& operator=(const HZBPass&) = delete;

        // Format backing every mip of the pyramid: R32G32_SFLOAT (min depth in R, max depth in G).
        // Mandated by the Vulkan 1.3 core spec to support both SAMPLED_IMAGE and STORAGE_IMAGE
        // optimal-tiling usage, so no format-support query is needed before using it as a storage
        // image here.
        //
        // --- Investigated 2026-07-18: KEPT at R32G32_SFLOAT, deliberately NOT narrowed to
        // R16G16_SFLOAT ---
        // An earlier audit flagged this as possibly oversized (most engines, UE5 included, often
        // get away with R16F/R32F single-channel or R16G16F for an HZB) and asked for this to be
        // profiled before changing, not blindly downsized. Investigated by reading the actual
        // consumer -- src/shaders/include/hzb_occlusion.glsl's IsClusterOccluded(), sampled by both
        // the early and late specializations of src/shaders/src/Culling/ClusterHZBOcclusionCull.comp
        // (see renderer::ClusterOcclusionCullingPass) -- rather than guessing. Conclusion: NOT
        // confident this is safe without a compensating change, so the format stays R32G32_SFLOAT
        // pending real GPU profiling data. Reasoning:
        //   1. IsClusterOccluded()'s final verdict is `return nearestDepth < storedFarthestDepth;`
        //      -- a bare strict inequality with ZERO epsilon/bias margin anywhere in the function
        //      (confirmed by reading the whole function, and ClusterHZBOcclusionCull.comp's call
        //      sites -- the only bounds inflation nearby, InflateBoundsForWPO in
        //      cluster_culling_tests.glsl, is a spatial AABB slop term for World Position Offset
        //      animation, a no-op for the common maxWPOAmplitude==0 static-cluster case, and not a
        //      depth-quantization tolerance). Nothing here would absorb added rounding noise from a
        //      coarser stored-depth format.
        //   2. This engine's main depth buffer (which the HZB is built from every frame -- see this
        //      class' own comment) is reversed-Z: VK_COMPARE_OP_GREATER + clear-to-0.0f (see
        //      renderer::ClusterRenderPipeline's depth attachment setup) and maths::mat4::
        //      PerspectiveVulkan's m[10]/m[11]/m[14] terms map the far plane to ndc.z=0 and the near
        //      plane to ndc.z=1 (verified algebraically: ndc.z(d) = zNear*zFar/((zFar-zNear)*d) -
        //      zNear/(zFar-zNear) for a point d world units in front of the camera). Floating-point
        //      reversed-Z is specifically chosen to give roughly CONSTANT RELATIVE depth precision
        //      across the whole camera range (unlike a fixed-point/non-reversed buffer, which loses
        //      almost all its precision at far distances) -- fp16 keeps that same relative-precision
        //      shape but with an ~11-bit mantissa instead of fp32's ~24-bit one (~2^13x coarser), so
        //      the ABSOLUTE world-space depth gap it can still resolve grows with distance from the
        //      camera instead of staying sub-millimeter everywhere the way fp32 does here.
        //   3. Quantified against this project's actual camera (Camera.h: zNear=0.1, zFar=1000.0 --
        //      see Camera::m_Near/m_Far) via the ndc.z(d) formula above: applying fp16's unit
        //      roundoff (2^-11) to the stored ndc.z value and mapping back through d(ndc.z)/dd gives
        //      an analytic (not GPU-measured) estimate of roughly a 5mm blind spot at 10 world units
        //      from the camera, ~4cm by 100 units, and ~12cm by 500 units -- and past d~=620 units
        //      (i.e. the outer ~38% of this camera's own 1000-unit zFar) ndc.z drops below fp16's
        //      normal-float floor (2^-14) entirely into denormal representation, where the step size
        //      is fixed rather than continuing to scale with distance (worse, not better), and where
        //      some GPU/driver/shader-compiler combinations flush fp16 subnormals to zero outright.
        //      Given this project's scene content (procedural terrain, mountains, rivers -- see
        //      CLAUDE.md) legitimately places clusters at these mid-to-far distances, and the
        //      comparison above has no tolerance to absorb that error, this reads as a concrete,
        //      non-hypothetical risk of exactly the "occlusion culling false-positive" failure mode
        //      (an object incorrectly vanishing) the original audit called out, not just a
        //      theoretical one -- though this whole point-3 estimate is analytic, derived from the
        //      projection formula, NOT from an actual GPU capture, which is precisely why this is
        //      "not confident it's safe" rather than "confirmed broken."
        // To revisit: either (a) add a small, principled epsilon/bias to IsClusterOccluded()'s
        // comparison, sized to fp16's worst-case error at this project's zFar, and re-derive whether
        // the test stays meaningfully conservative with it, or (b) capture an actual GPU profile
        // (Nsight/PIX) showing this image's bandwidth/cache footprint is a measured bottleneck
        // before trading away precision for it -- the whole pyramid is tiny in absolute VRAM either
        // way (a handful of MB at typical resolutions), so this is a pure correctness-risk-vs-
        // marginal-bandwidth trade, not a memory-pressure one.
        static constexpr VkFormat kFormat = VK_FORMAT_R32G32_SFLOAT;

        // Allocates the full HZB mip chain sized from `sourceDepthExtent` (mip 0 is
        // ceil(sourceDepthExtent / 2); see the class comment for how deeper mips are derived and
        // where the reduction stops, at a 1x1 mip), one VkImageView per mip level plus one
        // full-pyramid view, both compute pipelines, their descriptor set layouts / pool / sets
        // (written once here, since `sourceDepthView` is expected to stay valid and unchanged for
        // this object's entire lifetime -- this engine does not support runtime swapchain/depth
        // resize), and the nearest-filter sampler used to read `sourceDepthView`. Also performs the
        // image's one-time VK_IMAGE_LAYOUT_UNDEFINED -> VK_IMAGE_LAYOUT_GENERAL transition via a
        // blocking one-time submit on `queue`/`commandPool`, mirroring
        // VulkanContext::UploadEntityData's one-shot command buffer pattern -- the HZB image then
        // stays in GENERAL for its entire lifetime (see the class comment), so Generate() never
        // needs to transition it again.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkImageView sourceDepthView, VkExtent2D sourceDepthExtent);

        void Shutdown();

        // Records the full pyramid build into `cmd`: mip 0 = 2x2 min/max reduction of the source
        // depth view passed to Init(), each mip 1..mipCount-1 = 2x2 min/max reduction of the mip
        // directly below it. The source depth image must already be in
        // VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        // or VK_IMAGE_LAYOUT_GENERAL by the time this runs (see the class-level integration
        // contract) -- this function does not transition it. Every dispatch is separated from the
        // next by an explicit VkImageMemoryBarrier2 scoped to the exact mip subresource just
        // written (BarrierAfterMipWrite), so a reduction dispatch can never read a mip level whose
        // producing dispatch has not finished writing it. The final barrier's destination
        // stage/access is VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT / VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        // ready for a later occlusion-culling compute pass recorded into the same command buffer.
        void Generate(VkCommandBuffer cmd);

        uint32_t GetMipLevelCount() const { return static_cast<uint32_t>(m_MipExtents.size()); }
        VkExtent2D GetMipExtent(uint32_t level) const { return m_MipExtents[level]; }
        VkImage GetImage() const { return m_HZBImage.Image(); }
        VkImageView GetMipView(uint32_t level) const { return m_MipViews[level]; }
        VkImageView GetFullView() const { return m_HZBImage.View(); }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only to vmaDestroyImage() in Shutdown().

        GpuImage m_HZBImage;
        std::vector<VkImageView> m_MipViews; // One per mip level, levelCount = 1 each.
        std::vector<VkExtent2D> m_MipExtents; // Index-aligned with m_MipViews.
        VkExtent2D m_SourceExtent{ 0, 0 };

        VkSampler m_DepthSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_InitSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_ReduceSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_InitSet = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> m_ReduceSets; // One per mip level in [1, mipCount); index 0 == level 1.

        VkPipelineLayout m_InitPipelineLayout = VK_NULL_HANDLE;
        VkPipelineLayout m_ReducePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_InitPipeline = VK_NULL_HANDLE;
        VkPipeline m_ReducePipeline = VK_NULL_HANDLE;

        // Records one VkImageMemoryBarrier2, scoped to `mipLevel`'s subresource, that makes the
        // compute shader's imageStore into that mip level visible to `dstStage`/`dstAccess`. Shared
        // by every step of Generate() -- the image never changes layout (stays GENERAL throughout),
        // so this is a pure execution/memory dependency, not a layout transition.
        void BarrierAfterMipWrite(VkCommandBuffer cmd, uint32_t mipLevel,
            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const;
    };

}
