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
        VkImage GetImage() const { return m_Image; }

        // Single-mip-level view (levelCount == 1) of `level`, matching what HZBReduce.comp binds
        // internally -- exposed so an external debug view/readback could target one exact level.
        VkImageView GetMipView(uint32_t level) const { return m_MipViews[level]; }

        // Full-pyramid view (all mips, one array layer) for a future occlusion-culling shader to
        // sample with an explicit LOD (textureLod), e.g. to pick the mip level whose texel
        // footprint best matches an occludee's screen-space bounding box.
        VkImageView GetFullView() const { return m_FullView; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only to vmaDestroyImage() in Shutdown().

        VkImage m_Image = VK_NULL_HANDLE;
        VmaAllocation m_Allocation = VK_NULL_HANDLE;
        std::vector<VkImageView> m_MipViews; // One per mip level, levelCount = 1 each.
        std::vector<VkExtent2D> m_MipExtents; // Index-aligned with m_MipViews.
        VkImageView m_FullView = VK_NULL_HANDLE; // All mips, for a future sampled read.
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
