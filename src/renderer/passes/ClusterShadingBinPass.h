#pragma once
// Phase 1b of the UE5.8 Nanite/Lumen parity roadmap: a GPU counting sort that buckets Visibility
// Buffer pixels by materialID, so renderer::ClusterResolvePass::RecordResolveBinned can dispatch
// one shading pass per material bin with every thread sharing the same material -- the actual
// warp coherence real Nanite-style shading bins provide. Three GPU stages, each a distinct
// shader/pipeline (see src/shaders/src/Renderer/ClusterShadingBin{Classify,PrefixSum,Scatter}.comp
// for the exact algorithm on each), all recorded together by RecordClassifyAndSort():
//
//   A. Classify -- one invocation per output pixel (same 8x8 grid as ClusterResolve.comp).
//      Resolves hw-vs-software VisBuffer arbitration (via visbuffer_arbitration.glsl, shared
//      verbatim with ClusterResolve.comp), and for a background (no-geometry) pixel writes the
//      final GBuffer/output-color defaults directly (this pass is that pixel's only writer, ever).
//      For a foreground pixel, looks up its cluster's materialID, atomically increments that
//      material's histogram bucket, and stores {materialSlot, packedVisID, winningDepth} into a
//      per-pixel scratch buffer for stage C to re-read.
//   B. Prefix-sum -- a single invocation (local_size_x = 1) turning the 32-bucket histogram into
//      each bucket's exclusive prefix-sum offset into ONE shared sorted-pixel-list buffer (sized
//      renderExtent.width * renderExtent.height, NOT one fixed-capacity list per bucket -- that
//      would cost hundreds of MB at typical resolutions for no benefit), plus each bucket's
//      VkDispatchIndirectCommand for the eventual per-bin shading dispatch. 32 buckets is small
//      enough that a genuine parallel scan algorithm (Hillis-Steele/Blelloch) would be pure
//      overhead versus one thread's trivial serial loop.
//   C. Scatter -- one invocation per output pixel again. Re-reads stage A's scratch entry (no
//      re-deriving visibility arbitration a second time); a foreground pixel atomically claims a
//      slot within its bucket's own contiguous range (via a per-bucket cursor stage B initialized
//      to that bucket's offset) and writes itself into the single shared sorted list.
//
// The actual shading dispatch (Stage D: 32 indirect dispatches of a leaner, non-debug-view-capable
// shader, `binIndex` a push constant) is NOT owned by this class -- see
// renderer::ClusterResolvePass::RecordResolveBinned, which borrows this class's buffers via the
// getters below. Keeping stage D inside ClusterResolvePass (rather than duplicating most of that
// class's already-existing cluster-metadata/mask-texture/material-table/WPO/output-image
// descriptor wiring here) avoids a second near-identical set of bindings for what is, in the end,
// still "the resolve pass" -- this class owns only the sorting machinery, which is a genuinely
// distinct GPU algorithm.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class ClusterShadingBinPass {
    public:
        ClusterShadingBinPass() = default;

        ClusterShadingBinPass(const ClusterShadingBinPass&) = delete;
        ClusterShadingBinPass& operator=(const ClusterShadingBinPass&) = delete;

        // Matches ClusterShadingBinClassify.comp/ClusterShadingBinScatter.comp's local_size_x/y.
        static constexpr uint32_t kClassifyScatterWorkgroupSize = 8;

        // Allocates every buffer this 3-stage pipeline needs (sized against `renderExtent` -- the
        // two viewport-sized buffers, m_PixelScratchBuffer/m_SortedPixelListBuffer, dominate; the
        // four MATERIAL_TABLE_SIZE-indexed buffers are a few hundred bytes total) and the 3
        // pipelines. `clusterMetadataBuffer` is renderer::ClusterOcclusionCullingPass's own
        // ClusterCullMetadataSSBO (borrowed, same buffer renderer::ClusterResolvePass already
        // reads); `hwClusterIDView`/`hwTriangleIDView`/`hwDepthView`/`swVisBufferAtomicView` are
        // this frame's VisBuffer/depth sources (same 4 views ClusterResolvePass::Init() takes, same
        // imageLayout contract -- see that class's own Init() doc comment); `outputColorView`/
        // `outputNormalView`/`outputDepthView`/`outputAlbedoView`/`outputRoughnessMetallicView` are
        // renderer::ClusterResolvePass's own 5 output images (borrowed) -- stage A (Classify)
        // writes background pixels directly into them, so this class needs write access to the
        // exact same images ClusterResolvePass owns, not copies.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, VkBuffer clusterMetadataBuffer,
            VkImageView hwClusterIDView, VkImageView hwTriangleIDView, VkImageView hwDepthView,
            VkImageView swVisBufferAtomicView, VkImageView outputColorView, VkImageView outputNormalView,
            VkImageView outputDepthView, VkImageView outputAlbedoView, VkImageView outputRoughnessMetallicView);

        void Shutdown();

        // Records all 3 stages in order, each separated by the exact barrier its successor needs
        // (see the .cpp for the full stage-by-stage synchronization reasoning). Ends with a
        // trailing barrier making every buffer renderer::ClusterResolvePass::RecordResolveBinned
        // needs (m_SortedPixelListBuffer/m_BinOffsetsBuffer/m_BinHistogramBuffer for its shading
        // reads, m_BinDispatchArgsBuffer for its vkCmdDispatchIndirect calls) visible to that call.
        // Caller must ensure the 4 borrowed VisBuffer/depth views are already in the layouts
        // Init()'s own doc comment describes, and that the 5 borrowed output images are already
        // GENERAL (same contract as ClusterResolvePass::RecordResolve's own caller-side step).
        void RecordClassifyAndSort(VkCommandBuffer cmd, VkExtent2D renderExtent);

        VkBuffer GetSortedPixelListBuffer() const { return m_SortedPixelListBuffer.Handle(); }
        VkBuffer GetBinOffsetsBuffer() const { return m_BinOffsetsBuffer.Handle(); }
        VkBuffer GetBinHistogramBuffer() const { return m_BinHistogramBuffer.Handle(); }
        VkBuffer GetBinDispatchArgsBuffer() const { return m_BinDispatchArgsBuffer.Handle(); }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        // --- Buffers ---
        GpuBuffer m_BinHistogramBuffer;   // uint[kMaxMaterials], atomically incremented by Classify, read by PrefixSum/ResolveBinned.
        GpuBuffer m_PixelScratchBuffer;   // PixelScratchEntry[width*height], written by Classify, read by Scatter.
        GpuBuffer m_BinOffsetsBuffer;     // uint[kMaxMaterials], written by PrefixSum, read by Scatter/ResolveBinned.
        GpuBuffer m_BinCursorBuffer;      // uint[kMaxMaterials], initialized by PrefixSum, atomically advanced by Scatter.
        GpuBuffer m_BinDispatchArgsBuffer; // VkDispatchIndirectCommand[kMaxMaterials], written by PrefixSum, read by vkCmdDispatchIndirect.
        GpuBuffer m_SortedPixelListBuffer; // SortedPixelEntry[width*height], written by Scatter, read by ResolveBinned.

        // --- Stage A: Classify ---
        VkDescriptorSetLayout m_ClassifySetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_ClassifySet = VK_NULL_HANDLE;
        VkPipelineLayout m_ClassifyPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ClassifyPipeline = VK_NULL_HANDLE;

        // --- Stage B: Prefix-sum ---
        VkDescriptorSetLayout m_PrefixSumSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_PrefixSumSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PrefixSumPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_PrefixSumPipeline = VK_NULL_HANDLE;

        // --- Stage C: Scatter ---
        VkDescriptorSetLayout m_ScatterSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_ScatterSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ScatterPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ScatterPipeline = VK_NULL_HANDLE;

        // Single pool backing all 3 sets above (one alloc, not one pool per stage).
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;

        // Nearest-filter sampler for Stage A's g_HWDepthTexture read (visbuffer_arbitration.glsl)
        // -- renderer::ClusterResolvePass owns an equivalent sampler privately with no getter, so
        // this class owns its own small, cheap copy rather than plumbing a new accessor through
        // that class for one sampler.
        VkSampler m_DepthSampler = VK_NULL_HANDLE;
    };

}
