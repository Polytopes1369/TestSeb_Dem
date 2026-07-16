#pragma once
// Phase 3 (UE5.8 parity roadmap): the physical page pool backing every Virtual Shadow Map in
// renderer::VirtualShadowMapPass (the sun's clipmap levels AND every active point light's 6 cube
// faces) -- one SHARED pool of fixed-size depth pages, mirroring renderer::GpuGeometryPagePool's
// own "structured pool + GPU-resident indirection table" design (see that class's own header
// comment for the full rationale), adapted from cluster geometry bytes to depth texels:
//
//   - m_PhysicalPoolImage : one VkImage, VK_IMAGE_TYPE_2D, `physicalPageCapacity` array LAYERS of
//                           kShadowPageTexels^2 D32_SFLOAT texels each -- a texture ARRAY, not a
//                           flat packed atlas like renderer::SurfaceCachePass's cards (those need
//                           a rectangle-packer because cards are variably sized; shadow pages are
//                           all exactly the same size, so a layer-per-page array is the natural,
//                           gap-free fit and needs no packing algorithm at all). Each layer gets
//                           its own dedicated VK_IMAGE_VIEW_TYPE_2D view (for rendering ONE page
//                           as a depth attachment) plus one shared VK_IMAGE_VIEW_TYPE_2D_ARRAY view
//                           spanning every layer (for a shader to sample any resident page via
//                           sampler2DArray).
//   - m_PageTableBuffer   : a small STORAGE_BUFFER of `totalVSMCount * kShadowPagesPerVSM`
//                           uint32_t entries -- one per possible (VSM index, local page) pair,
//                           each holding the physical layer index that page currently resolves
//                           to, or kUnmappedSentinel. See src/shaders/include/shadow_page_table.glsl
//                           for the GPU-side read side of this table.
//
// --- CPU-side allocation (free-list + O(1) LRU) ---
// The allocation/eviction ALGORITHM is a direct copy of geometry::GpuPageTable's (free-list LIFO +
// bump counter for physical slots, O(1)-touch/O(1)-unlink intrusive doubly-linked LRU list for
// eviction order -- see that class's own header comment for why an explicit recency list beats
// per-page frame counters). It is a FRESH, self-contained copy here rather than a reuse of
// geometry::GpuPageTable itself: that class's public API is keyed by a geometry-specific
// page-aligned uint64_t byte address (ClusterFormat.h's virtualAddress convention) and its own
// header explicitly frames it as bookkeeping for "a cluster's logical virtual address" -- shadow
// pages have no byte address at all, just a small dense (vsmIndex, localPageIndex) domain, so a
// plain flat std::vector<uint32_t> logical->physical map (not geometry::GpuPageTable's
// std::unordered_map, since this domain is small and dense, not sparse) is both simpler and a
// better fit. Matches this codebase's established "a fresh, self-contained copy for a different
// domain" convention (e.g. TraceHWRT's 3-4 independent copies across GI trace shaders, Phase 2's
// own ggx_brdf.glsl header comment on the same point) rather than awkwardly repurposing a
// geometry-specific class.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    // Fixed page geometry -- see this file's own header comment for why 128x128 (standard VSM
    // page size) and 16x16 pages/axis (2048x2048 virtual resolution per VSM, matching
    // ShadowMapPass's own pre-Phase-3 resolution at the finest sun clipmap level).
    constexpr uint32_t kShadowPageTexels = 128u;
    constexpr uint32_t kShadowPagesPerAxis = 16u;
    constexpr uint32_t kShadowPagesPerVSM = kShadowPagesPerAxis * kShadowPagesPerAxis; // 256.
    constexpr uint32_t kInvalidShadowPhysicalPage = 0xFFFFFFFFu;

    class VirtualShadowMapPool {
    public:
        VirtualShadowMapPool() = default;

        VirtualShadowMapPool(const VirtualShadowMapPool&) = delete;
        VirtualShadowMapPool& operator=(const VirtualShadowMapPool&) = delete;

        // `totalVSMCount` sizes the GPU page table (`totalVSMCount * kShadowPagesPerVSM` entries);
        // `physicalPageCapacity` is the shared physical pool's layer count. No runtime validation
        // of the per-page virtual-viewport technique's dimensions (kShadowPagesPerAxis *
        // kShadowPageTexels = 2048) against VkPhysicalDeviceLimits::maxViewportDimensions /
        // viewportBoundsRange is needed: the Vulkan 1.0+ spec's own REQUIRED minimum for
        // maxViewportDimensions is 4096 (and viewportBoundsRange at least [-8192, 8191]) on every
        // conformant implementation, so a virtual resolution of 2048 (max negative viewport offset
        // -1920) is guaranteed safe on any Vulkan 1.3 device without a capability query -- see
        // renderer::VirtualShadowMapPass::RenderPage's own comment.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            uint32_t totalVSMCount, uint32_t physicalPageCapacity);

        void Shutdown();

        // Fills the GPU-resident page table with kUnmappedSentinel. Must be recorded once, after
        // Init(), before any shader dereferences the table -- mirrors
        // renderer::GpuGeometryPagePool::ClearPageTable exactly.
        void ClearPageTable(VkCommandBuffer cmd);

        // Binds `logicalPageID` (== vsmIndex * kShadowPagesPerVSM + localPageIndex, see
        // shadow_page_table.glsl) to a physical layer, evicting up to `maxEvictions`
        // least-recently-used resident pages first if the pool is full, and records the GPU page
        // table entry update + barrier into `cmd`. Returns the physical layer index (idempotent:
        // calling this again on an already-resident logicalPageID returns its existing layer
        // without touching any state) so the caller can immediately render into
        // GetPhysicalLayerView() for that layer -- unlike renderer::GpuGeometryPagePool::BindPage
        // (which copies pre-existing data), a shadow page's CONTENT does not exist yet at bind
        // time; the caller renders it via VirtualShadowMapPass::RenderPage AFTER this call returns,
        // then this class's page-table entry (already written here) makes it visible to a shader
        // that reads it starting next frame (the established one-frame-lag contract, see
        // VirtualShadowMapPass's own class comment). Returns kInvalidShadowPhysicalPage only if
        // eviction still could not free a slot (should not happen once `maxEvictions` >= 1 and at
        // least one page is resident; genuinely impossible on a cold/empty pool).
        uint32_t AllocatePage(VkCommandBuffer cmd, uint32_t logicalPageID, uint32_t maxEvictions = 4);

        // Marks `logicalPageID` as most-recently-used without changing residency. Called once per
        // frame for every page a consuming shader actually sampled this frame (mirrors
        // renderer::GpuGeometryPagePool::TouchPage) so a page still in active use is never mistaken
        // for an eviction candidate just because nothing re-requested it.
        void TouchPage(uint32_t logicalPageID);

        bool IsResident(uint32_t logicalPageID) const;
        uint32_t GetPhysicalLayer(uint32_t logicalPageID) const;
        uint32_t GetResidentPageCount() const { return m_ResidentCount; }
        uint32_t GetPhysicalCapacity() const { return m_PhysicalPageCapacity; }

        VkImage GetPhysicalPoolImage() const { return m_PhysicalPoolImage; }
        // Sampling view (sampler2DArray) spanning every physical layer.
        VkImageView GetPhysicalPoolArrayView() const { return m_PhysicalPoolArrayView; }
        // Rendering view for one specific physical layer (depth attachment target).
        VkImageView GetPhysicalLayerView(uint32_t physicalLayer) const { return m_PhysicalLayerViews[physicalLayer]; }
        VkSampler GetSampler() const { return m_Sampler; }
        VkBuffer GetPageTableBuffer() const { return m_PageTableBuffer.Handle(); }

        static constexpr uint32_t kUnmappedSentinel = 0xFFFFFFFFu;

    private:
        uint32_t AllocatePhysicalSlot();
        void EvictOneLeastRecentlyUsed(VkCommandBuffer cmd);
        void LinkAsMostRecentlyUsed(uint32_t physicalLayer);
        void UnlinkFromLRUList(uint32_t physicalLayer);
        void WritePageTableEntry(VkCommandBuffer cmd, uint32_t logicalPageID, uint32_t physicalLayer);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        uint32_t m_TotalVSMCount = 0;
        uint32_t m_PhysicalPageCapacity = 0;
        uint32_t m_ResidentCount = 0;

        VkImage m_PhysicalPoolImage = VK_NULL_HANDLE;
        VmaAllocation m_PhysicalPoolAllocation = VK_NULL_HANDLE;
        VkImageView m_PhysicalPoolArrayView = VK_NULL_HANDLE;
        std::vector<VkImageView> m_PhysicalLayerViews;
        VkSampler m_Sampler = VK_NULL_HANDLE;

        GpuBuffer m_PageTableBuffer; // uint32_t[totalVSMCount * kShadowPagesPerVSM], GPU_ONLY.

        // Intrusive doubly-linked LRU list node, indexed by physical layer -- see
        // geometry::GpuPageTable::LRUNode's own comment for the identical design this mirrors.
        struct LRUNode {
            uint32_t logicalPageID = 0;
            uint32_t prev = kInvalidShadowPhysicalPage;
            uint32_t next = kInvalidShadowPhysicalPage;
        };

        std::vector<uint32_t> m_FreePhysicalSlots; // LIFO stack of freed physical layers.
        uint32_t m_NextUnusedPhysicalSlot = 0;      // Bump counter for never-yet-used layers.
        // Dense flat map (not unordered_map: the logicalPageID domain is small -- totalVSMCount *
        // 256 -- and every ID in [0, size) is a valid index, so a plain array beats a hash map).
        std::vector<uint32_t> m_LogicalToPhysical;
        std::vector<LRUNode> m_LRUNodes; // Sized m_PhysicalPageCapacity.
        uint32_t m_LRUMostRecentPage = kInvalidShadowPhysicalPage;
        uint32_t m_LRULeastRecentPage = kInvalidShadowPhysicalPage;
    };

}
