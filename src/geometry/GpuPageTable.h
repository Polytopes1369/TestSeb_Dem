#pragma once
// CPU-side page table for the GPU geometry paging system.
//
// This is the authoritative bookkeeping structure that turns a "big virtual buffer" into a real,
// bounded GPU allocation: instead of relying on hardware sparse-residency binding
// (vkQueueBindSparse), the physical backing store is one ordinary VkBuffer treated as an array of
// fixed-size pages (a "structured SSBO pool", see renderer::GpuGeometryPagePool), and this class
// maps a cluster's logical virtual address (geometry::ClusterIndexEntry::virtualAddress -- a
// page-aligned byte offset from the .cache file, see ClusterFormat.h) to a physical page slot
// index inside that pool. The physical byte offset of a resident page is always
// `physicalPageIndex * kPageSizeBytes`, so once a logical address is mapped, translating it to a
// GPU buffer offset is a single multiply -- no traversal, no hashing on the GPU side.
//
// Why manual SSBO addressing instead of vkQueueBindSparse: the project does not currently enable
// VkPhysicalDeviceFeatures::sparseBinding / sparseResidencyBuffer, nor query a sparse-binding-
// capable queue family (VulkanContext only creates a combined graphics+compute queue). Sparse
// residency also requires page-granularity-aligned VkMemoryRequirements queried per-buffer and a
// dedicated vkQueueBindSparse submission path with its own semaphore choreography, none of which
// exists yet. Manual addressing achieves the same "virtual buffer larger than what's physically
// resident" property with a plain vkCmdCopyBuffer + a GPU-resident indirection table (see
// GpuGeometryPagePool), works on every Vulkan 1.3 implementation, and mirrors the page/
// virtualAddress convention already established by the on-disk cache format 1:1.
//
// This header has zero Vulkan dependency: it is pure CPU bookkeeping (a free-list allocator plus
// a logical->physical map), so it can be unit-tested without a GPU device (see
// tests/GpuPageTableTests.cpp) and is safe to call from any thread that owns its own instance.

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "geometry/ClusterFormat.h" // geometry::kPageSizeBytes

namespace geometry {

    // Sentinel returned by AllocatePage() when the request could not be satisfied (logical
    // address already resident, or every physical page slot is in use).
    constexpr uint32_t kInvalidPhysicalPage = 0xFFFFFFFFu;

    // Sentinel returned by GetPhysicalOffset() when the queried logical address is not resident.
    constexpr uint64_t kInvalidPhysicalOffset = 0xFFFFFFFFFFFFFFFFull;

    class GpuPageTable {
    public:
        // `physicalPageCapacity` is the fixed number of physical page slots backing this table
        // (i.e. the physical pool buffer is exactly `physicalPageCapacity * kPageSizeBytes`
        // bytes). The table never grows past this: once every slot is resident, AllocatePage()
        // fails until a FreePage() releases one.
        explicit GpuPageTable(uint32_t physicalPageCapacity);

        GpuPageTable(const GpuPageTable&) = delete;
        GpuPageTable& operator=(const GpuPageTable&) = delete;
        GpuPageTable(GpuPageTable&&) = default;
        GpuPageTable& operator=(GpuPageTable&&) = default;

        // Binds `logicalAddress` (must be a multiple of kPageSizeBytes) to a free physical page
        // slot and returns that slot's index. Returns kInvalidPhysicalPage without changing any
        // state if `logicalAddress` is already resident, or if no free physical slot is
        // available (every existing mapping's physical offset is left completely untouched in
        // both cases -- this is the invariant tests/GpuPageTableTests.cpp validates).
        uint32_t AllocatePage(uint64_t logicalAddress);

        // Releases the physical page slot bound to `logicalAddress`, making it eligible for reuse
        // by a future AllocatePage() call. Returns true if `logicalAddress` was resident (and has
        // now been freed); returns false, with no state change, if it was not resident. Every
        // other resident mapping's physical offset is left completely untouched.
        bool FreePage(uint64_t logicalAddress);

        bool IsResident(uint64_t logicalAddress) const;

        // Returns the physical page slot index bound to `logicalAddress`, or kInvalidPhysicalPage
        // if it is not resident.
        uint32_t GetPhysicalPageIndex(uint64_t logicalAddress) const;

        // Returns the physical byte offset (`physicalPageIndex * kPageSizeBytes`) into the
        // physical pool buffer for `logicalAddress`, or kInvalidPhysicalOffset if not resident.
        uint64_t GetPhysicalOffset(uint64_t logicalAddress) const;

        uint32_t GetResidentPageCount() const { return static_cast<uint32_t>(m_LogicalToPhysical.size()); }
        uint32_t GetCapacity() const { return m_PhysicalPageCapacity; }

        // Marks `logicalAddress` as the most-recently-used resident page for the LRU eviction
        // policy (see SelectLeastRecentlyUsedPages). AllocatePage() already calls this internally
        // the moment a page is bound, so callers only need to call it again on a later *reuse* of
        // an already-resident page -- typically once per frame, for every logical address that
        // frame's LOD cut still requires (see renderer::GpuGeometryPagePool::TouchPage/
        // TouchPages). No-op if `logicalAddress` is not resident.
        void TouchPage(uint64_t logicalAddress);

        // Returns up to `maxCount` resident logical addresses, ordered from least- to
        // most-recently-used (index 0, if present, is the single best eviction candidate).
        // Returns fewer than `maxCount` entries if fewer pages are resident, and an empty vector
        // if none are. Pure query -- does not free or otherwise change any state; the caller
        // decides whether/how to act on the result (see
        // renderer::GpuGeometryPagePool::EvictLeastRecentlyUsedPages, which calls FreePage/
        // UnbindPage for each candidate this returns).
        //
        // Backed by an O(1)-touch, O(1)-unlink intrusive doubly-linked list over physical page
        // slots (m_LRUNodes below), not per-page frame-number bookkeeping: because AllocatePage()
        // and TouchPage() always move a page to the most-recently-used end, the tail of that list
        // is always exactly the page that has gone the largest number of frames without being
        // (re)touched -- the same ordering an explicit "frames since last use" counter would
        // produce, at a fraction of the per-touch cost (no per-page counter comparison/scan is
        // ever needed to find an eviction candidate).
        std::vector<uint64_t> SelectLeastRecentlyUsedPages(uint32_t maxCount) const;

        // Converts a page-aligned logical byte address into the dense page index used to index
        // the GPU-resident indirection table (GpuGeometryPagePool::m_PageTableBuffer). Exposed as
        // a pure function so callers (and tests) can compute the same index independently.
        static uint32_t LogicalAddressToPageID(uint64_t logicalAddress) {
            return static_cast<uint32_t>(logicalAddress / kPageSizeBytes);
        }

    private:
        uint32_t m_PhysicalPageCapacity;

        // LIFO stack of physical page indices freed by FreePage(), reused before ever handing out
        // a never-before-used slot (m_NextUnusedPhysicalPage). Popping/pushing this stack never
        // touches m_LogicalToPhysical, which is what keeps every other resident page's physical
        // index stable across an alloc/free cycle.
        std::vector<uint32_t> m_FreePhysicalPages;

        // Physical page indices in [0, m_NextUnusedPhysicalPage) have been handed out at least
        // once; indices in [m_NextUnusedPhysicalPage, m_PhysicalPageCapacity) have never been
        // touched. Only grows.
        uint32_t m_NextUnusedPhysicalPage = 0;

        std::unordered_map<uint64_t, uint32_t> m_LogicalToPhysical;

        // Intrusive doubly-linked list node for the LRU eviction order, indexed by physical page
        // index (one node per physical slot -- see m_LRUNodes). `prev` points toward the
        // more-recently-used neighbor, `next` toward the less-recently-used neighbor;
        // kInvalidPhysicalPage marks a list end. A node's content is only meaningful while its
        // physical page index is resident (i.e. currently a value inside m_LogicalToPhysical);
        // it is stale until the slot is allocated again.
        struct LRUNode {
            uint64_t logicalAddress = 0;
            uint32_t prev = kInvalidPhysicalPage;
            uint32_t next = kInvalidPhysicalPage;
        };

        // Sized to m_PhysicalPageCapacity once, in the constructor, and never resized again, so a
        // physical page index is always a valid direct index into this vector -- no bounds check
        // is needed on the hot touch/link/unlink path.
        std::vector<LRUNode> m_LRUNodes;
        uint32_t m_LRUMostRecentPage = kInvalidPhysicalPage;  // List head; kInvalidPhysicalPage if empty.
        uint32_t m_LRULeastRecentPage = kInvalidPhysicalPage; // List tail; kInvalidPhysicalPage if empty.

        // Links `physicalPage` in as the new most-recently-used head. Assumes `physicalPage` is
        // NOT currently linked (either freshly handed out by AllocatePage, or already removed via
        // UnlinkFromLRUList) -- it does not unlink first, so calling it on an already-linked node
        // would corrupt the list.
        void LinkAsMostRecentlyUsed(uint32_t physicalPage);

        // Removes `physicalPage` from wherever it currently sits in the LRU list, patching its
        // neighbors' prev/next (and the head/tail pointers, if it was either) so the list stays
        // consistent. Must only be called on a currently-linked node (i.e. one reached via
        // m_LogicalToPhysical, or about to be freed by FreePage) -- calling it on a node that was
        // never linked would misinterpret its default prev/next as real list-end markers and
        // corrupt the head/tail pointers.
        void UnlinkFromLRUList(uint32_t physicalPage);
    };

}
