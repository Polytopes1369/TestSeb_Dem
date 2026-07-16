#pragma once
// GPU-resident geometry paging pool: the Vulkan-facing half of the paging system whose CPU-side
// bookkeeping lives in geometry::GpuPageTable (src/geometry/GpuPageTable.h).
//
// This owns two GPU buffers:
//   - m_PhysicalPool     : a big STORAGE_BUFFER treated as `maxPhysicalPages` fixed-size slots of
//                          geometry::kPageSizeBytes bytes each. This is the "structured SSBO
//                          pool" backing the virtual buffer -- physically resident cluster
//                          geometry (see geometry::ClusterData) lives here.
//   - m_PageTableBuffer  : a small STORAGE_BUFFER of `maxLogicalPages` uint32_t entries, one per
//                          possible logical page ID (geometry::GpuPageTable::LogicalAddressToPageID),
//                          each holding either the physical page slot index that logical page
//                          currently resolves to, or kUnmappedSentinel. This is the GPU-readable
//                          mirror of the CPU-side page table: a shader translates a logical
//                          cluster address to a physical byte offset with exactly one indexed
//                          load (`pageTable[logicalPageID] * kPageSizeBytes`), no CPU round trip.
//
// BindPage()/UnbindPage() record commands into a caller-supplied, already-recording
// VkCommandBuffer (matching this codebase's existing pattern of the caller owning single-time-
// submit lifetime -- see VulkanContext::UploadEntityData) rather than submitting their own; this
// keeps page (un)binding composable with a future streaming system that batches many page
// (un)binds into one command buffer per frame instead of one blocking submit per page.
//
// Deliberately out of scope here (left for the system that will drive this pool once geometry
// streaming is wired into the renderer): allocating/filling the host-visible staging buffer
// BindPage() copies from (naturally the job of a streaming component sitting on top of
// geometry::AsyncFileStreamer), and any shader-side consumption of the page table / physical pool
// (no mesh shader pipeline exists yet in this codebase -- see src/shaders/).
//
// LRU eviction: geometry::GpuPageTable tracks per-page recency (TouchPage/
// SelectLeastRecentlyUsedPages -- see GpuPageTable.h for why an O(1) recency list is used instead
// of per-page frame counters). This class exposes that policy at the GPU-paging level:
//   - TouchPage()/TouchPages() mark resident pages as still in use, without touching the GPU page
//     table at all -- the intended caller is whatever CPU-side code builds each frame's LOD-cut
//     "required cluster" list (the g_RequiredClusters/g_RequiredLogicalPageIDs SSBOs consumed by
//     src/shaders/src/Culling/ClusterResidencyCheck.comp), since that code already knows, on the
//     CPU, every logical address the GPU will check residency for this frame -- resident hits
//     included, which is exactly the signal a page-table-only feedback channel (misses only)
//     cannot provide.
//   - EvictLeastRecentlyUsedPages() frees the least-recently-touched resident pages, recording
//     the matching page-table-entry reset + barrier (via UnbindPage) for each.
//   - BindPageEvictingIfFull() is the streaming entry point: it evicts just enough LRU pages to
//     make room, then binds, in one call -- the physical slot an eviction frees is pushed onto
//     geometry::GpuPageTable's free list (see FreePage) and is immediately eligible for reuse, so
//     the very BindPage() this function performs afterward can, and typically does, land the new
//     page's data in that exact just-freed slot for the next completed geometry::AsyncFileStreamer
//     read (see geometry::StreamingRequestQueue for how a cluster ID reaches this call).

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include "geometry/GpuPageTable.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class GpuGeometryPagePool {
    public:
        GpuGeometryPagePool() = default;

        GpuGeometryPagePool(const GpuGeometryPagePool&) = delete;
        GpuGeometryPagePool& operator=(const GpuGeometryPagePool&) = delete;

        // Allocates the physical pool (`maxPhysicalPages * geometry::kPageSizeBytes` bytes) and
        // the GPU-resident page table (`maxLogicalPages * sizeof(uint32_t)` bytes). Does not
        // record any GPU commands -- call ClearPageTable() once afterwards (with a command buffer
        // the caller submits) before the pool is first read by a shader, so every entry starts
        // out as kUnmappedSentinel rather than undefined device memory content.
        void Init(VmaAllocator allocator, uint32_t maxLogicalPages, uint32_t maxPhysicalPages);

        void Shutdown();

        // Fills every entry of the GPU-resident page table with kUnmappedSentinel and inserts the
        // barrier making that fill visible to subsequent shader reads. Must be recorded once,
        // after Init(), before any shader dereferences the page table.
        void ClearPageTable(VkCommandBuffer cmd);

        // Allocates a free physical page slot for `logicalAddress` (must be a multiple of
        // geometry::kPageSizeBytes and < maxLogicalPages * kPageSizeBytes), copies
        // `dataSizeBytes` (<= kPageSizeBytes) bytes from `srcStagingBuffer[srcOffset]` into that
        // slot, and updates the GPU-resident page table entry to point at it. Records the copy,
        // the table update, and the VkMemoryBarrier2 synchronization needed before either
        // resource is read by a later shader stage into `cmd`. Returns false, recording nothing,
        // if `logicalAddress` is already resident or the physical pool is full.
        bool BindPage(VkCommandBuffer cmd, uint64_t logicalAddress, VkBuffer srcStagingBuffer,
            VkDeviceSize srcOffset, VkDeviceSize dataSizeBytes);

        // Frees the physical page slot bound to `logicalAddress` (making it eligible for reuse by
        // a future BindPage() call) and writes kUnmappedSentinel into its GPU-resident page table
        // entry, with the accompanying barrier. The freed slot's physical bytes are left
        // untouched -- once the table no longer indexes them they are simply unreachable from a
        // shader, so there is no need to zero or otherwise scrub them. Returns false, recording
        // nothing, if `logicalAddress` was not resident.
        bool UnbindPage(VkCommandBuffer cmd, uint64_t logicalAddress);

        // Marks `logicalAddress` as most-recently-used for the LRU eviction policy, without
        // recording any GPU commands or otherwise changing residency/physical offset. No-op if
        // `logicalAddress` is not resident. See geometry::GpuPageTable::TouchPage.
        void TouchPage(uint64_t logicalAddress) { m_PageTable.TouchPage(logicalAddress); }

        // Batched TouchPage(): marks every resident address in `logicalAddresses` as
        // most-recently-used. Intended to be called once per frame with that frame's full
        // required-cluster list (see the class-level comment above) so pages still in active use
        // are never mistaken for eviction candidates just because they were never re-bound.
        // Addresses that are not resident (a page-table miss the culling shader will report
        // through renderer::FeedbackBuffer instead) are silently skipped.
        void TouchPages(const std::vector<uint64_t>& logicalAddresses) {
            for (uint64_t logicalAddress : logicalAddresses) {
                m_PageTable.TouchPage(logicalAddress);
            }
        }

        // Frees up to `maxPagesToEvict` resident physical page slots, always choosing the
        // least-recently-used ones first (geometry::GpuPageTable::SelectLeastRecentlyUsedPages),
        // and records the matching UnbindPage() (page-table entry reset to kUnmappedSentinel +
        // barrier) for each into `cmd`. Returns the logical addresses actually evicted, ordered
        // from least- to most-recently-used; fewer than `maxPagesToEvict` if fewer pages were
        // resident (an empty vector if none were). Every evicted page's physical slot is pushed
        // onto geometry::GpuPageTable's free list by UnbindPage() as part of this call, so it is
        // immediately eligible for the very next BindPage() -- including one recorded later in
        // this same command buffer.
        std::vector<uint64_t> EvictLeastRecentlyUsedPages(VkCommandBuffer cmd, uint32_t maxPagesToEvict);

        // Streaming entry point: binds `logicalAddress` exactly as BindPage() does, but first
        // evicts up to `maxEvictions` least-recently-used resident pages (see
        // EvictLeastRecentlyUsedPages) if the physical pool is already at capacity, so a
        // freshly-completed disk read is not rejected purely because the pool was full at the
        // moment the LRU policy has an eviction candidate available. Returns false if
        // `logicalAddress` is already resident, or if the pool is still full after evicting up to
        // `maxEvictions` pages -- the same failure semantics as BindPage(), so callers do not need
        // to distinguish "already resident" from "pool exhausted."
        bool BindPageEvictingIfFull(VkCommandBuffer cmd, uint64_t logicalAddress, VkBuffer srcStagingBuffer,
            VkDeviceSize srcOffset, VkDeviceSize dataSizeBytes, uint32_t maxEvictions = 1);

        bool IsResident(uint64_t logicalAddress) const { return m_PageTable.IsResident(logicalAddress); }
        uint32_t GetResidentPageCount() const { return m_PageTable.GetResidentPageCount(); }
        uint32_t GetPhysicalCapacity() const { return m_PageTable.GetCapacity(); }

        // Physical page slot index currently bound to `logicalAddress` (geometry::kInvalidPhysicalPage
        // if not resident). Needed by renderer::ClusterRenderPipeline to derive each cluster's
        // firstIndex/vertexOffset (physicalPageIndex * kMaxClusterIndices/kMaxClusterVertices, the
        // layout contract renderer::GeometryDecompressionPass's pools are built on) right after
        // recording that cluster's BindPage() -- the mapping is CPU-side bookkeeping, valid the
        // moment BindPage() records, not only after the GPU executes it.
        uint32_t GetPhysicalPageIndex(uint64_t logicalAddress) const { return m_PageTable.GetPhysicalPageIndex(logicalAddress); }

        VkBuffer GetPhysicalPoolBuffer() const { return m_PhysicalPool.Handle(); }
        VkBuffer GetPageTableBuffer() const { return m_PageTableBuffer.Handle(); }

        static constexpr uint32_t kUnmappedSentinel = 0xFFFFFFFFu;

    private:
        geometry::GpuPageTable m_PageTable{ 0 };
        GpuBuffer m_PhysicalPool;
        GpuBuffer m_PageTableBuffer;
        uint32_t m_MaxLogicalPages = 0;
    };

}
