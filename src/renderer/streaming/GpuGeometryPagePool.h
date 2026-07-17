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
#ifndef NDEBUG
#include <unordered_map>
#endif

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
        // `transferQueueFamilyIndex`/`graphicsQueueFamilyIndex` (VulkanContext::
        // GetTransferQueueFamilyIndex/GetGraphicsQueueFamilyIndex) decide whether
        // UploadPageData()/ReleasePhysicalPoolOwnership()/AcquirePhysicalPoolOwnership() need to
        // record actual queue-family-ownership-transfer barriers -- when both indices are equal
        // (no dedicated transfer queue on this GPU, see VulkanContext's own fallback), those calls
        // record nothing, since same-queue-family data is already implicitly visible once its own
        // execution/memory barrier fires.
        void Init(VmaAllocator allocator, uint32_t maxLogicalPages, uint32_t maxPhysicalPages,
            uint32_t transferQueueFamilyIndex, uint32_t graphicsQueueFamilyIndex);

        void Shutdown();

        // Fills every entry of the GPU-resident page table with kUnmappedSentinel and inserts the
        // barrier making that fill visible to subsequent shader reads. Must be recorded once,
        // after Init(), before any shader dereferences the page table.
        void ClearPageTable(VkCommandBuffer cmd);

        // Allocates a free physical page slot for `logicalAddress` (must be a multiple of
        // geometry::kPageSizeBytes and < maxLogicalPages * kPageSizeBytes) and records the copy
        // of `dataSizeBytes` (<= kPageSizeBytes) bytes from `srcStagingBuffer[srcOffset]` into
        // that slot into `transferCmd` (VulkanContext::GetTransferCommandBuffer(), the dedicated
        // hardware copy queue's command buffer when one exists -- see Init()'s own comment).
        // Returns false, recording nothing, if `logicalAddress` is already resident or the
        // physical pool is full. Unlike the old single-call BindPage(), this does NOT update the
        // GPU-resident page table or make anything shader-visible -- the CPU-side allocation
        // (GetPhysicalPageIndex()/IsResident()) is valid immediately (same "valid the moment this
        // records" contract as before), but the caller MUST also call
        // ReleasePhysicalPoolOwnership()/AcquirePhysicalPoolOwnership() once, and FinalizeBoundPage()
        // once per page uploaded, before ANY shader reads this page -- see those methods' own
        // comments for the full 4-call sequence a streaming coordinator records each frame.
        bool UploadPageData(VkCommandBuffer transferCmd, uint64_t logicalAddress, VkBuffer srcStagingBuffer,
            VkDeviceSize srcOffset, VkDeviceSize dataSizeBytes);

        // Records the RELEASE half of a queue-family-ownership transfer for the WHOLE physical
        // pool buffer (not scoped to individual pages -- see class-level rationale: batching one
        // pair of barriers per frame instead of one pair per page trades a small amount of
        // granularity for a much simpler, easier-to-verify-correct synchronization story, and this
        // pool's per-frame upload volume is small enough that the extra breadth costs nothing
        // observable). No-op (records nothing) if Init() found no distinct transfer queue family.
        // Call once per frame on `transferCmd`, after every UploadPageData() call for the frame,
        // before that command buffer is submitted.
        void ReleasePhysicalPoolOwnership(VkCommandBuffer transferCmd);

        // The matching ACQUIRE half of ReleasePhysicalPoolOwnership() -- call once per frame on
        // the graphics command buffer, after the transfer queue's submission is guaranteed to have
        // completed (i.e. after that submission's signal semaphore is waited on by this
        // submission -- see main.cpp's per-frame sequence), before any FinalizeBoundPage() call.
        // No-op if Init() found no distinct transfer queue family (same-family data needs no
        // ownership transfer, only the usual execution/memory barrier FinalizeBoundPage() itself
        // records).
        void AcquirePhysicalPoolOwnership(VkCommandBuffer graphicsCmd);

        // Second half of the old BindPage(): writes the GPU-resident page table entry for
        // `logicalAddress` (already allocated by a prior, successful UploadPageData() call this
        // frame) and records the barrier making both the page table entry AND the physical pool
        // bytes UploadPageData() copied visible to a later shader read. Must be called on the
        // graphics command buffer, after AcquirePhysicalPoolOwnership() -- see that method's own
        // comment for why. No-op / undefined if `logicalAddress` was not just allocated by
        // UploadPageData() this frame (this method does not itself validate that, matching
        // BindPage()'s own prior trust-the-caller contract for its single-call form).
        void FinalizeBoundPage(VkCommandBuffer graphicsCmd, uint64_t logicalAddress);

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

        // Streaming entry point: uploads `logicalAddress` exactly as UploadPageData() does, but
        // first evicts up to `maxEvictions` least-recently-used resident pages (see
        // EvictLeastRecentlyUsedPages, recorded on `evictionCmd` -- the graphics command buffer,
        // since eviction only ever touches the page table, never physical pool bytes, so it needs
        // no transfer queue involvement) if the physical pool is already at capacity, so a
        // freshly-completed disk read is not rejected purely because the pool was full at the
        // moment the LRU policy has an eviction candidate available. Returns false if
        // `logicalAddress` is already resident, or if the pool is still full after evicting up to
        // `maxEvictions` pages -- the same failure semantics as UploadPageData(), so callers do
        // not need to distinguish "already resident" from "pool exhausted." Same remaining-call
        // obligations as UploadPageData() on success (Release/AcquirePhysicalPoolOwnership() once
        // per frame, FinalizeBoundPage() once per uploaded page).
        bool UploadPageDataEvictingIfFull(VkCommandBuffer transferCmd, VkCommandBuffer evictionCmd,
            uint64_t logicalAddress, VkBuffer srcStagingBuffer,
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

        // Debug-only "eviction churn" instrumentation (2026-07-16 "clusters missing / wrong LOD"
        // investigation; fixed 2026-07-17, see below). renderer::GpuGeometryPagePool::TouchPage/
        // TouchPages mark a still-in-use resident page as most-recently-used.
        // ClusterLODResidencyFallback.comp's RecordResidentTouch() reports every DRAW-decided node
        // found already resident (not just the non-resident ones it requests/substitutes), and
        // renderer::GeometryStreamingCoordinator::ProcessFeedbackAndDrainCompletions reads that
        // report back and calls TouchPages() with it every frame, before any eviction runs -- so a
        // page still being drawn every frame keeps its LRU recency fresh and is not evicted purely
        // for having gone unbound a while. BindPage() records the frame counter a page was (re)bound
        // at; UnbindPage() checks how many frames elapsed since and logs a warning if that gap is
        // suspiciously small -- with the touch wiring above in place, this should no longer fire
        // under normal (non-overflowing) camera movement; if it does, treat it as a regression, not
        // an expected condition. Call DebugAdvanceFrame() exactly once per frame
        // (renderer::GeometryStreamingCoordinator::ProcessFeedbackAndDrainCompletions does this) --
        // a no-op in Release, like every other Debug-only method in this codebase.
        void DebugAdvanceFrame() {
#ifndef NDEBUG
            ++m_DebugFrameCounter;
#endif
        }

    private:
        geometry::GpuPageTable m_PageTable{ 0 };
        GpuBuffer m_PhysicalPool;
        GpuBuffer m_PageTableBuffer;
        uint32_t m_MaxLogicalPages = 0;

        // See Init()'s own comment: when these differ, UploadPageData()'s copies land on a
        // different queue family than the one that later reads them, so
        // Release/AcquirePhysicalPoolOwnership() must record real ownership-transfer barriers.
        uint32_t m_TransferQueueFamilyIndex = 0;
        uint32_t m_GraphicsQueueFamilyIndex = 0;
        bool m_NeedsOwnershipTransfer = false;

#ifndef NDEBUG
        // Frames elapsed below this threshold between a page's bind and its eviction are logged as
        // likely thrashing rather than a normal, expected LRU turnover.
        static constexpr uint64_t kDebugChurnThresholdFrames = 5;
        uint64_t m_DebugFrameCounter = 0;
        std::unordered_map<uint64_t, uint64_t> m_DebugBoundAtFrame; // logicalAddress -> frame counter value at bind time.
#endif
    };

}
