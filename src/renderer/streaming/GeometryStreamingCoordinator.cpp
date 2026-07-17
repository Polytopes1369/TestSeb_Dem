#include "renderer/streaming/GeometryStreamingCoordinator.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "core/Logger.h"
#include "renderer/streaming/FeedbackBuffer.h"
#include "renderer/passes/GeometryDecompressionPass.h"
#include "renderer/streaming/GpuGeometryPagePool.h"

namespace renderer {

    void GeometryStreamingCoordinator::Init(VkDevice device, VmaAllocator allocator,
        const std::filesystem::path& cacheFilePath,
        const std::vector<geometry::ClusterIndexEntry>& indexEntries,
        const std::vector<geometry::DAGNodeEntry>& dagEntries) {
        Shutdown();

        m_Device = device;
        m_IndexEntries = indexEntries;

        // dagEntries is index-aligned with indexEntries (same clusterID == array position
        // convention, see geometry::VirtualGeometryCacheTest.cpp's documented invariant).
        m_ClusterDAGLevel.resize(dagEntries.size());
        for (size_t i = 0; i < dagEntries.size(); ++i) {
            m_ClusterDAGLevel[i] = dagEntries[i].level;
        }

        if (!m_Streamer.Open(cacheFilePath)) {
            LOG_ERROR(std::format("[GeometryStreamingCoordinator] Failed to open '{}' for async streaming.",
                cacheFilePath.string()));
        }

        for (uint32_t i = 0; i < kMaxInFlightReads; ++i) {
            m_RawBuffers[i] = geometry::AsyncFileStreamer::AllocateAlignedBuffer(geometry::kPageSizeBytes);
            m_SlotBusy[i] = false;
        }

        // CPU_ONLY + mapped: BindPage's own vkCmdCopyBuffer reads from this as a plain
        // TRANSFER_SRC staging buffer, exactly like every other staging buffer in this codebase.
        m_StagingRing.Create(allocator,
            static_cast<VkDeviceSize>(geometry::kPageSizeBytes) * kMaxPagesBoundPerFrame,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);

        LOG_INFO(std::format("[GeometryStreamingCoordinator] Initialized: {} clusters trackable, {} max in-flight reads.",
            m_IndexEntries.size(), kMaxInFlightReads));
    }

    void GeometryStreamingCoordinator::Shutdown() {
        // Close() blocks until every in-flight read's callback has fired (see AsyncFileStreamer.h)
        // -- safe to do unconditionally here, this class is never torn down mid-frame.
        m_Streamer.Close();

        for (uint32_t i = 0; i < kMaxInFlightReads; ++i) {
            if (m_RawBuffers[i] != nullptr) {
                geometry::AsyncFileStreamer::FreeAlignedBuffer(m_RawBuffers[i]);
            }
            m_RawBuffers[i] = nullptr;
            m_SlotBusy[i] = false;
        }

        m_StagingRing.Destroy();

        m_CompletedReads.clear();
        m_IndexEntries.clear();
        m_ClusterDAGLevel.clear();
        m_TotalBytesCompleted.store(0, std::memory_order_relaxed);
        m_Device = VK_NULL_HANDLE;
    }

    uint32_t GeometryStreamingCoordinator::FindFreeSlot() const {
        for (uint32_t i = 0; i < kMaxInFlightReads; ++i) {
            if (!m_SlotBusy[i]) {
                return i;
            }
        }
        return kInvalidSlot;
    }

    void GeometryStreamingCoordinator::ProcessFeedbackAndDrainCompletions(VkCommandBuffer cmd, VkCommandBuffer transferCmd,
        FeedbackBuffer& feedbackBuffer, GpuGeometryPagePool& pagePool, GeometryDecompressionPass& decompressionPass) {
        if (!m_Streamer.IsOpen()) {
            return; // Open() failed at Init -- streaming is simply inert (already logged).
        }

        pagePool.DebugAdvanceFrame(); // No-op in Release -- see GpuGeometryPagePool's own doc comment.

        // --- 1. Fold LAST frame's residency misses into the dedup request queue. Safe to read
        // here: the caller's own per-frame contract (main.cpp's single-frame-in-flight loop
        // already waited on the previous frame's fence before RecordFrame() runs again) guarantees
        // last frame's FeedbackBuffer::RecordReadback() copy has completed and is host-visible. ---
        uint32_t overflowedRequestCount = 0;
        std::vector<uint32_t> missedClusterIDs = feedbackBuffer.ReadRequestedClusterIDs(&overflowedRequestCount);
        if (overflowedRequestCount > 0) {
            // The GPU-side atomic counter went past capacity: some residency misses this frame
            // were never even reported, so they cannot be queued for a disk read until (if) a
            // future frame reports them again -- a persistent hole/wrong-LOD source distinct from
            // both the DAG-cut-gap bug and the page-eviction-churn one (see GpuGeometryPagePool).
            LOG_WARNING(std::format("[GeometryStreamingCoordinator] FeedbackBuffer saturated: {} residency-miss report(s) dropped this frame (capacity={}). "
                "Some non-resident clusters were not requested and will keep missing until pressure drops.",
                overflowedRequestCount, feedbackBuffer.GetCapacity()));
        }
        std::vector<float> missedPriorities;
        missedPriorities.reserve(missedClusterIDs.size());
        for (uint32_t clusterID : missedClusterIDs) {
            // Coarser cluster (higher DAG level) first -- see Init()'s own comment.
            missedPriorities.push_back(clusterID < m_ClusterDAGLevel.size() ? float(m_ClusterDAGLevel[clusterID]) : 0.0f);
        }
        m_RequestQueue.SubmitFrameRequests(missedClusterIDs, missedPriorities);

        // --- 1b. Fold LAST frame's resident-touch reports into the LRU BEFORE any eviction
        // decision below (step 3's BindPageEvictingIfFull) runs -- a page reported here was drawn
        // every frame up to and including last frame, so it must never be treated as a stale
        // eviction candidate purely because it was never re-bound. See GpuGeometryPagePool's
        // TouchPages doc comment and ClusterLODResidencyFallback.comp's RecordResidentTouch call. ---
        uint32_t overflowedTouchCount = 0;
        std::vector<uint32_t> touchedClusterIDs = feedbackBuffer.ReadTouchedClusterIDs(&overflowedTouchCount);
        if (overflowedTouchCount > 0) {
            LOG_WARNING(std::format("[GeometryStreamingCoordinator] FeedbackBuffer touch channel saturated: {} resident-touch report(s) dropped this frame (capacity={}). "
                "Some still-in-use pages were not marked most-recently-used this frame and could be mistaken for eviction candidates.",
                overflowedTouchCount, feedbackBuffer.GetCapacity()));
        }
        if (!touchedClusterIDs.empty()) {
            std::vector<uint64_t> touchedLogicalAddresses;
            touchedLogicalAddresses.reserve(touchedClusterIDs.size());
            for (uint32_t clusterID : touchedClusterIDs) {
                if (clusterID < m_IndexEntries.size()) {
                    touchedLogicalAddresses.push_back(m_IndexEntries[clusterID].virtualAddress);
                }
            }
            pagePool.TouchPages(touchedLogicalAddresses);
        }

        // --- 2. Issue up to kMaxNewReadsPerFrame new async reads, bounded by both the per-frame
        // budget and the number of free I/O slots. ---
        for (uint32_t issued = 0; issued < kMaxNewReadsPerFrame; ++issued) {
            uint32_t slot = FindFreeSlot();
            if (slot == kInvalidSlot) {
                break; // Every in-flight slot is busy -- try again next frame.
            }

            uint32_t clusterID = 0;
            if (!m_RequestQueue.PopNextRequest(clusterID)) {
                break; // Nothing left to request this frame.
            }
            if (clusterID >= m_IndexEntries.size()) {
                m_RequestQueue.MarkRequestCompleted(clusterID); // Defensive: out-of-range report, drop it.
                continue;
            }

            m_SlotBusy[slot] = true;
            void* destination = m_RawBuffers[slot];
            uint64_t fileOffset = m_IndexEntries[clusterID].virtualAddress; // Already an absolute, page-aligned file offset.

            bool submitted = m_Streamer.SubmitRead(fileOffset, destination, geometry::kPageSizeBytes,
                [this, clusterID, slot](bool success, uint32_t bytesTransferred) {
                    std::lock_guard<std::mutex> lock(m_CompletedMutex);
                    m_CompletedReads.push_back(CompletedRead{ clusterID, slot, bytesTransferred, success });
                });

            if (!submitted) {
                m_SlotBusy[slot] = false;
                m_RequestQueue.MarkRequestCompleted(clusterID); // Allow a future miss report to retry it.
            }
        }

        // --- 3. Drain up to kMaxPagesBoundPerFrame completed reads into the physical page pool. ---
        std::vector<CompletedRead> drained;
        {
            std::lock_guard<std::mutex> lock(m_CompletedMutex);
            uint32_t drainCount = std::min<uint32_t>(kMaxPagesBoundPerFrame, static_cast<uint32_t>(m_CompletedReads.size()));
            drained.assign(m_CompletedReads.begin(), m_CompletedReads.begin() + drainCount);
            m_CompletedReads.erase(m_CompletedReads.begin(), m_CompletedReads.begin() + drainCount);
        }

        uint32_t ringSlot = 0;
        // Uploaded on `transferCmd` this pass; each needs AcquirePhysicalPoolOwnership() +
        // FinalizeBoundPage() (+ decompression) recorded on `cmd` afterward -- see the loop below.
        // Copies of the owning ClusterIndexEntry (not just the logical address) since
        // DecompressPage() needs its bounds too, and re-deriving an entry from a logical address
        // would need the reverse of the clusterID -> virtualAddress mapping this class already
        // has forward (ClusterIndexEntry::clusterID is NOT recoverable from
        // GpuPageTable::LogicalAddressToPageID(virtualAddress), which yields a page-table index,
        // not a clusterID).
        std::vector<geometry::ClusterIndexEntry> uploadedEntries;
        uploadedEntries.reserve(drained.size());
        for (const CompletedRead& completed : drained) {
            m_SlotBusy[completed.slotIndex] = false; // Free the I/O slot for reuse regardless of outcome.
            m_RequestQueue.MarkRequestCompleted(completed.clusterID);

            if (!completed.success || completed.bytesTransferred < geometry::kPageSizeBytes) {
                LOG_WARNING(std::format("[GeometryStreamingCoordinator] Read failed or short for clusterID {}: success={}, bytes={}", completed.clusterID, completed.success, completed.bytesTransferred));
                continue; // Failed/short read: not bound this frame -- a future miss report retries it.
            }
            m_TotalBytesCompleted.fetch_add(completed.bytesTransferred, std::memory_order_relaxed);

            const geometry::ClusterIndexEntry& entry = m_IndexEntries[completed.clusterID];
            VkDeviceSize ringOffset = static_cast<VkDeviceSize>(ringSlot) * geometry::kPageSizeBytes;
            std::memcpy(static_cast<char*>(m_StagingRing.MappedData()) + ringOffset,
                m_RawBuffers[completed.slotIndex], geometry::kPageSizeBytes);
            ++ringSlot;

            bool uploaded = pagePool.UploadPageDataEvictingIfFull(transferCmd, cmd, entry.virtualAddress,
                m_StagingRing.Handle(), ringOffset, geometry::kPageSizeBytes);
            if (uploaded) {
                uploadedEntries.push_back(entry);
            } else {
                LOG_WARNING(std::format("[GeometryStreamingCoordinator] UploadPageDataEvictingIfFull failed for clusterID {} (virtualAddress {:#x}) -- "
                    "pool still full after eviction budget, or already resident. This cluster's completed read is discarded; it will only reappear "
                    "if a future frame reports it missing again.", completed.clusterID, entry.virtualAddress));
            }
        }

        // Finalize this frame's uploads: one ownership-transfer barrier pair for the whole batch
        // (see GpuGeometryPagePool::Release/AcquirePhysicalPoolOwnership's own comment on why one
        // pair per frame instead of one per page), then each page's table entry + decompression.
        if (!uploadedEntries.empty()) {
            pagePool.ReleasePhysicalPoolOwnership(transferCmd);
            pagePool.AcquirePhysicalPoolOwnership(cmd);
            for (const geometry::ClusterIndexEntry& entry : uploadedEntries) {
                pagePool.FinalizeBoundPage(cmd, entry.virtualAddress);
                uint32_t physicalPage = pagePool.GetPhysicalPageIndex(entry.virtualAddress);
                decompressionPass.DecompressPage(cmd, physicalPage,
                    maths::vec3{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] },
                    maths::vec3{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] });
            }
        }

        // Per-frame streaming summary -- only logged when there was actual activity to report, so
        // a steady-state frame with nothing missing/in-flight/bound does not spam demo_log.txt.
        if (!missedClusterIDs.empty() || !drained.empty()) {
            LOG_INFO(std::format("[GeometryStreamingCoordinator] frame summary: missed={}, pendingInQueue={}, inFlightReads={}, drainedThisFrame={}, residentPages={}/{}",
                missedClusterIDs.size(), m_RequestQueue.PendingCount(), m_Streamer.PendingRequestCount(),
                drained.size(), pagePool.GetResidentPageCount(), pagePool.GetPhysicalCapacity()));
        }
    }

}
