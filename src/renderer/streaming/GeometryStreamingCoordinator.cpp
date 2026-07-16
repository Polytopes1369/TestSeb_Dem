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
        const std::vector<geometry::ClusterIndexEntry>& indexEntries) {
        Shutdown();

        m_Device = device;
        m_IndexEntries = indexEntries;

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

    void GeometryStreamingCoordinator::ProcessFeedbackAndDrainCompletions(VkCommandBuffer cmd, FeedbackBuffer& feedbackBuffer,
        GpuGeometryPagePool& pagePool, GeometryDecompressionPass& decompressionPass) {
        if (!m_Streamer.IsOpen()) {
            return; // Open() failed at Init -- streaming is simply inert (already logged).
        }

        // --- 1. Fold LAST frame's residency misses into the dedup request queue. Safe to read
        // here: the caller's own per-frame contract (main.cpp's single-frame-in-flight loop
        // already waited on the previous frame's fence before RecordFrame() runs again) guarantees
        // last frame's FeedbackBuffer::RecordReadback() copy has completed and is host-visible. ---
        std::vector<uint32_t> missedClusterIDs = feedbackBuffer.ReadRequestedClusterIDs();
        m_RequestQueue.SubmitFrameRequests(missedClusterIDs);

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

            bool bound = pagePool.BindPageEvictingIfFull(cmd, entry.virtualAddress,
                m_StagingRing.Handle(), ringOffset, geometry::kPageSizeBytes);
            if (bound) {
                uint32_t physicalPage = pagePool.GetPhysicalPageIndex(entry.virtualAddress);
                decompressionPass.DecompressPage(cmd, physicalPage,
                    maths::vec3{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] },
                    maths::vec3{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] });
            }
        }
    }

}
