#include "renderer/streaming/VirtualTextureStreamingCoordinator.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "core/EngineConfig.h"
#include "core/Logger.h"

namespace renderer {

    uint32_t VirtualTextureStreamingCoordinator::BytesPerTexelForFormat(VkFormat format) {
        switch (format) {
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_R16G16_SFLOAT:
            case VK_FORMAT_R16G16_UINT:
            case VK_FORMAT_R32_SFLOAT:
                return 4;
            case VK_FORMAT_R8G8_UNORM:
            case VK_FORMAT_R8G8_UINT:
                return 2;
            case VK_FORMAT_R8_UNORM:
                return 1;
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return 8;
            default:
                LOG_ERROR(std::format(
                    "[VirtualTextureStreamingCoordinator] Unsupported physical pool format {} -- bytes-per-texel unknown, assuming 4.",
                    static_cast<int>(format)));
                return 4;
        }
    }

    bool VirtualTextureStreamingCoordinator::Init(VkDevice device, VmaAllocator allocator,
        VkCommandPool commandPool, VkQueue queue, const std::filesystem::path& cacheFilePath,
        VirtualTextureManager* vtManager) {
        (void)commandPool;
        (void)queue; // Reserved for parity with every other Init()-shaped class; no one-time submit needed here.
        Shutdown();

        m_Device = device;
        m_VTManager = vtManager;

        // --- Compute this deployment's fixed tile block size from the LIVE VirtualTextureManager
        // config (not from the cache file, which may not exist yet on a from-scratch run). ---
        uint32_t bytesPerTexel = 0;
        for (uint32_t i = 0; i < m_VTManager->GetPhysicalPoolCount(); ++i) {
            bytesPerTexel += BytesPerTexelForFormat(m_VTManager->GetPhysicalPoolFormat(i));
        }
        uint32_t tileSizeWithBorder = m_VTManager->GetTileSizeWithBorder();
        uint64_t computedTileBytes = static_cast<uint64_t>(tileSizeWithBorder) * tileSizeWithBorder * bytesPerTexel;
        m_TileBlockCapacityBytes = static_cast<uint32_t>(
            ((computedTileBytes + io::kVTPageSizeBytes - 1u) / io::kVTPageSizeBytes) * io::kVTPageSizeBytes);

        // --- Load the tile index table, if a cache file already exists. A missing file is NOT a
        // hard failure (see Init()'s own doc comment: a from-scratch demo run has nothing cached
        // yet) -- only a genuine I/O error reading an EXISTING file is. ---
        std::error_code existsEc;
        bool fileExists = std::filesystem::exists(cacheFilePath, existsEc);
        if (fileExists) {
            io::VirtualTextureCacheFileHeader header{};
            if (!m_CacheFileManager.ReadHeader(cacheFilePath, header)) {
                LOG_ERROR(std::format("[VirtualTextureStreamingCoordinator] Failed to read header from existing '{}'.",
                    cacheFilePath.string()));
                return false;
            }
            if (!m_CacheFileManager.ReadTileIndexTable(cacheFilePath, header, m_IndexEntries)) {
                LOG_ERROR("[VirtualTextureStreamingCoordinator] Failed to read the tile index table.");
                return false;
            }

            m_PageKeyToEntryIndex.reserve(m_IndexEntries.size());
            uint32_t maxBlockSizeBytes = m_TileBlockCapacityBytes;
            for (uint32_t i = 0; i < m_IndexEntries.size(); ++i) {
                m_PageKeyToEntryIndex[m_IndexEntries[i].pageKey] = i;
                maxBlockSizeBytes = std::max(maxBlockSizeBytes, m_IndexEntries[i].blockSizeBytes);
            }
            if (maxBlockSizeBytes > m_TileBlockCapacityBytes) {
                // Defensive widening: the cache file's own tiles are larger than the live manager's
                // current config would produce (e.g. built with a larger tileSize/border/pool set) --
                // widen the raw buffer/staging ring capacity to fit them rather than truncating reads.
                LOG_WARNING(std::format(
                    "[VirtualTextureStreamingCoordinator] '{}' tiles ({} B) are larger than the live "
                    "VirtualTextureManager config expects ({} B) -- widening I/O buffer capacity.",
                    cacheFilePath.string(), maxBlockSizeBytes, m_TileBlockCapacityBytes));
                m_TileBlockCapacityBytes = maxBlockSizeBytes;
            }

            if (!m_Streamer.Open(cacheFilePath)) {
                LOG_ERROR(std::format("[VirtualTextureStreamingCoordinator] Failed to open '{}' for async streaming.",
                    cacheFilePath.string()));
            }
        } else {
            LOG_WARNING(std::format(
                "[VirtualTextureStreamingCoordinator] '{}' does not exist yet -- streaming is inert "
                "(every page-table entry keeps falling back to the always-resident root page) until "
                "a future run writes one.", cacheFilePath.string()));
        }

        for (uint32_t i = 0; i < kMaxInFlightReads; ++i) {
            m_RawBuffers[i] = geometry::AsyncFileStreamer::AllocateAlignedBuffer(m_TileBlockCapacityBytes);
            m_SlotBusy[i] = false;
        }

        // CPU_ONLY + mapped: VirtualTextureManager::UploadTileData's own vkCmdCopyBufferToImage
        // reads from this as a plain TRANSFER_SRC staging buffer, exactly like
        // GeometryStreamingCoordinator's identical m_StagingRing.
        m_StagingRing.Create(allocator,
            static_cast<VkDeviceSize>(m_TileBlockCapacityBytes) * kMaxTilesBoundPerFrame,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);

        m_Feedback.Init(allocator, kFeedbackCapacity);

        LOG_INFO(std::format(
            "[VirtualTextureStreamingCoordinator] Initialized: {} tile(s) trackable, {} B/tile, {} max in-flight reads.",
            m_IndexEntries.size(), m_TileBlockCapacityBytes, kMaxInFlightReads));
        return true;
    }

    void VirtualTextureStreamingCoordinator::Shutdown() {
        m_Streamer.Close();

        for (uint32_t i = 0; i < kMaxInFlightReads; ++i) {
            if (m_RawBuffers[i] != nullptr) {
                geometry::AsyncFileStreamer::FreeAlignedBuffer(m_RawBuffers[i]);
            }
            m_RawBuffers[i] = nullptr;
            m_SlotBusy[i] = false;
        }

        m_StagingRing.Destroy();
        m_Feedback.Shutdown();

        m_CompletedReads.clear();
        m_IndexEntries.clear();
        m_PageKeyToEntryIndex.clear();
        m_TileBlockCapacityBytes = 0;
        m_TotalBytesCompleted.store(0, std::memory_order_relaxed);
        m_VTManager = nullptr;
        m_Device = VK_NULL_HANDLE;
    }

    uint32_t VirtualTextureStreamingCoordinator::FindFreeSlot() const {
        for (uint32_t i = 0; i < kMaxInFlightReads; ++i) {
            if (!m_SlotBusy[i]) {
                return i;
            }
        }
        return kInvalidSlot;
    }

    void VirtualTextureStreamingCoordinator::RecordBeginFrame(VkCommandBuffer cmd) {
        // Reset THIS frame's feedback counter before any shader that samples the virtual texture
        // (and might call RequestVirtualTexturePageResidency()) runs -- must happen every frame
        // regardless of whether the streamer itself is open, matching VirtualShadowMapPass's own
        // "clear always runs, the rest of the pipeline can be inert" convention.
        m_Feedback.RecordClear(cmd);

        // config::lumen::BUILD_VIRTUAL_TEXTURES temporary kill-switch: skipping this entire block
        // leaves every page-table entry permanently falling back to the always-resident, neutral-
        // white root page (see VirtualTextureManager::Init's own comment on why that fallback is
        // always safe to sample) -- m_Feedback's clear above still runs either way so the SSBO a
        // shader unconditionally writes into never accumulates stale atomic counts across frames.
        if (!config::lumen::BUILD_VIRTUAL_TEXTURES) {
            return;
        }

        if (!m_Streamer.IsOpen()) {
            return; // No cache file to stream from (see Init()'s own comment) -- inert.
        }

        // --- 1. Fold LAST frame's page-miss reports into the dedup request queue. ---
        uint32_t overflowedRequestCount = 0;
        std::vector<uint32_t> missedPageKeys = m_Feedback.ReadRequestedClusterIDs(&overflowedRequestCount);
        if (overflowedRequestCount > 0) {
            LOG_WARNING(std::format(
                "[VirtualTextureStreamingCoordinator] Feedback buffer saturated: {} page-miss report(s) dropped this frame (capacity={}).",
                overflowedRequestCount, m_Feedback.GetCapacity()));
        }
        // Priority = mip level (packed into the low 4 bits of the key, see
        // VirtualTextureManager::PackPageKey) -- a coarser (higher mip) tile is requested before a
        // finer one, matching geometry::GeometryStreamingCoordinator's DAG-level priority for the
        // same reason: it unlocks a larger visible fallback area sooner.
        std::vector<float> missedPriorities;
        missedPriorities.reserve(missedPageKeys.size());
        for (uint32_t pageKey : missedPageKeys) {
            uint32_t x = 0, y = 0, mip = 0;
            VirtualTextureManager::UnpackPageKey(pageKey, x, y, mip);
            missedPriorities.push_back(float(mip));
        }
        m_RequestQueue.SubmitFrameRequests(missedPageKeys, missedPriorities);

        // --- 2. Issue up to kMaxNewReadsPerFrame new async reads, bounded by free I/O slots. ---
        for (uint32_t issued = 0; issued < kMaxNewReadsPerFrame; ++issued) {
            uint32_t slot = FindFreeSlot();
            if (slot == kInvalidSlot) {
                break; // Every in-flight slot is busy -- try again next frame.
            }

            uint32_t pageKey = 0;
            if (!m_RequestQueue.PopNextRequest(pageKey)) {
                break; // Nothing left to request this frame.
            }

            auto it = m_PageKeyToEntryIndex.find(pageKey);
            if (it == m_PageKeyToEntryIndex.end()) {
                // No cached tile exists on disk for this page (e.g. a page never baked/persisted
                // yet) -- nothing to stream; drop the report so it isn't retried forever against a
                // tile that will never appear without an out-of-band bake/re-cache step.
                m_RequestQueue.MarkRequestCompleted(pageKey);
                continue;
            }

            const io::VirtualTextureTileIndexEntry& entry = m_IndexEntries[it->second];
            m_SlotBusy[slot] = true;
            void* destination = m_RawBuffers[slot];
            uint64_t fileOffset = entry.virtualAddress; // Already an absolute, page-aligned file offset.

            bool submitted = m_Streamer.SubmitRead(fileOffset, destination, m_TileBlockCapacityBytes,
                [this, pageKey, slot](bool success, uint32_t bytesTransferred) {
                    std::lock_guard<std::mutex> lock(m_CompletedMutex);
                    m_CompletedReads.push_back(CompletedRead{ pageKey, slot, bytesTransferred, success });
                });

            if (!submitted) {
                m_SlotBusy[slot] = false;
                m_RequestQueue.MarkRequestCompleted(pageKey); // Allow a future miss report to retry it.
            }
        }

        // --- 3. Drain completed reads into the physical pool, bounded by BOTH kMaxTilesBoundPerFrame
        // AND kMaxBytesUploadedPerFrame (explicit bandwidth throttling -- see that constant's own
        // comment on why a byte budget, not just a count, is what actually bounds upload cost). ---
        std::vector<CompletedRead> drained;
        {
            std::lock_guard<std::mutex> lock(m_CompletedMutex);
            uint32_t drainCount = std::min<uint32_t>(kMaxTilesBoundPerFrame, static_cast<uint32_t>(m_CompletedReads.size()));
            drained.assign(m_CompletedReads.begin(), m_CompletedReads.begin() + drainCount);
            m_CompletedReads.erase(m_CompletedReads.begin(), m_CompletedReads.begin() + drainCount);
        }

        uint32_t ringSlot = 0;
        uint64_t bytesUploadedThisFrame = 0;
        bool updatedAnyTile = false;
        for (const CompletedRead& completed : drained) {
            // Free the I/O slot and untrack the request regardless of outcome (mirrors
            // GeometryStreamingCoordinator's identical unconditional pattern): a read that is
            // skipped below for being over THIS frame's byte budget is not lost forever -- the page
            // is simply still non-resident, so a future frame's shader sample will report it missing
            // again (see virtual_texture_lookup.glsl's own miss-detection), re-queuing a fresh disk
            // read. The minor cost is re-reading a tile whose data briefly existed in a raw buffer
            // and was discarded -- an accepted, documented tradeoff for keeping the throttle logic
            // simple rather than threading a "put this back at the front of the queue" path through
            // geometry::StreamingRequestQueue's API.
            m_SlotBusy[completed.slotIndex] = false;
            m_RequestQueue.MarkRequestCompleted(completed.pageKey);

            if (!completed.success || completed.bytesTransferred < m_TileBlockCapacityBytes) {
                LOG_WARNING(std::format(
                    "[VirtualTextureStreamingCoordinator] Read failed or short for pageKey {}: success={}, bytes={}",
                    completed.pageKey, completed.success, completed.bytesTransferred));
                continue;
            }

            if (bytesUploadedThisFrame > 0 && bytesUploadedThisFrame + completed.bytesTransferred > kMaxBytesUploadedPerFrame) {
                continue; // Byte budget exhausted this frame -- see the throttle comment above.
            }

            auto it = m_PageKeyToEntryIndex.find(completed.pageKey);
            if (it == m_PageKeyToEntryIndex.end()) {
                continue; // Defensive: cannot happen (this pageKey was resolved via the same map in step 2).
            }
            const io::VirtualTextureTileIndexEntry& entry = m_IndexEntries[it->second];

            uint32_t x = 0, y = 0, mip = 0;
            VirtualTextureManager::UnpackPageKey(entry.pageKey, x, y, mip);

            VkDeviceSize ringOffset = static_cast<VkDeviceSize>(ringSlot) * m_TileBlockCapacityBytes;
            std::memcpy(static_cast<char*>(m_StagingRing.MappedData()) + ringOffset,
                m_RawBuffers[completed.slotIndex], m_TileBlockCapacityBytes);
            ++ringSlot;

            // Allocates (evicting LRU if the physical pool is full) and writes this frame's CPU-side
            // page-table mirror entry -- see renderer::VirtualTextureManager::RequestPageResidency's
            // own comment for the LRU eviction algorithm this call relies on.
            uint32_t physicalPageIndex = m_VTManager->RequestPageResidency(cmd, x, y, mip);

            // Upload each physical pool's own channel from its own sub-offset within the tile's
            // concatenated texel blob (see io::VirtualTextureTileData's own struct comment for the
            // pool-concatenation order the .vtcache writer used).
            uint32_t poolOffsetBytes = 0;
            uint32_t tileSizeWithBorder = m_VTManager->GetTileSizeWithBorder();
            for (uint32_t poolIndex = 0; poolIndex < m_VTManager->GetPhysicalPoolCount(); ++poolIndex) {
                uint32_t bytesPerTexel = BytesPerTexelForFormat(m_VTManager->GetPhysicalPoolFormat(poolIndex));
                uint32_t poolSizeBytes = tileSizeWithBorder * tileSizeWithBorder * bytesPerTexel;
                m_VTManager->UploadTileData(cmd, physicalPageIndex, m_StagingRing.Handle(),
                    ringOffset + poolOffsetBytes, poolIndex);
                poolOffsetBytes += poolSizeBytes;
            }

            m_TotalBytesCompleted.fetch_add(completed.bytesTransferred, std::memory_order_relaxed);
            bytesUploadedThisFrame += completed.bytesTransferred;
            updatedAnyTile = true;
        }

        if (updatedAnyTile) {
            // Flushes every RequestPageResidency() write above (CPU mirror -> GPU page table image)
            // in one batched call -- UpdatePageTableImage() early-outs entirely if nothing is dirty.
            m_VTManager->UpdatePageTableImage(cmd);
        }

        if (!missedPageKeys.empty() || !drained.empty()) {
            LOG_INFO(std::format(
                "[VirtualTextureStreamingCoordinator] frame summary: missed={}, pendingInQueue={}, inFlightReads={}, "
                "drainedThisFrame={}, bytesUploadedThisFrame={}, residentPages={}/{}",
                missedPageKeys.size(), m_RequestQueue.PendingCount(), m_Streamer.PendingRequestCount(),
                drained.size(), bytesUploadedThisFrame, m_VTManager->GetResidentPageCount(), m_VTManager->GetPhysicalPageCapacity()));
        }
    }

    void VirtualTextureStreamingCoordinator::RecordEndFrame(VkCommandBuffer cmd) {
        // Captures this frame's page-miss reports for RecordBeginFrame() to consume next frame --
        // see this class' own header comment for the full one-frame-lag contract.
        m_Feedback.RecordReadback(cmd);
    }

}
