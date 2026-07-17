#include "renderer/streaming/GpuGeometryPagePool.h"
#include "core/Logger.h"

#include <cassert>
#include <format>

namespace renderer {

    void GpuGeometryPagePool::Init(VmaAllocator allocator, uint32_t maxLogicalPages, uint32_t maxPhysicalPages,
        uint32_t transferQueueFamilyIndex, uint32_t graphicsQueueFamilyIndex) {
        Shutdown();

        m_MaxLogicalPages = maxLogicalPages;
        m_PageTable = geometry::GpuPageTable(maxPhysicalPages);
        m_TransferQueueFamilyIndex = transferQueueFamilyIndex;
        m_GraphicsQueueFamilyIndex = graphicsQueueFamilyIndex;
        m_NeedsOwnershipTransfer = transferQueueFamilyIndex != graphicsQueueFamilyIndex;

        m_PhysicalPool.Create(
            allocator,
            static_cast<VkDeviceSize>(maxPhysicalPages) * geometry::kPageSizeBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_PageTableBuffer.Create(
            allocator,
            static_cast<VkDeviceSize>(maxLogicalPages) * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        LOG_INFO(std::format("[GpuGeometryPagePool] Initialized page pool: maxLogicalPages={}, maxPhysicalPages={}, poolSize={} KB",
            maxLogicalPages, maxPhysicalPages, (maxPhysicalPages * geometry::kPageSizeBytes) / 1024));
    }

    void GpuGeometryPagePool::Shutdown() {
        LOG_INFO("[GpuGeometryPagePool] Shutting down page pool...");
        m_PhysicalPool.Destroy();
        m_PageTableBuffer.Destroy();
        m_PageTable = geometry::GpuPageTable(0);
        m_MaxLogicalPages = 0;
    }

    void GpuGeometryPagePool::ClearPageTable(VkCommandBuffer cmd) {
        // kUnmappedSentinel is 0xFFFFFFFFu -- every byte set -- so a single repeated-32-bit-word
        // vkCmdFillBuffer covers the whole table in one GPU-side command, no staging buffer or
        // per-entry vkCmdUpdateBuffer loop required.
        static_assert(kUnmappedSentinel == 0xFFFFFFFFu,
            "ClearPageTable relies on kUnmappedSentinel being an all-ones fill word");
        vkCmdFillBuffer(cmd, m_PageTableBuffer.Handle(), 0, VK_WHOLE_SIZE, kUnmappedSentinel);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    bool GpuGeometryPagePool::UploadPageData(VkCommandBuffer transferCmd, uint64_t logicalAddress, VkBuffer srcStagingBuffer,
        VkDeviceSize srcOffset, VkDeviceSize dataSizeBytes) {
        assert(logicalAddress % geometry::kPageSizeBytes == 0 && "logicalAddress must be page-aligned");
        assert(dataSizeBytes <= geometry::kPageSizeBytes && "a single page bind cannot exceed kPageSizeBytes");

        uint32_t pageID = geometry::GpuPageTable::LogicalAddressToPageID(logicalAddress);
        if (pageID >= m_MaxLogicalPages) {
            LOG_ERROR(std::format("[GpuGeometryPagePool] UploadPageData failed: pageID {} is out of range (max: {})", pageID, m_MaxLogicalPages));
            return false;
        }

        uint32_t physicalPage = m_PageTable.AllocatePage(logicalAddress);
        if (physicalPage == geometry::kInvalidPhysicalPage) {
            // Either already resident, or the physical pool is full -- either way, no other
            // resident page's mapping was touched by this failed attempt.
            if (!m_PageTable.IsResident(logicalAddress)) {
                LOG_WARNING(std::format("[GpuGeometryPagePool] Failed to allocate physical page for logicalAddress {:#x} (pool is full, capacity: {})",
                    logicalAddress, m_PageTable.GetCapacity()));
            }
            return false;
        }

        VkDeviceSize dstOffset = static_cast<VkDeviceSize>(physicalPage) * geometry::kPageSizeBytes;

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = srcOffset;
        copyRegion.dstOffset = dstOffset;
        copyRegion.size = dataSizeBytes;
        vkCmdCopyBuffer(transferCmd, srcStagingBuffer, m_PhysicalPool.Handle(), 1, &copyRegion);

        LOG_INFO(std::format("[GpuGeometryPagePool] Uploaded logicalAddress {:#x} to physical page slot {} (offset: {:#x})",
            logicalAddress, physicalPage, dstOffset));

#ifndef NDEBUG
        m_DebugBoundAtFrame[logicalAddress] = m_DebugFrameCounter;
#endif

        return true;
    }

    void GpuGeometryPagePool::ReleasePhysicalPoolOwnership(VkCommandBuffer transferCmd) {
        if (!m_NeedsOwnershipTransfer) {
            return; // Same queue family as the graphics/compute consumer -- nothing to release.
        }

        // RELEASE half of the ownership transfer: makes this frame's UploadPageData() copies
        // available to be acquired by m_GraphicsQueueFamilyIndex. Per the Vulkan spec, a queue
        // family ownership transfer's release operation's dstAccessMask is ignored (the acquiring
        // queue's own barrier is what actually makes the data visible to it), so it is left 0 here
        // -- only srcStageMask/srcAccessMask (what must complete before the release) are
        // meaningful on this side.
        VkBufferMemoryBarrier2 releaseBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        releaseBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        releaseBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        releaseBarrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        releaseBarrier.dstAccessMask = VK_ACCESS_2_NONE;
        releaseBarrier.srcQueueFamilyIndex = m_TransferQueueFamilyIndex;
        releaseBarrier.dstQueueFamilyIndex = m_GraphicsQueueFamilyIndex;
        releaseBarrier.buffer = m_PhysicalPool.Handle();
        releaseBarrier.offset = 0;
        releaseBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &releaseBarrier;
        vkCmdPipelineBarrier2(transferCmd, &depInfo);
    }

    void GpuGeometryPagePool::AcquirePhysicalPoolOwnership(VkCommandBuffer graphicsCmd) {
        if (!m_NeedsOwnershipTransfer) {
            return;
        }

        // ACQUIRE half: per the spec, the release side's srcAccessMask is the one that matters for
        // "what must complete first" (already encoded in the release barrier above via the
        // semaphore ordering main.cpp's submit establishes); this side's srcAccessMask is ignored
        // by the spec and left 0, while dstStageMask/dstAccessMask describe the first real
        // consumer -- FinalizeBoundPage()'s own barrier further narrows this to the exact shader
        // stages that read the pool, so this acquire only needs to make the transferred bytes
        // available to this queue family at all, not yet visible to any specific stage.
        VkBufferMemoryBarrier2 acquireBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        acquireBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        acquireBarrier.srcAccessMask = VK_ACCESS_2_NONE;
        acquireBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        acquireBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        acquireBarrier.srcQueueFamilyIndex = m_TransferQueueFamilyIndex;
        acquireBarrier.dstQueueFamilyIndex = m_GraphicsQueueFamilyIndex;
        acquireBarrier.buffer = m_PhysicalPool.Handle();
        acquireBarrier.offset = 0;
        acquireBarrier.size = VK_WHOLE_SIZE;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &acquireBarrier;
        vkCmdPipelineBarrier2(graphicsCmd, &depInfo);
    }

    void GpuGeometryPagePool::FinalizeBoundPage(VkCommandBuffer graphicsCmd, uint64_t logicalAddress) {
        uint32_t pageID = geometry::GpuPageTable::LogicalAddressToPageID(logicalAddress);
        uint32_t physicalPage = m_PageTable.GetPhysicalPageIndex(logicalAddress);

        // Page-table entry update: exactly 4 bytes at a 4-byte-aligned offset, well within
        // vkCmdUpdateBuffer's 65536-byte limit, so no staging buffer is needed for the
        // indirection-table half of this bind.
        VkDeviceSize entryOffset = static_cast<VkDeviceSize>(pageID) * sizeof(uint32_t);
        vkCmdUpdateBuffer(graphicsCmd, m_PageTableBuffer.Handle(), entryOffset, sizeof(uint32_t), &physicalPage);

        // Two writes (UploadPageData()'s geometry copy into m_PhysicalPool -- already made visible
        // to this queue family by AcquirePhysicalPoolOwnership(), or trivially visible if no
        // ownership transfer was needed -- and the table entry update into m_PageTableBuffer just
        // above) both need to become visible to shader storage-buffer reads before any subsequent
        // draw/dispatch can safely dereference either resource; a shader must read the table entry
        // to learn the offset, then read the pool at that offset, so both writes share the same
        // destination stage/access mask and are covered by one VkDependencyInfo.
        VkMemoryBarrier2 barriers[2]{};
        barriers[0].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        barriers[1].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT; // vkCmdUpdateBuffer is classified as CLEAR by the Vulkan spec's stage table.
        barriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 2;
        depInfo.pMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(graphicsCmd, &depInfo);

        LOG_INFO(std::format("[GpuGeometryPagePool] Finalized logicalAddress {:#x} at physical page slot {}",
            logicalAddress, physicalPage));
    }

    bool GpuGeometryPagePool::UnbindPage(VkCommandBuffer cmd, uint64_t logicalAddress) {
        if (!m_PageTable.IsResident(logicalAddress)) {
            return false;
        }

        uint32_t pageID = geometry::GpuPageTable::LogicalAddressToPageID(logicalAddress);
        uint32_t physicalPage = m_PageTable.GetPhysicalPageIndex(logicalAddress);

        LOG_INFO(std::format("[GpuGeometryPagePool] Unbound logicalAddress {:#x} from physical page slot {}",
            logicalAddress, physicalPage));

#ifndef NDEBUG
        // Eviction-churn check: see this class's DebugAdvanceFrame() doc comment for why a page
        // evicted only a handful of frames after being bound is suspicious -- it strongly suggests
        // the page was still being drawn every frame and got evicted purely because its LRU
        // recency was never refreshed, not because it genuinely stopped being needed.
        // renderer::GeometryStreamingCoordinator::ProcessFeedbackAndDrainCompletions now calls
        // TouchPages() every frame with resident-touch reports from
        // ClusterLODResidencyFallback.comp's RecordResidentTouch(), so this warning firing under
        // normal (non-overflowing) camera movement would indicate a regression in that wiring, not
        // an open/expected issue.
        auto boundIt = m_DebugBoundAtFrame.find(logicalAddress);
        if (boundIt != m_DebugBoundAtFrame.end()) {
            uint64_t framesSinceBind = m_DebugFrameCounter - boundIt->second;
            if (framesSinceBind < kDebugChurnThresholdFrames) {
                LOG_WARNING(std::format("[GpuGeometryPagePool] POSSIBLE THRASHING: logicalAddress {:#x} (physical slot {}) evicted only {} frame(s) after being bound -- "
                    "either the resident-touch feedback channel overflowed this frame (see FeedbackBuffer saturation warnings), or the page genuinely stopped "
                    "being required and this is expected LRU turnover. If clusters keep flickering between resident/non-resident with no matching overflow warning, investigate.",
                    logicalAddress, physicalPage, framesSinceBind));
            }
            m_DebugBoundAtFrame.erase(boundIt);
        }
#endif

        // Free the CPU-side mapping first: the physical slot is now eligible for reuse by a
        // concurrently-recorded future BindPage() call, but that call will not itself become
        // visible to any shader read until its own barrier -- ordering here is bookkeeping-only.
        m_PageTable.FreePage(logicalAddress);

        uint32_t sentinel = kUnmappedSentinel;
        VkDeviceSize entryOffset = static_cast<VkDeviceSize>(pageID) * sizeof(uint32_t);
        vkCmdUpdateBuffer(cmd, m_PageTableBuffer.Handle(), entryOffset, sizeof(uint32_t), &sentinel);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        return true;
    }

    std::vector<uint64_t> GpuGeometryPagePool::EvictLeastRecentlyUsedPages(VkCommandBuffer cmd, uint32_t maxPagesToEvict) {
        // Selecting every candidate up front (rather than re-querying the LRU list after each
        // individual eviction) is safe here: UnbindPage() only ever removes the exact physical
        // page backing the logical address it is given, so an earlier eviction in this loop can
        // never invalidate a later candidate's own eligibility.
        std::vector<uint64_t> candidates = m_PageTable.SelectLeastRecentlyUsedPages(maxPagesToEvict);

        if (!candidates.empty()) {
            LOG_INFO(std::format("[GpuGeometryPagePool] Evicting {} LRU pages from physical pool (resident: {}/{} before eviction).",
                candidates.size(), m_PageTable.GetResidentPageCount(), m_PageTable.GetCapacity()));
        }

        for (uint64_t logicalAddress : candidates) {
            bool unbound = UnbindPage(cmd, logicalAddress);
            assert(unbound && "SelectLeastRecentlyUsedPages returned a logical address that was not actually resident");
            (void)unbound; // Silence an unused-variable warning in builds where assert() compiles to nothing.
        }

        return candidates;
    }

    bool GpuGeometryPagePool::UploadPageDataEvictingIfFull(VkCommandBuffer transferCmd, VkCommandBuffer evictionCmd,
        uint64_t logicalAddress, VkBuffer srcStagingBuffer,
        VkDeviceSize srcOffset, VkDeviceSize dataSizeBytes, uint32_t maxEvictions) {
        if (m_PageTable.IsResident(logicalAddress)) {
            // Matches UploadPageData()'s own already-resident failure case -- no eviction should
            // ever be attempted for a page that does not actually need a free slot.
            return false;
        }

        if (m_PageTable.GetResidentPageCount() >= m_PageTable.GetCapacity()) {
            // Every physical slot is currently resident: free the least-recently-used ones first
            // so UploadPageData() below has somewhere to put the new page. Recorded on
            // `evictionCmd` (the graphics command buffer): eviction only ever touches the page
            // table, never physical pool bytes, so it needs no transfer-queue involvement at all.
            // The freed slot(s) go back onto geometry::GpuPageTable's free list as part of
            // EvictLeastRecentlyUsedPages(), making them immediately available to the
            // AllocatePage() call UploadPageData() performs next -- this is pure CPU-side
            // bookkeeping, so it is available the instant this function call returns, regardless
            // of which command buffer/queue either eviction or upload actually executes on.
            EvictLeastRecentlyUsedPages(evictionCmd, maxEvictions);
        }

        return UploadPageData(transferCmd, logicalAddress, srcStagingBuffer, srcOffset, dataSizeBytes);
    }

}
