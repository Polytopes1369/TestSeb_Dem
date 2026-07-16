#include "renderer/GpuGeometryPagePool.h"

#include <cassert>

namespace renderer {

    void GpuGeometryPagePool::Init(VmaAllocator allocator, uint32_t maxLogicalPages, uint32_t maxPhysicalPages) {
        Shutdown();

        m_MaxLogicalPages = maxLogicalPages;
        m_PageTable = geometry::GpuPageTable(maxPhysicalPages);

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
    }

    void GpuGeometryPagePool::Shutdown() {
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

    bool GpuGeometryPagePool::BindPage(VkCommandBuffer cmd, uint64_t logicalAddress, VkBuffer srcStagingBuffer,
        VkDeviceSize srcOffset, VkDeviceSize dataSizeBytes) {
        assert(logicalAddress % geometry::kPageSizeBytes == 0 && "logicalAddress must be page-aligned");
        assert(dataSizeBytes <= geometry::kPageSizeBytes && "a single page bind cannot exceed kPageSizeBytes");

        uint32_t pageID = geometry::GpuPageTable::LogicalAddressToPageID(logicalAddress);
        if (pageID >= m_MaxLogicalPages) {
            return false;
        }

        uint32_t physicalPage = m_PageTable.AllocatePage(logicalAddress);
        if (physicalPage == geometry::kInvalidPhysicalPage) {
            // Either already resident, or the physical pool is full -- either way, no other
            // resident page's mapping was touched by this failed attempt.
            return false;
        }

        VkDeviceSize dstOffset = static_cast<VkDeviceSize>(physicalPage) * geometry::kPageSizeBytes;

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = srcOffset;
        copyRegion.dstOffset = dstOffset;
        copyRegion.size = dataSizeBytes;
        vkCmdCopyBuffer(cmd, srcStagingBuffer, m_PhysicalPool.Handle(), 1, &copyRegion);

        // Page-table entry update: exactly 4 bytes at a 4-byte-aligned offset, well within
        // vkCmdUpdateBuffer's 65536-byte limit, so no staging buffer is needed for the
        // indirection-table half of this bind.
        VkDeviceSize entryOffset = static_cast<VkDeviceSize>(pageID) * sizeof(uint32_t);
        vkCmdUpdateBuffer(cmd, m_PageTableBuffer.Handle(), entryOffset, sizeof(uint32_t), &physicalPage);

        // Two writes (the geometry copy into m_PhysicalPool, the table entry update into
        // m_PageTableBuffer) both need to become visible to shader storage-buffer reads before
        // any subsequent draw/dispatch can safely dereference either resource; a shader must read
        // the table entry to learn the offset, then read the pool at that offset, so both writes
        // share the same destination stage/access mask and are covered by one VkDependencyInfo.
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
        vkCmdPipelineBarrier2(cmd, &depInfo);

        return true;
    }

    bool GpuGeometryPagePool::UnbindPage(VkCommandBuffer cmd, uint64_t logicalAddress) {
        if (!m_PageTable.IsResident(logicalAddress)) {
            return false;
        }

        uint32_t pageID = geometry::GpuPageTable::LogicalAddressToPageID(logicalAddress);

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

        for (uint64_t logicalAddress : candidates) {
            bool unbound = UnbindPage(cmd, logicalAddress);
            assert(unbound && "SelectLeastRecentlyUsedPages returned a logical address that was not actually resident");
            (void)unbound; // Silence an unused-variable warning in builds where assert() compiles to nothing.
        }

        return candidates;
    }

    bool GpuGeometryPagePool::BindPageEvictingIfFull(VkCommandBuffer cmd, uint64_t logicalAddress, VkBuffer srcStagingBuffer,
        VkDeviceSize srcOffset, VkDeviceSize dataSizeBytes, uint32_t maxEvictions) {
        if (m_PageTable.IsResident(logicalAddress)) {
            // Matches BindPage()'s own already-resident failure case -- no eviction should ever
            // be attempted for a page that does not actually need a free slot.
            return false;
        }

        if (m_PageTable.GetResidentPageCount() >= m_PageTable.GetCapacity()) {
            // Every physical slot is currently resident: free the least-recently-used ones first
            // so BindPage() below has somewhere to put the new page. The freed slot(s) go back
            // onto geometry::GpuPageTable's free list as part of EvictLeastRecentlyUsedPages(),
            // making them immediately available to the AllocatePage() call BindPage() performs
            // next, in this same command buffer.
            EvictLeastRecentlyUsedPages(cmd, maxEvictions);
        }

        return BindPage(cmd, logicalAddress, srcStagingBuffer, srcOffset, dataSizeBytes);
    }

}
