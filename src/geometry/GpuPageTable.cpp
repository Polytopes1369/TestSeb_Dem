#include "geometry/GpuPageTable.h"
#include "core/Logger.h"
#include <format>

namespace geometry {

    GpuPageTable::GpuPageTable(uint32_t physicalPageCapacity)
        : m_PhysicalPageCapacity(physicalPageCapacity) {
        m_FreePhysicalPages.reserve(physicalPageCapacity);
        // Every physical page index must be a valid direct index into m_LRUNodes the moment it is
        // first handed out by AllocatePage, so this is a resize (real, default-constructed
        // elements), not a reserve.
        m_LRUNodes.resize(physicalPageCapacity);
        LOG_INFO(std::format("[GpuPageTable] Initialized page table with capacity of {} physical pages ({} KB).", physicalPageCapacity, (physicalPageCapacity * kPageSizeBytes) / 1024));
    }

    uint32_t GpuPageTable::AllocatePage(uint64_t logicalAddress) {
        if (m_LogicalToPhysical.contains(logicalAddress)) {
            // Already resident: the caller must FreePage() first if it wants to remap this
            // logical address to a different physical slot.
            return kInvalidPhysicalPage;
        }

        uint32_t physicalPage;
        if (!m_FreePhysicalPages.empty()) {
            // Prefer reusing a freed slot over growing into never-touched capacity, so the
            // resident set stays packed toward low physical indices over the table's lifetime.
            physicalPage = m_FreePhysicalPages.back();
            m_FreePhysicalPages.pop_back();
        } else if (m_NextUnusedPhysicalPage < m_PhysicalPageCapacity) {
            physicalPage = m_NextUnusedPhysicalPage;
            ++m_NextUnusedPhysicalPage;
        } else {
            // Every physical slot is resident: the pool is full.
            return kInvalidPhysicalPage;
        }

        m_LogicalToPhysical.emplace(logicalAddress, physicalPage);

        // A freshly-bound page starts life as the most-recently-used entry in the LRU order --
        // it was just loaded because something needed it right now, so evicting it again before
        // anything else would be the worst possible choice.
        m_LRUNodes[physicalPage].logicalAddress = logicalAddress;
        LinkAsMostRecentlyUsed(physicalPage);

        return physicalPage;
    }

    bool GpuPageTable::FreePage(uint64_t logicalAddress) {
        auto it = m_LogicalToPhysical.find(logicalAddress);
        if (it == m_LogicalToPhysical.end()) {
            return false;
        }

        // Only this logical address's own entry is erased and its physical slot is only pushed
        // onto the free stack -- no other entry in m_LogicalToPhysical is touched, so every
        // other resident page's physical index (and therefore its physical byte offset) is
        // unaffected by this call.
        uint32_t physicalPage = it->second;
        UnlinkFromLRUList(physicalPage); // Must happen before the slot is offered for reuse below,
                                          // so a freed slot is never left dangling in the LRU list.
        m_FreePhysicalPages.push_back(physicalPage);
        m_LogicalToPhysical.erase(it);
        return true;
    }

    bool GpuPageTable::IsResident(uint64_t logicalAddress) const {
        return m_LogicalToPhysical.contains(logicalAddress);
    }

    uint32_t GpuPageTable::GetPhysicalPageIndex(uint64_t logicalAddress) const {
        auto it = m_LogicalToPhysical.find(logicalAddress);
        return it == m_LogicalToPhysical.end() ? kInvalidPhysicalPage : it->second;
    }

    uint64_t GpuPageTable::GetPhysicalOffset(uint64_t logicalAddress) const {
        uint32_t physicalPage = GetPhysicalPageIndex(logicalAddress);
        if (physicalPage == kInvalidPhysicalPage) {
            return kInvalidPhysicalOffset;
        }
        return static_cast<uint64_t>(physicalPage) * kPageSizeBytes;
    }

    void GpuPageTable::LinkAsMostRecentlyUsed(uint32_t physicalPage) {
        LRUNode& node = m_LRUNodes[physicalPage];
        node.prev = kInvalidPhysicalPage;
        node.next = m_LRUMostRecentPage;
        if (m_LRUMostRecentPage != kInvalidPhysicalPage) {
            m_LRUNodes[m_LRUMostRecentPage].prev = physicalPage;
        }
        m_LRUMostRecentPage = physicalPage;
        if (m_LRULeastRecentPage == kInvalidPhysicalPage) {
            // The list was empty: this single node is simultaneously the head and the tail.
            m_LRULeastRecentPage = physicalPage;
        }
    }

    void GpuPageTable::UnlinkFromLRUList(uint32_t physicalPage) {
        LRUNode& node = m_LRUNodes[physicalPage];
        if (node.prev != kInvalidPhysicalPage) {
            m_LRUNodes[node.prev].next = node.next;
        } else {
            m_LRUMostRecentPage = node.next;
        }
        if (node.next != kInvalidPhysicalPage) {
            m_LRUNodes[node.next].prev = node.prev;
        } else {
            m_LRULeastRecentPage = node.prev;
        }
        node.prev = kInvalidPhysicalPage;
        node.next = kInvalidPhysicalPage;
    }

    void GpuPageTable::TouchPage(uint64_t logicalAddress) {
        auto it = m_LogicalToPhysical.find(logicalAddress);
        if (it == m_LogicalToPhysical.end()) {
            return;
        }

        uint32_t physicalPage = it->second;
        if (m_LRUMostRecentPage == physicalPage) {
            return; // Already the most-recently-used page: nothing to move.
        }

        UnlinkFromLRUList(physicalPage);
        LinkAsMostRecentlyUsed(physicalPage);
    }

    std::vector<uint64_t> GpuPageTable::SelectLeastRecentlyUsedPages(uint32_t maxCount) const {
        std::vector<uint64_t> result;
        result.reserve(std::min<uint32_t>(maxCount, static_cast<uint32_t>(m_LogicalToPhysical.size())));

        uint32_t physicalPage = m_LRULeastRecentPage;
        while (physicalPage != kInvalidPhysicalPage && result.size() < maxCount) {
            const LRUNode& node = m_LRUNodes[physicalPage];
            result.push_back(node.logicalAddress);
            physicalPage = node.prev; // Walk from the tail toward the more-recently-used end.
        }
        return result;
    }

}
