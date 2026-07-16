// Standalone, framework-free unit test for geometry::GpuPageTable (src/geometry/GpuPageTable.h),
// the CPU-side page table backing the GPU geometry paging pool (renderer::GpuGeometryPagePool).
//
// Purely CPU-side: GpuPageTable has zero Vulkan dependency, so this needs neither a GPU device
// nor a window. Exits 0 if every check passes, non-zero otherwise, so it can be registered with
// CTest (see the top-level CMakeLists.txt) without pulling in any external test framework.
//
// This is the "GPU page binding test" requested for the paging system: what makes a page bind/
// unbind correct is that the physical byte offset handed to vkCmdCopyBuffer for one logical
// address never shifts underneath an unrelated, still-resident logical address -- that offset
// (physicalPageIndex * kPageSizeBytes) is entirely determined by GpuPageTable's bookkeeping, so
// proving the invariant holds here proves it holds for every real GPU bind that offset drives.
//
// What is validated:
//   1. Sequential allocation hands out distinct, page-aligned physical offsets within capacity.
//   2. Freeing one page, then allocating a new one, leaves every OTHER resident page's physical
//      offset completely unchanged (the core "neighbor pages are not disturbed" requirement).
//   3. The freed physical slot is the one reused by the next allocation (no wasted capacity).
//   4. Allocating past capacity fails cleanly and disturbs no existing mapping.
//   5. Freeing a non-resident address, or re-allocating an already-resident one, is a no-op that
//      changes no other mapping.
//   6. Queries on a never-resident logical address report "not resident" consistently.

#include "geometry/GpuPageTable.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>

namespace {

    int g_FailCount = 0;

    void Check(bool condition, const std::string& description) {
        if (!condition) {
            std::cerr << "[FAIL] " << description << std::endl;
            ++g_FailCount;
        }
    }

    uint64_t LogicalAddress(uint32_t clusterIndex) {
        return static_cast<uint64_t>(clusterIndex) * geometry::kPageSizeBytes;
    }

    // -----------------------------------------------------------------------------------------
    // Test 1: sequential allocation hands out distinct, page-aligned, in-bounds physical offsets.
    // -----------------------------------------------------------------------------------------
    void TestSequentialAllocation() {
        constexpr uint32_t kCapacity = 8;
        geometry::GpuPageTable table(kCapacity);

        std::unordered_map<uint64_t, uint64_t> observedOffsets;
        for (uint32_t i = 0; i < kCapacity; ++i) {
            uint64_t logicalAddress = LogicalAddress(i);
            uint32_t physicalPage = table.AllocatePage(logicalAddress);

            Check(physicalPage != geometry::kInvalidPhysicalPage,
                "TestSequentialAllocation: allocation " + std::to_string(i) + " should succeed");

            uint64_t offset = table.GetPhysicalOffset(logicalAddress);
            Check(offset % geometry::kPageSizeBytes == 0,
                "TestSequentialAllocation: offset must be page-aligned");
            Check(offset < static_cast<uint64_t>(kCapacity) * geometry::kPageSizeBytes,
                "TestSequentialAllocation: offset must be within the physical pool bounds");

            for (const auto& [existingLogical, existingOffset] : observedOffsets) {
                Check(existingOffset != offset,
                    "TestSequentialAllocation: two distinct logical addresses must never share a physical offset");
            }
            observedOffsets[logicalAddress] = offset;
        }

        Check(table.GetResidentPageCount() == kCapacity,
            "TestSequentialAllocation: resident count must equal the number of successful allocations");
    }

    // -----------------------------------------------------------------------------------------
    // Test 2 (the core requirement): freeing and reallocating a physical page must not alter the
    // physical offsets of any other, still-resident logical address.
    // -----------------------------------------------------------------------------------------
    void TestFreeAndReallocateDoesNotDisturbNeighbors() {
        constexpr uint32_t kCapacity = 10;
        geometry::GpuPageTable table(kCapacity);

        std::unordered_map<uint64_t, uint64_t> offsetsBeforeFree;
        for (uint32_t i = 0; i < kCapacity; ++i) {
            uint64_t logicalAddress = LogicalAddress(i);
            table.AllocatePage(logicalAddress);
            offsetsBeforeFree[logicalAddress] = table.GetPhysicalOffset(logicalAddress);
        }

        // Free a page in the middle of the range (neither the first nor the last allocated).
        constexpr uint32_t kFreedIndex = 5;
        uint64_t freedLogicalAddress = LogicalAddress(kFreedIndex);
        uint64_t freedOffsetBeforeFree = offsetsBeforeFree[freedLogicalAddress];

        bool freed = table.FreePage(freedLogicalAddress);
        Check(freed, "TestFreeAndReallocateDoesNotDisturbNeighbors: FreePage must succeed for a resident page");
        Check(!table.IsResident(freedLogicalAddress),
            "TestFreeAndReallocateDoesNotDisturbNeighbors: freed logical address must no longer be resident");

        // Every OTHER logical address's physical offset must be exactly what it was before the free.
        for (const auto& [logicalAddress, offsetBefore] : offsetsBeforeFree) {
            if (logicalAddress == freedLogicalAddress) {
                continue;
            }
            uint64_t offsetAfterFree = table.GetPhysicalOffset(logicalAddress);
            Check(offsetAfterFree == offsetBefore,
                "TestFreeAndReallocateDoesNotDisturbNeighbors: neighbor offset must survive an unrelated FreePage");
        }

        // Allocate a brand-new logical address; it must reuse the freed physical slot rather than
        // growing the pool (the pool is already at capacity minus the one just-freed slot).
        uint64_t newLogicalAddress = LogicalAddress(kCapacity + 1); // outside the original [0, kCapacity) range
        uint32_t newPhysicalPage = table.AllocatePage(newLogicalAddress);
        Check(newPhysicalPage != geometry::kInvalidPhysicalPage,
            "TestFreeAndReallocateDoesNotDisturbNeighbors: reallocation into a freed slot must succeed");
        Check(table.GetPhysicalOffset(newLogicalAddress) == freedOffsetBeforeFree,
            "TestFreeAndReallocateDoesNotDisturbNeighbors: the freed physical slot must be the one reused");

        // Again, every OTHER logical address's physical offset must be completely unchanged.
        for (const auto& [logicalAddress, offsetBefore] : offsetsBeforeFree) {
            if (logicalAddress == freedLogicalAddress) {
                continue;
            }
            uint64_t offsetAfterRealloc = table.GetPhysicalOffset(logicalAddress);
            Check(offsetAfterRealloc == offsetBefore,
                "TestFreeAndReallocateDoesNotDisturbNeighbors: neighbor offset must survive an unrelated AllocatePage reusing a freed slot");
        }

        Check(table.GetResidentPageCount() == kCapacity,
            "TestFreeAndReallocateDoesNotDisturbNeighbors: resident count must return to full capacity after reallocation");
    }

    // -----------------------------------------------------------------------------------------
    // Test 3: allocating past capacity fails cleanly, without disturbing any existing mapping.
    // -----------------------------------------------------------------------------------------
    void TestCapacityExhaustion() {
        constexpr uint32_t kCapacity = 4;
        geometry::GpuPageTable table(kCapacity);

        std::unordered_map<uint64_t, uint64_t> offsets;
        for (uint32_t i = 0; i < kCapacity; ++i) {
            uint64_t logicalAddress = LogicalAddress(i);
            table.AllocatePage(logicalAddress);
            offsets[logicalAddress] = table.GetPhysicalOffset(logicalAddress);
        }

        uint64_t overflowLogicalAddress = LogicalAddress(kCapacity);
        uint32_t overflowResult = table.AllocatePage(overflowLogicalAddress);
        Check(overflowResult == geometry::kInvalidPhysicalPage,
            "TestCapacityExhaustion: allocating beyond capacity must fail");
        Check(!table.IsResident(overflowLogicalAddress),
            "TestCapacityExhaustion: a failed allocation must not mark the address resident");

        for (const auto& [logicalAddress, offsetBefore] : offsets) {
            Check(table.GetPhysicalOffset(logicalAddress) == offsetBefore,
                "TestCapacityExhaustion: a failed allocation must not disturb any existing mapping");
        }
    }

    // -----------------------------------------------------------------------------------------
    // Test 4: freeing a non-resident address, and re-allocating an already-resident address, are
    // both no-ops that change no state.
    // -----------------------------------------------------------------------------------------
    void TestNoOpEdgeCases() {
        constexpr uint32_t kCapacity = 4;
        geometry::GpuPageTable table(kCapacity);

        uint64_t neverAllocated = LogicalAddress(100);
        Check(!table.FreePage(neverAllocated),
            "TestNoOpEdgeCases: freeing a never-allocated logical address must return false");
        Check(!table.IsResident(neverAllocated),
            "TestNoOpEdgeCases: a never-allocated logical address must not be resident");
        Check(table.GetPhysicalOffset(neverAllocated) == geometry::kInvalidPhysicalOffset,
            "TestNoOpEdgeCases: GetPhysicalOffset on a non-resident address must return kInvalidPhysicalOffset");

        uint64_t logicalAddress = LogicalAddress(0);
        uint32_t physicalPage = table.AllocatePage(logicalAddress);
        Check(physicalPage != geometry::kInvalidPhysicalPage, "TestNoOpEdgeCases: initial allocation must succeed");
        uint64_t offsetBefore = table.GetPhysicalOffset(logicalAddress);

        // Re-allocating an already-resident address must fail and must not change its offset.
        uint32_t reallocResult = table.AllocatePage(logicalAddress);
        Check(reallocResult == geometry::kInvalidPhysicalPage,
            "TestNoOpEdgeCases: re-allocating an already-resident address must fail");
        Check(table.GetPhysicalOffset(logicalAddress) == offsetBefore,
            "TestNoOpEdgeCases: a failed re-allocation must not change the existing offset");

        // Double-freeing must fail the second time and leave resident count untouched.
        Check(table.FreePage(logicalAddress), "TestNoOpEdgeCases: first FreePage of a resident address must succeed");
        Check(!table.FreePage(logicalAddress), "TestNoOpEdgeCases: second FreePage of the same address must fail");
    }

    // -----------------------------------------------------------------------------------------
    // Test 5: SelectLeastRecentlyUsedPages must report pages oldest-touch-first, and
    // AllocatePage() must place a freshly-bound page at the most-recently-used end (so it is
    // never the very next eviction candidate while anything older is still resident).
    // -----------------------------------------------------------------------------------------
    void TestLRUOrderAfterAllocation() {
        constexpr uint32_t kCapacity = 5;
        geometry::GpuPageTable table(kCapacity);

        for (uint32_t i = 0; i < kCapacity; ++i) {
            table.AllocatePage(LogicalAddress(i));
        }

        // No page has been touched since allocation: LRU order must exactly mirror allocation
        // order, oldest (index 0) first.
        std::vector<uint64_t> lruOrder = table.SelectLeastRecentlyUsedPages(kCapacity);
        Check(lruOrder.size() == kCapacity, "TestLRUOrderAfterAllocation: must report every resident page");
        for (uint32_t i = 0; i < kCapacity; ++i) {
            Check(lruOrder[i] == LogicalAddress(i),
                "TestLRUOrderAfterAllocation: LRU order must match allocation order when nothing was touched");
        }

        // Asking for fewer than the full resident count must return exactly that many, still
        // starting from the least-recently-used end.
        std::vector<uint64_t> topTwo = table.SelectLeastRecentlyUsedPages(2);
        Check(topTwo.size() == 2, "TestLRUOrderAfterAllocation: maxCount must cap the returned list size");
        Check(topTwo[0] == LogicalAddress(0) && topTwo[1] == LogicalAddress(1),
            "TestLRUOrderAfterAllocation: a capped query must still return the truly-oldest entries first");
    }

    // -----------------------------------------------------------------------------------------
    // Test 6 (the core LRU requirement): TouchPage() must move a page to the most-recently-used
    // end, so it is no longer selected as an eviction candidate ahead of pages that were not
    // touched, even though it was allocated first.
    // -----------------------------------------------------------------------------------------
    void TestTouchPageReordersEvictionCandidates() {
        constexpr uint32_t kCapacity = 4;
        geometry::GpuPageTable table(kCapacity);

        for (uint32_t i = 0; i < kCapacity; ++i) {
            table.AllocatePage(LogicalAddress(i));
        }
        // Allocation order (oldest -> newest): 0, 1, 2, 3.

        // Touch the oldest page (0): it must jump to the most-recently-used end, making 1 the new
        // least-recently-used candidate.
        table.TouchPage(LogicalAddress(0));

        std::vector<uint64_t> lruOrder = table.SelectLeastRecentlyUsedPages(kCapacity);
        std::vector<uint64_t> expectedOrder = { LogicalAddress(1), LogicalAddress(2), LogicalAddress(3), LogicalAddress(0) };
        Check(lruOrder == expectedOrder,
            "TestTouchPageReordersEvictionCandidates: touching the oldest page must make it the newest");

        // Touching an already-most-recently-used page must be a no-op on the ordering.
        table.TouchPage(LogicalAddress(0));
        Check(table.SelectLeastRecentlyUsedPages(kCapacity) == expectedOrder,
            "TestTouchPageReordersEvictionCandidates: re-touching the most-recently-used page must not change order");

        // Touching a non-resident address must be a silent no-op.
        table.TouchPage(LogicalAddress(100));
        Check(table.SelectLeastRecentlyUsedPages(kCapacity) == expectedOrder,
            "TestTouchPageReordersEvictionCandidates: touching a non-resident address must not change order");

        Check(table.SelectLeastRecentlyUsedPages(1)[0] == LogicalAddress(1),
            "TestTouchPageReordersEvictionCandidates: the single best eviction candidate must be the least recently touched page");
    }

    // -----------------------------------------------------------------------------------------
    // Test 7: freeing a page must remove it from the LRU list without disturbing the recency
    // order of any other resident page, and a page reallocated into the freed slot must reappear
    // at the most-recently-used end (LRU state must not leak across a free/realloc cycle).
    // -----------------------------------------------------------------------------------------
    void TestLRUSurvivesFreeAndReallocate() {
        constexpr uint32_t kCapacity = 4;
        geometry::GpuPageTable table(kCapacity);

        for (uint32_t i = 0; i < kCapacity; ++i) {
            table.AllocatePage(LogicalAddress(i));
        }
        // Allocation order (oldest -> newest): 0, 1, 2, 3.

        // Free the second-oldest page (1); it must vanish from the LRU list entirely, and the
        // relative order of every other page must be unaffected.
        table.FreePage(LogicalAddress(1));
        std::vector<uint64_t> lruOrder = table.SelectLeastRecentlyUsedPages(kCapacity);
        std::vector<uint64_t> expectedOrder = { LogicalAddress(0), LogicalAddress(2), LogicalAddress(3) };
        Check(lruOrder == expectedOrder,
            "TestLRUSurvivesFreeAndReallocate: freeing a page must remove it from the LRU list without reordering the rest");

        // Reallocate into the freed slot with a brand-new logical address: it must land at the
        // most-recently-used end, not inherit the freed page's old position.
        uint64_t newLogicalAddress = LogicalAddress(kCapacity + 1);
        table.AllocatePage(newLogicalAddress);
        lruOrder = table.SelectLeastRecentlyUsedPages(kCapacity);
        std::vector<uint64_t> expectedOrderAfterRealloc = { LogicalAddress(0), LogicalAddress(2), LogicalAddress(3), newLogicalAddress };
        Check(lruOrder == expectedOrderAfterRealloc,
            "TestLRUSurvivesFreeAndReallocate: a page reallocated into a freed slot must start as most-recently-used");

        // Freeing every remaining page must leave the LRU list empty.
        table.FreePage(LogicalAddress(0));
        table.FreePage(LogicalAddress(2));
        table.FreePage(LogicalAddress(3));
        table.FreePage(newLogicalAddress);
        Check(table.SelectLeastRecentlyUsedPages(kCapacity).empty(),
            "TestLRUSurvivesFreeAndReallocate: the LRU list must be empty once every page is freed");
    }

}

int main() {
    TestSequentialAllocation();
    TestFreeAndReallocateDoesNotDisturbNeighbors();
    TestCapacityExhaustion();
    TestNoOpEdgeCases();
    TestLRUOrderAfterAllocation();
    TestTouchPageReordersEvictionCandidates();
    TestLRUSurvivesFreeAndReallocate();

    if (g_FailCount == 0) {
        std::cout << "[GpuPageTableTests] All checks passed." << std::endl;
        return 0;
    }

    std::cerr << "[GpuPageTableTests] " << g_FailCount << " check(s) failed." << std::endl;
    return 1;
}
