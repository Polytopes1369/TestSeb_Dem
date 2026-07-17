#include "renderer/streaming/VirtualTextureManager.h"
#include <iostream>
#include <string>
#include <vector>
#include <cassert>

namespace renderer {

    class VirtualTextureTests {
    public:
        static int Run() {
            int failCount = 0;

            auto Check = [&](bool condition, const std::string& desc) {
                if (!condition) {
                    std::cerr << "[FAIL] " << desc << std::endl;
                    failCount++;
                }
            };

            // Test 1: Key Packing/Unpacking
            {
                uint32_t x = 142;
                uint32_t y = 512;
                uint32_t mip = 5;
                uint32_t key = VirtualTextureManager::PackPageKey(x, y, mip);
                
                uint32_t ux, uy, umip;
                VirtualTextureManager::UnpackPageKey(key, ux, uy, umip);
                Check(ux == x && uy == y && umip == mip, "Key packing round-trip mismatch");
            }

            // Test 2: Grid calculations & Initialization
            {
                VirtualTextureManager mgr;
                VirtualTextureConfig config;
                config.virtualWidth = 16384;
                config.virtualHeight = 16384;
                config.tileSize = 128;
                config.borderSize = 4;
                config.physicalPageCapacity = 8;

                // Configure bookkeeping manually to bypass Vulkan device calls
                mgr.m_Config = config;
                mgr.m_GridWidth = config.virtualWidth / config.tileSize;
                mgr.m_GridHeight = config.virtualHeight / config.tileSize;
                mgr.m_MipCount = static_cast<uint32_t>(std::log2(mgr.m_GridWidth)) + 1;
                mgr.BuildMipOffsets();
                mgr.ClearPageTable(VK_NULL_HANDLE);

                Check(mgr.m_GridWidth == 128, "Grid width must be 128");
                Check(mgr.m_GridHeight == 128, "Grid height must be 128");
                Check(mgr.m_MipCount == 8, "Mip count must be 8");
                Check(mgr.m_NextUnusedPhysicalSlot == 1, "Next unused physical slot must be 1 (slot 0 reserved for root)");
                Check(mgr.m_ResidentCount == 0, "Initial dynamic resident count must be 0");
            }

            // Test 3: Root Page Fallback Propagation
            {
                VirtualTextureManager mgr;
                VirtualTextureConfig config;
                config.virtualWidth = 1024; // 8x8 pages at mip 0
                config.tileSize = 128;
                config.physicalPageCapacity = 4;
                
                mgr.m_Config = config;
                mgr.m_GridWidth = config.virtualWidth / config.tileSize;
                mgr.m_GridHeight = config.virtualHeight / config.tileSize;
                mgr.m_MipCount = static_cast<uint32_t>(std::log2(mgr.m_GridWidth)) + 1; // log2(8) + 1 = 4 mips (0, 1, 2, 3)
                mgr.BuildMipOffsets();
                mgr.ClearPageTable(VK_NULL_HANDLE);

                // Mips:
                // Mip 3: 1x1
                // Mip 2: 2x2
                // Mip 1: 4x4
                // Mip 0: 8x8

                // Initially, all pages at all mips must point to slot 0 (root page) and resident mip 3 (root level)
                for (uint32_t mip = 0; mip < mgr.m_MipCount; ++mip) {
                    uint32_t w = 1 << (mgr.m_MipCount - 1 - mip);
                    for (uint32_t y = 0; y < w; ++y) {
                        for (uint32_t x = 0; x < w; ++x) {
                            uint32_t idx = mgr.GetPageTableIndex(x, y, mip);
                            Check(mgr.m_PageTableEntries[idx].physicalPageIndex == 0, "Initial page mapping must point to slot 0");
                            Check(mgr.m_PageTableEntries[idx].residentMip == 3, "Initial resident mip must point to root mip 3");
                        }
                    }
                }
            }

            // Test 4: Page request and hierarchical down-propagation
            {
                VirtualTextureManager mgr;
                VirtualTextureConfig config;
                config.virtualWidth = 1024; // 8x8 pages
                config.tileSize = 128;
                config.physicalPageCapacity = 4;
                
                mgr.m_Config = config;
                mgr.m_GridWidth = config.virtualWidth / config.tileSize;
                mgr.m_GridHeight = config.virtualHeight / config.tileSize;
                mgr.m_MipCount = 4; // mips 0, 1, 2, 3
                mgr.BuildMipOffsets();
                mgr.ClearPageTable(VK_NULL_HANDLE);

                // Request page at mip 2: (0, 0, 2). Should allocate slot 1
                uint32_t pIdx = mgr.RequestPageResidency(VK_NULL_HANDLE, 0, 0, 2);
                Check(pIdx == 1, "First allocated page must get physical slot 1");
                Check(mgr.m_ResidentCount == 1, "Resident count must be 1 after allocation");

                // Check that (0, 0, 2) is mapped to slot 1, resident mip 2
                uint32_t idx2 = mgr.GetPageTableIndex(0, 0, 2);
                Check(mgr.m_PageTableEntries[idx2].physicalPageIndex == 1, "Page table entry at requested level must update");
                Check(mgr.m_PageTableEntries[idx2].residentMip == 2, "Page table entry resident mip must match requested mip");

                // Check propagation to descendants at mip 1 (0,0), (1,0), (0,1), (1,1)
                for (uint32_t y = 0; y < 2; ++y) {
                    for (uint32_t x = 0; x < 2; ++x) {
                        uint32_t idx = mgr.GetPageTableIndex(x, y, 1);
                        Check(mgr.m_PageTableEntries[idx].physicalPageIndex == 1, "Descendant page mapping must propagate down");
                        Check(mgr.m_PageTableEntries[idx].residentMip == 2, "Descendant resident mip must report parent level");
                    }
                }

                // Check propagation to descendants at mip 0: first 4x4 grid
                for (uint32_t y = 0; y < 4; ++y) {
                    for (uint32_t x = 0; x < 4; ++x) {
                        uint32_t idx = mgr.GetPageTableIndex(x, y, 0);
                        Check(mgr.m_PageTableEntries[idx].physicalPageIndex == 1, "Descendant page mapping must propagate to leaf levels");
                        Check(mgr.m_PageTableEntries[idx].residentMip == 2, "Descendant resident mip must report ancestor level");
                    }
                }

                // Check that other pages at mip 2 (e.g. (1, 0, 2)) still point to root (slot 0)
                uint32_t idxOther = mgr.GetPageTableIndex(1, 0, 2);
                Check(mgr.m_PageTableEntries[idxOther].physicalPageIndex == 0, "Unrelated branches must remain unaffected");
                Check(mgr.m_PageTableEntries[idxOther].residentMip == 3, "Unrelated branches must point to root mip");
            }

            // Test 5: Overwriting coarser fallbacks but keeping finer ones
            {
                VirtualTextureManager mgr;
                VirtualTextureConfig config;
                config.virtualWidth = 1024;
                config.tileSize = 128;
                config.physicalPageCapacity = 4;
                
                mgr.m_Config = config;
                mgr.m_GridWidth = config.virtualWidth / config.tileSize;
                mgr.m_GridHeight = config.virtualHeight / config.tileSize;
                mgr.m_MipCount = 4;
                mgr.BuildMipOffsets();
                mgr.ClearPageTable(VK_NULL_HANDLE);

                // 1. Request leaf page at mip 0: (0, 0, 0) -> slot 1
                uint32_t p0 = mgr.RequestPageResidency(VK_NULL_HANDLE, 0, 0, 0);
                Check(p0 == 1, "First page must get slot 1");

                // 2. Request parent page at mip 1: (0, 0, 1) -> slot 2
                uint32_t p1 = mgr.RequestPageResidency(VK_NULL_HANDLE, 0, 0, 1);
                Check(p1 == 2, "Second page must get slot 2");

                // Check that (0, 0, 1) points to slot 2, resident mip 1
                Check(mgr.m_PageTableEntries[mgr.GetPageTableIndex(0, 0, 1)].physicalPageIndex == 2, "Mip 1 must point to slot 2");
                Check(mgr.m_PageTableEntries[mgr.GetPageTableIndex(0, 0, 1)].residentMip == 1, "Mip 1 resident mip must be 1");

                // Check that (0, 0, 0) STILL points to slot 1, resident mip 0 (finer mapping is kept!)
                Check(mgr.m_PageTableEntries[mgr.GetPageTableIndex(0, 0, 0)].physicalPageIndex == 1, "Finer mapping at mip 0 must not be overwritten by parent");
                Check(mgr.m_PageTableEntries[mgr.GetPageTableIndex(0, 0, 0)].residentMip == 0, "Finer resident mip must remain 0");

                // Check that (1, 0, 0), which is child of (0,0,1) but not requested directly, points to slot 2, resident mip 1 (fallback)
                Check(mgr.m_PageTableEntries[mgr.GetPageTableIndex(1, 0, 0)].physicalPageIndex == 2, "Coarser sibling must fallback to parent slot 2");
                Check(mgr.m_PageTableEntries[mgr.GetPageTableIndex(1, 0, 0)].residentMip == 1, "Coarser sibling resident mip must report parent level");
            }

            // Test 6: LRU Eviction & Fallback Restoration
            {
                VirtualTextureManager mgr;
                VirtualTextureConfig config;
                config.virtualWidth = 1024;
                config.tileSize = 128;
                config.physicalPageCapacity = 4; // slots 0 (root), 1, 2, 3 (LRU slots)
                
                mgr.m_Config = config;
                mgr.m_GridWidth = config.virtualWidth / config.tileSize;
                mgr.m_GridHeight = config.virtualHeight / config.tileSize;
                mgr.m_MipCount = 4;
                mgr.BuildMipOffsets();
                mgr.ClearPageTable(VK_NULL_HANDLE);

                // Request 3 distinct pages at mip 0 to fill the pool (slots 1, 2, 3)
                uint32_t a = mgr.RequestPageResidency(VK_NULL_HANDLE, 0, 0, 0); // slot 1
                uint32_t b = mgr.RequestPageResidency(VK_NULL_HANDLE, 1, 0, 0); // slot 2
                uint32_t c = mgr.RequestPageResidency(VK_NULL_HANDLE, 0, 1, 0); // slot 3

                Check(a == 1 && b == 2 && c == 3, "Sequentially allocated pages must occupy slots 1, 2, 3");
                Check(mgr.m_ResidentCount == 3, "Resident count must be 3");

                // LRU oldest-first order: slot 1 (a), slot 2 (b), slot 3 (c)
                // Touch page (0, 0, 0) [slot 1] to make it newest
                mgr.TouchPage(0, 0, 0);
                // LRU oldest-first order now: slot 2 (b), slot 3 (c), slot 1 (a)

                // Request page (1, 1, 0) -> pool is full, should evict slot 2 (b)
                uint32_t d = mgr.RequestPageResidency(VK_NULL_HANDLE, 1, 1, 0);
                Check(d == 2, "Eviction must reuse slot 2");
                Check(mgr.m_ResidentCount == 3, "Resident count must stay capped at 3");

                // Verify page b (1, 0, 0) is evicted and falls back to root (slot 0, mip 3)
                uint32_t idxB = mgr.GetPageTableIndex(1, 0, 0);
                Check(mgr.m_PageTableEntries[idxB].physicalPageIndex == 0, "Evicted page must fallback to root slot 0");
                Check(mgr.m_PageTableEntries[idxB].residentMip == 3, "Evicted page resident mip must revert to root");

                // Verify page d (1, 1, 0) is resident at slot 2
                uint32_t idxD = mgr.GetPageTableIndex(1, 1, 0);
                Check(mgr.m_PageTableEntries[idxD].physicalPageIndex == 2, "New page must be bound to reused slot");
                Check(mgr.m_PageTableEntries[idxD].residentMip == 0, "New page resident mip must match requested level");
            }

            if (failCount == 0) {
                std::cout << "[VirtualTextureTests] All checks passed successfully." << std::endl;
                return 0;
            }

            std::cerr << "[VirtualTextureTests] " << failCount << " check(s) failed." << std::endl;
            return 1;
        }
    };

}

int main() {
    return renderer::VirtualTextureTests::Run();
}
