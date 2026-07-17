#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdint>
#include <vector>
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/GpuImage.h"

namespace renderer {

    // Configuration structure for the Virtual Texture System
    struct VirtualTextureConfig {
        uint32_t virtualWidth = 16384;
        uint32_t virtualHeight = 16384;
        uint32_t tileSize = 128;
        uint32_t borderSize = 4;
        uint32_t physicalPageCapacity = 256;
        VkFormat pageTableFormat = VK_FORMAT_R16G16_UINT; // VK_FORMAT_R16G16_UINT or VK_FORMAT_R8G8_UINT
    };

    // Entry in the CPU-side mirror of the Page Table
    struct PageTableEntry {
        uint16_t physicalPageIndex = 0xFFFF; // Index into the physical pool layers (0xFFFF if unmapped)
        uint8_t residentMip = 0xFF;         // The actual resident mip level this page points to (0xFF if unmapped)
    };

    // Sentinel constants
    constexpr uint32_t kInvalidPhysicalPageIndex = 0xFFFFu;
    constexpr uint32_t kInvalidLogicalPageKey = 0xFFFFFFFFu;

    class VirtualTextureManager {
        friend class VirtualTextureTests;
    public:
        VirtualTextureManager() = default;
        ~VirtualTextureManager();

        VirtualTextureManager(const VirtualTextureManager&) = delete;
        VirtualTextureManager& operator=(const VirtualTextureManager&) = delete;
        VirtualTextureManager(VirtualTextureManager&& other) noexcept;
        VirtualTextureManager& operator=(VirtualTextureManager&& other) noexcept;

        // Initializes page table texture, physical pool texture(s), samplers, and bookkeeping structures.
        // Multiple formats can be passed in physicalPoolFormats (e.g. Albedo, Normal) to create multiple synchronized pools.
        bool Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
                  VkCommandPool commandPool, VkQueue queue, const VirtualTextureConfig& config,
                  const std::vector<VkFormat>& physicalPoolFormats);

        // Safely releases all allocated Vulkan resources
        void Shutdown();

        // Requests that a page at (x, y, mip) be resident.
        // If already resident, touches it (MRU) and returns the physical page index.
        // If not resident, allocates a slot (evicting LRU if full), updates the hierarchy, and returns the physical index.
        // The caller is expected to upload the actual page data into this physical index afterwards.
        uint32_t RequestPageResidency(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t mip);

        // Touches a page at (x, y, mip) to mark it as most recently used (LRU tracking) without modifying residency.
        void TouchPage(uint32_t x, uint32_t y, uint32_t mip);

        // Uploads raw tile data from a staging buffer into a specific physical pool slice
        void UploadTileData(VkCommandBuffer cmd, uint32_t physicalPageIndex, VkBuffer srcStagingBuffer,
                            VkDeviceSize srcOffset, uint32_t poolIndex = 0);

        // Flushes CPU-side page table modifications to the GPU-resident page table texture
        void UpdatePageTableImage(VkCommandBuffer cmd);

        // Clears the GPU-resident page table texture to the unmapped sentinel values
        void ClearPageTable(VkCommandBuffer cmd);

        // Accessors
        VkImage GetPageTableImage() const { return m_PageTableImage.Image(); }
        VkImageView GetPageTableImageView() const { return m_PageTableImage.View(); }
        VkSampler GetPageTableSampler() const { return m_PageTableSampler; }

        size_t GetPhysicalPoolCount() const { return m_PhysicalPoolImages.size(); }
        VkImage GetPhysicalPoolImage(uint32_t poolIndex = 0) const { return m_PhysicalPoolImages[poolIndex].Image(); }
        VkImageView GetPhysicalPoolImageView(uint32_t poolIndex = 0) const { return m_PhysicalPoolViews[poolIndex]; }
        VkSampler GetPhysicalPoolSampler() const { return m_PhysicalPoolSampler; }

        uint32_t GetPhysicalPageCapacity() const { return m_Config.physicalPageCapacity; }
        uint32_t GetTileSizeWithBorder() const { return m_Config.tileSize + 2 * m_Config.borderSize; }
        uint32_t GetVirtualWidth() const { return m_Config.virtualWidth; }
        uint32_t GetVirtualHeight() const { return m_Config.virtualHeight; }
        uint32_t GetTileSize() const { return m_Config.tileSize; }
        uint32_t GetBorderSize() const { return m_Config.borderSize; }
        uint32_t GetMipCount() const { return m_MipCount; }
        uint32_t GetGridWidth() const { return m_GridWidth; }
        uint32_t GetGridHeight() const { return m_GridHeight; }
        uint32_t GetResidentPageCount() const { return m_ResidentCount; }

        // Packs page coordinates into a single 32-bit key for LRU tracking
        static uint32_t PackPageKey(uint32_t x, uint32_t y, uint32_t mip) {
            return (mip & 0xF) | ((x & 0x3FFF) << 4) | ((y & 0x3FFF) << 18);
        }

        // Unpacks page coordinates from a single 32-bit key
        static void UnpackPageKey(uint32_t key, uint32_t& x, uint32_t& y, uint32_t& mip) {
            mip = key & 0xF;
            x = (key >> 4) & 0x3FFF;
            y = (key >> 18) & 0x3FFF;
        }

    private:
        uint32_t AllocatePhysicalSlot(VkCommandBuffer cmd);
        void EvictOneLeastRecentlyUsed(VkCommandBuffer cmd);
        void LinkAsMostRecentlyUsed(uint32_t physicalPageIndex);
        void UnlinkFromLRUList(uint32_t physicalPageIndex);

        // Page Table propagation helpers
        void PropagatePageTable(uint32_t x, uint32_t y, uint32_t mip, uint16_t physIndex, uint8_t resMip);
        void PropagateEviction(uint32_t x, uint32_t y, uint32_t mip, uint16_t evictedPhysIndex,
                               uint16_t ancPhysIndex, uint8_t ancResMip);

        uint32_t GetPageTableIndex(uint32_t x, uint32_t y, uint32_t mip) const;
        uint32_t GetMipOffset(uint32_t mip) const;
        void BuildMipOffsets();

        // Vulkan Context
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkCommandPool m_CommandPool = VK_NULL_HANDLE;
        VkQueue m_Queue = VK_NULL_HANDLE;

        VirtualTextureConfig m_Config;

        // GPU Resources
        GpuImage m_PageTableImage;
        std::vector<GpuImage> m_PhysicalPoolImages;
        std::vector<VkImageView> m_PhysicalPoolViews;
        VkSampler m_PageTableSampler = VK_NULL_HANDLE;
        VkSampler m_PhysicalPoolSampler = VK_NULL_HANDLE;

        // Page Table staging buffer for updates
        GpuBuffer m_PageTableStagingBuffer;
        void* m_PageTableStagingMapped = nullptr;

        // Bookkeeping
        uint32_t m_MipCount = 0;
        uint32_t m_GridWidth = 0;  // Number of pages wide at mip 0
        uint32_t m_GridHeight = 0; // Number of pages high at mip 0
        std::vector<PageTableEntry> m_PageTableEntries;
        std::vector<uint32_t> m_MipOffsets;
        bool m_PageTableDirty = false;

        // LRU list structures
        struct LRUNode {
            uint32_t logicalPageKey = kInvalidLogicalPageKey;
            uint32_t prev = kInvalidPhysicalPageIndex;
            uint32_t next = kInvalidPhysicalPageIndex;
        };

        std::vector<LRUNode> m_LRUNodes;
        std::vector<uint32_t> m_PhysicalSlotToLogicalKey;
        std::vector<uint32_t> m_FreePhysicalSlots;
        uint32_t m_NextUnusedPhysicalSlot = 0;
        uint32_t m_LRUMostRecentPage = kInvalidPhysicalPageIndex;
        uint32_t m_LRULeastRecentPage = kInvalidPhysicalPageIndex;
        uint32_t m_ResidentCount = 0;
    };

}
