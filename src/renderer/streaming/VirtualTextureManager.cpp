#include "renderer/streaming/VirtualTextureManager.h"
#include "renderer/vulkan/VulkanUtils.h"
#include "core/Logger.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <format>

namespace renderer {

    VirtualTextureManager::~VirtualTextureManager() {
        Shutdown();
    }

    VirtualTextureManager::VirtualTextureManager(VirtualTextureManager&& other) noexcept
        : m_Device(other.m_Device)
        , m_Allocator(other.m_Allocator)
        , m_CommandPool(other.m_CommandPool)
        , m_Queue(other.m_Queue)
        , m_Config(other.m_Config)
        , m_PageTableImage(std::move(other.m_PageTableImage))
        , m_PhysicalPoolImages(std::move(other.m_PhysicalPoolImages))
        , m_PhysicalPoolViews(std::move(other.m_PhysicalPoolViews))
        , m_PhysicalPoolFormats(std::move(other.m_PhysicalPoolFormats))
        , m_PhysicalPoolLayerViews(std::move(other.m_PhysicalPoolLayerViews))
        , m_PageTableSampler(other.m_PageTableSampler)
        , m_PhysicalPoolSampler(other.m_PhysicalPoolSampler)
        , m_PageTableStagingBuffer(std::move(other.m_PageTableStagingBuffer))
        , m_PageTableStagingMapped(other.m_PageTableStagingMapped)
        , m_MipCount(other.m_MipCount)
        , m_GridWidth(other.m_GridWidth)
        , m_GridHeight(other.m_GridHeight)
        , m_PageTableEntries(std::move(other.m_PageTableEntries))
        , m_MipOffsets(std::move(other.m_MipOffsets))
        , m_PageTableDirty(other.m_PageTableDirty)
        , m_LRUNodes(std::move(other.m_LRUNodes))
        , m_PhysicalSlotToLogicalKey(std::move(other.m_PhysicalSlotToLogicalKey))
        , m_FreePhysicalSlots(std::move(other.m_FreePhysicalSlots))
        , m_NextUnusedPhysicalSlot(other.m_NextUnusedPhysicalSlot)
        , m_LRUMostRecentPage(other.m_LRUMostRecentPage)
        , m_LRULeastRecentPage(other.m_LRULeastRecentPage)
        , m_ResidentCount(other.m_ResidentCount) {
        other.m_Device = VK_NULL_HANDLE;
        other.m_Allocator = VK_NULL_HANDLE;
        other.m_CommandPool = VK_NULL_HANDLE;
        other.m_Queue = VK_NULL_HANDLE;
        other.m_PageTableSampler = VK_NULL_HANDLE;
        other.m_PhysicalPoolSampler = VK_NULL_HANDLE;
        other.m_PageTableStagingMapped = nullptr;
        other.m_MipCount = 0;
        other.m_GridWidth = 0;
        other.m_GridHeight = 0;
        other.m_PageTableDirty = false;
        other.m_NextUnusedPhysicalSlot = 0;
        other.m_LRUMostRecentPage = kInvalidPhysicalPageIndex;
        other.m_LRULeastRecentPage = kInvalidPhysicalPageIndex;
        other.m_ResidentCount = 0;
    }

    VirtualTextureManager& VirtualTextureManager::operator=(VirtualTextureManager&& other) noexcept {
        if (this != &other) {
            Shutdown();
            m_Device = other.m_Device;
            m_Allocator = other.m_Allocator;
            m_CommandPool = other.m_CommandPool;
            m_Queue = other.m_Queue;
            m_Config = other.m_Config;
            m_PageTableImage = std::move(other.m_PageTableImage);
            m_PhysicalPoolImages = std::move(other.m_PhysicalPoolImages);
            m_PhysicalPoolViews = std::move(other.m_PhysicalPoolViews);
            m_PhysicalPoolFormats = std::move(other.m_PhysicalPoolFormats);
            m_PhysicalPoolLayerViews = std::move(other.m_PhysicalPoolLayerViews);
            m_PageTableSampler = other.m_PageTableSampler;
            m_PhysicalPoolSampler = other.m_PhysicalPoolSampler;
            m_PageTableStagingBuffer = std::move(other.m_PageTableStagingBuffer);
            m_PageTableStagingMapped = other.m_PageTableStagingMapped;
            m_MipCount = other.m_MipCount;
            m_GridWidth = other.m_GridWidth;
            m_GridHeight = other.m_GridHeight;
            m_PageTableEntries = std::move(other.m_PageTableEntries);
            m_MipOffsets = std::move(other.m_MipOffsets);
            m_PageTableDirty = other.m_PageTableDirty;
            m_LRUNodes = std::move(other.m_LRUNodes);
            m_PhysicalSlotToLogicalKey = std::move(other.m_PhysicalSlotToLogicalKey);
            m_FreePhysicalSlots = std::move(other.m_FreePhysicalSlots);
            m_NextUnusedPhysicalSlot = other.m_NextUnusedPhysicalSlot;
            m_LRUMostRecentPage = other.m_LRUMostRecentPage;
            m_LRULeastRecentPage = other.m_LRULeastRecentPage;
            m_ResidentCount = other.m_ResidentCount;

            other.m_Device = VK_NULL_HANDLE;
            other.m_Allocator = VK_NULL_HANDLE;
            other.m_CommandPool = VK_NULL_HANDLE;
            other.m_Queue = VK_NULL_HANDLE;
            other.m_PageTableSampler = VK_NULL_HANDLE;
            other.m_PhysicalPoolSampler = VK_NULL_HANDLE;
            other.m_PageTableStagingMapped = nullptr;
            other.m_MipCount = 0;
            other.m_GridWidth = 0;
            other.m_GridHeight = 0;
            other.m_PageTableDirty = false;
            other.m_NextUnusedPhysicalSlot = 0;
            other.m_LRUMostRecentPage = kInvalidPhysicalPageIndex;
            other.m_LRULeastRecentPage = kInvalidPhysicalPageIndex;
            other.m_ResidentCount = 0;
        }
        return *this;
    }

    bool VirtualTextureManager::Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
                                     VkCommandPool commandPool, VkQueue queue, const VirtualTextureConfig& config,
                                     const std::vector<VkFormat>& physicalPoolFormats) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_CommandPool = commandPool;
        m_Queue = queue;
        m_Config = config;

        // 1. Grid calculations
        if (config.virtualWidth % config.tileSize != 0 || config.virtualHeight % config.tileSize != 0) {
            LOG_ERROR("[VirtualTextureManager] Virtual texture dimensions must be multiples of tile size.");
            return false;
        }

        m_GridWidth = config.virtualWidth / config.tileSize;
        m_GridHeight = config.virtualHeight / config.tileSize;
        m_MipCount = static_cast<uint32_t>(std::log2(std::max(m_GridWidth, m_GridHeight))) + 1;

        BuildMipOffsets();

        // Initialize LRU systems
        m_LRUNodes.assign(config.physicalPageCapacity, LRUNode{});
        m_PhysicalSlotToLogicalKey.assign(config.physicalPageCapacity, kInvalidLogicalPageKey);
        m_FreePhysicalSlots.clear();
        m_NextUnusedPhysicalSlot = 0;
        m_LRUMostRecentPage = kInvalidPhysicalPageIndex;
        m_LRULeastRecentPage = kInvalidPhysicalPageIndex;
        m_ResidentCount = 0;

        // 2. Create Page Table Sampler (Nearest filtering, nearest mipmapping)
        VkSamplerCreateInfo ptSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        ptSamplerInfo.magFilter = VK_FILTER_NEAREST;
        ptSamplerInfo.minFilter = VK_FILTER_NEAREST;
        ptSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ptSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ptSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ptSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ptSamplerInfo.minLod = 0.0f;
        ptSamplerInfo.maxLod = static_cast<float>(m_MipCount - 1);
        VK_CHECK(vkCreateSampler(m_Device, &ptSamplerInfo, nullptr, &m_PageTableSampler));

        // 3. Create Physical Pool Sampler (Bilinear/Trilinear filtering, clamp to edge)
        VkSamplerCreateInfo poolSamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        poolSamplerInfo.magFilter = VK_FILTER_LINEAR;
        poolSamplerInfo.minFilter = VK_FILTER_LINEAR;
        poolSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        poolSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        poolSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        poolSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        poolSamplerInfo.minLod = 0.0f;
        poolSamplerInfo.maxLod = 0.0f; // Pool textures do not hold mips on their own slices
        VK_CHECK(vkCreateSampler(m_Device, &poolSamplerInfo, nullptr, &m_PhysicalPoolSampler));

        // 4. Create Page Table image
        VkImageCreateInfo ptImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ptImageInfo.imageType = VK_IMAGE_TYPE_2D;
        ptImageInfo.format = config.pageTableFormat;
        ptImageInfo.extent = { m_GridWidth, m_GridHeight, 1 };
        ptImageInfo.mipLevels = m_MipCount;
        ptImageInfo.arrayLayers = 1;
        ptImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        ptImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        ptImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ptImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ptImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        m_PageTableImage.Create(allocator, device, ptImageInfo, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_ASPECT_COLOR_BIT);

        // 5. Allocate Page Table Staging Buffer
        uint32_t bytesPerTexel = (config.pageTableFormat == VK_FORMAT_R16G16_UINT) ? 4 : 2;
        VkDeviceSize stagingSize = m_PageTableEntries.size() * bytesPerTexel;
        m_PageTableStagingBuffer.Create(allocator, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
        m_PageTableStagingMapped = m_PageTableStagingBuffer.MappedData();

        // 6. Create Physical Pool images (2D Array)
        // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT is required alongside TRANSFER_DST/SAMPLED because
        // renderer::VirtualTextureRenderPass (Step 3, RVT) writes tiles directly via Dynamic
        // Rendering (a color attachment), not only via UploadTileData's buffer-to-image copy.
        uint32_t tileSizeWithBorder = GetTileSizeWithBorder();
        m_PhysicalPoolFormats = physicalPoolFormats;
        m_PhysicalPoolImages.resize(physicalPoolFormats.size());
        m_PhysicalPoolViews.resize(physicalPoolFormats.size());
        m_PhysicalPoolLayerViews.resize(physicalPoolFormats.size());
        for (size_t i = 0; i < physicalPoolFormats.size(); ++i) {
            VkImageCreateInfo poolImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            poolImageInfo.imageType = VK_IMAGE_TYPE_2D;
            poolImageInfo.format = physicalPoolFormats[i];
            poolImageInfo.extent = { tileSizeWithBorder, tileSizeWithBorder, 1 };
            poolImageInfo.mipLevels = 1;
            poolImageInfo.arrayLayers = config.physicalPageCapacity;
            poolImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            poolImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            poolImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            poolImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            poolImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            // Create image without a default view
            m_PhysicalPoolImages[i].Create(allocator, device, poolImageInfo, VMA_MEMORY_USAGE_GPU_ONLY, 0);

            // Create the custom view of type VK_IMAGE_VIEW_TYPE_2D_ARRAY manually.
            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_PhysicalPoolImages[i].Image();
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            viewInfo.format = physicalPoolFormats[i];
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = config.physicalPageCapacity;

            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_PhysicalPoolViews[i]));

            // Per-layer VK_IMAGE_VIEW_TYPE_2D views: one per physical page slot, used as
            // VkRenderingAttachmentInfo color attachment targets by VirtualTextureRenderPass (a
            // VK_IMAGE_VIEW_TYPE_2D_ARRAY view cannot be bound as a single-layer color attachment).
            m_PhysicalPoolLayerViews[i].resize(config.physicalPageCapacity);
            for (uint32_t layer = 0; layer < config.physicalPageCapacity; ++layer) {
                VkImageViewCreateInfo layerViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
                layerViewInfo.image = m_PhysicalPoolImages[i].Image();
                layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                layerViewInfo.format = physicalPoolFormats[i];
                layerViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, layer, 1 };
                VK_CHECK(vkCreateImageView(m_Device, &layerViewInfo, nullptr, &m_PhysicalPoolLayerViews[i][layer]));
            }
        }

        // Initialize layouts via a one-shot command
        VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            ClearPageTable(cmd);

            // Clear every physical pool to opaque white BEFORE transitioning to SHADER_READ_ONLY:
            // slot 0 (the root page) is permanently marked resident by ClearPageTable() above from
            // the very first frame (see that function's own comment), and every other page-table
            // entry initially falls back to it via PropagatePageTable -- so a consumer shader can
            // legally sample ANY physical pool slot before renderer::VirtualTextureRenderPass or
            // renderer::VirtualTextureStreamingCoordinator has ever written real content into it.
            // Leaving that memory at its post-vmaCreateImage UNDEFINED contents would make that
            // sample read undefined/garbage data (not a Vulkan validation error, but a real visual
            // corruption risk the moment this manager is wired into a live material shader). White
            // is chosen -- not black or magenta -- so a "multiply the procedural albedo by the VT
            // sample" integration (see ClusterResolve.comp) is a no-op until real content streams
            // in, exactly like Unreal Engine 5.8's own RVT fallback-to-neutral convention.
            for (size_t i = 0; i < m_PhysicalPoolImages.size(); ++i) {
                VkImageMemoryBarrier2 toClearDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                toClearDst.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                toClearDst.srcAccessMask = 0;
                toClearDst.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                toClearDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toClearDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                toClearDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toClearDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toClearDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toClearDst.image = m_PhysicalPoolImages[i].Image();
                toClearDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, config.physicalPageCapacity };

                VkDependencyInfo toClearDstDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                toClearDstDep.imageMemoryBarrierCount = 1;
                toClearDstDep.pImageMemoryBarriers = &toClearDst;
                vkCmdPipelineBarrier2(cmd, &toClearDstDep);

                VkClearColorValue white{};
                white.float32[0] = 1.0f; white.float32[1] = 1.0f; white.float32[2] = 1.0f; white.float32[3] = 1.0f;
                VkImageSubresourceRange clearRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, config.physicalPageCapacity };
                vkCmdClearColorImage(cmd, m_PhysicalPoolImages[i].Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &clearRange);
            }

            // Transition physical pools
            for (size_t i = 0; i < m_PhysicalPoolImages.size(); ++i) {
                VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = m_PhysicalPoolImages[i].Image();
                barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, config.physicalPageCapacity };

                VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd, &dep);
            }
        });

        LOG_INFO(std::format("[VirtualTextureManager] Initialized: space={}x{}, grid={}x{}, {} mip levels, {} physical pages.",
            config.virtualWidth, config.virtualHeight, m_GridWidth, m_GridHeight, m_MipCount, config.physicalPageCapacity));

        return true;
    }

    void VirtualTextureManager::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            for (VkImageView view : m_PhysicalPoolViews) {
                if (view != VK_NULL_HANDLE) {
                    vkDestroyImageView(m_Device, view, nullptr);
                }
            }
            m_PhysicalPoolViews.clear();

            for (auto& poolLayerViews : m_PhysicalPoolLayerViews) {
                for (VkImageView view : poolLayerViews) {
                    if (view != VK_NULL_HANDLE) {
                        vkDestroyImageView(m_Device, view, nullptr);
                    }
                }
            }
            m_PhysicalPoolLayerViews.clear();

            if (m_PageTableSampler != VK_NULL_HANDLE) {
                vkDestroySampler(m_Device, m_PageTableSampler, nullptr);
                m_PageTableSampler = VK_NULL_HANDLE;
            }
            if (m_PhysicalPoolSampler != VK_NULL_HANDLE) {
                vkDestroySampler(m_Device, m_PhysicalPoolSampler, nullptr);
                m_PhysicalPoolSampler = VK_NULL_HANDLE;
            }
        }

        m_PageTableImage.Destroy();
        for (auto& poolImage : m_PhysicalPoolImages) {
            poolImage.Destroy();
        }
        m_PhysicalPoolImages.clear();
        m_PhysicalPoolFormats.clear();

        m_PageTableStagingBuffer.Destroy();
        m_PageTableStagingMapped = nullptr;

        m_Device = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
        m_CommandPool = VK_NULL_HANDLE;
        m_Queue = VK_NULL_HANDLE;
    }

    uint32_t VirtualTextureManager::RequestPageResidency(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t mip) {
        uint32_t index = GetPageTableIndex(x, y, mip);
        
        // If already resident, touch to refresh LRU and return physical index
        if (m_PageTableEntries[index].residentMip == mip && m_PageTableEntries[index].physicalPageIndex != 0xFFFF) {
            uint32_t physIndex = m_PageTableEntries[index].physicalPageIndex;
            TouchPage(x, y, mip);
            return physIndex;
        }

        // Allocate a new physical slot (might cause eviction)
        uint32_t physIndex = AllocatePhysicalSlot(cmd);

        // Bind the page to this physical slot
        uint32_t key = PackPageKey(x, y, mip);
        m_PhysicalSlotToLogicalKey[physIndex] = key;
        m_LRUNodes[physIndex].logicalPageKey = key;
        LinkAsMostRecentlyUsed(physIndex);
        m_ResidentCount++;

        // Propagate the resident mapping down to descendants in finer mip levels
        PropagatePageTable(x, y, mip, static_cast<uint16_t>(physIndex), static_cast<uint8_t>(mip));

        return physIndex;
    }

    void VirtualTextureManager::TouchPage(uint32_t x, uint32_t y, uint32_t mip) {
        uint32_t index = GetPageTableIndex(x, y, mip);
        if (m_PageTableEntries[index].residentMip == mip) {
            uint16_t physIndex = m_PageTableEntries[index].physicalPageIndex;
            if (physIndex != 0xFFFF && physIndex != 0) { // Keep root page (slot 0) permanent and outside LRU
                UnlinkFromLRUList(physIndex);
                LinkAsMostRecentlyUsed(physIndex);
            }
        }
    }

    void VirtualTextureManager::UploadTileData(VkCommandBuffer cmd, uint32_t physicalPageIndex, VkBuffer srcStagingBuffer,
                                              VkDeviceSize srcOffset, uint32_t poolIndex) {
        if (cmd == VK_NULL_HANDLE) return;

        if (poolIndex >= m_PhysicalPoolImages.size()) {
            throw std::runtime_error("VirtualTextureManager::UploadTileData -- poolIndex out of range");
        }

        VkImage poolImage = m_PhysicalPoolImages[poolIndex].Image();
        uint32_t tileSizeWithBorder = GetTileSizeWithBorder();

        // 1. Transition physical pool layer to TRANSFER_DST_OPTIMAL (old layout UNDEFINED since we overwrite)
        VkImageMemoryBarrier2 barrierToDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrierToDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrierToDst.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrierToDst.srcAccessMask = 0;
        barrierToDst.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrierToDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrierToDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrierToDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrierToDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToDst.image = poolImage;
        barrierToDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, physicalPageIndex, 1 };

        VkDependencyInfo depToDst{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depToDst.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToDst.imageMemoryBarrierCount = 1;
        depToDst.pImageMemoryBarriers = &barrierToDst;
        vkCmdPipelineBarrier2(cmd, &depToDst);

        // 2. Copy buffer to image layer
        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = srcOffset;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, physicalPageIndex, 1 };
        copyRegion.imageOffset = { 0, 0, 0 };
        copyRegion.imageExtent = { tileSizeWithBorder, tileSizeWithBorder, 1 };

        vkCmdCopyBufferToImage(cmd, srcStagingBuffer, poolImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        // 3. Transition back to SHADER_READ_ONLY_OPTIMAL
        VkImageMemoryBarrier2 barrierToShader{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrierToShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrierToShader.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrierToShader.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrierToShader.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrierToShader.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrierToShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrierToShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrierToShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToShader.image = poolImage;
        barrierToShader.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, physicalPageIndex, 1 };

        VkDependencyInfo depToShader{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depToShader.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToShader.imageMemoryBarrierCount = 1;
        depToShader.pImageMemoryBarriers = &barrierToShader;
        vkCmdPipelineBarrier2(cmd, &depToShader);
    }

    void VirtualTextureManager::UpdatePageTableImage(VkCommandBuffer cmd) {
        if (!m_PageTableDirty) return;

        // 1. Copy CPU page table entries to mapped staging buffer
        if (m_PageTableStagingMapped != nullptr) {
            if (m_Config.pageTableFormat == VK_FORMAT_R16G16_UINT) {
                uint16_t* dst = static_cast<uint16_t*>(m_PageTableStagingMapped);
                for (const auto& entry : m_PageTableEntries) {
                    *dst++ = entry.physicalPageIndex;
                    *dst++ = static_cast<uint16_t>(entry.residentMip);
                }
            } else {
                uint8_t* dst = static_cast<uint8_t*>(m_PageTableStagingMapped);
                for (const auto& entry : m_PageTableEntries) {
                    *dst++ = static_cast<uint8_t>(entry.physicalPageIndex);
                    *dst++ = entry.residentMip;
                }
            }
        }

        if (cmd == VK_NULL_HANDLE) {
            m_PageTableDirty = false;
            return;
        }

        // 2. Transition Page Table Image layout to TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier2 barrierToDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrierToDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrierToDst.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrierToDst.srcAccessMask = 0;
        barrierToDst.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrierToDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrierToDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrierToDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrierToDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToDst.image = m_PageTableImage.Image();
        barrierToDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, m_MipCount, 0, 1 };

        VkDependencyInfo depToDst{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depToDst.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToDst.imageMemoryBarrierCount = 1;
        depToDst.pImageMemoryBarriers = &barrierToDst;
        vkCmdPipelineBarrier2(cmd, &depToDst);

        // 3. Queue copy commands for each mip level
        uint32_t bytesPerTexel = (m_Config.pageTableFormat == VK_FORMAT_R16G16_UINT) ? 4 : 2;
        std::vector<VkBufferImageCopy> copyRegions(m_MipCount);
        for (uint32_t level = 0; level < m_MipCount; ++level) {
            uint32_t w = std::max(1u, m_GridWidth >> level);
            uint32_t h = std::max(1u, m_GridHeight >> level);

            copyRegions[level].bufferOffset = m_MipOffsets[level] * bytesPerTexel;
            copyRegions[level].bufferRowLength = 0;
            copyRegions[level].bufferImageHeight = 0;
            copyRegions[level].imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 };
            copyRegions[level].imageOffset = { 0, 0, 0 };
            copyRegions[level].imageExtent = { w, h, 1 };
        }

        vkCmdCopyBufferToImage(cmd, m_PageTableStagingBuffer.Handle(), m_PageTableImage.Image(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(copyRegions.size()),
                               copyRegions.data());

        // 4. Transition Page Table Image layout to SHADER_READ_ONLY_OPTIMAL
        VkImageMemoryBarrier2 barrierToShader{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrierToShader.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrierToShader.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrierToShader.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrierToShader.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrierToShader.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrierToShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrierToShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrierToShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierToShader.image = m_PageTableImage.Image();
        barrierToShader.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, m_MipCount, 0, 1 };

        VkDependencyInfo depToShader{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depToShader.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToShader.imageMemoryBarrierCount = 1;
        depToShader.pImageMemoryBarriers = &barrierToShader;
        vkCmdPipelineBarrier2(cmd, &depToShader);

        m_PageTableDirty = false;
    }

    void VirtualTextureManager::ClearPageTable(VkCommandBuffer cmd) {
        // Reset CPU Mirror
        m_PageTableEntries.assign(m_PageTableEntries.size(), PageTableEntry{ 0xFFFF, 0xFF });
        m_ResidentCount = 0;
        m_NextUnusedPhysicalSlot = 1; // Slot 0 is reserved for root page
        m_FreePhysicalSlots.clear();
        m_LRUMostRecentPage = kInvalidPhysicalPageIndex;
        m_LRULeastRecentPage = kInvalidPhysicalPageIndex;

        m_LRUNodes.assign(m_Config.physicalPageCapacity, LRUNode{});
        m_PhysicalSlotToLogicalKey.assign(m_Config.physicalPageCapacity, kInvalidLogicalPageKey);

        // Permanently bind root page (0, 0) at coarsest mip to physical slot 0
        uint32_t rootMip = m_MipCount - 1;
        PropagatePageTable(0, 0, rootMip, 0, static_cast<uint8_t>(rootMip));
        m_PhysicalSlotToLogicalKey[0] = PackPageKey(0, 0, rootMip);
        m_LRUNodes[0].logicalPageKey = PackPageKey(0, 0, rootMip);
        // Slot 0 is deliberately never linked in the LRU list, keeping it permanently cached.

        m_PageTableDirty = true;
        UpdatePageTableImage(cmd);
    }

    uint32_t VirtualTextureManager::AllocatePhysicalSlot(VkCommandBuffer cmd) {
        // 1. Try free list
        if (!m_FreePhysicalSlots.empty()) {
            uint32_t slot = m_FreePhysicalSlots.back();
            m_FreePhysicalSlots.pop_back();
            return slot;
        }

        // 2. Try expanding pool up to capacity
        if (m_NextUnusedPhysicalSlot < m_Config.physicalPageCapacity) {
            return m_NextUnusedPhysicalSlot++;
        }

        // 3. Pool is fully allocated: evict the Least Recently Used (LRU) slot
        uint32_t slotToEvict = m_LRULeastRecentPage;
        if (slotToEvict == kInvalidPhysicalPageIndex) {
            throw std::runtime_error("VirtualTextureManager: Physical pool at capacity, but eviction candidate not found.");
        }

        uint32_t keyToEvict = m_PhysicalSlotToLogicalKey[slotToEvict];
        uint32_t ex, ey, emip;
        UnpackPageKey(keyToEvict, ex, ey, emip);

        // Evict physical slot representation
        UnlinkFromLRUList(slotToEvict);
        m_PhysicalSlotToLogicalKey[slotToEvict] = kInvalidLogicalPageKey;
        m_LRUNodes[slotToEvict].logicalPageKey = kInvalidLogicalPageKey;
        m_ResidentCount--;

        // Determine closest resident parent mapping to restore fallbacks (always exists due to root page at slot 0)
        uint32_t parentIndex = GetPageTableIndex(ex / 2, ey / 2, emip + 1);
        uint16_t ancPhysIndex = m_PageTableEntries[parentIndex].physicalPageIndex;
        uint8_t ancResMip = m_PageTableEntries[parentIndex].residentMip;

        // Propagate parent fallback restore throughout the subtree
        PropagateEviction(ex, ey, emip, static_cast<uint16_t>(slotToEvict), ancPhysIndex, ancResMip);

        return slotToEvict;
    }

    void VirtualTextureManager::LinkAsMostRecentlyUsed(uint32_t physicalPageIndex) {
        LRUNode& node = m_LRUNodes[physicalPageIndex];
        node.prev = kInvalidPhysicalPageIndex;
        node.next = m_LRUMostRecentPage;

        if (m_LRUMostRecentPage != kInvalidPhysicalPageIndex) {
            m_LRUNodes[m_LRUMostRecentPage].prev = physicalPageIndex;
        }
        m_LRUMostRecentPage = physicalPageIndex;

        if (m_LRULeastRecentPage == kInvalidPhysicalPageIndex) {
            m_LRULeastRecentPage = physicalPageIndex;
        }
    }

    void VirtualTextureManager::UnlinkFromLRUList(uint32_t physicalPageIndex) {
        LRUNode& node = m_LRUNodes[physicalPageIndex];

        // Ensure node is currently linked
        if (node.logicalPageKey == kInvalidLogicalPageKey && m_LRUMostRecentPage != physicalPageIndex) {
            return;
        }

        if (node.prev != kInvalidPhysicalPageIndex) {
            m_LRUNodes[node.prev].next = node.next;
        } else {
            m_LRUMostRecentPage = node.next;
        }

        if (node.next != kInvalidPhysicalPageIndex) {
            m_LRUNodes[node.next].prev = node.prev;
        } else {
            m_LRULeastRecentPage = node.prev;
        }

        node.prev = kInvalidPhysicalPageIndex;
        node.next = kInvalidPhysicalPageIndex;
    }

    void VirtualTextureManager::PropagatePageTable(uint32_t x, uint32_t y, uint32_t mip, uint16_t physIndex, uint8_t resMip) {
        uint32_t index = GetPageTableIndex(x, y, mip);
        m_PageTableEntries[index] = { physIndex, resMip };
        m_PageTableDirty = true;

        if (mip == 0) return;

        uint32_t childMip = mip - 1;
        uint32_t childGridW = std::max(1u, m_GridWidth >> childMip);
        uint32_t childGridH = std::max(1u, m_GridHeight >> childMip);

        for (uint32_t dy = 0; dy < 2; ++dy) {
            for (uint32_t dx = 0; dx < 2; ++dx) {
                uint32_t cx = x * 2 + dx;
                uint32_t cy = y * 2 + dy;
                if (cx < childGridW && cy < childGridH) {
                    uint32_t childIndex = GetPageTableIndex(cx, cy, childMip);
                    if (m_PageTableEntries[childIndex].residentMip > resMip) {
                        PropagatePageTable(cx, cy, childMip, physIndex, resMip);
                    }
                }
            }
        }
    }

    void VirtualTextureManager::PropagateEviction(uint32_t x, uint32_t y, uint32_t mip, uint16_t evictedPhysIndex,
                                                 uint16_t ancPhysIndex, uint8_t ancResMip) {
        uint32_t index = GetPageTableIndex(x, y, mip);
        if (m_PageTableEntries[index].physicalPageIndex == evictedPhysIndex) {
            m_PageTableEntries[index] = { ancPhysIndex, ancResMip };
            m_PageTableDirty = true;

            if (mip == 0) return;

            uint32_t childMip = mip - 1;
            uint32_t childGridW = std::max(1u, m_GridWidth >> childMip);
            uint32_t childGridH = std::max(1u, m_GridHeight >> childMip);

            for (uint32_t dy = 0; dy < 2; ++dy) {
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    uint32_t cx = x * 2 + dx;
                    uint32_t cy = y * 2 + dy;
                    if (cx < childGridW && cy < childGridH) {
                        PropagateEviction(cx, cy, childMip, evictedPhysIndex, ancPhysIndex, ancResMip);
                    }
                }
            }
        }
    }

    uint32_t VirtualTextureManager::GetPageTableIndex(uint32_t x, uint32_t y, uint32_t mip) const {
        return GetMipOffset(mip) + y * std::max(1u, m_GridWidth >> mip) + x;
    }

    uint32_t VirtualTextureManager::GetMipOffset(uint32_t mip) const {
        return m_MipOffsets[mip];
    }

    void VirtualTextureManager::BuildMipOffsets() {
        m_MipOffsets.resize(m_MipCount);
        uint32_t offset = 0;
        for (uint32_t mip = 0; mip < m_MipCount; ++mip) {
            m_MipOffsets[mip] = offset;
            uint32_t w = std::max(1u, m_GridWidth >> mip);
            uint32_t h = std::max(1u, m_GridHeight >> mip);
            offset += w * h;
        }
        m_PageTableEntries.assign(offset, PageTableEntry{});
    }

}
