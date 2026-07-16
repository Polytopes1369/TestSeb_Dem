#include "renderer/passes/VirtualShadowMapPool.h"

#include <cassert>
#include <format>

#include "core/Logger.h"

namespace renderer {

    namespace {

        void TransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
            uint32_t layerCount, VkImageLayout oldLayout, VkImageLayout newLayout,
            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
            VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = srcStage;
            barrier.srcAccessMask = srcAccess;
            barrier.dstStageMask = dstStage;
            barrier.dstAccessMask = dstAccess;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = { aspect, 0, 1, 0, layerCount };
            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

    } // namespace

    bool VirtualShadowMapPool::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        uint32_t totalVSMCount, uint32_t physicalPageCapacity) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_TotalVSMCount = totalVSMCount;
        m_PhysicalPageCapacity = physicalPageCapacity;

        // =====================================================================================
        // STEP 1 -- Physical page pool: one VK_IMAGE_TYPE_2D, `physicalPageCapacity` array layers
        // of kShadowPageTexels^2 D32_SFLOAT texels each (see this class's own header comment on
        // why a texture array, not a packed atlas, is the right fit for fixed-size pages).
        // =====================================================================================
        VkImageCreateInfo poolImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        poolImageInfo.imageType = VK_IMAGE_TYPE_2D;
        poolImageInfo.format = VK_FORMAT_D32_SFLOAT;
        poolImageInfo.extent = { kShadowPageTexels, kShadowPageTexels, 1 };
        poolImageInfo.mipLevels = 1;
        poolImageInfo.arrayLayers = physicalPageCapacity;
        poolImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        poolImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        poolImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        poolImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        poolImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo gpuOnlyAlloc{};
        gpuOnlyAlloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &poolImageInfo, &gpuOnlyAlloc, &m_PhysicalPoolImage, &m_PhysicalPoolAllocation, nullptr));

        VkImageViewCreateInfo arrayViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        arrayViewInfo.image = m_PhysicalPoolImage;
        arrayViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        arrayViewInfo.format = VK_FORMAT_D32_SFLOAT;
        arrayViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, physicalPageCapacity };
        VK_CHECK(vkCreateImageView(m_Device, &arrayViewInfo, nullptr, &m_PhysicalPoolArrayView));

        m_PhysicalLayerViews.resize(physicalPageCapacity);
        for (uint32_t layer = 0; layer < physicalPageCapacity; ++layer) {
            VkImageViewCreateInfo layerViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            layerViewInfo.image = m_PhysicalPoolImage;
            layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            layerViewInfo.format = VK_FORMAT_D32_SFLOAT;
            layerViewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, layer, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &layerViewInfo, nullptr, &m_PhysicalLayerViews[layer]));
        }

        // Plain (non-comparison) sampler, same convention as renderer::ShadowMapPass's own --
        // consumers do their own manual PCF depth comparison. Border color: max-depth opaque white
        // so sampling past a page's edge reads as "far/unshadowed" (see ShadowMapPass.cpp's
        // identical rationale) -- relevant here for the PCF taps near a page's own border, since
        // unlike the old single whole-scene map, a shadow page's neighbor in world space is not
        // necessarily its neighbor in the physical atlas (a different array layer entirely).
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_Sampler));

        // =====================================================================================
        // STEP 2 -- GPU-resident page table: totalVSMCount * kShadowPagesPerVSM uint32 entries.
        // =====================================================================================
        m_PageTableBuffer.Create(allocator,
            static_cast<VkDeviceSize>(totalVSMCount) * kShadowPagesPerVSM * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // =====================================================================================
        // STEP 3 -- One-time layout transition: GENERAL for the whole lifetime (valid for both a
        // depth-attachment write AND a later sampled read), mirroring ShadowMapPass's own choice
        // (see that class's Init()) so no page ever needs to ping-pong layouts between being
        // rendered and being sampled.
        // =====================================================================================
        {
            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            TransitionImageLayout(cmd, m_PhysicalPoolImage, VK_IMAGE_ASPECT_DEPTH_BIT, physicalPageCapacity,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

            ClearPageTable(cmd);

            vkEndCommandBuffer(cmd);
            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(m_Device, commandPool, 1, &cmd);
        }

        // =====================================================================================
        // STEP 4 -- CPU-side allocator state.
        // =====================================================================================
        m_LogicalToPhysical.assign(static_cast<size_t>(totalVSMCount) * kShadowPagesPerVSM, kInvalidShadowPhysicalPage);
        m_LRUNodes.assign(physicalPageCapacity, LRUNode{});
        m_FreePhysicalSlots.clear();
        m_NextUnusedPhysicalSlot = 0;
        m_ResidentCount = 0;
        m_LRUMostRecentPage = kInvalidShadowPhysicalPage;
        m_LRULeastRecentPage = kInvalidShadowPhysicalPage;

        LOG_INFO(std::format("[VirtualShadowMapPool] Initialized: {} VSM(s), {} physical page(s) ({}x{} texels each, {} MB total).",
            totalVSMCount, physicalPageCapacity, kShadowPageTexels, kShadowPageTexels,
            (static_cast<uint64_t>(physicalPageCapacity) * kShadowPageTexels * kShadowPageTexels * sizeof(float)) / (1024 * 1024)));
        return true;
    }

    void VirtualShadowMapPool::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Sampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_Sampler, nullptr);
            if (m_PhysicalPoolArrayView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_PhysicalPoolArrayView, nullptr);
            for (VkImageView view : m_PhysicalLayerViews) {
                if (view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, view, nullptr);
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            if (m_PhysicalPoolImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_PhysicalPoolImage, m_PhysicalPoolAllocation);
        }
        m_PageTableBuffer.Destroy();

        m_PhysicalLayerViews.clear();
        m_PhysicalPoolArrayView = VK_NULL_HANDLE;
        m_PhysicalPoolImage = VK_NULL_HANDLE;
        m_PhysicalPoolAllocation = VK_NULL_HANDLE;
        m_Sampler = VK_NULL_HANDLE;

        m_LogicalToPhysical.clear();
        m_LRUNodes.clear();
        m_FreePhysicalSlots.clear();
        m_NextUnusedPhysicalSlot = 0;
        m_ResidentCount = 0;
        m_LRUMostRecentPage = kInvalidShadowPhysicalPage;
        m_LRULeastRecentPage = kInvalidShadowPhysicalPage;
        m_TotalVSMCount = 0;
        m_PhysicalPageCapacity = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void VirtualShadowMapPool::ClearPageTable(VkCommandBuffer cmd) {
        static_assert(kUnmappedSentinel == 0xFFFFFFFFu,
            "ClearPageTable relies on kUnmappedSentinel being an all-ones fill word");
        vkCmdFillBuffer(cmd, m_PageTableBuffer.Handle(), 0, VK_WHOLE_SIZE, kUnmappedSentinel);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    uint32_t VirtualShadowMapPool::AllocatePhysicalSlot() {
        if (!m_FreePhysicalSlots.empty()) {
            uint32_t slot = m_FreePhysicalSlots.back();
            m_FreePhysicalSlots.pop_back();
            return slot;
        }
        assert(m_NextUnusedPhysicalSlot < m_PhysicalPageCapacity);
        return m_NextUnusedPhysicalSlot++;
    }

    void VirtualShadowMapPool::LinkAsMostRecentlyUsed(uint32_t physicalLayer) {
        m_LRUNodes[physicalLayer].prev = kInvalidShadowPhysicalPage;
        m_LRUNodes[physicalLayer].next = m_LRUMostRecentPage;
        if (m_LRUMostRecentPage != kInvalidShadowPhysicalPage) {
            m_LRUNodes[m_LRUMostRecentPage].prev = physicalLayer;
        }
        m_LRUMostRecentPage = physicalLayer;
        if (m_LRULeastRecentPage == kInvalidShadowPhysicalPage) {
            m_LRULeastRecentPage = physicalLayer;
        }
    }

    void VirtualShadowMapPool::UnlinkFromLRUList(uint32_t physicalLayer) {
        LRUNode& node = m_LRUNodes[physicalLayer];
        if (node.prev != kInvalidShadowPhysicalPage) {
            m_LRUNodes[node.prev].next = node.next;
        } else {
            m_LRUMostRecentPage = node.next;
        }
        if (node.next != kInvalidShadowPhysicalPage) {
            m_LRUNodes[node.next].prev = node.prev;
        } else {
            m_LRULeastRecentPage = node.prev;
        }
        node.prev = kInvalidShadowPhysicalPage;
        node.next = kInvalidShadowPhysicalPage;
    }

    void VirtualShadowMapPool::WritePageTableEntry(VkCommandBuffer cmd, uint32_t logicalPageID, uint32_t physicalLayer) {
        VkDeviceSize entryOffset = static_cast<VkDeviceSize>(logicalPageID) * sizeof(uint32_t);
        vkCmdUpdateBuffer(cmd, m_PageTableBuffer.Handle(), entryOffset, sizeof(uint32_t), &physicalLayer);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT; // vkCmdUpdateBuffer is classified as CLEAR by the Vulkan spec's stage table.
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void VirtualShadowMapPool::EvictOneLeastRecentlyUsed(VkCommandBuffer cmd) {
        if (m_LRULeastRecentPage == kInvalidShadowPhysicalPage) {
            return; // Nothing resident to evict.
        }
        uint32_t physicalLayer = m_LRULeastRecentPage;
        uint32_t logicalPageID = m_LRUNodes[physicalLayer].logicalPageID;

        UnlinkFromLRUList(physicalLayer);
        m_LogicalToPhysical[logicalPageID] = kInvalidShadowPhysicalPage;
        m_FreePhysicalSlots.push_back(physicalLayer);
        --m_ResidentCount;

        WritePageTableEntry(cmd, logicalPageID, kUnmappedSentinel);
    }

    uint32_t VirtualShadowMapPool::AllocatePage(VkCommandBuffer cmd, uint32_t logicalPageID, uint32_t maxEvictions) {
        assert(logicalPageID < m_LogicalToPhysical.size());

        uint32_t existing = m_LogicalToPhysical[logicalPageID];
        if (existing != kInvalidShadowPhysicalPage) {
            return existing; // Already resident -- idempotent, no state change.
        }

        bool poolFull = m_FreePhysicalSlots.empty() && m_NextUnusedPhysicalSlot >= m_PhysicalPageCapacity;
        for (uint32_t evicted = 0; evicted < maxEvictions && poolFull; ++evicted) {
            EvictOneLeastRecentlyUsed(cmd);
            poolFull = m_FreePhysicalSlots.empty() && m_NextUnusedPhysicalSlot >= m_PhysicalPageCapacity;
        }
        if (poolFull) {
            return kInvalidShadowPhysicalPage; // Every page resident and none evicted -- should not happen with maxEvictions >= 1.
        }

        uint32_t physicalLayer = AllocatePhysicalSlot();
        m_LogicalToPhysical[logicalPageID] = physicalLayer;
        m_LRUNodes[physicalLayer].logicalPageID = logicalPageID;
        LinkAsMostRecentlyUsed(physicalLayer);
        ++m_ResidentCount;

        WritePageTableEntry(cmd, logicalPageID, physicalLayer);
        return physicalLayer;
    }

    void VirtualShadowMapPool::TouchPage(uint32_t logicalPageID) {
        assert(logicalPageID < m_LogicalToPhysical.size());
        uint32_t physicalLayer = m_LogicalToPhysical[logicalPageID];
        if (physicalLayer == kInvalidShadowPhysicalPage) {
            return;
        }
        UnlinkFromLRUList(physicalLayer);
        LinkAsMostRecentlyUsed(physicalLayer);
    }

    bool VirtualShadowMapPool::IsResident(uint32_t logicalPageID) const {
        assert(logicalPageID < m_LogicalToPhysical.size());
        return m_LogicalToPhysical[logicalPageID] != kInvalidShadowPhysicalPage;
    }

    uint32_t VirtualShadowMapPool::GetPhysicalLayer(uint32_t logicalPageID) const {
        assert(logicalPageID < m_LogicalToPhysical.size());
        return m_LogicalToPhysical[logicalPageID];
    }

}
