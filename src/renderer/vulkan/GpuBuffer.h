#pragma once
// Minimal RAII ownership wrapper around a VMA-backed VkBuffer.
//
// The rest of this codebase currently creates every VkBuffer/VmaAllocation pair with raw
// vmaCreateBuffer/vmaDestroyBuffer calls inline in VulkanContext (see VulkanContext.cpp), with no
// shared ownership type. GpuGeometryPagePool (GpuGeometryPagePool.h) is the first consumer of
// this wrapper: it owns two such buffers (the physical page pool and the GPU-resident page
// table), and per the project's coding rules resource ownership must not be managed through raw
// handles -- move-only RAII here guarantees vmaDestroyBuffer is always called exactly once, even
// on an exception path, without the caller having to remember to.

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace renderer {

    class GpuBuffer {
    public:
        GpuBuffer() = default;
        ~GpuBuffer();

        GpuBuffer(const GpuBuffer&) = delete;
        GpuBuffer& operator=(const GpuBuffer&) = delete;

        GpuBuffer(GpuBuffer&& other) noexcept;
        GpuBuffer& operator=(GpuBuffer&& other) noexcept;

        // Allocates a new VkBuffer + VmaAllocation of `sizeBytes` bytes with the given usage
        // flags and VMA memory-usage hint. Destroys any buffer this instance already owned
        // first. `mapped` requests a persistently host-mapped allocation (only meaningful for
        // host-visible memoryUsage values, e.g. VMA_MEMORY_USAGE_CPU_ONLY /
        // VMA_MEMORY_USAGE_CPU_TO_GPU); MappedData() returns nullptr when it was not requested.
        // Throws std::runtime_error on allocation failure, matching this codebase's existing
        // VulkanContext buffer-creation convention.
        void Create(VmaAllocator allocator, VkDeviceSize sizeBytes, VkBufferUsageFlags usage,
            VmaMemoryUsage memoryUsage, bool mapped = false);

        // Destroys the owned buffer/allocation, if any, and resets this instance to empty.
        // Safe to call on an already-empty instance (no-op).
        void Destroy();

        VkBuffer Handle() const { return m_Buffer; }
        VmaAllocation Allocation() const { return m_Allocation; }
        VkDeviceSize Size() const { return m_SizeBytes; }
        void* MappedData() const { return m_MappedData; }
        bool IsValid() const { return m_Buffer != VK_NULL_HANDLE; }

    private:
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkBuffer m_Buffer = VK_NULL_HANDLE;
        VmaAllocation m_Allocation = VK_NULL_HANDLE;
        VkDeviceSize m_SizeBytes = 0;
        void* m_MappedData = nullptr;
    };

}
