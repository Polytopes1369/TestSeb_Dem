#pragma once
// Minimal RAII ownership wrapper around a VMA-backed VkImage and optionally its default VkImageView.
//
// Move-only semantics ensure that vmaDestroyImage and vkDestroyImageView are called exactly once.

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace renderer {

    class GpuImage {
    public:
        GpuImage() = default;
        ~GpuImage();

        GpuImage(const GpuImage&) = delete;
        GpuImage& operator=(const GpuImage&) = delete;

        GpuImage(GpuImage&& other) noexcept;
        GpuImage& operator=(GpuImage&& other) noexcept;

        // Allocates a new VkImage + VmaAllocation using the provided creation info and memory usage.
        // If aspectMask is not 0, a default VkImageView covering all layers and mips will be created.
        // Throws std::runtime_error on failure.
        void Create(VmaAllocator allocator, VkDevice device,
            const VkImageCreateInfo& imageInfo, VmaMemoryUsage memoryUsage,
            VkImageAspectFlags aspectMask = 0);

        // Destroys the owned image, view, and allocation, resetting to empty.
        void Destroy();

        VkImage Image() const { return m_Image; }
        VkImageView View() const { return m_View; }
        VmaAllocation Allocation() const { return m_Allocation; }
        VkFormat Format() const { return m_Format; }
        VkExtent3D Extent() const { return m_Extent; }
        bool IsValid() const { return m_Image != VK_NULL_HANDLE; }

    private:
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkDevice m_Device = VK_NULL_HANDLE;
        VkImage m_Image = VK_NULL_HANDLE;
        VmaAllocation m_Allocation = VK_NULL_HANDLE;
        VkImageView m_View = VK_NULL_HANDLE;
        VkFormat m_Format = VK_FORMAT_UNDEFINED;
        VkExtent3D m_Extent = { 0, 0, 0 };
    };

}
