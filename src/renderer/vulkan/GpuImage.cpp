#include "renderer/vulkan/GpuImage.h"
#include "core/Logger.h"

#include <format>
#include <stdexcept>
#include <utility>

namespace renderer {

    GpuImage::~GpuImage() {
        Destroy();
    }

    GpuImage::GpuImage(GpuImage&& other) noexcept
        : m_Allocator(other.m_Allocator)
        , m_Device(other.m_Device)
        , m_Image(other.m_Image)
        , m_Allocation(other.m_Allocation)
        , m_View(other.m_View)
        , m_Format(other.m_Format)
        , m_Extent(other.m_Extent) {
        other.m_Allocator = VK_NULL_HANDLE;
        other.m_Device = VK_NULL_HANDLE;
        other.m_Image = VK_NULL_HANDLE;
        other.m_Allocation = VK_NULL_HANDLE;
        other.m_View = VK_NULL_HANDLE;
        other.m_Format = VK_FORMAT_UNDEFINED;
        other.m_Extent = { 0, 0, 0 };
    }

    GpuImage& GpuImage::operator=(GpuImage&& other) noexcept {
        if (this != &other) {
            Destroy();
            m_Allocator = other.m_Allocator;
            m_Device = other.m_Device;
            m_Image = other.m_Image;
            m_Allocation = other.m_Allocation;
            m_View = other.m_View;
            m_Format = other.m_Format;
            m_Extent = other.m_Extent;

            other.m_Allocator = VK_NULL_HANDLE;
            other.m_Device = VK_NULL_HANDLE;
            other.m_Image = VK_NULL_HANDLE;
            other.m_Allocation = VK_NULL_HANDLE;
            other.m_View = VK_NULL_HANDLE;
            other.m_Format = VK_FORMAT_UNDEFINED;
            other.m_Extent = { 0, 0, 0 };
        }
        return *this;
    }

    void GpuImage::Create(VmaAllocator allocator, VkDevice device,
        const VkImageCreateInfo& imageInfo, VmaMemoryUsage memoryUsage,
        VkImageAspectFlags aspectMask) {
        Destroy();

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = memoryUsage;

        VmaAllocationInfo resultInfo{};
        if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &m_Image, &m_Allocation, &resultInfo) != VK_SUCCESS) {
            m_Image = VK_NULL_HANDLE;
            m_Allocation = VK_NULL_HANDLE;
            LOG_ERROR(std::format("[GpuImage] Failed to allocate image of size {}x{}x{} (format: 0x{:08X})!",
                imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth, static_cast<uint32_t>(imageInfo.format)));
            throw std::runtime_error("GpuImage::Create -- vmaCreateImage failed");
        }

        m_Allocator = allocator;
        m_Device = device;
        m_Format = imageInfo.format;
        m_Extent = imageInfo.extent;

        if (aspectMask != 0) {
            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_Image;
            viewInfo.viewType = (imageInfo.imageType == VK_IMAGE_TYPE_3D) ? VK_IMAGE_VIEW_TYPE_3D :
                                (imageInfo.imageType == VK_IMAGE_TYPE_2D) ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_1D;
            viewInfo.format = imageInfo.format;
            viewInfo.subresourceRange.aspectMask = aspectMask;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = imageInfo.mipLevels;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = imageInfo.arrayLayers;

            if (vkCreateImageView(device, &viewInfo, nullptr, &m_View) != VK_SUCCESS) {
                m_View = VK_NULL_HANDLE;
                vmaDestroyImage(m_Allocator, m_Image, m_Allocation);
                m_Image = VK_NULL_HANDLE;
                m_Allocation = VK_NULL_HANDLE;
                LOG_ERROR("[GpuImage] Failed to create VkImageView!");
                throw std::runtime_error("GpuImage::Create -- vkCreateImageView failed");
            }
        }

        LOG_INFO(std::format("[GpuImage] Allocated image: handle={:#x}, view={:#x}, format={}, extent={}x{}x{}",
            reinterpret_cast<uintptr_t>(m_Image), reinterpret_cast<uintptr_t>(m_View),
            static_cast<int>(m_Format), m_Extent.width, m_Extent.height, m_Extent.depth));
    }

    void GpuImage::Destroy() {
        if (m_View != VK_NULL_HANDLE) {
            vkDestroyImageView(m_Device, m_View, nullptr);
            m_View = VK_NULL_HANDLE;
        }
        if (m_Image != VK_NULL_HANDLE) {
            LOG_INFO(std::format("[GpuImage] Destroying image: handle={:#x}", reinterpret_cast<uintptr_t>(m_Image)));
            vmaDestroyImage(m_Allocator, m_Image, m_Allocation);
            m_Image = VK_NULL_HANDLE;
            m_Allocation = VK_NULL_HANDLE;
        }
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
        m_Format = VK_FORMAT_UNDEFINED;
        m_Extent = { 0, 0, 0 };
    }

}
