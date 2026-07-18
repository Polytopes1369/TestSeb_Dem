#include "VulkanImageUtils.h"
#include "core/Logger.h"
#include <format>
#include <stdexcept>

namespace VulkanImageUtils {

// Helper to convert usage pattern to VkImageUsageFlags
static VkImageUsageFlags GetImageUsageFlags(ImageUsagePattern usage) {
    switch (usage) {
        case ImageUsagePattern::ColorAttachment:
            return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        case ImageUsagePattern::DepthAttachment:
            return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        case ImageUsagePattern::StorageSampled:
            return VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        case ImageUsagePattern::StorageImage:
            return VK_IMAGE_USAGE_STORAGE_BIT;
        case ImageUsagePattern::SampledImage:
            return VK_IMAGE_USAGE_SAMPLED_BIT;
        case ImageUsagePattern::TransferDst:
            return VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        default:
            return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
}

// Helper to get appropriate aspect mask for format
static VkImageAspectFlags GetImageAspectFlags(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// Generic image creation (shared by all Create* functions)
static VulkanImage CreateImageInternal(
    VkDevice device,
    VmaAllocator allocator,
    const VkImageCreateInfo& imageInfo,
    const char* debugName
) {
    VulkanImage result{};

    // Create image
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &result.image, &result.allocation, nullptr) != VK_SUCCESS) {
        LOG_ERROR(std::format("[VulkanImageUtils] Failed to create image: {}", debugName));
        throw std::runtime_error("vmaCreateImage failed");
    }

    // Create image view
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = result.image;
    viewInfo.viewType = (imageInfo.imageType == VK_IMAGE_TYPE_2D && imageInfo.arrayLayers > 1)
                            ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                        : (imageInfo.imageType == VK_IMAGE_TYPE_3D)
                            ? VK_IMAGE_VIEW_TYPE_3D
                        : (imageInfo.imageType == VK_IMAGE_TYPE_1D)
                            ? VK_IMAGE_VIEW_TYPE_1D
                            : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = imageInfo.format;
    viewInfo.subresourceRange.aspectMask = GetImageAspectFlags(imageInfo.format);
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = imageInfo.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = imageInfo.arrayLayers;

    if (vkCreateImageView(device, &viewInfo, nullptr, &result.view) != VK_SUCCESS) {
        vmaDestroyImage(allocator, result.image, result.allocation);
        LOG_ERROR(std::format("[VulkanImageUtils] Failed to create image view for: {}", debugName));
        throw std::runtime_error("vkCreateImageView failed");
    }

    return result;
}

VulkanImage Create2DImage(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    ImageUsagePattern usage
) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = GetImageUsageFlags(usage);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    return CreateImageInternal(device, allocator, imageInfo, "2D Image");
}

VulkanImage Create3DImage(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    VkFormat format,
    ImageUsagePattern usage
) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, depth};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = GetImageUsageFlags(usage);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    return CreateImageInternal(device, allocator, imageInfo, "3D Image (Volume)");
}

VulkanImage CreateCubeImage(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t size,
    VkFormat format,
    ImageUsagePattern usage
) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {size, size, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 6;  // Cube has 6 faces
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = GetImageUsageFlags(usage);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    return CreateImageInternal(device, allocator, imageInfo, "Cube Image");
}

VulkanImage Create2DImageArray(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t width,
    uint32_t height,
    uint32_t layers,
    VkFormat format,
    ImageUsagePattern usage
) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = layers;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = GetImageUsageFlags(usage);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    return CreateImageInternal(device, allocator, imageInfo, "2D Image Array");
}

void DestroyImage(VkDevice device, VmaAllocator allocator, VulkanImage& image) {
    if (image.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, image.view, nullptr);
        image.view = VK_NULL_HANDLE;
    }
    if (image.image != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator, image.image, image.allocation);
        image.image = VK_NULL_HANDLE;
        image.allocation = VK_NULL_HANDLE;
    }
}

} // namespace VulkanImageUtils
