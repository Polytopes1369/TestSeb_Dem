#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>
#include <cstdint>

namespace VulkanImageUtils {

// Image creation result (encapsulates common pattern)
struct VulkanImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

// Predefined image usage patterns (eliminates 34 occurrences of boilerplate)
enum class ImageUsagePattern {
    ColorAttachment,      // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    DepthAttachment,      // VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    StorageSampled,       // VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
    StorageImage,         // VK_IMAGE_USAGE_STORAGE_BIT only
    SampledImage,         // VK_IMAGE_USAGE_SAMPLED_BIT only
    TransferDst,          // VK_IMAGE_USAGE_TRANSFER_DST_BIT
};

// Create a 2D image with standard parameters
// Eliminates 34 × 12 lines = 408 lines of boilerplate across passes
VulkanImage Create2DImage(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t width,
    uint32_t height,
    VkFormat format,
    ImageUsagePattern usage
);

// Create a 3D image (volume, for volumetric effects)
VulkanImage Create3DImage(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t width,
    uint32_t height,
    uint32_t depth,
    VkFormat format,
    ImageUsagePattern usage
);

// Create a cube image (for environment maps, reflections)
VulkanImage CreateCubeImage(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t size,
    VkFormat format,
    ImageUsagePattern usage
);

// Create an image array (for virtual textures, texture atlasing)
VulkanImage Create2DImageArray(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t width,
    uint32_t height,
    uint32_t layers,
    VkFormat format,
    ImageUsagePattern usage
);

// Destroy image and associated resources
void DestroyImage(VkDevice device, VmaAllocator allocator, VulkanImage& image);

} // namespace VulkanImageUtils
