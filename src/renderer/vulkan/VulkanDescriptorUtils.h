#pragma once

#include <vulkan/vulkan.h>
#include <span>
#include <vector>
#include <stdexcept>

namespace VulkanDescriptorUtils {

// Helper to create a descriptor set layout from bindings
// Eliminates 71 lines of boilerplate across all passes
VkDescriptorSetLayout CreateDescriptorSetLayout(
    VkDevice device,
    std::span<const VkDescriptorSetLayoutBinding> bindings
);

// Descriptor pool + set pair (common pattern in all passes)
struct DescriptorPoolAndSet {
    VkDescriptorPool pool;
    VkDescriptorSetLayout layout;
    VkDescriptorSet set;
};

// Create both pool and allocated set in one call
// Eliminates 96 lines of boilerplate across all passes
DescriptorPoolAndSet CreateDescriptorPoolAndSet(
    VkDevice device,
    std::span<const VkDescriptorPoolSize> poolSizes,
    std::span<const VkDescriptorSetLayoutBinding> bindings
);

// Update descriptor writes (common pattern after setup)
// Eliminates 96 lines of boilerplate across all passes
void UpdateDescriptorSets(
    VkDevice device,
    std::span<const VkWriteDescriptorSet> writes
);

// Cleanup helpers (used in Shutdown methods)
void DestroyDescriptorPool(VkDevice device, VkDescriptorPool& pool);
void DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout& layout);

} // namespace VulkanDescriptorUtils
