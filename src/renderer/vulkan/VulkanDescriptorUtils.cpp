#include "VulkanDescriptorUtils.h"
#include "core/Logger.h"
#include <format>

namespace VulkanDescriptorUtils {

VkDescriptorSetLayout CreateDescriptorSetLayout(
    VkDevice device,
    std::span<const VkDescriptorSetLayoutBinding> bindings
) {
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
        LOG_ERROR(std::format("[VulkanDescriptorUtils] Failed to create descriptor set layout with {} bindings", bindings.size()));
        throw std::runtime_error("vkCreateDescriptorSetLayout failed");
    }

    return layout;
}

DescriptorPoolAndSet CreateDescriptorPoolAndSet(
    VkDevice device,
    std::span<const VkDescriptorPoolSize> poolSizes,
    std::span<const VkDescriptorSetLayoutBinding> bindings
) {
    DescriptorPoolAndSet result{};

    // Create descriptor set layout
    result.layout = CreateDescriptorSetLayout(device, bindings);

    // Create descriptor pool
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &result.pool) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device, result.layout, nullptr);
        LOG_ERROR(std::format("[VulkanDescriptorUtils] Failed to create descriptor pool with {} pool sizes", poolSizes.size()));
        throw std::runtime_error("vkCreateDescriptorPool failed");
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = result.pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &result.layout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &result.set) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device, result.pool, nullptr);
        vkDestroyDescriptorSetLayout(device, result.layout, nullptr);
        LOG_ERROR("[VulkanDescriptorUtils] Failed to allocate descriptor set");
        throw std::runtime_error("vkAllocateDescriptorSets failed");
    }

    return result;
}

void UpdateDescriptorSets(
    VkDevice device,
    std::span<const VkWriteDescriptorSet> writes
) {
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void DestroyDescriptorPool(VkDevice device, VkDescriptorPool& pool) {
    if (pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, pool, nullptr);
        pool = VK_NULL_HANDLE;
    }
}

void DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout& layout) {
    if (layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
}

} // namespace VulkanDescriptorUtils
