#include "renderer/vulkan/VulkanUtils.h"

#include <stdexcept>
#include "renderer/vulkan/VulkanPipeline.h"

namespace renderer {

    void VulkanUtils::ExecuteOneShotCommands(
        VkDevice device,
        VkCommandPool commandPool,
        VkQueue queue,
        const std::function<void(VkCommandBuffer)>& recordFunc
    ) {
        VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd;
        if (vkAllocateCommandBuffers(device, &allocInfo, &cmd) != VK_SUCCESS) {
            throw std::runtime_error("VulkanUtils: Failed to allocate one-shot command buffer");
        }

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            throw std::runtime_error("VulkanUtils: Failed to begin one-shot command buffer");
        }

        recordFunc(cmd);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            throw std::runtime_error("VulkanUtils: Failed to end one-shot command buffer");
        }

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            throw std::runtime_error("VulkanUtils: Failed to submit one-shot command buffer");
        }

        if (vkQueueWaitIdle(queue) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
            throw std::runtime_error("VulkanUtils: Failed to wait idle on queue for one-shot commands");
        }

        vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    }

    void VulkanUtils::TransitionImageLayout(
        VkCommandBuffer cmd,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkPipelineStageFlags2 srcStage,
        VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage,
        VkAccessFlags2 dstAccess,
        VkImageAspectFlags aspectMask,
        uint32_t baseMip,
        uint32_t mipCount,
        uint32_t baseLayer,
        uint32_t layerCount
    ) {
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
        barrier.subresourceRange = { aspectMask, baseMip, mipCount, baseLayer, layerCount };

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void VulkanUtils::TransitionImageLayoutOneShot(
        VkDevice device,
        VkCommandPool commandPool,
        VkQueue queue,
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkPipelineStageFlags2 srcStage,
        VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage,
        VkAccessFlags2 dstAccess,
        VkImageAspectFlags aspectMask,
        uint32_t baseMip,
        uint32_t mipCount,
        uint32_t baseLayer,
        uint32_t layerCount
    ) {
        ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            TransitionImageLayout(cmd, image, oldLayout, newLayout, srcStage, srcAccess, dstStage,
                dstAccess, aspectMask, baseMip, mipCount, baseLayer, layerCount);
            });
    }

    VkSampler VulkanUtils::CreateNearestSampler(VkDevice device, float maxLod) {
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = maxLod;
        samplerInfo.compareEnable = VK_FALSE; // Plain sampler2D read, not a shadow/compare sampler.
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        VkSampler sampler = VK_NULL_HANDLE;
        if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
            throw std::runtime_error("VulkanUtils: Failed to create nearest-filter sampler");
        }
        return sampler;
    }

}
