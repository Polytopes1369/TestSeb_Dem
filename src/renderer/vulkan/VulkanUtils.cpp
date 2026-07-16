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
        uint32_t mipCount
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
        barrier.subresourceRange = { aspectMask, baseMip, mipCount, 0, 1 };

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

}
