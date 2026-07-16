#pragma once

#include <vulkan/vulkan.h>
#include <functional>

namespace renderer {

    class VulkanUtils {
    public:
        // Allocates a primary one-time submit command buffer, executes the provided record function,
        // submits it to the queue, blocks until completion, and frees the command buffer.
        static void ExecuteOneShotCommands(
            VkDevice device,
            VkCommandPool commandPool,
            VkQueue queue,
            const std::function<void(VkCommandBuffer)>& recordFunc
        );

        // Performs a Vulkan 2 image layout transition using vkCmdPipelineBarrier2.
        static void TransitionImageLayout(
            VkCommandBuffer cmd,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkPipelineStageFlags2 srcStage,
            VkAccessFlags2 srcAccess,
            VkPipelineStageFlags2 dstStage,
            VkAccessFlags2 dstAccess,
            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            uint32_t baseMip = 0,
            uint32_t mipCount = 1
        );
    };

}
