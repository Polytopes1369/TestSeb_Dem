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
            uint32_t mipCount = 1,
            uint32_t baseLayer = 0,
            uint32_t layerCount = 1
        );

        // Convenience wrapper: allocates/submits/waits on a one-shot command buffer (via
        // ExecuteOneShotCommands) whose sole content is a single TransitionImageLayout call. Used
        // by passes that only need one image transitioned outside their normal per-frame recording
        // (e.g. an UNDEFINED -> GENERAL transition performed once at pass construction time).
        static void TransitionImageLayoutOneShot(
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
            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            uint32_t baseMip = 0,
            uint32_t mipCount = 1,
            uint32_t baseLayer = 0,
            uint32_t layerCount = 1
        );

        // Creates a simple nearest-filter, clamp-to-edge sampler with no mip/anisotropic filtering
        // beyond a caller-provided max LOD -- the common "point-sample this compute-written image
        // with no wraparound" convention used by HZB/occlusion/resolve/shading-bin passes.
        static VkSampler CreateNearestSampler(VkDevice device, float maxLod = 0.0f);
    };

}
