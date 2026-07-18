#pragma once

#include <vulkan/vulkan.h>

namespace VulkanBarrierUtils {

// Eliminates 68 occurrences of barrier boilerplate across compute/graphics passes

// Generic shader barrier (read/write sync for compute)
void RecordComputeBarrier(VkCommandBuffer cmd);

// Compute shader write → compute shader read (within same stage)
void RecordComputeToComputeBarrier(VkCommandBuffer cmd);

// Compute → Graphics (transfer data from compute to graphics pipeline)
void RecordComputeToGraphicsBarrier(VkCommandBuffer cmd);

// Graphics → Compute (transfer data from graphics to compute pipeline)
void RecordGraphicsToComputeBarrier(VkCommandBuffer cmd);

// Image layout transition with implicit memory barriers
void RecordImageLayoutTransition(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkImageAspectFlags aspectMask,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess
);

// Common transitions (pre-configured for typical scenarios)
void TransitionImageToShaderRead(VkCommandBuffer cmd, VkImage image);
void TransitionImageToStorageWrite(VkCommandBuffer cmd, VkImage image);
void TransitionImageToColorAttachment(VkCommandBuffer cmd, VkImage image);
void TransitionImageToDepthAttachment(VkCommandBuffer cmd, VkImage image);

} // namespace VulkanBarrierUtils
