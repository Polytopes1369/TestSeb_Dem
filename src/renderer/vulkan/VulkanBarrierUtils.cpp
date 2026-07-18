#include "VulkanBarrierUtils.h"

namespace VulkanBarrierUtils {

void RecordComputeBarrier(VkCommandBuffer cmd) {
    // Compute shader write → compute shader read synchronization
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void RecordComputeToComputeBarrier(VkCommandBuffer cmd) {
    RecordComputeBarrier(cmd);  // Identical to standard compute barrier
}

void RecordComputeToGraphicsBarrier(VkCommandBuffer cmd) {
    // Compute shader write → graphics pipeline read
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void RecordGraphicsToComputeBarrier(VkCommandBuffer cmd) {
    // Graphics pipeline write → compute shader read
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

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
) {
    VkImageMemoryBarrier2 imageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    imageBarrier.srcStageMask = srcStage;
    imageBarrier.srcAccessMask = srcAccess;
    imageBarrier.dstStageMask = dstStage;
    imageBarrier.dstAccessMask = dstAccess;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.image = image;
    imageBarrier.subresourceRange.aspectMask = aspectMask;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &imageBarrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void TransitionImageToShaderRead(VkCommandBuffer cmd, VkImage image) {
    RecordImageLayoutTransition(
        cmd,
        image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        0,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
    );
}

void TransitionImageToStorageWrite(VkCommandBuffer cmd, VkImage image) {
    RecordImageLayoutTransition(
        cmd,
        image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        0,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    );
}

void TransitionImageToColorAttachment(VkCommandBuffer cmd, VkImage image) {
    RecordImageLayoutTransition(
        cmd,
        image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        0,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
    );
}

void TransitionImageToDepthAttachment(VkCommandBuffer cmd, VkImage image) {
    RecordImageLayoutTransition(
        cmd,
        image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        0,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
    );
}

} // namespace VulkanBarrierUtils
