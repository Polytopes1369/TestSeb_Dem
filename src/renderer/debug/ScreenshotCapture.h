#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below): captures
// the current swapchain image to a BMP file on disk, used by DebugTestPipeline to attach a visual
// per feature test to its Markdown report. No new third-party dependency (no stb_image_write or
// similar exists anywhere in this project, matching CLAUDE.md's "no heavy frameworks" rule) --
// VulkanContext::CreateSwapchain() hardcodes VK_FORMAT_B8G8R8A8_UNORM (see VulkanContext.cpp), so
// a plain 32-bit BMP (whose pixel order is already B,G,R,[pad]) is a direct byte-for-byte copy of
// the readback, no channel-swizzle work needed.
//
// Split into a Record/Write pair rather than one self-contained one-shot-command-buffer call: per
// the Vulkan spec, a presentable image may only be used (including a layout transition) between
// vkAcquireNextImageKHR and the matching vkQueuePresentKHR -- capturing via a SEPARATE one-shot
// command buffer submitted AFTER present is a validation error
// (VUID-vkQueueSubmit-pCommandBuffers-04591-class "used before being acquired since last present").
// RecordCapture() must therefore be called from INSIDE the frame's own command buffer, before that
// frame's vkQueuePresentKHR; WriteStagingToBmp() runs afterward, once the caller has waited for
// that frame's GPU work to complete.
#ifndef NDEBUG

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <string>

namespace debugpipeline {

    class ScreenshotCapture {
    public:
        // Records, into the caller's already-open `cmd` (mid-frame, right after
        // ClusterRenderPipeline::RecordFrameLate() has recorded its own final PRESENT_SRC_KHR barrier
        // and before vkEndCommandBuffer()), a transition to TRANSFER_SRC_OPTIMAL, a copy into a
        // freshly allocated staging buffer, and a transition back to PRESENT_SRC_KHR (matching the
        // layout RecordFrame's own barrier already left the image in, so this frame's own
        // vkQueuePresentKHR sees the same steady state as if no capture had happened). Returns the
        // staging buffer/allocation via the out-parameters; false (logged) on allocation failure.
        static bool RecordCapture(
            VkCommandBuffer cmd,
            VmaAllocator allocator,
            VkImage swapchainImage,
            VkFormat swapchainFormat,
            VkExtent2D extent,
            VkBuffer& outStagingBuffer,
            VmaAllocation& outStagingAllocation);

        // Maps `stagingBuffer` (the caller must already have waited for the frame that recorded
        // the copy into it -- e.g. via vkQueueWaitIdle -- so the GPU write is guaranteed visible),
        // writes it to `outputFilePath` as a BMP, and destroys the staging buffer. Returns false
        // (logged) on any failure -- a screenshot failure must never abort the test pipeline run.
        static bool WriteStagingToBmp(
            VmaAllocator allocator,
            VkBuffer stagingBuffer,
            VmaAllocation stagingAllocation,
            VkExtent2D extent,
            const std::string& outputFilePath);
    };

}

#endif // NDEBUG
