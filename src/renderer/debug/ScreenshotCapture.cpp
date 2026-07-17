#ifndef NDEBUG

#include "renderer/debug/ScreenshotCapture.h"
#include "renderer/vulkan/VulkanUtils.h"
#include "core/Logger.h"

#include <cstdint>
#include <cstdio>

namespace debugpipeline {

    namespace {

        // Minimal, dependency-free 32-bit uncompressed BMP writer (BITMAPFILEHEADER +
        // BITMAPINFOHEADER, BI_RGB, bottom-up row order). `pixels` must already be in B,G,R,A byte
        // order (matches VK_FORMAT_B8G8R8A8_UNORM's own in-memory layout, so the GPU readback below
        // needs no channel swizzle).
#pragma pack(push, 1)
        struct BmpFileHeader {
            uint16_t type = 0x4D42; // 'BM'
            uint32_t fileSize = 0;
            uint16_t reserved1 = 0;
            uint16_t reserved2 = 0;
            uint32_t pixelDataOffset = 0;
        };

        struct BmpInfoHeader {
            uint32_t headerSize = 40;
            int32_t width = 0;
            int32_t height = 0;
            uint16_t planes = 1;
            uint16_t bitsPerPixel = 32;
            uint32_t compression = 0; // BI_RGB
            uint32_t imageSize = 0;
            int32_t xPixelsPerMeter = 2835; // ~72 DPI
            int32_t yPixelsPerMeter = 2835;
            uint32_t colorsUsed = 0;
            uint32_t colorsImportant = 0;
        };
#pragma pack(pop)

        bool WriteBmp32(const std::string& path, uint32_t width, uint32_t height, const uint8_t* bgraPixels, size_t rowPitchBytes) {
            const uint32_t imageSize = width * height * 4u;

            BmpFileHeader fileHeader{};
            fileHeader.pixelDataOffset = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader);
            fileHeader.fileSize = fileHeader.pixelDataOffset + imageSize;

            BmpInfoHeader infoHeader{};
            infoHeader.width = static_cast<int32_t>(width);
            infoHeader.height = static_cast<int32_t>(height); // Positive: bottom-up row order.
            infoHeader.imageSize = imageSize;

            FILE* file = nullptr;
            if (fopen_s(&file, path.c_str(), "wb") != 0 || file == nullptr) {
                LOG_ERROR(std::string("[ScreenshotCapture] Failed to open output file: ") + path);
                return false;
            }

            fwrite(&fileHeader, sizeof(fileHeader), 1, file);
            fwrite(&infoHeader, sizeof(infoHeader), 1, file);

            // BMP row order is bottom-up; the GPU readback is top-down (row 0 = top scanline), so
            // rows are written back-to-front here rather than pre-flipping the staging buffer.
            for (uint32_t row = 0; row < height; ++row) {
                const uint8_t* srcRow = bgraPixels + static_cast<size_t>(height - 1 - row) * rowPitchBytes;
                fwrite(srcRow, 4, width, file);
            }

            fclose(file);
            return true;
        }

    } // namespace

    bool ScreenshotCapture::RecordCapture(
        VkCommandBuffer cmd,
        VmaAllocator allocator,
        VkImage swapchainImage,
        VkFormat swapchainFormat,
        VkExtent2D extent,
        VkBuffer& outStagingBuffer,
        VmaAllocation& outStagingAllocation) {

        outStagingBuffer = VK_NULL_HANDLE;
        outStagingAllocation = VK_NULL_HANDLE;

        if (swapchainFormat != VK_FORMAT_B8G8R8A8_UNORM) {
            // The BMP writer above assumes B,G,R,A byte order with no conversion -- see this
            // file's own header comment. VulkanContext::CreateSwapchain() hardcodes this format,
            // so this branch only fires if that assumption is ever changed without updating this
            // capture path too.
            LOG_ERROR("[ScreenshotCapture] Unsupported swapchain format for BMP capture (expected B8G8R8A8_UNORM).");
            return false;
        }

        const VkDeviceSize bufferBytes = static_cast<VkDeviceSize>(extent.width) * extent.height * 4u;

        VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = bufferBytes;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocCreateInfo{};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;

        if (vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &outStagingBuffer, &outStagingAllocation, nullptr) != VK_SUCCESS) {
            LOG_ERROR("[ScreenshotCapture] Failed to allocate readback staging buffer.");
            return false;
        }

        // The image arrives here in PRESENT_SRC_KHR (RecordFrame's own final barrier already left
        // it there -- see this file's own header comment for why this MUST run inside the frame's
        // own command buffer, before that frame's vkQueuePresentKHR): transition to
        // TRANSFER_SRC_OPTIMAL for the copy, then back to PRESENT_SRC_KHR so this frame's own
        // present sees the same state as if no capture had happened.
        renderer::VulkanUtils::TransitionImageLayout(
            cmd, swapchainImage,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;   // Tightly packed.
        region.bufferImageHeight = 0; // Tightly packed.
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = { extent.width, extent.height, 1 };

        vkCmdCopyImageToBuffer(cmd, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                outStagingBuffer, 1, &region);

        renderer::VulkanUtils::TransitionImageLayout(
            cmd, swapchainImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_NONE);

        return true;
    }

    bool ScreenshotCapture::WriteStagingToBmp(
        VmaAllocator allocator,
        VkBuffer stagingBuffer,
        VmaAllocation stagingAllocation,
        VkExtent2D extent,
        const std::string& outputFilePath) {

        void* mapped = nullptr;
        bool ok = false;
        if (vmaMapMemory(allocator, stagingAllocation, &mapped) == VK_SUCCESS) {
            const size_t rowPitchBytes = static_cast<size_t>(extent.width) * 4u; // Tightly packed copy in RecordCapture().
            ok = WriteBmp32(outputFilePath, extent.width, extent.height,
                             reinterpret_cast<const uint8_t*>(mapped), rowPitchBytes);
            vmaUnmapMemory(allocator, stagingAllocation);
        } else {
            LOG_ERROR("[ScreenshotCapture] Failed to map readback staging buffer.");
        }

        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        return ok;
    }

}

#endif // NDEBUG
