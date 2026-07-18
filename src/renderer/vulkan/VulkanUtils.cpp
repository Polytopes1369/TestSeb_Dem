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

    void VulkanUtils::RecordMemoryBarrier(
        VkCommandBuffer cmd,
        VkPipelineStageFlags2 srcStage,
        VkAccessFlags2 srcAccess,
        VkPipelineStageFlags2 dstStage,
        VkAccessFlags2 dstAccess
    ) {
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
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

    void VulkanUtils::WriteRayBuffersDescriptorSet(
        VkDevice device,
        VkDescriptorSet raySet,
        VkBuffer rayBuffer,
        VkDeviceSize rayBufferSize,
        VkBuffer resultBuffer,
        VkDeviceSize resultBufferSize
    ) {
        VkDescriptorBufferInfo rayInfo{ rayBuffer, 0, rayBufferSize };
        VkDescriptorBufferInfo resultInfo{ resultBuffer, 0, resultBufferSize };

        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = raySet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &rayInfo;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = raySet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &resultInfo;

        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    }

    void VulkanUtils::WriteSharedGeometryBindings(
        VkDevice device,
        VkDescriptorSet set,
        uint32_t baseBinding,
        VkAccelerationStructureKHR tlas,
        VkBuffer vertexBuffer,
        VkBuffer indexBuffer,
        VkBuffer drawRangeBuffer
    ) {
        VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        accelWrite.accelerationStructureCount = 1;
        accelWrite.pAccelerationStructures = &tlas;

        VkDescriptorBufferInfo vertexBufferInfo{ vertexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo indexBufferInfo{ indexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo drawRangeBufferInfo{ drawRangeBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[4]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].pNext = &accelWrite;
        writes[0].dstSet = set;
        writes[0].dstBinding = baseBinding;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = set;
        writes[1].dstBinding = baseBinding + 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &vertexBufferInfo;

        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[2].dstSet = set;
        writes[2].dstBinding = baseBinding + 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &indexBufferInfo;

        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[3].dstSet = set;
        writes[3].dstBinding = baseBinding + 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &drawRangeBufferInfo;

        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
    }

    VulkanUtils::DescriptorSetLayoutPoolAndSet VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(
        VkDevice device,
        std::span<const VkDescriptorSetLayoutBinding> bindings,
        std::span<const VkDescriptorPoolSize> poolSizes
    ) {
        DescriptorSetLayoutPoolAndSet result{};

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &result.layout) != VK_SUCCESS) {
            throw std::runtime_error("VulkanUtils: Failed to create descriptor set layout");
        }

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &result.pool) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(device, result.layout, nullptr);
            throw std::runtime_error("VulkanUtils: Failed to create descriptor pool");
        }

        VkDescriptorSetAllocateInfo allocSet{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocSet.descriptorPool = result.pool;
        allocSet.descriptorSetCount = 1;
        allocSet.pSetLayouts = &result.layout;
        if (vkAllocateDescriptorSets(device, &allocSet, &result.set) != VK_SUCCESS) {
            vkDestroyDescriptorPool(device, result.pool, nullptr);
            vkDestroyDescriptorSetLayout(device, result.layout, nullptr);
            throw std::runtime_error("VulkanUtils: Failed to allocate descriptor set");
        }

        return result;
    }

    void VulkanUtils::CreateStorageSampledImage2D(
        VmaAllocator allocator,
        VkDevice device,
        VkFormat format,
        VkExtent2D extent,
        VkImage& outImage,
        VmaAllocation& outAllocation,
        VkImageView& outView
    ) {
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = { extent.width, extent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage, &outAllocation, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("VulkanUtils: Failed to create storage/sampled 2D image");
        }

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = outImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device, &viewInfo, nullptr, &outView) != VK_SUCCESS) {
            throw std::runtime_error("VulkanUtils: Failed to create storage/sampled 2D image view");
        }
    }

    void VulkanUtils::ClearComputeImageToGeneral(
        VkCommandBuffer cmd,
        VkImage image,
        const VkClearColorValue& clearColor
    ) {
        VkImageSubresourceRange colorRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        TransitionImageLayout(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &colorRange);
        TransitionImageLayout(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
