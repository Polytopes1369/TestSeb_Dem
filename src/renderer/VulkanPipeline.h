#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <vector>
#include <string>

class VulkanPipeline {
public:
    static VkPipelineLayout CreatePipelineLayout(VkDevice device, VkDescriptorSetLayout bindlessLayout);

    // Simplification : pas besoin de RenderPass, on utilise PipelineRenderingCreateInfo
    // Visibility Buffer: 2 color attachments (ClusterID + local TriangleID, both single-channel
    // uint formats) instead of one classic lit-color attachment -- see VulkanContext.h's
    // kVisBufferFormat comment for why 2 separate R32_UINT images are used over one R64_UINT.
    static VkPipeline CreateGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule vertShader, VkShaderModule fragShader, const std::array<VkFormat, 2>& colorFormats, VkFormat depthFormat);

    static VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code);

    static VkPipeline CreateComputePipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule computeShader);
};