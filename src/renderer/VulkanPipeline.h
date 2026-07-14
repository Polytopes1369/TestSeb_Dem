#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

class VulkanPipeline {
public:
    static VkPipelineLayout CreatePipelineLayout(VkDevice device, VkDescriptorSetLayout bindlessLayout);

    // Simplification : pas besoin de RenderPass, on utilise PipelineRenderingCreateInfo
    static VkPipeline CreateGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule meshShader, VkShaderModule fragShader);

    static VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code);

    static VkPipeline CreateComputePipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule computeShader);
};