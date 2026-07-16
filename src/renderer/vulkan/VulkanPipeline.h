#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

class VulkanPipeline {
public:
    static VkPipelineLayout CreatePipelineLayout(VkDevice device, VkDescriptorSetLayout bindlessLayout);

    // Simplification: no need for RenderPass, we use PipelineRenderingCreateInfo
    // Visibility Buffer: 2 color attachments (ClusterID + local TriangleID, both single-channel
    // uint formats) instead of one classic lit-color attachment -- see VulkanContext.h's
    // kVisBufferFormat comment for why 2 separate R32_UINT images are used over one R64_UINT.
    static VkPipeline CreateGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule vertShader, VkShaderModule fragShader, const std::array<VkFormat, 2>& colorFormats, VkFormat depthFormat);

    // Single source of truth for SPIR-V loading -- replaces the 17 anonymous-namespace copies
    // scattered across renderer .cpp files (each carrying an identical ReadShaderFile +
    // CreateShaderModule pair). All passes must call these instead of defining their own local
    // versions. Throws std::runtime_error on I/O or Vulkan failure.
    static std::vector<char>  ReadShaderFile(const std::string& path);
    static VkShaderModule     CreateShaderModule(VkDevice device, const std::vector<char>& code);
    // Convenience: ReadShaderFile + CreateShaderModule in one call.
    static VkShaderModule     LoadShaderModule(VkDevice device, const std::string& path);

    static VkPipeline CreateComputePipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule computeShader);
};