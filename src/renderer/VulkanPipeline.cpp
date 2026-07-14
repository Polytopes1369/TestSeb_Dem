#include "VulkanPipeline.h"
#include "core/Logger.h"
#include <fstream>

VkShaderModule VulkanPipeline::CreateShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
    return shaderModule;
}

VkPipelineLayout VulkanPipeline::CreatePipelineLayout(VkDevice device, VkDescriptorSetLayout bindlessLayout) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &bindlessLayout; // Le layout bindless créé précédemment

    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &layout));
    return layout;
}

VkPipeline VulkanPipeline::CreateGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule meshShader, VkShaderModule fragShader) {
    // Pipeline Shader Stages
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    stages[0].module = meshShader;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShader;
    stages[1].pName = "main";

    // Dynamic Rendering configuration (le cœur du sujet)
    VkPipelineRenderingCreateInfo pipelineRendering{};
    pipelineRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRendering.colorAttachmentCount = 1;
    pipelineRendering.pColorAttachmentFormats = new VkFormat(VK_FORMAT_B8G8R8A8_SRGB); // À adapter selon swapchain

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pNext = &pipelineRendering; // C'est ici que le Dynamic Rendering est lié
    pipelineInfo.layout = layout;

    // ... Remplir Rasterizer, Multisampling, ColorBlend, etc. (simplifié ici)

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
    return pipeline;
}

VkPipeline VulkanPipeline::CreateComputePipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule computeShader) {
    VkComputePipelineCreateInfo computeInfo{};
    computeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computeInfo.layout = layout;
    computeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeInfo.stage.module = computeShader;
    computeInfo.stage.pName = "main";

    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computeInfo, nullptr, &pipeline));
    return pipeline;
}