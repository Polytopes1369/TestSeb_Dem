#include "renderer/vulkan/VulkanPipeline.h"
#include "core/Logger.h"
#include "core/ResourcePath.h"
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

std::vector<char> VulkanPipeline::ReadShaderFile(const std::string& path) {
    // Shader paths are baked-in relative literals (e.g. "shaders/geom_cone.comp.spv") resolved
    // against the exe's own directory -- the process CWD depends on how DemoSceneVK.exe was
    // launched and is not reliably the build/deploy directory the .spv files live next to.
    const std::filesystem::path resolvedPath = core::ResolveExeRelativePath(path);
    std::ifstream file(resolvedPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::format("VulkanPipeline: failed to open SPIR-V file: {}", resolvedPath.string()));
    }
    const size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
    return buffer;
}

VkShaderModule VulkanPipeline::LoadShaderModule(VkDevice device, const std::string& path) {
    return CreateShaderModule(device, ReadShaderFile(path));
}

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
    pipelineLayoutInfo.pSetLayouts = &bindlessLayout; // The bindless layout created previously

    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &layout));
    return layout;
}

VkPipeline VulkanPipeline::CreateGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule vertShader, VkShaderModule fragShader, const std::array<VkFormat, 2>& colorFormats, VkFormat depthFormat) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; // Using standard Vertex stage for now
    stages[0].module = vertShader;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragShader;
    stages[1].pName = "main";

    // Bindless Architecture: Vertex Input is completely empty. We read directly from SSBO.
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Rasterization: Crucial for a sphere without a Depth Buffer.
    // Back-face culling ensures we don't render triangles behind the sphere.
    // NOTE: PerspectiveVulkan() negates the Y scale (m[5] = -g) to correct for Vulkan's
    // Y-down NDC convention. That Y flip reverses the apparent 2D winding of every triangle
    // in framebuffer space, so front-facing (CCW in object/world space, verified consistent
    // across all 20 icosahedron base faces) triangles arrive as CLOCKWISE on screen.
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Visibility Buffer output: both attachments are single-channel R32_UINT ID buffers, not a
    // classic lit RGBA color -- colorWriteMask is restricted to the R component (the only channel
    // these formats have) and blendEnable is left VK_FALSE, since blending is not defined (and
    // would be a validation error to enable) for an integer format in the first place.
    VkPipelineColorBlendAttachmentState colorBlendAttachments[2]{};
    colorBlendAttachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    colorBlendAttachments[0].blendEnable = VK_FALSE;
    colorBlendAttachments[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    colorBlendAttachments[1].blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 2;
    colorBlending.pAttachments = colorBlendAttachments;

    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // Dynamic Rendering configuration -- 2 color attachments (Visibility Buffer: ClusterID +
    // local TriangleID) instead of 1.
    VkPipelineRenderingCreateInfo pipelineRendering{};
    pipelineRendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRendering.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
    pipelineRendering.pColorAttachmentFormats = colorFormats.data();
    pipelineRendering.depthAttachmentFormat = depthFormat;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = (depthFormat != VK_FORMAT_UNDEFINED) ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = (depthFormat != VK_FORMAT_UNDEFINED) ? VK_TRUE : VK_FALSE;
    // Reversed-Z (see maths::mat4::PerspectiveVulkan's own comment): larger NDC depth is now
    // nearer, so the surviving fragment at a pixel is the one with the GREATER depth value. The
    // only live consumer of this shared helper is renderer::ClusterHardwareRasterPass (the main
    // camera's VisBuffer raster); VulkanContext's own separate, dead legacy m_GraphicsPipeline also
    // uses this helper but is never recorded by main.cpp's frame loop.
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = layout;
    pipelineInfo.pNext = &pipelineRendering;

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

VkDescriptorSetLayout VulkanPipeline::CreateBuildDispatchIndirectArgsSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SourceCountSSBO
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // DispatchArgsSSBO

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &setLayout));
    return setLayout;
}

void VulkanPipeline::CreateBuildDispatchIndirectArgsPipeline(
    VkDevice device,
    VkDescriptorSetLayout setLayout,
    VkPipelineLayout& outPipelineLayout,
    VkPipeline& outPipeline
) {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 2 * sizeof(uint32_t); // Matches BuildDispatchArgsPushConstants { workgroupSize; perElementMultiplier; }.

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &outPipelineLayout));

    VkShaderModule shaderModule = LoadShaderModule(device, "shaders/BuildDispatchIndirectArgs.comp.spv");
    outPipeline = CreateComputePipeline(device, outPipelineLayout, shaderModule);
    vkDestroyShaderModule(device, shaderModule, nullptr);
}
