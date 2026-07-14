#include "VulkanContext.h"
#include "VulkanPipeline.h"
#include "core/Logger.h"
#include "core/EntityData.h"
#include "core/EngineConfig.h" // Centralized engine configurations (config::WINDOW_WIDTH, etc.)
#include "renderer/RenderTypes.h" // renderer::Vertex, used to interpret the DEBUG readback bytes
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <format>

// Validation Layers array definition
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    LogLevel level = LogLevel::Info;
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        level = LogLevel::Error;
        __debugbreak();
    }
    else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        level = LogLevel::Warning;
    }
    Logger::Log(level, pCallbackData->pMessage);
    return VK_FALSE;
}

void VulkanContext::Init(std::string_view appName, GLFWwindow* window) {
    CreateInstance(appName);
    SetupDebugMessenger();
    CreateSurface(window);
    PickPhysicalDevice();
    CreateLogicalDevice();

    CreateCommandPool();
    AllocateCommandBuffer();

    CreateBindlessDescriptorSetLayout();
    CreateDescriptorPool();
    AllocateBindlessDescriptorSet();

    m_PipelineLayout = VulkanPipeline::CreatePipelineLayout(m_Device, m_BindlessLayout);

    // Initialize VMA Allocator
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.instance = m_Instance;
    allocatorInfo.physicalDevice = m_PhysicalDevice;
    allocatorInfo.device = m_Device;
    if (vmaCreateAllocator(&allocatorInfo, &m_Allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator!");
    }

    // Allocate Entity Tracking Buffer
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = sizeof(core::EntityData) * 10000;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateBuffer(m_Allocator, &bufferInfo, &allocInfo, &m_EntityBuffer, &m_EntityAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate entity buffer!");
    }

    // Allocate Procedural Geometry Buffers (512KB vertices, 128KB indices, 256B for internal parameters)
    VkBufferCreateInfo vInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    vInfo.size = 512 * 1024;
    // TRANSFER_SRC_BIT is required for the DEBUG readback (vkCmdCopyBuffer) in DebugReadbackGeometrySample()
    vInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo vAlloc{ .usage = VMA_MEMORY_USAGE_GPU_ONLY };
    if (vmaCreateBuffer(m_Allocator, &vInfo, &vAlloc, &m_VertexBuffer, &m_VertexAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate vertex SSBO!");
    }

    VkBufferCreateInfo iInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    iInfo.size = 128 * 1024;
    // TRANSFER_SRC_BIT is required for the DEBUG readback (vkCmdCopyBuffer) in DebugReadbackGeometrySample()
    iInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vmaCreateBuffer(m_Allocator, &iInfo, &vAlloc, &m_IndexBuffer, &m_IndexAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate index SSBO!");
    }

    VkBufferCreateInfo pInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    pInfo.size = 256;
    pInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VmaAllocationCreateInfo pAlloc{ .usage = VMA_MEMORY_USAGE_CPU_TO_GPU };
    if (vmaCreateBuffer(m_Allocator, &pInfo, &pAlloc, &m_ParamsBuffer, &m_ParamsAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate compute uniform block!");
    }

    CreateSwapchain(window);
    CreateImageViews();
    CreateSyncObjects();

    CreatePipelinesAndDescriptors();
    GenerateGeometry(); // Dispatches the Icosphere generator compute shader once at startup
}

void VulkanContext::CreateInstance(std::string_view appName) {
    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = appName.data();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "DemosceneEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    createInfo.pApplicationInfo = &appInfo;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (m_EnableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_Instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan Instance!");
    }
}

void VulkanContext::SetupDebugMessenger() {
    if (!m_EnableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    // Ajouter le flag de message fatal pour casser l'exécution immédiatement
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_Instance, "vkCreateDebugUtilsMessengerEXT");
    if (func == nullptr || func(m_Instance, &createInfo, nullptr, &m_DebugMessenger) != VK_SUCCESS) {
        throw std::runtime_error("Failed to set up debug messenger!");
    }
}

void VulkanContext::CreateSurface(GLFWwindow* window) {
    if (glfwCreateWindowSurface(m_Instance, window, nullptr, &m_Surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface via GLFW!");
    }
}

void VulkanContext::PickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);
    if (deviceCount == 0) throw std::runtime_error("No GPUs with Vulkan support found!");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_Instance, &deviceCount, devices.data());
    m_PhysicalDevice = devices[0];
}

void VulkanContext::CreateLogicalDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_PhysicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_PhysicalDevice, &queueFamilyCount, queueFamilies.data());

    bool found = false;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            m_GraphicsQueueFamilyIndex = i;
            found = true;
            break;
        }
    }
    if (!found) throw std::runtime_error("Could not find combined graphics and compute queue!");

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueCreateInfo.queueFamilyIndex = m_GraphicsQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceVulkan13Features features13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceFeatures2 deviceFeatures2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    deviceFeatures2.pNext = &features13;

    VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    createInfo.pNext = &deviceFeatures2;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;

    const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(m_PhysicalDevice, &createInfo, nullptr, &m_Device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Logical Device!");
    }

    vkGetDeviceQueue(m_Device, m_GraphicsQueueFamilyIndex, 0, &m_GraphicsQueue);
}

void VulkanContext::CreateCommandPool() {
    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_GraphicsQueueFamilyIndex;
    if (vkCreateCommandPool(m_Device, &poolInfo, nullptr, &m_CommandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Command Pool!");
    }
}

void VulkanContext::AllocateCommandBuffer() {
    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = m_CommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(m_Device, &allocInfo, &m_CommandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate principal Command Buffer!");
    }
}

void VulkanContext::CreateBindlessDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_BindlessLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create bindless set layout!");
    }
}

void VulkanContext::CreateDescriptorPool() {
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create main descriptor pool!");
    }
}

void VulkanContext::AllocateBindlessDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_DescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_BindlessLayout;
    if (vkAllocateDescriptorSets(m_Device, &allocInfo, &m_BindlessSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate bindless descriptor set!");
    }
}

void VulkanContext::CreateSwapchain(GLFWwindow* window) {
    m_SwapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    m_SwapchainExtent = { config::WINDOW_WIDTH, config::WINDOW_HEIGHT }; // Linked directly to config constants

    VkSwapchainCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    createInfo.surface = m_Surface;
    createInfo.minImageCount = 2;
    createInfo.imageFormat = m_SwapchainImageFormat;
    createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    createInfo.imageExtent = m_SwapchainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // Fixed incorrect enum value
    createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(m_Device, &createInfo, nullptr, &m_Swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Swapchain!");
    }

    uint32_t imageCount = 0;
    vkGetSwapchainImagesKHR(m_Device, m_Swapchain, &imageCount, nullptr);
    m_SwapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_Device, m_Swapchain, &imageCount, m_SwapchainImages.data());
}

void VulkanContext::CreateImageViews() {
    m_SwapchainImageViews.resize(m_SwapchainImages.size());
    for (size_t i = 0; i < m_SwapchainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        createInfo.image = m_SwapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = m_SwapchainImageFormat;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_Device, &createInfo, nullptr, &m_SwapchainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to generate Swapchain Image Views!");
        }
    }
}

void VulkanContext::CreateSyncObjects() {
    VkSemaphoreCreateInfo semaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    if (vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &m_ImageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &m_RenderFinishedSemaphore) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create synchronization semaphores!");
    }
}

void VulkanContext::CreatePipelinesAndDescriptors() {
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_GeometryDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Geometry Descriptor Pool!");
    }

    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0; // Vertices SSBO
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1; // Indices SSBO
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

    bindings[2].binding = 2; // Parameters UBO
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_GeometryLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Geometry Descriptor Layout!");
    }

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_GeometryDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_GeometryLayout;
    if (vkAllocateDescriptorSets(m_Device, &allocInfo, &m_GeometryDescriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate Geometry Descriptor Set!");
    }

    VkDescriptorBufferInfo vertInfo{ m_VertexBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo indexInfo{ m_IndexBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo paramsInfo{ m_ParamsBuffer, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_GeometryDescriptorSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &vertInfo;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &indexInfo;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].pBufferInfo = &paramsInfo;

    vkUpdateDescriptorSets(m_Device, 3, writes, 0, nullptr);

    // Compute layout setup
    VkPipelineLayoutCreateInfo computeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    computeLayoutInfo.setLayoutCount = 1;
    computeLayoutInfo.pSetLayouts = &m_GeometryLayout;
    if (vkCreatePipelineLayout(m_Device, &computeLayoutInfo, nullptr, &m_ComputePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Compute Pipeline Layout!");
    }

    auto computeCode = ReadShaderFile("shaders/geom_icosphere.comp.spv");
    VkShaderModule computeModule = VulkanPipeline::CreateShaderModule(m_Device, computeCode);

    VkComputePipelineCreateInfo computePipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    computePipelineInfo.layout = m_ComputePipelineLayout;
    computePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computePipelineInfo.stage.module = computeModule;
    computePipelineInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &m_ComputePipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to generate Icosphere Compute Pipeline!");
    }
    vkDestroyShaderModule(m_Device, computeModule, nullptr);

    // Graphics layout setup with custom PushConstant sizing matching CameraPushConstants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(CameraPushConstants);

    VkPipelineLayoutCreateInfo graphicsLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    graphicsLayoutInfo.setLayoutCount = 1;
    graphicsLayoutInfo.pSetLayouts = &m_GeometryLayout;
    graphicsLayoutInfo.pushConstantRangeCount = 1;
    graphicsLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(m_Device, &graphicsLayoutInfo, nullptr, &m_GraphicsPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Graphics Pipeline Layout!");
    }

    auto vertCode = ReadShaderFile("shaders/draw.vert.spv");
    auto fragCode = ReadShaderFile("shaders/draw.frag.spv");
    VkShaderModule vertModule = VulkanPipeline::CreateShaderModule(m_Device, vertCode);
    VkShaderModule fragModule = VulkanPipeline::CreateShaderModule(m_Device, fragCode);

    m_GraphicsPipeline = VulkanPipeline::CreateGraphicsPipeline(m_Device, m_GraphicsPipelineLayout, vertModule, fragModule, m_SwapchainImageFormat);

    vkDestroyShaderModule(m_Device, vertModule, nullptr);
    vkDestroyShaderModule(m_Device, fragModule, nullptr);
}

void VulkanContext::GenerateGeometry() {
    struct Params {
        float radius = 1.0f;
        uint32_t subdiv = 16;
        uint32_t meshID = 0;
        float materialID = 0.0f;
        uint32_t vertexOffset = 0;
        uint32_t indexOffset = 0;
    } params;

    void* data;
    if (vmaMapMemory(m_Allocator, m_ParamsAllocation, &data) == VK_SUCCESS) {
        std::memcpy(data, &params, sizeof(Params));
        vmaUnmapMemory(m_Allocator, m_ParamsAllocation);
    }

    // --- DEBUG: log the geometry parameters actually uploaded to the compute shader,
    // and the vertex/index counts the icosphere generator is expected to produce. ---
    constexpr uint32_t kIcosphereFaceCount = 20u;
    uint32_t vertsPerFace = (params.subdiv + 1u) * (params.subdiv + 2u) / 2u;
    uint32_t expectedVertexCount = vertsPerFace * kIcosphereFaceCount;
    uint32_t expectedIndexCount = kIcosphereFaceCount * params.subdiv * params.subdiv * 6u;
    Logger::Log(LogLevel::Info, std::format(
        "[GenerateGeometry] radius={} subdiv={} vertexOffset={} indexOffset={} -> expectedVertexCount={} expectedIndexCount={} (buffers hold {} verts / {} indices max)",
        params.radius, params.subdiv, params.vertexOffset, params.indexOffset,
        expectedVertexCount, expectedIndexCount,
        (512u * 1024u) / sizeof(renderer::Vertex), (128u * 1024u) / sizeof(uint32_t)));

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = m_CommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(m_Device, &allocInfo, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed allocating single submit compute command buffer!");
    }

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ComputePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ComputePipelineLayout, 0, 1, &m_GeometryDescriptorSet, 0, nullptr);

    // Dispatch sizing must match geom_icosphere.comp's local_size = (8, 8, 1):
    // X and Y cover the triangular grid coordinates (i, j) in [0, subdiv],
    // Z covers the 20 base icosahedron faces (one face per Z-invocation since local_size_z = 1).
    constexpr uint32_t kLocalSizeXY = 8u;

    uint32_t gridExtent = params.subdiv + 1u; // valid i,j range is [0, subdiv] inclusive
    uint32_t groupCountXY = (gridExtent + kLocalSizeXY - 1u) / kLocalSizeXY;
    uint32_t groupCountZ = kIcosphereFaceCount;

    // --- DEBUG: log the actual dispatch dimensions and resulting invocation coverage. ---
    Logger::Log(LogLevel::Info, std::format(
        "[GenerateGeometry] Dispatching compute: groupCount=({}, {}, {}) -> covers invocations x,y in [0,{}) z in [0,{})",
        groupCountXY, groupCountXY, groupCountZ,
        groupCountXY * kLocalSizeXY, groupCountZ));

    // Dispatch across the (i, j, face) domain
    vkCmdDispatch(cmd, groupCountXY, groupCountXY, groupCountZ);

    // Memory layout safe transition barrier execution
    VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &memBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

    vkQueueWaitIdle(m_GraphicsQueue);
    vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &cmd);

    Logger::Log(LogLevel::Info, "GPU Geometry generated successfully during bootstrap context phase.");

    // --- DEBUG: read back a sample of generated vertices/indices to verify the compute
    // shader actually wrote valid data across multiple icosahedron faces (not just face 0). ---
    DebugReadbackGeometrySample(vertsPerFace, expectedIndexCount);
}

void VulkanContext::DebugReadbackGeometrySample(uint32_t vertsPerFace, uint32_t expectedIndexCount) {
    // Sample window: covers the tail of face 0 and the head of face 1, so the log proves
    // (or disproves) that the compute dispatch wrote data past the first face.
    const uint32_t maxVertsInBuffer = (512u * 1024u) / sizeof(renderer::Vertex);
    const uint32_t sampleVertexCount = std::min<uint32_t>(vertsPerFace + 8u, maxVertsInBuffer);
    const uint32_t sampleIndexCount = std::min<uint32_t>(12u, expectedIndexCount);

    const VkDeviceSize vertexSampleBytes = static_cast<VkDeviceSize>(sampleVertexCount) * sizeof(renderer::Vertex);
    const VkDeviceSize indexSampleBytes = static_cast<VkDeviceSize>(sampleIndexCount) * sizeof(uint32_t);

    VkBuffer stagingVertexBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingVertexAlloc = VK_NULL_HANDLE;
    VkBufferCreateInfo stagingVertexInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stagingVertexInfo.size = vertexSampleBytes;
    stagingVertexInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo stagingAllocInfo{ .usage = VMA_MEMORY_USAGE_GPU_TO_CPU };
    if (vmaCreateBuffer(m_Allocator, &stagingVertexInfo, &stagingAllocInfo, &stagingVertexBuffer, &stagingVertexAlloc, nullptr) != VK_SUCCESS) {
        Logger::Log(LogLevel::Error, "[DebugReadback] Failed to allocate vertex staging buffer!");
        return;
    }

    VkBuffer stagingIndexBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingIndexAlloc = VK_NULL_HANDLE;
    VkBufferCreateInfo stagingIndexInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stagingIndexInfo.size = indexSampleBytes;
    stagingIndexInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vmaCreateBuffer(m_Allocator, &stagingIndexInfo, &stagingAllocInfo, &stagingIndexBuffer, &stagingIndexAlloc, nullptr) != VK_SUCCESS) {
        Logger::Log(LogLevel::Error, "[DebugReadback] Failed to allocate index staging buffer!");
        vmaDestroyBuffer(m_Allocator, stagingVertexBuffer, stagingVertexAlloc);
        return;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAllocInfo.commandPool = m_CommandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd) != VK_SUCCESS) {
        Logger::Log(LogLevel::Error, "[DebugReadback] Failed allocating readback command buffer!");
        vmaDestroyBuffer(m_Allocator, stagingVertexBuffer, stagingVertexAlloc);
        vmaDestroyBuffer(m_Allocator, stagingIndexBuffer, stagingIndexAlloc);
        return;
    }

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // GenerateGeometry() already vkQueueWaitIdle'd after the compute dispatch before calling
    // this function, so the compute writes are complete and visible: a plain copy is safe here.
    VkBufferCopy vertexCopyRegion{ 0, 0, vertexSampleBytes };
    vkCmdCopyBuffer(cmd, m_VertexBuffer, stagingVertexBuffer, 1, &vertexCopyRegion);

    VkBufferCopy indexCopyRegion{ 0, 0, indexSampleBytes };
    vkCmdCopyBuffer(cmd, m_IndexBuffer, stagingIndexBuffer, 1, &indexCopyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(m_GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_GraphicsQueue);
    vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &cmd);

    // Log a handful of representative vertices: the first vertex of face 0, the last vertex
    // of face 0, and the first vertex of face 1 (proves writes actually crossed a face boundary).
    void* mappedVerts = nullptr;
    if (vmaMapMemory(m_Allocator, stagingVertexAlloc, &mappedVerts) == VK_SUCCESS) {
        const renderer::Vertex* verts = reinterpret_cast<const renderer::Vertex*>(mappedVerts);

        auto logVertex = [](const char* label, const renderer::Vertex& v) {
            float len = std::sqrt(v.position.x * v.position.x + v.position.y * v.position.y + v.position.z * v.position.z);
            Logger::Log(LogLevel::Info, std::format(
                "[DebugReadback] {} pos=({:.4f}, {:.4f}, {:.4f}) |pos|={:.4f} meshID={} materialID={}",
                label, v.position.x, v.position.y, v.position.z, len, v.meshID, v.materialID));
            };

        logVertex("vertex[0] (face 0 head)", verts[0]);

        uint32_t faceZeroLastIdx = vertsPerFace - 1u;
        if (faceZeroLastIdx < sampleVertexCount) {
            logVertex("vertex[faceZeroLast] (face 0 tail)", verts[faceZeroLastIdx]);
        }

        uint32_t faceOneFirstIdx = vertsPerFace;
        if (faceOneFirstIdx < sampleVertexCount) {
            logVertex("vertex[faceOneFirst] (face 1 head)", verts[faceOneFirstIdx]);
        }

        vmaUnmapMemory(m_Allocator, stagingVertexAlloc);
    }
    else {
        Logger::Log(LogLevel::Error, "[DebugReadback] Failed to map vertex staging buffer for readback!");
    }

    void* mappedIndices = nullptr;
    if (vmaMapMemory(m_Allocator, stagingIndexAlloc, &mappedIndices) == VK_SUCCESS) {
        const uint32_t* idx = reinterpret_cast<const uint32_t*>(mappedIndices);
        std::string idxDump;
        for (uint32_t i = 0; i < sampleIndexCount; ++i) {
            idxDump += std::format("{} ", idx[i]);
        }
        Logger::Log(LogLevel::Info, std::format("[DebugReadback] First {} indices: {}", sampleIndexCount, idxDump));
        vmaUnmapMemory(m_Allocator, stagingIndexAlloc);
    }
    else {
        Logger::Log(LogLevel::Error, "[DebugReadback] Failed to map index staging buffer for readback!");
    }

    vmaDestroyBuffer(m_Allocator, stagingVertexBuffer, stagingVertexAlloc);
    vmaDestroyBuffer(m_Allocator, stagingIndexBuffer, stagingIndexAlloc);
}

std::vector<char> VulkanContext::ReadShaderFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open explicit SPIR-V file: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

void VulkanContext::Shutdown() {
    if (m_Device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_Device);
    }

    if (m_VertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_Allocator, m_VertexBuffer, m_VertexAllocation);
    }
    if (m_IndexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_Allocator, m_IndexBuffer, m_IndexAllocation);
    }
    if (m_ParamsBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_Allocator, m_ParamsBuffer, m_ParamsAllocation);
    }
    if (m_EntityBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_Allocator, m_EntityBuffer, m_EntityAllocation);
    }

    if (m_ImageAvailableSemaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(m_Device, m_ImageAvailableSemaphore, nullptr);
    }
    if (m_RenderFinishedSemaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(m_Device, m_RenderFinishedSemaphore, nullptr);
    }

    if (m_ComputePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_Device, m_ComputePipeline, nullptr);
    }
    if (m_GraphicsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_Device, m_GraphicsPipeline, nullptr);
    }

    if (m_PipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
    }
    if (m_ComputePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_Device, m_ComputePipelineLayout, nullptr);
    }
    if (m_GraphicsPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_Device, m_GraphicsPipelineLayout, nullptr);
    }

    for (auto imageView : m_SwapchainImageViews) {
        vkDestroyImageView(m_Device, imageView, nullptr);
    }
    m_SwapchainImageViews.clear();

    if (m_Swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_Device, m_Swapchain, nullptr);
    }

    if (m_GeometryDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_Device, m_GeometryDescriptorPool, nullptr);
    }
    if (m_GeometryLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_Device, m_GeometryLayout, nullptr);
    }

    if (m_DescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
    }
    if (m_BindlessLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_Device, m_BindlessLayout, nullptr);
    }

    if (m_CommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_Device, m_CommandPool, nullptr);
    }

    if (m_Allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(m_Allocator);
    }

    if (m_Device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_Device, nullptr);
    }

    if (m_Surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
    }

    if (m_EnableValidationLayers && m_DebugMessenger != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_Instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(m_Instance, m_DebugMessenger, nullptr);
        }
    }

    if (m_Instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_Instance, nullptr);
    }
}