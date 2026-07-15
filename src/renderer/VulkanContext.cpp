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
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <format>

// Validation Layers array definition
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

namespace {
    // --- Per-shader Params blocks, mirrored 1:1 from their geom_*.comp UBO/push-constant
    // declarations (all fields are 4-byte scalars, so std140 layout == plain C++ struct
    // layout here: no vec3 members means no extra padding to account for). worldOffsetX/Y/Z
    // are appended at the end of every block and translate the generated geometry to its
    // grid slot directly on the GPU, avoiding a separate per-instance transform stage.

    struct ConeParams {
        float height;
        float bottomRadius;
        float topRadius;
        uint32_t nbSides;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    struct IcosphereParams {
        float radius;
        uint32_t subdiv;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    struct PlaneParams {
        float width;
        float length_;
        uint32_t widthSegs;
        uint32_t lengthSegs;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    struct SphereParams {
        float radius;
        uint32_t sideSegs;
        uint32_t heightSegs;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    struct TorusParams {
        float radius1;
        float radius2;
        uint32_t nbRadSeg;
        uint32_t nbSides;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    struct TubeParams {
        float height;
        uint32_t nbSides;
        float bottomRadius1;
        float bottomRadius2;
        float topRadius1;
        float topRadius2;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    struct CapsuleParams {
        float radius;
        float height;       // cylindrical body length (= total height - 2*radius)
        uint32_t nbSides;
        uint32_t nbHeightSegs;
        uint32_t nbCapSegs;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    struct CylinderParams {
        float radius;
        float height;
        uint32_t nbSides;
        uint32_t nbHeightSegs;
        uint32_t nbCapSegs;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    // CPU mirror of the GLSL EntityTransform struct (struct_custo.glsl): must match its
    // std430 layout exactly (mat4 = 64 bytes, vec3 + pad = 16 bytes) since it is memcpy'd
    // wholesale into m_EntityTransformBuffer every frame.
    struct EntityTransform {
        maths::mat4 rotation;
        float centerX;
        float centerY;
        float centerZ;
        float _pad0;
    };

    // geom_box.comp reads its Params via push constants (not the shared UBO), because each
    // of its 6 face dispatches also needs distinct specialization constants baked per-pipeline.
    struct BoxPushConstants {
        float faceWidth;
        float faceHeight;
        float lengthOffset; // constant coordinate along wAxis (±half box size)
        uint32_t uSegsCount;
        uint32_t vSegsCount;
        uint32_t meshID;
        float materialID;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        float worldOffsetX;
        float worldOffsetY;
        float worldOffsetZ;
    };

    // Specialization constants for geom_box.comp's 6 face dispatches (constant_id 0..5:
    // uAxis, vAxis, wAxis, faceMode, udir, vdir — axis indices 0=x,1=y,2=z, matching the
    // shader's setComp() convention). Each face's (uAxis,vAxis,wAxis,udir,vdir) is chosen so
    // that the shader's (a,b,d) triangle — built from vectors vAxis*(segH*vdir) and
    // uAxis*(segW*udir) — satisfies cross(b-a, d-a) == wSign * wAxisVector, which is the
    // condition for consistently outward normals with the winding VulkanPipeline expects
    // (front faces wind CCW in object space, becoming CLOCKWISE on screen after the
    // projection's Y-flip). Derivation: cross(b-a,d-a) = -segH*segW*udir*vdir*P*wAxisVector,
    // where P = +1 if (uAxis,vAxis,wAxis) is an even permutation of (x,y,z) and -1 if odd —
    // so the required condition reduces to udir*vdir*P == -wSign (solved here with vdir=1
    // fixed, udir = -wSign*P).
    struct BoxFaceSpecConstants {
        int32_t uAxis;
        int32_t vAxis;
        int32_t wAxis;
        int32_t faceMode; // 0: w==z, 1: w==y, 2: w==x (selects the shader's UV formula branch)
        float   udir;
        float   vdir;
    };

    constexpr BoxFaceSpecConstants kBoxFaceSpecs[6] = {
        { 0, 1, 2, 0, -1.0f, 1.0f }, // +Z  (u=x,v=y,w=z, P=+1, wSign=+1)
        { 0, 1, 2, 0,  1.0f, 1.0f }, // -Z  (wSign=-1)
        { 0, 2, 1, 1,  1.0f, 1.0f }, // +Y  (u=x,v=z,w=y, P=-1, wSign=+1)
        { 0, 2, 1, 1, -1.0f, 1.0f }, // -Y  (wSign=-1)
        { 1, 2, 0, 2, -1.0f, 1.0f }, // +X  (u=y,v=z,w=x, P=+1, wSign=+1)
        { 1, 2, 0, 2,  1.0f, 1.0f }, // -X  (wSign=-1)
    };

    // lengthOffset sign per face, matching kBoxFaceSpecs order: +Z,-Z,+Y,-Y,+X,-X.
    constexpr float kBoxFaceLengthOffsetSign[6] = { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f };
}

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

    // Author entity data on the CPU (meshID assigned via core::IDManager) before any GPU
    // buffer exists for it, per the engine's CPU-authored / GPU-generated architecture.
    BuildEntityData();

    // Allocate Entity Tracking Buffer
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = sizeof(core::EntityData) * 10000;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateBuffer(m_Allocator, &bufferInfo, &allocInfo, &m_EntityBuffer, &m_EntityAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate entity buffer!");
    }

    // Upload the CPU-authored entity records now that both m_EntityBuffer and the command
    // pool (created above, CreateCommandPool()/AllocateCommandBuffer()) exist.
    UploadEntityData();

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

    // Per-entity rotation buffer: host-visible so UpdateEntityRotations() can re-upload the
    // whole array with a plain memcpy every frame (mirrors the m_ParamsBuffer update pattern).
    VkBufferCreateInfo etInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    etInfo.size = sizeof(EntityTransform) * kEntityCount;
    etInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VmaAllocationCreateInfo etAlloc{ .usage = VMA_MEMORY_USAGE_CPU_TO_GPU };
    if (vmaCreateBuffer(m_Allocator, &etInfo, &etAlloc, &m_EntityTransformBuffer, &m_EntityTransformAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate entity transform buffer!");
    }

    CreateSwapchain(window);
    CreateImageViews();

    // Create Depth Image
    VkImageCreateInfo depthImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
    depthImageInfo.format = m_DepthFormat;
    depthImageInfo.extent = { m_SwapchainExtent.width, m_SwapchainExtent.height, 1 };
    depthImageInfo.mipLevels = 1;
    depthImageInfo.arrayLayers = 1;
    depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo depthAllocInfo{};
    depthAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    if (vmaCreateImage(m_Allocator, &depthImageInfo, &depthAllocInfo, &m_DepthImage, &m_DepthAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image!");
    }

    // Create Depth Image View
    VkImageViewCreateInfo depthViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    depthViewInfo.image = m_DepthImage;
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.format = m_DepthFormat;
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthViewInfo.subresourceRange.baseMipLevel = 0;
    depthViewInfo.subresourceRange.levelCount = 1;
    depthViewInfo.subresourceRange.baseArrayLayer = 0;
    depthViewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_Device, &depthViewInfo, nullptr, &m_DepthImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create depth image view!");
    }

    CreateSyncObjects();

    CreatePipelinesAndDescriptors();
    GenerateGeometry(); // Dispatches all 7 procedural primitive generators once at startup
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
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_GeometryDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Geometry Descriptor Pool!");
    }

    VkDescriptorSetLayoutBinding bindings[5] = {};
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

    bindings[3].binding = 3; // Per-entity rotation SSBO (read only by the vertex shader)
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[4].binding = 4; // Entity data SSBO (CPU-authored, IDManager-assigned meshID; read only by the vertex shader)
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 5;
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
    VkDescriptorBufferInfo entityTransformInfo{ m_EntityTransformBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo entityDataInfo{ m_EntityBuffer, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet writes[5] = {};
    for (int i = 0; i < 5; i++) {
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
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo = &entityTransformInfo;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &entityDataInfo;

    vkUpdateDescriptorSets(m_Device, 5, writes, 0, nullptr);

    // Compute layout setup: shared by all non-box primitive generators, which read their
    // per-dispatch parameters from the Params UBO (binding = 2) bound in m_GeometryDescriptorSet.
    VkPipelineLayoutCreateInfo computeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    computeLayoutInfo.setLayoutCount = 1;
    computeLayoutInfo.pSetLayouts = &m_GeometryLayout;
    if (vkCreatePipelineLayout(m_Device, &computeLayoutInfo, nullptr, &m_ComputePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Compute Pipeline Layout!");
    }

    // Box push-constant layout: geom_box.comp reads its Params via push constants instead of
    // the shared UBO (see BoxPushConstants above), since each face dispatch also needs its
    // own specialization constants baked at pipeline-creation time.
    VkPushConstantRange boxPushConstantRange{};
    boxPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    boxPushConstantRange.offset = 0;
    boxPushConstantRange.size = sizeof(BoxPushConstants);

    VkPipelineLayoutCreateInfo boxComputeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    boxComputeLayoutInfo.setLayoutCount = 1;
    boxComputeLayoutInfo.pSetLayouts = &m_GeometryLayout;
    boxComputeLayoutInfo.pushConstantRangeCount = 1;
    boxComputeLayoutInfo.pPushConstantRanges = &boxPushConstantRange;
    if (vkCreatePipelineLayout(m_Device, &boxComputeLayoutInfo, nullptr, &m_BoxComputePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Box Compute Pipeline Layout!");
    }

    // One compute pipeline per non-box primitive: same descriptor-set-only layout, each with
    // its own shader module generating its own topology into the shared Vertex/Index SSBOs.
    struct SimplePrimitivePipelineDesc {
        const char* shaderFile;
        VkPipeline* outPipeline;
    };
    const SimplePrimitivePipelineDesc simplePrimitives[] = {
        { "shaders/geom_cone.comp.spv",      &m_ConePipeline },
        { "shaders/geom_icosphere.comp.spv", &m_IcospherePipeline },
        { "shaders/geom_plane.comp.spv",     &m_PlanePipeline },
        { "shaders/geom_sphere.comp.spv",    &m_SpherePipeline },
        { "shaders/geom_torus.comp.spv",     &m_TorusPipeline },
        { "shaders/geom_tube.comp.spv",      &m_TubePipeline },
        { "shaders/geom_capsule.comp.spv",   &m_CapsulePipeline },
        { "shaders/geom_cylinder.comp.spv",  &m_CylinderPipeline },
    };
    for (const auto& desc : simplePrimitives) {
        auto code = ReadShaderFile(desc.shaderFile);
        VkShaderModule module = VulkanPipeline::CreateShaderModule(m_Device, code);

        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_ComputePipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";

        if (vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, desc.outPipeline) != VK_SUCCESS) {
            throw std::runtime_error(std::string("Failed to create compute pipeline for ") + desc.shaderFile);
        }
        vkDestroyShaderModule(m_Device, module, nullptr);
    }

    // Box: 6 compute pipelines built from the same shader module, one per cube face,
    // differentiated only by VkSpecializationInfo (axis mapping + winding, see kBoxFaceSpecs).
    {
        auto boxCode = ReadShaderFile("shaders/geom_box.comp.spv");
        VkShaderModule boxModule = VulkanPipeline::CreateShaderModule(m_Device, boxCode);

        VkSpecializationMapEntry mapEntries[6] = {
            { 0, offsetof(BoxFaceSpecConstants, uAxis),    sizeof(int32_t) },
            { 1, offsetof(BoxFaceSpecConstants, vAxis),    sizeof(int32_t) },
            { 2, offsetof(BoxFaceSpecConstants, wAxis),    sizeof(int32_t) },
            { 3, offsetof(BoxFaceSpecConstants, faceMode), sizeof(int32_t) },
            { 4, offsetof(BoxFaceSpecConstants, udir),     sizeof(float) },
            { 5, offsetof(BoxFaceSpecConstants, vdir),     sizeof(float) },
        };

        for (uint32_t face = 0; face < 6u; ++face) {
            VkSpecializationInfo specInfo{};
            specInfo.mapEntryCount = 6;
            specInfo.pMapEntries = mapEntries;
            specInfo.dataSize = sizeof(BoxFaceSpecConstants);
            specInfo.pData = &kBoxFaceSpecs[face];

            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_BoxComputePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = boxModule;
            pipelineInfo.stage.pName = "main";
            pipelineInfo.stage.pSpecializationInfo = &specInfo;

            if (vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_BoxFacePipelines[face]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create Box face compute pipeline!");
            }
        }
        vkDestroyShaderModule(m_Device, boxModule, nullptr);
    }

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

    m_GraphicsPipeline = VulkanPipeline::CreateGraphicsPipeline(m_Device, m_GraphicsPipelineLayout, vertModule, fragModule, m_SwapchainImageFormat, m_DepthFormat);

    vkDestroyShaderModule(m_Device, vertModule, nullptr);
    vkDestroyShaderModule(m_Device, fragModule, nullptr);
}

void VulkanContext::DispatchGeometryCompute(
    VkPipeline pipeline,
    VkPipelineLayout layout,
    const void* uboParamsData,
    size_t uboParamsSize,
    const void* pushConstantData,
    size_t pushConstantSize,
    uint32_t groupCountX,
    uint32_t groupCountY,
    uint32_t groupCountZ)
{
    // Every primitive except the box drives geom_*.comp through the shared Params UBO
    // (binding = 2): overwrite it with this dispatch's parameters before recording the
    // command buffer that reads it.
    if (uboParamsData != nullptr) {
        void* mapped = nullptr;
        if (vmaMapMemory(m_Allocator, m_ParamsAllocation, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map Params UBO for geometry dispatch!");
        }
        std::memcpy(mapped, uboParamsData, uboParamsSize);
        vmaUnmapMemory(m_Allocator, m_ParamsAllocation);
    }

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

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &m_GeometryDescriptorSet, 0, nullptr);

    if (pushConstantData != nullptr) {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(pushConstantSize), pushConstantData);
    }

    vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);

    // Memory layout safe transition barrier execution: compute writes to the shared
    // Vertex/Index SSBOs must be visible to the vertex shader stage that reads them at draw time.
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

    // Blocking wait between dispatches: this is a one-time startup pass (not a per-frame
    // path), so trading dispatch-level parallelism for simplicity is an explicit, deliberate
    // choice here — it also guarantees the next dispatch's Params UBO overwrite (above) never
    // races a still-in-flight read of the previous dispatch's parameters.
    vkQueueWaitIdle(m_GraphicsQueue);
    vkFreeCommandBuffers(m_Device, m_CommandPool, 1, &cmd);
}

maths::vec2 VulkanContext::GridSlot(int slotIndex) const {
    // 3x3 grid centered on (0,0,0), in the XZ plane (Y = 0 ground level), spaced
    // kGridSpacing apart. Used both by GenerateGeometry() (to bake each primitive's world
    // position at generation time) and UpdateEntityRotations() (to recover each entity's
    // rotation pivot), so this is the single source of truth for the layout.
    constexpr float kGridSpacing = 3.0f;
    int col = slotIndex % 3;
    int row = slotIndex / 3;
    return maths::vec2{ (col - 1) * kGridSpacing, (row - 1) * kGridSpacing };
}

void VulkanContext::BuildEntityData() {
    // Context 0: this is currently the only "factory" allocating entity IDs in the engine.
    // Sequence starts at 0, so the 9 sequential GetNextID() calls below deterministically
    // produce 0..8 in the low 32 bits, matching kEntityCount's dense-index requirement.
    core::IDManager::Init(0);

    for (uint32_t i = 0; i < kEntityCount; ++i) {
        core::EntityID id = core::IDManager::GetNextID();

        core::EntityData& entity = m_EntityData[i];
        entity.meshID = static_cast<uint32_t>(id & 0xFFFFFFFFu);
        entity.materialID = 0u;
        entity.cellID = 0u;
        entity.flags = 0u;
        core::SetFlag(entity.flags, core::EntityFlags::CastShadows, true);
    }
}

void VulkanContext::UploadEntityData() {
    // m_EntityBuffer is VMA_MEMORY_USAGE_GPU_ONLY (not host-visible), so uploading the
    // CPU-authored m_EntityData requires a temporary host-visible staging buffer plus an
    // explicit GPU-side copy, mirroring the one-time-submit pattern used elsewhere in Init().
    VkDeviceSize uploadSize = sizeof(core::EntityData) * kEntityCount;

    VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    stagingInfo.size = uploadSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo stagingAllocResultInfo{};
    if (vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate entity data staging buffer!");
    }
    std::memcpy(stagingAllocResultInfo.pMappedData, m_EntityData.data(), static_cast<size_t>(uploadSize));

    VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAllocInfo.commandPool = m_CommandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed allocating single submit command buffer for entity data upload!");
    }

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = uploadSize;
    vkCmdCopyBuffer(cmd, stagingBuffer, m_EntityBuffer, 1, &copyRegion);

    // Explicit layout/ownership-free memory barrier: the copy's writes must be visible to the
    // vertex shader's storage-buffer reads (draw.vert, binding = 4) before any draw call.
    VkMemoryBarrier2 memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
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
    vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
}

void VulkanContext::GenerateGeometry() {
    // --- Grid layout: 9 primitives arranged on a 3x3 grid (see GridSlot()). Generation order
    // below is Icosphere-first (so it lands at vertexOffset/indexOffset 0, keeping
    // DebugReadbackGeometrySample's fixed sampling window — designed around a single
    // icosphere living at the start of the buffers — valid unchanged); each primitive's grid
    // *position* is independent of generation order and set via GridSlot(slotIndex) below.
    auto gridSlot = [this](int slotIndex) -> maths::vec2 { return GridSlot(slotIndex); };

    uint32_t runningVertexOffset = 0;
    uint32_t runningIndexOffset = 0;

    Logger::Log(LogLevel::Info, "[GenerateGeometry] Generating 9 procedural primitives on a 3x3 grid...");

    // ---------------------------------------------------------------------------------
    // ICOSPHERE (slot 2 visually) — generated first so it occupies buffer offset 0.
    // ---------------------------------------------------------------------------------
    uint32_t icosphereVertsPerFace = 0;
    uint32_t icosphereIndexCount = 0;
    {
        maths::vec2 slot = gridSlot(2);
        IcosphereParams params{};
        params.radius = 0.8f;
        params.subdiv = 8u;
        params.meshID = m_EntityData[2].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = 0.0f;
        params.worldOffsetZ = slot.y;

        // Dispatch sizing must match geom_icosphere.comp's local_size = (8, 8, 1): X and Y
        // cover the triangular grid coordinates (i, j) in [0, subdiv], Z covers the 20 base
        // icosahedron faces (one face per Z-invocation since local_size_z = 1).
        constexpr uint32_t kIcosphereFaceCount = 20u;
        constexpr uint32_t kLocalSizeXY = 8u;
        uint32_t gridExtent = params.subdiv + 1u; // valid i,j range is [0, subdiv] inclusive
        uint32_t groupCountXY = (gridExtent + kLocalSizeXY - 1u) / kLocalSizeXY;

        DispatchGeometryCompute(m_IcospherePipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0,
            groupCountXY, groupCountXY, kIcosphereFaceCount);

        icosphereVertsPerFace = (params.subdiv + 1u) * (params.subdiv + 2u) / 2u;
        uint32_t totalVerts = icosphereVertsPerFace * kIcosphereFaceCount;
        // Each face writes subdiv*subdiv triangles (barycentric subdivision), 3 indices each —
        // NOT subdiv*subdiv*6 as an earlier debug estimate assumed for the single-primitive case.
        icosphereIndexCount = kIcosphereFaceCount * params.subdiv * params.subdiv * 3u;

        runningVertexOffset += totalVerts;
        runningIndexOffset += icosphereIndexCount;
    }
    // DEBUG: sample-readback the icosphere geometry, exactly as before this feature, to catch
    // regressions in the (still buffer-offset-0) icosphere generation.
    DebugReadbackGeometrySample(icosphereVertsPerFace, icosphereIndexCount);

    // ---------------------------------------------------------------------------------
    // BOX (slot 0) — 6 compute dispatches, one per cube face, chained onto the same meshID.
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(0);
        constexpr float kHalfSize = 0.7f;
        constexpr uint32_t kSegs = 2u; // a flat face only needs its 4 corners

        for (uint32_t face = 0; face < 6u; ++face) {
            BoxPushConstants params{};
            params.faceWidth = kHalfSize * 2.0f;
            params.faceHeight = kHalfSize * 2.0f;
            params.lengthOffset = kBoxFaceLengthOffsetSign[face] * kHalfSize;
            params.uSegsCount = kSegs;
            params.vSegsCount = kSegs;
            params.meshID = m_EntityData[0].meshID;
            params.materialID = 0.0f;
            params.vertexOffset = runningVertexOffset;
            params.indexOffset = runningIndexOffset;
            params.worldOffsetX = slot.x;
            params.worldOffsetY = 0.0f;
            params.worldOffsetZ = slot.y;

            constexpr uint32_t kLocalSizeXY = 8u; // geom_box.comp local_size = (8, 8, 1)
            uint32_t groupCount = (kSegs + kLocalSizeXY - 1u) / kLocalSizeXY;

            DispatchGeometryCompute(m_BoxFacePipelines[face], m_BoxComputePipelineLayout,
                nullptr, 0, &params, sizeof(params),
                groupCount, groupCount, 1);

            runningVertexOffset += kSegs * kSegs;
            runningIndexOffset += (kSegs - 1u) * (kSegs - 1u) * 6u;
        }
    }

    // ---------------------------------------------------------------------------------
    // CONE (slot 1)
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(1);
        ConeParams params{};
        params.height = 1.4f;
        params.bottomRadius = 0.7f;
        params.topRadius = 0.35f;
        params.nbSides = 32u;
        params.meshID = m_EntityData[1].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = -params.height * 0.5f; // recenter: geometry spans y=[0,height]
        params.worldOffsetZ = slot.y;

        uint32_t totalVerts = 4u * (params.nbSides + 1u); // geom_cone.comp local_size_x = 64
        constexpr uint32_t kLocalSizeX = 64u;
        uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

        DispatchGeometryCompute(m_ConePipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

        runningVertexOffset += totalVerts;
        runningIndexOffset += 12u * params.nbSides;
    }

    // ---------------------------------------------------------------------------------
    // PLANE (slot 3)
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(3);
        PlaneParams params{};
        params.width = 1.4f;
        params.length_ = 1.4f;
        params.widthSegs = 2u; // flat surface: 4 corners is all that's needed
        params.lengthSegs = 2u;
        params.meshID = m_EntityData[3].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = 0.0f;
        params.worldOffsetZ = slot.y;

        constexpr uint32_t kLocalSizeXY = 8u; // geom_plane.comp local_size = (8, 8, 1)
        uint32_t groupCountX = (params.widthSegs + kLocalSizeXY - 1u) / kLocalSizeXY;
        uint32_t groupCountY = (params.lengthSegs + kLocalSizeXY - 1u) / kLocalSizeXY;

        DispatchGeometryCompute(m_PlanePipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);

        runningVertexOffset += params.widthSegs * params.lengthSegs;
        runningIndexOffset += 6u * (params.widthSegs - 1u) * (params.lengthSegs - 1u);
    }

    // ---------------------------------------------------------------------------------
    // SPHERE / UV sphere (slot 4)
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(4);
        SphereParams params{};
        params.radius = 0.8f;
        params.sideSegs = 24u;
        params.heightSegs = 16u;
        params.meshID = m_EntityData[4].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = 0.0f;
        params.worldOffsetZ = slot.y;

        uint32_t ringCount = params.heightSegs - 2u; // interior latitude rings (excludes poles)
        uint32_t ringStride = params.sideSegs + 1u;
        uint32_t vertCount = ringCount * ringStride + 2u;
        constexpr uint32_t kLocalSizeX = 64u; // geom_sphere.comp local_size_x = 64
        uint32_t groupCount = (vertCount + kLocalSizeX - 1u) / kLocalSizeX;

        DispatchGeometryCompute(m_SpherePipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

        runningVertexOffset += vertCount;
        // Dense packing after the geom_sphere.comp bottom-cap fix: sideSegs*3 (top fan) +
        // (ringCount-1)*sideSegs*6 (middle quads) + sideSegs*3 (bottom fan) = 6*sideSegs*ringCount.
        runningIndexOffset += 6u * params.sideSegs * ringCount;
    }

    // ---------------------------------------------------------------------------------
    // TORUS (slot 5)
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(5);
        TorusParams params{};
        params.radius1 = 0.7f; // major (ring) radius
        params.radius2 = 0.22f; // minor (tube) radius
        params.nbRadSeg = 32u;
        params.nbSides = 16u;
        params.meshID = m_EntityData[5].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = 0.0f;
        params.worldOffsetZ = slot.y;

        uint32_t vertCount = (params.nbRadSeg + 1u) * (params.nbSides + 1u);
        constexpr uint32_t kLocalSizeX = 64u; // geom_torus.comp local_size_x = 64
        uint32_t groupCount = (vertCount + kLocalSizeX - 1u) / kLocalSizeX;

        DispatchGeometryCompute(m_TorusPipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

        runningVertexOffset += vertCount;
        runningIndexOffset += 6u * params.nbRadSeg * params.nbSides;
    }

    // ---------------------------------------------------------------------------------
    // TUBE (slot 6) — hollow cylinder (pipe): outer/inner wall + top/bottom rings.
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(6);
        TubeParams params{};
        params.height = 1.4f;
        params.nbSides = 24u;
        params.bottomRadius1 = 0.7f;
        params.bottomRadius2 = 0.18f; // wall thickness
        params.topRadius1 = 0.7f;
        params.topRadius2 = 0.18f;
        params.meshID = m_EntityData[6].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = -params.height * 0.5f; // recenter: geometry spans y=[0,height]
        params.worldOffsetZ = slot.y;

        uint32_t capCount = params.nbSides * 2u + 2u;
        uint32_t totalVerts = capCount * 4u; // geom_tube.comp local_size_x = 64
        constexpr uint32_t kLocalSizeX = 64u;
        uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

        DispatchGeometryCompute(m_TubePipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

        runningVertexOffset += totalVerts;
        runningIndexOffset += 24u * params.nbSides;
    }

    // ---------------------------------------------------------------------------------
    // CAPSULE (slot 7) — cylindrical body + two hemispherical caps.
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(7);
        CapsuleParams params{};
        params.radius = 0.5f;
        params.height = 0.8f; // cylindrical body length; full shape spans y=[-radius, height+radius]
        params.nbSides = 24u;
        params.nbHeightSegs = 4u;
        params.nbCapSegs = 8u;
        params.meshID = m_EntityData[7].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = -params.height * 0.5f; // recenter: shape's vertical midpoint is height/2
        params.worldOffsetZ = slot.y;

        uint32_t sideColumns = params.nbSides + 1u;
        uint32_t hemiVerts = (params.nbCapSegs + 1u) * sideColumns;
        uint32_t bodyVerts = (params.nbHeightSegs + 1u) * sideColumns;
        uint32_t totalVerts = hemiVerts * 2u + bodyVerts;
        constexpr uint32_t kLocalSizeX = 64u; // geom_capsule.comp local_size_x = 64
        uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

        DispatchGeometryCompute(m_CapsulePipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

        runningVertexOffset += totalVerts;
        // 2 hemispheres (nbCapSegs*nbSides quads each) + body (nbHeightSegs*nbSides quads), 6 indices/quad.
        runningIndexOffset += 6u * params.nbSides * (2u * params.nbCapSegs + params.nbHeightSegs);
    }

    // ---------------------------------------------------------------------------------
    // CYLINDER (slot 8) — flat top/bottom caps (unlike TUBE, which is hollow).
    // ---------------------------------------------------------------------------------
    {
        maths::vec2 slot = gridSlot(8);
        CylinderParams params{};
        params.radius = 0.7f;
        params.height = 1.4f;
        params.nbSides = 24u;
        params.nbHeightSegs = 4u;
        params.nbCapSegs = 4u;
        params.meshID = m_EntityData[8].meshID;
        params.materialID = 0.0f;
        params.vertexOffset = runningVertexOffset;
        params.indexOffset = runningIndexOffset;
        params.worldOffsetX = slot.x;
        params.worldOffsetY = -params.height * 0.5f; // recenter: geometry spans y=[0,height]
        params.worldOffsetZ = slot.y;

        uint32_t sideColumns = params.nbSides + 1u;
        uint32_t sideVertCount = sideColumns * (params.nbHeightSegs + 1u);
        uint32_t capVertCount = 1u + params.nbCapSegs * sideColumns; // center + rings
        uint32_t totalVerts = sideVertCount + 2u * capVertCount;
        constexpr uint32_t kLocalSizeX = 64u; // geom_cylinder.comp local_size_x = 64
        uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

        DispatchGeometryCompute(m_CylinderPipeline, m_ComputePipelineLayout,
            &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

        runningVertexOffset += totalVerts;
        // Side quads (nbHeightSegs*nbSides) + 2 caps (nbCapSegs*nbSides each), 6 indices/quad.
        runningIndexOffset += 6u * params.nbSides * (params.nbHeightSegs + 2u * params.nbCapSegs);
    }

    m_TotalVertexCount = runningVertexOffset;
    m_TotalIndexCount = runningIndexOffset;
    Logger::Log(LogLevel::Info, std::format(
        "[GenerateGeometry] All 9 primitives generated: totalVertexCount={} totalIndexCount={} "
        "(buffers hold {} verts / {} indices max)",
        runningVertexOffset, runningIndexOffset,
        (512u * 1024u) / sizeof(renderer::Vertex), (128u * 1024u) / sizeof(uint32_t)));
}

void VulkanContext::UpdateEntityRotations(float timeSeconds) {
    // Distinct per-axis angular speeds (radians/sec) and a per-entity phase offset so the 9
    // primitives tumble out of sync with each other rather than spinning in lockstep.
    constexpr float kSpeedX = 0.7f;
    constexpr float kSpeedY = 1.1f;
    constexpr float kSpeedZ = 0.5f;
    constexpr float kPhaseStep = 0.6f; // radians of extra offset per meshID

    std::array<EntityTransform, kEntityCount> transforms{};

    for (uint32_t meshID = 0; meshID < kEntityCount; ++meshID) {
        float phase = static_cast<float>(meshID) * kPhaseStep;

        maths::mat4 rotation =
            maths::mat4::RotateY(timeSeconds * kSpeedY + phase) *
            maths::mat4::RotateX(timeSeconds * kSpeedX + phase) *
            maths::mat4::RotateZ(timeSeconds * kSpeedZ + phase);

        // Every primitive's baked world-space center coincides exactly with its grid slot
        // position at Y=0: each geom_*.comp shader either generates a shape already centered
        // on its own local origin (icosphere/box/sphere/torus/plane), or one that spans
        // y=[0,height] recentered via worldOffsetY=-height/2 in GenerateGeometry() (cone/tube/
        // cylinder/capsule) — both cases land the shape's true vertical midpoint at Y=0.
        maths::vec2 slot = GridSlot(static_cast<int>(meshID));

        EntityTransform& xform = transforms[meshID];
        xform.rotation = rotation;
        xform.centerX = slot.x;
        xform.centerY = 0.0f;
        xform.centerZ = slot.y;
        xform._pad0 = 0.0f;
    }

    void* mapped = nullptr;
    if (vmaMapMemory(m_Allocator, m_EntityTransformAllocation, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map Entity Transform buffer for per-frame update!");
    }
    std::memcpy(mapped, transforms.data(), sizeof(EntityTransform) * kEntityCount);
    vmaUnmapMemory(m_Allocator, m_EntityTransformAllocation);
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

    if (m_DepthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_Device, m_DepthImageView, nullptr);
        m_DepthImageView = VK_NULL_HANDLE;
    }
    if (m_DepthImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_Allocator, m_DepthImage, m_DepthAllocation);
        m_DepthImage = VK_NULL_HANDLE;
        m_DepthAllocation = VK_NULL_HANDLE;
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
    if (m_EntityTransformBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_Allocator, m_EntityTransformBuffer, m_EntityTransformAllocation);
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

    for (VkPipeline pipeline : { m_ConePipeline, m_IcospherePipeline, m_PlanePipeline,
                                  m_SpherePipeline, m_TorusPipeline, m_TubePipeline,
                                  m_CapsulePipeline, m_CylinderPipeline }) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_Device, pipeline, nullptr);
        }
    }
    for (VkPipeline facePipeline : m_BoxFacePipelines) {
        if (facePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_Device, facePipeline, nullptr);
        }
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
    if (m_BoxComputePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_Device, m_BoxComputePipelineLayout, nullptr);
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