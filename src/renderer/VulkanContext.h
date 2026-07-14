#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string_view>
#include <string>
#include <optional>
#include <vk_mem_alloc.h>
#include "core/EntityData.h"

// Forward declaration
struct GLFWwindow;

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsComputePresentFamily;

    bool IsComplete() const {
        return graphicsComputePresentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanContext {
public:
    void Init(std::string_view appName, GLFWwindow* window);
    void Shutdown();

    void UploadEntityData(const std::vector<core::EntityData>& entities);

    // --- Getters Publics ---
    VkInstance GetInstance() const { return m_Instance; }
    VkPhysicalDevice GetPhysicalDevice() const { return m_PhysicalDevice; }
    VkDevice GetDevice() const { return m_Device; }
    VkQueue GetGraphicsQueue() const { return m_GraphicsQueue; }
    VkSwapchainKHR GetSwapchain() const { return m_Swapchain; }
    VkCommandBuffer GetCommandBuffer() const { return m_CommandBuffer; }
    VkExtent2D GetSwapchainExtent() const { return m_SwapchainExtent; }
    const std::vector<VkImageView>& GetSwapchainImageViews() const { return m_SwapchainImageViews; }
    const std::vector<VkImage>& GetSwapchainImages() const { return m_SwapchainImages; }

    // Synchronisation & Pipelines Compute
    VkSemaphore GetImageAvailableSemaphore() const { return m_ImageAvailableSemaphore; }
    VkSemaphore GetRenderFinishedSemaphore() const { return m_RenderFinishedSemaphore; }
    VkPipeline GetComputePipeline() const { return m_ComputePipeline; }
    VkPipelineLayout GetPipelineLayout() const { return m_PipelineLayout; }
    VkPipelineLayout GetComputePipelineLayout() const { return m_ComputePipelineLayout; }
    const std::vector<VkDescriptorSet>& GetComputeDescriptorSets() const { return m_ComputeDescriptorSets; }

private:
    // Initialisation Core
    void CreateInstance(std::string_view appName);
    void SetupDebugMessenger();
    void CreateSurface(GLFWwindow* window);
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateSwapchain(GLFWwindow* window);
    void CreateImageViews();

    // Commandes & Descripteurs Globaux (Bindless)
    void CreateCommandPool();
    void AllocateCommandBuffer();
    void CreateBindlessDescriptorSetLayout();
    void CreateDescriptorPool();
    void AllocateBindlessDescriptorSet();

    // Pipeline Compute dédié (Phase 4)
    void CreateComputePipelineAndDescriptors();
    void CreateSyncObjects();
    std::vector<char> ReadShaderFile(const std::string& filename);

    // Helpers de sélection hardware
    bool IsDeviceSuitable(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);
    bool CheckValidationLayerSupport();
    std::vector<const char*> GetRequiredExtensions();

    // Helpers WSI / Swapchain
    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window);

    // --- Variables Membres ---
    VkInstance m_Instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_DebugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_Surface = VK_NULL_HANDLE;

    VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;
    VkDevice m_Device = VK_NULL_HANDLE;
    VkQueue m_GraphicsQueue = VK_NULL_HANDLE;

    // Infrastructure d'affichage
    VkSwapchainKHR m_Swapchain = VK_NULL_HANDLE;
    VkFormat m_SwapchainImageFormat;
    VkExtent2D m_SwapchainExtent;
    std::vector<VkImage> m_SwapchainImages;
    std::vector<VkImageView> m_SwapchainImageViews;

    // Commandes GPU
    VkCommandPool m_CommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_CommandBuffer = VK_NULL_HANDLE;

    // Synchronisation Frame-by-Frame
    VkSemaphore m_ImageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore m_RenderFinishedSemaphore = VK_NULL_HANDLE;

    // Infrastructure Bindless Globale
    VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_BindlessLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_BindlessDescriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;

    // Allocateur VMA
    VmaAllocator m_Allocator = VK_NULL_HANDLE;

    // Pipeline de Calcul Local (Compute Target)
    VkPipeline m_ComputePipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_ComputeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_ComputeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_ComputePipelineLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_ComputeDescriptorSets;

    VkBuffer m_EntityBuffer = VK_NULL_HANDLE;
    VmaAllocation m_EntityAllocation = VK_NULL_HANDLE;

    // Extensions & Couches de Validation
    const std::vector<const char*> m_ValidationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> m_DeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME
    };

#ifdef NDEBUG
    const bool m_EnableValidationLayers = false;
#else
    const bool m_EnableValidationLayers = true;
#endif
};