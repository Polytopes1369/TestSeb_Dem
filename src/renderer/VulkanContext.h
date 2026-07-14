#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>
#include "core/Camera.h"
#include <array>
#include <string_view>
#include <vector>
#include <string>

class VulkanContext {
public:
    void Init(std::string_view appName, GLFWwindow* window);
    void Shutdown();

    VkDevice GetDevice() const { return m_Device; }
    VkSwapchainKHR GetSwapchain() const { return m_Swapchain; }
    VkExtent2D GetSwapchainExtent() const { return m_SwapchainExtent; }
    const std::vector<VkImage>& GetSwapchainImages() const { return m_SwapchainImages; }
    const std::vector<VkImageView>& GetSwapchainImageViews() const { return m_SwapchainImageViews; }
    VkImageView GetDepthImageView() const { return m_DepthImageView; }
    VkImage GetDepthImage() const { return m_DepthImage; }
    VkCommandBuffer GetCommandBuffer() const { return m_CommandBuffer; }

    VkSemaphore GetImageAvailableSemaphore() const { return m_ImageAvailableSemaphore; }
    VkSemaphore GetRenderFinishedSemaphore() const { return m_RenderFinishedSemaphore; }
    VkQueue GetGraphicsQueue() const { return m_GraphicsQueue; }

    VkPipeline GetGraphicsPipeline() const { return m_GraphicsPipeline; }
    VkPipelineLayout GetGraphicsPipelineLayout() const { return m_GraphicsPipelineLayout; }
    VkDescriptorSet GetGeometryDescriptorSet() const { return m_GeometryDescriptorSet; }

    // Total number of indices written across all 7 procedurally-generated primitives; the
    // single scene draw call in main.cpp draws exactly this many indices in one vkCmdDraw.
    uint32_t GetTotalIndexCount() const { return m_TotalIndexCount; }

private:
    VkInstance m_Instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_DebugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_Surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;
    VkDevice m_Device = VK_NULL_HANDLE;

    VkQueue m_GraphicsQueue = VK_NULL_HANDLE;
    uint32_t m_GraphicsQueueFamilyIndex = 0;
    VkCommandPool m_CommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_CommandBuffer = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_BindlessLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_BindlessSet = VK_NULL_HANDLE;
    VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;

    VmaAllocator m_Allocator = VK_NULL_HANDLE;

    VkSwapchainKHR m_Swapchain = VK_NULL_HANDLE;
    VkFormat m_SwapchainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_SwapchainExtent = { 0, 0 };
    std::vector<VkImage> m_SwapchainImages;
    std::vector<VkImageView> m_SwapchainImageViews;

    VkSemaphore m_ImageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore m_RenderFinishedSemaphore = VK_NULL_HANDLE;

    VkBuffer m_EntityBuffer = VK_NULL_HANDLE;
    VmaAllocation m_EntityAllocation = VK_NULL_HANDLE;

    VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
    VmaAllocation m_VertexAllocation = VK_NULL_HANDLE;
    VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
    VmaAllocation m_IndexAllocation = VK_NULL_HANDLE;
    VkBuffer m_ParamsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_ParamsAllocation = VK_NULL_HANDLE;

    VkImage m_DepthImage = VK_NULL_HANDLE;
    VmaAllocation m_DepthAllocation = VK_NULL_HANDLE;
    VkImageView m_DepthImageView = VK_NULL_HANDLE;
    VkFormat m_DepthFormat = VK_FORMAT_D32_SFLOAT;

    // One compute pipeline per non-box primitive, all sharing m_ComputePipelineLayout and
    // reading their per-dispatch parameters from the shared Params UBO (m_ParamsBuffer).
    VkPipeline m_ConePipeline = VK_NULL_HANDLE;
    VkPipeline m_IcospherePipeline = VK_NULL_HANDLE;
    VkPipeline m_PlanePipeline = VK_NULL_HANDLE;
    VkPipeline m_SpherePipeline = VK_NULL_HANDLE;
    VkPipeline m_TorusPipeline = VK_NULL_HANDLE;
    VkPipeline m_TubePipeline = VK_NULL_HANDLE;

    // The box is generated via 6 dispatches (one per cube face) of the same geom_box.comp
    // module, each specialized with a different VkSpecializationInfo (axis mapping / winding)
    // and driven by push constants instead of the shared Params UBO.
    std::array<VkPipeline, 6> m_BoxFacePipelines{};
    VkPipelineLayout m_BoxComputePipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_GraphicsPipeline = VK_NULL_HANDLE;

    VkDescriptorPool m_GeometryDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_GeometryLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_GeometryDescriptorSet = VK_NULL_HANDLE;

    VkPipelineLayout m_ComputePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_GraphicsPipelineLayout = VK_NULL_HANDLE;

    // Running total of indices written by GenerateGeometry() across all 7 primitives.
    uint32_t m_TotalIndexCount = 0;

    const bool m_EnableValidationLayers = true;

    void CreateInstance(std::string_view appName);
    void SetupDebugMessenger();
    void CreateSurface(GLFWwindow* window);
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateCommandPool();
    void AllocateCommandBuffer();
    void CreateBindlessDescriptorSetLayout();
    void CreateDescriptorPool();
    void AllocateBindlessDescriptorSet();
    void CreateSwapchain(GLFWwindow* window);
    void CreateImageViews();
    void CreateSyncObjects();

    void CreatePipelinesAndDescriptors();
    void GenerateGeometry();

    // Records, submits, and blocks on a single one-shot compute dispatch that generates one
    // primitive (or one box face) into the shared Vertex/Index SSBOs. Exactly one of
    // uboParamsData / pushConstantData must be non-null, matching how the target shader
    // expects its Params block: every primitive except the box reads a UBO (binding = 2,
    // copied here into m_ParamsBuffer); the box reads push constants instead (see
    // geom_box.comp). Blocking (vkQueueWaitIdle) between dispatches keeps this simple and
    // correct for a one-time startup pass, matching the pre-existing single-primitive path.
    void DispatchGeometryCompute(
        VkPipeline pipeline,
        VkPipelineLayout layout,
        const void* uboParamsData,
        size_t uboParamsSize,
        const void* pushConstantData,
        size_t pushConstantSize,
        uint32_t groupCountX,
        uint32_t groupCountY,
        uint32_t groupCountZ);

    // DEBUG: copies back a small sample of the generated vertex/index SSBOs to host memory
    // and logs it via Logger, to verify the compute dispatch actually produced valid geometry.
    void DebugReadbackGeometrySample(uint32_t vertsPerFace, uint32_t expectedIndexCount);

    std::vector<char> ReadShaderFile(const std::string& filename);
};