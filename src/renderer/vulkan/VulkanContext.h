#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>
#include "core/Camera.h"
#include "core/EngineConfig.h" // config::VERTEX_SPACING default arg for GeneratePlane()
#include "core/EntityData.h"
#include "core/IDManager.h"
#include "renderer/MaterialParameterTable.h"
#include <array>
#include <string_view>
#include <vector>
#include <string>

class VulkanContext {
public:
    void Init(std::string_view appName, GLFWwindow* window);
    void Shutdown();

    VkDevice GetDevice() const { return m_Device; }
    // Needed by renderer::SurfaceCacheRayTracingPass::Init, which queries
    // VkPhysicalDeviceRayTracingPipelinePropertiesKHR (Shader Binding Table alignment) -- the one
    // consumer in this codebase that needs the physical device handle outside VulkanContext itself.
    VkPhysicalDevice GetPhysicalDevice() const { return m_PhysicalDevice; }
    VkInstance GetInstance() const { return m_Instance; }
    uint32_t GetGraphicsQueueFamilyIndex() const { return m_GraphicsQueueFamilyIndex; }
    VkFormat GetSwapchainImageFormat() const { return m_SwapchainImageFormat; }
    VkSwapchainKHR GetSwapchain() const { return m_Swapchain; }
    VkExtent2D GetSwapchainExtent() const { return m_SwapchainExtent; }
    const std::vector<VkImage>& GetSwapchainImages() const { return m_SwapchainImages; }
    const std::vector<VkImageView>& GetSwapchainImageViews() const { return m_SwapchainImageViews; }
    VkImageView GetDepthImageView() const { return m_DepthImageView; }
    VkImage GetDepthImage() const { return m_DepthImage; }
    VkFormat GetDepthFormat() const { return m_DepthFormat; }
    VkCommandBuffer GetCommandBuffer() const { return m_CommandBuffer; }

    // Visibility Buffer attachments (replaces the classic lit-color G-Buffer target): two
    // single-channel R32_UINT images, index-aligned per-pixel -- ClusterID and local TriangleID
    // written together by draw.frag form one logical 64-bit visibility ID, split across two
    // mandatory-format images instead of one VK_FORMAT_R64_UINT attachment (whose color-attachment
    // support is NOT guaranteed by the Vulkan spec, unlike R32_UINT). See CreatePipelinesAndDescriptors()
    // and draw.vert/draw.frag.
    VkImage GetVisBufferClusterIDImage() const { return m_VisBufferClusterIDImage; }
    VkImageView GetVisBufferClusterIDView() const { return m_VisBufferClusterIDImageView; }
    VkImage GetVisBufferTriangleIDImage() const { return m_VisBufferTriangleIDImage; }
    VkImageView GetVisBufferTriangleIDView() const { return m_VisBufferTriangleIDImageView; }
    static constexpr VkFormat GetVisBufferFormat() { return kVisBufferFormat; }

    // Dedicated hardware copy queue (UE 5.8 RHI parity), used by renderer::
    // GeometryStreamingCoordinator for its per-frame page uploads so they never contend for the
    // graphics queue's own command submission. Falls back to the graphics queue/family when the
    // GPU exposes no distinct transfer-only family (see CreateLogicalDevice()'s own comment) --
    // HasDedicatedTransferQueue() tells a caller whether queue-family-ownership-transfer barriers
    // are actually needed (same family == none needed).
    VkQueue GetTransferQueue() const { return m_TransferQueue; }
    uint32_t GetTransferQueueFamilyIndex() const { return m_TransferQueueFamilyIndex; }
    bool HasDedicatedTransferQueue() const { return m_HasDedicatedTransferQueue; }
    VkCommandBuffer GetTransferCommandBuffer() const { return m_TransferCommandBuffer; }
    // Signaled when GetTransferCommandBuffer()'s submission finishes -- the graphics submission
    // waits on this before consuming anything the transfer queue uploaded this frame. See
    // main.cpp's per-frame submit sequence.
    VkSemaphore GetTransferFinishedSemaphore() const { return m_TransferFinishedSemaphore; }

    VkSemaphore GetImageAvailableSemaphore() const { return m_ImageAvailableSemaphore; }
    // One render-finished semaphore per swapchain image (indexed by the acquired image index),
    // NOT a single shared one: vkQueuePresentKHR's wait on this semaphore is not guaranteed
    // retired by the time a later frame's vkQueueSubmit would re-signal it (a frame's own fence
    // only guards that frame's GPU work, not the present engine's internal semaphore consumption),
    // so a single semaphore reused across the swapchain's images racily double-signals
    // (VUID-vkQueueSubmit-pSignalSemaphores-00067). See VulkanContext.cpp's CreateSyncObjects.
    VkSemaphore GetRenderFinishedSemaphore(uint32_t imageIndex) const { return m_RenderFinishedSemaphores[imageIndex]; }
    VkQueue GetGraphicsQueue() const { return m_GraphicsQueue; }

    VkPipeline GetGraphicsPipeline() const { return m_GraphicsPipeline; }
    VkPipelineLayout GetGraphicsPipelineLayout() const { return m_GraphicsPipelineLayout; }
    VkDescriptorSet GetGeometryDescriptorSet() const { return m_GeometryDescriptorSet; }

    // Total number of indices written across all 12 procedurally-generated primitives; the
    // single scene draw call in main.cpp draws exactly this many indices in one vkCmdDraw.
    uint32_t GetTotalIndexCount() const { return m_TotalIndexCount; }

    // Total number of vertices written across all 12 procedurally-generated primitives.
    uint32_t GetTotalVertexCount() const { return m_TotalVertexCount; }

    // Recomputes every entity's self-rotation (tumbling on all 3 axes) from elapsed scene
    // time and re-uploads the whole EntityTransform array to the GPU. Must be called once per
    // frame, before recording the draw, so the vertex shader picks up this frame's rotation.
    void UpdateEntityRotations(float timeSeconds);

    // --- Accessors exposing the raw GPU handles needed by geometry::RunVirtualGeometryCacheTest
    // to read back the live procedural Vertex/Index SSBOs for the virtual geometry cache test.
    // Kept minimal (handles + counts only) so the geometry/ module never needs to include this
    // header's private implementation details.
    VmaAllocator GetAllocator() const { return m_Allocator; }
    VkCommandPool GetCommandPool() const { return m_CommandPool; }
    VkBuffer GetVertexBuffer() const { return m_VertexBuffer; }
    VkBuffer GetIndexBuffer() const { return m_IndexBuffer; }
    const core::EntityData* GetEntityData() const { return m_EntityData.data(); }
    uint32_t GetEntityCount() const { return kEntityCount; }
    VkBuffer GetEntityTransformBuffer() const { return m_EntityTransformBuffer; }
    // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): CPU-readable mirror of
    // this frame's per-entity rotation, refreshed every UpdateEntityRotations() call alongside its
    // GPU SSBO upload. Consumed by renderer:: passes that need the ACTUAL rotation matrix (not just
    // a GPU buffer handle to bind) -- currently only SurfaceCacheRayTracingPass's per-frame TLAS
    // refit and GlobalSDFPass's object-space compositing (see core::EntityTransformCPU's own
    // comment for why this lives in EntityData.h, not here).
    const core::EntityTransformCPU* GetEntityTransformsCPU() const { return m_EntityTransformsCPU.data(); }
    VkBuffer GetEntityBuffer() const { return m_EntityBuffer; }
    const renderer::MaterialTable& GetMaterialTable() const { return m_MaterialTable; }

private:
    VkInstance m_Instance = VK_NULL_HANDLE;
#ifndef NDEBUG
    VkDebugUtilsMessengerEXT m_DebugMessenger = VK_NULL_HANDLE;
#endif
    VkSurfaceKHR m_Surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_PhysicalDevice = VK_NULL_HANDLE;
    VkDevice m_Device = VK_NULL_HANDLE;

    VkQueue m_GraphicsQueue = VK_NULL_HANDLE;
    uint32_t m_GraphicsQueueFamilyIndex = 0;
    VkCommandPool m_CommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_CommandBuffer = VK_NULL_HANDLE;

    // Dedicated transfer queue (or a fallback alias of m_GraphicsQueue/m_GraphicsQueueFamilyIndex)
    // -- see GetTransferQueue()'s own comment.
    VkQueue m_TransferQueue = VK_NULL_HANDLE;
    uint32_t m_TransferQueueFamilyIndex = 0;
    bool m_HasDedicatedTransferQueue = false;
    VkCommandPool m_TransferCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_TransferCommandBuffer = VK_NULL_HANDLE;
    VkSemaphore m_TransferFinishedSemaphore = VK_NULL_HANDLE;

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
    std::vector<VkSemaphore> m_RenderFinishedSemaphores; // One per swapchain image -- see GetRenderFinishedSemaphore's comment.

    VkBuffer m_EntityBuffer = VK_NULL_HANDLE;
    VmaAllocation m_EntityAllocation = VK_NULL_HANDLE;

    VkDeviceSize m_VertexBufferBytes = 0;
    VkDeviceSize m_IndexBufferBytes = 0;
    VkBuffer m_VertexBuffer = VK_NULL_HANDLE;
    VmaAllocation m_VertexAllocation = VK_NULL_HANDLE;
    VkBuffer m_IndexBuffer = VK_NULL_HANDLE;
    VmaAllocation m_IndexAllocation = VK_NULL_HANDLE;
    VkBuffer m_ParamsBuffer = VK_NULL_HANDLE;
    VmaAllocation m_ParamsAllocation = VK_NULL_HANDLE;

    // Per-entity self-rotation array (one EntityTransform per meshID, see struct_custo.glsl),
    // host-visible so UpdateEntityRotations() can re-upload it every frame with a plain memcpy.
    VkBuffer m_EntityTransformBuffer = VK_NULL_HANDLE;
    VmaAllocation m_EntityTransformAllocation = VK_NULL_HANDLE;

    VkImage m_DepthImage = VK_NULL_HANDLE;
    VmaAllocation m_DepthAllocation = VK_NULL_HANDLE;
    VkImageView m_DepthImageView = VK_NULL_HANDLE;
    VkFormat m_DepthFormat = VK_FORMAT_D32_SFLOAT;

    // Visibility Buffer attachments -- VK_FORMAT_R32_UINT is mandated by the Vulkan 1.3 core spec
    // to support VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT (unlike R64_UINT, whose color-attachment
    // support is optional and would require a runtime VkFormatProperties query), so no
    // format-support query is needed before using it as a render target here -- the same reasoning
    // HZBPass documents for its own R32G32_SFLOAT pyramid format.
    static constexpr VkFormat kVisBufferFormat = VK_FORMAT_R32_UINT;
    VkImage m_VisBufferClusterIDImage = VK_NULL_HANDLE;
    VmaAllocation m_VisBufferClusterIDAllocation = VK_NULL_HANDLE;
    VkImageView m_VisBufferClusterIDImageView = VK_NULL_HANDLE;
    VkImage m_VisBufferTriangleIDImage = VK_NULL_HANDLE;
    VmaAllocation m_VisBufferTriangleIDAllocation = VK_NULL_HANDLE;
    VkImageView m_VisBufferTriangleIDImageView = VK_NULL_HANDLE;

    // One compute pipeline per non-box primitive, all sharing m_ComputePipelineLayout and
    // reading their per-dispatch parameters from the shared Params UBO (m_ParamsBuffer).
    VkPipeline m_ConePipeline = VK_NULL_HANDLE;
    VkPipeline m_IcospherePipeline = VK_NULL_HANDLE;
    VkPipeline m_PlanePipeline = VK_NULL_HANDLE;
    VkPipeline m_SpherePipeline = VK_NULL_HANDLE;
    VkPipeline m_TorusPipeline = VK_NULL_HANDLE;
    VkPipeline m_TubePipeline = VK_NULL_HANDLE;
    VkPipeline m_CapsulePipeline = VK_NULL_HANDLE;
    VkPipeline m_CylinderPipeline = VK_NULL_HANDLE;
    VkPipeline m_PyramidPipeline = VK_NULL_HANDLE;
    VkPipeline m_TorusKnotPipeline = VK_NULL_HANDLE;
    VkPipeline m_ChamferBoxPipeline = VK_NULL_HANDLE;
    VkPipeline m_AutosmoothPipeline = VK_NULL_HANDLE;

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

    // Running total of vertices/indices written by GenerateGeometry() across all 12 primitives,
    // the 2 Lumen-corner walls, and the floor.
    uint32_t m_TotalVertexCount = 0;
    uint32_t m_TotalIndexCount = 0;

    // One EntityTransform slot per entity (box=0 .. chamferBox=11, wallA=12, wallB=13, floor=14);
    // see struct_custo.glsl. The base scene is a feature "gallery": 12 primitives laid out as 9
    // widely-separated zones (see GridSlot()) each demonstrating one engine feature explicitly
    // (Nanite density, WPO displacement, metal, dielectric, glass, translucent, emissive,
    // MegaLights, Lumen GI), plus 2 static colored walls forming the Lumen GI corner and the floor.
    static constexpr uint32_t kEntityCount = 15;
    // The 2 static walls that form the Lumen/GI showcase corner (see GenerateGeometry()'s wall
    // blocks and UpdateEntityRotations()'s fixed-rotation branch for them) -- generated right
    // before the floor, so kFloorEntityIndex below stays "last".
    static constexpr uint32_t kWallEntityIndexA = kEntityCount - 3;
    static constexpr uint32_t kWallEntityIndexB = kEntityCount - 2;
    // The floor plane is always generated last (see GenerateGeometry()'s own GeneratePlane() call).
    static constexpr uint32_t kFloorEntityIndex = kEntityCount - 1;

    // CPU-authoritative entity records: built once by BuildEntityData() (meshID assigned via
    // core::IDManager) before GenerateGeometry() runs, then copied to m_EntityBuffer by
    // UploadEntityData(). One entry per entity (box=0 .. floor=14), see struct_custo.glsl.
    std::array<core::EntityData, kEntityCount> m_EntityData{};

    // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): CPU-readable mirror of
    // m_EntityTransformBuffer's own per-frame contents -- see GetEntityTransformsCPU()'s own
    // comment.
    std::array<core::EntityTransformCPU, kEntityCount> m_EntityTransformsCPU{};

    // Hand-authored PBR showcase materials (renderer::GenerateShowcaseMaterialTable), one slot per
    // entity -- built once by BuildEntityData(), uploaded to the GPU by ClusterResolvePass::Init()
    // via ClusterRenderPipelineCreateInfo::materialTable.
    renderer::MaterialTable m_MaterialTable{};

#ifndef NDEBUG
    const bool m_EnableValidationLayers = true;
#endif

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

    void GenerateBox(
        float Width, float Length, float Height,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GenerateCone(
        float Radius1, float Radius2, float Height,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GenerateSphere(
        float Radius,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GenerateIcosphere(
        float Radius, bool Tetra, bool Octa, bool Icosa,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset,
        uint32_t& outBaseFaceCount, uint32_t& outVertsPerFace);

    void GenerateCylinder(
        float Radius, float Height,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GenerateTube(
        float Radius1, float Radius2, float Height,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GenerateTorus(
        float Radius1, float Radius2, float Rotation, float Twist,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GeneratePyramid(
        float Width, float Depth, float Height,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GeneratePlane(
        float Length, float Width,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset,
        float worldOffsetY = 0.0f, float spacing = config::VERTEX_SPACING);

    void GenerateCapsule(
        float Radius, float Height,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    // Authors m_EntityData on the CPU: assigns each of the kEntityCount entities a meshID via
    // core::IDManager::GetNextID() (instead of a hardcoded literal). Must run before
    // GenerateGeometry(), which reads m_EntityData[i].meshID into each primitive's push
    // constants / Params UBO.
    void BuildEntityData();

    // One-shot upload of m_EntityData to the GPU-only m_EntityBuffer via a temporary staging
    // buffer + vkCmdCopyBuffer, matching DispatchGeometryCompute's blocking one-time-submit
    // pattern. Must run after m_EntityBuffer is allocated and before it is bound for reading.
    void UploadEntityData();

    // Single source of truth for the feature-gallery layout (also used by
    // UpdateEntityRotations() to recover each entity's rotation pivot, and duplicated by
    // renderer::MegaLightsTypes.cpp's EntityGridPosition() -- see that function's own comment):
    // primitive slot index in [0, 11], returns the (X, Z) world position of that primitive's own
    // showcase zone (Y = 0). Zones are spaced 4 units apart so each feature reads as its own
    // distinct area rather than one continuous grid -- see the .cpp definition for the full
    // zone -> feature mapping.
    maths::vec2 GridSlot(int slotIndex) const;

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
};
