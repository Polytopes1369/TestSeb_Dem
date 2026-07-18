#include "renderer/vulkan/VulkanContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "core/EngineConfig.h" // Centralized engine configurations (config::WINDOW_WIDTH, etc.)
#include "core/EntityData.h"
#include "core/Logger.h"
#include "renderer/MaterialParameterTable.h"
#include "renderer/RenderTypes.h"
#include "renderer/vulkan/RayTracingFunctions.h"
#include "renderer/vulkan/VulkanUtils.h"
#include "core/debug/ValidationMessageSink.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>


// Validation layers, the debug messenger and the code that wires them up are
// debug-only tooling (project rule: nothing debug-related is compiled into a
// Release binary). Everything gated by NDEBUG below -- the array,
// DebugCallback, and the extension/messenger setup in CreateInstance /
// SetupDebugMessenger / the shutdown path -- compiles to nothing in Release.
#ifndef NDEBUG

// For IsDebuggerPresent() in DebugCallback -- lean include, and NOMINMAX so
// windows.h's min/max macros never shadow the std:: ones used throughout this
// file.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

// Validation Layers array definition
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#endif // NDEBUG

namespace {

// --- Per-shader Params blocks, mirrored 1:1 from their geom_*.comp
// UBO/push-constant declarations (all fields are 4-byte scalars, so std140
// layout == plain C++ struct layout here: no vec3 members means no extra
// padding to account for). worldOffsetX/Y/Z are appended at the end of every
// block and translate the generated geometry to its grid slot directly on the
// GPU, avoiding a separate per-instance transform stage.

struct ConeParams {
  float radius1;
  float radius2;
  float height;
  uint32_t heightSegments;
  uint32_t capSegments;
  uint32_t sides;
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
  uint32_t segments;
  uint32_t tetra;
  uint32_t octa;
  uint32_t icosa;
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
  uint32_t widthSegments;
  uint32_t lengthSegments;
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
  uint32_t segments;
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
  float rotation;
  float twist;
  uint32_t segments;
  uint32_t sides;
  uint32_t meshID;
  float materialID;
  uint32_t vertexOffset;
  uint32_t indexOffset;
  float worldOffsetX;
  float worldOffsetY;
  float worldOffsetZ;
};

struct TubeParams {
  float radius1;
  float radius2;
  float height;
  uint32_t heightSegments;
  uint32_t capSegments;
  uint32_t sides;
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
  float height; // cylindrical body length (= total height - 2*radius)
  uint32_t sides;
  uint32_t heightSegs;
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
  uint32_t heightSegments;
  uint32_t capSegments;
  uint32_t sides;
  uint32_t meshID;
  float materialID;
  uint32_t vertexOffset;
  uint32_t indexOffset;
  float worldOffsetX;
  float worldOffsetY;
  float worldOffsetZ;
};

struct PyramidParams {
  float width;
  float depth;
  float height;
  uint32_t widthSegments;
  uint32_t depthSegments;
  uint32_t heightSegments;
  uint32_t meshID;
  float materialID;
  uint32_t vertexOffset;
  uint32_t indexOffset;
  float worldOffsetX;
  float worldOffsetY;
  float worldOffsetZ;
};

struct TorusKnotParams {
  float radius;
  float tube;
  uint32_t p;
  uint32_t q;
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

struct ChamferBoxParams {
  float width;
  float height;
  float depth;
  uint32_t sideSegs;
  uint32_t heightSegs;
  float chamferPower;
  uint32_t meshID;
  float materialID;
  uint32_t vertexOffset;
  uint32_t indexOffset;
  float worldOffsetX;
  float worldOffsetY;
  float worldOffsetZ;
};

// CPU mirror of the GLSL EntityTransform struct (struct_custo.glsl): must match
// its std430 layout exactly (mat4 = 64 bytes, vec3 + pad = 16 bytes, vec3 + pad = 16 bytes) since
// it is memcpy'd wholesale into m_EntityTransformBuffer every frame.
struct EntityTransform {
  maths::mat4 rotation;
  float centerX;
  float centerY;
  float centerZ;
  float _pad0;
  float translationX;
  float translationY;
  float translationZ;
  float _pad1;
};

// geom_box.comp reads its Params via push constants (not the shared UBO),
// because each of its 6 face dispatches also needs distinct specialization
// constants baked per-pipeline.
struct BoxPushConstants {
  float width;
  float length;
  float height;
  uint32_t widthSegments;
  uint32_t lengthSegments;
  uint32_t heightSegments;
  uint32_t meshID;
  float materialID;
  uint32_t vertexOffset;
  uint32_t indexOffset;
  float worldOffsetX;
  float worldOffsetY;
  float worldOffsetZ;
};

// Specialization constants for geom_box.comp's 6 face dispatches (constant_id
// 0..6: uAxis, vAxis, wAxis, faceMode, udir, vdir, wSign — axis indices 0=x,1=y,2=z,
// matching the shader's setComp() convention). Each face's
// (uAxis,vAxis,wAxis,udir,vdir) is chosen so that the shader's (a,b,d) triangle
// — built from vectors vAxis*(segH*vdir) and uAxis*(segW*udir) — satisfies
// cross(b-a, d-a) == wSign * wAxisVector, which is the condition for
// consistently outward normals with the winding VulkanPipeline expects (front
// faces wind CCW in object space, becoming CLOCKWISE on screen after the
// projection's Y-flip). Derivation: cross(b-a,d-a) =
// -segH*segW*udir*vdir*P*wAxisVector, where P = +1 if (uAxis,vAxis,wAxis) is an
// even permutation of (x,y,z) and -1 if odd — so the required condition reduces
// to udir*vdir*P == -wSign (solved here with vdir=1 fixed, udir = -wSign*P).
struct BoxFaceSpecConstants {
  int32_t uAxis;
  int32_t vAxis;
  int32_t wAxis;
  int32_t faceMode; // 0: w==z, 1: w==y, 2: w==x (selects the shader's UV
                    // formula branch)
  float udir;
  float vdir;
  float wSign;
};

constexpr BoxFaceSpecConstants kBoxFaceSpecs[6] = {
    {0, 1, 2, 0, -1.0f, 1.0f, 1.0f},  // +Z  (u=x,v=y,w=z, P=+1, wSign=+1)
    {0, 1, 2, 0, 1.0f,  1.0f, -1.0f}, // -Z  (wSign=-1)
    {0, 2, 1, 1, 1.0f,  1.0f, 1.0f},  // +Y  (u=x,v=z,w=y, P=-1, wSign=+1)
    {0, 2, 1, 1, -1.0f, 1.0f, -1.0f}, // -Y  (wSign=-1)
    {1, 2, 0, 2, -1.0f, 1.0f, 1.0f},  // +X  (u=y,v=z,w=x, P=+1, wSign=+1)
    {1, 2, 0, 2, 1.0f,  1.0f, -1.0f}, // -X  (wSign=-1)
};
} // namespace

#ifndef NDEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL
DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  LogLevel level = LogLevel::Info;
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    level = LogLevel::Error;
  } else if (messageSeverity >=
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    level = LogLevel::Warning;
  }

  // ALWAYS log before any break: __debugbreak() without an attached debugger
  // terminates the process on the spot, and breaking before logging used to
  // kill the app with the validation message still unwritten -- a validation
  // error then looked like a silent crash right after whatever the last
  // successful log line happened to be. Breaking is also gated on an actual
  // debugger being attached, so a plain console launch now logs the error and
  // keeps going instead of dying on an int3 the user can never see.
  LOG(level, pCallbackData->pMessage);

  // Also feed DebugTestPipeline's capture buffer (see ValidationMessageSink's own class comment):
  // a plain observer, changes nothing about the logging/break behavior above.
  debugpipeline::ValidationMessageSink::Push(static_cast<uint32_t>(messageSeverity),
                                              pCallbackData->pMessage);

  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT &&
      IsDebuggerPresent()) {
    __debugbreak();
  }
  return VK_FALSE;
}
#endif // NDEBUG
void VulkanContext::Init(std::string_view appName, GLFWwindow *window) {
  CreateInstance(appName);
  SetupDebugMessenger();
  CreateSurface(window);
  PickPhysicalDevice();

  // Resize GLFW window to match the loaded/detected profile dimensions
  glfwSetWindowSize(window, config::WINDOW_WIDTH, config::WINDOW_HEIGHT);

  m_VertexBufferBytes = config::nanite::VERTEX_BUFFER_BYTES;
  m_IndexBufferBytes = config::nanite::INDEX_BUFFER_BYTES;

  CreateLogicalDevice();

  CreateCommandPool();
  AllocateCommandBuffer();

  CreateBindlessDescriptorSetLayout();
  CreateDescriptorPool();
  AllocateBindlessDescriptorSet();

  m_PipelineLayout =
      VulkanPipeline::CreatePipelineLayout(m_Device, m_BindlessLayout);

  // Initialize VMA Allocator
  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
  allocatorInfo.instance = m_Instance;
  allocatorInfo.physicalDevice = m_PhysicalDevice;
  allocatorInfo.device = m_Device;
  // Required by VMA itself (a hard assert, not just a validation warning) for any
  // vmaCreateBuffer() call using VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT -- SurfaceCachePass's
  // vertex/index buffers and every renderer::AccelerationStructure/SurfaceCacheTraceContext
  // buffer use that flag (see SurfaceCachePass::GetVertexBuffer()'s own comment), on top of the
  // bufferDeviceAddress DEVICE feature already enabled in CreateLogicalDevice above -- VMA needs
  // to be told separately, at allocator-creation time, that it is allowed to service such
  // requests.
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  if (vmaCreateAllocator(&allocatorInfo, &m_Allocator) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create VMA allocator!");
  }
  LOG_INFO("[VulkanContext] VMA Allocator initialized successfully.");

  // Author entity data on the CPU (meshID assigned via core::IDManager) before
  // any GPU buffer exists for it, per the engine's CPU-authored / GPU-generated
  // architecture.
  BuildEntityData();

  // Allocate Entity Tracking Buffer
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = sizeof(core::EntityData) * 10000;
  bufferInfo.usage =
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  if (vmaCreateBuffer(m_Allocator, &bufferInfo, &allocInfo, &m_EntityBuffer,
                      &m_EntityAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate entity buffer!");
  }
  LOG_INFO("[VulkanContext] Allocated entity buffer (10000 elements).");

  // Upload the CPU-authored entity records now that both m_EntityBuffer and the
  // command pool (created above, CreateCommandPool()/AllocateCommandBuffer())
  // exist.
  UploadEntityData();

  // Allocate Procedural Geometry Buffers (see
  // m_VertexBufferBytes/m_IndexBufferBytes; 256B for internal parameters)
  VkBufferCreateInfo vInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  vInfo.size = m_VertexBufferBytes;
  // TRANSFER_SRC_BIT is required for the DEBUG readback (vkCmdCopyBuffer) in
  // DebugReadbackGeometrySample()
  vInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VmaAllocationCreateInfo vAlloc{.usage = VMA_MEMORY_USAGE_GPU_ONLY};
  if (vmaCreateBuffer(m_Allocator, &vInfo, &vAlloc, &m_VertexBuffer,
                      &m_VertexAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate vertex SSBO!");
  }
  LOG_INFO(std::format("[VulkanContext] Allocated vertex SSBO: size={} MB.", m_VertexBufferBytes / (1024 * 1024)));

  VkBufferCreateInfo iInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  iInfo.size = m_IndexBufferBytes;
  // TRANSFER_SRC_BIT is required for the DEBUG readback (vkCmdCopyBuffer) in
  // DebugReadbackGeometrySample()
  iInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (vmaCreateBuffer(m_Allocator, &iInfo, &vAlloc, &m_IndexBuffer,
                      &m_IndexAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate index SSBO!");
  }
  LOG_INFO(std::format("[VulkanContext] Allocated index SSBO: size={} MB.", m_IndexBufferBytes / (1024 * 1024)));

  VkBufferCreateInfo pInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  pInfo.size = 256;
  pInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  VmaAllocationCreateInfo pAlloc{.usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
  if (vmaCreateBuffer(m_Allocator, &pInfo, &pAlloc, &m_ParamsBuffer,
                      &m_ParamsAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate compute uniform block!");
  }
  LOG_INFO("[VulkanContext] Allocated params uniform buffer (256 bytes).");

  // Per-entity rotation buffer: host-visible so UpdateEntityRotations() can
  // re-upload the whole array with a plain memcpy every frame (mirrors the
  // m_ParamsBuffer update pattern).
  VkBufferCreateInfo etInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  etInfo.size = sizeof(EntityTransform) * kTotalEntityCount;
  etInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VmaAllocationCreateInfo etAlloc{.usage = VMA_MEMORY_USAGE_CPU_TO_GPU};
  if (vmaCreateBuffer(m_Allocator, &etInfo, &etAlloc, &m_EntityTransformBuffer,
                      &m_EntityTransformAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate entity transform buffer!");
  }

  CreateSwapchain(window);
  CreateImageViews();

  // Compute scaled rendering resolution for TAA / TSR
  VkExtent2D renderExtent = m_SwapchainExtent;
  renderExtent.width = static_cast<uint32_t>(static_cast<float>(m_SwapchainExtent.width) * config::temporal::RENDER_SCALE);
  renderExtent.height = static_cast<uint32_t>(static_cast<float>(m_SwapchainExtent.height) * config::temporal::RENDER_SCALE);
  renderExtent.width = (renderExtent.width + 7) & ~7;
  renderExtent.height = (renderExtent.height + 7) & ~7;

  // Create Depth Image
  VkImageCreateInfo depthImageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
  depthImageInfo.format = m_DepthFormat;
  depthImageInfo.extent = {renderExtent.width, renderExtent.height, 1};
  depthImageInfo.mipLevels = 1;
  depthImageInfo.arrayLayers = 1;
  depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  // SAMPLED_BIT (in addition to the attachment usage) lets renderer::HZBPass
  // read this image's resolved depth through a combined image sampler when
  // building the Hierarchical Z-Buffer pyramid -- D32_SFLOAT is mandated to
  // support both DEPTH_STENCIL_ATTACHMENT and SAMPLED_IMAGE optimal-tiling
  // usage, so this adds no extra format-support risk.
  depthImageInfo.usage =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo depthAllocInfo{};
  depthAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  if (vmaCreateImage(m_Allocator, &depthImageInfo, &depthAllocInfo,
                      &m_DepthImage, &m_DepthAllocation,
                      nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create depth image!");
  }

  // Create Depth Image View
  VkImageViewCreateInfo depthViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  depthViewInfo.image = m_DepthImage;
  depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  depthViewInfo.format = m_DepthFormat;
  depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  depthViewInfo.subresourceRange.baseMipLevel = 0;
  depthViewInfo.subresourceRange.levelCount = 1;
  depthViewInfo.subresourceRange.baseArrayLayer = 0;
  depthViewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(m_Device, &depthViewInfo, nullptr, &m_DepthImageView) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create depth image view!");
  }

  // Create Visibility Buffer attachments (ClusterID + local TriangleID, see
  // VulkanContext.h's
  // GetVisBufferClusterIDImage()/GetVisBufferTriangleIDImage() comment). Two
  // separate single-channel R32_UINT images rather than one 2-channel image,
  // sized to the rendering extent.
  auto createVisBufferImage = [&](VkImage &outImage,
                                  VmaAllocation &outAllocation,
                                  VkImageView &outView, const char *debugName) {
    VkImageCreateInfo visBufferImageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    visBufferImageInfo.imageType = VK_IMAGE_TYPE_2D;
    visBufferImageInfo.format = kVisBufferFormat;
    visBufferImageInfo.extent = {renderExtent.width,
                                 renderExtent.height, 1};
    visBufferImageInfo.mipLevels = 1;
    visBufferImageInfo.arrayLayers = 1;
    visBufferImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    visBufferImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    visBufferImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_STORAGE_BIT;
    visBufferImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo visBufferAllocInfo{};
    visBufferAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;


    if (vmaCreateImage(m_Allocator, &visBufferImageInfo, &visBufferAllocInfo,
                       &outImage, &outAllocation, nullptr) != VK_SUCCESS) {
      throw std::runtime_error(
          std::string("Failed to create Visibility Buffer image: ") +
          debugName);
    }

    VkImageViewCreateInfo visBufferViewInfo{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    visBufferViewInfo.image = outImage;
    visBufferViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    visBufferViewInfo.format = kVisBufferFormat;
    visBufferViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    visBufferViewInfo.subresourceRange.baseMipLevel = 0;
    visBufferViewInfo.subresourceRange.levelCount = 1;
    visBufferViewInfo.subresourceRange.baseArrayLayer = 0;
    visBufferViewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_Device, &visBufferViewInfo, nullptr, &outView) !=
        VK_SUCCESS) {
      throw std::runtime_error(
          std::string("Failed to create Visibility Buffer image view: ") +
          debugName);
    }
  };

  createVisBufferImage(m_VisBufferClusterIDImage,
                       m_VisBufferClusterIDAllocation,
                       m_VisBufferClusterIDImageView, "ClusterID");
  createVisBufferImage(m_VisBufferTriangleIDImage,
                       m_VisBufferTriangleIDAllocation,
                       m_VisBufferTriangleIDImageView, "TriangleID");

  CreateSyncObjects();

  CreatePipelinesAndDescriptors();
  GenerateGeometry(); // Dispatches all 7 procedural primitive generators once
                      // at startup
}

void VulkanContext::CreateInstance(std::string_view appName) {
  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pApplicationName = appName.data();
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "DemosceneEngine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  createInfo.pApplicationInfo = &appInfo;

  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions =
      glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);

#ifndef NDEBUG
  if (m_EnableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  }
#endif

  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  if (vkCreateInstance(&createInfo, nullptr, &m_Instance) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Vulkan Instance!");
  }
}

void VulkanContext::SetupDebugMessenger() {
#ifndef NDEBUG
  if (!m_EnableValidationLayers)
    return;

  VkDebugUtilsMessengerCreateInfoEXT createInfo{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  // Add the fatal message flag to break execution immediately
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = DebugCallback;

  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      m_Instance, "vkCreateDebugUtilsMessengerEXT");
  if (func == nullptr ||
      func(m_Instance, &createInfo, nullptr, &m_DebugMessenger) != VK_SUCCESS) {
    throw std::runtime_error("Failed to set up debug messenger!");
  }
#endif // NDEBUG
}

void VulkanContext::CreateSurface(GLFWwindow *window) {
  if (glfwCreateWindowSurface(m_Instance, window, nullptr, &m_Surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface via GLFW!");
  }
}

void VulkanContext::PickPhysicalDevice() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(m_Instance, &deviceCount, nullptr);
  if (deviceCount == 0)
    throw std::runtime_error("No GPUs with Vulkan support found!");

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(m_Instance, &deviceCount, devices.data());
  m_PhysicalDevice = devices[0];

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(m_PhysicalDevice, &properties);
  std::string deviceName(properties.deviceName);
  LOG_INFO(std::format("[VulkanContext] Detected physical device: {}", deviceName));

  config::InitializeProfileFromGPU(deviceName);
}

void VulkanContext::CreateLogicalDevice() {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_PhysicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(m_PhysicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  bool found = false;
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      m_GraphicsQueueFamilyIndex = i;
      found = true;
      break;
    }
  }
  if (!found)
    throw std::runtime_error(
        "Could not find combined graphics and compute queue!");

  // Dedicated transfer queue (UE 5.8 RHI parity): a queue family advertising ONLY
  // VK_QUEUE_TRANSFER_BIT (no GRAPHICS, no COMPUTE) is a hardware copy engine on most discrete
  // GPUs -- streaming uploads (renderer::GeometryStreamingCoordinator's page copies) submitted
  // there run in parallel with, and never contend for, the graphics queue's own command
  // submission. Not every GPU exposes one (integrated GPUs / some vendors expose only the
  // combined graphics+compute family), so this is a graceful runtime fallback, never a hard
  // requirement -- m_HasDedicatedTransferQueue tells a consumer whether queue-family-ownership
  // transfer barriers are actually needed (same family == no transfer needed, see
  // GpuGeometryPagePool::UploadPageData/FinalizeBoundPage).
  m_TransferQueueFamilyIndex = m_GraphicsQueueFamilyIndex;
  m_HasDedicatedTransferQueue = false;
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
        !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        !(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      m_TransferQueueFamilyIndex = i;
      m_HasDedicatedTransferQueue = true;
      break;
    }
  }

  float queuePriority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  queueCreateInfos.reserve(2); // Fixed upfront: emplace_back below must never trigger a reallocation
                               // while an earlier element's pQueuePriorities (below) still points
                               // at this same-scope `queuePriority` local -- reserving avoids any
                               // risk of that, though pQueuePriorities is re-pointed at &queuePriority
                               // fresh for each entry regardless, so this is defense in depth.

  VkDeviceQueueCreateInfo graphicsQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  graphicsQueueCreateInfo.queueFamilyIndex = m_GraphicsQueueFamilyIndex;
  graphicsQueueCreateInfo.queueCount = 1;
  graphicsQueueCreateInfo.pQueuePriorities = &queuePriority;
  queueCreateInfos.push_back(graphicsQueueCreateInfo);

  if (m_HasDedicatedTransferQueue) {
    VkDeviceQueueCreateInfo transferQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    transferQueueCreateInfo.queueFamilyIndex = m_TransferQueueFamilyIndex;
    transferQueueCreateInfo.queueCount = 1;
    transferQueueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(transferQueueCreateInfo);
  }

  VkPhysicalDeviceVulkan13Features features13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  features13.synchronization2 = VK_TRUE;
  features13.dynamicRendering = VK_TRUE;

  // renderer::ClusterHardwareRasterPass / renderer::ClusterOcclusionCullingPass
  // consume this feature's vkCmdDrawIndexedIndirectCount (the whole point of
  // driving the hardware raster path's draw count from a GPU-written buffer
  // instead of a CPU readback) -- without it the Vulkan 1.3 core function is
  // present but every call is a validation error / undefined behavior
  // (VUID-vkCmdDrawIndexedIndirectCount-None-04445), and no hardware-routed
  // cluster ever actually draws.
  VkPhysicalDeviceVulkan12Features features12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  features12.drawIndirectCount = VK_TRUE;
  // Required by VK_KHR_acceleration_structure (BLAS/TLAS build input buffers and the
  // acceleration structure's own backing buffer are addressed by GPU virtual address, not bound
  // via a descriptor) -- see renderer::AccelerationStructure / SurfaceCachePass's vertex/index
  // buffers, both created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT for exactly this reason.
  features12.bufferDeviceAddress = VK_TRUE;

  // renderer::SDFRayMarchPass's compute shader (SDFRayMarch.comp) samples one of a fixed-size
  // array of per-entity Mesh SDF textures, picking WHICH element per invocation from a CPU-built
  // BVH traversal result (geometry::EntityBVH) -- a genuinely bindless-style access pattern per
  // CLAUDE.md's "Descriptor Array massif (Bindless)" architecture mandate, the first consumer of
  // it in this codebase. Because the chosen array index varies per shader invocation (two
  // neighboring rays can easily pick different entities) rather than being uniform across the
  // whole subgroup, GLSL requires the access be wrapped in nonuniformEXT()
  // (GL_EXT_nonuniform_qualifier) to avoid undefined behavior on hardware that would otherwise
  // assume a uniform index -- and that qualifier itself requires this feature bit enabled at
  // device creation (VUID-RuntimeSpirv-NonUniform-06274 without it).
  features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

  // renderer::ClusterSoftwareRasterPass's atomic Visibility Buffer
  // (ClusterSoftwareRaster.comp / ClearVisBufferAtomic.comp) needs
  // imageAtomicMax on a VK_FORMAT_R64_UINT storage image -- SPIR-V's
  // Int64ImageEXT capability, gated behind this extension's
  // shaderImageInt64Atomics feature (VK_EXT_shader_image_atomic_int64, widely
  // supported on desktop GPUs). shaderInt64 (core Vulkan 1.0 feature, below) is
  // the separate prerequisite for the uint64_t scalar type itself
  // (GL_EXT_shader_explicit_arithmetic_types_int64), used to pack/unpack the
  // atomic word.
  VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT imageAtomicInt64Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};
  imageAtomicInt64Features.shaderImageInt64Atomics = VK_TRUE;

  // Ray Tracing feature trio, per CLAUDE.md's mandatory "Required Hardware Extensions" list --
  // consumed by renderer::AccelerationStructure (BLAS/TLAS builder), renderer::
  // SurfaceCacheRayTracingPass (the VK_KHR_ray_tracing_pipeline SBT/rgen-rchit-rmiss pipeline,
  // Lumen-style Surface Cache hit lighting) and SurfaceCacheGIInject.comp's inline
  // rayQueryEXT-based hardware trace path (VK_KHR_ray_query -- an HWRT path callable from a plain
  // compute shader, which vkCmdTraceRaysKHR itself cannot be).
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelStructFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  accelStructFeatures.accelerationStructure = VK_TRUE;

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;

  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  rayQueryFeatures.rayQuery = VK_TRUE;

  imageAtomicInt64Features.pNext = &accelStructFeatures;
  accelStructFeatures.pNext = &rayTracingPipelineFeatures;
  rayTracingPipelineFeatures.pNext = &rayQueryFeatures;
  features12.pNext = &imageAtomicInt64Features;
  features13.pNext = &features12;

  VkPhysicalDeviceFeatures2 deviceFeatures2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  deviceFeatures2.pNext = &features13;
  // geometryShader: NOT used for an actual geometry-shader pipeline stage
  // anywhere in this engine -- required purely because SPIR-V's PrimitiveId
  // BuiltIn (gl_PrimitiveID), read by ClusterRaster.frag to derive each
  // cluster's local TriangleID for the Visibility Buffer, mandates the SPIR-V
  // "Geometry" capability even when only used in a fragment shader with no
  // geometry stage present in the pipeline (see the Vulkan/SPIR-V spec's
  // BuiltIn table). A near-universally-supported core Vulkan 1.0 feature bit on
  // desktop GPUs, so enabled unconditionally here, matching this function's
  // existing feature-enablement rigor ( synchronization2/dynamicRendering above
  // are requested the same way, with no prior vkGetPhysicalDeviceFeatures
  // support query).
  deviceFeatures2.features.geometryShader = VK_TRUE;
  // shaderInt64: see imageAtomicInt64Features's comment above.
  deviceFeatures2.features.shaderInt64 = VK_TRUE;
  // fragmentStoresAndAtomics (Phase 3, UE5.8 parity roadmap): SurfaceCacheCapture.frag -- a
  // fragment shader -- calls shadow_feedback.glsl's RequestShadowPageResidency(), which does an
  // atomicAdd + indexed write into a writable STORAGE_BUFFER (renderer::VirtualShadowMapPass's
  // feedback buffer). Per the Vulkan spec, ANY writable storage buffer/image/texel-buffer
  // variable in the fragment stage requires this feature enabled, or vkCreateGraphicsPipelines
  // fails validation (VUID-RuntimeSpirv-NonWritable-06340) -- a near-universally-supported core
  // Vulkan 1.0 feature bit on desktop GPUs (every GPU capable of this project's own mandatory
  // ray tracing / Int64 image atomics requirements already supports it), so enabled
  // unconditionally here, matching geometryShader's own enablement rigor above.
  deviceFeatures2.features.fragmentStoresAndAtomics = VK_TRUE;
  // multiDrawIndirect: renderer::TransparentForwardPass::RecordDraw() issues ONE
  // vkCmdDrawIndexedIndirect covering every static transparent leaf cluster in a single call
  // (drawCount = cluster count, e.g. 712 in this demo's scene -- see that class' own class
  // comment). Without this feature, the spec caps drawCount at 1
  // (VUID-vkCmdDrawIndexedIndirect-drawCount-02718) -- a near-universally-supported core Vulkan
  // 1.0 feature bit on desktop GPUs (every GPU capable of this project's own mandatory ray
  // tracing / mesh shader requirements already supports it), so enabled unconditionally here,
  // matching geometryShader's/fragmentStoresAndAtomics' own enablement rigor above.
  deviceFeatures2.features.multiDrawIndirect = VK_TRUE;
  // independentBlend (Phase PP3, post-process stack roadmap): renderer::TransparentForwardPass's
  // own forward pipeline now has 2 color attachments with DIFFERENT blend states (color: alpha-
  // blended "over" compositing; g_RefractionOffset: a plain overwrite, blendEnable=FALSE -- see
  // that pass' own pipeline-creation comment) -- without this feature, the spec requires every
  // element of VkPipelineColorBlendStateCreateInfo::pAttachments to be IDENTICAL
  // (VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605), which vkCreateGraphicsPipelines
  // then rejects. A near-universally-supported core Vulkan 1.0 feature bit on desktop GPUs, so
  // enabled unconditionally here, matching multiDrawIndirect's own enablement rigor above.
  deviceFeatures2.features.independentBlend = VK_TRUE;
  // tessellationShader (Phase 7a, UE5.8 parity roadmap): core Vulkan 1.0 feature bit, no device
  // extension required -- gates VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT/_EVALUATION_BIT
  // (renderer::HeroTessellationPass, the hero Icosphere's own screen-space-adaptive
  // displacement-mapped pipeline). A near-universally-supported core feature on desktop GPUs
  // (same rigor as geometryShader/fragmentStoresAndAtomics/multiDrawIndirect above), enabled
  // unconditionally here.
  deviceFeatures2.features.tessellationShader = VK_TRUE;

  VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  createInfo.pNext = &deviceFeatures2;
  createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();

  const std::vector<const char *> deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME,
      // VK_KHR_deferred_host_operations: required dependency of both extensions below (the
      // Vulkan spec lets a BLAS/TLAS build or an RT pipeline compile run as a deferred host
      // operation; this codebase never actually defers one -- every build/compile below passes
      // VK_NULL_HANDLE for the VkDeferredOperationKHR parameter and blocks -- but the extension
      // must still be enabled, since VK_KHR_acceleration_structure and
      // VK_KHR_ray_tracing_pipeline both declare it as a required device extension).
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_RAY_QUERY_EXTENSION_NAME};
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  if (vkCreateDevice(m_PhysicalDevice, &createInfo, nullptr, &m_Device) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create Logical Device!");
  }

  vkGetDeviceQueue(m_Device, m_GraphicsQueueFamilyIndex, 0, &m_GraphicsQueue);
  vkGetDeviceQueue(m_Device, m_TransferQueueFamilyIndex, 0, &m_TransferQueue);
  LOG_INFO(std::format("[VulkanContext] Transfer queue: {} (family {})",
      m_HasDedicatedTransferQueue ? "dedicated hardware copy queue" : "falling back to the graphics queue",
      m_TransferQueueFamilyIndex));

  // See RayTracingFunctions.h's own comment: VK_KHR_acceleration_structure /
  // VK_KHR_ray_tracing_pipeline's entry points are not re-exported by this SDK's loader import
  // library, so they must be resolved via vkGetDeviceProcAddr right here, once, immediately after
  // the device (which just enabled both extensions above) comes up.
  renderer::LoadRayTracingFunctions(m_Device, renderer::g_RTFunctions);
}

void VulkanContext::CreateCommandPool() {
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = m_GraphicsQueueFamilyIndex;
  if (vkCreateCommandPool(m_Device, &poolInfo, nullptr, &m_CommandPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create Command Pool!");
  }

  // Own pool for the transfer queue's per-frame command buffer (renderer::
  // GeometryStreamingCoordinator's page-copy uploads, see GetTransferCommandBuffer()) -- a command
  // pool is always tied to one specific queue family, so this must be a distinct pool even when
  // m_HasDedicatedTransferQueue is false and it targets the SAME family as m_CommandPool above
  // (Vulkan permits multiple pools per family; nothing here needs them to be the same pool).
  VkCommandPoolCreateInfo transferPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  transferPoolInfo.queueFamilyIndex = m_TransferQueueFamilyIndex;
  if (vkCreateCommandPool(m_Device, &transferPoolInfo, nullptr, &m_TransferCommandPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create Transfer Command Pool!");
  }
}

void VulkanContext::AllocateCommandBuffer() {
  VkCommandBufferAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.commandPool = m_CommandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(m_Device, &allocInfo, &m_CommandBuffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate principal Command Buffer!");
  }

  VkCommandBufferAllocateInfo transferAllocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  transferAllocInfo.commandPool = m_TransferCommandPool;
  transferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  transferAllocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(m_Device, &transferAllocInfo, &m_TransferCommandBuffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate Transfer Command Buffer!");
  }
}

void VulkanContext::CreateBindlessDescriptorSetLayout() {
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &binding;
  if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr,
                                  &m_BindlessLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create bindless set layout!");
  }
}

void VulkanContext::CreateDescriptorPool() {
  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  if (vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create main descriptor pool!");
  }
}

void VulkanContext::AllocateBindlessDescriptorSet() {
  VkDescriptorSetAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool = m_DescriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &m_BindlessLayout;
  if (vkAllocateDescriptorSets(m_Device, &allocInfo, &m_BindlessSet) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate bindless descriptor set!");
  }
}

void VulkanContext::CreateSwapchain(GLFWwindow *window) {
  m_SwapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

  // The window/config constants (config::WINDOW_WIDTH/HEIGHT) are a target, not a guarantee:
  // on a scaled/HiDPI display the compositor can report a currentExtent smaller than the
  // requested client size, and VK_KHR_surface REQUIRES imageExtent to fall within
  // [minImageExtent, maxImageExtent] or vkCreateSwapchainKHR is invalid usage (VUID-
  // VkSwapchainCreateInfoKHR-pNext-07781). Querying and clamping here is mandatory, not
  // optional -- skipping it previously threw "Failed to create Swapchain!" on any surface whose
  // reported extent didn't happen to match the config constants exactly.
  VkSurfaceCapabilitiesKHR surfaceCapabilities{};
  if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_PhysicalDevice, m_Surface,
                                                &surfaceCapabilities) !=
      VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to query physical device surface capabilities!");
  }

  // currentExtent.width == 0xFFFFFFFF is the surface's way of saying "you choose": in that case
  // fall back to the configured window size, still clamped to the surface's min/max. Otherwise
  // the compositor dictates a fixed extent (as on this HiDPI/scaled display) and every swapchain
  // image MUST match it exactly.
  if (surfaceCapabilities.currentExtent.width != 0xFFFFFFFFu) {
    m_SwapchainExtent = surfaceCapabilities.currentExtent;
  } else {
    m_SwapchainExtent = {config::WINDOW_WIDTH, config::WINDOW_HEIGHT};
    m_SwapchainExtent.width =
        std::clamp(m_SwapchainExtent.width,
                  surfaceCapabilities.minImageExtent.width,
                  surfaceCapabilities.maxImageExtent.width);
    m_SwapchainExtent.height =
        std::clamp(m_SwapchainExtent.height,
                  surfaceCapabilities.minImageExtent.height,
                  surfaceCapabilities.maxImageExtent.height);
  }

  VkSwapchainCreateInfoKHR createInfo{
      VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  createInfo.surface = m_Surface;
  uint32_t minImageCount = surfaceCapabilities.minImageCount + 1;
  if (surfaceCapabilities.maxImageCount > 0) {
    minImageCount =
        std::min(minImageCount, surfaceCapabilities.maxImageCount);
  }
  createInfo.minImageCount = std::max(minImageCount, 2u);
  createInfo.imageFormat = m_SwapchainImageFormat;
  createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  createInfo.imageExtent = m_SwapchainExtent;
  createInfo.imageArrayLayers = 1;
  // TRANSFER_SRC_BIT (in addition to the pre-existing DST_BIT used by RecordFrame's own final
  // blit) lets debugpipeline::ScreenshotCapture vkCmdCopyImageToBuffer directly out of a presented
  // swapchain image -- Debug-only consumer, but the bit itself must be requested here since
  // swapchain image usage is fixed at creation time. Universally supported alongside
  // COLOR_ATTACHMENT_BIT/TRANSFER_DST_BIT on every desktop Vulkan present engine.
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  createInfo.imageSharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // Fixed incorrect enum value
  createInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  createInfo.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(m_Device, &createInfo, nullptr, &m_Swapchain) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create Swapchain!");
  }

  uint32_t imageCount = 0;
  vkGetSwapchainImagesKHR(m_Device, m_Swapchain, &imageCount, nullptr);
  m_SwapchainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(m_Device, m_Swapchain, &imageCount,
                          m_SwapchainImages.data());
}

void VulkanContext::CreateImageViews() {
  m_SwapchainImageViews.resize(m_SwapchainImages.size());
  for (size_t i = 0; i < m_SwapchainImages.size(); i++) {
    VkImageViewCreateInfo createInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    createInfo.image = m_SwapchainImages[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = m_SwapchainImageFormat;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_Device, &createInfo, nullptr,
                          &m_SwapchainImageViews[i]) != VK_SUCCESS) {
      throw std::runtime_error("Failed to generate Swapchain Image Views!");
    }
  }
}

void VulkanContext::CreateSyncObjects() {
  VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  if (vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr,
                        &m_ImageAvailableSemaphore) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create synchronization semaphores!");
  }

  // One render-finished semaphore per swapchain image -- see
  // GetRenderFinishedSemaphore()'s header comment for why a single shared one
  // is unsafe once the per-frame vkDeviceWaitIdle is gone. Requires
  // m_SwapchainImages to already be populated (CreateSwapchain() runs before
  // CreateSyncObjects() in Init()).
  m_RenderFinishedSemaphores.resize(m_SwapchainImages.size());
  for (VkSemaphore &semaphore : m_RenderFinishedSemaphores) {
    if (vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &semaphore) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create synchronization semaphores!");
    }
  }

  // Signaled once per frame when the transfer queue's command buffer (GetTransferCommandBuffer())
  // finishes -- the graphics submission waits on this before touching any data it uploaded (see
  // main.cpp's per-frame submit sequence). A single binary semaphore is safe here (unlike
  // m_RenderFinishedSemaphores needing one per swapchain image): this one is only ever
  // signaled/waited within the same frame's own pair of submissions, never handed to
  // vkQueuePresentKHR or reused across a frame boundary before both its signal and its wait have
  // retired -- the frameFence wait at the top of main.cpp's loop already guarantees that.
  if (vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &m_TransferFinishedSemaphore) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create transfer-finished semaphore!");
  }
}

void VulkanContext::CreatePipelinesAndDescriptors() {
  VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
                                      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 1;
  poolInfo.poolSizeCount = 2;
  poolInfo.pPoolSizes = poolSizes;
  if (vkCreateDescriptorPool(m_Device, &poolInfo, nullptr,
                             &m_GeometryDescriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Geometry Descriptor Pool!");
  }

  VkDescriptorSetLayoutBinding bindings[5] = {};
  bindings[0].binding = 0; // Vertices SSBO
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags =
      VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

  bindings[1].binding = 1; // Indices SSBO
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags =
      VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;

  bindings[2].binding = 2; // Parameters UBO
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[3].binding =
      3; // Per-entity rotation SSBO (read only by the vertex shader)
  bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[3].descriptorCount = 1;
  bindings[3].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  bindings[4].binding = 4; // Entity data SSBO (CPU-authored, IDManager-assigned
                           // meshID; read only by the vertex shader)
  bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[4].descriptorCount = 1;
  bindings[4].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = 5;
  layoutInfo.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr,
                                  &m_GeometryLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Geometry Descriptor Layout!");
  }

  VkDescriptorSetAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool = m_GeometryDescriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &m_GeometryLayout;
  if (vkAllocateDescriptorSets(m_Device, &allocInfo,
                               &m_GeometryDescriptorSet) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate Geometry Descriptor Set!");
  }

  VkDescriptorBufferInfo vertInfo{m_VertexBuffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo indexInfo{m_IndexBuffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo paramsInfo{m_ParamsBuffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo entityTransformInfo{m_EntityTransformBuffer, 0,
                                             VK_WHOLE_SIZE};
  VkDescriptorBufferInfo entityDataInfo{m_EntityBuffer, 0, VK_WHOLE_SIZE};

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

  // Compute layout setup: shared by all non-box primitive generators, which
  // read their per-dispatch parameters from the Params UBO (binding = 2) bound
  // in m_GeometryDescriptorSet.
  VkPipelineLayoutCreateInfo computeLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  computeLayoutInfo.setLayoutCount = 1;
  computeLayoutInfo.pSetLayouts = &m_GeometryLayout;
  if (vkCreatePipelineLayout(m_Device, &computeLayoutInfo, nullptr,
                             &m_ComputePipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Compute Pipeline Layout!");
  }

  // Box push-constant layout: geom_box.comp reads its Params via push constants
  // instead of the shared UBO (see BoxPushConstants above), since each face
  // dispatch also needs its own specialization constants baked at
  // pipeline-creation time.
  VkPushConstantRange boxPushConstantRange{};
  boxPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  boxPushConstantRange.offset = 0;
  boxPushConstantRange.size = sizeof(BoxPushConstants);

  VkPipelineLayoutCreateInfo boxComputeLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  boxComputeLayoutInfo.setLayoutCount = 1;
  boxComputeLayoutInfo.pSetLayouts = &m_GeometryLayout;
  boxComputeLayoutInfo.pushConstantRangeCount = 1;
  boxComputeLayoutInfo.pPushConstantRanges = &boxPushConstantRange;
  if (vkCreatePipelineLayout(m_Device, &boxComputeLayoutInfo, nullptr,
                             &m_BoxComputePipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Box Compute Pipeline Layout!");
  }

  // One compute pipeline per non-box primitive: same descriptor-set-only
  // layout, each with its own shader module generating its own topology into
  // the shared Vertex/Index SSBOs.
  struct SimplePrimitivePipelineDesc {
    const char *shaderFile;
    VkPipeline *outPipeline;
  };
  const SimplePrimitivePipelineDesc simplePrimitives[] = {
      {"shaders/geom_cone.comp.spv", &m_ConePipeline},
      {"shaders/geom_icosphere.comp.spv", &m_IcospherePipeline},
      {"shaders/geom_plane.comp.spv", &m_PlanePipeline},
      {"shaders/geom_sphere.comp.spv", &m_SpherePipeline},
      {"shaders/geom_torus.comp.spv", &m_TorusPipeline},
      {"shaders/geom_tube.comp.spv", &m_TubePipeline},
      {"shaders/geom_capsule.comp.spv", &m_CapsulePipeline},
      {"shaders/geom_cylinder.comp.spv", &m_CylinderPipeline},
      {"shaders/geom_pyramide.comp.spv", &m_PyramidPipeline},
      {"shaders/geom_TorusKnot.comp.spv", &m_TorusKnotPipeline},
      {"shaders/geom_chamferBox.comp.spv", &m_ChamferBoxPipeline},
      {"shaders/geom_terrain.comp.spv", &m_TerrainPipeline},
      {"shaders/autosmooth.comp.spv", &m_AutosmoothPipeline},
  };
  for (const auto &desc : simplePrimitives) {
    auto code = VulkanPipeline::ReadShaderFile(desc.shaderFile);
    VkShaderModule module = VulkanPipeline::CreateShaderModule(m_Device, code);

    VkComputePipelineCreateInfo pipelineInfo{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.layout = m_ComputePipelineLayout;
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                 nullptr, desc.outPipeline) != VK_SUCCESS) {
      throw std::runtime_error(
          std::string("Failed to create compute pipeline for ") +
          desc.shaderFile);
    }
    vkDestroyShaderModule(m_Device, module, nullptr);
  }

  // Box: 6 compute pipelines built from the same shader module, one per cube
  // face, differentiated only by VkSpecializationInfo (axis mapping + winding,
  // see kBoxFaceSpecs).
  {
    auto boxCode = VulkanPipeline::ReadShaderFile("shaders/geom_box.comp.spv");
    VkShaderModule boxModule =
        VulkanPipeline::CreateShaderModule(m_Device, boxCode);

    VkSpecializationMapEntry mapEntries[7] = {
        {0, offsetof(BoxFaceSpecConstants, uAxis), sizeof(int32_t)},
        {1, offsetof(BoxFaceSpecConstants, vAxis), sizeof(int32_t)},
        {2, offsetof(BoxFaceSpecConstants, wAxis), sizeof(int32_t)},
        {3, offsetof(BoxFaceSpecConstants, faceMode), sizeof(int32_t)},
        {4, offsetof(BoxFaceSpecConstants, udir), sizeof(float)},
        {5, offsetof(BoxFaceSpecConstants, vdir), sizeof(float)},
        {6, offsetof(BoxFaceSpecConstants, wSign), sizeof(float)},
    };

    for (uint32_t face = 0; face < 6u; ++face) {
      VkSpecializationInfo specInfo{};
      specInfo.mapEntryCount = 7;
      specInfo.pMapEntries = mapEntries;
      specInfo.dataSize = sizeof(BoxFaceSpecConstants);
      specInfo.pData = &kBoxFaceSpecs[face];

      VkComputePipelineCreateInfo pipelineInfo{
          VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipelineInfo.layout = m_BoxComputePipelineLayout;
      pipelineInfo.stage.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pipelineInfo.stage.module = boxModule;
      pipelineInfo.stage.pName = "main";
      pipelineInfo.stage.pSpecializationInfo = &specInfo;

      if (vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                   nullptr,
                                   &m_BoxFacePipelines[face]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Box face compute pipeline!");
      }
    }
    vkDestroyShaderModule(m_Device, boxModule, nullptr);
  }

  // Graphics layout setup with custom PushConstant sizing matching
  // CameraPushConstants
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(CameraPushConstants);

  VkPipelineLayoutCreateInfo graphicsLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  graphicsLayoutInfo.setLayoutCount = 1;
  graphicsLayoutInfo.pSetLayouts = &m_GeometryLayout;
  graphicsLayoutInfo.pushConstantRangeCount = 1;
  graphicsLayoutInfo.pPushConstantRanges = &pushConstantRange;
  if (vkCreatePipelineLayout(m_Device, &graphicsLayoutInfo, nullptr,
                             &m_GraphicsPipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Graphics Pipeline Layout!");
  }

  auto vertCode = VulkanPipeline::ReadShaderFile("shaders/draw.vert.spv");
  auto fragCode = VulkanPipeline::ReadShaderFile("shaders/draw.frag.spv");
  VkShaderModule vertModule =
      VulkanPipeline::CreateShaderModule(m_Device, vertCode);
  VkShaderModule fragModule =
      VulkanPipeline::CreateShaderModule(m_Device, fragCode);

  // Visibility Buffer: 2 color attachments (ClusterID, local TriangleID)
  // instead of the swapchain's own presentable format -- see VulkanContext.h's
  // kVisBufferFormat comment.
  std::array<VkFormat, 2> visBufferFormats{kVisBufferFormat, kVisBufferFormat};
  m_GraphicsPipeline = VulkanPipeline::CreateGraphicsPipeline(
      m_Device, m_GraphicsPipelineLayout, vertModule, fragModule,
      visBufferFormats, m_DepthFormat);

  vkDestroyShaderModule(m_Device, vertModule, nullptr);
  vkDestroyShaderModule(m_Device, fragModule, nullptr);
}

void VulkanContext::DispatchGeometryCompute(
    VkPipeline pipeline, VkPipelineLayout layout, const void *uboParamsData,
    size_t uboParamsSize, const void *pushConstantData, size_t pushConstantSize,
    uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
  // Every primitive except the box drives geom_*.comp through the shared Params
  // UBO (binding = 2): overwrite it with this dispatch's parameters before
  // recording the command buffer that reads it.
  if (uboParamsData != nullptr) {
    void *mapped = nullptr;
    if (vmaMapMemory(m_Allocator, m_ParamsAllocation, &mapped) != VK_SUCCESS) {
      throw std::runtime_error(
          "Failed to map Params UBO for geometry dispatch!");
    }
    std::memcpy(mapped, uboParamsData, uboParamsSize);
    vmaUnmapMemory(m_Allocator, m_ParamsAllocation);
  }

  renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                            &m_GeometryDescriptorSet, 0, nullptr);

    if (pushConstantData != nullptr) {
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         static_cast<uint32_t>(pushConstantSize),
                         pushConstantData);
    }

    vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);

    // Memory layout safe transition barrier execution: compute writes to the
    // shared Vertex/Index SSBOs must be visible to the vertex shader stage that
    // reads them at draw time.
    VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &memBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  });
}

maths::vec2 VulkanContext::GridSlot(int slotIndex) const {
  // Feature-gallery layout: 9 zones on a 3x3 macro-grid (kZonePitch apart, in the XZ plane,
  // Y = 0 ground level), each zone dedicated to ONE engine feature so the base scene reads as a
  // side-by-side showcase instead of one dense grid of randomly-skinned primitives. Zones that
  // hold two primitives offset them by kPairOffset either side of the zone center so both remain
  // clearly separated (max primitive half-width here is 0.8, so a 2.0-unit pair separation never
  // overlaps). Used both by GenerateGeometry() (to bake each primitive's world position at
  // generation time) and UpdateEntityRotations() (to recover each entity's rotation pivot) --
  // single source of truth for the layout, duplicated (deliberately, see its own comment) by
  // renderer::MegaLightsTypes.cpp's EntityGridPosition() for MegaLights placement.
  //
  //   col -1        col 0          col 1
  //   row -1  NANITE zone   WPO/displacement   METAL zone
  //           (icosphere +  zone (cone, sways   (chrome box +
  //           torus knot)   via WPO)            gold capsule)
  //   row  0  DIELECTRIC    LUMEN/GI zone       TRANSPARENT/
  //           zone (plane + (chamfer box in a   GLASS zone
  //           pyramid)      2-wall color-bounce (clear sphere)
  //                         corner)
  //   row  1  TRANSLUCENT   EMISSIVE zone       MEGALIGHTS zone
  //           zone (torus)  (glowing tube)      (cylinder lit by
  //                                             ~200 stochastic
  //                                             point lights)
  constexpr float kZonePitch = 4.0f;
  constexpr float kPairOffset = 1.0f;

  struct ZoneEntry { float col, row, pairOffset; };
  static constexpr std::array<ZoneEntry, 12> kLayout = {{
      /* 0  Box (metal A, chrome)      */ {  1.0f, -1.0f, -kPairOffset },
      /* 1  Cone (WPO/displacement)    */ {  0.0f, -1.0f,  0.0f },
      /* 2  Icosphere (Nanite A)       */ { -1.0f, -1.0f, -kPairOffset },
      /* 3  Plane (dielectric A)       */ { -1.0f,  0.0f, -kPairOffset },
      /* 4  Sphere (glass/transparent) */ {  1.0f,  0.0f,  0.0f },
      /* 5  Torus (translucent)        */ { -1.0f,  1.0f,  0.0f },
      /* 6  Tube (emissive)            */ {  0.0f,  1.0f,  0.0f },
      /* 7  Capsule (metal B, gold)    */ {  1.0f, -1.0f,  kPairOffset },
      /* 8  Cylinder (MegaLights hero) */ {  1.0f,  1.0f,  0.0f },
      /* 9  Pyramid (dielectric B)     */ { -1.0f,  0.0f,  kPairOffset },
      /* 10 TorusKnot (Nanite B)       */ { -1.0f, -1.0f,  kPairOffset },
      /* 11 ChamferBox (Lumen/GI hero) */ {  0.0f,  0.0f,  0.0f },
  }};

  const ZoneEntry& z = kLayout[static_cast<size_t>(slotIndex)];
  return maths::vec2{z.col * kZonePitch + z.pairOffset, z.row * kZonePitch};
}

void VulkanContext::BuildEntityData() {
  // Context 0: this is currently the only "factory" allocating entity IDs in
  // the engine. Sequence starts at 0, so the 12 sequential GetNextID() calls
  // below deterministically produce 0..11 in the low 32 bits, matching
  // kEntityCount's dense-index requirement.
  core::IDManager::Init(0);

  // One deliberately hand-authored material per entity (materialID == entity index), each entity
  // demonstrating exactly one named engine feature (metal, dielectric, glass, translucent,
  // emissive, ...) instead of a randomly-rolled category -- see
  // renderer::GenerateShowcaseMaterialTable's own comment for the full per-entity mapping.
  m_MaterialTable = renderer::GenerateShowcaseMaterialTable();

  for (uint32_t i = 0; i < kEntityCount; ++i) {
    core::EntityID id = core::IDManager::GetNextID();

    core::EntityData &entity = m_EntityData[i];
    entity.meshID = static_cast<uint32_t>(id & 0xFFFFFFFFu);
    entity.materialID = i;
    entity.cellID = 0u;
    entity.flags = 0u;
    core::SetFlag(entity.flags, core::EntityFlags::CastShadows, true);

    bool isTransparent = m_MaterialTable.isTransparent[i];
    // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): the Icosphere (kHeroEntityIndex)
    // is the single tessellated/displaced hero asset, rendered ONLY by
    // renderer::HeroTessellationPass -- never by the opaque Nanite VisBuffer pipeline (no
    // representation for runtime-displaced geometry there) nor by TransparentForwardPass.
    // Overrides its materialID to the reserved renderer::kHeroMaterialID slot (see that
    // constant's own comment) and forces its entity IsTransparent flag true -- NOT because it's
    // actually alpha-blended (kHeroMaterialID's own alpha is 1.0, fully opaque), but because
    // ClusterLODCompact.comp's existing per-entity IsTransparent exclusion (see that shader's own
    // EntityDataBuffer comment) is the exact "never enters the opaque candidate list" mechanism
    // this entity also needs. TransparentForwardPass itself stays unaffected: it filters by
    // materialTable.isTransparent[materialID], which correctly stays false for kHeroMaterialID
    // (GenerateShowcaseMaterialTable() never sets it true -- see that function's own hero-recipe
    // comment), so the hero entity's clusters never enter ITS candidate list either.
    if (i == kHeroEntityIndex) {
      entity.materialID = renderer::kHeroMaterialID;
      isTransparent = true;
    }
    // Phase 7b (UE5.8 parity roadmap, terrain heightfield): the floor entity is now a procedural
    // terrain heightfield (see GenerateGeometry()'s own terrain block), not a flat plane -- override
    // its materialID to the reserved renderer::kTerrainMaterialID slot so ClusterResolve.comp/
    // ClusterResolveBinned.comp's height/slope biome blend (terrain_shading.glsl) applies to it.
    // Unlike the hero entity, the terrain stays fully opaque and needs no IsTransparent exclusion --
    // it renders through the normal opaque Nanite path unmodified (see kTerrainMaterialID's own
    // comment).
    if (i == kFloorEntityIndex) {
      entity.materialID = renderer::kTerrainMaterialID;
    }
    // Phase 7c (UE5.8 parity roadmap, water/erosion): the water entity -- same exclusion mechanism
    // as the hero entity's own override above (forced IsTransparent so ClusterLODCompact.comp
    // never routes its clusters into the opaque candidate list; TransparentForwardPass itself stays
    // unaffected since materialTable.isTransparent[kWaterMaterialID] correctly stays false, see
    // that constant's own comment) -- water is rendered ONLY by renderer::WaterForwardPass.
    if (i == kWaterEntityIndex) {
      entity.materialID = renderer::kWaterMaterialID;
      isTransparent = true;
    }
    core::SetFlag(entity.flags, core::EntityFlags::IsTransparent, isTransparent);

    // Phase 1 (Nanite advanced): entity 6 (Tube) demos runtime Hermite-spline bending, always
    // opaque (GenerateShowcaseMaterialTable() already curates slot 6 with alpha == 1.0, see that
    // function's own comment) and unaffected by the hero-tessellation override above (kHeroEntityIndex
    // == 2, not 6), so it stays on the normal Nanite VisBuffer/ClusterResolve path where
    // ApplySplineDeformation() runs.
    //
    // Enhanced procedural displacement was originally authored on entity 2 (Icosphere), but Phase 7a's
    // concurrently-developed hero-asset tessellation independently claimed the same entity
    // (kHeroEntityIndex == 2) and routes it exclusively through renderer::HeroTessellationPass, which
    // never reaches ClusterRaster.vert/cluster_software_raster_core.glsl/ClusterResolve*.comp -- the
    // only places ApplyEnhancedDisplacement() is ever called. Rather than ship an EntityFlags bit that
    // is silently a no-op, reassigned to entity 10 (TorusKnot), the zone layout's own "Nanite B" pairing
    // to entity 2's "Nanite A" (see kLayout above) -- a normal opaque entity untouched by any other
    // concurrent feature's override, so ApplyEnhancedDisplacement() actually runs for it.
    if (i == 10u) {
      core::SetFlag(entity.flags, core::EntityFlags::HasEnhancedDisplacement, true);
    }
    if (i == 6u) {
      core::SetFlag(entity.flags, core::EntityFlags::HasSplineDeformation, true);
    }
  }

  // --- Runtime World Partition streaming pool (see kStreamingUnitCount's own comment) ---
  // Continues the SAME core::IDManager sequence the loop above started (deliberately not reset --
  // see BuildEntityData()'s own "Context 0" comment), so meshIDs stay dense across the whole
  // [0, kTotalEntityCount) range. Every slot starts fully valid (a real, small, pre-baked mesh --
  // see GenerateGeometry()'s streaming block) but core::EntityFlags::StreamingInactive so it never
  // draws until world::WorldCellStreamingLoader's main-thread pump claims it via
  // SetStreamingUnitState().
  for (uint32_t i = kEntityCount; i < kTotalEntityCount; ++i) {
    core::EntityID id = core::IDManager::GetNextID();
    uint32_t unit = (i - kEntityCount) / kStreamingSlotsPerUnit;
    uint32_t shape = unit % kStreamingArchetypeShapeCount;

    core::EntityData &entity = m_EntityData[i];
    entity.meshID = static_cast<uint32_t>(id & 0xFFFFFFFFu);
    entity.materialID = renderer::kStreamingArchetypeMaterialIDBase + shape;
    entity.cellID = 0u;
    entity.flags = 0u;
    core::SetFlag(entity.flags, core::EntityFlags::CastShadows, true);
    core::SetFlag(entity.flags, core::EntityFlags::StreamingInactive, true);

    // Streaming archetype materials are always opaque (see MaterialParameterTable.h's own
    // kStreamingArchetypeMaterialIDBase comment) -- looked up by materialID, NOT by entity index i
    // as the loop above does, since materialID == i only holds for the fixed showcase gallery.
    bool isTransparent = m_MaterialTable.isTransparent[entity.materialID];
    core::SetFlag(entity.flags, core::EntityFlags::IsTransparent, isTransparent);
  }
}

void VulkanContext::UploadEntityData() {
  // m_EntityBuffer is VMA_MEMORY_USAGE_GPU_ONLY (not host-visible), so
  // uploading the CPU-authored m_EntityData requires a temporary host-visible
  // staging buffer plus an explicit GPU-side copy, mirroring the
  // one-time-submit pattern used elsewhere in Init().
  VkDeviceSize uploadSize = sizeof(core::EntityData) * kTotalEntityCount;

  VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  stagingInfo.size = uploadSize;
  stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo stagingAllocInfo{};
  stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
  stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingAllocation = VK_NULL_HANDLE;
  VmaAllocationInfo stagingAllocResultInfo{};
  if (vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo,
                      &stagingBuffer, &stagingAllocation,
                      &stagingAllocResultInfo) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate entity data staging buffer!");
  }
  std::memcpy(stagingAllocResultInfo.pMappedData, m_EntityData.data(),
              static_cast<size_t>(uploadSize));

  VkCommandBufferAllocateInfo cmdAllocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = uploadSize;
    vkCmdCopyBuffer(cmd, stagingBuffer, m_EntityBuffer, 1, &copyRegion);

    // Explicit layout/ownership-free memory barrier: the copy's writes must be
    // visible to the vertex shader's storage-buffer reads (draw.vert, binding =
    // 4) before any draw call.
    VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &memBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  });

  vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
}

void VulkanContext::GenerateBox(
    float Width, float Length, float Height,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Width > 0.0f);
  assert(Length > 0.0f);
  assert(Height > 0.0f);

  uint32_t WidthSegments = std::max(1u, static_cast<uint32_t>(std::round(Width / config::VERTEX_SPACING)));
  uint32_t LengthSegments = std::max(1u, static_cast<uint32_t>(std::round(Length / config::VERTEX_SPACING)));
  uint32_t HeightSegments = std::max(1u, static_cast<uint32_t>(std::round(Height / config::VERTEX_SPACING)));

  for (uint32_t face = 0; face < 6u; ++face) {
    BoxPushConstants params{};
    params.width = Width;
    params.length = Length;
    params.height = Height;
    params.widthSegments = WidthSegments;
    params.lengthSegments = LengthSegments;
    params.heightSegments = HeightSegments;
    params.meshID = meshID;
    params.materialID = 0.0f;
    params.vertexOffset = runningVertexOffset;
    params.indexOffset = runningIndexOffset;
    params.worldOffsetX = slot.x;
    params.worldOffsetY = 0.0f;
    params.worldOffsetZ = slot.y;

    uint32_t uSegs = 0, vSegs = 0;
    if (face == 0u || face == 1u) {
      uSegs = WidthSegments;
      vSegs = HeightSegments;
    } else if (face == 2u || face == 3u) {
      uSegs = WidthSegments;
      vSegs = LengthSegments;
    } else {
      uSegs = HeightSegments;
      vSegs = LengthSegments;
    }

    uint32_t uSegsCount = uSegs + 1u;
    uint32_t vSegsCount = vSegs + 1u;

    constexpr uint32_t kLocalSizeXY = 8u; // geom_box.comp local_size = (8, 8, 1)
    uint32_t groupCountX = (uSegsCount + kLocalSizeXY - 1u) / kLocalSizeXY;
    uint32_t groupCountY = (vSegsCount + kLocalSizeXY - 1u) / kLocalSizeXY;

    DispatchGeometryCompute(m_BoxFacePipelines[face],
                            m_BoxComputePipelineLayout, nullptr, 0, &params,
                            sizeof(params), groupCountX, groupCountY, 1);

    runningVertexOffset += uSegsCount * vSegsCount;
    runningIndexOffset += uSegs * vSegs * 6u;
  }
}

void VulkanContext::GenerateCone(
    float Radius1, float Radius2, float Height,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Radius1 > 0.0f);
  assert(Radius2 > 0.0f);
  assert(Height > 0.0f);

  constexpr float PI = 3.1415926535f;
  float maxRadius = std::max(Radius1, Radius2);
  uint32_t HeightSegments = std::max(1u, static_cast<uint32_t>(std::round(Height / config::VERTEX_SPACING)));
  uint32_t CapSegments = std::max(1u, static_cast<uint32_t>(std::round(maxRadius / config::VERTEX_SPACING)));
  uint32_t Sides = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * maxRadius / config::VERTEX_SPACING)));

  ConeParams params{};
  params.radius1 = Radius1;
  params.radius2 = Radius2;
  params.height = Height;
  params.heightSegments = HeightSegments;
  params.capSegments = CapSegments;
  params.sides = Sides;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = -params.height * 0.5f; // recenter: geometry spans y=[0,height]
  params.worldOffsetZ = slot.y;

  uint32_t sideVertCount = (params.sides + 1u) * (params.heightSegments + 1u);
  uint32_t capVertCount = 1u + params.capSegments * (params.sides + 1u);
  uint32_t totalVerts = sideVertCount + 2u * capVertCount;

  constexpr uint32_t kLocalSizeX = 64u;
  uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

  DispatchGeometryCompute(m_ConePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;
  // Each cap's innermost ring is a fan from the center vertex (1 triangle/column, 3 indices),
  // every ring above it a genuine quad strip (2 triangles/column, 6 indices) -- see
  // geom_cone.comp's own comment on why this differs from a uniform 6-indices-per-ring stride.
  uint32_t capIndexCount = params.sides * 3u + (params.capSegments - 1u) * params.sides * 6u;
  runningIndexOffset += params.sides * params.heightSegments * 6u + 2u * capIndexCount;
}

void VulkanContext::GenerateSphere(
    float Radius,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Radius > 0.0f);

  constexpr float PI = 3.1415926535f;
  uint32_t Segments = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * Radius / config::VERTEX_SPACING)));

  SphereParams params{};
  params.radius = Radius;
  params.segments = Segments;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = 0.0f;
  params.worldOffsetZ = slot.y;

  uint32_t ringCount = params.segments - 2u; // interior latitude rings (excludes poles)
  uint32_t ringStride = params.segments + 1u;
  uint32_t vertCount = ringCount * ringStride + 2u;
  constexpr uint32_t kLocalSizeX = 64u; // geom_sphere.comp local_size_x = 64
  uint32_t groupCount = (vertCount + kLocalSizeX - 1u) / kLocalSizeX;

  DispatchGeometryCompute(m_SpherePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += vertCount;
  runningIndexOffset += 6u * params.segments * ringCount;
}

void VulkanContext::GenerateIcosphere(
    float Radius, bool Tetra, bool Octa, bool Icosa,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset,
    uint32_t& outBaseFaceCount, uint32_t& outVertsPerFace) {
  // Validate dimensions and segments are positive and non-zero
  assert(Radius > 0.0f);
  assert((Tetra ? 1 : 0) + (Octa ? 1 : 0) + (Icosa ? 1 : 0) == 1);

  float edgeLength = (Icosa ? 1.05f : (Octa ? 1.41f : 1.63f)) * Radius;
  uint32_t Segments = std::max(1u, static_cast<uint32_t>(std::round(edgeLength / config::VERTEX_SPACING)));

  IcosphereParams params{};
  params.radius = Radius;
  params.segments = Segments;
  params.tetra = Tetra ? 1u : 0u;
  params.octa = Octa ? 1u : 0u;
  params.icosa = Icosa ? 1u : 0u;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = 0.0f;
  params.worldOffsetZ = slot.y;

  outBaseFaceCount = Tetra ? 4u : (Octa ? 8u : 20u);

  // Dispatch sizing must match geom_icosphere.comp's local_size = (8, 8, 1):
  // X and Y cover the triangular grid coordinates (i, j) in [0, segments], Z
  // covers the base faces.
  constexpr uint32_t kLocalSizeXY = 8u;
  uint32_t gridExtent = params.segments + 1u; // valid i,j range is [0, segments] inclusive
  uint32_t groupCountXY = (gridExtent + kLocalSizeXY - 1u) / kLocalSizeXY;

  DispatchGeometryCompute(m_IcospherePipeline, m_ComputePipelineLayout,
                          &params, sizeof(params), nullptr, 0, groupCountXY,
                          groupCountXY, outBaseFaceCount);

  outVertsPerFace = (params.segments + 1u) * (params.segments + 2u) / 2u;
  uint32_t totalVerts = outVertsPerFace * outBaseFaceCount;
  uint32_t icosphereIndexCount = outBaseFaceCount * params.segments * params.segments * 3u;

  runningVertexOffset += totalVerts;
  runningIndexOffset += icosphereIndexCount;
}

void VulkanContext::GenerateCylinder(
    float Radius, float Height,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Radius > 0.0f);
  assert(Height > 0.0f);

  constexpr float PI = 3.1415926535f;
  uint32_t HeightSegments = std::max(1u, static_cast<uint32_t>(std::round(Height / config::VERTEX_SPACING)));
  uint32_t CapSegments = std::max(1u, static_cast<uint32_t>(std::round(Radius / config::VERTEX_SPACING)));
  uint32_t Sides = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * Radius / config::VERTEX_SPACING)));

  CylinderParams params{};
  params.radius = Radius;
  params.height = Height;
  params.heightSegments = HeightSegments;
  params.capSegments = CapSegments;
  params.sides = Sides;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = -params.height * 0.5f; // recenter: geometry spans y=[0,height]
  params.worldOffsetZ = slot.y;

  uint32_t sideColumns = params.sides + 1u;
  uint32_t sideVertCount = sideColumns * (params.heightSegments + 1u);
  uint32_t capVertCount = 1u + params.capSegments * sideColumns; // center + rings
  uint32_t totalVerts = sideVertCount + 2u * capVertCount;
  constexpr uint32_t kLocalSizeX = 64u; // geom_cylinder.comp local_size_x = 64
  uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

  DispatchGeometryCompute(m_CylinderPipeline, m_ComputePipelineLayout,
                          &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;
  // Each cap's innermost ring is a fan from the center vertex (1 triangle/column, 3 indices),
  // every ring above it a genuine quad strip (2 triangles/column, 6 indices) -- see
  // geom_cylinder.comp's own comment on why this differs from a uniform 6-indices-per-ring stride.
  uint32_t capIndexCount = params.sides * 3u + (params.capSegments - 1u) * params.sides * 6u;
  runningIndexOffset += 6u * params.sides * params.heightSegments + 2u * capIndexCount;
}

void VulkanContext::GenerateTube(
    float Radius1, float Radius2, float Height,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Radius1 > 0.0f);
  assert(Radius2 > 0.0f);
  assert(Radius1 > Radius2);
  assert(Height > 0.0f);

  constexpr float PI = 3.1415926535f;
  uint32_t HeightSegments = std::max(1u, static_cast<uint32_t>(std::round(Height / config::VERTEX_SPACING)));
  uint32_t CapSegments = std::max(1u, static_cast<uint32_t>(std::round((Radius1 - Radius2) / config::VERTEX_SPACING)));
  uint32_t Sides = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * Radius1 / config::VERTEX_SPACING)));

  TubeParams params{};
  params.radius1 = Radius1;
  params.radius2 = Radius2;
  params.height = Height;
  params.heightSegments = HeightSegments;
  params.capSegments = CapSegments;
  params.sides = Sides;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = -params.height * 0.5f; // recenter: geometry spans y=[0,height]
  params.worldOffsetZ = slot.y;

  uint32_t sideColumns = params.sides + 1u;
  uint32_t outerSideVertCount = sideColumns * (params.heightSegments + 1u);
  uint32_t innerSideVertCount = sideColumns * (params.heightSegments + 1u);
  uint32_t bottomCapVertCount = sideColumns * (params.capSegments + 1u);
  uint32_t topCapVertCount = sideColumns * (params.capSegments + 1u);
  uint32_t totalVerts = outerSideVertCount + innerSideVertCount +
                        bottomCapVertCount + topCapVertCount;

  constexpr uint32_t kLocalSizeX = 64u;
  uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

  DispatchGeometryCompute(m_TubePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 12u * params.sides * params.heightSegments +
                        12u * params.sides * params.capSegments;
}

void VulkanContext::GenerateTorus(
    float Radius1, float Radius2, float Rotation, float Twist,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Radius1 > 0.0f);
  assert(Radius2 > 0.0f);

  constexpr float PI = 3.1415926535f;
  uint32_t Segments = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * Radius1 / config::VERTEX_SPACING)));
  uint32_t Sides = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * Radius2 / config::VERTEX_SPACING)));

  TorusParams params{};
  params.radius1 = Radius1;
  params.radius2 = Radius2;
  params.rotation = Rotation;
  params.twist = Twist;
  params.segments = Segments;
  params.sides = Sides;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = 0.0f;
  params.worldOffsetZ = slot.y;

  uint32_t stride = params.sides + 1u;
  uint32_t vertCount = (params.segments + 1u) * stride;
  constexpr uint32_t kLocalSizeX = 64u;
  uint32_t groupCount = (vertCount + kLocalSizeX - 1u) / kLocalSizeX;

  DispatchGeometryCompute(m_TorusPipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += vertCount;
  runningIndexOffset += 6u * params.segments * params.sides;
}

void VulkanContext::GeneratePyramid(
    float Width, float Depth, float Height,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Width > 0.0f);
  assert(Depth > 0.0f);
  assert(Height > 0.0f);

  uint32_t WidthSegments = std::max(1u, static_cast<uint32_t>(std::round(Width / config::VERTEX_SPACING)));
  uint32_t DepthSegments = std::max(1u, static_cast<uint32_t>(std::round(Depth / config::VERTEX_SPACING)));
  uint32_t HeightSegments = std::max(1u, static_cast<uint32_t>(std::round(Height / config::VERTEX_SPACING)));

  PyramidParams params{};
  params.width = Width;
  params.depth = Depth;
  params.height = Height;
  params.widthSegments = WidthSegments;
  params.depthSegments = DepthSegments;
  params.heightSegments = HeightSegments;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = -params.height * 0.5f; // recenter
  params.worldOffsetZ = slot.y;

  uint32_t baseColumns = params.widthSegments + 1u;
  uint32_t baseRows = params.depthSegments + 1u;
  uint32_t baseVertCount = baseColumns * baseRows;

  uint32_t sideZCols = params.widthSegments + 1u;
  uint32_t sideZRows = params.heightSegments + 1u;
  uint32_t sideZVerts = sideZCols * sideZRows;

  uint32_t sideXCols = params.depthSegments + 1u;
  uint32_t sideXRows = params.heightSegments + 1u;
  uint32_t sideXVerts = sideXCols * sideXRows;

  uint32_t totalVerts = baseVertCount + 2u * sideZVerts + 2u * sideXVerts;
  constexpr uint32_t kLocalSizeX = 64u;
  uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

  DispatchGeometryCompute(m_PyramidPipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;

  uint32_t baseQuads = params.widthSegments * params.depthSegments;
  // Each side face's topmost quad row borders the apex, where every column collapses to the
  // same duplicated vertex position (1 triangle/column there, 3 indices) instead of a genuine
  // quad (2 triangles/column, 6 indices) -- see geom_pyramide.comp's own comment on why this
  // differs from a uniform 6-indices-per-row stride.
  uint32_t sideZQuadIndexCount = params.widthSegments * 3u + (params.heightSegments - 1u) * params.widthSegments * 6u;
  uint32_t sideXQuadIndexCount = params.depthSegments * 3u + (params.heightSegments - 1u) * params.depthSegments * 6u;
  runningIndexOffset += 6u * baseQuads + 2u * sideZQuadIndexCount + 2u * sideXQuadIndexCount;
}

void VulkanContext::GeneratePlane(
    float Length, float Width,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset,
    float worldOffsetY, float spacing) {
  // Validate dimensions are positive and non-zero
  assert(Length > 0.0f);
  assert(Width > 0.0f);

  uint32_t LengthSegments = std::max(2u, static_cast<uint32_t>(std::round(Length / spacing)));
  uint32_t WidthSegments = std::max(2u, static_cast<uint32_t>(std::round(Width / spacing)));

  PlaneParams params{};
  params.width = Width;
  params.length_ = Length;
  params.widthSegments = WidthSegments;
  params.lengthSegments = LengthSegments;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = worldOffsetY;
  params.worldOffsetZ = slot.y;

  uint32_t totalVerts = params.widthSegments * params.lengthSegments;
  constexpr uint32_t kLocalSizeXY = 8u; // geom_plane.comp local_size = (8, 8, 1)
  uint32_t groupCountX = (params.widthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;
  uint32_t groupCountY = (params.lengthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;

  DispatchGeometryCompute(m_PlanePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (params.widthSegments - 1u) * (params.lengthSegments - 1u);
}

void VulkanContext::GenerateTerrain(
    float Length, float Width,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset,
    float worldOffsetY, float spacing) {
  // Validate dimensions are positive and non-zero
  assert(Length > 0.0f);
  assert(Width > 0.0f);

  uint32_t LengthSegments = std::max(2u, static_cast<uint32_t>(std::round(Length / spacing)));
  uint32_t WidthSegments = std::max(2u, static_cast<uint32_t>(std::round(Width / spacing)));

  // geom_terrain.comp's Params UBO is byte-identical to PlaneParams (see that struct's own
  // comment) -- reused directly rather than duplicating an identical struct.
  PlaneParams params{};
  params.width = Width;
  params.length_ = Length;
  params.widthSegments = WidthSegments;
  params.lengthSegments = LengthSegments;
  params.meshID = meshID;
  // Matches every other Generate*() call: the vertex-baked materialID field is not what drives
  // runtime shading (see geometry::ClusterIndexEntry::materialID's own comment -- that's stamped
  // from core::EntityData::materialID at cook time instead). The real terrain materialID override
  // is set in BuildEntityData(), not here.
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = worldOffsetY;
  params.worldOffsetZ = slot.y;

  uint32_t totalVerts = params.widthSegments * params.lengthSegments;
  constexpr uint32_t kLocalSizeXY = 8u; // geom_terrain.comp local_size = (8, 8, 1)
  uint32_t groupCountX = (params.widthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;
  uint32_t groupCountY = (params.lengthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;

  DispatchGeometryCompute(m_TerrainPipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (params.widthSegments - 1u) * (params.lengthSegments - 1u);
}

void VulkanContext::GenerateWaterPlane(
    float Width, float Length, uint32_t WidthSegments, uint32_t LengthSegments,
    uint32_t meshID, maths::vec2 slot, float worldOffsetY,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions and segments are positive and non-zero
  assert(Width > 0.0f);
  assert(Length > 0.0f);
  assert(WidthSegments > 0u);
  assert(LengthSegments > 0u);

  // geom_plane.comp's Params UBO -- reused directly, PlaneParams already declared above.
  PlaneParams params{};
  params.width = Width;
  params.length_ = Length;
  params.widthSegments = WidthSegments;
  params.lengthSegments = LengthSegments;
  params.meshID = meshID;
  params.materialID = 0.0f; // See GenerateTerrain()'s own identical comment on this field.
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = worldOffsetY; // Fixed water level -- NOT added to a sampled height.
  params.worldOffsetZ = slot.y;

  uint32_t totalVerts = params.widthSegments * params.lengthSegments;
  constexpr uint32_t kLocalSizeXY = 8u; // geom_plane.comp local_size = (8, 8, 1)
  uint32_t groupCountX = (params.widthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;
  uint32_t groupCountY = (params.lengthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;

  DispatchGeometryCompute(m_PlanePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (params.widthSegments - 1u) * (params.lengthSegments - 1u);
}

void VulkanContext::GenerateCapsule(
    float Radius, float Height,
    uint32_t meshID, maths::vec2 slot,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  // Validate dimensions are positive and non-zero
  assert(Radius > 0.0f);
  assert(Height > 0.0f);

  constexpr float PI = 3.1415926535f;
  uint32_t HeightSegs = std::max(1u, static_cast<uint32_t>(std::round(Height / config::VERTEX_SPACING)));
  uint32_t Sides = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * Radius / config::VERTEX_SPACING)));

  CapsuleParams params{};
  params.radius = Radius;
  params.height = Height;
  params.sides = Sides;
  params.heightSegs = HeightSegs;
  params.meshID = meshID;
  params.materialID = 0.0f;
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = -params.height * 0.5f; // recenter: vertical midpoint is height/2
  params.worldOffsetZ = slot.y;

  uint32_t sideColumns = params.sides + 1u;
  uint32_t hemiVerts = (params.heightSegs + 1u) * sideColumns; // nbCapSegs derived as heightSegs in shader
  uint32_t bodyVerts = (params.heightSegs + 1u) * sideColumns;
  uint32_t totalVerts = hemiVerts * 2u + bodyVerts;

  constexpr uint32_t kLocalSizeX = 64u; // geom_capsule.comp local_size_x = 64
  uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

  DispatchGeometryCompute(m_CapsulePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;
  // Each hemisphere's pole-adjacent ring is a fan over sideColumns duplicate pole vertices (1
  // triangle/column, 3 indices), every other ring a genuine quad strip (2 triangles/column, 6
  // indices) -- see geom_capsule.comp's own comment on why this differs from a uniform
  // 6-indices-per-ring stride. nbCapSegs == heightSegs in the shader (see its own comment).
  uint32_t hemiIndexCount = params.sides * 3u + (params.heightSegs - 1u) * params.sides * 6u;
  uint32_t bodyIndexCount = 6u * params.sides * params.heightSegs;
  runningIndexOffset += 2u * hemiIndexCount + bodyIndexCount;
}

void VulkanContext::GenerateGeometry() {
  // --- Gallery layout: 12 primitives arranged into 9 widely-separated feature zones (see
  // GridSlot()'s own comment for the full zone -> feature mapping). Generation order below is
  // Icosphere-first (so it lands at vertexOffset/indexOffset 0, keeping
  // DebugReadbackGeometrySample's fixed sampling window — designed around a single icosphere
  // living at the start of the buffers — valid unchanged); each primitive's gallery *position* is
  // independent of generation order and set via GridSlot(slotIndex) below.
  auto gridSlot = [this](int slotIndex) -> maths::vec2 {
    return GridSlot(slotIndex);
  };

  uint32_t runningVertexOffset = 0;
  uint32_t runningIndexOffset = 0;

  LOG_INFO("[GenerateGeometry] Generating 12 procedural primitives across a 9-zone "
           "feature-showcase gallery, plus the Lumen-corner walls and floor...");

  // -------------------------------------------------------------------------
  // ICOSPHERE (slot 2 visually) — generated first so it occupies buffer offset
  // 0.
  // -------------------------------------------------------------------------
  uint32_t icosphereVertsPerFace = 0;
  uint32_t icosphereIndexCount = 0;
  uint32_t icosphereBaseFaceCount = 20u;
  {
    maths::vec2 slot = gridSlot(2);

    // Target parameters to accept and strictly validate:
    // IcoSphere: Radius, Segments, Tetra, Octa, Icosa
    float Radius = 0.8f;
    bool Tetra = false;
    bool Octa = false;
    bool Icosa = true;

    GenerateIcosphere(Radius, Tetra, Octa, Icosa,
                      m_EntityData[2].meshID, slot,
                      runningVertexOffset, runningIndexOffset,
                      icosphereBaseFaceCount, icosphereVertsPerFace);

    float edgeLength = (Icosa ? 1.05f : (Octa ? 1.41f : 1.63f)) * Radius;
    uint32_t Segments = std::max(1u, static_cast<uint32_t>(std::round(edgeLength / config::VERTEX_SPACING)));
    icosphereIndexCount = icosphereBaseFaceCount * Segments * Segments * 3u;
  }
  // DEBUG: sample-readback the icosphere geometry, exactly as before this
  // feature, to catch regressions in the (still buffer-offset-0) icosphere
  // generation.
  DebugReadbackGeometrySample(icosphereVertsPerFace, icosphereIndexCount);

  // -------------------------------------------------------------------------
  // BOX (slot 0) — 6 compute dispatches, one per cube face, chained onto the
  // same meshID.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(0);
    GenerateBox(1.4f, 1.4f, 1.4f, m_EntityData[0].meshID, slot, runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // CONE (slot 1)
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(1);
    GenerateCone(0.7f, 0.35f, 1.4f, m_EntityData[1].meshID, slot, runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // PLANE (slot 3 visually)
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(3);

    // Target parameters to accept and strictly validate:
    // Plane: Length, Width, LengthSegments, WidthSegments
    float Length = 1.4f;
    float Width = 1.4f;

    GeneratePlane(Length, Width,
                  m_EntityData[3].meshID, slot,
                  runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // SPHERE / UV sphere (slot 4)
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(4);
    GenerateSphere(0.8f, m_EntityData[4].meshID, slot, runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // TORUS (slot 5)
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(5);

    // Target parameters to accept and strictly validate:
    // Torus: Radius 1, Radius 2, Rotation, Twist, Segments, Sides
    float Radius1 = 0.7f;
    float Radius2 = 0.22f;
    float Rotation = 0.0f;
    float Twist = 0.0f;

    GenerateTorus(Radius1, Radius2, Rotation, Twist,
                  m_EntityData[5].meshID, slot,
                  runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // TUBE (slot 6) — hollow cylinder (pipe): outer/inner wall + top/bottom
  // rings.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(6);

    // Target parameters to accept and strictly validate:
    // Tube: Radius 1, Radius 2, Height, HeightSegments, CapSegments, Sides
    float Radius1 = 0.7f; // outer
    float Radius2 = 0.5f; // inner
    float Height = 1.4f;

    GenerateTube(Radius1, Radius2, Height,
                 m_EntityData[6].meshID, slot,
                 runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // CAPSULE (slot 7) — cylindrical body + two hemispherical caps.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(7);

    // Target parameters to accept and strictly validate:
    // Capsule: Radius, Height, Sides, HeightSegs
    float Radius = 0.5f;
    float Height = 0.8f; // cylindrical body length

    GenerateCapsule(Radius, Height,
                    m_EntityData[7].meshID, slot,
                    runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // CYLINDER (slot 8) — flat top/bottom caps (unlike TUBE, which is hollow).
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(8);

    // Target parameters to accept and strictly validate:
    // Cylinder: Radius, Height, HeightSegments, CapSegments, Sides
    float Radius = 0.7f;
    float Height = 1.4f;

    GenerateCylinder(Radius, Height,
                     m_EntityData[8].meshID, slot,
                     runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // PYRAMID (slot 9) — square (4-sided) flat-shaded pyramid.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(9);

    // Target parameters to accept and strictly validate:
    // Pyramide: Width, Depth, Height, WidthSegments, DepthSegments,
    // HeightSegments
    float Width = 1.4f;
    float Depth = 1.4f;
    float Height = 1.2f;

    GeneratePyramid(Width, Depth, Height,
                    m_EntityData[9].meshID, slot,
                    runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // TORUS KNOT (slot 10) — (p,q) knotted tube.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(10);
    TorusKnotParams params{};
    params.radius = 0.5f;
    params.tube = 0.15f;
    params.p = 2u;
    params.q = 3u;
    
    // Calculate segments based on density: 1 vertex per 0.01m
    constexpr float PI = 3.1415926535f;
    float curveLength = 2.0f * PI * params.radius * std::max(params.p, params.q);
    params.nbRadSeg = std::max(3u, static_cast<uint32_t>(std::round(curveLength / config::VERTEX_SPACING)));
    params.nbSides = std::max(3u, static_cast<uint32_t>(std::round(2.0f * PI * params.tube / config::VERTEX_SPACING)));

    params.meshID = m_EntityData[10].meshID;
    params.materialID = 0.0f;
    params.vertexOffset = runningVertexOffset;
    params.indexOffset = runningIndexOffset;
    params.worldOffsetX = slot.x;
    params.worldOffsetY =
        0.0f; // shape is already centered on its own local origin
    params.worldOffsetZ = slot.y;

    uint32_t vertCount = (params.nbRadSeg + 1u) * (params.nbSides + 1u);
    constexpr uint32_t kLocalSizeX =
        64u; // geom_TorusKnot.comp local_size_x = 64
    uint32_t groupCount = (vertCount + kLocalSizeX - 1u) / kLocalSizeX;

    DispatchGeometryCompute(m_TorusKnotPipeline, m_ComputePipelineLayout,
                            &params, sizeof(params), nullptr, 0, groupCount, 1,
                            1);

    runningVertexOffset += vertCount;
    runningIndexOffset += 6u * params.nbRadSeg * params.nbSides;
  }

  // -------------------------------------------------------------------------
  // CHAMFER BOX (slot 11) — rounded/chamfered box built as a superellipsoid.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(11);
    ChamferBoxParams params{};
    params.width = 1.4f;
    params.height = 1.4f;
    params.depth = 1.4f;
    
    // Calculate segments based on density: 1 vertex per 0.01m
    params.heightSegs = std::max(3u, static_cast<uint32_t>(std::round(params.height / config::VERTEX_SPACING)));
    params.sideSegs = std::max(3u, static_cast<uint32_t>(std::round(2.0f * (params.width + params.depth) / config::VERTEX_SPACING)));

    params.chamferPower =
        6.0f; // > 2 required (see geom_chamferBox.comp); higher = sharper edges
    params.meshID = m_EntityData[11].meshID;
    params.materialID = 0.0f;
    params.vertexOffset = runningVertexOffset;
    params.indexOffset = runningIndexOffset;
    params.worldOffsetX = slot.x;
    params.worldOffsetY = 0.0f;
    params.worldOffsetZ = slot.y;

    uint32_t ringCount =
        params.heightSegs - 2u; // interior latitude rings (excludes poles)
    uint32_t ringStride = params.sideSegs + 1u;
    uint32_t vertCount = ringCount * ringStride + 2u;
    constexpr uint32_t kLocalSizeX =
        64u; // geom_chamferBox.comp local_size_x = 64
    uint32_t groupCount = (vertCount + kLocalSizeX - 1u) / kLocalSizeX;

    DispatchGeometryCompute(m_ChamferBoxPipeline, m_ComputePipelineLayout,
                            &params, sizeof(params), nullptr, 0, groupCount, 1,
                            1);

    runningVertexOffset += vertCount;
    // Same index topology as geom_sphere.comp: sideSegs*3 (top fan) +
    // (ringCount-1)*sideSegs*6 (middle quads) + sideSegs*3 (bottom fan) =
    // 6*sideSegs*ringCount.
    runningIndexOffset += 6u * params.sideSegs * ringCount;
  }

  // -------------------------------------------------------------------------
  // LUMEN/GI SHOWCASE CORNER (slots 12/13) — two static colored walls meeting at a right angle
  // around the ChamferBox (slot 11, zone (0,0)): a plain-colored diffuse corner is the clearest
  // possible demonstration of Lumen-style indirect color bounce (the neutral-gray ChamferBox
  // picks up the red/green tint purely from bounced light, with no material trickery of its own).
  //
  // Both walls are baked FLAT (geom_plane.comp always emits a horizontal Y = worldOffsetY
  // surface, same as the floor below) and stood up vertically at render time via a fixed
  // (non-animated) 90-degree rotation set in UpdateEntityRotations() -- reusing the exact same
  // "rotate about xform.center" pivot formula every spinning primitive already uses, just with a
  // constant angle instead of a time-varying one. kWallSpan is used for both the plane's `width`
  // (baked along local X) and `length` (baked along local Z) so each wall is square; after
  // rotation whichever baked axis lands on world Y becomes the wall's height.
  //
  // Pivot math (see UpdateEntityRotations()): baking flat at worldOffsetY = kWallCenterY and then
  // rotating about a pivot whose Y ALSO equals kWallCenterY keeps the wall's world-space footprint
  // (its X or Z position) exactly pinned at the wall's own slot -- any mismatch between the bake
  // height and the pivot height would shear the wall sideways instead of just standing it up.
  {
    constexpr float kWallSpan = 3.0f;     // Both baked axes; one becomes wall height after rotation.
    constexpr float kFloorTopY = -0.8f;   // Must match the floor's own worldOffsetY below.
    constexpr float kWallCenterY = kFloorTopY + kWallSpan * 0.5f; // Wall base sits exactly on the floor.

    // WALL A (slot 12) — red, vertical plane fixed at X = -1.8, spanning Z in [-1.5, 1.5].
    // Baked flat then stood up about the X axis... no -- about the Z axis (see
    // UpdateEntityRotations(): RotateZ maps the baked local-X extent onto world Y, leaving X and Z
    // pinned), which is why the wall's OWN world-space X position comes from `slot.x` below.
    {
      maths::vec2 slot = {-1.8f, 0.0f};
      GeneratePlane(kWallSpan, kWallSpan, m_EntityData[kWallEntityIndexA].meshID, slot,
                    runningVertexOffset, runningIndexOffset, kWallCenterY,
                    config::FLOOR_VERTEX_SPACING);
    }

    // WALL B (slot 13) — green, vertical plane fixed at Z = -1.8, spanning X in [-1.5, 1.5].
    // Stood up about the X axis instead (RotateX maps the baked local-Z extent onto world Y,
    // leaving X and Z pinned) -- perpendicular to Wall A, closing the corner.
    {
      maths::vec2 slot = {0.0f, -1.8f};
      GeneratePlane(kWallSpan, kWallSpan, m_EntityData[kWallEntityIndexB].meshID, slot,
                    runningVertexOffset, runningIndexOffset, kWallCenterY,
                    config::FLOOR_VERTEX_SPACING);
    }
  }

  // -------------------------------------------------------------------------
  // TERRAIN HEIGHTFIELD (slot 14) — Phase 7b (UE5.8 parity roadmap): 300m x 300m procedural
  // terrain replacing what used to be a flat floor plane, at the same world footprint -- see
  // GenerateTerrain()'s own comment and terrain_noise.glsl's kTerrainAmplitude comment for why its
  // height variation stays small (this gallery floats every zone primitive at a fixed clearance
  // above the ground rather than deriving per-entity placement from it).
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = {0.0f, 0.0f}; // centered at the world origin
    // Deliberately coarser than FLOOR_VERTEX_SPACING (1.0): unlike the flat floor it replaces
    // (near-zero curvature everywhere, so ClusterDAG's simplifier folds its 90000 vertices into a
    // handful of DAG nodes almost for free), a genuinely undulating heightfield has real curvature
    // at every vertex, forcing real partitioning into hundreds/thousands of small clusters. At
    // FLOOR_VERTEX_SPACING (90000 vertices) this pushed the scene's total concurrent DAG-build
    // load (LoadingManager's parallel per-mesh workers, see core::LoadingManager) far past what
    // every other primitive's mesh needs, and reproducibly crashed partway through cold-cache
    // rebuild during Phase 7b's own verification -- a pre-existing concurrency issue in the DAG-
    // build pipeline that was never triggered before because no primitive's mesh had ever
    // combined this vertex count with this much genuine local curvature. Rather than chase that
    // pipeline bug (out of Phase 7b's own scope -- see the geometry-only feature this phase is
    // meant to add), staying at a lower vertex/cluster budget comparable to every other primitive's
    // own mesh sidesteps it entirely: 4x coarser (75x75 = 5625 vertices) still reads as a genuine
    // rolling backdrop at this terrain's own small kTerrainAmplitude.
    constexpr float kTerrainVertexSpacing = 4.0f;
    GenerateTerrain(300.0f, 300.0f, m_EntityData[kFloorEntityIndex].meshID, slot,
                  runningVertexOffset, runningIndexOffset, -0.8f,
                  kTerrainVertexSpacing);
  }

  // -------------------------------------------------------------------------
  // WATER (slot 15) -- Phase 7c (UE5.8 parity roadmap, water/erosion): 16th entity, a flat plane
  // sized to the showcase zone-grid's own footprint (not the full 300x300 terrain -- the depth
  // test against the already-rasterized terrain naturally clips this flat quad to whatever low-
  // lying basin actually falls within it, so an oversized water plane would just waste fragment-
  // shader work on pixels the terrain always occludes). 2x2 segments (a single quad): wave
  // perturbation is per-fragment (WaterForward.frag), not per-vertex, so more segments would add
  // nothing -- see GenerateWaterPlane()'s own comment.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = {0.0f, 0.0f}; // centered at the world origin, same as the terrain itself
    constexpr float kWaterPlaneSpan = 24.0f; // Mirrors the zone-grid's own ~16-unit extent with margin.
    constexpr float kWaterLevel = -1.0f; // Mirrors water_params.glsl's kWaterLevel -- keep in sync.
    GenerateWaterPlane(kWaterPlaneSpan, kWaterPlaneSpan, 2u, 2u, m_EntityData[kWaterEntityIndex].meshID, slot,
                       kWaterLevel, runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // RUNTIME WORLD PARTITION STREAMING POOL (entities kStreamingSlotBase..kTotalEntityCount-1) --
  // see kStreamingUnitCount's own header comment for why streamed-in content is drawn from a small
  // fixed set of pre-baked shapes (live per-cell Nanite cluster DAG builds are not feasible on a
  // streaming budget) rather than unique per-cell geometry.
  //
  // Every unit's 2 slots are baked at their OWN distinct, widely-separated parking position (never
  // the origin, and never shared with another slot) -- geom_autosmooth.comp's post-pass welds
  // normals across ANY vertices within a small world-space epsilon regardless of meshID (see that
  // shader's own comment), so baking multiple archetypes on top of each other at (0,0,0) would
  // silently corrupt their normals. world::WorldCellStreamingLoader's runtime translation (see
  // struct_custo.glsl's EntityTransform comment) is an ADDITIVE offset on top of whatever position
  // a mesh is baked at -- exactly the same rotation-pivot math every other entity already uses, see
  // UpdateEntityRotations()'s streaming-pool loop below -- so parking slots stay fully compatible
  // with being moved to an arbitrary cell position at runtime.
  // -------------------------------------------------------------------------
  {
    constexpr float kParkSpacing = 3.0f;
    constexpr float kParkBaseX = -400.0f; // Far outside the showcase gallery (a few meters around the origin) and the streaming demo world itself (see BakeDemoWorld.cpp's kWorldCenterX).
    constexpr float kParkBaseZ = -400.0f;

    for (uint32_t unit = 0; unit < kStreamingUnitCount; ++unit) {
      uint32_t shape = unit % kStreamingArchetypeShapeCount;

      uint32_t coarseIdx = StreamingUnitCoarseSlot(unit);
      maths::vec2 coarseSlot{ kParkBaseX + static_cast<float>(coarseIdx) * kParkSpacing, kParkBaseZ };
      GenerateBox(0.6f, 0.6f, 0.6f, m_EntityData[coarseIdx].meshID, coarseSlot, runningVertexOffset, runningIndexOffset);

      uint32_t fineIdx = StreamingUnitFineSlot(unit);
      maths::vec2 fineSlot{ kParkBaseX + static_cast<float>(fineIdx) * kParkSpacing, kParkBaseZ };
      switch (shape) {
        case 0: { // Rock: icosphere.
          uint32_t baseFaceCount = 20u, vertsPerFace = 0u;
          GenerateIcosphere(0.5f, false, false, true, m_EntityData[fineIdx].meshID, fineSlot,
                             runningVertexOffset, runningIndexOffset, baseFaceCount, vertsPerFace);
          break;
        }
        case 1: // Bush: UV sphere.
          GenerateSphere(0.5f, m_EntityData[fineIdx].meshID, fineSlot, runningVertexOffset, runningIndexOffset);
          break;
        case 2: // Tree: capsule (trunk + canopy silhouette stand-in).
          GenerateCapsule(0.25f, 0.8f, m_EntityData[fineIdx].meshID, fineSlot, runningVertexOffset, runningIndexOffset);
          break;
        default: // Debris: torus (irregular scrap silhouette stand-in).
          GenerateTorus(0.35f, 0.12f, 0.0f, 0.0f, m_EntityData[fineIdx].meshID, fineSlot, runningVertexOffset, runningIndexOffset);
          break;
      }
    }
  }

  m_TotalVertexCount = runningVertexOffset;
  m_TotalIndexCount = runningIndexOffset;

  // -------------------------------------------------------------------------
  // AUTOSMOOTH POST-PASS (autosmooth at 45.0 degrees)
  // -------------------------------------------------------------------------
  if (m_AutosmoothPipeline != VK_NULL_HANDLE) {
    struct AutosmoothParams {
      uint32_t totalVertexCount;
      uint32_t totalIndexCount;
    } params;
    params.totalVertexCount = m_TotalVertexCount;
    params.totalIndexCount = m_TotalIndexCount;

    constexpr uint32_t kLocalSizeX = 64u;
    uint32_t groupCount = (m_TotalVertexCount + kLocalSizeX - 1u) / kLocalSizeX;

    DispatchGeometryCompute(m_AutosmoothPipeline, m_ComputePipelineLayout,
                            &params, sizeof(params), nullptr, 0,
                            groupCount, 1, 1);
  }

  const VkDeviceSize vertexBytesUsed =
      static_cast<VkDeviceSize>(m_TotalVertexCount) * sizeof(renderer::Vertex);
  const VkDeviceSize indexBytesUsed =
      static_cast<VkDeviceSize>(m_TotalIndexCount) * sizeof(uint32_t);
  if (vertexBytesUsed > m_VertexBufferBytes ||
      indexBytesUsed > m_IndexBufferBytes) {
    LOG_CRITICAL(std::format("[GenerateGeometry] Procedural geometry "
                             "OVERFLOWED its fixed-size SSBOs: "
                             "vertices used {}/{} bytes, indices used {}/{} "
                             "bytes. GPU writes past this point are "
                             "undefined behavior (silent corruption of "
                             "adjacent buffers). Increase VERTEX_BUFFER_BYTES/"
                             "INDEX_BUFFER_BYTES in the chosen configuration profile.",
                             vertexBytesUsed, m_VertexBufferBytes,
                             indexBytesUsed, m_IndexBufferBytes));
    throw std::runtime_error(
        "Procedural geometry buffers overflowed -- see log for exact sizes.");
  }

  LOG_INFO(std::format("[GenerateGeometry] All 12 primitives + 2 Lumen walls + floor generated: "
                       "totalVertexCount={} totalIndexCount={} "
                       "(buffers hold {} verts / {} indices max)",
                       runningVertexOffset, runningIndexOffset,
                       m_VertexBufferBytes / sizeof(renderer::Vertex),
                       m_IndexBufferBytes / sizeof(uint32_t)));
}

void VulkanContext::UpdateEntityRotations(float timeSeconds) {
  // Distinct per-axis angular speeds (radians/sec) and a per-entity phase
  // offset so the 12 primitives tumble out of sync with each other rather than
  // spinning in lockstep. The floor plane and the 2 Lumen-corner walls remain
  // completely static (the walls at a fixed 90-degree stand-up rotation, see below).
  constexpr float kSpeedX = 0.7f;
  constexpr float kSpeedY = 1.1f;
  constexpr float kSpeedZ = 0.5f;
  constexpr float kPhaseStep = 0.6f; // radians of extra offset per meshID

  // Must exactly match GenerateGeometry()'s Lumen-corner wall block (kWallSpan/kFloorTopY there):
  // the bake height and this pivot's Y both need to equal kWallCenterY, or the rotation shears the
  // wall sideways instead of standing it up in place -- see that block's own comment for the math.
  constexpr float kWallSpan = 3.0f;
  constexpr float kFloorTopY = -0.8f;
  constexpr float kWallCenterY = kFloorTopY + kWallSpan * 0.5f;

  std::array<EntityTransform, kTotalEntityCount> transforms{};

  for (uint32_t meshID = 0; meshID < kEntityCount; ++meshID) {
    EntityTransform &xform = transforms[meshID];
    if (meshID == kFloorEntityIndex) {
      // Floor plane: static at Y = kFloorTopY.
      xform.rotation = maths::mat4{};
      xform.centerX = 0.0f;
      xform.centerY = kFloorTopY;
      xform.centerZ = 0.0f;
      xform._pad0 = 0.0f;
    } else if (meshID == kWallEntityIndexA) {
      // Wall A (red): fixed 90-degree rotation about Z stands the flat-baked plane up so its
      // baked local-X extent becomes world Y, while X/Z stay pinned at the wall's own slot (-1.8,
      // 0.0) -- see GenerateGeometry()'s wall block for the full pivot derivation.
      constexpr float kHalfPi = 1.5707963267948966f;
      xform.rotation = maths::mat4::RotateZ(kHalfPi);
      xform.centerX = -1.8f;
      xform.centerY = kWallCenterY;
      xform.centerZ = 0.0f;
      xform._pad0 = 0.0f;
    } else if (meshID == kWallEntityIndexB) {
      // Wall B (green): fixed 90-degree rotation about X stands the flat-baked plane up so its
      // baked local-Z extent becomes world Y, while X/Z stay pinned at the wall's own slot (0.0,
      // -1.8) -- perpendicular to Wall A, closing the Lumen showcase corner.
      constexpr float kHalfPi = 1.5707963267948966f;
      xform.rotation = maths::mat4::RotateX(kHalfPi);
      xform.centerX = 0.0f;
      xform.centerY = kWallCenterY;
      xform.centerZ = -1.8f;
      xform._pad0 = 0.0f;
    } else if (meshID == kWaterEntityIndex) {
      // Phase 7c (UE5.8 parity roadmap, water/erosion): water plane, static at Y = kWaterLevel
      // (must match GenerateGeometry()'s own water block) -- wave motion is per-fragment shading
      // only (WaterForward.frag), not a vertex-level rotation/animation, see that shader's own
      // header comment.
      constexpr float kWaterLevel = -1.0f;
      xform.rotation = maths::mat4{};
      xform.centerX = 0.0f;
      xform.centerY = kWaterLevel;
      xform.centerZ = 0.0f;
      xform._pad0 = 0.0f;
    } else {
      float phase = static_cast<float>(meshID) * kPhaseStep;

      // config::ENTITY_SELF_ROTATION_ENABLED kill-switch (see its own comment in EngineConfig.h):
      // identity rotation when disabled -- every consumer (cluster_entity_transform.glsl's
      // helpers, ClusterRaster.vert's per-vertex transform) already degrades correctly to a no-op
      // for an identity matrix, so no shader-side change is needed to fully disable rotation.
      maths::mat4 rotation = config::ENTITY_SELF_ROTATION_ENABLED
          ? maths::mat4::RotateY(timeSeconds * kSpeedY + phase) *
            maths::mat4::RotateX(timeSeconds * kSpeedX + phase) *
            maths::mat4::RotateZ(timeSeconds * kSpeedZ + phase)
          : maths::mat4{};

      // Every primitive's baked world-space center coincides exactly with its
      // grid slot position at Y=0: each geom_*.comp shader either generates a
      // shape already centered on its own local origin
      // (icosphere/box/sphere/torus/plane/torusKnot/chamferBox), or one that
      // spans y=[0,height] recentered via worldOffsetY=-height/2 in
      // GenerateGeometry() (cone/tube/cylinder/capsule/pyramid) — both cases land
      // the shape's true vertical midpoint at Y=0.
      maths::vec2 slot = GridSlot(static_cast<int>(meshID));

      xform.rotation = rotation;
      xform.centerX = slot.x;
      xform.centerY = 0.0f;
      xform.centerZ = slot.y;
      xform._pad0 = 0.0f;
    }

    // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): CPU-readable mirror of
    // the SAME values just computed above for GPU upload -- zero extra computation, see
    // GetEntityTransformsCPU()'s own comment on why this exists.
    m_EntityTransformsCPU[meshID] = core::EntityTransformCPU{
        xform.rotation, maths::vec3{xform.centerX, xform.centerY, xform.centerZ}};
  }

  // --- Runtime World Partition streaming pool: static (no self-rotation), positioned entirely by
  // m_StreamingUnitTranslation (see SetStreamingUnitState()) -- center/rotation stay exactly as
  // baked (identity rotation, center == the slot's own parking position, see GenerateGeometry()'s
  // streaming block), so worldPos == translation + bakedPos (this class's own EntityTransform
  // comment). Both slots of a unit always share the same translation. ---
  for (uint32_t unit = 0; unit < kStreamingUnitCount; ++unit) {
    const maths::vec3 &t = m_StreamingUnitTranslation[unit];
    for (uint32_t slotInUnit = 0; slotInUnit < kStreamingSlotsPerUnit; ++slotInUnit) {
      uint32_t i = kStreamingSlotBase + unit * kStreamingSlotsPerUnit + slotInUnit;
      EntityTransform &xform = transforms[i];
      xform.rotation = maths::mat4{};
      xform.centerX = 0.0f;
      xform.centerY = 0.0f;
      xform.centerZ = 0.0f;
      xform._pad0 = 0.0f;
      xform.translationX = t.x;
      xform.translationY = t.y;
      xform.translationZ = t.z;
      xform._pad1 = 0.0f;
      m_EntityTransformsCPU[i] = core::EntityTransformCPU{ xform.rotation, maths::vec3{0.0f, 0.0f, 0.0f}, t };
    }
  }

  void *mapped = nullptr;
  if (vmaMapMemory(m_Allocator, m_EntityTransformAllocation, &mapped) !=
      VK_SUCCESS) {
    throw std::runtime_error(
        "Failed to map Entity Transform buffer for per-frame update!");
  }
  std::memcpy(mapped, transforms.data(),
              sizeof(EntityTransform) * kTotalEntityCount);
  vmaUnmapMemory(m_Allocator, m_EntityTransformAllocation);
}

void VulkanContext::SetStreamingUnitState(uint32_t unit, bool active, bool useFineVariant,
                                           const maths::vec3 &worldPos, uint32_t cellID) {
  assert(unit < kStreamingUnitCount);

  // Translation takes effect on the NEXT UpdateEntityRotations() call (once per frame, see that
  // function's own streaming-pool loop) -- no GPU touch needed here, unlike EntityData below,
  // because the whole transform buffer is already re-uploaded wholesale every frame regardless.
  m_StreamingUnitTranslation[unit] = active ? worldPos : maths::vec3{0.0f, 0.0f, 0.0f};

  uint32_t coarseIdx = StreamingUnitCoarseSlot(unit);
  uint32_t fineIdx = StreamingUnitFineSlot(unit);

  bool coarseActive = active && !useFineVariant;
  bool fineActive = active && useFineVariant;

  core::EntityData &coarse = m_EntityData[coarseIdx];
  core::SetFlag(coarse.flags, core::EntityFlags::StreamingInactive, !coarseActive);
  coarse.cellID = coarseActive ? cellID : 0u;

  core::EntityData &fine = m_EntityData[fineIdx];
  core::SetFlag(fine.flags, core::EntityFlags::StreamingInactive, !fineActive);
  fine.cellID = fineActive ? cellID : 0u;

  PatchStreamingUnitEntityData(unit);
}

void VulkanContext::PatchStreamingUnitEntityData(uint32_t unit) {
  uint32_t coarseIdx = StreamingUnitCoarseSlot(unit);
  // The 2 slots of a unit are always adjacent (see StreamingUnitCoarseSlot/StreamingUnitFineSlot),
  // so a single 2-element staging buffer + one vkCmdCopyBuffer region covers both.
  VkDeviceSize patchSize = sizeof(core::EntityData) * kStreamingSlotsPerUnit;
  VkDeviceSize dstOffset = sizeof(core::EntityData) * coarseIdx;

  VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  stagingInfo.size = patchSize;
  stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo stagingAllocInfo{};
  stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
  stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingAllocation = VK_NULL_HANDLE;
  VmaAllocationInfo stagingAllocResultInfo{};
  if (vmaCreateBuffer(m_Allocator, &stagingInfo, &stagingAllocInfo,
                      &stagingBuffer, &stagingAllocation,
                      &stagingAllocResultInfo) != VK_SUCCESS) {
    LOG_ERROR("[VulkanContext] Failed to allocate streaming-slot EntityData staging buffer!");
    return;
  }
  std::memcpy(stagingAllocResultInfo.pMappedData, &m_EntityData[coarseIdx],
              static_cast<size_t>(patchSize));

  renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = patchSize;
    vkCmdCopyBuffer(cmd, stagingBuffer, m_EntityBuffer, 1, &copyRegion);

    // Same transfer-write -> vertex/compute-shader-read visibility barrier as UploadEntityData().
    VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &memBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
  });

  vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
}

void VulkanContext::DebugReadbackGeometrySample(uint32_t vertsPerFace,
                                                uint32_t expectedIndexCount) {
  // Sample window: covers the tail of face 0 and the head of face 1, so the log
  // proves (or disproves) that the compute dispatch wrote data past the first
  // face.
  const uint32_t maxVertsInBuffer =
      static_cast<uint32_t>(m_VertexBufferBytes / sizeof(renderer::Vertex));
  const uint32_t sampleVertexCount =
      std::min<uint32_t>(vertsPerFace + 8u, maxVertsInBuffer);
  const uint32_t sampleIndexCount = std::min<uint32_t>(12u, expectedIndexCount);

  const VkDeviceSize vertexSampleBytes =
      static_cast<VkDeviceSize>(sampleVertexCount) * sizeof(renderer::Vertex);
  const VkDeviceSize indexSampleBytes =
      static_cast<VkDeviceSize>(sampleIndexCount) * sizeof(uint32_t);

  VkBuffer stagingVertexBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingVertexAlloc = VK_NULL_HANDLE;
  VkBufferCreateInfo stagingVertexInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  stagingVertexInfo.size = vertexSampleBytes;
  stagingVertexInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo stagingAllocInfo{.usage =
                                               VMA_MEMORY_USAGE_GPU_TO_CPU};
  if (vmaCreateBuffer(m_Allocator, &stagingVertexInfo, &stagingAllocInfo,
                      &stagingVertexBuffer, &stagingVertexAlloc,
                      nullptr) != VK_SUCCESS) {
    LOG_ERROR("[DebugReadback] Failed to allocate vertex staging buffer!");
    return;
  }

  VkBuffer stagingIndexBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingIndexAlloc = VK_NULL_HANDLE;
  VkBufferCreateInfo stagingIndexInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  stagingIndexInfo.size = indexSampleBytes;
  stagingIndexInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vmaCreateBuffer(m_Allocator, &stagingIndexInfo, &stagingAllocInfo,
                      &stagingIndexBuffer, &stagingIndexAlloc,
                      nullptr) != VK_SUCCESS) {
    LOG_ERROR("[DebugReadback] Failed to allocate index staging buffer!");
    vmaDestroyBuffer(m_Allocator, stagingVertexBuffer, stagingVertexAlloc);
    return;
  }

  renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
    // GenerateGeometry() already vkQueueWaitIdle'd after the compute dispatch
    // before calling this function, so the compute writes are complete and
    // visible: a plain copy is safe here.
    VkBufferCopy vertexCopyRegion{0, 0, vertexSampleBytes};
    vkCmdCopyBuffer(cmd, m_VertexBuffer, stagingVertexBuffer, 1,
                    &vertexCopyRegion);

    VkBufferCopy indexCopyRegion{0, 0, indexSampleBytes};
    vkCmdCopyBuffer(cmd, m_IndexBuffer, stagingIndexBuffer, 1, &indexCopyRegion);
  });

  // Log a handful of representative vertices: the first vertex of face 0, the
  // last vertex of face 0, and the first vertex of face 1 (proves writes
  // actually crossed a face boundary).
  void *mappedVerts = nullptr;
  if (vmaMapMemory(m_Allocator, stagingVertexAlloc, &mappedVerts) ==
      VK_SUCCESS) {
    const renderer::Vertex *verts =
        reinterpret_cast<const renderer::Vertex *>(mappedVerts);

    auto logVertex = [](const char *label, const renderer::Vertex &v) {
      float len =
          std::sqrt(v.position.x * v.position.x + v.position.y * v.position.y +
                    v.position.z * v.position.z);
      LOG_INFO(std::format("[DebugReadback] {} pos=({:.4f}, {:.4f}, {:.4f}) "
                           "|pos|={:.4f} meshID={} materialID={}",
                           label, v.position.x, v.position.y, v.position.z, len,
                           v.meshID, v.materialID));
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
  } else {
    LOG_ERROR(
        "[DebugReadback] Failed to map vertex staging buffer for readback!");
  }

  void *mappedIndices = nullptr;
  if (vmaMapMemory(m_Allocator, stagingIndexAlloc, &mappedIndices) ==
      VK_SUCCESS) {
    const uint32_t *idx = reinterpret_cast<const uint32_t *>(mappedIndices);
    std::string idxDump;
    for (uint32_t i = 0; i < sampleIndexCount; ++i) {
      idxDump += std::format("{} ", idx[i]);
    }
    LOG_INFO(std::format("[DebugReadback] First {} indices: {}",
                         sampleIndexCount, idxDump));
    vmaUnmapMemory(m_Allocator, stagingIndexAlloc);
  } else {
    LOG_ERROR(
        "[DebugReadback] Failed to map index staging buffer for readback!");
  }

  vmaDestroyBuffer(m_Allocator, stagingVertexBuffer, stagingVertexAlloc);
  vmaDestroyBuffer(m_Allocator, stagingIndexBuffer, stagingIndexAlloc);
}

void VulkanContext::Shutdown() {
  if (m_Device != VK_NULL_HANDLE) {
    LOG_INFO("[VulkanContext] Shutting down Vulkan context...");
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

  if (m_VisBufferClusterIDImageView != VK_NULL_HANDLE) {
    vkDestroyImageView(m_Device, m_VisBufferClusterIDImageView, nullptr);
    m_VisBufferClusterIDImageView = VK_NULL_HANDLE;
  }
  if (m_VisBufferClusterIDImage != VK_NULL_HANDLE) {
    vmaDestroyImage(m_Allocator, m_VisBufferClusterIDImage,
                    m_VisBufferClusterIDAllocation);
    m_VisBufferClusterIDImage = VK_NULL_HANDLE;
    m_VisBufferClusterIDAllocation = VK_NULL_HANDLE;
  }
  if (m_VisBufferTriangleIDImageView != VK_NULL_HANDLE) {
    vkDestroyImageView(m_Device, m_VisBufferTriangleIDImageView, nullptr);
    m_VisBufferTriangleIDImageView = VK_NULL_HANDLE;
  }
  if (m_VisBufferTriangleIDImage != VK_NULL_HANDLE) {
    vmaDestroyImage(m_Allocator, m_VisBufferTriangleIDImage,
                    m_VisBufferTriangleIDAllocation);
    m_VisBufferTriangleIDImage = VK_NULL_HANDLE;
    m_VisBufferTriangleIDAllocation = VK_NULL_HANDLE;
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
    vmaDestroyBuffer(m_Allocator, m_EntityTransformBuffer,
                     m_EntityTransformAllocation);
  }
  if (m_EntityBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_Allocator, m_EntityBuffer, m_EntityAllocation);
  }

  if (m_ImageAvailableSemaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_Device, m_ImageAvailableSemaphore, nullptr);
  }
  for (VkSemaphore semaphore : m_RenderFinishedSemaphores) {
    vkDestroySemaphore(m_Device, semaphore, nullptr);
  }
  m_RenderFinishedSemaphores.clear();
  if (m_TransferFinishedSemaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_Device, m_TransferFinishedSemaphore, nullptr);
    m_TransferFinishedSemaphore = VK_NULL_HANDLE;
  }

  for (VkPipeline pipeline :
       {m_ConePipeline, m_IcospherePipeline, m_PlanePipeline, m_SpherePipeline,
        m_TorusPipeline, m_TubePipeline, m_CapsulePipeline, m_CylinderPipeline,
        m_PyramidPipeline, m_TorusKnotPipeline, m_ChamferBoxPipeline,
        m_TerrainPipeline, m_AutosmoothPipeline}) {
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
  if (m_TransferCommandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(m_Device, m_TransferCommandPool, nullptr);
    m_TransferCommandPool = VK_NULL_HANDLE;
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

#ifndef NDEBUG
  if (m_EnableValidationLayers && m_DebugMessenger != VK_NULL_HANDLE) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        m_Instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
      func(m_Instance, m_DebugMessenger, nullptr);
    }
  }
#endif // NDEBUG

  if (m_Instance != VK_NULL_HANDLE) {
    vkDestroyInstance(m_Instance, nullptr);
  }
}
