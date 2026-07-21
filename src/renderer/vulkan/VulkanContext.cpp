#include "renderer/vulkan/VulkanContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "core/EngineConfig.h" // Centralized engine configurations (config::WINDOW_WIDTH, etc.)
#include "core/EntityData.h"
#include "core/InstanceRegistry.h"
#include "core/Logger.h"
#include "core/ResourcePath.h"
#include "renderer/MaterialParameterTable.h"
#include "renderer/RenderTypes.h"
#include "renderer/passes/ProceduralTreePass.h"
#include "renderer/vulkan/RayTracingFunctions.h"
#include "renderer/vulkan/VulkanUtils.h"
#include "core/debug/ValidationMessageSink.h"
// Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): geometry::SimplifiableMesh/
// ComputeFaceAccumulatedNormals -- BakeHlodProxyIntoSlot() reuses the SAME normal-recomputation
// helper the offline HLOD bake chain documents (world::CellHlodVertex's own header comment) rather
// than inventing a second implementation.
#include "animation/SkeletalAnimator.h" // animation::SkeletalAnimator::kBoneCount/kSegmentLength -- see GenerateCreature()'s own comment.
#include "geometry/ClusterFormat.h" // geometry::ClusterVertexSkin -- see GetVertexSkinBuffer()'s own comment.
#include "geometry/MeshSimplifier.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>
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

// Minimal-scene mode (2026-07-21): the default scene is now just the UV sphere (slot 4,
// GenerateGeometry()'s own SPHERE block) and a flat ground plane (kFloorEntityIndex, generated via
// GeneratePlane instead of GenerateTerrain), lit by the existing always-on sun/sky -- nothing else
// (the other 11 showcase primitives, the creature, the Lumen-corner walls, water, and the 10-
// species tree grove) is generated. Shared between BuildEntityData() (floor materialID) and
// GenerateGeometry() (which meshes actually get dispatched) so the two can never disagree about
// what the floor is. Every entity this skips keeps its own allocated slot (kEntityCount/
// kTotalEntityCount and every index derived from them are UNCHANGED) but simply never has any
// geometry generated for it -- an already-proven-safe path throughout this pipeline (geometry::
// BuildClusterDAG's own "No triangles matched..." early-return, geometry::
// VirtualGeometryCacheTest.cpp's "produced zero clusters; skipping" continue, and renderer::
// GlobalSDFPass's fallback-table-driven bake all already tolerate a zero-geometry entity
// gracefully), not a new/untested code path -- so the rich showcase scene's own code stays fully
// intact and easy to bring back (flip this back to false) without re-deriving any of it.
constexpr bool kMinimalSceneMode = true;

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

// Rivers/waterfalls feature: geom_river.comp's own Params UBO -- deliberately NOT byte-identical
// to PlaneParams (unlike GenerateTerrain's reuse of it): this shape has no width/length/
// worldOffsetX/Z of its own (the path lives entirely in river_spline.glsl, shared by both this
// dispatch and terrain_noise.glsl's channel carve -- see that file's own header comment), only a
// generation-grid resolution and the ribbon half-width. Must match geom_river.comp's own Params
// block exactly (std140).
struct RiverParams {
  uint32_t segmentsAlong;
  uint32_t segmentsAcross;
  float halfWidth;
  float _padUnused;
  uint32_t meshID;
  float materialID;
  uint32_t vertexOffset;
  uint32_t indexOffset;
  float worldOffsetX;
  float worldOffsetY;
  float worldOffsetZ;
};

// Skeletal-animation feature: geom_creature.comp's own Params UBO -- must match that shader's
// Params block exactly (std140). boneCount/segmentLength MUST equal animation::SkeletalAnimator::
// kBoneCount/kSegmentLength exactly (see GenerateCreature()'s own comment) so the generated
// bind-pose mesh matches the CPU-side skeleton's own analytic bind pose bit-for-bit.
struct CreatureParams {
  uint32_t boneCount;
  uint32_t sidesPerRing;
  uint32_t ringsPerSegment;
  float segmentLength;
  float radiusMin;
  float radiusMax;
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

  // E1 (loading-time optimization): must run right after the device exists (vkCreatePipelineCache
  // needs m_Device) and before ANY renderer:: pass is constructed (every pass reads the resulting
  // handle via VulkanPipeline::GetPipelineCache() at its own pipeline-creation call site) -- see
  // CreatePipelineCache()'s own header comment for the full load/validate contract.
  CreatePipelineCache();

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

  // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): load the SAME world_data/cellmanifest.bin
  // main.cpp's own world::CellManifest instance loads (see world::kDefaultManifestPath's own
  // comment for why both sides share one literal) BEFORE BuildEntityData()/GenerateGeometry() run,
  // so both can key each dedicated streaming unit's materialID/shape (BuildEntityData()) and actual
  // baked geometry (GenerateGeometry()) off the SAME real per-cell archetype data for that SAME
  // unit. Loaded once here (not lazily inside either) so a missing/unreadable file degrades exactly
  // once, gracefully -- matching this codebase's "streaming is additive, not load-bearing"
  // convention (see world::CellManifest::Load()'s own header comment and main.cpp's own identical
  // fallback log): every dedicated-unit lookup below then simply finds nothing and every streaming
  // unit falls back to the pre-Phase-5 shared 4-archetype rotation, exactly as before this feature.
  if (m_CellManifest.Load(core::ResolveExeRelativePath(world::kDefaultManifestPath))) {
    LOG_INFO(std::format("[VulkanContext] Loaded {} authored cells from '{}' for HLOD proxy bake-in "
                         "(kStreamingUnitCount={}).", m_CellManifest.RecordCount(),
                         world::kDefaultManifestPath, kStreamingUnitCount));
  } else {
    LOG_WARNING(std::format("[VulkanContext] '{}' not found or unreadable -- HLOD proxy bake-in "
                            "skipped; every streaming unit falls back to the shared 4-archetype "
                            "rotation (run WorldPartitionBakeTool.exe once to enable real per-cell "
                            "geometry).", world::kDefaultManifestPath));
  }

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
  // DebugReadbackGeometrySample(). TRANSFER_DST_BIT (Phase 5, Streaming & Monde roadmap, Part 2,
  // Gap 3) is required by BakeHlodProxyIntoSlot()'s own vkCmdCopyBuffer, which writes a real,
  // offline-baked HLOD proxy mesh straight into this SSBO from a staging buffer -- every other
  // primitive in this buffer is instead written by a compute shader's storage-buffer WRITE (no
  // TRANSFER_DST_BIT needed for that path), so this bit was never required until this feature.
  vInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo vAlloc{.usage = VMA_MEMORY_USAGE_GPU_ONLY};
  if (vmaCreateBuffer(m_Allocator, &vInfo, &vAlloc, &m_VertexBuffer,
                      &m_VertexAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate vertex SSBO!");
  }
  LOG_INFO(std::format("[VulkanContext] Allocated vertex SSBO: size={} MB.", m_VertexBufferBytes / (1024 * 1024)));

  VkBufferCreateInfo iInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  iInfo.size = m_IndexBufferBytes;
  // TRANSFER_SRC_BIT is required for the DEBUG readback (vkCmdCopyBuffer) in
  // DebugReadbackGeometrySample(). TRANSFER_DST_BIT -- see vInfo.usage's own comment above, same
  // reasoning applies to the index SSBO.
  iInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vmaCreateBuffer(m_Allocator, &iInfo, &vAlloc, &m_IndexBuffer,
                      &m_IndexAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate index SSBO!");
  }
  LOG_INFO(std::format("[VulkanContext] Allocated index SSBO: size={} MB.", m_IndexBufferBytes / (1024 * 1024)));

  // Skeletal-animation feature -- see GetVertexSkinBuffer()'s own comment: index-aligned 1:1 with
  // m_VertexBuffer (same element count), so its byte size is scaled off the exact same vertex
  // capacity, just with geometry::ClusterVertexSkin (8 bytes/vertex) instead of renderer::Vertex
  // (48 bytes/vertex) per slot. No TRANSFER_DST_BIT (unlike m_VertexBuffer/m_IndexBuffer): this
  // buffer is only ever written by geom_creature.comp's compute dispatch, never by a staging-buffer
  // upload. TRANSFER_SRC_BIT is required for geometry::RunVirtualGeometryCacheTest's own readback.
  VkDeviceSize vertexSkinBufferBytes =
      (m_VertexBufferBytes / sizeof(renderer::Vertex)) * sizeof(geometry::ClusterVertexSkin);
  VkBufferCreateInfo vsInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  vsInfo.size = vertexSkinBufferBytes;
  vsInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (vmaCreateBuffer(m_Allocator, &vsInfo, &vAlloc, &m_VertexSkinBuffer,
                      &m_VertexSkinAllocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate vertex-skin SSBO!");
  }
  LOG_INFO(std::format("[VulkanContext] Allocated vertex-skin SSBO: size={} MB.", vertexSkinBufferBytes / (1024 * 1024)));

  VkBufferCreateInfo pInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  pInfo.size = 256;
  // TRANSFER_DST_BIT added for the startup-batching fix: DispatchGeometryCompute() now writes each
  // dispatch's Params block via vkCmdUpdateBuffer (recorded into a shared command buffer alongside
  // many other dispatches) instead of a host-mapped memcpy -- see that function's own comment.
  pInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
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

  // Dedicated async-compute queue (Phase 2, Lumen advanced roadmap -- UE 5.8 RHI parity): mirrors
  // the dedicated-transfer-queue search immediately above exactly, just with a different family
  // predicate -- a family advertising COMPUTE_BIT but NOT GRAPHICS_BIT is this GPU's separate
  // "async compute" hardware queue (an ACE-style engine on AMD, or a genuinely independent compute
  // queue on NVIDIA/Intel where one exists) that can execute concurrently with the graphics queue's
  // own submission instead of merely time-slicing the same combined graphics+compute queue. Cannot
  // collide with m_TransferQueueFamilyIndex found above: that search explicitly REQUIRES the
  // family to lack COMPUTE_BIT, while this one explicitly REQUIRES it -- so the two predicates are
  // mutually exclusive by construction, no distinctness check needed. Not every GPU exposes one
  // (integrated GPUs / some vendors expose only the combined graphics+compute family), so this is a
  // graceful runtime fallback exactly like the transfer queue's own -- see
  // GetAsyncComputeQueue()'s own header comment.
  m_AsyncComputeQueueFamilyIndex = m_GraphicsQueueFamilyIndex;
  m_HasDedicatedAsyncComputeQueue = false;
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      m_AsyncComputeQueueFamilyIndex = i;
      m_HasDedicatedAsyncComputeQueue = true;
      break;
    }
  }

  float queuePriority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  queueCreateInfos.reserve(3); // Fixed upfront: emplace_back below must never trigger a reallocation
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

  if (m_HasDedicatedAsyncComputeQueue) {
    VkDeviceQueueCreateInfo asyncComputeQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    asyncComputeQueueCreateInfo.queueFamilyIndex = m_AsyncComputeQueueFamilyIndex;
    asyncComputeQueueCreateInfo.queueCount = 1;
    asyncComputeQueueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(asyncComputeQueueCreateInfo);
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
  // vertexPipelineStoresAndAtomics (particle system Subtask 4): ParticleRender.vert -- a vertex
  // shader -- includes ParticleCommon.glsl and reads (never writes) its writable
  // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER `ParticleBuffer`/`SortedPairsBuffer` blocks; the SPIR-V still
  // declares them writable (the shared include has no readonly variant, since
  // ParticleSimulation.comp genuinely writes the same block). Per the Vulkan spec, ANY writable
  // storage buffer/image/texel-buffer variable in the vertex/tessellation/geometry stages requires
  // this feature enabled, or vkCreateGraphicsPipelines fails validation
  // (VUID-RuntimeSpirv-NonWritable-06341) -- the vertex-stage sibling of fragmentStoresAndAtomics
  // immediately above, same near-universal desktop-GPU support, enabled unconditionally here for
  // the same reason.
  deviceFeatures2.features.vertexPipelineStoresAndAtomics = VK_TRUE;
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
  // tessellationShader (Phase 7a, UE5.8 parity roadmap; generalized to multiple entities by the
  // Nanite Tessellation generalization): core Vulkan 1.0 feature bit, no device extension required
  // -- gates VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT/_EVALUATION_BIT (renderer::TessellationPass,
  // every core::EntityFlags::IsTessellated entity's own screen-space-adaptive displacement-mapped
  // pipeline). A near-universally-supported core feature on desktop GPUs (same rigor as
  // geometryShader/fragmentStoresAndAtomics/multiDrawIndirect above), enabled unconditionally here.
  deviceFeatures2.features.tessellationShader = VK_TRUE;
  // fillModeNonSolid (UE5.8 rendering-parity gap G2, vegetation scatter): gates
  // VK_POLYGON_MODE_LINE, used by renderer::VegetationScatterPass' Debug-only wireframe/bounds
  // visualization pipeline (that pipeline itself is #ifndef NDEBUG, but the feature bit is a core
  // Vulkan 1.0 capability with no runtime cost when unused, so it is enabled unconditionally here
  // -- same near-universally-supported-desktop-feature rigor as tessellationShader/independentBlend
  // above -- rather than special-casing device creation per build type).
  deviceFeatures2.features.fillModeNonSolid = VK_TRUE;

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

  // Same fallback-aliasing idiom as the transfer queue above: when no dedicated async-compute
  // family was found, m_AsyncComputeQueueFamilyIndex already equals m_GraphicsQueueFamilyIndex (set
  // above), so this just re-fetches the SAME VkQueue handle as m_GraphicsQueue -- harmless, and
  // keeps every consumer of GetAsyncComputeQueue() ignorant of the fallback (same convention
  // GetTransferQueue() already establishes).
  vkGetDeviceQueue(m_Device, m_AsyncComputeQueueFamilyIndex, 0, &m_AsyncComputeQueue);
  LOG_INFO(std::format("[VulkanContext] Async-compute queue: {} (family {})",
      m_HasDedicatedAsyncComputeQueue ? "dedicated async-compute queue" : "falling back to the graphics queue",
      m_AsyncComputeQueueFamilyIndex));

  // See RayTracingFunctions.h's own comment: VK_KHR_acceleration_structure /
  // VK_KHR_ray_tracing_pipeline's entry points are not re-exported by this SDK's loader import
  // library, so they must be resolved via vkGetDeviceProcAddr right here, once, immediately after
  // the device (which just enabled both extensions above) comes up.
  renderer::LoadRayTracingFunctions(m_Device, renderer::g_RTFunctions);
}

void VulkanContext::CreatePipelineCache() {
  // E1 (loading-time optimization): "pipeline.cache" persists the driver's own compiled-pipeline
  // blob across runs. Every VkPipeline in this codebase (~58 vkCreateComputePipelines/
  // vkCreateGraphicsPipelines/vkCreateRayTracingPipelinesKHR call sites across ~35 files) is
  // routed through the handle created here via VulkanPipeline::GetPipelineCache() -- see that
  // static accessor's own header comment. Resolved exe-relative (core::ResolveExeRelativePath),
  // same convention as scene.cache/gpu_profile.cfg.
  const std::filesystem::path cachePath = core::ResolveExeRelativePath("pipeline.cache");

  VkPhysicalDeviceProperties deviceProperties{};
  vkGetPhysicalDeviceProperties(m_PhysicalDevice, &deviceProperties);

  std::vector<char> cacheBlob;
  bool blobValid = false;

  std::ifstream inFile(cachePath, std::ios::binary | std::ios::ate);
  if (inFile.is_open()) {
    const std::streamsize fileSize = inFile.tellg();
    if (fileSize >= static_cast<std::streamsize>(sizeof(VkPipelineCacheHeaderVersionOne))) {
      cacheBlob.resize(static_cast<size_t>(fileSize));
      inFile.seekg(0);
      inFile.read(cacheBlob.data(), fileSize);
      if (inFile.good() || inFile.eof()) {
        // Validate the VkPipelineCacheHeaderVersionOne prefix ourselves (rather than only relying
        // on vkCreatePipelineCache to reject/ignore a non-matching blob -- the spec permits that
        // fallback but never requires it) so a mismatch can be logged with the SPECIFIC reason
        // (stale GPU/driver vs. truncated/corrupt file) instead of failing silently. This prefix is
        // the one portable part of an otherwise fully driver-defined, non-portable cache blob.
        VkPipelineCacheHeaderVersionOne header{};
        std::memcpy(&header, cacheBlob.data(), sizeof(header));
        const bool headerSizeOk = header.headerSize == sizeof(VkPipelineCacheHeaderVersionOne);
        const bool versionOk = header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
        const bool vendorOk = header.vendorID == deviceProperties.vendorID;
        const bool deviceOk = header.deviceID == deviceProperties.deviceID;
        const bool uuidOk = std::memcmp(header.pipelineCacheUUID, deviceProperties.pipelineCacheUUID, VK_UUID_SIZE) == 0;
        blobValid = headerSizeOk && versionOk && vendorOk && deviceOk && uuidOk;
        if (!blobValid) {
          LOG_INFO(std::format(
              "[VulkanContext] pipeline.cache header mismatch (headerSize {}=={}: {}, headerVersion "
              "{}=={}: {}, vendorID 0x{:X}==0x{:X}: {}, deviceID 0x{:X}==0x{:X}: {}, "
              "pipelineCacheUUID: {}) -- discarding, starting from an empty cache.",
              header.headerSize, sizeof(VkPipelineCacheHeaderVersionOne), headerSizeOk,
              static_cast<uint32_t>(header.headerVersion), static_cast<uint32_t>(VK_PIPELINE_CACHE_HEADER_VERSION_ONE), versionOk,
              header.vendorID, deviceProperties.vendorID, vendorOk,
              header.deviceID, deviceProperties.deviceID, deviceOk,
              uuidOk ? "match" : "mismatch"));
        }
      }
    } else if (fileSize >= 0) {
      LOG_INFO(std::format("[VulkanContext] pipeline.cache is too small to contain a valid header "
                           "({} bytes) -- discarding, starting from an empty cache.",
                           static_cast<long long>(fileSize)));
    }
  } else {
    LOG_INFO("[VulkanContext] No pipeline.cache found on disk (expected on first launch, or after a "
             "driver/GPU change) -- starting from an empty cache.");
  }

  VkPipelineCacheCreateInfo cacheInfo{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  if (blobValid) {
    cacheInfo.initialDataSize = cacheBlob.size();
    cacheInfo.pInitialData = cacheBlob.data();
  }

  VK_CHECK(vkCreatePipelineCache(m_Device, &cacheInfo, nullptr, &m_PipelineCache));

  m_PipelineCacheWasHit = blobValid;
  m_PipelineCacheLoadedBytes = blobValid ? cacheBlob.size() : 0;
  LOG_INFO(std::format("[VulkanContext] Pipeline cache {}: {} bytes loaded from '{}'.",
                       blobValid ? "HIT" : "MISS", m_PipelineCacheLoadedBytes, cachePath.string()));

  // Publish the handle for every renderer:: pass' own pipeline-creation call site -- see
  // VulkanPipeline::SetPipelineCache()'s own comment for why a static accessor is used here instead
  // of threading a VkPipelineCache parameter through every pass' constructor/Init() signature.
  VulkanPipeline::SetPipelineCache(m_PipelineCache);
}

void VulkanContext::SavePipelineCache() {
  if (m_PipelineCache == VK_NULL_HANDLE) {
    return;
  }

  size_t dataSize = 0;
  VK_CHECK(vkGetPipelineCacheData(m_Device, m_PipelineCache, &dataSize, nullptr));
  if (dataSize == 0) {
    LOG_INFO("[VulkanContext] SavePipelineCache: driver reports 0 bytes of pipeline cache data -- "
             "nothing to save.");
    return;
  }

  std::vector<char> data(dataSize);
  VK_CHECK(vkGetPipelineCacheData(m_Device, m_PipelineCache, &dataSize, data.data()));
  // vkGetPipelineCacheData permits the second call to report a SMALLER size than the first query
  // (e.g. a concurrent vkCreate*Pipelines* call shrinking the serialized form) -- trim to what was
  // actually written so the saved file's own size always matches its real content.
  data.resize(dataSize);

  const std::filesystem::path finalPath = core::ResolveExeRelativePath("pipeline.cache");
  const std::filesystem::path tempPath = core::ResolveExeRelativePath("pipeline.cache.tmp");

  std::ofstream outFile(tempPath, std::ios::binary | std::ios::trunc);
  if (!outFile.is_open()) {
    LOG_WARNING(std::format("[VulkanContext] SavePipelineCache: failed to open '{}' for writing.", tempPath.string()));
    return;
  }
  outFile.write(data.data(), static_cast<std::streamsize>(data.size()));
  outFile.close();
  if (!outFile) {
    LOG_WARNING(std::format("[VulkanContext] SavePipelineCache: write to '{}' failed.", tempPath.string()));
    std::error_code removeEc;
    std::filesystem::remove(tempPath, removeEc);
    return;
  }

  // Atomic on the same volume (both paths are the exe's own directory): a crash or power-loss
  // between the write above and this rename can never leave pipeline.cache itself half-written --
  // the OS either completes the rename or leaves the OLD pipeline.cache untouched, so a future
  // launch's CreatePipelineCache() only ever sees a fully-written file (or none at all).
  std::error_code renameEc;
  std::filesystem::rename(tempPath, finalPath, renameEc);
  if (renameEc) {
    LOG_WARNING(std::format("[VulkanContext] SavePipelineCache: rename '{}' -> '{}' failed: {}",
                            tempPath.string(), finalPath.string(), renameEc.message()));
    return;
  }

  LOG_INFO(std::format("[VulkanContext] Saved pipeline cache: {} bytes to '{}'.", data.size(), finalPath.string()));
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

  // Own pool for the async-compute queue's per-frame command buffer (Phase 2, Lumen advanced
  // roadmap -- see GetAsyncComputeCommandBuffer()'s own comment), same "always a distinct pool,
  // even when the family falls back to the graphics family" reasoning as m_TransferCommandPool
  // above.
  VkCommandPoolCreateInfo asyncComputePoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  asyncComputePoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  asyncComputePoolInfo.queueFamilyIndex = m_AsyncComputeQueueFamilyIndex;
  if (vkCreateCommandPool(m_Device, &asyncComputePoolInfo, nullptr, &m_AsyncComputeCommandPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create Async-Compute Command Pool!");
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
    throw std::runtime_error("Failed to allocate principal (cmdEarly) Command Buffer!");
  }

  // Phase 2 (Lumen advanced roadmap) fix: 2 more graphics-queue command buffers ("cmdMid"/
  // "cmdLate", same pool/family as m_CommandBuffer above) -- see GetCommandBufferMid()/
  // GetCommandBufferLate()'s own comment for why this frame's graphics work is now split 3 ways.
  VkCommandBufferAllocateInfo midAllocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  midAllocInfo.commandPool = m_CommandPool;
  midAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  midAllocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(m_Device, &midAllocInfo, &m_CommandBufferMid) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate cmdMid Command Buffer!");
  }

  VkCommandBufferAllocateInfo lateAllocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  lateAllocInfo.commandPool = m_CommandPool;
  lateAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  lateAllocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(m_Device, &lateAllocInfo, &m_CommandBufferLate) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate cmdLate Command Buffer!");
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

  VkCommandBufferAllocateInfo asyncComputeAllocInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  asyncComputeAllocInfo.commandPool = m_AsyncComputeCommandPool;
  asyncComputeAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  asyncComputeAllocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(m_Device, &asyncComputeAllocInfo, &m_AsyncComputeCommandBuffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate Async-Compute Command Buffer!");
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

  // Phase 2 (Lumen advanced roadmap): bidirectional async-compute semaphore pair -- unlike
  // m_TransferFinishedSemaphore's one-way (transfer -> graphics) dependency, the async-compute
  // queue's work this frame both STARTS after a graphics-queue release (m_AsyncComputeCanStartSemaphore)
  // AND must FINISH before the graphics queue consumes its results later the same frame
  // (m_AsyncComputeFinishedSemaphore) -- see GetAsyncComputeCanStartSemaphore()/
  // GetAsyncComputeFinishedSemaphore()'s own header comments. Both are plain binary semaphores,
  // safe under the same "signaled/waited within one frame only, never handed to vkQueuePresentKHR,
  // never reused before the frameFence wait retires both ends" reasoning
  // m_TransferFinishedSemaphore's own comment above documents -- always created, even when this GPU
  // exposes no dedicated async-compute family: the fallback path still submits (and waits on) them,
  // just with both ends on the same graphics queue (see main.cpp's per-frame submit sequence), so
  // no separate creation branch is needed here.
  if (vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &m_AsyncComputeCanStartSemaphore) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create async-compute-can-start semaphore!");
  }
  if (vkCreateSemaphore(m_Device, &semaphoreInfo, nullptr, &m_AsyncComputeFinishedSemaphore) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create async-compute-finished semaphore!");
  }
}

void VulkanContext::CreatePipelinesAndDescriptors() {
  // Terrain hydrology feature: run the GPU water & erosion bake FIRST -- the geometry descriptor
  // sets written below reference its output textures (binding 6), and GenerateGeometry() (called
  // right after this function) samples them from geom_terrain.comp / geom_water_surface.comp.
  // Blocking/synchronous by design, see TerrainHydrologySim::Init's own comment.
  m_TerrainHydrology.Init(m_Device, m_Allocator, m_CommandPool, m_GraphicsQueue);

  // maxSets 2: the main geometry set + the water-surface variant (binding 6 differs -- see
  // m_GeometryDescriptorSetWaterSurface's own header comment). Pool sizes doubled to match.
  VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
                                      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}};
  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets = 2;
  poolInfo.poolSizeCount = 3;
  poolInfo.pPoolSizes = poolSizes;
  if (vkCreateDescriptorPool(m_Device, &poolInfo, nullptr,
                             &m_GeometryDescriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Geometry Descriptor Pool!");
  }

  VkDescriptorSetLayoutBinding bindings[7] = {};
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

  // Skeletal-animation feature: vertex-skin SSBO (geometry::ClusterVertexSkin per vertex slot,
  // index-aligned with binding 0's Vertex SSBO) -- compute-stage-only WRITE, since only
  // geom_creature.comp's dispatch ever references binding 5 in its own GLSL source (every other
  // geom_*.comp shader shares this same descriptor set/layout but simply never declares this
  // binding, an already-established harmless pattern in this shared-descriptor-set design -- see
  // e.g. how the box-face generator never references binding 2's Params UBO either). Not read by
  // any vertex/fragment shader (unlike bindings 0/1/3/4): its only consumer is the CPU-side
  // readback geometry::RunVirtualGeometryCacheTest performs once at startup.
  bindings[5].binding = 5;
  bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[5].descriptorCount = 1;
  bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  // Terrain hydrology feature: the erosion bake's sampled height texture -- referenced only by
  // geom_terrain.comp (mesh height) and geom_water_surface.comp (water-surface height, via the
  // second set below), the same "declared in the shared layout, referenced by few shaders"
  // pattern binding 5 already establishes.
  bindings[6].binding = 6;
  bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bindings[6].descriptorCount = 1;
  bindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = 7;
  layoutInfo.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr,
                                  &m_GeometryLayout) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create Geometry Descriptor Layout!");
  }

  VkDescriptorSetLayout setLayouts[2] = {m_GeometryLayout, m_GeometryLayout};
  VkDescriptorSet geometrySets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkDescriptorSetAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool = m_GeometryDescriptorPool;
  allocInfo.descriptorSetCount = 2;
  allocInfo.pSetLayouts = setLayouts;
  if (vkAllocateDescriptorSets(m_Device, &allocInfo, geometrySets) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate Geometry Descriptor Sets!");
  }
  m_GeometryDescriptorSet = geometrySets[0];
  m_GeometryDescriptorSetWaterSurface = geometrySets[1];

  VkDescriptorBufferInfo vertInfo{m_VertexBuffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo indexInfo{m_IndexBuffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo paramsInfo{m_ParamsBuffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo entityTransformInfo{m_EntityTransformBuffer, 0,
                                             VK_WHOLE_SIZE};
  VkDescriptorBufferInfo entityDataInfo{m_EntityBuffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo vertexSkinInfo{m_VertexSkinBuffer, 0, VK_WHOLE_SIZE};
  // Binding 6 differs between the two sets: mesh height for every ordinary primitive dispatch
  // (only geom_terrain.comp actually reads it), water surface for geom_water_surface.comp's.
  VkDescriptorImageInfo hydroMeshHeightInfo{m_TerrainHydrology.GetLinearSampler(),
                                            m_TerrainHydrology.GetMeshHeightView(),
                                            VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo hydroWaterSurfaceInfo{m_TerrainHydrology.GetLinearSampler(),
                                              m_TerrainHydrology.GetWaterSurfaceView(),
                                              VK_IMAGE_LAYOUT_GENERAL};

  auto writeGeometrySet = [&](VkDescriptorSet set, const VkDescriptorImageInfo* hydroInfo) {
    VkWriteDescriptorSet writes[7] = {};
    for (int i = 0; i < 7; i++) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
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
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[5].pBufferInfo = &vertexSkinInfo;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].pImageInfo = hydroInfo;
    vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);
  };
  writeGeometrySet(m_GeometryDescriptorSet, &hydroMeshHeightInfo);
  writeGeometrySet(m_GeometryDescriptorSetWaterSurface, &hydroWaterSurfaceInfo);

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
      {"shaders/geom_river.comp.spv", &m_RiverPipeline},
      {"shaders/geom_water_surface.comp.spv", &m_WaterSurfacePipeline},
      {"shaders/geom_creature.comp.spv", &m_CreaturePipeline},
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

    if (vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo,
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

      if (vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo,
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
    VkCommandBuffer cmd,
    VkPipeline pipeline, VkPipelineLayout layout, const void *uboParamsData,
    size_t uboParamsSize, const void *pushConstantData, size_t pushConstantSize,
    uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ,
    VkDescriptorSet descriptorSetOverride) {
  // Every primitive except the box drives geom_*.comp through the shared Params UBO (binding = 2):
  // overwrite it with this dispatch's parameters before recording the dispatch that reads it.
  //
  // Startup-latency fix: this used to be a host-mapped vmaMapMemory()+memcpy, safe only because
  // every call was ALSO its own one-shot submit+vkQueueWaitIdle (so the write for call N+1 could
  // never happen until the GPU had already finished reading call N's data). Now that many calls are
  // batched into one shared `cmd` and submitted together (see this function's own header comment in
  // VulkanContext.h), that assumption no longer holds: every CPU-side memcpy would happen before the
  // GPU executes ANY of them, leaving m_ParamsBuffer holding only the LAST call's bytes by the time
  // the GPU actually runs. vkCmdUpdateBuffer avoids this entirely -- its source data is copied into
  // the command buffer's own storage immediately at record time (same semantics already relied on by
  // FlushPendingEntityDataPatches()-style updates elsewhere in this codebase), so each dispatch keeps
  // its own correct snapshot no matter how many later calls reuse the same buffer bytes afterward.
  if (uboParamsData != nullptr) {
    vkCmdUpdateBuffer(cmd, m_ParamsBuffer, 0, static_cast<VkDeviceSize>(uboParamsSize), uboParamsData);

    // RAW barrier: the update above must be visible to the compute dispatch below, which reads it
    // through the Params UBO. srcStageMask = CLEAR_BIT (not TRANSFER_BIT) because vkCmdUpdateBuffer
    // is classified under VK_PIPELINE_STAGE_2_CLEAR_BIT by the Vulkan spec's pipeline-stage table --
    // same classification this codebase's own per-frame EntityData patch path already relies on.
    VkMemoryBarrier2 uboWriteBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    uboWriteBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    uboWriteBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    uboWriteBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    uboWriteBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;

    VkDependencyInfo uboWriteDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    uboWriteDep.memoryBarrierCount = 1;
    uboWriteDep.pMemoryBarriers = &uboWriteBarrier;
    vkCmdPipelineBarrier2(cmd, &uboWriteDep);
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  VkDescriptorSet set = (descriptorSetOverride != VK_NULL_HANDLE)
                            ? descriptorSetOverride
                            : m_GeometryDescriptorSet;
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                          &set, 0, nullptr);

  if (pushConstantData != nullptr) {
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(pushConstantSize),
                       pushConstantData);
  }

  vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);

  // Memory layout safe transition barrier execution: compute writes to the shared Vertex/Index
  // SSBOs must be visible to (a) the vertex shader stage that reads them at draw time [pre-existing
  // purpose], (b) any LATER compute dispatch batched into this same `cmd` that also reads them --
  // concretely GenerateGeometry()'s AUTOSMOOTH post-pass, which reads every vertex written by every
  // call that precedes it in the batch, hence COMPUTE_SHADER_BIT added to dstStageMask below (a
  // dispatch-write -> vertex-shader-read-only barrier would silently under-synchronize that pass
  // once per-call vkQueueWaitIdle stopped papering over it), and (c) this dispatch's own Params-UBO
  // read (if any, see above) must fully finish before the NEXT call's vkCmdUpdateBuffer is allowed to
  // overwrite the same m_ParamsBuffer bytes -- a write-after-read hazard needing only the execution
  // ordering a barrier already provides, so CLEAR_BIT is folded into this same barrier's dstStageMask
  // rather than issuing a separate one.
  VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
  memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_CLEAR_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

  VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  depInfo.memoryBarrierCount = 1;
  depInfo.pMemoryBarriers = &memBarrier;
  vkCmdPipelineBarrier2(cmd, &depInfo);
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

namespace {

// ---------------------------------------------------------------------------------------------
// 10-tree-species recipe table (renderer::ProceduralTreePass) -- one distinct L-system
// parameterization per species, shared by BuildEntityData() (per-entity bark/leaf materialID
// assignment) and GenerateGeometry()'s TREES block (shape parameters + grove placement), so the
// two can never drift apart. Every numeric field must respect geom_tree_bark.comp/
// geom_tree_leaves.comp's shared validity guard (trunkHeight/trunkRadius/leafSize > 0,
// lengthTaper/radiusTaper strictly inside (0, 1), branchFactor >= 2, 1 <= depth <=
// tree_lsystem.glsl's TREE_MAX_DEPTH) -- a violating recipe doesn't crash, the dispatch just
// silently generates nothing, which is far worse to diagnose.
//
// Species silhouettes come purely from the recipe (see tree_lsystem.glsl's transform model --
// note every level-1 branch fans out from the trunk's single tip, so a species' character is
// its CROWN shape + trunk proportions):
//   oak:    broad spreading crown (branchFactor 4, wide angle, low damping).
//   pine:   tall trunk, compact drooping conifer crown (short branches, steep angle + damping).
//   birch:  slender white trunk, small light foliage (its own bark/leaf material variants).
//   willow: trailing cascade (high pitchDamping accumulates pitch past horizontal -> droop).
//   poplar: tight columnar crown (near-vertical branch angle, short branches).
//   maple:  round crown, autumn orange-red foliage variant.
//   baobab: squat massive trunk (largest trunkRadius, shallow depth, 5-way branching).
//   palm:   bare tall trunk + one tier of 7 huge leaf cards (depth 1 -- the crown IS the
//           leaf tier, the closest this L-system gets to fronds).
//   shrub:  no real trunk (0.55 tall), dense 5-way ball of foliage at ground level.
//   dead:   gnarled near-bare snag (binary branching, depth 5, gray wood, tiny dry remnants).
// ---------------------------------------------------------------------------------------------
struct TreeSpeciesRecipe {
  uint32_t depth;            // L-system recursion levels past the trunk.
  uint32_t branchFactor;     // Children per branching node.
  float trunkHeight;
  float trunkRadius;
  float lengthTaper;         // Per-level segment-length multiplier, strictly (0, 1).
  float radiusTaper;         // Per-level segment-radius multiplier, strictly (0, 1).
  float branchAngleRadians;  // Cone half-angle added away from vertical per level.
  float pitchDamping;        // Parent pitch carry-over factor, [0, 1].
  uint32_t sides;            // Bark cylinder cross-section side count.
  float leafSize;            // Leaf cross-quad half-extent, world units.
  uint32_t barkMaterialID;   // renderer::kTreeBark*MaterialID family.
  uint32_t leafMaterialID;   // renderer::kTreeLeaf*MaterialID family.
  float groveOffsetX;        // Placement inside the grove clearing, relative to its anchor.
  float groveOffsetZ;
};

// Grove layout: two staggered columns (quincunx) at grove-local X = 0 and -3.5, marching NORTH
// (+Z) from the clearing anchor (kTreeClearingX, VulkanContext::GenerateGeometry's TREES block --
// now -10.0, WEST of the origin). Placement here is driven by the default camera
// (main.cpp's `Camera camera({12.3613, 6.5726, 0}, {0, 4, 0})`, 45-degree vertical FOV), verified
// numerically (camera-space horizontal/vertical angle for every one of these 10 positions stays
// within ~92% of both frustum half-angles, at both ground level and each species' own canopy
// height) rather than eyeballed: this grove replaced an EARLIER version placed east of the
// gallery (X=+10..+14.5, Z=0..-18) that was never actually visible on launch -- that placement
// put every tree at a shallow forward-depth from the camera (which itself sits at X=+12.36,
// nearly co-located in X with that old clearing), so even modest vertical/lateral offsets
// produced huge camera-space angles. West of the origin gives far more forward depth (~22-27
// units) for the same lateral spread, at the cost of switching sides from the ORIGINAL 4-tree
// layout's east-side placement (which itself predates the terrain-hydrology river and was never
// re-verified against the camera after the tree-species/grove-widening change). Min Z (5.0) stays
// >= 5 world units clear of both the gallery's own row=1 zones (Torus/Tube at world Z=+4, see
// GridSlot()'s own layout) and the skeletal-animation creature's clearing (kCreatureClearingX
// =-10, kCreatureClearingZ=0, ~5-unit bind-pose chain length) -- the river itself (terrain-
// hydrology feature, control points (30,30)..(6,6), all positive X/Z) is trivially clear on this
// west side regardless of Z, since ClosestPointOnRiverPolyline clamps to the segment's own
// endpoints rather than an infinite diagonal.
constexpr std::array<TreeSpeciesRecipe, 10> kTreeSpecies = {{
    // depth bF  trunkH trunkR lenTap radTap angle  damp  sides leafSz barkMat                              leafMat                               offX   offZ
    {  4u,   4u, 2.6f,  0.22f, 0.70f, 0.62f, 0.85f, 0.45f, 7u, 0.26f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafMaterialID,        0.0f,   5.0f }, // oak
    {  4u,   3u, 3.6f,  0.15f, 0.50f, 0.58f, 1.25f, 0.35f, 6u, 0.20f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafPineMaterialID,    0.0f,   7.0f }, // pine
    {  4u,   3u, 3.2f,  0.10f, 0.66f, 0.55f, 0.42f, 0.55f, 5u, 0.16f, renderer::kTreeBarkBirchMaterialID, renderer::kTreeLeafBirchMaterialID,   0.0f,   9.0f }, // birch
    {  5u,   3u, 2.4f,  0.17f, 0.78f, 0.55f, 0.55f, 0.95f, 6u, 0.18f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafWillowMaterialID,  0.0f,  11.0f }, // willow
    {  4u,   3u, 3.8f,  0.13f, 0.58f, 0.60f, 0.20f, 0.30f, 6u, 0.17f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafMaterialID,        0.0f,  13.0f }, // poplar
    {  4u,   4u, 2.2f,  0.16f, 0.72f, 0.62f, 0.65f, 0.50f, 6u, 0.24f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafAutumnMaterialID, -3.5f,   6.5f }, // autumn maple
    {  3u,   5u, 1.9f,  0.45f, 0.55f, 0.48f, 0.75f, 0.40f, 8u, 0.18f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafMaterialID,       -3.5f,   8.5f }, // baobab
    {  1u,   7u, 3.4f,  0.14f, 0.35f, 0.30f, 1.05f, 0.00f, 6u, 0.45f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafMaterialID,       -3.5f,  10.5f }, // palm
    {  3u,   5u, 0.55f, 0.07f, 0.75f, 0.60f, 0.90f, 0.60f, 5u, 0.20f, renderer::kTreeBarkMaterialID,      renderer::kTreeLeafBirchMaterialID,  -3.5f,  12.5f }, // shrub
    {  5u,   2u, 2.9f,  0.18f, 0.76f, 0.66f, 0.70f, 0.85f, 5u, 0.055f, renderer::kTreeBarkDeadMaterialID, renderer::kTreeLeafDeadMaterialID,   -3.5f,  14.5f }, // dead tree
}};

#ifndef NDEBUG
// Log labels only -- kept OUT of TreeSpeciesRecipe itself so Release builds embed no species
// string literals at all (CLAUDE.md's zero-debug-string rule), rather than relying on the
// optimizer to dead-strip an unused const char* field.
constexpr std::array<const char*, 10> kTreeSpeciesNames = {
    "oak", "pine", "birch", "willow", "poplar",
    "autumn maple", "baobab", "palm", "shrub", "dead tree",
};
#endif

} // namespace

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

  // Cornell-box wall colors (GenerateShowcaseMaterialTable()'s own hand-authored table.params[12]/
  // [13] recipes) are hand-authored at these two FIXED materialID slots, independent of which
  // entity index actually renders them -- before the procedural-tree entities below were inserted,
  // kWallEntityIndexA/B happened to equal these same two numbers, so the pre-existing
  // `entity.materialID = i` default silently did the right thing. That coincidence no longer holds
  // once kTreeEntityCount pushes kWallEntityIndexA/B upward, so both are now named explicitly and
  // assigned via the same override mechanism the hero/floor/water entities already use below.
  constexpr uint32_t kWallMaterialIDA = 12u;
  constexpr uint32_t kWallMaterialIDB = 13u;

  for (uint32_t i = 0; i < kEntityCount; ++i) {
    core::EntityID id = core::IDManager::GetNextID();

    // Phase 0.1 (dynamic instance registry): built as a local value, not a reference into the
    // registry, then handed to AcquireSlot() at the end of the loop body -- see
    // m_InstanceRegistry's own declaration-site comment for why AcquireSlot() is guaranteed to
    // return index == i here.
    core::EntityData entity{};
    entity.meshID = static_cast<uint32_t>(id & 0xFFFFFFFFu);
    entity.materialID = i;
    entity.cellID = 0u;
    entity.flags = 0u;
    core::SetFlag(entity.flags, core::EntityFlags::CastShadows, true);

    // VSM advanced roadmap, Feature 2 (real static-vs-dynamic page invalidation): activates the
    // previously-dead IsDynamic bit on the 12 continuously-rotating showcase entities (indices
    // [0, kTreeEntityIndexBase) == [0, 12) -- see UpdateEntityRotations(), which leaves every
    // entity from kTreeEntityIndexBase onward (the procedural trees, the 2 Lumen-corner walls, the
    // floor, the water plane) permanently static. Gives renderer::VirtualShadowMapPass's own
    // per-page dynamic-content classification a real, meaningful flag to test instead of the dead
    // default of always-0.
    if (i < kTreeEntityIndexBase) {
      core::SetFlag(entity.flags, core::EntityFlags::IsDynamic, true);
    }

    // Default: materialID == entity index for the 12 showcase primitives (BY DESIGN, see
    // GenerateShowcaseMaterialTable()'s own zone-layout comment); every entity beyond that
    // (procedural trees, walls, floor, water, ...) gets its real materialID (and, from it,
    // isTransparent) from an explicit override further below instead -- this default is only ever
    // actually consumed by the 12 showcase primitives. The bounds guard is NOT decorative:
    // kEntityCount (37 since the 10-tree-species scene) now exceeds renderer::kMaxMaterials (32),
    // so the tail entity indices (creature/walls/floor/water, 32..36) would read past the
    // 32-element isTransparent array here -- an MSVC-debug std::array assertion abort at startup
    // -- if this default were still read unconditionally. Every one of those tail entities takes
    // one of the explicit overrides below, so the placeholder `false` is never actually consumed.
    bool isTransparent = (i < renderer::kMaxMaterials) ? m_MaterialTable.isTransparent[i] : false;
    // Generalized Nanite Tessellation (renderer::TessellationPass -- see kTessellatedEntityIndices'
    // own comment for the full generalization rationale and exact entity choices): every entity
    // index in kTessellatedEntityIndices is rendered ONLY by renderer::TessellationPass -- never by
    // the opaque Nanite VisBuffer pipeline (no representation for runtime-displaced geometry there)
    // nor by TransparentForwardPass. Forces the entity's IsTransparent flag true -- NOT because
    // it's actually alpha-blended, but because ClusterLODCompact.comp's existing per-entity
    // IsTransparent exclusion (see that shader's own EntityDataBuffer comment) is the exact "never
    // enters the opaque candidate list" mechanism every tessellated entity also needs.
    // TransparentForwardPass itself stays unaffected: it filters by
    // materialTable.isTransparent[materialID], and every tessellated entity's OWN materialID
    // (unmodified showcase recipe, except the hero -- see below) correctly stays non-transparent
    // there, so none of these entities' clusters ever enter ITS candidate list either.
    bool isTessellated = std::find(kTessellatedEntityIndices.begin(), kTessellatedEntityIndices.end(), i)
        != kTessellatedEntityIndices.end();
    if (isTessellated) {
      core::SetFlag(entity.flags, core::EntityFlags::IsTessellated, true);
      isTransparent = true;
    }
    // The original hero entity (kHeroEntityIndex, the Icosphere) keeps its own pre-existing
    // materialID override to the reserved renderer::kHeroMaterialID slot (see that constant's own
    // comment) -- every OTHER tessellated entity (slots 3/9 above) keeps its own regular showcase
    // materialID unmodified, so renderer::TessellationPass shades each with its own distinct
    // look (see ClusterRenderPipeline's own m_Tessellation Init() call site for how each entity's
    // materialID is resolved into its GPU draw info).
    if (i == kHeroEntityIndex) {
      entity.materialID = renderer::kHeroMaterialID;
    }
    // Procedural tree generator (renderer::ProceduralTreePass): each tree is baked as a bark
    // entity followed immediately by its leaf entity (see VulkanContext.h's own kTreeEntityCount
    // comment) -- even local offset = bark (opaque solid cylinder mesh), odd local offset = leaves
    // (opaque base material too: the leaf cards use an opacity-CUTOUT mask, not alpha-blend
    // transparency, see MaterialParameterTable.h's own kTreeLeafMaterialID comment for why
    // isTransparent correctly stays false for it). 10-tree-species scene: bark/leaf materialIDs
    // now come from the species recipe table (kTreeSpecies above) instead of one shared pair, so
    // e.g. the birch gets its white bark and the maple its autumn foliage -- every one of these
    // material slots is a hand-authored opaque recipe, so isTransparent stays false throughout.
    if (i >= kTreeEntityIndexBase && i < kTreeEntityIndexBase + kTreeEntityCount) {
      static_assert(kTreeSpecies.size() == kTreeVisualCount,
                    "kTreeSpecies (VulkanContext.cpp) and kTreeVisualCount (VulkanContext.h) must "
                    "describe the same number of trees -- update both together.");
      uint32_t localIdx = i - kTreeEntityIndexBase;
      uint32_t speciesIdx = localIdx / 2u;
      bool isLeafPart = (localIdx % 2u) == 1u;
      const TreeSpeciesRecipe& species = kTreeSpecies[speciesIdx];
      entity.materialID = isLeafPart ? species.leafMaterialID : species.barkMaterialID;
      isTransparent = m_MaterialTable.isTransparent[entity.materialID];
    }
    // Lumen/GI showcase corner walls -- see kWallMaterialIDA/B's own comment above for why this
    // explicit override is now required (materialID == entity index no longer coincidentally holds
    // for these two once the procedural-tree entities shifted kWallEntityIndexA/B upward).
    if (i == kWallEntityIndexA) {
      entity.materialID = kWallMaterialIDA;
      isTransparent = m_MaterialTable.isTransparent[entity.materialID];
    }
    if (i == kWallEntityIndexB) {
      entity.materialID = kWallMaterialIDB;
      isTransparent = m_MaterialTable.isTransparent[entity.materialID];
    }
    // Phase 7b (UE5.8 parity roadmap, terrain heightfield): the floor entity is now a procedural
    // terrain heightfield (see GenerateGeometry()'s own terrain block), not a flat plane -- override
    // its materialID to the reserved renderer::kTerrainMaterialID slot so ClusterResolve.comp/
    // ClusterResolveBinned.comp's height/slope biome blend (terrain_shading.glsl) applies to it.
    // Unlike the hero entity, the terrain stays fully opaque and needs no IsTransparent exclusion --
    // it renders through the normal opaque Nanite path unmodified (see kTerrainMaterialID's own
    // comment).
    if (i == kFloorEntityIndex) {
      // Minimal-scene mode: the floor is now a flat GeneratePlane() (GenerateGeometry()'s own
      // floor block), not the terrain heightfield -- MaterialParameterTable.h's slot 14 ("Floor
      // (slot 14): neutral matte gray ground") is the exact pre-hydrology floor material this
      // reverts to; kTerrainMaterialID's biome-blend shading (terrain_shading.glsl) is for a real
      // heightfield's slope/height variation, which a flat plane doesn't have.
      entity.materialID = kMinimalSceneMode ? 14u : renderer::kTerrainMaterialID;
      isTransparent = m_MaterialTable.isTransparent[entity.materialID];
    }
    // Skeletal-animation feature: the procedural creature -- stays fully opaque (no exclusion
    // mechanism needed, unlike the hero/tessellated/water entities: skinned geometry renders
    // through the NORMAL opaque Nanite VisBuffer pipeline, gated purely by
    // core::EntityFlags::IsSkeletallyAnimated below -- see that flag's own comment for why this is
    // the whole point of "Nanite skinning" vs. a separate forward pass).
    if (i == kCreatureEntityIndex) {
      entity.materialID = renderer::kCreatureMaterialID;
      isTransparent = m_MaterialTable.isTransparent[entity.materialID];
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
    // (kHeroEntityIndex == 2) and routes it exclusively through renderer::TessellationPass, which
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
    // Skeletal-animation feature: opts the procedural creature into GPU linear-blend vertex
    // skinning inside the normal Nanite cluster pipeline -- see core::EntityFlags::
    // IsSkeletallyAnimated's own comment.
    if (i == kCreatureEntityIndex) {
      core::SetFlag(entity.flags, core::EntityFlags::IsSkeletallyAnimated, true);
    }

    // Phase 0.1 (dynamic instance registry): claims this entity's slot via the registry's
    // Acquire path instead of writing directly into a fixed array element. The registry starts
    // completely empty and this loop runs before anything else ever calls AcquireSlot(), so the
    // "bump the high-water mark" path is guaranteed to hand back slotIndex == i.
    uint32_t slotIndex = m_InstanceRegistry.AcquireSlot(entity);
    assert(slotIndex == i && "BuildEntityData(): showcase entity did not land at its expected absolute index");
    (void)slotIndex;
  }

  // --- Runtime World Partition streaming pool (see kStreamingUnitCount's own comment) ---
  // Continues the SAME core::IDManager sequence the loop above started (deliberately not reset --
  // see BuildEntityData()'s own "Context 0" comment), so meshIDs stay dense across the whole
  // [0, kTotalEntityCount) range. Every slot starts fully valid (a real, small, pre-baked mesh --
  // see GenerateGeometry()'s streaming block) but core::EntityFlags::StreamingInactive so it never
  // draws until world::WorldCellStreamingLoader's main-thread pump claims it via
  // SetStreamingUnitState().
  // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): dedicated units get their materialID from
  // the SAME real per-cell archetypeShape GenerateGeometry() later bakes their fine mesh's
  // silhouette from (see that function's own streaming-pool block) -- both must agree, or a unit's
  // material (e.g. green "Bush") would visibly mismatch its baked mesh shape (e.g. a "Tree"
  // capsule). Units beyond the dedicated range (no manifest, or this unit's index exceeds the
  // authored cell count) keep the pre-Phase-5 unit%4 shared rotation.
  const std::vector<world::CellCoord>& orderedCellsForShape = m_CellManifest.GetOrderedCells();
  for (uint32_t i = kEntityCount; i < kTotalEntityCount; ++i) {
    core::EntityID id = core::IDManager::GetNextID();
    uint32_t unit = (i - kEntityCount) / kStreamingSlotsPerUnit;
    uint32_t shape = unit % kStreamingArchetypeShapeCount;
    if (unit < orderedCellsForShape.size()) {
      std::optional<world::CellPlacement> dedicatedPlacement = m_CellManifest.GetPlacement(orderedCellsForShape[unit]);
      if (dedicatedPlacement.has_value()) {
        shape = dedicatedPlacement->archetypeShape % kStreamingArchetypeShapeCount;
      }
    }

    // Phase 0.1 (dynamic instance registry): same local-value + AcquireSlot() pattern as the
    // showcase loop above -- this loop continues immediately where that one left off (registry
    // high-water mark == kEntityCount already), so slotIndex is guaranteed == i here too.
    core::EntityData entity{};
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

    uint32_t slotIndex = m_InstanceRegistry.AcquireSlot(entity);
    assert(slotIndex == i && "BuildEntityData(): streaming-pool slot did not land at its expected absolute index");
    (void)slotIndex;
  }
}

void VulkanContext::UploadEntityData() {
  // m_EntityBuffer is VMA_MEMORY_USAGE_GPU_ONLY (not host-visible), so
  // uploading the CPU-authored entity records (m_InstanceRegistry's backing store) requires a
  // temporary host-visible staging buffer plus an explicit GPU-side copy, mirroring the
  // one-time-submit pattern used elsewhere in Init(). Only the first kTotalEntityCount slots are
  // uploaded -- m_InstanceRegistry's own Debug-only headroom capacity (if any) never reaches the
  // GPU, since it exists solely for RunInstanceRegistrySmokeTest()'s own probe slots.
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
  std::memcpy(stagingAllocResultInfo.pMappedData, m_InstanceRegistry.Data(),
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
    VkCommandBuffer cmd,
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

    DispatchGeometryCompute(cmd, m_BoxFacePipelines[face],
                            m_BoxComputePipelineLayout, nullptr, 0, &params,
                            sizeof(params), groupCountX, groupCountY, 1);

    runningVertexOffset += uSegsCount * vSegsCount;
    runningIndexOffset += uSegs * vSegs * 6u;
  }
}

void VulkanContext::GenerateCone(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_ConePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;
  // Each cap's innermost ring is a fan from the center vertex (1 triangle/column, 3 indices),
  // every ring above it a genuine quad strip (2 triangles/column, 6 indices) -- see
  // geom_cone.comp's own comment on why this differs from a uniform 6-indices-per-ring stride.
  uint32_t capIndexCount = params.sides * 3u + (params.capSegments - 1u) * params.sides * 6u;
  runningIndexOffset += params.sides * params.heightSegments * 6u + 2u * capIndexCount;
}

void VulkanContext::GenerateSphere(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_SpherePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += vertCount;
  runningIndexOffset += 6u * params.segments * ringCount;
}

void VulkanContext::GenerateIcosphere(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_IcospherePipeline, m_ComputePipelineLayout,
                          &params, sizeof(params), nullptr, 0, groupCountXY,
                          groupCountXY, outBaseFaceCount);

  outVertsPerFace = (params.segments + 1u) * (params.segments + 2u) / 2u;
  uint32_t totalVerts = outVertsPerFace * outBaseFaceCount;
  uint32_t icosphereIndexCount = outBaseFaceCount * params.segments * params.segments * 3u;

  runningVertexOffset += totalVerts;
  runningIndexOffset += icosphereIndexCount;
}

void VulkanContext::GenerateCylinder(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_CylinderPipeline, m_ComputePipelineLayout,
                          &params, sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;
  // Each cap's innermost ring is a fan from the center vertex (1 triangle/column, 3 indices),
  // every ring above it a genuine quad strip (2 triangles/column, 6 indices) -- see
  // geom_cylinder.comp's own comment on why this differs from a uniform 6-indices-per-ring stride.
  uint32_t capIndexCount = params.sides * 3u + (params.capSegments - 1u) * params.sides * 6u;
  runningIndexOffset += 6u * params.sides * params.heightSegments + 2u * capIndexCount;
}

void VulkanContext::GenerateTube(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_TubePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 12u * params.sides * params.heightSegments +
                        12u * params.sides * params.capSegments;
}

void VulkanContext::GenerateTorus(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_TorusPipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCount, 1, 1);

  runningVertexOffset += vertCount;
  runningIndexOffset += 6u * params.segments * params.sides;
}

void VulkanContext::GeneratePyramid(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_PyramidPipeline, m_ComputePipelineLayout, &params,
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
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_PlanePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (params.widthSegments - 1u) * (params.lengthSegments - 1u);
}

void VulkanContext::GenerateTerrain(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_TerrainPipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (params.widthSegments - 1u) * (params.lengthSegments - 1u);
}

void VulkanContext::GenerateWaterPlane(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_PlanePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (params.widthSegments - 1u) * (params.lengthSegments - 1u);
}

void VulkanContext::GenerateWaterSurface(
    VkCommandBuffer cmd,
    float Span, uint32_t Segments,
    uint32_t meshID, maths::vec2 slot, float worldOffsetY,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  assert(Span > 0.0f);
  assert(Segments >= 2u);

  // geom_water_surface.comp's Params UBO is byte-identical to PlaneParams -- reused directly,
  // same convention as GenerateTerrain().
  PlaneParams params{};
  params.width = Span;
  params.length_ = Span;
  params.widthSegments = Segments;
  params.lengthSegments = Segments;
  params.meshID = meshID;
  params.materialID = 0.0f; // See GenerateTerrain()'s own identical comment on this field.
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  // The TERRAIN's anchor, not a fixed water level: vertex Y comes from the hydrology bake's
  // water-surface texture, which lives in terrain-local height space (see this function's own
  // header comment in VulkanContext.h).
  params.worldOffsetY = worldOffsetY;
  params.worldOffsetZ = slot.y;

  uint32_t totalVerts = params.widthSegments * params.lengthSegments;
  constexpr uint32_t kLocalSizeXY = 8u; // geom_water_surface.comp local_size = (8, 8, 1)
  uint32_t groupCountX = (params.widthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;
  uint32_t groupCountY = (params.lengthSegments + kLocalSizeXY - 1u) / kLocalSizeXY;

  // The one caller of the descriptor-set override: binding 6 must be the WATER-SURFACE texture
  // here (see m_GeometryDescriptorSetWaterSurface's own header comment).
  DispatchGeometryCompute(cmd, m_WaterSurfacePipeline, m_ComputePipelineLayout, &params,
                          sizeof(params), nullptr, 0, groupCountX, groupCountY, 1,
                          m_GeometryDescriptorSetWaterSurface);

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (params.widthSegments - 1u) * (params.lengthSegments - 1u);
}

void VulkanContext::GenerateRiver(
    uint32_t meshID, uint32_t segmentsAlong, uint32_t segmentsAcross,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  assert(segmentsAlong >= 2u);
  assert(segmentsAcross >= 2u);

  RiverParams params{};
  params.segmentsAlong = segmentsAlong;
  params.segmentsAcross = segmentsAcross;
  // Must match river_spline.glsl's own kRiverHalfWidth exactly -- passed through explicitly (see
  // this struct's own comment) rather than re-derived here.
  params.halfWidth = 2.2f;
  params.meshID = meshID;
  params.materialID = 0.0f; // See GenerateTerrain()'s own identical comment on this field.
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = 0.0f; // river_spline.glsl's control points are already world-space.
  params.worldOffsetY = 0.0f;
  params.worldOffsetZ = 0.0f;

  uint32_t totalVerts = segmentsAlong * segmentsAcross;
  constexpr uint32_t kLocalSizeXY = 8u; // geom_river.comp local_size = (8, 8, 1)
  uint32_t groupCountX = (segmentsAlong + kLocalSizeXY - 1u) / kLocalSizeXY;
  uint32_t groupCountY = (segmentsAcross + kLocalSizeXY - 1u) / kLocalSizeXY;

  // Not threaded into GenerateGeometry()'s shared batch command buffer (see that function's own
  // header comment on the startup-latency fix): this function predates that refactor and is always
  // called chained directly onto GenerateWaterPlane() with no cmd of its own to record into, so it
  // keeps its own one-shot submission here, exactly as DispatchGeometryCompute() itself used to do
  // internally before batching required every caller to supply the command buffer explicitly.
  renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
    DispatchGeometryCompute(cmd, m_RiverPipeline, m_ComputePipelineLayout, &params,
                            sizeof(params), nullptr, 0, groupCountX, groupCountY, 1);
  });

  runningVertexOffset += totalVerts;
  runningIndexOffset += 6u * (segmentsAlong - 1u) * (segmentsAcross - 1u);
}

void VulkanContext::GenerateCreature(
    uint32_t meshID, maths::vec2 slot, float worldOffsetY,
    uint32_t sidesPerRing, uint32_t ringsPerSegment, float radiusMin, float radiusMax,
    uint32_t& runningVertexOffset, uint32_t& runningIndexOffset) {
  assert(sidesPerRing >= 3u);
  assert(ringsPerSegment >= 1u);
  assert(radiusMin > 0.0f);
  assert(radiusMax > radiusMin);

  CreatureParams params{};
  // MUST equal animation::SkeletalAnimator::kBoneCount/kSegmentLength exactly -- see this
  // function's own header comment (VulkanContext.h) for why the generated bind-pose mesh and the
  // CPU-side skeleton's own analytic bind pose can never be allowed to drift apart.
  params.boneCount = animation::SkeletalAnimator::kBoneCount;
  params.segmentLength = animation::SkeletalAnimator::kSegmentLength;
  params.sidesPerRing = sidesPerRing;
  params.ringsPerSegment = ringsPerSegment;
  params.radiusMin = radiusMin;
  params.radiusMax = radiusMax;
  params.meshID = meshID;
  params.materialID = 0.0f; // See GenerateTerrain()'s own identical comment on this field.
  params.vertexOffset = runningVertexOffset;
  params.indexOffset = runningIndexOffset;
  params.worldOffsetX = slot.x;
  params.worldOffsetY = worldOffsetY;
  params.worldOffsetZ = slot.y;

  uint32_t totalRings = (params.boneCount - 1u) * params.ringsPerSegment + 1u;
  uint32_t ringVertexCount = totalRings * params.sidesPerRing;
  // +2: tail pole cap + head pole cap (see geom_creature.comp's own vertex-generation comment).
  uint32_t totalVerts = ringVertexCount + 2u;

  constexpr uint32_t kLocalSizeX = 64u; // geom_creature.comp local_size_x = 64
  uint32_t groupCount = (totalVerts + kLocalSizeX - 1u) / kLocalSizeX;

  // Not threaded into GenerateGeometry()'s shared batch command buffer -- see GenerateRiver()'s
  // identical comment just above for why this function keeps its own one-shot submission.
  renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
    DispatchGeometryCompute(cmd, m_CreaturePipeline, m_ComputePipelineLayout, &params,
                            sizeof(params), nullptr, 0, groupCount, 1, 1);
  });

  runningVertexOffset += totalVerts;
  // Body quads between consecutive rings (6 indices/quad) + 2 pole fans (3 indices/triangle,
  // sidesPerRing triangles/fan) -- see geom_creature.comp's own index-generation comment.
  uint32_t bodyIndexCount = (totalRings - 1u) * params.sidesPerRing * 6u;
  uint32_t capIndexCount = 2u * params.sidesPerRing * 3u;
  runningIndexOffset += bodyIndexCount + capIndexCount;
}

void VulkanContext::GenerateCapsule(
    VkCommandBuffer cmd,
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

  DispatchGeometryCompute(cmd, m_CapsulePipeline, m_ComputePipelineLayout, &params,
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
  //
  // Startup-latency fix: every primitive generator below used to be its own blocking one-shot GPU
  // submission (allocate command buffer -> record -> submit -> vkQueueWaitIdle -> free), via
  // DispatchGeometryCompute()'s former internal ExecuteOneShotCommands call -- hundreds of full
  // CPU/GPU round-trips in a row just for this function (12 gallery primitives, several needing
  // multiple dispatches -- the box alone is 6 faces -- plus 2 walls, terrain, water, and
  // kStreamingUnitCount streaming-pool archetypes at up to 7 dispatches apiece). None of these
  // calls has any GPU-side data dependency on another: every one writes into a CPU-computed,
  // disjoint sub-range of the same fixed-size m_VertexBuffer/m_IndexBuffer SSBOs
  // (runningVertexOffset/runningIndexOffset are plain CPU counters threaded through by reference,
  // advanced synchronously with no GPU readback involved), and every downstream consumer
  // (BuildClusterDAG, the AUTOSMOOTH pass below) selects by meshID tag or world-space position,
  // never by buffer-layout adjacency or call order. They are therefore now batched into ONE shared
  // command buffer below (the "GALLERY + STREAMING POOL BATCH"), submitted and waited on ONCE
  // instead of once per call -- see DispatchGeometryCompute()'s own header comment for how its
  // Params-UBO write was made safe to batch this way (vkCmdUpdateBuffer instead of a host memcpy)
  // and how its trailing barrier was widened to keep the AUTOSMOOTH cross-dependency below correct.
  //
  // Two real exceptions, both preserved exactly as before:
  //  1. The icosphere is generated in its OWN one-shot submission, separate from the batch below,
  //     because DebugReadbackGeometrySample() right after it does an immediate CPU-side GPU
  //     readback that hard-depends on the icosphere's compute writes already being complete and
  //     visible (see that function's own comment: "GenerateGeometry() already vkQueueWaitIdle'd
  //     after the compute dispatch before calling this function"). Deferring it into the batch
  //     below (which is not submitted until everything else has also been recorded) would make that
  //     readback race a still-unsubmitted, unexecuted dispatch and read stale/garbage data. Since
  //     the icosphere is a single dispatch anyway, no batching win is being given up here.
  //  2. The AUTOSMOOTH post-pass genuinely depends on every OTHER dispatch's vertex writes (it
  //     welds normals across the ENTIRE accumulated vertex buffer, see its own block below) -- a
  //     real cross-dispatch data dependency, not just a shared-buffer-region false alarm. It is
  //     still safe to fold into the SAME batch as everything else (rather than needing its own
  //     submission) because it is recorded strictly after every generator call it depends on, and
  //     DispatchGeometryCompute()'s trailing barrier now also orders any later COMPUTE_SHADER_BIT
  //     consumer batched into the same command buffer, not just VERTEX_SHADER_BIT draw-time reads.
  auto gridSlot = [this](int slotIndex) -> maths::vec2 {
    return GridSlot(slotIndex);
  };

  uint32_t runningVertexOffset = 0;
  uint32_t runningIndexOffset = 0;

  LOG_INFO("[GenerateGeometry] Generating 12 procedural primitives across a 9-zone "
           "feature-showcase gallery, plus the Lumen-corner walls and floor...");

  // -------------------------------------------------------------------------
  // ICOSPHERE (slot 2 visually) — generated first so it occupies buffer offset 0. Kept as its own
  // one-shot submission -- see this function's own header comment (point 1) for why.
  // -------------------------------------------------------------------------
  uint32_t icosphereVertsPerFace = 0;
  uint32_t icosphereIndexCount = 0;
  uint32_t icosphereBaseFaceCount = 20u;
  {
    maths::vec2 slot = gridSlot(2);

    // Target parameters to accept and strictly validate:
    // IcoSphere: Radius, Segments, Tetra, Octa, Icosa
    // Minimal-scene mode: shrunk to a visually negligible size rather than skipped outright --
    // this is kHeroEntityIndex, hard-wired into DebugReadbackGeometrySample() (assumes buffer
    // offset 0) and renderer::TessellationPass (its own dedicated forward path), so leaving its
    // generation running (just tiny) is far lower-risk than unwiring either of those.
    float Radius = kMinimalSceneMode ? 0.001f : 0.8f;
    bool Tetra = false;
    bool Octa = false;
    bool Icosa = true;

    renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
      GenerateIcosphere(cmd, Radius, Tetra, Octa, Icosa,
                        m_InstanceRegistry[2].meshID, slot,
                        runningVertexOffset, runningIndexOffset,
                        icosphereBaseFaceCount, icosphereVertsPerFace);
    });

    float edgeLength = (Icosa ? 1.05f : (Octa ? 1.41f : 1.63f)) * Radius;
    uint32_t Segments = std::max(1u, static_cast<uint32_t>(std::round(edgeLength / config::VERTEX_SPACING)));
    icosphereIndexCount = icosphereBaseFaceCount * Segments * Segments * 3u;
  }
  // DEBUG: sample-readback the icosphere geometry, exactly as before this
  // feature, to catch regressions in the (still buffer-offset-0) icosphere
  // generation.
  DebugReadbackGeometrySample(icosphereVertsPerFace, icosphereIndexCount);

  // -------------------------------------------------------------------------
  // GALLERY + STREAMING POOL BATCH -- every remaining primitive generator call in this function
  // (Box through the streaming pool's archetypes) plus the AUTOSMOOTH post-pass, all recorded into
  // ONE shared command buffer and submitted/waited on exactly once -- see this function's own
  // header comment for why this is safe.
  // -------------------------------------------------------------------------
  renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
  // Minimal-scene mode: every OTHER showcase primitive besides the UV sphere below (Box, Cone,
  // Torus, Tube, Capsule, Cylinder, TorusKnot, ChamferBox) is skipped entirely -- see
  // kMinimalSceneMode's own declaration comment for why this is safe. Plane (slot 3, just below)
  // is NOT in that group: it's one of VulkanContext::kTessellatedEntityIndices/
  // ClusterRenderPipeline.cpp's kTessellatedEntityIDs = {2, 3, 9} (mirrors kHeroEntityIndex=2's own
  // "shrink, don't skip" treatment above), and TessellationPass::Init() hard-fails (LOG_ERROR +
  // return false) if any of those 3 entities has no Fallback Mesh draw range -- i.e. zero baked
  // geometry there is a startup crash, not a graceful no-op like every other skipped entity.
  if (!kMinimalSceneMode) {
  // -------------------------------------------------------------------------
  // BOX (slot 0) — 6 compute dispatches, one per cube face, chained onto the
  // same meshID.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(0);
    GenerateBox(cmd, 1.4f, 1.4f, 1.4f, m_InstanceRegistry[0].meshID, slot, runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // CONE (slot 1)
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(1);
    GenerateCone(cmd, 0.7f, 0.35f, 1.4f, m_InstanceRegistry[1].meshID, slot, runningVertexOffset, runningIndexOffset);
  }
  } // !kMinimalSceneMode (Box/Cone)

  // -------------------------------------------------------------------------
  // PLANE (slot 3 visually) — always generated, see this block's own kMinimalSceneMode comment
  // above (kTessellatedEntityIndices hard requirement, mirrors the icosphere/hero's treatment).
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(3);

    // Target parameters to accept and strictly validate:
    // Plane: Length, Width, LengthSegments, WidthSegments
    float Length = kMinimalSceneMode ? 0.001f : 1.4f;
    float Width = kMinimalSceneMode ? 0.001f : 1.4f;

    GeneratePlane(cmd, Length, Width,
                  m_InstanceRegistry[3].meshID, slot,
                  runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // SPHERE / UV sphere (slot 4) — the minimal scene's own sphere. Always generated.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(4);
    GenerateSphere(cmd, 0.8f, m_InstanceRegistry[4].meshID, slot, runningVertexOffset, runningIndexOffset);
  }

  if (!kMinimalSceneMode) {
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

    GenerateTorus(cmd, Radius1, Radius2, Rotation, Twist,
                  m_InstanceRegistry[5].meshID, slot,
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

    GenerateTube(cmd, Radius1, Radius2, Height,
                 m_InstanceRegistry[6].meshID, slot,
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

    GenerateCapsule(cmd, Radius, Height,
                    m_InstanceRegistry[7].meshID, slot,
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

    GenerateCylinder(cmd, Radius, Height,
                     m_InstanceRegistry[8].meshID, slot,
                     runningVertexOffset, runningIndexOffset);
  }
  } // !kMinimalSceneMode (Torus/Tube/Capsule/Cylinder)

  // -------------------------------------------------------------------------
  // PYRAMID (slot 9) — square (4-sided) flat-shaded pyramid. Always generated -- one of
  // kTessellatedEntityIndices, see the PLANE (slot 3) block's own kMinimalSceneMode comment above
  // for why this can be shrunk but never fully skipped.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = gridSlot(9);

    // Target parameters to accept and strictly validate:
    // Pyramide: Width, Depth, Height, WidthSegments, DepthSegments,
    // HeightSegments
    float Width = kMinimalSceneMode ? 0.001f : 1.4f;
    float Depth = kMinimalSceneMode ? 0.001f : 1.4f;
    float Height = kMinimalSceneMode ? 0.001f : 1.2f;

    GeneratePyramid(cmd, Width, Depth, Height,
                    m_InstanceRegistry[9].meshID, slot,
                    runningVertexOffset, runningIndexOffset);
  }

  if (!kMinimalSceneMode) {
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

    params.meshID = m_InstanceRegistry[10].meshID;
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

    DispatchGeometryCompute(cmd, m_TorusKnotPipeline, m_ComputePipelineLayout,
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
    params.meshID = m_InstanceRegistry[11].meshID;
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

    DispatchGeometryCompute(cmd, m_ChamferBoxPipeline, m_ComputePipelineLayout,
                            &params, sizeof(params), nullptr, 0, groupCount, 1,
                            1);

    runningVertexOffset += vertCount;
    // Same index topology as geom_sphere.comp: sideSegs*3 (top fan) +
    // (ringCount-1)*sideSegs*6 (middle quads) + sideSegs*3 (bottom fan) =
    // 6*sideSegs*ringCount.
    runningIndexOffset += 6u * params.sideSegs * ringCount;
  }
  } // !kMinimalSceneMode (TorusKnot/ChamferBox)

  // -------------------------------------------------------------------------
  // TREES (entities kTreeEntityIndexBase..kTreeEntityIndexBase+kTreeEntityCount-1) -- Procedural
  // SpeedTree-style trees (renderer::ProceduralTreePass), fulfilling CLAUDE.md's "Arbres (generes
  // par du code style speedtree)" requirement: kTreeVisualCount distinct trees, each two entities
  // (bark + leaves -- see ProceduralTreePass.h's own class comment on why one entity can't hold
  // both materials), placed in their own small clearing WEST of the 3x3 showcase grid (see
  // kTreeSpecies's own header comment, right above BuildEntityData(), for exactly why west and
  // exactly these coordinates -- verified numerically against the default camera's actual frustum,
  // not eyeballed: an EARLIER east-side placement here was never actually visible on launch).
  // Each tree's bark+leaf pair is generated by its own scoped ProceduralTreePass instance
  // (Init -> RecordGenerate -> Shutdown, all within this block) -- see that class's own header
  // comment for why this is a bake-time-only pass, not a persistent per-frame one like
  // renderer::GlobalSDFPass.
  // -------------------------------------------------------------------------
  if (!kMinimalSceneMode) {
    renderer::ProceduralTreePass treePass;
    // m_TerrainHydrology.Init() already ran (VulkanContext::Init(), well before GenerateGeometry())
    // -- see ProceduralTreePass::Init()'s own header comment for why trees sample this same bake.
    treePass.Init(m_Device, m_CommandPool, m_GraphicsQueue, m_VertexBuffer, m_IndexBuffer,
                  m_TerrainHydrology.GetMeshHeightView(), m_TerrainHydrology.GetLinearSampler());

    // See kTreeSpecies's own header comment for the exact camera-frustum verification this value
    // (and every per-species groveOffsetX/Z) is derived from.
    constexpr float kTreeClearingX = -10.0f;
    // Matches the terrain/floor's own worldOffsetY (GenerateTerrain()'s -0.8f call-site argument
    // below) -- trees are planted AT ground level, unlike the showcase primitives above (which
    // deliberately float at the gallery's own y=0 "display stage" height, see GridSlot()'s header
    // comment).
    constexpr float kTreeGroundY = -0.8f;

    LOG_INFO(std::format("[GenerateGeometry] Generating {} procedural trees, one per species "
                         "(renderer::ProceduralTreePass, see kTreeSpecies)...",
                          kTreeVisualCount));

    // 10-tree-species scene: shape/material/placement all come from the kTreeSpecies recipe table
    // (see its own header comment above BuildEntityData() for the per-species silhouette design
    // and the river-avoiding grove layout) -- this loop only contributes what the table cannot
    // know at namespace scope: the per-tree deterministic seed, the registry-assigned mesh/
    // material IDs, and the grove's world-space anchor.
    for (uint32_t t = 0; t < kTreeVisualCount; ++t) {
      const TreeSpeciesRecipe& species = kTreeSpecies[t];
      uint32_t barkEntityIndex = kTreeEntityIndexBase + t * 2u;
      uint32_t leafEntityIndex = barkEntityIndex + 1u;

      renderer::ProceduralTreePass::TreeParams params{};
      // Deterministic per-tree seed (see tree_lsystem.glsl's own "fully deterministic" comment --
      // this demo must play back identically every run) -- an arbitrary odd multiplier spreads the
      // low tree index `t` across the hash's full bit range rather than leaving seeds that only
      // differ in their low bits.
      params.seed = 0x5EED0000u + t * 0x1000193u;
      params.depth = species.depth;
      params.branchFactor = species.branchFactor;
      params.trunkHeight = species.trunkHeight;
      params.trunkRadius = species.trunkRadius;
      params.lengthTaper = species.lengthTaper;
      params.radiusTaper = species.radiusTaper;
      params.branchAngleRadians = species.branchAngleRadians;
      params.pitchDamping = species.pitchDamping;
      params.sides = species.sides;
      params.leafSize = species.leafSize;

      // BuildEntityData() already assigned each bark/leaf entity its species materialID from this
      // SAME kTreeSpecies row (see its own tree block), so reading the registry back here keeps
      // the vertex-stamped materialID and the entity's own materialID in guaranteed agreement.
      params.barkMeshID = m_InstanceRegistry[barkEntityIndex].meshID;
      params.barkMaterialID = static_cast<float>(m_InstanceRegistry[barkEntityIndex].materialID);
      params.leafMeshID = m_InstanceRegistry[leafEntityIndex].meshID;
      params.leafMaterialID = static_cast<float>(m_InstanceRegistry[leafEntityIndex].materialID);

      params.worldOffsetX = kTreeClearingX + species.groveOffsetX;
      params.worldOffsetY = kTreeGroundY;
      params.worldOffsetZ = species.groveOffsetZ;

#ifndef NDEBUG
      renderer::ProceduralTreePass::GeometryFootprint fp =
          renderer::ProceduralTreePass::ComputeFootprint(params);
      LOG_INFO(std::format("[GenerateGeometry]   Tree {}/{} '{}': depth={} branchFactor={} at "
                           "({:.1f}, {:.1f}) -- {} bark verts + {} leaf verts.",
                           t + 1u, kTreeVisualCount, kTreeSpeciesNames[t], species.depth,
                           species.branchFactor, params.worldOffsetX, params.worldOffsetZ,
                           fp.barkVertexCount, fp.leafVertexCount));
#endif

      treePass.RecordGenerate(params, runningVertexOffset, runningIndexOffset);
    }

    treePass.Shutdown();
  }

  // -------------------------------------------------------------------------
  // SKELETAL-ANIMATION CREATURE (entity kCreatureEntityIndex) — a procedurally-generated,
  // procedurally-animated segmented "worm/snake" (16-bone single-chain skeleton, animation::
  // SkeletalAnimator), the one entity in the whole showcase carrying core::EntityFlags::
  // IsSkeletallyAnimated. Placed in its own small clearing on the opposite side of the gallery from
  // the trees (see kTreeClearingX's own comment for that precedent) so it reads as a distinct
  // feature area, resting directly on the terrain (worldOffsetY = floor top + radiusMax, so its
  // belly touches the ground -- its idle undulation only sways side-to-side in X/Z, never
  // vertically, so no further ground clearance is needed).
  // -------------------------------------------------------------------------
  if (!kMinimalSceneMode) {
    // Placement/radius constants now live as public VulkanContext class constants (kCreatureClearingX
    // /kCreatureGroundY/kCreatureRadiusMin/kCreatureRadiusMax) -- the single source of truth shared
    // with renderer::FurStrandPass (UE5.8 rendering-parity gap G10a), which grows fur strands off this
    // exact bind-pose surface. Only the two mesh-resolution knobs stay local here (the fur system
    // distributes its roots parametrically, independent of mesh resolution, so it never needs them).
    constexpr uint32_t kCreatureSidesPerRing = 8u;
    constexpr uint32_t kCreatureRingsPerSegment = 2u; // >1 so mid-segment vertices get a genuine 2-bone blend (see geom_creature.comp).

    maths::vec2 slot = { kCreatureClearingX, kCreatureClearingZ };
    GenerateCreature(m_InstanceRegistry[kCreatureEntityIndex].meshID, slot,
                      kCreatureGroundY + kCreatureRadiusMax,
                      kCreatureSidesPerRing, kCreatureRingsPerSegment,
                      kCreatureRadiusMin, kCreatureRadiusMax,
                      runningVertexOffset, runningIndexOffset);
  }

  // -------------------------------------------------------------------------
  // LUMEN/GI SHOWCASE CORNER (entities kWallEntityIndexA/B) — two static colored walls meeting at a right angle
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
  if (!kMinimalSceneMode) {
    constexpr float kWallSpan = 3.0f;     // Both baked axes; one becomes wall height after rotation.
    constexpr float kFloorTopY = -0.8f;   // Must match the floor's own worldOffsetY below.
    constexpr float kWallCenterY = kFloorTopY + kWallSpan * 0.5f; // Wall base sits exactly on the floor.

    // WALL A (entity kWallEntityIndexA) — red, vertical plane fixed at X = -1.8, spanning Z in [-1.5, 1.5].
    // Baked flat then stood up about the X axis... no -- about the Z axis (see
    // UpdateEntityRotations(): RotateZ maps the baked local-X extent onto world Y, leaving X and Z
    // pinned), which is why the wall's OWN world-space X position comes from `slot.x` below.
    {
      maths::vec2 slot = {-1.8f, 0.0f};
      GeneratePlane(cmd, kWallSpan, kWallSpan, m_InstanceRegistry[kWallEntityIndexA].meshID, slot,
                    runningVertexOffset, runningIndexOffset, kWallCenterY,
                    config::FLOOR_VERTEX_SPACING);
    }

    // WALL B (entity kWallEntityIndexB) — green, vertical plane fixed at Z = -1.8, spanning X in [-1.5, 1.5].
    // Stood up about the X axis instead (RotateX maps the baked local-Z extent onto world Y,
    // leaving X and Z pinned) -- perpendicular to Wall A, closing the corner.
    {
      maths::vec2 slot = {0.0f, -1.8f};
      GeneratePlane(cmd, kWallSpan, kWallSpan, m_InstanceRegistry[kWallEntityIndexB].meshID, slot,
                    runningVertexOffset, runningIndexOffset, kWallCenterY,
                    config::FLOOR_VERTEX_SPACING);
    }
  }

  // -------------------------------------------------------------------------
  // TERRAIN HEIGHTFIELD (entity kFloorEntityIndex) — Phase 7b (UE5.8 parity roadmap): 300m x 300m procedural
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
    // Minimal-scene mode: a flat GeneratePlane() instead of the eroded heightfield -- same
    // footprint/anchor/spacing as GenerateTerrain's own call below (kept for when
    // kMinimalSceneMode is flipped back off), see BuildEntityData()'s matching floor materialID
    // override for why this pairs with MaterialParameterTable.h slot 14, not kTerrainMaterialID.
    if (kMinimalSceneMode) {
      GeneratePlane(cmd, 300.0f, 300.0f, m_InstanceRegistry[kFloorEntityIndex].meshID, slot,
                    runningVertexOffset, runningIndexOffset, -0.8f,
                    kTerrainVertexSpacing);
    } else {
      GenerateTerrain(cmd, 300.0f, 300.0f, m_InstanceRegistry[kFloorEntityIndex].meshID, slot,
                    runningVertexOffset, runningIndexOffset, -0.8f,
                    kTerrainVertexSpacing);
    }
  }

  // -------------------------------------------------------------------------
  // WATER (entity kWaterEntityIndex) -- Phase 7c (UE5.8 parity roadmap, water/erosion): a flat plane
  // sized to the showcase zone-grid's own footprint (not the full 300x300 terrain -- the depth
  // test against the already-rasterized terrain naturally clips this flat quad to whatever low-
  // lying basin actually falls within it, so an oversized water plane would just waste fragment-
  // shader work on pixels the terrain always occludes). 2x2 segments (a single quad): wave
  // perturbation is per-fragment (WaterForward.frag), not per-vertex, so more segments would add
  // nothing -- see GenerateWaterPlane()'s own comment.
  //
  // Rivers/waterfalls feature: immediately CHAINS a second dispatch (GenerateRiver(), geom_river.
  // comp) onto this SAME meshID/entity, continuing runningVertexOffset/runningIndexOffset exactly
  // like the earlier BOX block's own 6 chained per-face dispatches -- the flowing-water ribbon
  // (river course + one waterfall segment, path authored once in river_spline.glsl) becomes part
  // of the SAME Fallback Mesh / EntityDrawRange / renderer::WaterForwardPass draw call as the flat
  // lake plane, needing no new renderer::EntityData slot, no ClusterRenderPipeline wiring, and no
  // new MaterialParameterTable entry -- see geom_river.comp's own header comment for the full
  // reasoning. 96x10 segments: long enough along the ~55-world-unit path (see river_spline.glsl's
  // own kRiverControlXZ) to resolve the sharp waterfall segment without the QEM-simplified
  // Fallback Mesh (geometry::BuildFallbackMesh, error-driven, not a fixed decimation ratio --
  // see that header's own comment) collapsing it away entirely, and wide enough across (10) for
  // the ribbon's own lateral wave/normal detail.
  // -------------------------------------------------------------------------
  {
    maths::vec2 slot = {0.0f, 0.0f}; // centered at the world origin, same as the terrain itself
    // Terrain hydrology feature: the old flat 24x24 lake quad (GenerateWaterPlane) is replaced by
    // the SIMULATED water surface -- sea + lakes + eroded river channels in one 600x600 grid
    // (kHydroWaterMeshSpan: wider than the terrain so the ocean extends toward the horizon), Y
    // sampled from the erosion bake's water-surface texture and anchored at the TERRAIN's own
    // worldOffsetY (-0.8, matching GenerateTerrain's call above -- the surface heights are
    // terrain-local). 128x128 segments (~4.7-unit spacing): enough to follow coastlines/lake
    // shores; the open sea between them is flat, which QEM simplifies for free. The authored
    // river ribbon (GenerateRiver below) still chains onto this same entity unchanged.
    //
    // Minimal-scene mode: shrunk to a 2x2-segment, near-zero-span quad rather than skipped
    // outright -- ClusterRenderPipeline.cpp's own kWaterEntityID hard-fails Init() (LOG_ERROR +
    // return false) if the water entity has no Fallback Mesh draw range at all, the same
    // kTessellatedEntityIndices-style hard requirement the PLANE (slot 3)/PYRAMID (slot 9) blocks
    // above already work around. GenerateRiver below is skipped, not shrunk: its ribbon is
    // authored directly in river_spline.glsl at a fixed ~55-world-unit path/world position, not
    // parameterized by this span, so shrinking the water surface alone already removes it from
    // view -- generating it would just be a full-size, unused mesh.
    constexpr float kWaterSurfaceSpan = kMinimalSceneMode ? 0.001f : 600.0f;   // Mirrors terrain_hydrology_params.glsl's kHydroWaterMeshSpan.
    constexpr uint32_t kWaterSurfaceSegments = kMinimalSceneMode ? 2u : 128u;
    constexpr float kTerrainAnchorY = -0.8f;      // Mirrors GenerateTerrain's own call-site worldOffsetY.
    GenerateWaterSurface(cmd, kWaterSurfaceSpan, kWaterSurfaceSegments,
                         m_InstanceRegistry[kWaterEntityIndex].meshID, slot,
                         kTerrainAnchorY, runningVertexOffset, runningIndexOffset);
    if (!kMinimalSceneMode) {
      GenerateRiver(m_InstanceRegistry[kWaterEntityIndex].meshID, 96u, 10u, runningVertexOffset, runningIndexOffset);
    }
  }

  // -------------------------------------------------------------------------
  // RUNTIME WORLD PARTITION STREAMING POOL (entities kStreamingSlotBase..kTotalEntityCount-1) --
  // see kStreamingUnitCount's own header comment for why streamed-in content is drawn from a small
  // fixed set of pre-baked shapes (live per-cell Nanite cluster DAG builds are not feasible on a
  // streaming budget) rather than unique per-cell geometry.
  //
  // Every unit's 2 slots are baked at their OWN distinct, widely-separated parking position (never
  // the origin, and never shared with another slot, see StreamingSlotParkPosition()) --
  // geom_autosmooth.comp's post-pass welds normals across ANY vertices within a small world-space
  // epsilon regardless of meshID (see that shader's own comment), so baking multiple archetypes on
  // top of each other at (0,0,0) would silently corrupt their normals. world::
  // WorldCellStreamingLoader's runtime translation (see struct_custo.glsl's EntityTransform
  // comment) cancels this SAME park offset back out before adding the desired absolute world
  // position (see SetStreamingUnitState()'s own comment for the double-offset bug this fixes) --
  // exactly the same rotation-pivot math every other entity already uses, see
  // UpdateEntityRotations()'s streaming-pool loop below -- so parking slots stay fully compatible
  // with being moved to an arbitrary cell position at runtime.
  //
  // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): each unit that has a corresponding authored
  // cell in m_CellManifest (unit index == that cell's index in GetOrderedCells(), see
  // m_CellToStreamingUnit's own comment) gets its COARSE slot baked from that cell's REAL,
  // offline-simplified HLOD proxy mesh (BakeHlodProxyIntoSlot()) instead of a generic box, and its
  // FINE slot baked from that SAME cell's real authored archetype shape (not an arbitrary unit%4
  // rotation) -- giving a genuine, visibly different triangle count between the two representations
  // of the SAME cell, per-cell, exactly as authored. Units with no corresponding cell (manifest
  // missing/unreadable, or this unit's index exceeds the authored cell count) fall back to the
  // pre-Phase-5 generic-box-coarse / unit%4-rotation-fine behavior unchanged.
  //
  // BUG FOUND + FIXED while growing kStreamingUnitCount 6 -> 50 for this feature: GenerateIcosphere/
  // GenerateSphere/GenerateCapsule/GenerateTorus/GenerateBox all read the GLOBAL, MUTABLE
  // config::VERTEX_SPACING directly (no per-call override, unlike GeneratePlane's own `spacing`
  // parameter) -- under the "Extrem" quality profile (config::EngineConfig_Extrem.h,
  // VERTEX_SPACING=0.02) a single fine archetype mesh tessellates into tens of thousands of
  // indices. geom_autosmooth.comp's post-pass (right after this block) is an O(totalVertexCount *
  // totalIndexCount) brute-force neighbor scan (see that shader's own comment) -- multiplying BOTH
  // dimensions by the streaming pool's own ~8.3x unit-count growth pushed its cost far enough past
  // what completes within the GPU driver's TDR window to hard-hang the process (confirmed
  // empirically: reproduced a "VulkanUtils: Failed to wait idle on queue for one-shot commands"
  // throw from this exact function at kStreamingUnitCount=50 with VERTEX_SPACING left at the active
  // profile's value). Streaming archetypes are small background props, never meant to track the
  // interactive showcase gallery's own hero-asset fidelity -- config::VERTEX_SPACING is temporarily
  // overridden to a small, FIXED, profile-independent value for this block only (saved/restored
  // around it), capping every fine mesh's triangle count to a small constant regardless of
  // kStreamingUnitCount or the active quality profile, exactly the same reasoning GeneratePlane's
  // own `spacing` parameter already documents for the analogous terrain-vs-hero-asset tradeoff.
  // -------------------------------------------------------------------------
  {
    const std::vector<world::CellCoord>& orderedCells = m_CellManifest.GetOrderedCells();
    m_CellToStreamingUnit.clear();

    const float savedVertexSpacing = config::VERTEX_SPACING;
    config::VERTEX_SPACING = 0.12f; // See this block's own header comment for why this must be fixed, not profile-dependent.

    // Startup-latency fix: BakeHlodProxyIntoSlot() used to be called inline in a single pass below,
    // each call opening its OWN one-shot command buffer and blocking the CPU on vkQueueWaitIdle
    // before returning -- up to kStreamingUnitCount (50) sequential blocking GPU round-trips just
    // for this block. Nothing in BakeHlodProxyIntoSlot()'s own logic actually depends on waiting
    // between calls: runningVertexOffset/runningIndexOffset are plain CPU-side counters threaded
    // through by reference (advanced synchronously, no GPU readback involved anywhere in this
    // function), and every call writes into a disjoint, CPU-computed sub-range of the same
    // fixed-size m_VertexBuffer/m_IndexBuffer SSBOs -- which sub-range lands where is functionally
    // irrelevant, since every vertex is tagged with its owning meshID and every downstream consumer
    // (BuildClusterDAG, the AUTOSMOOTH compute pass) selects/matches by that tag or by world-space
    // position, never by buffer-layout adjacency or call order. This first pass therefore precomputes
    // each unit's dedicated-cell lookup (populating m_CellToStreamingUnit exactly once per unit, same
    // as before) and attempts every qualifying unit's HLOD proxy bake, all recorded into ONE shared
    // command buffer, submitted and waited on exactly ONCE afterward -- not once per unit.
    //
    // The fallback plain-box coarse mesh and every fine-mesh archetype (GenerateBox/GenerateIcosphere/
    // GenerateSphere/GenerateCapsule/GenerateTorus, all DispatchGeometryCompute-driven) are
    // deliberately left as a SECOND pass below, each still its own blocking one-shot submit -- out of
    // scope for this fix (the exact same pattern is shared by every procedural primitive generator in
    // this entire function, not unique to the streaming pool, so batching it too would be a much
    // larger change than the specific staging-buffer/wait-idle stall this block targets) -- and that
    // second pass needs bakedRealCoarseProxy's final per-unit result anyway to decide whether to fall
    // back to a plain box.
    std::vector<std::optional<world::CellPlacement>> dedicatedPlacements(kStreamingUnitCount);
    std::vector<bool> bakedRealCoarseProxy(kStreamingUnitCount, false);
    {
      std::vector<std::pair<VkBuffer, VmaAllocation>> pendingStagingBuffers;
      renderer::VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_GraphicsQueue, [&](VkCommandBuffer cmd) {
        for (uint32_t unit = 0; unit < kStreamingUnitCount; ++unit) {
          if (unit < orderedCells.size()) {
            dedicatedPlacements[unit] = m_CellManifest.GetPlacement(orderedCells[unit]);
            if (dedicatedPlacements[unit].has_value()) {
              m_CellToStreamingUnit[orderedCells[unit]] = unit;
            }
          }

          const std::optional<world::CellPlacement>& dedicatedPlacement = dedicatedPlacements[unit];
          if (dedicatedPlacement.has_value() && dedicatedPlacement->hlodVertexCount > 0 && dedicatedPlacement->hlodIndexCount > 0) {
            uint32_t coarseIdx = StreamingUnitCoarseSlot(unit);
            maths::vec2 coarseSlot = StreamingSlotParkPosition(coarseIdx);
            bakedRealCoarseProxy[unit] = BakeHlodProxyIntoSlot(cmd, *dedicatedPlacement, m_InstanceRegistry[coarseIdx].meshID,
                                                                coarseSlot, runningVertexOffset, runningIndexOffset,
                                                                pendingStagingBuffers);
          }
        }

        // One shared transfer-write -> vertex/compute-shader-read visibility barrier for every
        // vkCmdCopyBuffer recorded by the loop above (both the vertex and index copy of every unit
        // that got a real HLOD bake this call) -- safe to cover them all with a single barrier since
        // they share the exact same hazard (transfer write into m_VertexBuffer/m_IndexBuffer vs. a
        // later vertex/compute-shader read of either), and every one of those writes is already
        // ordered before this point in submission order within this one command buffer. Matches
        // BakeHlodProxyIntoSlot()'s own former per-call barrier (same stage/access masks); skipped
        // entirely when nothing was baked this call (e.g. no manifest loaded), same as before.
        if (!pendingStagingBuffers.empty()) {
          VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
          memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
          memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
          memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
          memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

          VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
          depInfo.memoryBarrierCount = 1;
          depInfo.pMemoryBarriers = &memBarrier;
          vkCmdPipelineBarrier2(cmd, &depInfo);
        }
      });

      // Safe to destroy every staging buffer now: ExecuteOneShotCommands() above already blocked
      // until the GPU finished executing every vkCmdCopyBuffer that reads from them.
      for (auto& [stagingBuffer, stagingAllocation] : pendingStagingBuffers) {
        vmaDestroyBuffer(m_Allocator, stagingBuffer, stagingAllocation);
      }

      LOG_INFO(std::format("[GenerateGeometry] Streaming pool: baked {} cell(s)' real HLOD proxies "
                           "via ONE batched GPU submission (previously {} separate blocking "
                           "one-shot submits).",
                           pendingStagingBuffers.size() / 2u, pendingStagingBuffers.size() / 2u));
    }

    for (uint32_t unit = 0; unit < kStreamingUnitCount; ++unit) {
      const std::optional<world::CellPlacement>& dedicatedPlacement = dedicatedPlacements[unit];
      uint32_t shape = dedicatedPlacement.has_value()
          ? (dedicatedPlacement->archetypeShape % kStreamingArchetypeShapeCount)
          : (unit % kStreamingArchetypeShapeCount);

      uint32_t coarseIdx = StreamingUnitCoarseSlot(unit);
      maths::vec2 coarseSlot = StreamingSlotParkPosition(coarseIdx);

      if (!bakedRealCoarseProxy[unit]) {
        // Fallback: no manifest, this unit exceeds the authored cell count, this cell's HLOD proxy
        // was empty (BuildHlodForCell produced zero triangles -- degrades exactly like "no
        // authored content", see world::CellPlacement's own hlodIndexCount==0 comment), or the
        // manifest's own blob offsets failed BakeHlodProxyIntoSlot()'s bounds check -- the
        // pre-Phase-5 plain-box coarse mesh either way.
        GenerateBox(cmd, 0.6f, 0.6f, 0.6f, m_InstanceRegistry[coarseIdx].meshID, coarseSlot, runningVertexOffset, runningIndexOffset);
      }

      uint32_t fineIdx = StreamingUnitFineSlot(unit);
      maths::vec2 fineSlot = StreamingSlotParkPosition(fineIdx);
      switch (shape) {
        case 0: { // Rock: icosphere.
          uint32_t baseFaceCount = 20u, vertsPerFace = 0u;
          GenerateIcosphere(cmd, 0.5f, false, false, true, m_InstanceRegistry[fineIdx].meshID, fineSlot,
                             runningVertexOffset, runningIndexOffset, baseFaceCount, vertsPerFace);
          break;
        }
        case 1: // Bush: UV sphere.
          GenerateSphere(cmd, 0.5f, m_InstanceRegistry[fineIdx].meshID, fineSlot, runningVertexOffset, runningIndexOffset);
          break;
        case 2: // Tree: capsule (trunk + canopy silhouette stand-in).
          GenerateCapsule(cmd, 0.25f, 0.8f, m_InstanceRegistry[fineIdx].meshID, fineSlot, runningVertexOffset, runningIndexOffset);
          break;
        default: // Debris: torus (irregular scrap silhouette stand-in).
          GenerateTorus(cmd, 0.35f, 0.12f, 0.0f, 0.0f, m_InstanceRegistry[fineIdx].meshID, fineSlot, runningVertexOffset, runningIndexOffset);
          break;
      }
    }

    // Restore the active quality profile's own VERTEX_SPACING immediately -- see this block's own
    // header comment. Every OTHER GenerateXxx() call in this function (gallery primitives, walls,
    // floor, water) runs either before this block or is unaffected since streaming is the LAST
    // block GenerateGeometry() executes, but restoring defensively (rather than relying on that
    // ordering never changing) costs nothing and avoids a silent, hard-to-diagnose regression if a
    // future edit ever adds a primitive after this block.
    config::VERTEX_SPACING = savedVertexSpacing;

    LOG_INFO(std::format("[GenerateGeometry] Streaming pool: {}/{} units dedicated to real authored "
                         "cells with baked HLOD proxies (out of {} authored cells in the manifest).",
                         m_CellToStreamingUnit.size(), kStreamingUnitCount, orderedCells.size()));
  }

  m_TotalVertexCount = runningVertexOffset;
  m_TotalIndexCount = runningIndexOffset;

  // -------------------------------------------------------------------------
  // AUTOSMOOTH POST-PASS (autosmooth at 45.0 degrees) -- reads every vertex written by every call
  // above (all recorded into this SAME command buffer) -- see this function's own header comment
  // (point 2) for why it is safe to fold into this batch instead of needing its own submission.
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

    DispatchGeometryCompute(cmd, m_AutosmoothPipeline, m_ComputePipelineLayout,
                            &params, sizeof(params), nullptr, 0,
                            groupCount, 1, 1);
  }
  }); // end GALLERY + STREAMING POOL BATCH: single submit + single vkQueueWaitIdle for everything
      // recorded into `cmd` above, executed synchronously before ExecuteOneShotCommands() returns --
      // GenerateGeometry() (and therefore VulkanContext::Init(), which calls it last) still only
      // returns once every byte of procedural geometry is fully generated and GPU-visible, exactly
      // as before batching: RunVirtualGeometryCacheTest()'s ReadbackFullGeometry() (called from
      // main.cpp right after Init() returns) still finds the Vertex/Index SSBOs completely populated.

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

  LOG_INFO(std::format("[GenerateGeometry] All 12 primitives + {} procedural trees + 2 Lumen walls "
                       "+ floor + water generated: "
                       "totalVertexCount={} totalIndexCount={} "
                       "(buffers hold {} verts / {} indices max)",
                       kTreeVisualCount, runningVertexOffset, runningIndexOffset,
                       m_VertexBufferBytes / sizeof(renderer::Vertex),
                       m_IndexBufferBytes / sizeof(uint32_t)));
}

maths::vec2 VulkanContext::StreamingSlotParkPosition(uint32_t slotIndex) {
  // See this function's own header-comment (VulkanContext.h) for why this must be the ONE shared
  // implementation both GenerateGeometry() (bake time) and SetStreamingUnitState() (runtime, to
  // cancel this same offset back out) call -- these constants must never drift into two copies.
  constexpr float kParkSpacing = 3.0f;
  constexpr float kParkBaseX = -400.0f; // Far outside the showcase gallery (a few meters around the origin) and the streaming demo world itself (see BakeDemoWorld.cpp's kWorldCenterX).
  constexpr float kParkBaseZ = -400.0f;
  return maths::vec2{ kParkBaseX + static_cast<float>(slotIndex) * kParkSpacing, kParkBaseZ };
}

bool VulkanContext::BakeHlodProxyIntoSlot(VkCommandBuffer cmd, const world::CellPlacement &placement, uint32_t meshID,
                                          const maths::vec2 &worldOffset,
                                          uint32_t &runningVertexOffset, uint32_t &runningIndexOffset,
                                          std::vector<std::pair<VkBuffer, VmaAllocation>> &pendingStagingBuffers) {
  const std::vector<world::CellHlodVertex> &blobVerts = m_CellManifest.GetHlodVertices();
  const std::vector<uint32_t> &blobIndices = m_CellManifest.GetHlodIndices();

  // Bounds-check the manifest's own claimed ranges against the actual blob arrays before trusting
  // them -- see this function's own header comment (VulkanContext.h).
  if (static_cast<uint64_t>(placement.hlodVertexOffset) + placement.hlodVertexCount > blobVerts.size() ||
      static_cast<uint64_t>(placement.hlodIndexOffset) + placement.hlodIndexCount > blobIndices.size()) {
    LOG_ERROR("[GenerateGeometry] HLOD proxy record references an out-of-range blob slice -- "
             "skipping, falling back to the plain-box coarse mesh for this unit.");
    return false;
  }

  // Also guard the FIXED-size vertex/index SSBOs themselves -- same overflow contract
  // GenerateGeometry()'s own post-loop check enforces for the whole scene, checked per-slot here
  // too so one oversized manifest can't silently corrupt whatever geometry is generated after it in
  // this same function.
  const VkDeviceSize vertexBytesAfter = static_cast<VkDeviceSize>(runningVertexOffset + placement.hlodVertexCount) * sizeof(renderer::Vertex);
  const VkDeviceSize indexBytesAfter = static_cast<VkDeviceSize>(runningIndexOffset + placement.hlodIndexCount) * sizeof(uint32_t);
  if (vertexBytesAfter > m_VertexBufferBytes || indexBytesAfter > m_IndexBufferBytes) {
    LOG_ERROR("[GenerateGeometry] Baking this cell's HLOD proxy would overflow the fixed-size "
             "vertex/index SSBOs -- skipping, falling back to the plain-box coarse mesh for this unit.");
    return false;
  }

  // Build a temporary geometry::SimplifiableMesh purely to recompute per-vertex normals via
  // ComputeFaceAccumulatedNormals -- the manifest's blob deliberately stores positions+UVs only,
  // never normals (see RuntimeCellManifest.h's own header comment). `triangles` uses the SAME
  // cell-local 0-based indexing the manifest's own hlodIndexOffset/Count range already is (see
  // world::CellPlacement's own comment), so no rebase is needed for this local computation.
  geometry::SimplifiableMesh localMesh;
  localMesh.positions.reserve(placement.hlodVertexCount);
  for (uint32_t v = 0; v < placement.hlodVertexCount; ++v) {
    const world::CellHlodVertex &src = blobVerts[placement.hlodVertexOffset + v];
    localMesh.positions.push_back(maths::vec3{ src.x, src.y, src.z });
  }
  localMesh.triangles.reserve(placement.hlodIndexCount);
  for (uint32_t idx = 0; idx < placement.hlodIndexCount; ++idx) {
    localMesh.triangles.push_back(blobIndices[placement.hlodIndexOffset + idx]);
  }
  std::vector<maths::vec3> normals = geometry::ComputeFaceAccumulatedNormals(localMesh);

  // Assemble the CPU-side renderer::Vertex array (position rebased by worldOffset -- the SAME
  // "baked at the slot's own park position" convention every other streaming-slot generator in
  // this block already uses, see StreamingSlotParkPosition()'s own comment) ready for a straight
  // memcpy into the staging buffer.
  std::vector<renderer::Vertex> vertices(placement.hlodVertexCount);
  for (uint32_t v = 0; v < placement.hlodVertexCount; ++v) {
    const world::CellHlodVertex &src = blobVerts[placement.hlodVertexOffset + v];
    renderer::Vertex &dst = vertices[v];
    dst.position = maths::vec3{ src.x + worldOffset.x, src.y, src.z + worldOffset.y };
    dst.materialID = 0.0f; // Dead field -- see GenerateBox()'s own "params.materialID = 0.0f" precedent: actual material comes from EntityData::materialID via meshID indirection, never from this per-vertex field.
    dst.normal = normals[v];
    dst.meshID = meshID;
    dst.uv = maths::vec2{ src.u, src.v };
    dst.uv2 = maths::vec2{ src.u, src.v };
  }

  // Global (buffer-absolute) index array: the manifest's own indices are cell-local (0-based into
  // THIS record's own vertex range, see world::CellPlacement's own comment) -- rebase each one by
  // runningVertexOffset so it correctly addresses this slot's own destination range once copied.
  std::vector<uint32_t> indices(placement.hlodIndexCount);
  for (uint32_t idx = 0; idx < placement.hlodIndexCount; ++idx) {
    indices[idx] = blobIndices[placement.hlodIndexOffset + idx] + runningVertexOffset;
  }

  const VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(vertices.size()) * sizeof(renderer::Vertex);
  const VkDeviceSize indexBytes = static_cast<VkDeviceSize>(indices.size()) * sizeof(uint32_t);

  // --- Staging + vkCmdCopyBuffer, mirroring UploadEntityData()'s own staging-buffer setup (same
  // VMA_MEMORY_USAGE_CPU_ONLY + VMA_ALLOCATION_CREATE_MAPPED_BIT staging buffer). UNLIKE
  // UploadEntityData(), the copy is recorded into the CALLER's own `cmd` instead of a one-shot
  // command buffer this function opens/submits/waits on itself -- see this function's own header
  // comment (VulkanContext.h) for why: GenerateGeometry()'s streaming-pool block batches up to
  // kStreamingUnitCount calls into ONE shared submission instead of one blocking round-trip each.
  // The staging buffers are therefore NOT destroyed here either -- ownership moves to
  // `pendingStagingBuffers`; the caller destroys them only after its own shared submission's
  // vkQueueWaitIdle confirms the GPU has actually finished reading from them. ---
  VkBufferCreateInfo stagingVertexInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  stagingVertexInfo.size = vertexBytes;
  stagingVertexInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VmaAllocationCreateInfo stagingAllocInfo{};
  stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
  stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer stagingVertexBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingVertexAlloc = VK_NULL_HANDLE;
  VmaAllocationInfo stagingVertexAllocResult{};
  if (vmaCreateBuffer(m_Allocator, &stagingVertexInfo, &stagingAllocInfo, &stagingVertexBuffer,
                      &stagingVertexAlloc, &stagingVertexAllocResult) != VK_SUCCESS) {
    LOG_ERROR("[GenerateGeometry] Failed to allocate HLOD proxy vertex staging buffer!");
    return false;
  }
  std::memcpy(stagingVertexAllocResult.pMappedData, vertices.data(), static_cast<size_t>(vertexBytes));

  VkBufferCreateInfo stagingIndexInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  stagingIndexInfo.size = indexBytes;
  stagingIndexInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkBuffer stagingIndexBuffer = VK_NULL_HANDLE;
  VmaAllocation stagingIndexAlloc = VK_NULL_HANDLE;
  VmaAllocationInfo stagingIndexAllocResult{};
  if (vmaCreateBuffer(m_Allocator, &stagingIndexInfo, &stagingAllocInfo, &stagingIndexBuffer,
                      &stagingIndexAlloc, &stagingIndexAllocResult) != VK_SUCCESS) {
    LOG_ERROR("[GenerateGeometry] Failed to allocate HLOD proxy index staging buffer!");
    vmaDestroyBuffer(m_Allocator, stagingVertexBuffer, stagingVertexAlloc);
    return false;
  }
  std::memcpy(stagingIndexAllocResult.pMappedData, indices.data(), static_cast<size_t>(indexBytes));

  // Recorded into the caller's shared `cmd` -- NOT submitted, waited on, barriered, or cleaned up
  // here. The caller (GenerateGeometry()'s streaming-pool block) records one shared
  // VkMemoryBarrier2 after every unit's bake has been recorded this way (same transfer-write ->
  // vertex/compute-shader-read visibility this function used to barrier individually -- see that
  // block's own comment for why one barrier safely covers every copy: same hazard, same
  // destination buffers, all ordered before it within this one command buffer), then submits once
  // and destroys every staging buffer in `pendingStagingBuffers` only after that single
  // vkQueueWaitIdle confirms completion.
  VkBufferCopy vertexCopyRegion{};
  vertexCopyRegion.srcOffset = 0;
  vertexCopyRegion.dstOffset = static_cast<VkDeviceSize>(runningVertexOffset) * sizeof(renderer::Vertex);
  vertexCopyRegion.size = vertexBytes;
  vkCmdCopyBuffer(cmd, stagingVertexBuffer, m_VertexBuffer, 1, &vertexCopyRegion);

  VkBufferCopy indexCopyRegion{};
  indexCopyRegion.srcOffset = 0;
  indexCopyRegion.dstOffset = static_cast<VkDeviceSize>(runningIndexOffset) * sizeof(uint32_t);
  indexCopyRegion.size = indexBytes;
  vkCmdCopyBuffer(cmd, stagingIndexBuffer, m_IndexBuffer, 1, &indexCopyRegion);

  pendingStagingBuffers.emplace_back(stagingVertexBuffer, stagingVertexAlloc);
  pendingStagingBuffers.emplace_back(stagingIndexBuffer, stagingIndexAlloc);

  runningVertexOffset += placement.hlodVertexCount;
  runningIndexOffset += placement.hlodIndexCount;
  return true;
}

std::optional<uint32_t> VulkanContext::GetDedicatedStreamingUnitForCell(const world::CellCoord &coord) const {
  auto it = m_CellToStreamingUnit.find(coord);
  if (it == m_CellToStreamingUnit.end()) return std::nullopt;
  return it->second;
}

void VulkanContext::UpdateEntityRotations(float timeSeconds, const maths::vec3 &originOffset) {
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
    } else if (meshID == kCreatureEntityIndex) {
      // Skeletal-animation creature (renderer::ClusterRenderPipeline's own m_SkeletalAnimator):
      // static, no self-rotation here -- exactly like the procedural trees branch below, its full
      // world placement (kCreatureClearingX/kCreatureGroundY, see GenerateGeometry()'s own CREATURE
      // block) is already baked into restPos by geom_creature.comp's worldOffsetX/Y/Z push-constant
      // fields, and ALL of its visible motion comes from GPU-side skinning
      // (skeletal_animation.glsl's ApplySkeletalSkinning), not from this per-frame CPU rotation
      // matrix. With rotation == identity, `center + rotation*(restPos - center)` reduces to exactly
      // `restPos` regardless of center (same fact the floor/wallA/wallB/water/tree branches above
      // already rely on) -- this branch exists purely to keep GridSlot(meshID) (below, sized for
      // only the original 12 gallery primitives) from ever being called with kCreatureEntityIndex,
      // which is always well outside that [0,12) range.
      xform.rotation = maths::mat4{};
      xform.centerX = 0.0f;
      xform.centerY = 0.0f;
      xform.centerZ = 0.0f;
      xform._pad0 = 0.0f;
    } else if (meshID >= kTreeEntityIndexBase && meshID < kTreeEntityIndexBase + kTreeEntityCount) {
      // Procedural trees (renderer::ProceduralTreePass): static, no self-rotation (a real tree
      // doesn't spin like the showcase primitives below) -- `center`'s exact value is mathematically
      // irrelevant here: with rotation == identity, `center + rotation*(restPos - center)` reduces
      // to exactly `restPos` regardless of center (the identical fact this function's own floor/
      // wallA/wallB/water branches above already rely on), and geom_tree_bark.comp/
      // geom_tree_leaves.comp already bake each tree's full world placement directly into restPos
      // via their own worldOffsetX/Y/Z push-constant fields (see VulkanContext::GenerateGeometry()'s
      // own TREES block) -- so this branch exists purely to keep GridSlot(meshID) (below, sized for
      // only the original 12 gallery primitives) from ever being called with an out-of-range index.
      xform.rotation = maths::mat4{};
      xform.centerX = 0.0f;
      xform.centerY = 0.0f;
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

    // Phase 5 (Streaming & Monde roadmap, Part 1): rebase this entity into the current LWC origin
    // cell's frame by subtracting `originOffset` from the `translation` channel ONLY (never
    // `center` -- see this function's own declaration comment in VulkanContext.h for the full
    // worked-out reason). Every entity above leaves `translation` at its default zero (the "no
    // additional world-space offset" case struct_custo.glsl's own comment documents for every
    // original showcase/wall/floor/water entity), so this uniformly turns that zero into exactly
    // `-originOffset` -- the entire rebase, applied once, correct regardless of this entity's own
    // rotation.
    xform.translationX = -originOffset.x;
    xform.translationY = -originOffset.y;
    xform.translationZ = -originOffset.z;
    xform._pad1 = 0.0f;

    // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): CPU-readable mirror of
    // the SAME values just computed above for GPU upload -- zero extra computation, see
    // GetEntityTransformsCPU()'s own comment on why this exists. Phase 5: now includes the same
    // rebased `translation`, so every consumer of this CPU mirror (TLAS refit, Global SDF
    // object-space compositing) operates in the SAME current-frame LWC-rebased reference frame as
    // the rasterized VisBuffer pipeline -- one single "world space" notion per frame, never two
    // divergent ones (see renderer::ClusterRenderPipeline.h's own m_FrameScratch header comment for
    // why that single-choke-point property matters).
    m_InstanceRegistry.Transform(meshID) = core::EntityTransformCPU{
        xform.rotation, maths::vec3{xform.centerX, xform.centerY, xform.centerZ},
        maths::vec3{xform.translationX, xform.translationY, xform.translationZ}};
  }

  // --- Runtime World Partition streaming pool: static (no self-rotation), positioned entirely by
  // m_StreamingUnitTranslation (see SetStreamingUnitState()) -- center/rotation stay exactly as
  // baked (identity rotation, center == 0), so worldPos == translation + bakedPos (this class's own
  // EntityTransform comment). Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3) BUG FIX: indexed
  // per-SLOT now, NOT per-unit -- coarse and fine no longer share one translation value, see
  // m_StreamingUnitTranslation's own declaration-site comment for exactly why (each slot's baked
  // position already includes its own DIFFERENT park offset, which SetStreamingUnitState() must
  // cancel independently per slot). ---
  for (uint32_t unit = 0; unit < kStreamingUnitCount; ++unit) {
    for (uint32_t slotInUnit = 0; slotInUnit < kStreamingSlotsPerUnit; ++slotInUnit) {
      uint32_t i = kStreamingSlotBase + unit * kStreamingSlotsPerUnit + slotInUnit;
      const maths::vec3 &t = m_StreamingUnitTranslation[i - kStreamingSlotBase];
      EntityTransform &xform = transforms[i];
      xform.rotation = maths::mat4{};
      xform.centerX = 0.0f;
      xform.centerY = 0.0f;
      xform.centerZ = 0.0f;
      xform._pad0 = 0.0f;
      // Phase 5 (Streaming & Monde roadmap, Part 1): same rebase as the gallery loop above --
      // subtract `originOffset` from `translation` only (`center` is already 0 here, the streaming
      // pool's own local-origin-baked convention, see struct_custo.glsl's EntityTransform comment).
      xform.translationX = t.x - originOffset.x;
      xform.translationY = t.y - originOffset.y;
      xform.translationZ = t.z - originOffset.z;
      xform._pad1 = 0.0f;
      maths::vec3 rebasedTranslation{ xform.translationX, xform.translationY, xform.translationZ };
      m_InstanceRegistry.Transform(i) = core::EntityTransformCPU{ xform.rotation, maths::vec3{0.0f, 0.0f, 0.0f}, rebasedTranslation };
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

  uint32_t coarseIdx = StreamingUnitCoarseSlot(unit);
  uint32_t fineIdx = StreamingUnitFineSlot(unit);

  bool coarseActive = active && !useFineVariant;
  bool fineActive = active && useFineVariant;

  // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3) BUG FIX: translation must cancel THIS
  // slot's own baked park offset (StreamingSlotParkPosition()) before adding the desired absolute
  // world position, or the composed worldPos = translation + center + rotation*(restPos-center)
  // formula (center==0, rotation==identity here, so this collapses to translation + restPos)
  // double-counts the park offset -- the entity would render up to ~560 units away from where the
  // streaming system intended it (see m_StreamingUnitTranslation's own declaration-site comment for
  // the full derivation). Coarse and fine slots have DIFFERENT park positions (by design, so
  // GenerateGeometry()'s autosmooth post-pass never welds their normals together at bake time), so
  // each needs its own independently-canceled translation -- takes effect on the NEXT
  // UpdateEntityRotations() call (once per frame, see that function's own streaming-pool loop); no
  // GPU touch needed here, unlike EntityData below, because the whole transform buffer is already
  // re-uploaded wholesale every frame regardless.
  m_StreamingUnitTranslation[coarseIdx - kStreamingSlotBase] = coarseActive
      ? (worldPos - maths::vec3{ StreamingSlotParkPosition(coarseIdx).x, 0.0f, StreamingSlotParkPosition(coarseIdx).y })
      : maths::vec3{0.0f, 0.0f, 0.0f};
  m_StreamingUnitTranslation[fineIdx - kStreamingSlotBase] = fineActive
      ? (worldPos - maths::vec3{ StreamingSlotParkPosition(fineIdx).x, 0.0f, StreamingSlotParkPosition(fineIdx).y })
      : maths::vec3{0.0f, 0.0f, 0.0f};

  core::EntityData &coarse = m_InstanceRegistry[coarseIdx];
  core::SetFlag(coarse.flags, core::EntityFlags::StreamingInactive, !coarseActive);
  coarse.cellID = coarseActive ? cellID : 0u;

  core::EntityData &fine = m_InstanceRegistry[fineIdx];
  core::SetFlag(fine.flags, core::EntityFlags::StreamingInactive, !fineActive);
  fine.cellID = fineActive ? cellID : 0u;

  PatchStreamingUnitEntityData(unit);
}

void VulkanContext::PatchStreamingUnitEntityData(uint32_t unit) {
  // Purely CPU-side: m_InstanceRegistry[coarseIdx]/[fineIdx] were already updated in place by
  // SetStreamingUnitState() just above this call, so all that is needed here is to remember that
  // this unit's slice of m_EntityBuffer is now stale. FlushPendingEntityDataPatches() re-reads
  // m_InstanceRegistry fresh (never a snapshot) once per frame, so marking the same unit dirty more
  // than once before the next flush is harmless -- the flush only ever uploads the final, fully
  // up-to-date state.
  m_StreamingUnitEntityDataDirty[unit] = true;
}

void VulkanContext::FlushPendingEntityDataPatches(VkCommandBuffer cmd) {
  bool anyPatched = false;
  for (uint32_t unit = 0; unit < kStreamingUnitCount; ++unit) {
    if (!m_StreamingUnitEntityDataDirty[unit]) {
      continue;
    }
    m_StreamingUnitEntityDataDirty[unit] = false;
    anyPatched = true;

    uint32_t coarseIdx = StreamingUnitCoarseSlot(unit);
    // The 2 slots of a unit are always adjacent (see StreamingUnitCoarseSlot/StreamingUnitFineSlot),
    // so a single 32-byte vkCmdUpdateBuffer call covers both -- well under the 65536-byte inline-
    // update size limit, and both dstOffset and patchSize are always multiples of 4 (core::EntityData
    // is a flat 16-byte/4-uint32_t struct, see its own comment), satisfying vkCmdUpdateBuffer's
    // alignment requirement. Data is copied into the command buffer's own storage immediately at
    // record time -- the same well-established Vulkan semantics renderer::GpuGeometryPagePool::
    // FinalizeBoundPage() already relies on elsewhere in this codebase (it passes the address of a
    // local stack variable that goes out of scope right after the call returns) -- so pointing
    // straight at the live m_InstanceRegistry storage here is safe even though it may be overwritten
    // again before this command buffer actually executes on the GPU.
    VkDeviceSize patchSize = sizeof(core::EntityData) * kStreamingSlotsPerUnit;
    VkDeviceSize dstOffset = sizeof(core::EntityData) * coarseIdx;
    vkCmdUpdateBuffer(cmd, m_EntityBuffer, dstOffset, patchSize, &m_InstanceRegistry[coarseIdx]);
  }

  if (!anyPatched) {
    return;
  }

  // Same transfer-write -> vertex/compute-shader-read visibility barrier PatchStreamingUnitEntityData()
  // used to issue per-call (matching UploadEntityData()'s own dstStage/dstAccess for this same
  // buffer) -- srcStageMask changed from TRANSFER to CLEAR since vkCmdUpdateBuffer, unlike
  // vkCmdCopyBuffer, is classified under VK_PIPELINE_STAGE_2_CLEAR_BIT by the Vulkan spec's pipeline-
  // stage table, exactly like renderer::GpuGeometryPagePool's own FinalizeBoundPage()/UnbindPage()
  // page-table-entry updates (see those functions' own comments for the same CLEAR-not-TRANSFER
  // classification). One barrier covers every patch applied by the loop above, regardless of how many
  // units were dirty this frame (at most maxConcurrentLoads, see world::StreamingManager) -- cheaper
  // and just as correct as one barrier per patch, since they all guard the exact same hazard (transfer
  // write into m_EntityBuffer vs. a later vertex/compute shader read of it).
  VkMemoryBarrier2 memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
  memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
  memBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
  memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

  VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  depInfo.memoryBarrierCount = 1;
  depInfo.pMemoryBarriers = &memBarrier;
  vkCmdPipelineBarrier2(cmd, &depInfo);
}

#ifndef NDEBUG
bool VulkanContext::RunInstanceRegistrySmokeTest() {
  // Phase 0.1 (UE5.8-parity PCG roadmap, "Dynamic Instance Registry"): proves core::InstanceRegistry's
  // AcquireSlot()/ReleaseSlot() bookkeeping actually works end-to-end -- LIFO free-list reuse, correct
  // live-count/high-water-mark tracking -- WITHOUT ever touching (let alone corrupting) any of the
  // kTotalEntityCount real entities BuildEntityData() already acquired. Every probe slot this test
  // uses comes from m_InstanceRegistry's own kInstanceRegistryDebugHeadroom (its declaration-site
  // comment explains why that headroom exists), so even a bug in this test's own logic cannot alias a
  // live showcase/streaming-pool index. Called once from main.cpp right after VulkanContext::Init()
  // returns (i.e. after BuildEntityData() has already run). Whole function compiled out of Release.
  LOG_INFO("[VulkanContext] Running core::InstanceRegistry Acquire/Release smoke test...");

  // Phase 9.2 (test-pipeline integration roadmap): default to "ran, failed, generic pointer to the
  // log" up front -- every early `return false` below (there are several distinct checks) leaves
  // this default in place, so DebugTestPipeline::RunAll()'s later query always sees SOME real
  // result rather than a stale/empty struct, without needing per-branch instrumentation of every
  // individual check (matching this codebase's existing AudioEngine-smoke-test failure-reporting
  // convention -- see GetInstanceRegistrySmokeTestResult()'s own comment). Overwritten with the
  // real success details only at the very end, right before the final `return true`.
  m_InstanceRegistrySmokeTestResult = InstanceRegistrySmokeTestResult{
      /*ran=*/true, /*passed=*/false,
      /*details=*/"FAILED -- see demo_log.txt for the specific '[VulkanContext] InstanceRegistry "
                  "smoke test FAILED: ...' line logged during this run."
  };

  // Snapshot every currently-live entity's data before touching the registry at all, so any
  // corruption of the real showcase/streaming slots is caught by a plain field comparison at the end,
  // regardless of what the Acquire/Release calls below actually did internally.
  std::array<core::EntityData, kTotalEntityCount> beforeSnapshot{};
  std::copy_n(m_InstanceRegistry.Data(), kTotalEntityCount, beforeSnapshot.data());
  const uint32_t liveCountBefore = m_InstanceRegistry.GetLiveCount();
  const uint32_t highWaterMarkBefore = m_InstanceRegistry.GetHighWaterMark();

  if (highWaterMarkBefore != kTotalEntityCount) {
    LOG_ERROR(std::format(
        "[VulkanContext] InstanceRegistry smoke test FAILED: expected high-water mark == {} "
        "(BuildEntityData() should have already acquired exactly the showcase + streaming pool), got {}.",
        kTotalEntityCount, highWaterMarkBefore));
    return false;
  }

  // --- Acquire a handful of BRAND NEW slots, exercising the "grow into never-yet-used headroom"
  // path -- never touches/releases any of the kTotalEntityCount already-live entities. ---
  constexpr uint32_t kProbeCount = 3;
  static_assert(kProbeCount <= kInstanceRegistryDebugHeadroom,
      "RunInstanceRegistrySmokeTest needs at least kProbeCount slots of spare registry headroom.");

  using Registry = core::InstanceRegistry<kInstanceRegistryCapacity>;
  std::array<uint32_t, kProbeCount> probeIndices{};
  for (uint32_t p = 0; p < kProbeCount; ++p) {
    core::EntityData probe{};
    // Distinctive sentinel values, never used by real geometry (BuildEntityData() only ever
    // assigns meshIDs sequentially starting at 0 via core::IDManager).
    probe.meshID = 0xDEADBEEFu - p;
    probe.materialID = 0u;
    probe.cellID = 0u;
    probe.flags = 0u;

    uint32_t index = m_InstanceRegistry.AcquireSlot(probe);
    if (index == Registry::kInvalidSlot) {
      LOG_ERROR("[VulkanContext] InstanceRegistry smoke test FAILED: AcquireSlot() returned "
                "kInvalidSlot despite spare Debug headroom being available.");
      return false;
    }
    if (index < kTotalEntityCount) {
      LOG_ERROR(std::format(
          "[VulkanContext] InstanceRegistry smoke test FAILED: AcquireSlot() returned index {} "
          "which aliases a live showcase/streaming entity (< kTotalEntityCount={}).",
          index, kTotalEntityCount));
      return false;
    }
    if (m_InstanceRegistry[index].meshID != probe.meshID) {
      LOG_ERROR("[VulkanContext] InstanceRegistry smoke test FAILED: readback meshID does not "
                "match what AcquireSlot() was given.");
      return false;
    }
    probeIndices[p] = index;
  }

  if (m_InstanceRegistry.GetLiveCount() != liveCountBefore + kProbeCount) {
    LOG_ERROR("[VulkanContext] InstanceRegistry smoke test FAILED: GetLiveCount() did not "
              "increase by kProbeCount after acquiring the probe slots.");
    return false;
  }

  // --- Release every probe, then re-acquire the same count: the free list is LIFO, so this must
  // hand back the exact same indices in reverse-release order -- proving ReleaseSlot() actually
  // recycles capacity instead of leaking it. ---
  for (uint32_t p = kProbeCount; p-- > 0; ) {
    m_InstanceRegistry.ReleaseSlot(probeIndices[p]);
  }
  if (m_InstanceRegistry.GetLiveCount() != liveCountBefore) {
    LOG_ERROR("[VulkanContext] InstanceRegistry smoke test FAILED: GetLiveCount() did not return "
              "to its pre-probe value after releasing every probe slot.");
    return false;
  }

  // IMPORTANT: verify every re-acquired index FIRST, in its own complete pass, THEN release them
  // all in a second pass below -- releasing a probe immediately inside this same loop (before the
  // next iteration's AcquireSlot() call) would put that just-verified index back on top of the
  // free-list stack, so the very next AcquireSlot() would just hand it right back out again
  // instead of exercising the LIFO order this test is actually trying to prove.
  for (uint32_t p = 0; p < kProbeCount; ++p) {
    core::EntityData probe{};
    probe.meshID = 0xC0FFEEu + p;
    uint32_t index = m_InstanceRegistry.AcquireSlot(probe);
    if (index != probeIndices[p]) {
      LOG_ERROR(std::format(
          "[VulkanContext] InstanceRegistry smoke test FAILED: LIFO free-list re-acquire returned "
          "index {}, expected {} (release order was not preserved).",
          index, probeIndices[p]));
      return false;
    }
  }
  // Leave the registry in exactly the state BuildEntityData() left it: every probe released again
  // (release order does not matter here -- all kProbeCount are being freed, not re-verified).
  for (uint32_t p = 0; p < kProbeCount; ++p) {
    m_InstanceRegistry.ReleaseSlot(probeIndices[p]);
  }
  if (m_InstanceRegistry.GetLiveCount() != liveCountBefore) {
    LOG_ERROR("[VulkanContext] InstanceRegistry smoke test FAILED: GetLiveCount() did not return "
              "to its pre-probe value after the final cleanup release pass.");
    return false;
  }

  if (m_InstanceRegistry.GetHighWaterMark() != highWaterMarkBefore + kProbeCount) {
    // The high-water mark only ever grows (ReleaseSlot() recycles via the free list, it never
    // rewinds the mark -- see InstanceRegistry.h's own comment): it should have advanced by
    // exactly kProbeCount from the very first AcquireSlot() burst above, then stayed flat through
    // the LIFO release/re-acquire/release pass, which only ever reused those same free-listed
    // indices.
    LOG_ERROR(std::format(
        "[VulkanContext] InstanceRegistry smoke test FAILED: unexpected final high-water mark "
        "(expected {}, got {}).", highWaterMarkBefore + kProbeCount, m_InstanceRegistry.GetHighWaterMark()));
    return false;
  }

  // --- Final check: every one of the kTotalEntityCount real entities is byte-identical to the
  // snapshot taken before any of the above ran -- the actual "did not corrupt other entities" proof. ---
  for (uint32_t i = 0; i < kTotalEntityCount; ++i) {
    const core::EntityData& current = m_InstanceRegistry[i];
    const core::EntityData& snapshot = beforeSnapshot[i];
    if (current.meshID != snapshot.meshID || current.materialID != snapshot.materialID ||
        current.cellID != snapshot.cellID || current.flags != snapshot.flags) {
      LOG_ERROR(std::format(
          "[VulkanContext] InstanceRegistry smoke test FAILED: entity index {} was mutated by the "
          "Acquire/Release probe sequence (meshID {}->{}, materialID {}->{}, cellID {}->{}, flags {}->{}).",
          i, snapshot.meshID, current.meshID, snapshot.materialID, current.materialID,
          snapshot.cellID, current.cellID, snapshot.flags, current.flags));
      return false;
    }
  }

  std::string passMsg = std::format(
      "[VulkanContext] InstanceRegistry smoke test PASSED ({} probe slots acquired/released via "
      "LIFO free-list reuse in Debug headroom, all {} live entities unchanged).",
      kProbeCount, kTotalEntityCount);
  LOG_INFO(passMsg);
  m_InstanceRegistrySmokeTestResult.passed = true;
  m_InstanceRegistrySmokeTestResult.details = std::move(passMsg);
  return true;
}
#endif // NDEBUG

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
  if (m_VertexSkinBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_Allocator, m_VertexSkinBuffer, m_VertexSkinAllocation);
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
  if (m_AsyncComputeCanStartSemaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_Device, m_AsyncComputeCanStartSemaphore, nullptr);
    m_AsyncComputeCanStartSemaphore = VK_NULL_HANDLE;
  }
  if (m_AsyncComputeFinishedSemaphore != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_Device, m_AsyncComputeFinishedSemaphore, nullptr);
    m_AsyncComputeFinishedSemaphore = VK_NULL_HANDLE;
  }

  for (VkPipeline pipeline :
       {m_ConePipeline, m_IcospherePipeline, m_PlanePipeline, m_SpherePipeline,
        m_TorusPipeline, m_TubePipeline, m_CapsulePipeline, m_CylinderPipeline,
        m_PyramidPipeline, m_TorusKnotPipeline, m_ChamferBoxPipeline,
        m_TerrainPipeline, m_RiverPipeline, m_WaterSurfacePipeline,
        m_CreaturePipeline, m_AutosmoothPipeline}) {
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

  // Terrain hydrology feature: destroys the bake's images/pipeline/sampler -- must run while the
  // device and allocator are both still live (it uses both).
  m_TerrainHydrology.Shutdown();

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
  if (m_AsyncComputeCommandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(m_Device, m_AsyncComputeCommandPool, nullptr);
    m_AsyncComputeCommandPool = VK_NULL_HANDLE;
  }

  if (m_Allocator != VK_NULL_HANDLE) {
    vmaDestroyAllocator(m_Allocator);
  }

  // E1 (loading-time optimization): persist this run's final pipeline-cache contents -- captures
  // any pipeline created after main.cpp's own post-ClusterRenderPipeline::Init() save point (e.g. a
  // Debug-only pass constructed later) -- before the cache and device are destroyed. Must run before
  // vkDestroyDevice below: SavePipelineCache()'s vkGetPipelineCacheData call needs a live device.
  SavePipelineCache();
  // Clear the static accessor before destroying the handle it points at, so no dangling call
  // (there should be none left at shutdown, but this keeps VulkanPipeline::GetPipelineCache() safe
  // -- VK_NULL_HANDLE is always a legal "no cache" argument -- rather than a stale/destroyed one).
  VulkanPipeline::SetPipelineCache(VK_NULL_HANDLE);
  if (m_PipelineCache != VK_NULL_HANDLE) {
    vkDestroyPipelineCache(m_Device, m_PipelineCache, nullptr);
    m_PipelineCache = VK_NULL_HANDLE;
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
