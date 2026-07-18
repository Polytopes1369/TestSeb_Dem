#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>
#include "core/Camera.h"
#include "core/EngineConfig.h" // config::VERTEX_SPACING default arg for GeneratePlane()
#include "core/EntityData.h"
#include "core/IDManager.h"
#include "core/InstanceRegistry.h"
#include "renderer/MaterialParameterTable.h"
// Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): world::CellManifest/world::CellCoord --
// VulkanContext::GenerateGeometry() reads the same offline-baked world_data/cellmanifest.bin
// main.cpp's own world::CellManifest instance loads, to bake each authored cell's real HLOD proxy
// into a dedicated streaming unit at startup (see kStreamingUnitCount's own comment). Both headers
// are Vulkan-free and have zero dependency back onto renderer::, so this is a one-directional,
// acyclic addition.
#include "world/CellManifest.h"
#include <array>
#include <cassert>
#include <optional>
#include <string_view>
#include <unordered_map>
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
    // Phase 2 (Lumen advanced roadmap) fix: this frame's Nanite/GI work is now split across THREE
    // separate graphics-queue command buffers/submissions (previously one) so the async-compute
    // queue's Surface Cache TLAS-refit + radiosity injection can run genuinely CONCURRENTLY with
    // this queue's own Nanite VisBuffer culling/raster work, instead of only ever starting after
    // the entire graphics frame had already finished (a single vkQueueSubmit's signal semaphores
    // only fire once every command buffer in that submission completes -- see
    // renderer::ClusterRenderPipeline::RecordFrameEarly's own class-comment addendum for the full
    // root-cause explanation). All three are allocated from the SAME m_CommandPool (same graphics
    // queue family) -- same-queue submission order alone (no semaphore) is what makes cmdEarly's
    // trailing barriers visible to cmdMid, and cmdMid's to cmdLate, exactly like this codebase's
    // existing single-command-buffer intra-frame barriers already relied on same-queue ordering
    // across the [13] HZB rebuild -> next frame boundary.
    //   - GetCommandBufferEarly(): Virtual Shadow Map/Virtual Texture "begin frame", Surface Cache
    //     capture, Global SDF update, and (only when async compute is active this frame) the
    //     RELEASE half of the Surface Cache atlas/TLAS ownership transfer to the async-compute
    //     queue family. Submitted first, signals GetAsyncComputeCanStartSemaphore(), waits on
    //     nothing.
    //   - GetCommandBufferMid(): the Nanite VisBuffer pipeline (LOD selection/culling, HW+SW
    //     raster, resolve) plus this frame's geometry-streaming triage -- none of this samples the
    //     Surface Cache atlas/TLAS, so it needs no wait on the async-compute queue at all. It DOES
    //     wait on GetTransferFinishedSemaphore() (the geometry page pool's own queue-family-
    //     ownership acquire + decompression + raster reads run here, not in cmdEarly/cmdLate).
    //     Submitted second (same queue as cmdEarly, so implicitly ordered after it).
    //   - GetCommandBufferLate(): (only when async compute is active this frame) the ACQUIRE half
    //     of the Surface Cache ownership transfer back from the async-compute queue family, then
    //     every GI/reflection/forward pass that reads the Surface Cache atlas/TLAS this same frame
    //     (Reflection, World Probes, MegaLights' shadow rays, the 3 forward passes), post-process,
    //     and the final swapchain blit + present transition. Submitted third (same queue,
    //     implicitly ordered after cmdMid), additionally waits on GetImageAvailableSemaphore() and
    //     GetAsyncComputeFinishedSemaphore(), and is the ONE submission guarded by the per-frame
    //     frameFence (main.cpp) -- see that fence's own declaration-site comment for why cmdEarly/
    //     cmdMid need no fence of their own.
    VkCommandBuffer GetCommandBufferEarly() const { return m_CommandBuffer; }
    VkCommandBuffer GetCommandBufferMid() const { return m_CommandBufferMid; }
    VkCommandBuffer GetCommandBufferLate() const { return m_CommandBufferLate; }

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

    // Dedicated async-compute queue (Phase 2, Lumen advanced roadmap -- UE 5.8 RHI parity, mirrors
    // GetTransferQueue()'s own "prefer dedicated, fall back to graphics family" query exactly, just
    // searching for a family that advertises COMPUTE_BIT but NOT GRAPHICS_BIT instead of a
    // transfer-only family). Used by renderer::ClusterRenderPipeline to move
    // SurfaceCacheRayTracingPass::RecordRefreshTLAS + SurfaceCacheGIInjectPass::RecordInject's
    // radiosity bounce loop off the graphics queue -- see that class' own RecordFrame comment for
    // the full per-frame cross-queue sequencing contract. Falls back to the graphics queue/family
    // when the GPU exposes no distinct async-compute-capable family (e.g. Intel iGPUs, or any GPU
    // whose only non-graphics queue happens to also be transfer-only) -- HasDedicatedAsyncComputeQueue()
    // tells a caller whether queue-family-ownership-transfer barriers are actually needed (same
    // family == none needed), exactly like HasDedicatedTransferQueue() above.
    VkQueue GetAsyncComputeQueue() const { return m_AsyncComputeQueue; }
    uint32_t GetAsyncComputeQueueFamilyIndex() const { return m_AsyncComputeQueueFamilyIndex; }
    bool HasDedicatedAsyncComputeQueue() const { return m_HasDedicatedAsyncComputeQueue; }
    VkCommandBuffer GetAsyncComputeCommandBuffer() const { return m_AsyncComputeCommandBuffer; }
    // Bidirectional semaphore pair (unlike the transfer queue's one-way m_TransferFinishedSemaphore):
    // signaled by the GRAPHICS queue after this frame's Surface Cache capture writes the atlas
    // (releasing ownership of the 5 atlas images the async-compute work reads/writes), waited on by
    // the ASYNC-COMPUTE queue's own submission before it acquires them and starts TLAS refit + GI
    // injection. See main.cpp's per-frame submit sequence.
    VkSemaphore GetAsyncComputeCanStartSemaphore() const { return m_AsyncComputeCanStartSemaphore; }
    // Signaled by the ASYNC-COMPUTE queue after it releases m_Radiance + the TLAS back to the
    // graphics queue family, waited on by the GRAPHICS queue's own submission before any same-frame
    // consumer (Screen Trace / World Probes / GI Composite / Reflection) reads either. See main.cpp's
    // per-frame submit sequence.
    VkSemaphore GetAsyncComputeFinishedSemaphore() const { return m_AsyncComputeFinishedSemaphore; }

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
    //
    // Phase 5 (Streaming & Monde roadmap, Part 1): `originOffset` is the current LWC origin cell's
    // world-space center (world::LwcOrigin::GetCurrentOffset(), computed by main.cpp earlier this
    // same frame, after the fly-camera movement block so this call sees this frame's fresh origin,
    // never a stale one -- see main.cpp's own per-frame ordering comment). Subtracted from every
    // entity's `translation` channel ONLY, not `center` -- see this method's .cpp definition for the
    // exact worked-out reason `center` must stay untouched (it is also the rotation pivot inside
    // struct_custo.glsl's `rotation*(restPos-center)` term, baked against the immutable, always-
    // absolute GPU vertex buffer; rebasing it there would introduce a spurious rotation*offset error
    // term on every rotating entity, growing with the offset's own magnitude -- exactly the kind of
    // "close enough" approximation CLAUDE.md's zero-approximation rule forbids). `translation` is
    // already the pure world-space-additive-after-rotation channel struct_custo.glsl's own
    // composition comment documents, so subtracting there alone rebases the FINAL composed world
    // position by exactly `-originOffset`, correct regardless of any entity's current rotation.
    void UpdateEntityRotations(float timeSeconds, const maths::vec3& originOffset);

    // --- Accessors exposing the raw GPU handles needed by geometry::RunVirtualGeometryCacheTest
    // to read back the live procedural Vertex/Index SSBOs for the virtual geometry cache test.
    // Kept minimal (handles + counts only) so the geometry/ module never needs to include this
    // header's private implementation details.
    VmaAllocator GetAllocator() const { return m_Allocator; }
    VkCommandPool GetCommandPool() const { return m_CommandPool; }
    VkBuffer GetVertexBuffer() const { return m_VertexBuffer; }
    VkBuffer GetIndexBuffer() const { return m_IndexBuffer; }
    const core::EntityData* GetEntityData() const { return m_InstanceRegistry.Data(); }
    // Returns the TOTAL entity count including the streaming pool (kTotalEntityCount), not just
    // the fixed showcase gallery -- every existing "iterate every entity" consumer (GlobalSDFPass's
    // per-entity SDF bake, EntityBVH, TLAS build, shadow maps) picks up the streaming slots
    // automatically and for free this way, each slot getting a small, cheap, always-valid bake
    // (they are small pre-baked archetype props, not the scene's large primitives) rather than
    // needing special-casing in every one of those passes.
    uint32_t GetEntityCount() const { return kTotalEntityCount; }
    VkBuffer GetEntityTransformBuffer() const { return m_EntityTransformBuffer; }
    // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): CPU-readable mirror of
    // this frame's per-entity rotation, refreshed every UpdateEntityRotations() call alongside its
    // GPU SSBO upload. Consumed by renderer:: passes that need the ACTUAL rotation matrix (not just
    // a GPU buffer handle to bind) -- currently only SurfaceCacheRayTracingPass's per-frame TLAS
    // refit and GlobalSDFPass's object-space compositing (see core::EntityTransformCPU's own
    // comment for why this lives in EntityData.h, not here).
    const core::EntityTransformCPU* GetEntityTransformsCPU() const { return m_InstanceRegistry.TransformData(); }
    VkBuffer GetEntityBuffer() const { return m_EntityBuffer; }
    const renderer::MaterialTable& GetMaterialTable() const { return m_MaterialTable; }

    // --- Runtime World Partition streaming pool control (see kStreamingUnitCount's own comment).
    // Main-thread-only (patches m_EntityBuffer via a one-shot command submission -- see .cpp). ---
    uint32_t GetStreamingUnitCount() const { return kStreamingUnitCount; }

    // Activates (or deactivates) one streaming unit: `active` claims the unit for a world cell at
    // `worldPos`, showing its coarse mesh if `useFineVariant` is false or its fine mesh if true,
    // and stamping `cellID` into both slots' core::EntityData::cellID; `!active` parks the unit
    // (both slots marked core::EntityFlags::StreamingInactive, never drawn) for later reuse by a
    // different cell. Safe to call from the main thread only, at any point before this frame's
    // command buffer recording begins (see main.cpp's own call site, right after
    // StreamingManager::Update() and before ClusterRenderPipeline::RecordFrame()).
    void SetStreamingUnitState(uint32_t unit, bool active, bool useFineVariant, const maths::vec3& worldPos, uint32_t cellID);

    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): returns the streaming unit
    // GenerateGeometry() dedicated to `coord` at startup (real, baked HLOD proxy + fine archetype
    // mesh -- see that function's own streaming-pool bake-in block), or std::nullopt if this cell
    // has no dedicated unit (world_data/cellmanifest.bin was missing/unreadable at startup, this
    // cell is unauthored, or the manifest's authored-cell count exceeded kStreamingUnitCount's fixed
    // capacity). Callers (main.cpp's own streaming-activation loop) must fall back to the shared
    // free-list pool over the remaining spare units in that case -- see
    // GetDedicatedStreamingUnitCount()'s own comment for the exact fallback contract.
    std::optional<uint32_t> GetDedicatedStreamingUnitForCell(const world::CellCoord& coord) const;

    // Number of units GenerateGeometry() dedicated to a real authored cell at startup -- ALWAYS the
    // contiguous range [0, GetDedicatedStreamingUnitCount()) (unit index == that cell's index in
    // world::CellManifest::GetOrderedCells(), see that function's own bake-in loop), so a caller
    // building its own free-list over the REMAINING spare units (main.cpp's own
    // freeStreamingUnits) can simply start iterating at this value instead of at 0. Returns 0 if
    // world_data/cellmanifest.bin was missing/unreadable at startup (every unit then falls back to
    // the pre-Phase-5 shared 4-archetype rotation, exactly as before this feature).
    uint32_t GetDedicatedStreamingUnitCount() const { return static_cast<uint32_t>(m_CellToStreamingUnit.size()); }

    // Phase 0.2 (UE5.8-parity PCG roadmap, "PCG Instance Draw Path"): the real, already-baked
    // meshID/materialID of streaming unit `unit`'s FINE (full-detail) archetype variant -- lets a
    // caller (main.cpp's PcgInstanceDrawPass smoke test) point new PCG draw instances at already-
    // resident geometry without needing this class's own private streaming-slot layout constants
    // (StreamingUnitFineSlot/kStreamingSlotBase, both private -- see this class' own header comment
    // on why every existing "iterate every entity" consumer instead goes through GetEntityData()/
    // GetEntityCount()). `unit` must be < GetStreamingUnitCount(). Valid any time after Init() has
    // run BuildEntityData() -- every streaming unit's fine-variant EntityData is authored there
    // unconditionally, regardless of whether any world cell has actually claimed the unit yet (see
    // SetStreamingUnitState()'s own comment: an idle unit still carries a real, valid meshID, only
    // core::EntityFlags::StreamingInactive hides it from being drawn by the normal per-frame cut).
    struct StreamingArchetypeMeshInfo { uint32_t meshID = 0; uint32_t materialID = 0; };
    StreamingArchetypeMeshInfo GetStreamingArchetypeFineMeshInfo(uint32_t unit) const {
        assert(unit < kStreamingUnitCount);
        uint32_t fineIdx = StreamingUnitFineSlot(unit);
        return { m_InstanceRegistry[fineIdx].meshID, m_InstanceRegistry[fineIdx].materialID };
    }

#ifndef NDEBUG
    // Phase 0.1 (UE5.8-parity PCG roadmap, "Dynamic Instance Registry"): CPU-only startup smoke test
    // proving core::InstanceRegistry::AcquireSlot()/ReleaseSlot() free-list bookkeeping is correct
    // and cannot corrupt any of the kTotalEntityCount already-live entities BuildEntityData()
    // acquired at Init() time. Whole declaration + definition compiled out in Release (matching this
    // class's existing m_DebugMessenger/m_EnableValidationLayers convention below) -- see the .cpp
    // definition for exactly what it checks. Returns false (after LOG_ERROR-ing the specific failing
    // check) on the first failing check, true if every check passes. Safe to call any time after
    // Init() has run BuildEntityData() -- main.cpp calls it exactly once, right after
    // VulkanContext::Init() returns.
    bool RunInstanceRegistrySmokeTest();
#endif

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
    VkCommandBuffer m_CommandBuffer = VK_NULL_HANDLE; // "cmdEarly" -- see GetCommandBufferEarly()'s own comment.
    // "cmdMid"/"cmdLate" -- see GetCommandBufferMid()/GetCommandBufferLate()'s own comment. Same
    // pool as m_CommandBuffer above (same graphics queue family), just 2 more primary command
    // buffers allocated out of it.
    VkCommandBuffer m_CommandBufferMid = VK_NULL_HANDLE;
    VkCommandBuffer m_CommandBufferLate = VK_NULL_HANDLE;

    // Dedicated transfer queue (or a fallback alias of m_GraphicsQueue/m_GraphicsQueueFamilyIndex)
    // -- see GetTransferQueue()'s own comment.
    VkQueue m_TransferQueue = VK_NULL_HANDLE;
    uint32_t m_TransferQueueFamilyIndex = 0;
    bool m_HasDedicatedTransferQueue = false;
    VkCommandPool m_TransferCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_TransferCommandBuffer = VK_NULL_HANDLE;
    VkSemaphore m_TransferFinishedSemaphore = VK_NULL_HANDLE;

    // Dedicated async-compute queue (or a fallback alias of m_GraphicsQueue/m_GraphicsQueueFamilyIndex)
    // -- see GetAsyncComputeQueue()'s own comment.
    VkQueue m_AsyncComputeQueue = VK_NULL_HANDLE;
    uint32_t m_AsyncComputeQueueFamilyIndex = 0;
    bool m_HasDedicatedAsyncComputeQueue = false;
    VkCommandPool m_AsyncComputeCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_AsyncComputeCommandBuffer = VK_NULL_HANDLE;
    VkSemaphore m_AsyncComputeCanStartSemaphore = VK_NULL_HANDLE;
    VkSemaphore m_AsyncComputeFinishedSemaphore = VK_NULL_HANDLE;

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
    // Phase 7b (UE5.8 parity roadmap, terrain heightfield): geom_terrain.comp -- same shared
    // Params UBO / DispatchGeometryCompute path as every other non-box primitive above (in fact
    // byte-identical to PlaneParams, see GenerateTerrain()'s own comment).
    VkPipeline m_TerrainPipeline = VK_NULL_HANDLE;
    // Rivers/waterfalls feature: geom_river.comp -- same shared Params UBO / DispatchGeometryCompute
    // path, own RiverParams struct (see GenerateRiver()'s own comment).
    VkPipeline m_RiverPipeline = VK_NULL_HANDLE;

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

    // One EntityTransform slot per entity (box=0 .. chamferBox=11, wallA=12, wallB=13, floor=14,
    // water=15 since Phase 7c); see struct_custo.glsl. The base scene is a feature "gallery": 12
    // primitives laid out as 9 widely-separated zones (see GridSlot()) each demonstrating one
    // engine feature explicitly (Nanite density, WPO displacement, metal, dielectric, glass,
    // translucent, emissive, MegaLights, Lumen GI), plus 2 static colored walls forming the Lumen
    // GI corner, the floor (a Phase 7b procedural terrain heightfield, not a flat plane), and a
    // Phase 7c water plane.
    // Procedural tree generator (renderer::ProceduralTreePass -- CLAUDE.md's "Arbres (generes par
    // du code style speedtree)" requirement): kTreeVisualCount distinct trees, each baked as TWO
    // entities (bark + leaves, see ProceduralTreePass.h's own class comment for why one entity
    // can't hold both materials), inserted right after the 12 gallery primitives (indices
    // [kTreeEntityIndexBase, kTreeEntityIndexBase + kTreeEntityCount)). The walls/floor/water block
    // below is keyed off kEntityCount's OWN END (kEntityCount - 4/3/2/1), so growing kEntityCount
    // to make room for the tree entities automatically shifts those to the new end without any
    // other change -- exactly the same "keyed by absolute index, not distance from the start"
    // mechanism GenerateShowcaseMaterialTable()/GridSlot() already rely on for the first 12.
    static constexpr uint32_t kTreeVisualCount = 4;
    static constexpr uint32_t kTreeEntityIndexBase = 12;
    static constexpr uint32_t kTreeEntityCount = kTreeVisualCount * 2u; // bark + leaves per tree.

    static constexpr uint32_t kEntityCount = 16 + kTreeEntityCount;
    // The 2 static walls that form the Lumen/GI showcase corner (see GenerateGeometry()'s wall
    // blocks and UpdateEntityRotations()'s fixed-rotation branch for them) -- generated right
    // before the floor. Deliberately offset from kEntityCount by 4/3 (not 3/2, as before Phase 7c)
    // so adding the water plane as the new last entity does not shift these -- both stay at
    // kEntityCount-4/kEntityCount-3 (12/13 before the tree entities above were inserted, now
    // shifted to 20/21), since GenerateShowcaseMaterialTable()'s own per-slot recipes are keyed by
    // an EXPLICIT materialID override at those two entity indices (see BuildEntityData()'s own
    // kWallMaterialIDA/B constants), not by an implicit materialID==entityIndex assumption.
    static constexpr uint32_t kWallEntityIndexA = kEntityCount - 4;
    static constexpr uint32_t kWallEntityIndexB = kEntityCount - 3;
    // The floor (a Phase 7b terrain heightfield) is generated right before the water plane -- see
    // this constant's own "kEntityCount - 4/3" sibling comment above for why this is offset by 2,
    // not 1, now that water is the new last entity.
    static constexpr uint32_t kFloorEntityIndex = kEntityCount - 2;
    // Phase 7c (UE5.8 parity roadmap, water/erosion): the water plane, always generated last (see
    // GenerateGeometry()'s own GenerateWaterPlane() call) -- rendered by renderer::
    // WaterForwardPass instead of the opaque Nanite path, same core::EntityFlags::IsTransparent
    // exclusion mechanism as the hero entity (see BuildEntityData()'s own kWaterMaterialID
    // override).
    static constexpr uint32_t kWaterEntityIndex = kEntityCount - 1;
    // Phase 7a (UE5.8 parity roadmap, hero asset tessellation): the Icosphere -- generated FIRST
    // (see GenerateGeometry()'s own "Icosphere-first" comment, `m_InstanceRegistry[2].meshID` used
    // directly). Originally the ONE hardcoded tessellated/displaced hero asset (back when this
    // feature only ever rendered renderer::kHeroMaterialID via the single-entity
    // "HeroTessellationPass"); still tessellated today, but now just one entry in
    // kTessellatedEntityIndices below, same as any other opted-in entity -- see that constant's
    // own comment for the generalization.
    static constexpr uint32_t kHeroEntityIndex = 2;

    // Generalized Nanite Tessellation (renderer::TessellationPass, real UE5.8 Nanite Tessellation
    // parity -- 5.5+ applies to any flagged mesh, not one hardcoded hero asset): every entity index
    // BuildEntityData() marks core::EntityFlags::IsTessellated, rendered by renderer::
    // TessellationPass's own screen-space-adaptive tessellation + procedural displacement instead
    // of the opaque Nanite VisBuffer pipeline (see that class' own header comment). Deliberately
    // chosen to be VISIBLY different from the pre-generalization single-hero-sphere look:
    // kHeroEntityIndex (2, Icosphere -- already tessellated before this generalization, kept for
    // continuity), slot 3 (Plane, "Dielectric A" -- originally a perfectly flat surface; tessellated
    // displacement turns it into a rocky/eroded ground patch, the clearest possible before/after
    // demonstration of this feature), and slot 9 (Pyramid, "Dielectric B" -- a simple flat-faced
    // primitive that likewise reads very differently once its faces are subdivided and displaced).
    // Every one of these keeps its own showcase materialID (see GenerateShowcaseMaterialTable()'s
    // own zone-layout comment) unmodified except the hero (still overridden to kHeroMaterialID,
    // exactly as before) -- renderer::TessellationPass now shades each entity with ITS OWN
    // material, not a single shared one (see BuildEntityData()'s own IsTessellated assignment for
    // the exact override rules).
    static constexpr std::array<uint32_t, 3> kTessellatedEntityIndices = { kHeroEntityIndex, 3u, 9u };

    // --- Runtime World Partition streaming pool (world::StreamingManager / world::WorldCellStreamingLoader) ---
    // Bounded pool of extra entity slots appended AFTER the fixed showcase gallery (indices
    // [kEntityCount, kTotalEntityCount)) -- every constant above (kWallEntityIndexA/B,
    // kFloorEntityIndex, kWaterEntityIndex, kHeroEntityIndex) stays keyed off kEntityCount (16)
    // unchanged, so this pool can never collide with or shift any existing showcase entity index.
    //
    // Each "streaming unit" is 2 physical slots (a coarse + a fine pre-baked mesh variant of the
    // same archetype shape, both baked at local origin -- see GenerateGeometry()'s streaming block)
    // sharing one world position: HLOD <-> FullDetail transitions are a flag swap between the two,
    // never a geometry re-bake, because re-writing baked vertex data after the Nanite cluster DAG
    // has been built from it would desync the DAG's cluster bounds from the raw vertices it indexes
    // (see ClusterLODCompact.comp's own boundsMin/boundsMax comment) -- live re-baking of resident
    // Nanite geometry is not a supported operation in this pipeline.
    static constexpr uint32_t kStreamingArchetypeShapeCount = 4; // Rock, Bush, Tree, Debris -- see BuildEntityData()'s archetype block.
    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): grown from 6 (a small shared-archetype
    // rotation pool) to 50, so GenerateGeometry() can dedicate one real, individually-baked unit to
    // every one of BakeDemoWorld.cpp's 49 authored demo-grid cells (7x7, see that tool's own
    // kGridRadiusCells) with one spare unit left over for the shared-archetype fallback pool (see
    // main.cpp's own freeStreamingUnits comment). Ceiling verified, not guessed: kEntityCount (16) +
    // kStreamingUnitCount*kStreamingSlotsPerUnit must stay <= SurfaceCacheTraceContext::
    // kMaxTracedEntities (128, mesh_sdf_trace.glsl's own compile-time array size -- exceeding it
    // doesn't crash, GlobalSDFPass/SurfaceCacheTraceContext just silently truncate the overflow
    // entities out of GI/SDF tracing, see that constant's own comment) -- 16 + 50*2 = 116, leaving
    // 12 of headroom. A prior attempt at 64 units (16 + 64*2 = 144) confirmed this ceiling by
    // exceeding it by exactly 16 entities.
    static constexpr uint32_t kStreamingUnitCount = 50;
    static constexpr uint32_t kStreamingSlotsPerUnit = 2; // 0 = coarse (HLOD), 1 = fine (FullDetail).
    static constexpr uint32_t kStreamingSlotCount = kStreamingUnitCount * kStreamingSlotsPerUnit;
    static constexpr uint32_t kStreamingSlotBase = kEntityCount;
    static constexpr uint32_t kTotalEntityCount = kEntityCount + kStreamingSlotCount;

    static constexpr uint32_t StreamingUnitCoarseSlot(uint32_t unit) { return kStreamingSlotBase + unit * kStreamingSlotsPerUnit; }
    static constexpr uint32_t StreamingUnitFineSlot(uint32_t unit) { return kStreamingSlotBase + unit * kStreamingSlotsPerUnit + 1u; }

    // Phase 0.1 (UE5.8-parity PCG roadmap, "Dynamic Instance Registry"): extra core::InstanceRegistry
    // capacity, beyond kTotalEntityCount, reserved ONLY for RunInstanceRegistrySmokeTest()'s own
    // probe slots (see that method's own comment) -- lets the smoke test AcquireSlot() genuinely
    // brand-new indices instead of borrowing and restoring one of the kTotalEntityCount already-live
    // real entities, which would risk corrupting it if the restore step ever had a bug. Zero in
    // Release (#ifndef NDEBUG) so this headroom -- and the smoke test that is its only consumer --
    // costs nothing in the shipping executable, matching this header's existing m_DebugMessenger/
    // m_EnableValidationLayers convention below.
#ifndef NDEBUG
    static constexpr uint32_t kInstanceRegistryDebugHeadroom = 4;
#else
    static constexpr uint32_t kInstanceRegistryDebugHeadroom = 0;
#endif
    static constexpr uint32_t kInstanceRegistryCapacity = kTotalEntityCount + kInstanceRegistryDebugHeadroom;

    // CPU-authoritative entity records + their CPU-side transform mirror. Generalized (Phase 0.1)
    // from a fixed std::array<EntityData, kTotalEntityCount> into a core::InstanceRegistry: a
    // capacity-bounded pool supporting runtime AcquireSlot()/ReleaseSlot() with LIFO free-list reuse
    // (see InstanceRegistry.h's own header comment). BuildEntityData() acquires exactly
    // kTotalEntityCount slots once at startup -- the 16 fixed showcase entities (indices
    // [0, kEntityCount)) then the streaming pool (indices [kStreamingSlotBase, kTotalEntityCount)) --
    // in that exact deterministic order on a registry that starts completely empty, so AcquireSlot()
    // is guaranteed to hand back index == the loop's own absolute index every time, preserving every
    // existing absolute-index constant (kWallEntityIndexA/B, kFloorEntityIndex, kWaterEntityIndex,
    // kHeroEntityIndex, GridSlot()'s own zone layout) completely unchanged, and never releases them.
    // Every existing consumer that iterates by GetEntityCount() (== kTotalEntityCount, unchanged)
    // keeps working completely unmodified, since none of them can observe the registry's own extra
    // Debug-only headroom capacity above that count. Data()/TransformData() give the exact same
    // contiguous-array pointer semantics GetEntityData()/GetEntityTransformsCPU() always returned.
    core::InstanceRegistry<kInstanceRegistryCapacity> m_InstanceRegistry;

    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3) BUG FIX: per-SLOT (not per-unit as before
    // this fix), indexed by physical slot (StreamingUnitCoarseSlot(unit)/StreamingUnitFineSlot(unit)
    // minus kStreamingSlotBase). struct_custo.glsl's EntityTransform composition is
    // `worldPos = translation + center + rotation*(restPos - center)`; with center == 0 and
    // rotation == identity for every streaming slot (see UpdateEntityRotations()'s own streaming-
    // pool loop), this collapses to `worldPos = translation + restPos`, and restPos already bakes in
    // this slot's own unique park offset (StreamingSlotParkPosition() -- GenerateGeometry()'s
    // streaming block bakes every slot at its own parking position, never the origin, see that
    // block's own header comment on why coarse/fine must never share one). A single shared
    // per-UNIT translation set to the raw desired world position (the pre-fix behavior) therefore
    // silently double-counted that park offset, rendering an active streaming slot ~560 units away
    // from its intended cell -- SetStreamingUnitState() now subtracts EACH slot's own
    // StreamingSlotParkPosition() before storing here, and coarse/fine need independent storage
    // because their park positions differ (by kParkSpacing, see that function's own local
    // constants) even though they represent the same unit/cell. Set by SetStreamingUnitState() and
    // consumed every frame by UpdateEntityRotations() -- persists across frames since
    // UpdateEntityRotations() rebuilds the whole upload array from scratch every call and has no
    // other memory of where a streaming slot was last placed.
    std::array<maths::vec3, kStreamingSlotCount> m_StreamingUnitTranslation{};

    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): the manifest GenerateGeometry() reads once
    // at startup to bake real per-cell HLOD proxies into the streaming pool (see that function's own
    // streaming-pool block) -- kept alive as a member (not a Init()-local) only because
    // BuildEntityData() ALSO needs it (to assign each dedicated unit's materialID/shape consistently
    // with what GenerateGeometry() later bakes for that SAME unit) and runs before GenerateGeometry()
    // in Init()'s own call order. Small enough to keep resident for the process's whole lifetime
    // without concern (see world::CellManifest's own header comment on why reading it wholesale up
    // front is fine at this demo's scale).
    world::CellManifest m_CellManifest;

    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): built once by GenerateGeometry()'s
    // streaming-pool block, ALWAYS mapping to the contiguous range [0, GetDedicatedStreamingUnitCount())
    // (unit index == that cell's index in m_CellManifest.GetOrderedCells()) -- backs
    // GetDedicatedStreamingUnitForCell()/GetDedicatedStreamingUnitCount() above. Empty if
    // world_data/cellmanifest.bin was missing/unreadable at startup.
    std::unordered_map<world::CellCoord, uint32_t, world::CellCoordHash> m_CellToStreamingUnit;

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

    // Phase 7b (UE5.8 parity roadmap, terrain heightfield): dispatches geom_terrain.comp. Same
    // signature shape as GeneratePlane (reuses its own PlaneParams struct -- geom_terrain.comp's
    // Params UBO is byte-identical) since the terrain entity replaces what used to be a flat
    // GeneratePlane() call at the floor slot -- see GenerateGeometry()'s own floor block.
    void GenerateTerrain(
        float Length, float Width,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset,
        float worldOffsetY, float spacing);

    // Phase 7c (UE5.8 parity roadmap, water/erosion): dispatches m_PlanePipeline (geom_plane.comp),
    // NOT m_TerrainPipeline -- unlike GenerateTerrain, worldOffsetY here is a FIXED water-level Y,
    // not an addition on top of a sampled height. Zero new shader/pipeline: reuses the exact
    // pipeline already created for GeneratePlane's own dielectric-plane entity.
    void GenerateWaterPlane(
        float Width, float Length, uint32_t WidthSegments, uint32_t LengthSegments,
        uint32_t meshID, maths::vec2 slot, float worldOffsetY,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    void GenerateCapsule(
        float Radius, float Height,
        uint32_t meshID, maths::vec2 slot,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    // Rivers/waterfalls feature: dispatches m_RiverPipeline (geom_river.comp), generating the
    // spline-following water ribbon (river course + waterfall segment, see river_spline.glsl's own
    // kRiverControlHeight comment) CHAINED onto the same `meshID`/`runningVertexOffset`/
    // `runningIndexOffset` as an immediately-preceding GenerateWaterPlane() call for the flat lake
    // quad -- deliberately NOT a new renderer::EntityData slot (see geom_river.comp's own header
    // comment for why: one shared renderer::WaterForwardPass draw, one shared kWaterMaterialID,
    // zero VulkanContext::kEntityCount ripple). `segmentsAlong`/`segmentsAcross` size the ribbon's
    // generation grid; the actual world-space path/width come from river_spline.glsl's own
    // constants (kRiverControlXZ/kRiverControlHeight/kRiverHalfWidth), not from parameters here --
    // unlike every other Generate*() primitive, this shape's authoring lives in the shared GLSL
    // include specifically so terrain_noise.glsl's channel carve and this mesh generator can never
    // silently drift apart (see that file's own header comment for the full 3-consumer contract).
    void GenerateRiver(
        uint32_t meshID, uint32_t segmentsAlong, uint32_t segmentsAcross,
        uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

    // Authors m_InstanceRegistry's entity records on the CPU: assigns each of the kEntityCount
    // entities a meshID via core::IDManager::GetNextID() (instead of a hardcoded literal), then
    // claims its slot via core::InstanceRegistry::AcquireSlot(). Must run before
    // GenerateGeometry(), which reads m_InstanceRegistry[i].meshID into each primitive's push
    // constants / Params UBO.
    void BuildEntityData();

    // One-shot upload of m_InstanceRegistry's backing store to the GPU-only m_EntityBuffer via a
    // temporary staging buffer + vkCmdCopyBuffer, matching DispatchGeometryCompute's blocking
    // one-time-submit pattern. Must run after m_EntityBuffer is allocated and before it is bound
    // for reading.
    void UploadEntityData();

    // Patches exactly the 2 core::EntityData elements at [kStreamingSlotBase + unit*2, +2) into
    // m_EntityBuffer via a small one-shot staging-buffer copy (same pattern as UploadEntityData(),
    // just a 2-element slice instead of the whole array) -- called by SetStreamingUnitState() after
    // updating m_InstanceRegistry in place. A rare, small, main-thread-only patch (at most
    // maxConcurrentLoads streaming events per frame, see world::StreamingManager), so the brief
    // synchronous stall from ExecuteOneShotCommands' vkQueueWaitIdle is an accepted tradeoff over
    // building a batched per-frame dirty-range upload path for what is not a per-frame-hot path.
    void PatchStreamingUnitEntityData(uint32_t unit);

    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): the deterministic (bake-time AND
    // runtime-time both call this SAME function) parking position for ONE physical streaming slot,
    // unique per slot (never per unit) so autosmooth's post-pass (GenerateGeometry()'s own
    // AUTOSMOOTH POST-PASS block) never welds normals across two different archetype shapes baked
    // in the same run -- see kStreamingUnitCount's own header comment on why this pool exists.
    // Called by GenerateGeometry() (to bake each slot's mesh AT this offset) and by
    // SetStreamingUnitState() (to CANCEL this same offset back out of the desired world position --
    // see m_StreamingUnitTranslation's own comment for the exact bug this shared function fixes).
    static maths::vec2 StreamingSlotParkPosition(uint32_t slotIndex);

    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 3): CPU memcpy-into-staging + vkCmdCopyBuffer
    // bake-in of one authored cell's already-simplified HLOD proxy mesh (`placement`, sliced out of
    // m_CellManifest's own shared blob arrays) directly into the vertex/index SSBOs at
    // [runningVertexOffset, +placement.hlodVertexCount) / [runningIndexOffset, +placement.hlodIndexCount)
    // -- no compute shader dispatch, unlike every other GenerateXxx() primitive, because this
    // geometry is already finalized on disk (see kStreamingUnitCount's own header comment on why
    // live per-cell Nanite DAG builds are infeasible and proxies must instead be baked once at
    // startup like this). Advances runningVertexOffset/runningIndexOffset by the ACTUAL proxy size
    // (varies per cell, unlike the fixed-size archetype generators) on success. Returns false (both
    // offsets left untouched, caller must fall back to a plain generated mesh) if the manifest's
    // blob offsets are out of range for this record, or baking would overflow the fixed-size SSBOs
    // -- defensive; should never trigger against a manifest this same build's WorldPartitionBakeTool
    // produced, but a hand-edited/stale file must never read or write out of bounds.
    bool BakeHlodProxyIntoSlot(const world::CellPlacement& placement, uint32_t meshID,
                                const maths::vec2& worldOffset,
                                uint32_t& runningVertexOffset, uint32_t& runningIndexOffset);

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
