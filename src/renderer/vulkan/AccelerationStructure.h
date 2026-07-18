#pragma once
// RAII ownership wrapper + one-shot builders for VK_KHR_acceleration_structure BLAS/TLAS objects
// (Bottom/Top-Level Acceleration Structures), used by renderer::SurfaceCacheRayTracingPass to
// build one BLAS per traced entity directly against renderer::SurfaceCachePass's existing
// combined Fallback Mesh vertex/index buffers (no geometry duplication -- see that class'
// GetVertexBuffer()/GetIndexBuffer() comment) and one TLAS instancing them.
//
// BLAS geometry is static for every entity EXCEPT the engine's one skeletally-animated entity (the
// procedural creature, core::EntityFlags::IsSkeletallyAnimated -- see animation::SkeletalAnimator),
// so every BuildBLAS() call is a one-shot, Init()-time-only build with
// VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR (never ALLOW_UPDATE) UNLESS the new
// `allowUpdate` parameter is set, in which case the flags become ALLOW_UPDATE_BIT_KHR |
// PREFER_FAST_BUILD_BIT_KHR instead (the standard pairing for frequently-updated dynamic geometry --
// PREFER_FAST_TRACE would otherwise force a full high-quality rebuild's worth of build cost on
// every single UPDATE call, defeating the point of updating at all). `allowUpdate` is set for
// exactly one BLAS in this codebase: the skeletally-animated creature's, built by
// renderer::SurfaceCacheRayTracingPass::Init() against a DEDICATED per-frame-skinned vertex buffer
// (see that class' own RecordCreatureBlasUpdate) rather than the shared, static Fallback Mesh
// buffer every other entity's one-shot BLAS still reads directly -- see RecordUpdateBLAS below for
// the per-frame UPDATE-mode refit path this one BLAS gets that no other BLAS in this file does.
// The TLAS, separately, DOES get a per-frame refit path of its own
// (TlasRefitResources/CreateTlasRefitResources/RecordRefitTLAS below, Phase 4 integration) --
// entity RIGID TRANSFORMS can change per frame (renderer::VulkanContext::UpdateEntityRotations,
// gated by config::ENTITY_SELF_ROTATION_ENABLED), and a TLAS instance's transform is exactly the
// per-instance data that needs refreshing to keep ray-traced GI/reflections consistent with what
// the rasterized/culled path already shows. Still always MODE_BUILD, never MODE_UPDATE/
// ALLOW_UPDATE -- see RecordRefitTLAS's own comment for why (the TLAS's own per-frame refit is
// unrelated to, and unaffected by, the creature BLAS's separate per-frame UPDATE below -- the two
// are sequenced by the caller, see RecordUpdateBLAS's own comment on ordering).

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class AccelerationStructure {
    public:
        AccelerationStructure() = default;
        ~AccelerationStructure();

        AccelerationStructure(const AccelerationStructure&) = delete;
        AccelerationStructure& operator=(const AccelerationStructure&) = delete;

        AccelerationStructure(AccelerationStructure&& other) noexcept;
        AccelerationStructure& operator=(AccelerationStructure&& other) noexcept;

        void Destroy();

        VkAccelerationStructureKHR Handle() const { return m_Handle; }
        VkDeviceAddress DeviceAddress() const { return m_DeviceAddress; }
        bool IsValid() const { return m_Handle != VK_NULL_HANDLE; }

        // Package-private to this .cpp's builder functions (BuildBLAS/BuildTLAS) -- not part of
        // the public construction API, since correctly sizing/building an acceleration structure
        // requires the multi-step vkGetAccelerationStructureBuildSizesKHR ->
        // vmaCreateBuffer -> vkCreateAccelerationStructureKHR -> vkCmdBuildAccelerationStructuresKHR
        // sequence those free functions already implement in full.
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkAccelerationStructureKHR m_Handle = VK_NULL_HANDLE;
        GpuBuffer m_Buffer; // Backing storage for this acceleration structure's own data.
        VkDeviceAddress m_DeviceAddress = 0;
    };

    // Builds one BLAS directly against geometry already resident in `vertexBuffer`/`indexBuffer`
    // (both must have been created with VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
    // | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT -- see renderer::SurfaceCachePass::GetVertexBuffer()'s
    // own comment): `vertexOffsetBytes`/`indexOffsetBytes` address one entity's own span inside a
    // larger combined buffer (renderer::SurfaceCachePass::EntityDrawRange), so no per-entity
    // geometry copy is ever made. One-shot: opens its own command buffer on `commandPool`,
    // records the build, submits to `queue`, and blocks (vkQueueWaitIdle) before returning --
    // acceptable because this only ever runs at Init() time, once per entity (see this file's own
    // class comment). `allowUpdate` (default false, matching every call site but the skeletally-
    // animated creature's) selects VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
    // VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR instead of the default
    // PREFER_FAST_TRACE_BIT_KHR -- see this file's own header comment. This initial build is still
    // always MODE_BUILD (never MODE_UPDATE) regardless of `allowUpdate`: it only controls whether
    // the RESULTING acceleration structure is eligible for a LATER RecordUpdateBLAS() call, exactly
    // like a VkBuffer's usage flags declare eligibility for a later operation without performing it
    // now.
    AccelerationStructure BuildBLAS(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkBuffer vertexBuffer, VkDeviceSize vertexStride, uint32_t maxVertex, VkDeviceSize vertexOffsetBytes,
        VkBuffer indexBuffer, VkDeviceSize indexOffsetBytes, uint32_t triangleCount, bool allowUpdate = false);

    // Builds one TLAS from `instances` (each entry's transform/instanceCustomIndex/mask/
    // instanceShaderBindingTableRecordOffset/flags/accelerationStructureReference already filled
    // in by the caller -- see renderer::SurfaceCacheRayTracingPass::Init()). Same one-shot,
    // Init()-time-only discipline as BuildBLAS().
    AccelerationStructure BuildTLAS(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const std::vector<VkAccelerationStructureInstanceKHR>& instances);

    // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): PERSISTENT buffers for
    // a per-frame TLAS refit (see RecordRefitTLAS below) -- deliberately separate from the one-shot
    // builders above, which this file's own header comment documents as intentionally lacking any
    // refit/update path. `instanceBuffer` is host-visible mapped and memcpy'd fresh every frame by
    // the caller; `scratchBuffer` is sized once (instance count is fixed for this engine's static
    // entity list) and reused every frame -- neither is ever reallocated, so a refit costs no
    // per-frame VMA allocation churn.
    struct TlasRefitResources {
        GpuBuffer instanceBuffer;
        GpuBuffer scratchBuffer;
        VkDeviceAddress scratchAddress = 0; // Pre-aligned device address into scratchBuffer, cached once.
    };

    // Allocates a TlasRefitResources sized for exactly `instanceCount` TLAS instances -- call once
    // at Init() time, immediately after the first (one-shot) BuildTLAS() call for the same
    // instance count.
    TlasRefitResources CreateTlasRefitResources(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, uint32_t instanceCount);

    // Records a full MODE_BUILD rebuild (never ALLOW_UPDATE -- see this file's own header comment:
    // the instance count here is tiny, ~a dozen, so there is no real performance case for an
    // incremental update) DIRECTLY INTO `cmd` -- no separate one-shot submit/wait, unlike
    // BuildTLAS() above. Targets the SAME already-allocated `dstTlas` handle every frame (a
    // MODE_BUILD rebuild does not depend on the acceleration structure's previous contents, unlike
    // MODE_UPDATE, so reusing the same backing buffer across frames is valid as long as the
    // instance count -- and therefore the required backing-buffer size -- never changes, which it
    // doesn't for this engine's fixed, static entity list). Includes its own pre-build (WAR) and
    // post-build (RAW) VkMemoryBarrier2 pair -- see the .cpp's own comment for the exact stage/
    // access masks and why a same-command-buffer barrier is sufficient given this engine's
    // single-frame-in-flight model (no extra cross-frame semaphore needed).
    void RecordRefitTLAS(VkCommandBuffer cmd, VkDevice device, VkAccelerationStructureKHR dstTlas,
        TlasRefitResources& resources, const std::vector<VkAccelerationStructureInstanceKHR>& instances);

    // Skeletal-animation feature: PERSISTENT scratch buffer for a per-frame BLAS UPDATE-mode refit
    // (see RecordUpdateBLAS below) -- the BLAS-update analogue of TlasRefitResources above, minus
    // an instance buffer (a BLAS update has no per-instance array to re-upload; only the vertex
    // buffer's CONTENTS change frame to frame, its device address/stride/count stay fixed, so the
    // geometry description RecordUpdateBLAS builds needs no persistent buffer of its own beyond the
    // scratch). `scratchBuffer` is sized from updateScratchSize (VkAccelerationStructureBuildSizesInfoKHR),
    // which the spec allows to differ from -- typically be smaller than -- the ORIGINAL build's
    // buildScratchSize, so it is NOT safe to reuse whatever one-shot scratch buffer BuildBLAS()
    // itself used (that one is destroyed immediately after BuildAccelerationStructure() returns,
    // per that function's own one-shot discipline) -- a fresh, persistent, appropriately-sized
    // allocation is required.
    struct BlasUpdateResources {
        GpuBuffer scratchBuffer;
        VkDeviceAddress scratchAddress = 0; // Pre-aligned device address into scratchBuffer, cached once.
    };

    // Allocates a BlasUpdateResources sized for an UPDATE-mode rebuild of a BLAS with exactly
    // `triangleCount`/`maxVertex` geometry (must match the geometry every RecordUpdateBLAS() call
    // for this BLAS will describe -- an update's geometry shape, unlike its vertex buffer CONTENTS,
    // must stay identical to what the original BuildBLAS() call used, per the
    // VK_KHR_acceleration_structure spec's UPDATE-mode contract). Call once at Init() time,
    // immediately after the BuildBLAS(..., /*allowUpdate=*/true) call it refits.
    BlasUpdateResources CreateBlasUpdateResources(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
        VkDeviceSize vertexStride, uint32_t maxVertex, uint32_t triangleCount);

    // Records a MODE_UPDATE refit DIRECTLY into `cmd` -- no separate one-shot submit/wait, unlike
    // BuildBLAS() above. `blas` MUST have been built by BuildBLAS(..., /*allowUpdate=*/true, ...)
    // (VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) -- both srcAccelerationStructure and
    // dstAccelerationStructure are `blas` itself (an in-place refit: the spec explicitly permits
    // src==dst for an UPDATE build, and this is the standard choice when the caller has no use for
    // keeping the pre-update structure around, exactly this codebase's case). `vertexBuffer`/
    // `vertexOffsetBytes` may point at a COMPLETELY DIFFERENT buffer than the ORIGINAL BuildBLAS()
    // call used (only the geometry's COUNT/topology must stay identical, not its source buffer --
    // see this function's own caller, renderer::SurfaceCacheRayTracingPass::RecordCreatureBlasUpdate,
    // which updates from a dedicated per-frame-skinned buffer instead of the shared static Fallback
    // Mesh buffer the initial Init()-time build read from); `indexBuffer`/`indexOffsetBytes`/
    // `triangleCount` are expected to stay byte-identical to the original build every call (this
    // BLAS's topology never changes, only vertex POSITIONS do) but are re-specified fresh anyway,
    // mirroring RecordRefitTLAS's own "every build call gets a complete, self-contained geometry
    // description" discipline -- no partial/implicit reuse of a previous call's geometry struct.
    // Includes its own pre-update (WAR: previous frame's TLAS-build/ray-trace read of this BLAS
    // must retire before this update starts writing) and post-update (RAW: this update's write must
    // be visible to the TLAS refit that reads this BLAS's device address/bounds next) VkMemoryBarrier2
    // pair, using VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR /
    // VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR|WRITE_BIT_KHR throughout -- see the .cpp's own
    // comment for the exact reasoning. Does NOT include the barrier between the skinning compute
    // shader's vertex write and this update's own read of that same buffer -- that boundary crosses
    // a DIFFERENT pipeline stage (COMPUTE_SHADER, not ACCELERATION_STRUCTURE_BUILD) and is the
    // caller's own responsibility (RecordCreatureBlasUpdate records it immediately before calling
    // this function). Caller must also ensure this call is sequenced BEFORE the TLAS refit that
    // references `blas`'s device address this same frame (RecordRefitTLAS above) -- see
    // RecordCreatureBlasUpdate's own comment for the exact call-site ordering this codebase uses.
    void RecordUpdateBLAS(VkCommandBuffer cmd, VkDevice device, VkAccelerationStructureKHR blas, BlasUpdateResources& resources,
        VkBuffer vertexBuffer, VkDeviceSize vertexStride, uint32_t maxVertex, VkDeviceSize vertexOffsetBytes,
        VkBuffer indexBuffer, VkDeviceSize indexOffsetBytes, uint32_t triangleCount);

    // vkGetBufferDeviceAddress wrapper -- every BLAS/TLAS geometry description below addresses its
    // input buffers by GPU virtual address rather than a bound descriptor (the whole reason
    // bufferDeviceAddress is enabled unconditionally in VulkanContext::CreateLogicalDevice).
    VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer);

}
