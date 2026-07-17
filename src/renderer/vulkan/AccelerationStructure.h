#pragma once
// RAII ownership wrapper + one-shot builders for VK_KHR_acceleration_structure BLAS/TLAS objects
// (Bottom/Top-Level Acceleration Structures), used by renderer::SurfaceCacheRayTracingPass to
// build one BLAS per traced entity directly against renderer::SurfaceCachePass's existing
// combined Fallback Mesh vertex/index buffers (no geometry duplication -- see that class'
// GetVertexBuffer()/GetIndexBuffer() comment) and one TLAS instancing them.
//
// BLAS geometry is static in this engine (no vertex displacement/skinning), so every BuildBLAS()
// call here is a one-shot, Init()-time-only build (VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
// never VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) -- there is deliberately no BLAS
// refit/update path anywhere in this file. The TLAS, however, DOES get a per-frame refit path
// (TlasRefitResources/CreateTlasRefitResources/RecordRefitTLAS below, Phase 4 integration) --
// entity RIGID TRANSFORMS can change per frame (renderer::VulkanContext::UpdateEntityRotations,
// gated by config::ENTITY_SELF_ROTATION_ENABLED), and a TLAS instance's transform is exactly the
// per-instance data that needs refreshing to keep ray-traced GI/reflections consistent with what
// the rasterized/culled path already shows. Still always MODE_BUILD, never MODE_UPDATE/
// ALLOW_UPDATE -- see RecordRefitTLAS's own comment for why.

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
    // acceptable because this only ever runs at Init() time, once per static entity (see this
    // file's own class comment).
    AccelerationStructure BuildBLAS(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkBuffer vertexBuffer, VkDeviceSize vertexStride, uint32_t maxVertex, VkDeviceSize vertexOffsetBytes,
        VkBuffer indexBuffer, VkDeviceSize indexOffsetBytes, uint32_t triangleCount);

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

    // vkGetBufferDeviceAddress wrapper -- every BLAS/TLAS geometry description below addresses its
    // input buffers by GPU virtual address rather than a bound descriptor (the whole reason
    // bufferDeviceAddress is enabled unconditionally in VulkanContext::CreateLogicalDevice).
    VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer);

}
