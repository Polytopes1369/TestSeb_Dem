#pragma once
// RAII ownership wrapper + one-shot builders for VK_KHR_acceleration_structure BLAS/TLAS objects
// (Bottom/Top-Level Acceleration Structures), used by renderer::SurfaceCacheRayTracingPass to
// build one BLAS per traced entity directly against renderer::SurfaceCachePass's existing
// combined Fallback Mesh vertex/index buffers (no geometry duplication -- see that class'
// GetVertexBuffer()/GetIndexBuffer() comment) and one TLAS instancing them.
//
// Entities are static in this engine (see renderer::GlobalSDFPass's own class comment: "dynamic
// object add/remove/move is future work"), so every BuildBLAS()/BuildTLAS() call here is a
// one-shot, Init()-time-only build (VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
// never VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) -- there is deliberately no
// RecordRefit()/Update() anywhere in this file.

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

    // vkGetBufferDeviceAddress wrapper -- every BLAS/TLAS geometry description below addresses its
    // input buffers by GPU virtual address rather than a bound descriptor (the whole reason
    // bufferDeviceAddress is enabled unconditionally in VulkanContext::CreateLogicalDevice).
    VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer);

}
