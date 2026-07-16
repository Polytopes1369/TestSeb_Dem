#include "renderer/AccelerationStructure.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "core/Logger.h"
#include "renderer/RayTracingFunctions.h"

namespace renderer {

    namespace {

        VkCommandBuffer BeginOneShotCommandBuffer(VkDevice device, VkCommandPool commandPool) {
            VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            allocInfo.commandPool = commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
            return cmd;
        }

        void EndAndSubmitOneShotCommandBuffer(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkCommandBuffer cmd) {
            VK_CHECK(vkEndCommandBuffer(cmd));
            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);
        }

        // VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
        // (queried fresh each call -- this only ever runs a handful of times at Init(), so caching
        // is not worth the extra state). scratchData.deviceAddress must be a multiple of this
        // value per the VK_KHR_acceleration_structure spec (VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03802);
        // a VMA-backed dedicated allocation's base address is not otherwise guaranteed to already
        // satisfy it, so AllocateAlignedScratchBuffer below explicitly pads and re-aligns.
        VkDeviceSize QueryMinScratchOffsetAlignment(VkPhysicalDevice physicalDevice) {
            VkPhysicalDeviceAccelerationStructurePropertiesKHR accelStructProps{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
            VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
            props2.pNext = &accelStructProps;
            vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
            return std::max<VkDeviceSize>(accelStructProps.minAccelerationStructureScratchOffsetAlignment, 1);
        }

        // Allocates a scratch buffer at least `requiredSize` bytes, USABLE from a device address
        // that is itself a multiple of `alignment` -- over-allocates by (alignment - 1) bytes and
        // returns the aligned address alongside the raw buffer, since a VkBuffer's device address
        // cannot otherwise be offset without an explicitly aligned sub-allocation.
        struct AlignedScratchBuffer {
            GpuBuffer buffer;
            VkDeviceAddress alignedAddress = 0;
        };

        AlignedScratchBuffer AllocateAlignedScratchBuffer(VmaAllocator allocator, VkDevice device, VkDeviceSize requiredSize, VkDeviceSize alignment) {
            AlignedScratchBuffer result;
            const VkDeviceSize paddedSize = requiredSize + alignment;
            result.buffer.Create(allocator, paddedSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
            VkDeviceAddress baseAddress = GetBufferDeviceAddress(device, result.buffer.Handle());
            result.alignedAddress = (baseAddress + alignment - 1) & ~(alignment - 1);
            return result;
        }

        // Shared build sequence for both BLAS and TLAS: query build sizes, allocate the backing
        // storage buffer, create the VkAccelerationStructureKHR handle, allocate an aligned
        // scratch buffer, record + submit the build, fetch the resulting device address. The only
        // difference between a BLAS and a TLAS build is `type` and the contents of `geometry` /
        // `rangeInfo` -- both fully prepared by the caller (BuildBLAS/BuildTLAS below).
        AccelerationStructure BuildAccelerationStructure(
            VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR& geometry,
            const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo, uint32_t primitiveCount) {

            VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
            buildInfo.type = type;
            // PREFER_FAST_TRACE, never ALLOW_UPDATE: every acceleration structure in this codebase
            // is built exactly once at Init() and never refit -- see this file's own header
            // comment on why entities are treated as static.
            buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            buildInfo.geometryCount = 1;
            buildInfo.pGeometries = &geometry;

            VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
            g_RTFunctions.vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &buildInfo, &primitiveCount, &sizeInfo);

            AccelerationStructure result;
            result.m_Device = device;
            result.m_Allocator = allocator;
            result.m_Buffer.Create(allocator, sizeInfo.accelerationStructureSize,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);

            VkAccelerationStructureCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
            createInfo.buffer = result.m_Buffer.Handle();
            createInfo.offset = 0;
            createInfo.size = sizeInfo.accelerationStructureSize;
            createInfo.type = type;
            VK_CHECK(g_RTFunctions.vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &result.m_Handle));

            const VkDeviceSize scratchAlignment = QueryMinScratchOffsetAlignment(physicalDevice);
            AlignedScratchBuffer scratch = AllocateAlignedScratchBuffer(allocator, device, sizeInfo.buildScratchSize, scratchAlignment);

            buildInfo.dstAccelerationStructure = result.m_Handle;
            buildInfo.scratchData.deviceAddress = scratch.alignedAddress;

            const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

            VkCommandBuffer cmd = BeginOneShotCommandBuffer(device, commandPool);
            g_RTFunctions.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

            // Makes the freshly-built acceleration structure's data visible to a later
            // acceleration-structure read (a TLAS build referencing this BLAS, or a
            // traceRayEXT/rayQueryEXT traversal reading this TLAS) -- explicit VkMemoryBarrier2 per
            // CLAUDE.md's synchronization rule, no shortcut via the implicit submission-boundary
            // ordering alone.
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
            barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            EndAndSubmitOneShotCommandBuffer(device, commandPool, queue, cmd);
            scratch.buffer.Destroy();

            VkAccelerationStructureDeviceAddressInfoKHR addressInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
            addressInfo.accelerationStructure = result.m_Handle;
            result.m_DeviceAddress = g_RTFunctions.vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);

            return result;
        }

    } // namespace

    VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer) {
        VkBufferDeviceAddressInfo info{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        info.buffer = buffer;
        return vkGetBufferDeviceAddress(device, &info);
    }

    AccelerationStructure::~AccelerationStructure() {
        Destroy();
    }

    AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept
        : m_Device(other.m_Device), m_Allocator(other.m_Allocator), m_Handle(other.m_Handle),
        m_Buffer(std::move(other.m_Buffer)), m_DeviceAddress(other.m_DeviceAddress) {
        other.m_Device = VK_NULL_HANDLE;
        other.m_Allocator = VK_NULL_HANDLE;
        other.m_Handle = VK_NULL_HANDLE;
        other.m_DeviceAddress = 0;
    }

    AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&& other) noexcept {
        if (this != &other) {
            Destroy();
            m_Device = other.m_Device;
            m_Allocator = other.m_Allocator;
            m_Handle = other.m_Handle;
            m_Buffer = std::move(other.m_Buffer);
            m_DeviceAddress = other.m_DeviceAddress;
            other.m_Device = VK_NULL_HANDLE;
            other.m_Allocator = VK_NULL_HANDLE;
            other.m_Handle = VK_NULL_HANDLE;
            other.m_DeviceAddress = 0;
        }
        return *this;
    }

    void AccelerationStructure::Destroy() {
        if (m_Handle != VK_NULL_HANDLE && m_Device != VK_NULL_HANDLE) {
            g_RTFunctions.vkDestroyAccelerationStructureKHR(m_Device, m_Handle, nullptr);
        }
        m_Buffer.Destroy();
        m_Handle = VK_NULL_HANDLE;
        m_DeviceAddress = 0;
        m_Device = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
    }

    AccelerationStructure BuildBLAS(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkBuffer vertexBuffer, VkDeviceSize vertexStride, uint32_t maxVertex, VkDeviceSize vertexOffsetBytes,
        VkBuffer indexBuffer, VkDeviceSize indexOffsetBytes, uint32_t triangleCount) {

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // Matches geometry::FallbackVertex::position's layout exactly.
        triangles.vertexData.deviceAddress = GetBufferDeviceAddress(device, vertexBuffer) + vertexOffsetBytes;
        triangles.vertexStride = vertexStride;
        triangles.maxVertex = maxVertex;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = GetBufferDeviceAddress(device, indexBuffer) + indexOffsetBytes;

        VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles = triangles;
        // OPAQUE: this codebase's Fallback Mesh deliberately ignores the opaque/masked cluster
        // classification (see geometry::FallbackMeshBuilder.h's own header comment) -- every BLAS
        // triangle is unconditionally opaque, so SurfaceCacheHWRT.rchit is always the sole hit
        // shader invoked per ray, with no any-hit / alpha-test detour.
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = triangleCount;
        rangeInfo.primitiveOffset = 0;
        rangeInfo.firstVertex = 0;
        rangeInfo.transformOffset = 0;

        return BuildAccelerationStructure(physicalDevice, device, allocator, commandPool, queue,
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometry, rangeInfo, triangleCount);
    }

    AccelerationStructure BuildTLAS(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const std::vector<VkAccelerationStructureInstanceKHR>& instances) {

        const uint32_t instanceCount = static_cast<uint32_t>(instances.size());

        // The instance array itself must also live in a device-addressable buffer -- uploaded
        // host-visible + mapped (small, Init()-time-only data, mirroring
        // SurfaceCacheTraceContext's own UploadHostVisibleSSBO rationale) rather than through a
        // staging-buffer copy.
        GpuBuffer instanceBuffer;
        const VkDeviceSize instanceBytes = static_cast<VkDeviceSize>(std::max<size_t>(instances.size(), 1)) * sizeof(VkAccelerationStructureInstanceKHR);
        instanceBuffer.Create(allocator, instanceBytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        if (!instances.empty()) {
            std::memcpy(instanceBuffer.MappedData(), instances.data(), static_cast<size_t>(instances.size()) * sizeof(VkAccelerationStructureInstanceKHR));
        } else {
            std::memset(instanceBuffer.MappedData(), 0, static_cast<size_t>(instanceBytes));
        }

        VkAccelerationStructureGeometryInstancesDataKHR instancesData{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
        instancesData.arrayOfPointers = VK_FALSE;
        instancesData.data.deviceAddress = GetBufferDeviceAddress(device, instanceBuffer.Handle());

        VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.geometry.instances = instancesData;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = instanceCount;

        AccelerationStructure tlas = BuildAccelerationStructure(physicalDevice, device, allocator, commandPool, queue,
            VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, geometry, rangeInfo, instanceCount);

        // The instance buffer is only read DURING the build (vkCmdBuildAccelerationStructuresKHR
        // above, already submitted-and-waited-on inside BuildAccelerationStructure) -- safe to
        // free now that that call has returned.
        instanceBuffer.Destroy();
        return tlas;
    }

}
