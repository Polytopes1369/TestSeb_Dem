#include "renderer/vulkan/AccelerationStructure.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "core/Logger.h"
#include "renderer/vulkan/RayTracingFunctions.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {



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

        // Shared by CreateTlasRefitResources (build-size query) and BuildTLAS's own TLAS geometry
        // setup below -- factored out so both stay byte-for-byte consistent (a size query built
        // from a DIFFERENT geometry description than the actual build could under-size the scratch
        // buffer, a real correctness hazard, not just a style nit).
        VkAccelerationStructureGeometryKHR MakeTlasInstancesGeometry(VkDeviceAddress instanceBufferAddress) {
            VkAccelerationStructureGeometryInstancesDataKHR instancesData{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
            instancesData.arrayOfPointers = VK_FALSE;
            instancesData.data.deviceAddress = instanceBufferAddress;

            VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
            geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            geometry.geometry.instances = instancesData;
            return geometry;
        }

        // Shared build sequence for both BLAS and TLAS: query build sizes, allocate the backing
        // storage buffer, create the VkAccelerationStructureKHR handle, allocate an aligned
        // scratch buffer, record + submit the build, fetch the resulting device address. The only
        // difference between a BLAS and a TLAS build is `type` and the contents of `geometry` /
        // `rangeInfo` -- both fully prepared by the caller (BuildBLAS/BuildTLAS below). `flags`
        // defaults to PREFER_FAST_TRACE (every acceleration structure in this codebase except the
        // skeletally-animated creature's BLAS is built exactly once at Init() and never refit -- see
        // this file's own header comment); BuildBLAS's `allowUpdate` path is the one caller that
        // overrides it to ALLOW_UPDATE_BIT_KHR | PREFER_FAST_BUILD_BIT_KHR instead.
        AccelerationStructure BuildAccelerationStructure(
            VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR& geometry,
            const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo, uint32_t primitiveCount,
            VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) {

            VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
            buildInfo.type = type;
            buildInfo.flags = flags;
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

            VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
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
            });

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
        VkBuffer indexBuffer, VkDeviceSize indexOffsetBytes, uint32_t triangleCount, bool allowUpdate) {

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

        // Skeletal-animation feature: ALLOW_UPDATE_BIT_KHR | PREFER_FAST_BUILD_BIT_KHR for the one
        // BLAS that will later be refit every frame via RecordUpdateBLAS (the skeletally-animated
        // creature's) -- the standard flag pairing for frequently-updated dynamic geometry (fast
        // per-frame UPDATE cost matters far more than peak trace quality for a coarse Fallback Mesh
        // BVH proxy). Every other entity keeps this file's default PREFER_FAST_TRACE_BIT_KHR (see
        // this file's own header comment) via BuildAccelerationStructure's own default argument.
        VkBuildAccelerationStructureFlagsKHR flags = allowUpdate
            ? (VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
            : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

        return BuildAccelerationStructure(physicalDevice, device, allocator, commandPool, queue,
            VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometry, rangeInfo, triangleCount, flags);
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

        VkAccelerationStructureGeometryKHR geometry = MakeTlasInstancesGeometry(GetBufferDeviceAddress(device, instanceBuffer.Handle()));

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

    TlasRefitResources CreateTlasRefitResources(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, uint32_t instanceCount) {
        TlasRefitResources result;

        // Persistent instance buffer -- memcpy'd fresh every RecordRefitTLAS() call, never
        // reallocated (instanceCount is fixed for this engine's static entity list).
        const VkDeviceSize instanceBytes = static_cast<VkDeviceSize>(std::max<uint32_t>(instanceCount, 1)) * sizeof(VkAccelerationStructureInstanceKHR);
        result.instanceBuffer.Create(allocator, instanceBytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        // Query the build sizes a MODE_BUILD rebuild targeting `instanceCount` instances will
        // need, using the EXACT SAME geometry-description helper RecordRefitTLAS's own build uses
        // (MakeTlasInstancesGeometry) -- a size query built from a mismatched geometry description
        // would risk under-sizing the scratch buffer below.
        VkAccelerationStructureGeometryKHR geometry = MakeTlasInstancesGeometry(GetBufferDeviceAddress(device, result.instanceBuffer.Handle()));

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        g_RTFunctions.vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, &instanceCount, &sizeInfo);

        const VkDeviceSize scratchAlignment = QueryMinScratchOffsetAlignment(physicalDevice);
        AlignedScratchBuffer scratch = AllocateAlignedScratchBuffer(allocator, device, sizeInfo.buildScratchSize, scratchAlignment);
        result.scratchBuffer = std::move(scratch.buffer);
        result.scratchAddress = scratch.alignedAddress;

        return result;
    }

    void RecordRefitTLAS(VkCommandBuffer cmd, VkDevice device, VkAccelerationStructureKHR dstTlas,
        TlasRefitResources& resources, const std::vector<VkAccelerationStructureInstanceKHR>& instances) {

        const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
        if (instanceCount == 0) {
            return; // Nothing to refit -- matches BuildTLAS's own empty-scene tolerance.
        }

        // Fresh instance transforms this frame -- persistent host-visible buffer, no per-frame
        // VMA allocation.
        std::memcpy(resources.instanceBuffer.MappedData(), instances.data(),
            static_cast<size_t>(instanceCount) * sizeof(VkAccelerationStructureInstanceKHR));

        VkAccelerationStructureGeometryKHR geometry = MakeTlasInstancesGeometry(GetBufferDeviceAddress(device, resources.instanceBuffer.Handle()));

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        // MODE_BUILD, never MODE_UPDATE -- see this function's own declaration comment
        // (AccelerationStructure.h) for why: a full rebuild is cheap enough at this instance count
        // (~a dozen) that ALLOW_UPDATE's extra bookkeeping (srcAccelerationStructure, a build
        // originally flagged ALLOW_UPDATE) buys nothing here. A MODE_BUILD targeting an
        // already-populated dstAccelerationStructure is well-defined per the spec -- the build
        // does not read the structure's previous contents, only overwrite them.
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;
        buildInfo.dstAccelerationStructure = dstTlas;
        buildInfo.scratchData.deviceAddress = resources.scratchAddress;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = instanceCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

        // --- Pre-build barrier (WAR): the previous frame's HWRT consumers (rayQueryEXT reads
        // against this same TLAS, e.g. Surface Cache radiosity injection) must have finished
        // reading before this frame's build can start overwriting it.
        //
        // Phase 2 (Lumen advanced roadmap) fix, 2026-07-17: this engine's per-frame graphics work
        // is now split across 3 command buffers/submissions (cmdEarly/cmdMid/cmdLate -- see
        // renderer::ClusterRenderPipeline.h's own "Per-frame GPU work" class comment), so "this
        // frame's command buffer" is no longer singular -- this call site is reached from either
        // cmdEarly (the fallback, fully-graphics-serialized path) or asyncComputeCmd (the async-
        // compute-queue path, renderer::ClusterRenderPipeline::RecordAsyncCompute), depending on
        // this frame's own routing decision, which can differ from LAST frame's. The underlying
        // safety property still holds, just via a longer (still fully synchronous, no data race)
        // chain instead of literally being "the same command buffer": whichever queue the PREVIOUS
        // frame's HWRT consumers ran on (they always live in that frame's own cmdLate -- Reflection/
        // World Probes/MegaLights/the forward passes, or asyncComputeCmd's own GIInject bounce loop
        // when that frame used the async path), cmdLate's own submission always WAITS on that
        // frame's asyncComputeFinished semaphore before executing (see RecordFrameLate()'s own
        // ACQUIRE-barrier comment), and frameFence guards ONLY cmdLate's submission -- so frameFence
        // signaling transitively proves BOTH that frame's cmdLate AND its asyncComputeCmd have fully
        // retired. main.cpp's very first action every loop iteration is waiting on frameFence before
        // recording ANYTHING new (including this call, wherever it lands THIS frame) -- so by the
        // time this barrier's srcStage/srcAccess (above) actually need to be true, the previous
        // frame's read is unconditionally already retired, exactly as robustly as the old single-
        // command-buffer design guaranteed it, matching every other per-frame WAR pattern already
        // established in this codebase.
        // --- Post-build barrier (RAW): makes the freshly-rebuilt TLAS visible to every HWRT
        // consumer THIS frame (radiosity injection and anything recorded after it). ---
        VkMemoryBarrier2 preBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        preBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        preBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preDep.memoryBarrierCount = 1;
        preDep.pMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(cmd, &preDep);

        g_RTFunctions.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

        VkMemoryBarrier2 postBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        postBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        postBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        postDep.memoryBarrierCount = 1;
        postDep.pMemoryBarriers = &postBarrier;
        vkCmdPipelineBarrier2(cmd, &postDep);
    }

    BlasUpdateResources CreateBlasUpdateResources(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
        VkDeviceSize vertexStride, uint32_t maxVertex, uint32_t triangleCount) {
        BlasUpdateResources result;

        // Sizing-only geometry description: per the VK_KHR_acceleration_structure spec,
        // vkGetAccelerationStructureBuildSizesKHR reads only the geometry's SHAPE (format/stride/
        // maxVertex/indexType/triangleCount) from pBuildInfo, never the actual buffer device
        // addresses -- so a placeholder (0) deviceAddress here is legal and does not need to
        // resolve to any real buffer. Must otherwise match RecordUpdateBLAS's own per-frame
        // geometry description byte-for-byte (same rationale as CreateTlasRefitResources' own
        // MakeTlasInstancesGeometry reuse: a size query built from a mismatched geometry
        // description risks under-sizing the scratch buffer, a real correctness hazard).
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = 0;
        triangles.vertexStride = vertexStride;
        triangles.maxVertex = maxVertex;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = 0;

        VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles = triangles;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        // Must match BuildBLAS's own allowUpdate=true flags exactly -- the flags describe the
        // acceleration structure's build/update CAPABILITY, and an inconsistent flags value here
        // could under-size updateScratchSize relative to what an actual UPDATE build (RecordUpdateBLAS,
        // using the SAME flags) will really require.
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        // MODE_UPDATE here (unlike CreateTlasRefitResources' own MODE_BUILD size query): this call
        // exists specifically to size the UPDATE path's scratch requirement (updateScratchSize),
        // which the spec allows to differ from buildScratchSize -- querying with MODE_BUILD would
        // return the wrong (BUILD-path) scratch size instead.
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
        g_RTFunctions.vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, &triangleCount, &sizeInfo);

        const VkDeviceSize scratchAlignment = QueryMinScratchOffsetAlignment(physicalDevice);
        // updateScratchSize (NOT buildScratchSize) -- see this function's own header comment
        // (AccelerationStructure.h) for why an UPDATE-mode refit needs its own, separately-sized
        // scratch allocation rather than reusing BuildBLAS's one-shot (already-destroyed) scratch.
        AlignedScratchBuffer scratch = AllocateAlignedScratchBuffer(allocator, device, sizeInfo.updateScratchSize, scratchAlignment);
        result.scratchBuffer = std::move(scratch.buffer);
        result.scratchAddress = scratch.alignedAddress;

        return result;
    }

    void RecordUpdateBLAS(VkCommandBuffer cmd, VkDevice device, VkAccelerationStructureKHR blas, BlasUpdateResources& resources,
        VkBuffer vertexBuffer, VkDeviceSize vertexStride, uint32_t maxVertex, VkDeviceSize vertexOffsetBytes,
        VkBuffer indexBuffer, VkDeviceSize indexOffsetBytes, uint32_t triangleCount) {

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        // May legitimately point at a DIFFERENT buffer than the original BuildBLAS() call used --
        // see this function's own declaration comment (AccelerationStructure.h).
        triangles.vertexData.deviceAddress = GetBufferDeviceAddress(device, vertexBuffer) + vertexOffsetBytes;
        triangles.vertexStride = vertexStride;
        triangles.maxVertex = maxVertex;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = GetBufferDeviceAddress(device, indexBuffer) + indexOffsetBytes;

        VkAccelerationStructureGeometryKHR geometry{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles = triangles;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        // Must match the flags BuildBLAS(..., /*allowUpdate=*/true, ...) originally built `blas`
        // with -- see CreateBlasUpdateResources's own comment on why an inconsistent flags value
        // here is a correctness hazard, not just a style nit.
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        // MODE_UPDATE, src==dst: an in-place refit reusing `blas`'s own existing backing buffer/
        // internal BVH topology, recomputing leaf/node bounds from this frame's (skinned) vertex
        // positions -- see this function's own declaration comment (AccelerationStructure.h) for
        // why src==dst is the correct, standard choice here.
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.srcAccelerationStructure = blas;
        buildInfo.dstAccelerationStructure = blas;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;
        buildInfo.scratchData.deviceAddress = resources.scratchAddress;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = triangleCount;
        rangeInfo.primitiveOffset = 0;
        rangeInfo.firstVertex = 0;
        rangeInfo.transformOffset = 0;
        const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

        // --- Pre-update barrier (WAR): the previous frame's HWRT consumers of this exact BLAS must
        // have finished reading before this frame's update starts overwriting it. Three distinct
        // stages can have read it: (1) ACCELERATION_STRUCTURE_BUILD_BIT_KHR -- last frame's TLAS
        // refit (RecordRefitTLAS), which reads this BLAS's device address/bounds as ITS OWN build
        // input; (2) RAY_TRACING_SHADER_BIT_KHR -- SurfaceCacheHWRT's traceRayEXT traversal
        // descending into it through the TLAS; (3) COMPUTE_SHADER_BIT -- every OTHER HWRT consumer
        // in this codebase uses INLINE ray query (GL_EXT_ray_query) from an ordinary compute shader
        // instead of the dedicated ray-tracing pipeline (SurfaceCacheGIInject.comp, ScreenProbeTrace
        // .comp, WorldProbeInject.comp, ReflectionTrace.comp, MegaLightsShade.comp, etc. -- see
        // CMakeLists.txt's own SPIR-V-1.4 shader list for the full roster), which is a DIFFERENT
        // pipeline stage from RAY_TRACING_SHADER_BIT_KHR despite tracing through the same TLAS/BLAS
        // -- mirrors BuildAccelerationStructure's own post-build barrier just above (which already
        // combines all 3 for the identical "later AS read" reason) and RecordRefitTLAS's own
        // pre-build barrier (COMPUTE_SHADER_BIT | RAY_TRACING_SHADER_BIT_KHR -- that one omits
        // ACCELERATION_STRUCTURE_BUILD_BIT_KHR because nothing in this codebase ever builds another
        // acceleration structure FROM a TLAS the way the TLAS build reads FROM a BLAS). Same
        // "previous frame's read is unconditionally already retired by the time this call is
        // reached" reasoning as RecordRefitTLAS's own pre-build barrier (this codebase's per-frame
        // fence-wait-before-recording discipline, see that function's own comment) -- not re-derived
        // here to avoid duplicating that reasoning verbatim.
        // --- Post-update barrier (RAW): makes the freshly-updated BLAS visible to the TLAS refit
        // that reads it next THIS frame (the caller's own responsibility to sequence immediately
        // after this call -- see RecordUpdateBLAS's own declaration comment) and, transitively via
        // that TLAS refit's OWN post-build barrier, to every ray trace/query consumer after it --
        // narrowed to ACCELERATION_STRUCTURE_BUILD_BIT_KHR alone (unlike the pre-barrier's 3-stage
        // src, and unlike BuildAccelerationStructure's own broader post-build barrier) specifically
        // BECAUSE that strict "TLAS refit always runs next, same queue" sequencing contract is
        // guaranteed here (RecordCreatureBlasUpdate's own caller contract), so nothing else ever
        // reads this BLAS directly without going through that TLAS refit first. ---
        VkMemoryBarrier2 preBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        preBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preDep.memoryBarrierCount = 1;
        preDep.pMemoryBarriers = &preBarrier;
        vkCmdPipelineBarrier2(cmd, &preDep);

        g_RTFunctions.vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

        VkMemoryBarrier2 postBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        postBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        postBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        postDep.memoryBarrierCount = 1;
        postDep.pMemoryBarriers = &postBarrier;
        vkCmdPipelineBarrier2(cmd, &postDep);
    }

}
