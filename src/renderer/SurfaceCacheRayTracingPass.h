#pragma once
// Hardware Ray Tracing (HWRT) Surface Cache hit lighting, VK_KHR_ray_tracing_pipeline: builds one
// BLAS per traced entity directly against renderer::SurfaceCachePass's combined Fallback Mesh
// buffers (renderer::AccelerationStructure::BuildBLAS -- no geometry duplication), one TLAS
// instancing them (instanceCustomIndex == renderer::SurfaceCacheTraceContext's dense traced-entity
// index, letting SurfaceCacheHWRT.rchit resolve straight back to that index with no further
// lookup), the 3-stage ray tracing pipeline (SurfaceCacheHWRT.rgen/.rchit/.rmiss) and its Shader
// Binding Table, and dispatches vkCmdTraceRaysKHR over the same RayRequest/RayResult SSBO contract
// renderer::SurfaceCacheSWRTPass uses -- the two back-ends are interchangeable from a caller's
// point of view.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/AccelerationStructure.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;

    class SurfaceCacheRayTracingPass {
    public:
        SurfaceCacheRayTracingPass() = default;

        SurfaceCacheRayTracingPass(const SurfaceCacheRayTracingPass&) = delete;
        SurfaceCacheRayTracingPass& operator=(const SurfaceCacheRayTracingPass&) = delete;

        // Builds every BLAS + the TLAS + the RT pipeline + SBT + this pass' own set 0 (ray I/O +
        // TLAS) and set 3 (Fallback Mesh vertex/index/draw-range buffers) descriptor sets.
        // `traceContext` and `surfaceCache` must already be Init'd and must outlive this pass
        // (their sets 1/2 are reused unmodified in this pipeline's layout, and this pass builds
        // its BLAS geometry directly against surfaceCache's own vertex/index buffers).
        bool Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
            VkCommandPool commandPool, VkQueue queue,
            const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache);

        void Shutdown();

        // (Re)writes set 0's ray-request/result bindings -- same contract as
        // SurfaceCacheSWRTPass::SetRayBuffers(). Must be called at least once before RecordTrace().
        void SetRayBuffers(VkBuffer rayBuffer, VkDeviceSize rayBufferSize, VkBuffer resultBuffer, VkDeviceSize resultBufferSize);

        // Dispatches vkCmdTraceRaysKHR over a 1D launch of `rayCount` rays. Caller owns every
        // synchronization barrier before/after this call, same discipline as
        // SurfaceCacheSWRTPass::RecordTrace().
        void RecordTrace(VkCommandBuffer cmd, uint32_t rayCount, const SurfaceCacheTraceContext& traceContext);

        // Exposed so renderer::SurfaceCacheGIInjectPass can bind the SAME TLAS + draw-range table
        // into its own inline-ray-query (VK_KHR_ray_query) descriptor set, instead of building a
        // second TLAS over the same geometry -- see SurfaceCacheGIInject.comp's own TraceHWRT.
        VkAccelerationStructureKHR GetTLASHandle() const { return m_TLAS.Handle(); }
        VkBuffer GetDrawRangeBuffer() const { return m_DrawRangeBuffer.Handle(); }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        std::vector<AccelerationStructure> m_BLASList; // Index-aligned with SurfaceCacheTraceContext::GetTracedEntities().
        AccelerationStructure m_TLAS;

        GpuBuffer m_DrawRangeBuffer; // set 3, binding 2 -- EntityDrawRangeGpu[], index-aligned with m_BLASList.

        VkDescriptorSetLayout m_RaySetLayout = VK_NULL_HANDLE;      // set 0: ray I/O + TLAS.
        VkDescriptorSetLayout m_GeometrySetLayout = VK_NULL_HANDLE; // set 3: vertex/index/draw-range buffers.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_RaySet = VK_NULL_HANDLE;
        VkDescriptorSet m_GeometrySet = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        // Shader Binding Table: one dedicated buffer per region (raygen/miss/hit), each aligned
        // to VkPhysicalDeviceRayTracingPipelinePropertiesKHR::shaderGroupBaseAlignment -- see
        // Init()'s own comment on why a single shared buffer isn't used here.
        GpuBuffer m_RaygenSBT;
        GpuBuffer m_MissSBT;
        GpuBuffer m_HitSBT;
        VkStridedDeviceAddressRegionKHR m_RaygenRegion{};
        VkStridedDeviceAddressRegionKHR m_MissRegion{};
        VkStridedDeviceAddressRegionKHR m_HitRegion{};
        VkStridedDeviceAddressRegionKHR m_CallableRegion{}; // Unused (no callable shaders) -- stays zeroed.
    };

}
