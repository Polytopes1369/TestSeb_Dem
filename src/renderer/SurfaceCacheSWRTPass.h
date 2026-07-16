#pragma once
// Standalone dispatcher for the SWRT (SoftWare Ray Tracing) compute shader
// (src/shaders/src/GI/SurfaceCacheTraceSWRT.comp): owns the compute pipeline + its own set-0
// descriptor set (the ray-request/result SSBO pair), while descriptor sets 1 (mesh SDF trace
// scene) and 2 (surface cache sampling) come from an already-Init'd
// renderer::SurfaceCacheTraceContext, shared unmodified with SurfaceCacheRayTracingPass and
// SurfaceCacheGIInjectPass. Intended for a future screen-space (or any other caller-supplied ray
// batch) final-gather pass -- SurfaceCacheGIInjectPass's own hemisphere rays call
// TraceMeshSDFScene directly (see mesh_sdf_trace.glsl) instead of round-tripping through this
// class' SSBO interface, since that avoids a buffer upload/dispatch per texel batch.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace renderer {

    class SurfaceCacheTraceContext;

    class SurfaceCacheSWRTPass {
    public:
        SurfaceCacheSWRTPass() = default;

        SurfaceCacheSWRTPass(const SurfaceCacheSWRTPass&) = delete;
        SurfaceCacheSWRTPass& operator=(const SurfaceCacheSWRTPass&) = delete;

        // Matches SurfaceCacheTraceSWRT.comp's local_size_x exactly.
        static constexpr uint32_t kWorkgroupSize = 64;

        // Builds set 0's layout + the compute pipeline (whose layout also includes
        // traceContext.GetMeshSdfTraceSetLayout() / GetSurfaceCacheSamplingSetLayout() at sets
        // 1/2), plus this pass' own descriptor pool. `traceContext` must already be Init'd and
        // must outlive this pass.
        bool Init(VkDevice device, VmaAllocator allocator, const SurfaceCacheTraceContext& traceContext);

        void Shutdown();

        // (Re)writes set 0's two bindings to `rayBuffer` (SurfaceCacheTraceSWRT.comp's
        // RayRequestBuffer, readonly) / `resultBuffer` (its RayResultBuffer, writeonly). Call
        // whenever the caller's buffers change -- they need not be the same buffers every frame.
        // Must be called at least once before RecordTrace().
        void SetRayBuffers(VkBuffer rayBuffer, VkDeviceSize rayBufferSize, VkBuffer resultBuffer, VkDeviceSize resultBufferSize);

        // Dispatches ceil(rayCount / kWorkgroupSize) workgroups, tracing `rayCount` entries of the
        // currently-bound ray buffer (see SetRayBuffers()) against every entity `traceContext`
        // knows about. Caller owns every synchronization barrier before (ray buffer fully
        // written) and after (result buffer visible to the next consumer) this call -- this
        // function only records the bind/push-constant/dispatch, no barriers of its own, matching
        // every other compute pass in this codebase (e.g. renderer::GlobalSDFPass::RecordSlab).
        void RecordTrace(VkCommandBuffer cmd, uint32_t rayCount, const SurfaceCacheTraceContext& traceContext);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_RaySetLayout = VK_NULL_HANDLE; // set 0.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_RaySet = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
