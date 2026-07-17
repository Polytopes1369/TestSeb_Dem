#pragma once
// Part 3 deliverable: dispatches src/shaders/src/GI/SurfaceCacheGIInject.comp -- one compute
// dispatch per Surface Cache Card, round-robining across renderer::SurfaceCachePass::GetCards()
// with the same per-call budget discipline as SurfaceCachePass::RecordCapture (see that class'
// own "why asynchronous" comment), re-injecting a cosine-weighted hemisphere-sampled secondary
// light bounce into each card's texels of the radiance atlas.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "core/EngineConfig.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;

    class SurfaceCacheGIInjectPass {
    public:
        SurfaceCacheGIInjectPass() = default;

        SurfaceCacheGIInjectPass(const SurfaceCacheGIInjectPass&) = delete;
        SurfaceCacheGIInjectPass& operator=(const SurfaceCacheGIInjectPass&) = delete;

        // Matches SurfaceCacheGIInject.comp's local_size_x/y exactly.
        static constexpr uint32_t kWorkgroupSize = 8;
        // Mirrors renderer::SurfaceCachePass::kCardsPerFrameBudget's own budgeting rationale --
        // GI injection is at least as expensive per card as capture (sampleCount hemisphere
        // traces per texel vs. one rasterized fragment), so the same small per-call slice applies.
        static inline uint32_t kCardsPerFrameBudget = 16u;
        static inline uint32_t kSampleCountPerTexel = 64u;

        // `traceContext`, `surfaceCache` and `rtPass` must all already be Init'd and must outlive
        // this pass -- their resources (sets 1/2, the 5 atlas images, the TLAS + draw-range
        // table) are bound into this pass' own set 0 unmodified.
        bool Init(VkDevice device, VmaAllocator allocator, const SurfaceCacheTraceContext& traceContext,
            const SurfaceCachePass& surfaceCache, const SurfaceCacheRayTracingPass& rtPass);

        void Shutdown();

        // Dispatches up to kCardsPerFrameBudget cards' worth of GI injection, advancing the
        // internal round-robin cursor exactly like SurfaceCachePass::RecordCapture. `traceMode`
        // selects SurfaceCacheGIInject.comp's trace back-end uniformly for this call: 0 = SWRT
        // (mesh_sdf_trace.glsl), 1 = HWRT (inline rayQueryEXT against the shared TLAS). Caller
        // owns every synchronization barrier before (source atlases + TLAS visible) and after
        // (radiance atlas writes visible to the next consumer) this call.
        void RecordInject(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
            const SurfaceCachePass& surfaceCache, uint32_t traceMode);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE; // set 0.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        uint32_t m_InjectCursor = 0;
        uint32_t m_FrameIndex = 0; // Advances every RecordInject() call -- Halton sequence temporal offset.
    };

}
