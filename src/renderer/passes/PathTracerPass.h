#pragma once
// UE5.8 rendering-parity gap G10b -- offline/reference unbiased Path Tracer (ground-truth view to
// validate the real-time Lumen approximation against, exactly what UE5.8's own Path Tracer render
// mode is for). This is a DEBUG-ONLY validation tool, NOT a shipping render path, so per CLAUDE.md
// rule 8 ("modes de visualisation (Lumen/Nanite)" must be excluded from Release entirely) the ENTIRE
// file -- declaration and definition -- is wrapped in #ifndef NDEBUG: in a Release build this
// header declares nothing and PathTracerPass.cpp compiles to an empty translation unit, so zero
// code / strings / symbols from this feature survive into the production executable.
//
// Pipeline (follows the existing SurfaceCacheRayTracingPass ray-tracing-pipeline convention):
//   - A 3-stage RT pipeline (PathTracer.rgen/.rchit/.rmiss) + Shader Binding Table, built exactly
//     like SurfaceCacheRayTracingPass (reusing renderer::g_RTFunctions / RayTracingFunctions.cpp
//     helpers and the shaderGroupBaseAlignment SBT layout). The ray-gen shader is a single-kernel
//     path tracer (primary ray + indirect bounces + NEE shadow rays); see PathTracer.rgen's header
//     for the full sampling strategy.
//   - Reuses the SAME scene TLAS renderer::SurfaceCacheRayTracingPass already built for Lumen
//     (GetTLASHandle()) and the SAME combined Fallback Mesh vertex/index/draw-range buffers -- no
//     second acceleration structure, no geometry duplication.
//   - A tiny compute resolve pipeline (PathTracerResolve.comp) averages the progressive HDR
//     accumulation buffer and tonemaps it (same physical-exposure + ACES chain as the real-time
//     PostProcess path) into an rgba8 display image that renderer::ClusterRenderPipeline blits to
//     the swapchain in place of the normal composite while this mode is active.
//
// Progressive accumulation: RecordFrame() accumulates ONE sample-per-pixel per frame into a
// persistent rgba32f sum image while the camera is stationary, and RESETS the accumulation (sample
// count -> 0, sum overwritten) on any camera movement -- the standard behavior of every reference
// path-tracer mode, UE5.8's included. Movement is detected by comparing the camera view matrix
// frame-to-frame (jitter-independent -- the view matrix carries no TAA jitter, only the projection
// does).

#ifndef NDEBUG

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/LightingTypes.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace core { struct EntityData; }

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;

    class PathTracerPass {
    public:
        PathTracerPass() = default;

        PathTracerPass(const PathTracerPass&) = delete;
        PathTracerPass& operator=(const PathTracerPass&) = delete;

        static constexpr VkFormat kAccumFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // Running radiance SUM (precision over many samples).
        static constexpr VkFormat kDisplayFormat = VK_FORMAT_R8G8B8A8_UNORM;    // Tonemapped output, blitted to the swapchain.
        static constexpr uint32_t kWorkgroupSize = 8;

        // Builds the RT pipeline + SBT + the per-traced-entity materialID buffer + both descriptor
        // sets + the resolve pipeline. `traceContext`/`surfaceCache`/`rtPass` must already be Init'd
        // and outlive this pass (their TLAS + geometry buffers are bound into this pass' descriptor
        // set unmodified). `entityDataCPU` (index == meshID) supplies each traced entity's real
        // materialID; `materialParamsBuffer` is renderer::ClusterResolvePass's material-table SSBO;
        // `megaLightsBuffer` is renderer::MegaLightsPass's point-light SSBO -- both bound (not
        // copied) so the reference shades the exact same materials/lights as the real-time path.
        bool Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
            VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
            const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
            const SurfaceCacheRayTracingPass& rtPass,
            const core::EntityData* entityDataCPU,
            VkBuffer materialParamsBuffer, VkDeviceSize materialParamsSize,
            VkBuffer megaLightsBuffer, VkDeviceSize megaLightsSize);

        void Shutdown();

        // Records one accumulation frame: uploads this frame's view params, dispatches the path-trace
        // ray-gen (accumulating into the sum image), then the resolve/tonemap into the display image.
        // Detects camera movement via `cameraView` and resets the accumulation when it changes.
        // Caller owns the trailing barrier that makes the display image visible to the final blit
        // (renderer::ClusterRenderPipeline's existing resolve->blit barrier already covers it, since
        // the resolve dispatch's write is a COMPUTE_SHADER SHADER_STORAGE_WRITE). All handles/buffers
        // this pass binds must already be visible for RT/compute reads at this call site (true in
        // RecordFrameLate, where the TLAS/geometry/material/light buffers were all made available by
        // earlier GI passes this same frame).
        void RecordFrame(VkCommandBuffer cmd, const maths::mat4& invViewProj, const maths::mat4& cameraView,
            const maths::vec3& cameraPositionWorld, const DirectionalLight& sun, uint32_t frameIndex);

        VkImage GetDisplayImage() const { return m_DisplayImage; }
        VkImageView GetDisplayView() const { return m_DisplayView; }
        VkExtent2D GetExtent() const { return m_RenderExtent; }
        // Number of samples-per-pixel accumulated so far in the current stationary-camera period --
        // shown live in the Debug ImGui path-tracer panel.
        uint32_t GetAccumulatedSampleCount() const { return m_AccumulatedSamples; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        // Persistent HDR accumulation SUM image (rgba32f) + tonemapped display image (rgba8), both
        // kept in VK_IMAGE_LAYOUT_GENERAL for their whole lifetime (this codebase's convention for
        // compute-written intermediate images).
        VkImage m_AccumImage = VK_NULL_HANDLE;
        VmaAllocation m_AccumAllocation = VK_NULL_HANDLE;
        VkImageView m_AccumView = VK_NULL_HANDLE;
        VkImage m_DisplayImage = VK_NULL_HANDLE;
        VmaAllocation m_DisplayAllocation = VK_NULL_HANDLE;
        VkImageView m_DisplayView = VK_NULL_HANDLE;

        // Dense-traced-entity-index -> materialID (built once from core::EntityData at Init).
        GpuBuffer m_TracedMaterialIDBuffer;
        // Host-visible, persistently-mapped per-frame view params UBO (single-frame-in-flight, so a
        // plain memcpy each frame is safe -- see RecordFrame()).
        GpuBuffer m_ViewParamsBuffer;

        // RT path-trace pipeline (set 0: accum image + TLAS + geometry + materialID + material table
        // + lights + view params).
        VkDescriptorSetLayout m_TraceSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_TraceDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_TraceSet = VK_NULL_HANDLE;
        VkPipelineLayout m_TracePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_TracePipeline = VK_NULL_HANDLE;

        // Shader Binding Table -- one dedicated shaderGroupBaseAlignment-aligned buffer per region,
        // exactly like SurfaceCacheRayTracingPass.
        GpuBuffer m_RaygenSBT;
        GpuBuffer m_MissSBT;
        GpuBuffer m_HitSBT;
        VkStridedDeviceAddressRegionKHR m_RaygenRegion{};
        VkStridedDeviceAddressRegionKHR m_MissRegion{};
        VkStridedDeviceAddressRegionKHR m_HitRegion{};
        VkStridedDeviceAddressRegionKHR m_CallableRegion{}; // Unused -- stays zeroed.

        // Resolve/tonemap compute pipeline (set 0: accum image readonly + display image writeonly).
        VkDescriptorSetLayout m_ResolveSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_ResolveDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_ResolveSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ResolvePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ResolvePipeline = VK_NULL_HANDLE;

        // Progressive-accumulation state.
        uint32_t m_AccumulatedSamples = 0;
        maths::mat4 m_PrevCameraView{};
        bool m_HasPrevCameraView = false;
    };

}

#endif // NDEBUG
