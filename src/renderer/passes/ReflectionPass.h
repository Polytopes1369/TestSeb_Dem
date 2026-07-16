#pragma once
// Phase 2 (UE5.8 parity roadmap): specular reflections / GI, the Lumen pillar this engine had
// entirely none of before this pass -- see this repo's own 2026-07-16 audit (grep for
// GGX|BRDF|Fresnel|Schlick|NDF|ImportanceSample across src/shaders found zero real hits before
// include/ggx_brdf.glsl was written). Structurally mirrors renderer::ScreenProbeGIPass's own 4-call
// contract (RecordUpdateViewParams -> RecordTrace -> RecordTemporal -> RecordGather, double-buffered
// ping-pong, no copy pass) -- see that class' own class comment for the shared rationale -- but
// differs in two deliberate ways:
//
//   1. FULL RESOLUTION, not per-8x8-tile probes: a reflection must stay sharp at low roughness,
//      which a coarse probe tile would immediately blur away. One invocation per pixel in every
//      stage, dispatched (ceil(width/8), ceil(height/8), 1).
//   2. RAW RGBA16F RADIANCE, not L1 spherical harmonics: an SH order-1 basis can only reconstruct a
//      low-frequency cosine-lobe response -- physically incapable of representing a narrow specular
//      lobe at low roughness. ScreenProbeGIPass's SH storage is therefore not reusable here; only
//      its trace->temporal->gather CALL STRUCTURE is mirrored.
//
// --- Per-frame call order a caller must follow (identical shape to ScreenProbeGIPass's own) ---
//   1. RecordUpdateViewParams(cmd, viewProj, prevViewProj, cameraPositionWorld) -- uploads the
//      shared ReflectionViewParamsUBO every other call below reads; `prevViewProj` should be an
//      identity matrix on the very first frame ever recorded (see
//      renderer::ClusterRenderPipeline's own m_HasPrevViewProj guard).
//   2. RecordTrace(cmd, traceContext, entityCount, traceMode, frameIndex) -- needs
//      renderer::ClusterResolvePass's GBuffer (normal/depth/roughness-metallic) already visible to
//      COMPUTE_SHADER reads, and renderer::SurfaceCachePass's radiance atlas +
//      renderer::SurfaceCacheRayTracingPass's TLAS already visible.
//   3. RecordTemporal(cmd) -- needs RecordTrace's own trailing barrier (this class inserts one).
//   4. RecordGather(cmd) -- needs RecordTemporal's own trailing barrier (this class inserts one),
//      and renderer::ClusterResolvePass's output color image already in GENERAL (true for that
//      image's entire lifetime). Read-modify-writes that color image directly (additive, Fresnel-
//      weighted) -- the exact same direct-RMW compositing convention ScreenProbeGather.comp already
//      uses for diffuse GI, see ReflectionGather.comp's own header comment.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;
    class ClusterResolvePass;

    class ReflectionPass {
    public:
        ReflectionPass() = default;

        ReflectionPass(const ReflectionPass&) = delete;
        ReflectionPass& operator=(const ReflectionPass&) = delete;

        static constexpr uint32_t kGridWorkgroupSize = 8; // Every stage's local_size_x/y (full-res dispatch).

        static constexpr VkFormat kRadianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // rgb = radiance, a = validity.
        static constexpr VkFormat kWorldPosFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // rgb = world pos, a = validity.
        static constexpr VkFormat kNormalFormat = VK_FORMAT_R16G16_SFLOAT;         // Octahedral-encoded.

        // `traceContext`/`surfaceCache`/`rtPass`/`resolvePass` must all already be Init'd and must
        // outlive this pass -- their resources are bound into this pass' own descriptor sets
        // unmodified (mirrors renderer::ScreenProbeGIPass's own borrowing convention).
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, const SurfaceCacheTraceContext& traceContext,
            const SurfaceCachePass& surfaceCache, const SurfaceCacheRayTracingPass& rtPass,
            const ClusterResolvePass& resolvePass);

        void Shutdown();

        // Uploads this frame's ReflectionViewParamsUBO (invViewProj, prevViewProj,
        // cameraPositionWorld, viewport size) -- `viewProj` is this frame's combined matrix (its
        // inverse is computed here, CPU-side, via maths::mat4::Inverse()); `prevViewProj` is the
        // previous frame's own combined matrix (renderer::ClusterRenderPipeline's own
        // m_PrevViewProj); `cameraPositionWorld` is the eye point RecordTrace's view-dependent GGX-
        // VNDF sampling needs (unlike ScreenProbeGIPass's own full-sphere probe rays, a reflection
        // ray direction depends on the view vector).
        void RecordUpdateViewParams(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const maths::mat4& prevViewProj, const maths::vec3& cameraPositionWorld);

        // Flips m_CurrentSlotIndex FIRST (so this frame writes into whichever slot held the
        // PREVIOUS frame's history, and that previous frame's now-stale "current" slot becomes this
        // frame's reprojection source), then GGX-VNDF-importance-samples ONE reflection ray per
        // pixel and traces it via SWRT (traceMode = 0) or HWRT (traceMode = 1), writing
        // {radiance, validity} plus this pixel's own world position/normal into the new current
        // slot. RecordTemporal()/RecordGather() called right after this see that same slot as
        // "current".
        void RecordTrace(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
            uint32_t entityCount, uint32_t traceMode, uint32_t frameIndex);

        // Reprojects + roughness-weighted-exponentially blends the current slot's raw new radiance
        // sample against the history slot, in place -- see ReflectionTemporal.comp's own header
        // comment for the roughness-dependent blend factor rationale.
        void RecordTemporal(VkCommandBuffer cmd);

        // Full-resolution, pixel-aligned (no bilateral neighborhood search needed, unlike
        // ScreenProbeGather.comp) Fresnel-weighted composite, read-modify-writing `resolvePass`'s
        // output color image (the same instance passed to Init()).
        void RecordGather(VkCommandBuffer cmd);

    private:
        // One ping-pong slot's 3 fields, all sized to `m_RenderExtent` (full resolution, unlike
        // ScreenProbeGIPass::ProbeSlot's coarser probe-grid sizing).
        struct ReflectionSlot {
            VkImage radianceImage = VK_NULL_HANDLE; VmaAllocation radianceAllocation = VK_NULL_HANDLE; VkImageView radianceView = VK_NULL_HANDLE;
            VkImage worldPosImage = VK_NULL_HANDLE; VmaAllocation worldPosAllocation = VK_NULL_HANDLE; VkImageView worldPosView = VK_NULL_HANDLE;
            VkImage normalImage = VK_NULL_HANDLE; VmaAllocation normalAllocation = VK_NULL_HANDLE; VkImageView normalView = VK_NULL_HANDLE;
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        VkExtent2D m_RenderExtent{ 0, 0 };

        ReflectionSlot m_Slots[2];
        uint32_t m_CurrentSlotIndex = 0;

        VkSampler m_ReflectionSampler = VK_NULL_HANDLE; // Linear, clamp-to-edge -- history reprojection taps.

        // Shared view-params UBO (ReflectionViewParamsUBO, std140) -- bound identically into every
        // pass' own set 0.
        GpuBuffer m_ViewParamsBuffer;

        // Trace: 2 variants (indexed by m_CurrentSlotIndex at record time) since which physical slot
        // is "current" (write target) flips every frame.
        VkDescriptorSetLayout m_TraceSetLayout = VK_NULL_HANDLE; // set 0.
        VkDescriptorPool m_TraceDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_TraceSet[2]{};
        VkPipelineLayout m_TracePipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_TracePipeline = VK_NULL_HANDLE;

        // Temporal: 2 variants, same reason.
        VkDescriptorSetLayout m_TemporalSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_TemporalDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_TemporalSet[2]{};
        VkPipelineLayout m_TemporalPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_TemporalPipeline = VK_NULL_HANDLE;

        // Gather: 2 variants, same reason.
        VkDescriptorSetLayout m_GatherSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_GatherDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_GatherSet[2]{};
        VkPipelineLayout m_GatherPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_GatherPipeline = VK_NULL_HANDLE;
    };

}
