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
#include "renderer/vulkan/RenderPass.h"

namespace renderer {

    class SurfaceCacheTraceContext;
    class SurfaceCachePass;
    class SurfaceCacheRayTracingPass;
    class ClusterResolvePass;

    // Migrated to RenderPass<Derived> (see renderer/vulkan/RenderPass.h): Init()/Shutdown() are
    // inherited. Note InitImpl() still calls the inherited Shutdown() as its own first statement
    // (self-reinit, same pattern/reason as ShadowMapPass's own migration comment). The 3 descriptor
    // pools here (Trace/Temporal/Gather) each allocate 2 sets (maxSets=2, one per ping-pong slot) --
    // VulkanUtils::CreateDescriptorSetLayoutPoolAndSet only covers the far more common maxSets=1
    // case, so these 3 stay raw vkCreateDescriptorSetLayout/Pool/AllocateDescriptorSets calls,
    // wrapped in RegisterResource() rather than routed through that helper.
    class ReflectionPass : public RenderPass<ReflectionPass> {
        friend class RenderPass<ReflectionPass>; // Lets Init() call our private InitImpl().

    public:
        ReflectionPass() = default;

        ReflectionPass(const ReflectionPass&) = delete;
        ReflectionPass& operator=(const ReflectionPass&) = delete;

        static constexpr uint32_t kGridWorkgroupSize = 8; // Every stage's local_size_x/y (full-res dispatch).

        static constexpr VkFormat kRadianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // rgb = radiance, a = validity.
        static constexpr VkFormat kWorldPosFormat = VK_FORMAT_R32G32B32A32_SFLOAT; // rgb = world pos, a = validity.
        static constexpr VkFormat kNormalFormat = VK_FORMAT_R16G16_SFLOAT;         // Octahedral-encoded.

        // Phase 2 (Lumen advanced roadmap): fixed (non-ping-ponged) auxiliary images, same lifetime
        // convention as m_HitMaskImage below (consumed same-frame, never reprojected) -- see this
        // class' own member comments for why each exists.
        static constexpr VkFormat kRawRadianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // Same layout as kRadianceFormat.
        static constexpr VkFormat kHalfVectorFormat = VK_FORMAT_R16G16_SFLOAT;        // Octahedral-encoded, same layout as kNormalFormat.

        // `traceContext`/`surfaceCache`/`rtPass`/`resolvePass` must all already be Init'd and must
        // outlive this pass -- their resources are bound into this pass' own descriptor sets
        // unmodified (mirrors renderer::ScreenProbeGIPass's own borrowing convention).
        // Init(VkDevice, VmaAllocator, VkCommandPool, VkQueue, VkExtent2D,
        // const SurfaceCacheTraceContext&, const SurfaceCachePass&, const SurfaceCacheRayTracingPass&,
        // const ClusterResolvePass&) -> bool and Shutdown() are inherited from
        // RenderPass<ReflectionPass>; see InitImpl() below.

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

        // Phase PP4: 1.0 where RecordTrace()'s ray found real geometry this frame, 0.0 everywhere
        // else (background, grazing-angle skip, or a genuine miss) -- NOT ping-ponged (a single
        // fixed image, unlike the 3 ReflectionSlot fields above), since renderer::
        // ScreenSpaceEffectsPass's SSRFallback.comp consumes it the SAME frame RecordTrace() wrote
        // it, right after RecordGather() -- no cross-frame reprojection needed.
        VkImageView GetHitMaskView() const { return m_HitMaskView; }

    private:
        // One ping-pong slot's 3 fields, all sized to `m_RenderExtent` (full resolution, unlike
        // ScreenProbeGIPass::ProbeSlot's coarser probe-grid sizing).
        struct ReflectionSlot {
            VkImage radianceImage = VK_NULL_HANDLE; VmaAllocation radianceAllocation = VK_NULL_HANDLE; VkImageView radianceView = VK_NULL_HANDLE;
            VkImage worldPosImage = VK_NULL_HANDLE; VmaAllocation worldPosAllocation = VK_NULL_HANDLE; VkImageView worldPosView = VK_NULL_HANDLE;
            VkImage normalImage = VK_NULL_HANDLE; VmaAllocation normalAllocation = VK_NULL_HANDLE; VkImageView normalView = VK_NULL_HANDLE;
        };

        bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, const SurfaceCacheTraceContext& traceContext,
            const SurfaceCachePass& surfaceCache, const SurfaceCacheRayTracingPass& rtPass,
            const ClusterResolvePass& resolvePass);

        // m_Device / m_Allocator are inherited (protected) from RenderPass<ReflectionPass>.

        VkExtent2D m_RenderExtent{ 0, 0 };

        ReflectionSlot m_Slots[2];
        uint32_t m_CurrentSlotIndex = 0;

        // Phase PP4: single fixed (not ping-ponged) hit-mask image -- see GetHitMaskView()'s own comment.
        static constexpr VkFormat kHitMaskFormat = VK_FORMAT_R8_UNORM;
        VkImage m_HitMaskImage = VK_NULL_HANDLE;
        VmaAllocation m_HitMaskAllocation = VK_NULL_HANDLE;
        VkImageView m_HitMaskView = VK_NULL_HANDLE;

        // Phase 2 (Lumen advanced roadmap): raw (pre-temporal-blend) radiance, written by
        // ReflectionTrace.comp alongside its existing per-slot g_ReflectionRadiance write, read
        // read-only by ReflectionTemporal.comp's new 3x3 neighborhood variance-clamp box (mirrors
        // TAATSR.comp's own YCoCg mean/variance AABB technique, factored into include/
        // temporal_variance_clamp.glsl). A FIXED image, like m_HitMaskImage above, not one of the two
        // ping-pong ReflectionSlot fields: ReflectionTemporal.comp reads AND writes the CURRENT slot's
        // g_ReflectionRadiance in place (the roughness-weighted exponential history blend), so a
        // neighboring workgroup's own in-place write could race with this workgroup's neighborhood
        // read if the clamp sampled that same ping-pong image -- this dedicated, write-once-per-frame,
        // read-only-thereafter copy of the raw trace output removes that race entirely.
        VkImage m_RawRadianceImage = VK_NULL_HANDLE;
        VmaAllocation m_RawRadianceAllocation = VK_NULL_HANDLE;
        VkImageView m_RawRadianceView = VK_NULL_HANDLE;

        // Phase 2 (Lumen advanced roadmap): this pixel's GGX-VNDF-sampled half-vector (octahedral-
        // encoded, mirrors g_ReflectionNormal's own encoding), written by ReflectionTrace.comp right
        // after it computes halfVectorWorld for its VNDF sampling. A FIXED image (same "consumed
        // same-frame, never reprojected" rationale as m_HitMaskImage): ReflectionGather.comp
        // reconstructs H from this image to derive NdotH/NdotV/NdotL/VdotH and evaluate the complete
        // D_GGX * G_SmithGGXCorrelated * F_Schlick specular BRDF product, replacing the previous
        // Fresnel-only composite weight -- see ReflectionGather.comp's own header comment for why this
        // replaces (not adds to) the old bare Fresnel term.
        VkImage m_HalfVectorImage = VK_NULL_HANDLE;
        VmaAllocation m_HalfVectorAllocation = VK_NULL_HANDLE;
        VkImageView m_HalfVectorView = VK_NULL_HANDLE;

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
