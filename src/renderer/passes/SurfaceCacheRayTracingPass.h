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

#include "core/EntityData.h" // core::EntityTransformCPU
#include "renderer/vulkan/AccelerationStructure.h"
#include "renderer/vulkan/GpuBuffer.h"

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
        //
        // Skeletal-animation feature (ray-traced GI/reflections fix): `boneMatricesBuffer`
        // (animation::SkeletalAnimator::GetBoneMatricesBuffer(), the SAME per-frame SSBO
        // ClusterRaster.vert already binds) and `entityDataCPU` (index == meshID, the SAME array
        // renderer::VirtualShadowMapPass::Init already receives) together let this method find the
        // ONE traced entity with core::EntityFlags::IsSkeletallyAnimated set (at most one, by this
        // codebase's own design -- see CLAUDE.md's scope note on this being a single-creature
        // feature) and give ONLY that entity's BLAS the ALLOW_UPDATE build flags + a dedicated
        // per-frame-skinned vertex buffer + compute pipeline (see RecordCreatureBlasUpdate below) --
        // every other entity's BLAS is built exactly as before, unaffected. `entityDataCPU == nullptr`
        // is tolerated (skips creature detection entirely, degrading to this feature's pre-fix
        // behavior) but every real caller in this codebase always has a valid array to pass.
        bool Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
            VkCommandPool commandPool, VkQueue queue,
            const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
            VkBuffer boneMatricesBuffer, const core::EntityData* entityDataCPU);

        void Shutdown();

        // (Re)writes set 0's ray-request/result bindings -- same contract as
        // SurfaceCacheSWRTPass::SetRayBuffers(). Must be called at least once before RecordTrace().
        void SetRayBuffers(VkBuffer rayBuffer, VkDeviceSize rayBufferSize, VkBuffer resultBuffer, VkDeviceSize resultBufferSize);

        // Dispatches vkCmdTraceRaysKHR over a 1D launch of `rayCount` rays. Caller owns every
        // synchronization barrier before/after this call, same discipline as
        // SurfaceCacheSWRTPass::RecordTrace().
        void RecordTrace(VkCommandBuffer cmd, uint32_t rayCount, const SurfaceCacheTraceContext& traceContext);

        // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): rebuilds m_TLAS's
        // instance transforms from this frame's `entityTransformsCPU` (renderer::VulkanContext::
        // GetEntityTransformsCPU(), index == meshID) and records a full refit DIRECTLY into `cmd`
        // (see AccelerationStructure.h's own RecordRefitTLAS comment -- no separate submit/wait,
        // unlike Init()'s one-shot BuildTLAS). BLAS geometry/device addresses are untouched (rigid
        // transform only, no vertex displacement). Must be called before anything else this frame
        // traces against GetTLASHandle() -- see ClusterRenderPipeline::RecordFrame's own call site
        // comment for why it runs first in the [1z] GI block.
        void RecordRefreshTLAS(VkCommandBuffer cmd, const core::EntityTransformCPU* entityTransformsCPU);

        // Skeletal-animation feature (ray-traced GI/reflections fix): dispatches
        // CreatureBlasSkinning.comp (writes this frame's skinned Fallback Mesh vertex positions into
        // m_CreatureSkinnedVertexBuffer from `entityTransformsCPU`'s current bone matrices), records
        // the compute-write -> AS-build-read barrier, then calls RecordUpdateBLAS to refit the
        // creature's BLAS in UPDATE mode -- a no-op (returns immediately) if Init() found no
        // skeletally-animated entity. MUST be called:
        //   - AFTER animation::SkeletalAnimator::RecordUpdate has uploaded THIS frame's bone
        //     matrices (this method's own compute dispatch reads that SSBO) -- see
        //     renderer::ClusterRenderPipeline::RecordFrameEarly's own call-site comment for why that
        //     upload was moved earlier in the frame specifically to satisfy this.
        //   - ON THE SAME command buffer/queue as, and immediately BEFORE, whichever RecordRefreshTLAS
        //     call executes this same frame (the TLAS refit reads this BLAS's device address/bounds,
        //     so it must observe this update's result, not a stale one -- see RecordRefreshTLAS's own
        //     comment and renderer::ClusterRenderPipeline::RecordFrameEarly/RecordAsyncCompute's own
        //     two call sites for the exact fallback-path/async-compute-path pairing this requires).
        void RecordCreatureBlasUpdate(VkCommandBuffer cmd, const core::EntityTransformCPU* entityTransformsCPU);

        // KNOWN, DOCUMENTED SIMPLIFICATION (mirrors renderer::ClusterRenderPipeline::
        // RecordSurfaceCacheOwnershipTransfer's own identical, already-accepted precedent for
        // TlasRefitResources -- see that method's own header comment): m_CreatureSkinnedVertexBuffer,
        // the creature BLAS's own backing buffer, and m_CreatureBlasUpdateResources.scratchBuffer are
        // NOT included in any cross-queue ownership-transfer barrier. All 3 are written and read
        // ENTIRELY within one RecordCreatureBlasUpdate() call, always on whichever single queue that
        // call's own `cmd` targets THIS frame (the SAME queue RecordRefreshTLAS runs on this same
        // frame, by the caller's own sequencing contract -- see RecordCreatureBlasUpdate's own
        // comment) -- so within a frame, and across consecutive frames with the SAME routing
        // decision, no cross-queue hazard exists at all. The only residual risk, exactly like
        // TlasRefitResources' own accepted one, is SetDebugAsyncComputeEnabled() toggling routing
        // between two consecutive frames (rare, Debug-only) leaving one of these 3 buffers "owned"
        // by whichever queue family last touched it -- accepted here for the same reason: these are
        // purely-internal, single-call-scoped resources with no other consumer, not worth a third
        // whole-resource-group ownership transfer (alongside RecordSurfaceCacheOwnershipTransfer's
        // existing 9-resource one and ClusterRenderPipeline's own dedicated bone-matrices-buffer one,
        // which DOES need real per-frame handling -- see that method's own comment for why the
        // bone-matrices case is NOT analogous: it is read every async-compute-routed frame by THIS
        // method's own compute dispatch, not just on a rare toggle).
        bool HasCreatureBlas() const { return m_CreatureBlasListIndex != kInvalidBlasListIndex; }

        // Exposed so renderer::SurfaceCacheGIInjectPass can bind the SAME TLAS + draw-range table
        // into its own inline-ray-query (VK_KHR_ray_query) descriptor set, instead of building a
        // second TLAS over the same geometry -- see SurfaceCacheGIInject.comp's own TraceHWRT.
        VkAccelerationStructureKHR GetTLASHandle() const { return m_TLAS.Handle(); }
        VkBuffer GetDrawRangeBuffer() const { return m_DrawRangeBuffer.Handle(); }
        // Phase 2 (Lumen advanced roadmap): the TLAS's own backing storage buffer (renderer::
        // AccelerationStructure::m_Buffer) -- needed by renderer::ClusterRenderPipeline::RecordFrame
        // to record a real VkBufferMemoryBarrier2 queue-family-ownership-transfer pair around it when
        // RecordRefreshTLAS runs on the async-compute queue. VkAccelerationStructureKHR itself is not
        // a barrier-typed resource (per the Vulkan spec, ownership-transfer barriers operate on the
        // underlying VkBuffer memory, not the acceleration structure handle) -- this exposes exactly
        // that underlying buffer, nothing else.
        VkBuffer GetTLASBufferHandle() const { return m_TLAS.m_Buffer.Handle(); }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        std::vector<AccelerationStructure> m_BLASList; // Index-aligned with SurfaceCacheTraceContext::GetTracedEntities().
        AccelerationStructure m_TLAS;

        // Phase 4 integration: persistent per-frame TLAS refit buffers (see AccelerationStructure.h's
        // own TlasRefitResources comment) + the per-BLAS bookkeeping RecordRefreshTLAS needs to
        // rebuild `instances` every frame. NOT simply index-aligned with GetTracedEntities() --
        // Init()'s own loop below SKIPS (via `continue`) any traced entity with empty Fallback Mesh
        // geometry, so m_BLASList/m_RefitInstanceInfos are a COMPACTED subset, in traversal order,
        // of the entities that actually got a BLAS -- each entry explicitly records which
        // tracedEntities index (for instanceCustomIndex) and which entityID (to look up this
        // frame's rotation in entityTransformsCPU) it corresponds to, rather than assuming any
        // implicit 1:1 alignment.
        struct RefitInstanceInfo {
            uint32_t instanceCustomIndex = 0; // == the tracedEntities dense index (matches SurfaceCacheHWRT.rchit's own resolution).
            uint32_t entityID = 0;            // meshID -- indexes entityTransformsCPU.
        };
        std::vector<RefitInstanceInfo> m_RefitInstanceInfos; // Index-aligned with m_BLASList.
        TlasRefitResources m_TlasRefitResources;

        GpuBuffer m_DrawRangeBuffer; // set 3, binding 2 -- EntityDrawRangeGpu[], index-aligned with m_BLASList.

        // =====================================================================================
        // Skeletal-animation feature (ray-traced GI/reflections fix): per-frame BLAS refit state
        // for the ONE traced entity with core::EntityFlags::IsSkeletallyAnimated set (found during
        // Init()'s own per-entity loop -- see that method's own comment). kInvalidBlasListIndex
        // (never a valid m_BLASList index) means "no skeletally-animated entity was traced this
        // Init() call" -- RecordCreatureBlasUpdate is then a no-op and none of the resources below
        // are ever allocated.
        // =====================================================================================
        static constexpr uint32_t kInvalidBlasListIndex = 0xFFFFFFFFu;
        uint32_t m_CreatureBlasListIndex = kInvalidBlasListIndex; // Index into m_BLASList/m_RefitInstanceInfos.
        uint32_t m_CreatureEntityID = 0; // meshID -- indexes entityTransformsCPU in RecordCreatureBlasUpdate.
        uint32_t m_CreatureSrcVertexOffset = 0; // Element offset into surfaceCache.GetVertexBuffer() where this entity's span begins.
        uint32_t m_CreatureVertexCount = 0;     // Element count of that span == m_CreatureSkinnedVertexBuffer's own element count.
        uint32_t m_CreatureMaxVertex = 0;       // BLAS triangles.maxVertex, mirrors every other entity's own conservative-bound derivation (see Init()'s .cpp comment).
        VkBuffer m_CreatureIndexBuffer = VK_NULL_HANDLE;   // Shared surfaceCache.GetIndexBuffer() -- topology never changes, no dedicated copy needed.
        VkDeviceSize m_CreatureIndexOffsetBytes = 0;
        uint32_t m_CreatureTriangleCount = 0;

        // Dedicated per-frame-skinned Fallback Mesh vertex buffer (geometry::FallbackVertex-shaped,
        // GPU_ONLY) -- written every frame by CreatureBlasSkinning.comp, then read as RecordUpdateBLAS's
        // own UPDATE-mode geometry input. Kept SEPARATE from surfaceCache's own shared, static vertex
        // buffer (which every OTHER entity's one-shot BLAS still reads directly, and which this
        // pass' own m_GeometrySet/SurfaceCacheHWRT.rchit continue to sample unmodified -- see
        // CreatureBlasSkinning.comp's own header comment for why that shared buffer is deliberately
        // left untouched).
        GpuBuffer m_CreatureSkinnedVertexBuffer;
        BlasUpdateResources m_CreatureBlasUpdateResources;

        // CreatureBlasSkinning.comp's own compute pipeline + dedicated 3-binding descriptor set
        // (source vertex buffer / bone matrices SSBO / dest skinned vertex buffer, all COMPUTE-stage
        // STORAGE_BUFFER) -- kept separate from m_RaySetLayout/m_GeometrySetLayout above (different
        // pipeline bind point entirely, VK_PIPELINE_BIND_POINT_COMPUTE not ...RAY_TRACING_KHR).
        VkDescriptorSetLayout m_SkinningSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_SkinningDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_SkinningSet = VK_NULL_HANDLE;
        VkPipelineLayout m_SkinningPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_SkinningPipeline = VK_NULL_HANDLE;

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
