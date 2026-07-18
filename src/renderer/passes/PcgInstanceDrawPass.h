#pragma once
// Phase 0.2 (UE5.8-parity PCG roadmap, "PCG Instance Draw Path"): a GPU-driven instanced-mesh draw
// path for PCG-spawned content (rocks/trees/bushes/...), distinct from renderer::ParticleSystemPass
// (which hardcodes a billboard quad, ParticleRender.vert -- no per-instance mesh reference) -- this
// class draws real, baked Nanite cluster geometry per instance, each with its own meshID,
// materialID, and world transform, through the SAME cluster-culling/indirect-draw contract every
// fixed scene entity uses (renderer::ClusterCullingPass, composed here unmodified -- see that
// class' own header comment: "self-contained building block ... not wired into VulkanContext/
// main.cpp by this change", which this pass is the first real consumer of).
//
// --- Why this is a SEPARATE pool from core::InstanceRegistry / the fixed EntityTransformBuffer ---
// core::InstanceRegistry (Phase 0.1) is a CPU-side core::EntityData/EntityTransformCPU slot
// allocator backing the engine's ONE fixed, GPU-uploaded-wholesale entity array -- every entity in
// it gets exactly one EntityTransform slot (indexed by meshID) and its clusters are woven into the
// single shared Nanite DAG built once at cache-build time (see ClusterLODSelectionPass's own class
// comment). It has no notion of "many instances of the same baked mesh at different transforms."
// This class is the opposite shape: a bounded pool of DRAW instances that may freely repeat the
// SAME meshID (and therefore the same already-resident cluster geometry) at many different world
// transforms -- exactly what a future PCG scatter (Phase 4, the real spawner) needs to place, say,
// a thousand rocks from one baked archetype mesh. Reusing the fixed entity array for that would
// require either pre-baking one EntityTransform slot per POSSIBLE instance (wasteful, and still
// capped at compile time) or overloading a shared per-meshID transform with per-instance meaning
// (impossible -- one meshID, one transform, by that system's own design). So this pool owns its
// OWN small per-instance GPU transform buffer (GpuInstanceData) instead, entirely additively -- it
// does not replace or modify core::InstanceRegistry, the fixed EntityTransformBuffer, or anything
// ClusterLODSelectionPass/ClusterOcclusionCullingPass/ClusterHardwareRasterPass do for the existing
// fixed scene.
//
// --- Why this does NOT go through ClusterLODSelectionPass's per-frame DAG cut ---
// ClusterLODSelectionPass's GPU-driven LOD cut operates over ONE shared DAGNodesBuffer/
// LODNodeMetadataBuffer built once at Init() from the full scene's cache tables, keyed by
// clusterID/entityID exactly as baked -- it has no notion of "the same baked mesh, drawn twice with
// two different transforms and therefore two different on-screen LOD requirements." Building a
// genuinely per-instance LOD cut is real, non-trivial future work (Phase 4's own scope, per this
// class' own header note to that phase) -- Phase 0.2's job is proving the draw-instance plumbing
// itself works end-to-end, not a full LOD system for it. This class therefore always draws every
// acquired instance's LEAF-level (full detail, geometry::DAGNodeEntry::level == 0) clusters only,
// exactly like this codebase's OWN pre-ClusterLODSelectionPass baseline once did for the whole
// scene (see that class' own header comment: "replaces ... the previous static, CPU-baked 'always
// DAG level 0' candidate list").
//
// --- The actual draw pipeline (this class owns its own graphics pipeline, PcgInstanceDraw.vert/
// .frag -- see those shaders' own header comments for the full per-vertex contract) ---
// Composed on top of renderer::ClusterCullingPass (owned by value, m_Culling below) for the GPU
// frustum+backface cull and atomic compaction into a VkDrawIndexedIndirectCommand buffer with a
// GPU-computed (never CPU-computed) draw count -- UploadInstances() below is the only place this
// class touches the CPU/GPU boundary for instance data; RecordCull()'s survivor count and
// RecordDraw()'s vkCmdDrawIndexedIndirectCount instance count are both entirely GPU-side from then
// on, exactly like every other consumer of this codebase's Nanite cluster pipeline.
//
// --- Per-frame (or per-rebuild) sequence a caller must record, in order ---
//   1. AcquireInstance(...) / ReleaseInstance(...) -- CPU-side free-list bookkeeping only, whenever
//      the instance SET changes (mirrors core::InstanceRegistry's own AcquireSlot/ReleaseSlot LIFO
//      idiom). No GPU work happens here.
//   2. UploadInstances(...) -- whenever the instance set OR any instance's transform/meshID/
//      materialID changed since the last call: rebuilds every acquired instance's candidate leaf-
//      cluster list from the cache's full index/DAG tables and uploads it (see that method's own
//      comment). May be skipped on frames that reuse the previous upload unchanged.
//   3. RecordClear(cmd) -- resets the culling pass' draw counter, every frame.
//   4. RecordCull(cmd, viewParams) -- GPU frustum+backface cull against the most recent
//      UploadInstances() candidate list, every frame the instances should be considered for
//      drawing.
//   5. RecordDraw(cmd, camera, renderExtent, decompressedIndexPoolBuffer) -- must be called inside
//      an already-open vkCmdBeginRendering scope targeting `colorFormat`/`depthFormat` (Init()'s own
//      parameters) -- this class does not open that scope itself, matching every other pass in this
//      codebase's own "self-contained building block" convention (see e.g.
//      ClusterHardwareRasterPass's own class comment).

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/Camera.h" // CameraPushConstants
#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"
#include "renderer/MaterialParameterTable.h"
#include "renderer/passes/ClusterCullingPass.h"
#include "renderer/streaming/GpuGeometryPagePool.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class PcgInstanceDrawPass {
    public:
        PcgInstanceDrawPass() = default;

        PcgInstanceDrawPass(const PcgInstanceDrawPass&) = delete;
        PcgInstanceDrawPass& operator=(const PcgInstanceDrawPass&) = delete;

        static constexpr uint32_t kInvalidInstance = 0xFFFFFFFFu;

        // Byte-for-byte, std430-compatible mirror of pcg_instance_draw_common.glsl's PcgDrawInstance
        // -- 80 bytes (one 64-byte mat4 block + one 16-byte scalar block). Exposed publicly only so
        // a caller with unusual diagnostic needs could read it back; UploadInstances() is the only
        // normal way to populate it.
        struct GpuInstanceData {
            maths::mat4 localToWorld;
            uint32_t materialID = 0;
            uint32_t _pad0 = 0, _pad1 = 0, _pad2 = 0;
        };
        static_assert(sizeof(GpuInstanceData) == 80,
            "GpuInstanceData must match pcg_instance_draw_common.glsl's PcgDrawInstance exactly (std430 layout)");

        // Byte-for-byte, std430-compatible mirror of pcg_instance_draw_common.glsl's
        // PcgClusterLocalBounds -- 32 bytes. See that struct's own GLSL-side comment for why this
        // must stay a separate array from ClusterCullMetadata::boundsMin/boundsMax.
        struct LocalBoundsGpu {
            maths::vec3 boundsMin; float _pad0 = 0.0f;
            maths::vec3 boundsMax; float _pad1 = 0.0f;
        };
        static_assert(sizeof(LocalBoundsGpu) == 32,
            "LocalBoundsGpu must match pcg_instance_draw_common.glsl's PcgClusterLocalBounds exactly (std430 layout)");

        // Allocates every buffer/descriptor/pipeline this pass owns:
        //   - m_Culling (renderer::ClusterCullingPass) sized for `maxCandidateClusters` candidate
        //     clusters -- see that class' own Init() for exactly what it allocates.
        //   - the per-instance GPU transform/material buffer, sized for `maxInstances` slots.
        //   - the per-candidate-cluster LOCAL-space bounds buffer, sized for `maxCandidateClusters`
        //     entries (index-aligned with m_Culling's own ClusterCullMetadata array).
        //   - a private copy of `materialTable`'s PBR parameters (renderer::MaterialParameters[
        //     kMaxMaterials]), uploaded ONCE here via a one-shot staged copy (`commandPool`/`queue`)
        //     -- this pass never re-reads the live scene's own ClusterResolvePass material SSBO, so
        //     it has zero runtime coupling to that pass' lifetime.
        //   - a small fixed-lighting UBO (sun direction/color/intensity + flat ambient), also
        //     uploaded once here from `sunDirectionWorld`/`sunColor`/`sunIntensity`/`ambientColor` --
        //     see PcgInstanceDraw.frag's own header comment for why this pipeline has no per-frame
        //     lighting update yet (Phase 0.2 scope).
        //   - the PcgInstanceDraw.vert/.frag graphics pipeline itself, targeting `colorFormat`/
        //     `depthFormat` via dynamic rendering (opaque: depth test+write ON, VK_COMPARE_OP_GREATER
        //     reversed-Z, matching this codebase's site-wide convention -- see maths::mat4::
        //     PerspectiveVulkan's own comment).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkBuffer compressedPhysicalPoolBuffer, const MaterialTable& materialTable,
            const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
            const maths::vec3& ambientColor,
            VkFormat colorFormat, VkFormat depthFormat,
            uint32_t maxInstances, uint32_t maxCandidateClusters);

        void Shutdown();

        // --- CPU-side free-list allocator (LIFO, mirrors core::InstanceRegistry::AcquireSlot/
        // ReleaseSlot exactly) -- NOT thread-safe, main-thread-only, same contract as that class.
        // Neither call touches the GPU; call UploadInstances() afterward to actually publish any
        // change. `position`/`rotation`/`scale` compose this instance's local-to-world transform
        // exactly like pcg::PcgPoint::GetLocalToWorld() (Translate * FromQuat * Scale). Returns
        // kInvalidInstance if the pool is completely exhausted (mirrors core::InstanceRegistry::
        // kInvalidSlot). `meshID` must name an entity whose clusters are already resident in the
        // physical pool passed to Init() (compressedPhysicalPoolBuffer) -- this pass does no
        // streaming of its own, see this class' own header comment. ---
        uint32_t AcquireInstance(uint32_t meshID, uint32_t materialID,
            const maths::vec3& position, const maths::quat& rotation, const maths::vec3& scale);
        void ReleaseInstance(uint32_t instanceSlot);

        uint32_t GetLiveInstanceCount() const { return m_LiveCount; }
        uint32_t GetMaxInstances() const { return m_MaxInstances; }

        // Rebuilds the candidate leaf-cluster list for every currently-acquired instance and
        // uploads it (see this class' own header comment, step 2). For each acquired instance,
        // scans `indexEntries`/`dagEntries` (the SAME full, index-aligned cache tables
        // ClusterLODSelectionPass::Init() itself consumes -- geometry::ClusterIndexEntry::entityID
        // filtered against this instance's own `meshID`, geometry::DAGNodeEntry::level == 0 for
        // leaf/full-detail clusters only, see this class' own header comment on LOD scope) and, for
        // every matching cluster:
        //   - resolves its current physical page via `pagePool.GetPhysicalPageIndex()` (skipped,
        //     with a logged warning, if not resident -- should never happen for content streamed in
        //     at ClusterRenderPipeline::Init() time, see that class' own "everything ... already
        //     resident before the first frame" guarantee) to derive firstIndex/vertexOffset exactly
        //     like ClusterRenderPipeline::Init()'s own STEP 4 loop does;
        //   - transforms the cluster's LOCAL-space AABB corners / bounding sphere / cone axis by
        //     this instance's own world transform into WORLD-space ClusterCullMetadata fields (see
        //     this class' own header comment on why -- renderer::ClusterCullingPass's frustum/
        //     backface tests need world-space input, since it has no entity-transform buffer of its
        //     own to apply one internally, unlike the shared ClusterLODCompact.comp path);
        //   - stamps ClusterCullMetadata::entityID with this instance's OWN slot index (repurposed
        //     meaning, see PcgInstanceDraw.vert's own header comment) and ::materialID with this
        //     instance's own materialID (NOT necessarily the cluster's originally-baked materialID
        //     -- a PCG instance may recolor/re-material an archetype mesh freely).
        // Returns the number of candidate clusters actually uploaded (0 if no instance is currently
        // acquired, or none of their meshIDs have any matching leaf cluster in the supplied tables).
        // Logs and clamps (never overflows either GPU buffer) if the total candidate count would
        // exceed `maxCandidateClusters` (Init()'s own parameter).
        uint32_t UploadInstances(VkCommandPool commandPool, VkQueue queue,
            const std::vector<geometry::ClusterIndexEntry>& indexEntries,
            const std::vector<geometry::DAGNodeEntry>& dagEntries,
            const GpuGeometryPagePool& pagePool);

        // Forwards to m_Culling.RecordClear() -- see renderer::ClusterCullingPass::RecordClear's own
        // comment. Must be recorded once per frame (or per re-cull), before RecordCull().
        void RecordClear(VkCommandBuffer cmd);

        // Forwards to m_Culling.RecordCull(), passing the candidate count from the most recent
        // UploadInstances() call -- see renderer::ClusterCullingPass::RecordCull's own comment for
        // the full frustum/backface-cull + atomic-compaction contract.
        void RecordCull(VkCommandBuffer cmd, const ClusterCullViewParams& viewParams);

        // Records one draw of every currently-culled-visible instance: binds this pass' own
        // graphics pipeline + descriptor set, pushes `camera` (view/proj, matching
        // PcgInstanceDraw.vert's own {mat4 view; mat4 proj;} push-constant layout exactly), binds
        // `decompressedIndexPoolBuffer` (renderer::GeometryDecompressionPass::
        // GetDecompressedIndexPoolBuffer(), VK_INDEX_TYPE_UINT32) as the active index buffer, sets
        // the dynamic viewport/scissor to `renderExtent`, then issues exactly one
        // vkCmdDrawIndexedIndirectCount over m_Culling's own indirect-command/draw-count buffers
        // (GPU-computed instance count, never read back to the CPU -- see this class' own header
        // comment). Must be called inside an already-open vkCmdBeginRendering scope targeting
        // Init()'s `colorFormat`/`depthFormat` attachments -- this class does not open that scope
        // itself (see this class' own header comment, step 5).
        void RecordDraw(VkCommandBuffer cmd, const CameraPushConstants& camera, VkExtent2D renderExtent,
            VkBuffer decompressedIndexPoolBuffer);

        // Exposed for Debug-only validation (renderer::ClusterRenderPipeline::
        // RunPcgInstanceDrawSmokeTest): the GPU-atomic draw count RecordCull() most recently wrote,
        // and the CPU-known upper bound RecordDraw()'s vkCmdDrawIndexedIndirectCount call passes as
        // its own maxDrawCount -- a caller wanting to actually read the GPU-computed count back must
        // copy GetDrawCountBuffer() itself (this pass keeps no host-visible mirror of it, matching
        // renderer::ClusterCullingPass's own "GPU-only, no CPU readback" contract).
        VkBuffer GetDrawCountBuffer() const { return m_Culling.GetDrawCountBuffer(); }
        uint32_t GetLastCandidateClusterCount() const { return m_LastCandidateClusterCount; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        ClusterCullingPass m_Culling;

        GpuBuffer m_InstanceBuffer;       // GpuInstanceData[maxInstances], std430, GPU_ONLY.
        GpuBuffer m_LocalBoundsBuffer;    // LocalBoundsGpu[maxCandidateClusters], std430, GPU_ONLY.
        GpuBuffer m_MaterialParamsBuffer; // MaterialParameters[kMaxMaterials], std430, GPU_ONLY, written once at Init().
        GpuBuffer m_LightingParamsBuffer; // PcgLightingParamsUBO, std140, GPU_ONLY, written once at Init().

        uint32_t m_MaxInstances = 0;
        uint32_t m_MaxCandidateClusters = 0;
        uint32_t m_LastCandidateClusterCount = 0;

        // --- CPU-side instance bookkeeping -- see AcquireInstance()/ReleaseInstance()'s own
        // comment. Mirrors core::InstanceRegistry's own private-member shape exactly (LIFO free
        // list + occupancy flags + monotonic high-water mark), just without that class' template
        // parameter (this pool's capacity, m_MaxInstances, is a runtime Init() argument instead). ---
        struct InstanceSlotCPU {
            uint32_t meshID = 0;
            uint32_t materialID = 0;
            maths::vec3 position{};
            maths::quat rotation{};
            maths::vec3 scale{ 1.0f, 1.0f, 1.0f };
        };
        std::vector<InstanceSlotCPU> m_Slots;
        std::vector<bool> m_Occupied;
        std::vector<uint32_t> m_FreeList;
        uint32_t m_HighWaterMark = 0;
        uint32_t m_LiveCount = 0;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
