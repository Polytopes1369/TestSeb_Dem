#pragma once
// GPU-driven Nanite-style LOD DAG cut: replaces ClusterRenderPipeline's previous static, CPU-baked
// "always DAG level 0 (full detail)" candidate list with a per-frame, view-dependent cut computed
// entirely on the GPU. See src/shaders/src/Culling/ClusterDAGScreenError.comp (per-node screen-
// space error decision), ClusterLODResidencyFallback.comp (this engine's chosen parent-fallback
// residency policy -- a node whose fine geometry isn't resident yet has its nearest resident
// ancestor substituted instead of simply vanishing) and ClusterLODCompact.comp (final candidate-
// list emission) for the three shaders this class drives, in that order, every frame.
//
// Because geometry::ClusterDAG guarantees clusterError is strictly monotonically increasing from
// leaves to root (ValidateClusterDAG), the screen-error decision is a single-pass, embarrassingly
// parallel per-node test -- no tree traversal is needed for it. Only the residency dimension needs
// a (small, bounded) per-node ancestor walk, done once per DRAW-decided-but-non-resident node in
// ClusterLODResidencyFallback.comp.
//
// Exactly like every other piece of this Nanite-style pipeline, this class is a self-contained
// building block -- Init()/Shutdown()/per-frame Record*() only.
//
// --- Per-frame sequence a caller must record, in order ---
//   1. RecordClear(cmd) -- zeroes the candidate count, the force-draw flags, and the feedback
//      buffer's request count (see GetFeedbackBuffer()'s doc comment for who else touches it).
//   2. RecordEvaluateAndCompact(cmd, viewParams) -- the 3 dispatches described above, each
//      separated by the barrier making its output visible to the next; ends with the barrier
//      making GetCandidateMetadataBuffer()/GetCandidateCountBuffer() visible to a later compute
//      read (renderer::ClusterOcclusionCullingPass's early pass).
//   3. RecordBuildEarlyDispatchArgs(cmd) -- converts the candidate count into
//      GetEarlyDispatchArgsBuffer()'s VkDispatchIndirectCommand, which
//      renderer::ClusterOcclusionCullingPass's now-indirect RecordEarlyPass() consumes instead of
//      a CPU-known cluster count.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"
#include "renderer/ClusterCullingPass.h" // ClusterCullMetadata
#include "renderer/FeedbackBuffer.h"
#include "renderer/GpuBuffer.h"

namespace geometry { struct ClusterIndexEntry; struct DAGNodeEntry; }

namespace renderer {

    // GLSL-friendly, std430-compatible mirror of DAGNodePayload in ClusterDAGScreenError.comp.
    // Field order matches that struct exactly (sphereCenter first so every following scalar packs
    // tightly with zero implicit padding -- see that shader's own comment) -- 48 bytes, already a
    // multiple of its own 16-byte array stride, so no manual trailing pad is needed here either.
    struct DAGNodePayload {
        maths::vec3 sphereCenter;
        float sphereRadius = 0.0f;

        uint32_t clusterID = 0;
        uint32_t parentClusterID = 0xFFFFFFFFu;
        uint32_t childClusterID0 = 0xFFFFFFFFu;
        uint32_t childClusterID1 = 0xFFFFFFFFu;
        uint32_t level = 0;

        float clusterError = 0.0f;
        float parentError = 0.0f;
        float maxWPOAmplitude = 0.0f;
    };
    static_assert(sizeof(DAGNodePayload) == 48,
        "DAGNodePayload must match DAGNodePayload in ClusterDAGScreenError.comp exactly (std430 layout)");

    // GLSL-friendly, std430-compatible mirror of LODNodeMetadata in cluster_lod_node_metadata.glsl.
    // See that file's comment for the field-ordering rationale; the trailing 4-byte pad below
    // rounds this struct up to its actual 96-byte std430 array stride (GLSL infers this
    // automatically for its own copy, but the C++ mirror must declare it explicitly).
    struct LODNodeMetadata {
        maths::vec3 boundsMin; float _padBoundsMin = 0.0f;
        maths::vec3 boundsMax; float _padBoundsMax = 0.0f;
        maths::vec3 sphereCenter; float sphereRadius = 0.0f;
        maths::vec3 coneAxis; float coneCutoff = 0.0f;

        uint32_t indexCount = 0;
        uint32_t clusterID = 0;
        uint32_t parentClusterID = 0xFFFFFFFFu;
        uint32_t logicalPageID = 0;
        uint32_t maskTextureIndex = 0xFFFFFFFFu;
        float maxWPOAmplitude = 0.0f;
        // geometry::ClusterIndexEntry::entityID -- the owning entity's meshID, carried through the
        // LOD cut so ClusterLODCompact.comp can copy it verbatim into ClusterCullMetadata::entityID
        // (see that struct's own comment) for the resolve pass's NANITE_INSTANCES debug view.
        uint32_t entityID = 0;
        // geometry::ClusterIndexEntry::materialID, carried through the LOD cut the same way as
        // entityID above so ClusterLODCompact.comp can copy it verbatim into
        // ClusterCullMetadata::materialID for ClusterResolve.comp's real PBR material lookup.
        // Occupies what used to be this struct's trailing padding float -- same 96-byte std430
        // stride, no size change.
        uint32_t materialID = 0;
    };
    static_assert(sizeof(LODNodeMetadata) == 96,
        "LODNodeMetadata must match LODNodeMetadata in cluster_lod_node_metadata.glsl exactly (std430 layout)");

    class ClusterLODSelectionPass {
    public:
        ClusterLODSelectionPass() = default;

        ClusterLODSelectionPass(const ClusterLODSelectionPass&) = delete;
        ClusterLODSelectionPass& operator=(const ClusterLODSelectionPass&) = delete;

        // Builds DAGNodesSSBO/LODNodeMetadataSSBO from the FULL index/DAG tables (every DAG level,
        // not just leaves -- geometry::ClusterIndexEntry::virtualAddress feeds
        // geometry::GpuPageTable::LogicalAddressToPageID() for each entry's logicalPageID).
        // `leafCount` sizes the candidate output buffer (a valid cut's simultaneous draw count
        // never exceeds the total leaf count). `pageTableBuffer` is
        // renderer::GpuGeometryPagePool::GetPageTableBuffer().
        // `entityDataBuffer` (renderer::VulkanContext's own, same handle ClusterResolvePass::Init
        // receives) is bound read-only into ClusterLODCompact.comp's descriptor set (binding 6) so
        // it can exclude core::EntityFlags::IsTransparent entities from this candidate list --
        // see that shader's own comment.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkBuffer pageTableBuffer, uint32_t leafCount,
            const std::vector<geometry::ClusterIndexEntry>& indexEntries,
            const std::vector<geometry::DAGNodeEntry>& dagEntries,
            VkBuffer entityDataBuffer);

        void Shutdown();

        // Zeroes the candidate count, the force-draw flags, and the feedback buffer's request
        // count. Must be recorded once per frame, before RecordEvaluateAndCompact().
        void RecordClear(VkCommandBuffer cmd);

        // Records the 3-dispatch evaluate -> residency-fallback -> compact sequence described in
        // the class comment, uploading `viewParams` into the DAGViewParamsUBO first.
        struct ViewParams {
            maths::mat4 view;
            maths::mat4 proj;
            float pixelErrorThreshold = 1.0f;
            float fovYRadians = 0.0f;
            float viewportHeight = 0.0f;
            float aspectRatio = 0.0f;
        };
        void RecordEvaluateAndCompact(VkCommandBuffer cmd, const ViewParams& viewParams);

        // Converts the candidate count into GetEarlyDispatchArgsBuffer()'s VkDispatchIndirectCommand
        // (workgroup size 64, matching ClusterHZBOcclusionCull.comp's local_size_x). Must be
        // recorded after RecordEvaluateAndCompact(), before
        // renderer::ClusterOcclusionCullingPass::RecordEarlyPass().
        void RecordBuildEarlyDispatchArgs(VkCommandBuffer cmd);

        VkBuffer GetCandidateMetadataBuffer() const { return m_CandidateMetadataBuffer.Handle(); }
        VkBuffer GetCandidateCountBuffer() const { return m_CandidateCountBuffer.Handle(); }
        VkBuffer GetEarlyDispatchArgsBuffer() const { return m_EarlyDispatchArgsBuffer.Handle(); }
        uint32_t GetTotalNodeCount() const { return m_TotalNodeCount; }

        // The GPU-write side of the residency-miss feedback loop (ClusterLODResidencyFallback.comp
        // calls RequestClusterResidency() into this buffer's device-local half every frame, via
        // feedback_buffer.glsl). A future streaming coordinator reads it back
        // (FeedbackBuffer::RecordReadback()/ReadRequestedClusterIDs()) to drive real disk I/O --
        // this class only owns and clears it, it does not itself read it back.
        FeedbackBuffer& GetFeedbackBuffer() { return m_FeedbackBuffer; }

    private:
        static constexpr uint32_t kWorkgroupSize = 64; // Matches every LOD shader's local_size_x.

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        uint32_t m_TotalNodeCount = 0;
        uint32_t m_LeafCount = 0;

        FeedbackBuffer m_FeedbackBuffer;

        GpuBuffer m_DAGNodesBuffer;          // DAGNodePayload[totalNodeCount], std430, GPU_ONLY. Written once at Init.
        GpuBuffer m_LODNodeMetadataBuffer;   // LODNodeMetadata[totalNodeCount], std430, GPU_ONLY. Written once at Init.
        GpuBuffer m_DAGDecisionBuffer;       // uint[totalNodeCount], std430, GPU_ONLY. Rewritten every frame.
        GpuBuffer m_DAGLocalErrorBuffer;     // float[totalNodeCount] -- ClusterDAGScreenError.comp's diagnostic output, unread by any consumer.
        GpuBuffer m_DAGParentErrorBuffer;    // float[totalNodeCount] -- ditto.
        GpuBuffer m_ForceDrawBuffer;         // uint[totalNodeCount], std430, GPU_ONLY. Cleared every frame.
        GpuBuffer m_ViewParamsBuffer;        // DAGScreenErrorViewParams, std140 UBO, GPU_ONLY.
        GpuBuffer m_CandidateMetadataBuffer; // ClusterCullMetadata[leafCount], std430, GPU_ONLY.
        GpuBuffer m_CandidateCountBuffer;    // single uint32 atomic counter, GPU_ONLY.
        GpuBuffer m_EarlyDispatchArgsBuffer; // VkDispatchIndirectCommand (3x uint32), GPU_ONLY.

        // Dispatch 1: ClusterDAGScreenError.comp.
        VkDescriptorSetLayout m_ScreenErrorSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_ScreenErrorDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ScreenErrorPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ScreenErrorPipeline = VK_NULL_HANDLE;

        // Dispatch 2: ClusterLODResidencyFallback.comp.
        VkDescriptorSetLayout m_ResidencyFallbackSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_ResidencyFallbackDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ResidencyFallbackPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ResidencyFallbackPipeline = VK_NULL_HANDLE;

        // Dispatch 3: ClusterLODCompact.comp.
        VkDescriptorSetLayout m_CompactSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_CompactDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_CompactPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CompactPipeline = VK_NULL_HANDLE;

        // BuildDispatchIndirectArgs.comp instance (shared shader with
        // renderer::ClusterOcclusionCullingPass/ClusterSoftwareRasterPass, own separate
        // pipeline/descriptor instance).
        VkDescriptorSetLayout m_BuildArgsSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_BuildArgsDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_BuildArgsPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_BuildArgsPipeline = VK_NULL_HANDLE;

        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE; // Shared by all 4 descriptor sets above.
    };

}
