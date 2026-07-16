#pragma once
// GPU-driven cluster visibility pass: culls the current frame's candidate cluster list against
// the camera's 6-plane frustum (bounding-box test) and a normal-cone backface test, compacting
// every surviving cluster into a VkDrawIndexedIndirectCommand buffer that a later indirect draw
// call can consume directly -- see src/shaders/src/Culling/ClusterFrustumCull.comp for the
// compute shader this class drives, and src/shaders/include/cluster_culling_common.glsl for the
// GLSL-side mirror of the two structs declared below.
//
// Upstream of this pass: whatever produces the frame's candidate cluster list (the LOD cut --
// ClusterDAGScreenError.comp -- and residency check -- ClusterResidencyCheck.comp, both still
// self-contained building blocks themselves) is expected to fill each candidate's
// ClusterCullMetadata from its geometry::ClusterIndexEntry (see ClusterFormat.h) and pass the list
// to UploadClusterMetadata(). Exactly like renderer::HZBPass / renderer::FeedbackBuffer /
// renderer::GeometryDecompressionPass, this class is a self-contained building block -- Init() /
// Shutdown() / per-frame Record*() only -- not wired into VulkanContext/main.cpp by this change.
//
// --- Per-frame sequence a caller must record, in order ---
//   1. UploadClusterMetadata(...)  -- whenever this frame's candidate list changes (may be
//      skipped on frames that reuse the previous upload, e.g. while the camera alone changed).
//   2. RecordClear(cmd)            -- resets the draw counter to 0, with its cross-dispatch barrier.
//   3. RecordCull(cmd, viewParams, clusterCount) -- one invocation per candidate cluster; each
//      survivor claims a unique slot in the indirect command buffer via a global atomic, exactly
//      as renderer::FeedbackBuffer's RequestClusterResidency() compacts residency misses.
//   4. A later vkCmdDrawIndexedIndirectCount (device must support VK_KHR_draw_indirect_count, or
//      an equivalent CPU readback of GetDrawCountBuffer() feeding a plain
//      vkCmdDrawIndexedIndirect with a CPU-known draw count) may then consume
//      GetIndirectCommandBuffer() / GetDrawCountBuffer() -- RecordCull()'s final barrier already
//      makes both visible to VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT /
//      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT.

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    // GLSL-friendly, std430-compatible mirror of geometry::ClusterIndexEntry (ClusterFormat.h) --
    // must match ClusterCullMetadata in cluster_culling_common.glsl field-for-field and
    // byte-offset-for-byte-offset (verified by the static_assert below). The packed, pragma-pack(1)
    // C++ ClusterIndexEntry cannot be uploaded directly (its 8-bit cone fields have no std430
    // representation); a caller converts each candidate cluster's ClusterIndexEntry into one of
    // these -- widening coneAxisX/Y/Z and coneCutoff from int8_t back to normalized floats
    // (divide by 127.0f, the exact inverse of geometry::VirtualGeometryCacheTest.cpp's
    // BuildIndexEntry quantization) -- before calling UploadClusterMetadata.
    struct ClusterCullMetadata {
        maths::vec3 boundsMin;
        float _padBoundsMin = 0.0f;
        maths::vec3 boundsMax;
        float _padBoundsMax = 0.0f;
        maths::vec3 sphereCenter;
        float sphereRadius = 0.0f;
        maths::vec3 coneAxis;
        float coneCutoff = 0.0f;
        uint32_t indexCount = 0;   // Becomes VkDrawIndexedIndirectCommand::indexCount for this cluster.
        uint32_t firstIndex = 0;   // Base offset into the global decompressed index buffer.
        uint32_t vertexOffset = 0; // Base vertex added to every local index.
        uint32_t clusterID = 0;
    };
    static_assert(sizeof(ClusterCullMetadata) == 80,
        "ClusterCullMetadata must match ClusterCullMetadata in cluster_culling_common.glsl exactly (std430 layout)");

    // CPU-side mirror of CullingViewParams in cluster_culling_common.glsl (the std140 UBO
    // payload): 6 frustum planes (xyz = outward unit normal, w = signed distance) extracted from
    // the combined view-projection matrix, plus the camera's world-space position for the
    // backface cone test.
    struct ClusterCullViewParams {
        std::array<std::array<float, 4>, 6> frustumPlanes{};
        maths::vec3 cameraPositionWorld;
        float _padCameraPosition = 0.0f;
    };
    static_assert(sizeof(ClusterCullViewParams) == 112,
        "ClusterCullViewParams must match CullingViewParams in cluster_culling_common.glsl exactly (std140 layout)");

    // Extracts the 6 frustum planes (Gribb-Hartmann) from `viewProj`, which must already be
    // proj * view (matching CameraPushConstants / draw.vert's `camera.proj * camera.view *
    // vec4(worldPos, 1.0)` transform order). Assumes Vulkan's [0, 1] normalized device depth range
    // (see maths::mat4::PerspectiveVulkan) -- NOT OpenGL's [-1, 1] convention, which extracts a
    // different (wrong) near plane. Every returned plane is normalized so plane[i][0..2] is a unit
    // outward normal and plane[i][3] is the true signed distance from the origin; a point p is
    // outside plane i when dot(plane[i].xyz, p) + plane[i].w < 0.
    std::array<std::array<float, 4>, 6> ExtractFrustumPlanes(const maths::mat4& viewProj);

    class ClusterCullingPass {
    public:
        ClusterCullingPass() = default;

        ClusterCullingPass(const ClusterCullingPass&) = delete;
        ClusterCullingPass& operator=(const ClusterCullingPass&) = delete;

        // Allocates the cluster metadata SSBO (maxClusters entries), the view-params UBO, the
        // output indirect-command SSBO (maxClusters VkDrawIndexedIndirectCommand-sized slots), the
        // single-word draw-count SSBO, and the culling compute pipeline/descriptor set.
        void Init(VkDevice device, VmaAllocator allocator, uint32_t maxClusters);

        void Shutdown();

        // Uploads this frame's candidate cluster list (clusters.size() must be <= maxClusters)
        // into the metadata SSBO via a host-visible staging buffer + one-time command buffer copy,
        // mirroring VulkanContext::UploadEntityData's staging pattern. Must be called, at least
        // once, before RecordCull() is first invoked for a matching candidate list; a caller need
        // not re-upload every frame if the candidate list is unchanged from the previous frame.
        void UploadClusterMetadata(VkCommandPool commandPool, VkQueue queue, const std::vector<ClusterCullMetadata>& clusters);

        // Resets the draw counter to 0 in the draw-count buffer and inserts the barrier making
        // that clear visible to RecordCull()'s compute dispatch. Must be recorded once per frame,
        // before RecordCull().
        void RecordClear(VkCommandBuffer cmd);

        // Records the culling dispatch: uploads `viewParams` into the view-params UBO via
        // vkCmdUpdateBuffer (112 bytes, well within its 65536-byte/4-byte-multiple limits) and
        // dispatches one invocation per candidate cluster in [0, clusterCount) -- clusterCount
        // must be <= the entry count of the most recent UploadClusterMetadata() call. Every
        // surviving cluster claims a unique slot in the indirect command buffer via a global
        // atomic counter. Ends with the barrier making both the indirect command buffer and the
        // draw-count buffer visible to VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT /
        // VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT for a later vkCmdDrawIndexedIndirect(Count).
        void RecordCull(VkCommandBuffer cmd, const ClusterCullViewParams& viewParams, uint32_t clusterCount);

        VkBuffer GetIndirectCommandBuffer() const { return m_IndirectCommandBuffer.Handle(); }
        VkBuffer GetDrawCountBuffer() const { return m_DrawCountBuffer.Handle(); }
        // Exposed so a downstream hardware raster pass (renderer::ClusterHardwareRasterPass) can
        // bind the exact same ClusterCullMetadataSSBO this pass populated -- surviving clusters'
        // VkDrawIndexedIndirectCommand::firstInstance (see ClusterFrustumCull.comp) is this
        // buffer's own array index for that cluster, letting a vertex shader re-index it directly
        // via gl_InstanceIndex.
        VkBuffer GetClusterMetadataBuffer() const { return m_ClusterMetadataBuffer.Handle(); }
        uint32_t GetMaxClusters() const { return m_MaxClusters; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only for UploadClusterMetadata()'s staging buffer.
        uint32_t m_MaxClusters = 0;

        GpuBuffer m_ClusterMetadataBuffer; // binding 0: ClusterCullMetadata[maxClusters], std430, GPU_ONLY.
        GpuBuffer m_ViewParamsBuffer;      // binding 1: CullingViewParams, std140 UBO, GPU_ONLY (written via vkCmdUpdateBuffer).
        GpuBuffer m_IndirectCommandBuffer; // binding 2: VkDrawIndexedIndirectCommand[maxClusters], std430, GPU_ONLY.
        GpuBuffer m_DrawCountBuffer;       // binding 3: single uint32 atomic counter, GPU_ONLY.

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
