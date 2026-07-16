#pragma once
// Hardware rasterization pipeline for "large" clusters -- the Nanite-style dual rasterization
// path's fixed-function half: triangles covering more than a few pixels are efficiently rasterized
// by the ordinary hardware triangle pipeline, where per-pixel/per-quad fixed-function costs are
// well amortized. (A future compute-based software rasterizer would handle the *other* half --
// sub-pixel triangles, where hardware quad overdraw becomes wasteful -- and a size-classification
// step routing clusters to one path or the other; both are out of scope here. This class only
// builds the hardware path, and currently consumes whatever a culling pass's indirect buffers
// contain, unfiltered by triangle screen size.)
//
// Consumes, unmodified, whichever VkDrawIndexedIndirectCommand + draw-count buffer pair a culling
// pass already produced (renderer::ClusterCullingPass::GetIndirectCommandBuffer()/
// GetDrawCountBuffer(), or renderer::ClusterOcclusionCullingPass's early/late equivalents) via
// vkCmdDrawIndexedIndirectCount -- every render argument (which clusters, how many indices, at
// what buffer offsets) is generated entirely on the GPU by that upstream culling pass; this class
// never reads a draw count or cluster list back to the CPU, and RecordDraw() is a thin wrapper
// around exactly one vkCmdDrawIndexedIndirectCount call.
//
// See src/shaders/src/Renderer/ClusterRaster.vert / ClusterRaster.frag for the two shaders this
// pipeline runs:
//   - ClusterRaster.vert extracts one vertex's compressed position directly from the compressed
//     physical page pool (renderer::GpuGeometryPagePool::GetPhysicalPoolBuffer(), decoded on the
//     fly via cluster_vertex_decode.glsl -- no pre-decompression dependency for positions), using
//     gl_VertexIndex (the fixed-function-resolved absolute vertex slot) and gl_InstanceIndex (this
//     cluster's own slot in the same ClusterCullMetadataSSBO the culling pass read -- see
//     ClusterFrustumCull.comp / ClusterHZBOcclusionCull.comp, both of which now write that slot
//     index into every VkDrawIndexedIndirectCommand::firstInstance they emit).
//   - ClusterRaster.frag writes the Visibility Buffer: ClusterID (from the vertex shader) and
//     local TriangleID (gl_PrimitiveID, which the Vulkan spec resets to 0 at the start of each
//     indirect sub-draw -- i.e. each cluster's own draw command -- making it exactly the cluster-
//     local triangle ordinal with no manual bookkeeping needed).
//
// Indexed rendering requires a real, GPU-native index buffer -- unlike vertex positions, indices
// cannot be decoded on the fly by a shader (the fixed-function index-fetch stage runs before any
// shader). renderer::GeometryDecompressionPass now owns exactly that buffer
// (GetDecompressedIndexPoolBuffer(), VK_INDEX_TYPE_UINT32, expanded from the compressed pool ahead
// of time by DecompressClusterIndices.comp) -- RecordDraw() binds it every call, since which
// buffer is current can change as pages stream in/out.
//
// Exactly like every other piece of this Nanite-style pipeline (HZBPass, ClusterCullingPass,
// ClusterOcclusionCullingPass, GeometryDecompressionPass), this class is a self-contained building
// block -- Init()/Shutdown()/RecordDraw() only -- not wired into VulkanContext/main.cpp by this
// change. RecordDraw() must be called inside an already-open vkCmdBeginRendering scope targeting
// VisBuffer color attachments matching the formats passed to Init() (see
// VulkanContext::kVisBufferFormat) plus a depth attachment matching Init()'s depthFormat -- this
// class does not own or open that rendering scope itself, since it may need to share it with the
// flat/legacy VisBuffer draw (draw.vert/draw.frag) or a future software-rasterization path.

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include "core/Camera.h" // CameraPushConstants

namespace renderer {

    class ClusterHardwareRasterPass {
    public:
        ClusterHardwareRasterPass() = default;

        ClusterHardwareRasterPass(const ClusterHardwareRasterPass&) = delete;
        ClusterHardwareRasterPass& operator=(const ClusterHardwareRasterPass&) = delete;

        // Builds the descriptor set (binding 0 = ClusterCullMetadataSSBO, binding 1 =
        // CompressedClusterPoolSSBO, both vertex-stage-only) and the graphics pipeline (via
        // VulkanPipeline::CreateGraphicsPipeline, reusing the exact same MRT VisBuffer color-blend
        // / dynamic-rendering / depth-stencil setup the flat draw.vert/draw.frag pipeline already
        // uses). `clusterMetadataBuffer` is whichever culling pass's
        // GetClusterMetadataBuffer() the caller intends to feed RecordDraw()'s indirect buffers
        // from (renderer::ClusterCullingPass or renderer::ClusterOcclusionCullingPass) --
        // `firstInstance` in that pass's emitted VkDrawIndexedIndirectCommands indexes exactly this
        // buffer, so the two must come from the same culling pass instance/frame.
        // `compressedPhysicalPoolBuffer` is renderer::GpuGeometryPagePool::GetPhysicalPoolBuffer().
        // `visBufferColorFormats` must match VulkanContext::kVisBufferFormat x2 (ClusterID,
        // TriangleID) and `depthFormat` the depth image's format, so this pipeline is attachment-
        // compatible with the render targets RecordDraw() will be called against.
        void Init(VkDevice device, VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
            const std::array<VkFormat, 2>& visBufferColorFormats, VkFormat depthFormat);

        void Shutdown();

        // Records the hardware raster draw: binds the pipeline/descriptor set, pushes `camera`
        // (view/proj, vertex-stage only -- ClusterRaster.frag needs no push constants), binds
        // `decompressedIndexPoolBuffer` (renderer::GeometryDecompressionPass::
        // GetDecompressedIndexPoolBuffer(), VK_INDEX_TYPE_UINT32) as the active index buffer, sets
        // the dynamic viewport/scissor to `renderExtent`, then issues exactly one
        // vkCmdDrawIndexedIndirectCount over [indirectCommandBuffer, indirectCommandBuffer +
        // maxDrawCount * sizeof(VkDrawIndexedIndirectCommand)), reading the actual sub-draw count
        // from `drawCountBuffer` at GPU-execution time -- `maxDrawCount` only bounds the driver's
        // worst-case iteration, it is never read back to the CPU. Must be called inside an
        // already-open vkCmdBeginRendering scope -- see the class comment.
        void RecordDraw(VkCommandBuffer cmd, const CameraPushConstants& camera, VkExtent2D renderExtent,
            VkBuffer decompressedIndexPoolBuffer, VkBuffer indirectCommandBuffer, VkBuffer drawCountBuffer, uint32_t maxDrawCount);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
