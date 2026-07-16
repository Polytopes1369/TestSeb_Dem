#pragma once
// Hardware rasterization pipeline for "large" clusters -- the Nanite-style dual rasterization
// path's fixed-function half: triangles covering more than kSoftwareRasterThresholdPixels screen
// pixels (renderer::ClusterRenderPipeline's own constant) are efficiently rasterized by the
// ordinary hardware triangle pipeline, where per-pixel/per-quad fixed-function costs are well
// amortized. renderer::ClusterSoftwareRasterPass is this path's live sibling, handling the *other*
// half -- sub-pixel triangles, where hardware quad overdraw becomes wasteful -- via a 64-bit atomic
// software rasterizer; the size-classification step routing each cluster to one path or the other
// runs entirely on the GPU inside ClusterHZBOcclusionCull.comp's ShouldUseSoftwareRaster(), driven
// by renderer::ClusterOcclusionCullingPass. This class only builds the hardware path, consuming
// whichever subset of clusters that classification already routed here.
//
// Consumes, unmodified, whichever VkDrawIndexedIndirectCommand + draw-count buffer pair
// renderer::ClusterOcclusionCullingPass's early/late passes already produced via
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
// Exactly like every other piece of this Nanite-style pipeline (HZBPass, ClusterOcclusionCullingPass,
// ClusterSoftwareRasterPass, GeometryDecompressionPass), this class is a self-contained building
// block -- Init()/Shutdown()/RecordDraw() only; renderer::ClusterRenderPipeline is what actually
// wires it into the frame loop main.cpp drives. RecordDraw() must be called inside an already-open
// vkCmdBeginRendering scope targeting VisBuffer color attachments matching the formats passed to
// Init() (see VulkanContext::kVisBufferFormat) plus a depth attachment matching Init()'s
// depthFormat -- this class does not own or open that rendering scope itself, since
// ClusterRenderPipeline::BeginVisBufferRendering shares it across both the early and late hardware
// raster passes (see that method's own comment on the CLEAR-vs-LOAD distinction between the two).

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
        // CompressedClusterPoolSSBO, binding 2 = WPOGlobalsUBO, all vertex-stage-only) and TWO
        // graphics pipelines sharing that one descriptor set/layout (via
        // VulkanPipeline::CreateGraphicsPipeline, reusing the exact same MRT VisBuffer color-blend /
        // dynamic-rendering / depth-stencil setup the flat draw.vert/draw.frag pipeline already
        // uses): a masked pipeline (ClusterRaster.vert/ClusterRaster.frag, samples the bindless
        // cutout mask array and discards below the alpha-test cutoff -- this `discard` statically
        // disables hardware early-fragment-tests for the whole pipeline) and an opaque pipeline
        // (ClusterRaster.vert/ClusterRasterOpaque.frag, no mask sampling, no discard at all, so
        // early-Z is free to run) -- see ClusterRasterOpaque.frag's own doc comment. The opaque
        // frag module simply never references binding 3 (the mask array), so no second descriptor
        // set is needed. `clusterMetadataBuffer` is whichever culling pass's
        // GetClusterMetadataBuffer() the caller intends to feed RecordDraw()'s indirect buffers
        // from -- renderer::ClusterOcclusionCullingPass in this pipeline today --
        // `firstInstance` in that pass's emitted VkDrawIndexedIndirectCommands indexes exactly this
        // buffer, so the two must come from the same culling pass instance/frame.
        // `compressedPhysicalPoolBuffer` is renderer::GpuGeometryPagePool::GetPhysicalPoolBuffer().
        // `wpoGlobalsBuffer` is renderer::ClusterRenderPipeline's own WPOGlobalsUBO buffer (16
        // bytes, std140) -- see wpo_deformation.glsl's ApplyWPODeformation, called from
        // ClusterRaster.vert. `maskImageInfos` is renderer::ProceduralMaskGenerator::
        // GetMaskImageInfos() -- bound as binding 3 (fragment-stage-only, COMBINED_IMAGE_SAMPLER,
        // descriptorCount == maskImageInfos.size()) for ClusterRaster.frag's opacity-mask discard
        // (mask_sampling.glsl). `visBufferColorFormats` must match VulkanContext::kVisBufferFormat x2
        // (ClusterID, TriangleID) and `depthFormat` the depth image's format, so this pipeline is
        // attachment-compatible with the render targets RecordDraw() will be called against.
        void Init(VkDevice device, VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
            VkBuffer wpoGlobalsBuffer, const std::vector<VkDescriptorImageInfo>& maskImageInfos,
            const std::array<VkFormat, 2>& visBufferColorFormats, VkFormat depthFormat,
            VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer);

        void Shutdown();

        // Records one hardware raster draw: binds either the opaque or masked pipeline (per
        // `opaque`) plus the shared descriptor set, pushes `camera` (view/proj, vertex-stage only --
        // neither fragment shader needs push constants), binds `decompressedIndexPoolBuffer`
        // (renderer::GeometryDecompressionPass::GetDecompressedIndexPoolBuffer(),
        // VK_INDEX_TYPE_UINT32) as the active index buffer, sets the dynamic viewport/scissor to
        // `renderExtent`, then issues exactly one vkCmdDrawIndexedIndirectCount over
        // [indirectCommandBuffer, indirectCommandBuffer + maxDrawCount *
        // sizeof(VkDrawIndexedIndirectCommand)), reading the actual sub-draw count from
        // `drawCountBuffer` at GPU-execution time -- `maxDrawCount` only bounds the driver's
        // worst-case iteration, it is never read back to the CPU. `indirectCommandBuffer`/
        // `drawCountBuffer` must already be filtered to only the opaque (or only the masked)
        // cluster list matching `opaque` -- see renderer::ClusterOcclusionCullingPass's opaque/
        // masked list split; this call performs no filtering of its own. Must be called inside an
        // already-open vkCmdBeginRendering scope -- see the class comment. Callers wanting both
        // variants drawn this frame call this twice, once per `opaque` value, against each list's
        // own indirect buffers.
        void RecordDraw(VkCommandBuffer cmd, const CameraPushConstants& camera, VkExtent2D renderExtent,
            VkBuffer decompressedIndexPoolBuffer, VkBuffer indirectCommandBuffer, VkBuffer drawCountBuffer,
            uint32_t maxDrawCount, bool opaque);

    private:
        VkDevice m_Device = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_OpaquePipeline = VK_NULL_HANDLE; // ClusterRaster.vert + ClusterRasterOpaque.frag: no discard, early-Z eligible.
        VkPipeline m_MaskedPipeline = VK_NULL_HANDLE; // ClusterRaster.vert + ClusterRaster.frag: mask-sampling discard.
    };

}
