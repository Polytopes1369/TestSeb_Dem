#version 460

// Hardware rasterization fragment shader for "large" clusters (see
// renderer::ClusterHardwareRasterPass / ClusterRaster.vert). Writes the Visibility Buffer only --
// no lighting, no albedo, no standard color output -- exactly like draw.frag's own VisBuffer
// write, reusing the same 2x R32_UINT attachment convention (ClusterID + local TriangleID) so both
// the flat legacy draw path and this clustered hardware path can share one VisBuffer target.
//
// gl_PrimitiveID supplies the local TriangleID directly: the Vulkan spec resets primitive
// numbering to 0 at the start of each indirect sub-draw (i.e. each cluster's own
// VkDrawIndexedIndirectCommand within the vkCmdDrawIndexedIndirectCount batch), so it is exactly
// this cluster's own triangle ordinal in [0, indexCount / 3) with no manual computation needed in
// either shader stage.

layout(location = 0) flat in uint inClusterID;

layout(location = 0) out uint outClusterID;
layout(location = 1) out uint outTriangleID;

void main() {
    outClusterID = inClusterID;
    outTriangleID = uint(gl_PrimitiveID);
}
