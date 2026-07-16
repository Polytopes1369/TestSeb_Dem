#version 460

// Hardware rasterization fragment shader for clusters PROVEN fully opaque at cluster-build time
// (geometry::ClusterDAGNode::isMasked == false -- see ClusterPartitioner.cpp's opacity split).
// Byte-for-byte identical VisBuffer write to ClusterRaster.frag (the masked variant), but with
// ZERO mask-sampling instructions and, critically, NO discard anywhere in this shader: unlike
// ClusterRaster.frag, whose `discard` statically disables hardware early-fragment-tests for every
// cluster drawn through that pipeline (opaque or not), a pipeline built from THIS shader has no
// `OpKill` in its SPIR-V at all, so early-Z is free to run. renderer::ClusterHardwareRasterPass
// builds this as a second, separate VkPipeline sharing the same layout/descriptor set as the
// masked pipeline (see that class's Init()) and routes only clusters with
// ClusterCullMetadata::maskTextureIndex == kInvalidMaskTextureIndex through it (see
// ClusterOcclusionCullingPass's opaque/masked list split).
//
// gl_PrimitiveID supplies the local TriangleID directly, exactly as in ClusterRaster.frag -- see
// that shader's own comment for why no manual bookkeeping is needed.

layout(location = 0) flat in uint inClusterID;

layout(location = 0) out uint outClusterID;
layout(location = 1) out uint outTriangleID;

void main() {
    outClusterID = inClusterID;
    outTriangleID = uint(gl_PrimitiveID);
}
