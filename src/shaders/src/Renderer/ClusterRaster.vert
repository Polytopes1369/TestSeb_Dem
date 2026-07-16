#version 460
#extension GL_GOOGLE_include_directive : enable

// Hardware rasterization vertex shader for "large" clusters (Nanite-style dual rasterization path
// -- see renderer::ClusterHardwareRasterPass). Dispatched via vkCmdDrawIndexedIndirectCount, one
// VkDrawIndexedIndirectCommand per surviving cluster (produced entirely on the GPU by
// ClusterFrustumCull.comp / ClusterHZBOcclusionCull.comp): every render argument (which cluster,
// how many indices, at what buffer offsets) already comes from that indirect command, so this
// shader is only responsible for turning (gl_VertexIndex, gl_InstanceIndex) into a world-space
// clip-space position, decoded straight from the *compressed* physical page pool.
//
// gl_InstanceIndex identifies the cluster: every indirect command's instanceCount is 1 and
// firstInstance carries that cluster's own index into g_Clusters.clusters[] (see
// ClusterFrustumCull.comp / ClusterHZBOcclusionCull.comp's EmitEarlyDraw/EmitLateDraw), so
// gl_InstanceIndex == firstInstance exactly and re-indexes the very same
// ClusterCullMetadataSSBO the culling pass already read.
//
// gl_VertexIndex is the fixed-function-resolved absolute vertex slot: the index buffer bound via
// vkCmdBindIndexBuffer (renderer::GeometryDecompressionPass::GetDecompressedIndexPoolBuffer(),
// VK_INDEX_TYPE_UINT32, expanded ahead of time by DecompressClusterIndices.comp from
// ClusterData::indices) stores each cluster's *local* index values (0, kMaxClusterVertices)); the
// fixed-function stage adds VkDrawIndexedIndirectCommand::vertexOffset (== this cluster's
// ClusterCullMetadata::vertexOffset, physicalPageIndex * CLUSTER_MAX_VERTICES) before this shader
// ever runs. Subtracting that same vertexOffset back off here recovers the local vertex index
// needed to address the compressed page's SoA vertex block.

#include "include/cluster_culling_common.glsl"

#define COMPRESSED_POOL_SET 0
#define COMPRESSED_POOL_BINDING 1
#include "include/cluster_vertex_decode.glsl"

layout(std430, set = 0, binding = 0) readonly buffer ClusterCullMetadataSSBO {
    ClusterCullMetadata clusters[];
} g_Clusters;

// Camera matrices via Push Constants -- same layout as draw.vert's CameraPushConstants.
layout(push_constant) uniform CameraPushConstants {
    mat4 view;
    mat4 proj;
} camera;

// Visibility Buffer ClusterID, passed through to ClusterRaster.frag. This is gl_InstanceIndex
// (this cluster's own slot in ClusterCullMetadataSSBO), NOT geometry::ClusterIndexEntry::clusterID
// (the unrelated, persistent, potentially-sparse on-disk cache identifier) -- a later resolve pass
// (renderer::ClusterResolvePass) needs to re-index g_Clusters.clusters[] directly from whatever is
// stored in the VisBuffer to reconstruct per-pixel vertex data, which only a dense array slot
// index supports; the true clusterID remains reachable, if ever needed, via one extra indirection
// (g_Clusters.clusters[slotIndex].clusterID). Local TriangleID is NOT computed here --
// ClusterRaster.frag reads it directly from gl_PrimitiveID, which the fixed-function primitive
// assembly stage resets to 0 at the start of each indirect sub-draw (i.e. each cluster's own
// VkDrawIndexedIndirectCommand), making it exactly this cluster's local triangle ordinal with no
// manual bookkeeping needed here.
layout(location = 0) flat out uint outClusterID;

void main() {
    ClusterCullMetadata cluster = g_Clusters.clusters[gl_InstanceIndex];

    // physicalPageIndex is recovered from vertexOffset (== physicalPageIndex * CLUSTER_MAX_VERTICES
    // by construction, see renderer::GeometryDecompressionPass / ClusterCullMetadata::vertexOffset's
    // documented contract) rather than needing a second metadata field -- exact integer division
    // since vertexOffset is always page-aligned.
    uint physicalPageIndex = cluster.vertexOffset / CLUSTER_MAX_VERTICES;
    uint localVertexIndex = uint(gl_VertexIndex) - cluster.vertexOffset;
    uint pageByteBase = physicalPageIndex * CLUSTER_PAGE_SIZE_BYTES;

    // Decoded directly from the compressed pool -- see DecodeClusterPosition
    // (cluster_vertex_decode.glsl) -- no dependency on renderer::GeometryDecompressionPass's
    // vertex pool having already decompressed this exact page this frame.
    vec3 worldPos = DecodeClusterPosition(pageByteBase, localVertexIndex, cluster.boundsMin, cluster.boundsMax);

    outClusterID = uint(gl_InstanceIndex);

    gl_Position = camera.proj * camera.view * vec4(worldPos, 1.0);
}
