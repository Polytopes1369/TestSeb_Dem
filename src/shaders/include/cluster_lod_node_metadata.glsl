#ifndef CLUSTER_LOD_NODE_METADATA_GLSL
#define CLUSTER_LOD_NODE_METADATA_GLSL

// One entry per DAG node, ALL levels (not just leaves) -- index-aligned with DAGNodePayload
// (ClusterDAGScreenError.comp) and with the DAG's own dense, 0-based, per-file-unique clusterID
// numbering (geometry::ClusterIndexEntry::clusterID / DAGNodeEntry::clusterID). Populated once at
// renderer::ClusterLODSelectionPass::Init from the full geometry::ClusterIndexEntry table (every
// DAG level is already resident by the time this runs -- see ClusterRenderPipeline::Init's STEP 4),
// and consumed every frame by ClusterLODResidencyFallback.comp (the parent-fallback ancestor walk)
// and ClusterLODCompact.comp (final candidate-list emission).
//
// Field order mirrors ClusterCullMetadata's own convention (cluster_culling_common.glsl): every
// vec3 is immediately followed by a scalar that fills its std430 base-alignment padding, so each
// subsequent vec3 lands on a clean 16-byte boundary with no implicit gaps. The trailing scalar run
// (indexCount..maxWPOAmplitude) needs no per-field padding of its own; only the struct's own total
// size needs rounding up to its 16-byte array stride (96 bytes total), which the GLSL compiler
// infers automatically -- the C++ mirror (renderer::LODNodeMetadata, ClusterLODSelectionPass.h)
// declares that rounding explicitly (see its own comment for why).
struct LODNodeMetadata {
    vec3 boundsMin;
    float _padBoundsMin;
    vec3 boundsMax;
    float _padBoundsMax;
    vec3 sphereCenter;
    float sphereRadius;
    vec3 coneAxis;
    float coneCutoff;

    uint indexCount;
    uint clusterID;       // == this entry's own array index; kept explicit for readability.
    uint parentClusterID; // geometry::kInvalidClusterID (0xFFFFFFFF) for a DAG root.
    uint logicalPageID;   // virtualAddress / GEOMETRY_PAGE_SIZE_BYTES -- see geometry_page_table.glsl.
    uint maskTextureIndex;
    float maxWPOAmplitude;
};

#endif // CLUSTER_LOD_NODE_METADATA_GLSL
