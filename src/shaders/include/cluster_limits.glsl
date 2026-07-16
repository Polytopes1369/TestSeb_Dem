#ifndef CLUSTER_LIMITS_GLSL
#define CLUSTER_LIMITS_GLSL

// Matches geometry::kMaxClusterVertices / kMaxClusterTriangles / kMaxClusterIndices
// (ClusterFormat.h) exactly -- must be kept in sync with that C++-side header by hand, since
// GLSL has no way to #include a C++ constant. Split out from cluster_vertex_decode.glsl into its
// own binding-free header so shaders that only need these limits (not the compressed-cluster-pool
// SSBO decode functions) can include them without pulling in an unwanted SSBO binding.
#define CLUSTER_MAX_VERTICES 64u
#define CLUSTER_MAX_TRIANGLES 128u
#define CLUSTER_MAX_INDICES (CLUSTER_MAX_TRIANGLES * 3u)

#endif // CLUSTER_LIMITS_GLSL
