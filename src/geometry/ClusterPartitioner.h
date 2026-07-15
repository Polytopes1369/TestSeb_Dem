#pragma once
// Native, dependency-free spatial partitioning of a triangle mesh into small, GPU-meshlet-sized
// clusters (geometry::kMaxClusterVertices / geometry::kMaxClusterTriangles from ClusterFormat.h).
//
// Two families of algorithms are commonly used for this problem:
//   - Graph partitioning (METIS-style): build the triangle adjacency (dual) graph and run a
//     multilevel min-edge-cut partition. Produces excellent locality but requires a full
//     coarsening/uncoarsening pipeline and is usually pulled in as a heavy external library.
//   - Spatial clustering (k-means-style): partition triangles directly by centroid position.
//     No adjacency graph is needed, it converges in a handful of iterations, and the resulting
//     clusters are geometrically compact, which is exactly what bounding-volume culling wants.
//
// This file implements the second family: a *bisecting k-means* partitioner (recursive 2-means).
// It is the standard way to run k-means when the number of clusters isn't known in advance and
// a hard per-cluster size cap must be enforced: split the mesh's triangles into two spatially
// coherent halves via a few Lloyd iterations, then recurse into each half until it satisfies both
// the vertex and triangle caps. Because every recursive split strictly shrinks the group (see the
// termination guarantee on SplitGroupInTwo in the .cpp), and because a single triangle always
// trivially satisfies both caps, the recursion is guaranteed to terminate and to produce clusters
// that never exceed the limits — a property plain (non-bisecting) k-means cannot offer on its own.
// It is dependency-free, native C++23, and does not require building an explicit adjacency graph,
// which makes it both simpler to implement correctly and cheaper to run than a METIS-style
// multilevel partition for the procedurally generated primitives this engine produces.

#include <cstdint>
#include <vector>
#include "core/maths/Maths.h"
#include "renderer/RenderTypes.h"

namespace geometry {

    // One partitioned cluster: a self-contained triangle group with its own local vertex
    // renumbering (so it can be encoded into geometry::ClusterData) and bounding volumes.
    struct MeshCluster {
        // Global (into the caller's full vertex array) index for each vertex used by this
        // cluster, in first-seen order. Local triangle indices below index into this array.
        // size() is always <= geometry::kMaxClusterVertices.
        std::vector<uint32_t> globalVertexIndices;

        // Cluster-local triangle list: 3 consecutive entries per triangle, each indexing into
        // globalVertexIndices (i.e. in [0, globalVertexIndices.size())). size() / 3 is always
        // <= geometry::kMaxClusterTriangles.
        std::vector<uint8_t> localTriangleIndices;

        // Traceability: for each triangle in localTriangleIndices (same order, one entry per
        // triangle), the 0-based index of that triangle among the *original* triangles that
        // matched the requested meshID (i.e. its position in a left-to-right scan of the input
        // index buffer, counting only matching triangles). Lets callers (tests, LOD builders)
        // verify every original triangle ended up in exactly one output cluster.
        std::vector<uint32_t> originalTriangleIndices;

        // Axis-aligned bounding box, in the same space as the input vertex positions.
        maths::vec3 boundsMin;
        maths::vec3 boundsMax;

        // Bounding sphere guaranteed to enclose every vertex in the cluster (center of the AABB,
        // radius equal to its half-diagonal) — cheap to test and consistent with the enclosing
        // sphere convention already used elsewhere in the virtual geometry cache.
        maths::vec3 sphereCenter;
        float sphereRadius = 0.0f;
    };

    // Partitions every triangle of `allIndices` whose vertices carry Vertex::meshID == targetMeshID
    // (the per-vertex ID the procedural PrimitiveGen compute shaders already stamp) into spatially
    // compact clusters, each respecting geometry::kMaxClusterVertices / geometry::kMaxClusterTriangles.
    //
    // Returns an empty vector if no triangle matches targetMeshID. The union of every returned
    // cluster's originalTriangleIndices is exactly the set [0, N) with no gaps and no duplicates,
    // where N is the number of matching triangles.
    std::vector<MeshCluster> PartitionMeshIntoClusters(
        uint32_t targetMeshID,
        const std::vector<renderer::Vertex>& allVertices,
        const std::vector<uint32_t>& allIndices);

}
