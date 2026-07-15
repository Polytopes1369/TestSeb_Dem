#pragma once
// Cluster grouping (Nanite-style "cluster group" construction): pairs up spatially/topologically
// adjacent clusters produced by ClusterPartitioner.h and merges each pair into one local mesh
// ready for DAG-parent simplification (see MeshSimplifier.h).
//
// Two clusters coming out of geometry::PartitionMeshIntoClusters are considered adjacent if they
// share at least one *global* vertex index of the original mesh -- since the partitioner never
// duplicates vertices, a shared global index means the two clusters are stitched together along
// an exact edge/vertex chain of the source mesh, not just spatially close.
//
// Merging a pair lets the boundary that used to sit *between* them become interior (used by two
// triangles instead of one) in the merged mesh, so it can be freely simplified away. Only the
// resulting group's true outer boundary -- edges still used by exactly one triangle after the
// merge, because they border a cluster that was NOT included in this group -- is marked locked.
// This is the crack-prevention contract MeshSimplifier.h enforces: a locked vertex never moves
// and is never removed, so neighboring, differently-grouped/simplified geometry can still stitch
// against it exactly.

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "geometry/ClusterPartitioner.h"
#include "geometry/MeshSimplifier.h"
#include "renderer/RenderTypes.h"

namespace geometry {

    // Packs an unordered pair of indices into one 64-bit key (high 32 bits = the larger index,
    // low 32 bits = the smaller), used as the key for the adjacency-weight maps consumed by
    // GreedyPairByWeight. Exposed so callers building their own adjacency notion (e.g. ClusterDAG
    // pairing same-level DAG nodes by shared boundary position instead of shared cluster vertex)
    // can still reuse the pairing algorithm itself.
    uint64_t PackIndexPairKey(uint32_t a, uint32_t b);

    // Greedily pairs up `count` items (indices [0, count)) given their pairwise adjacency
    // weights (only pairs present as keys in `weights`, built via PackIndexPairKey, are
    // considered adjacent at all). Each item is paired with its still-unpaired neighbor of
    // highest weight (ties broken by smaller neighbor index); an item with no unpaired neighbor
    // left forms a singleton group of its own. Returns one entry per output group, each holding
    // its 1 or 2 member indices.
    std::vector<std::vector<uint32_t>> GreedyPairByWeight(
        uint32_t count, const std::unordered_map<uint64_t, uint32_t>& weights);

    // One cluster group: a pair (or, for a leftover odd cluster, a singleton) of source clusters
    // merged into a single boundary-locked local mesh.
    struct ClusterGroup {
        // Indices into the `clusters` vector passed to GroupAdjacentClusters (size 1 or 2).
        std::vector<uint32_t> memberClusterIndices;

        // Merged, deduplicated local mesh: mesh.locked[v] is true for every vertex lying on an
        // edge used by exactly one triangle within this group (i.e. the group's outer boundary).
        // Ready to pass directly to SimplifyMeshQEM.
        SimplifiableMesh mesh;

        // Sum of the member clusters' triangle counts, i.e. mesh.triangles.size() / 3 before any
        // simplification is applied.
        uint32_t originalTriangleCount = 0;
    };

    // Greedily pairs up adjacent clusters (see file header for the adjacency definition) and
    // builds one merged, boundary-locked ClusterGroup per pair. Pairing prefers, among an
    // unpaired cluster's still-unpaired neighbors, the one sharing the most global vertices (the
    // most natural/compact merge). A cluster with no unpaired adjacent neighbor forms a singleton
    // group on its own.
    //
    // `allVertices` must be the same array the clusters' MeshCluster::globalVertexIndices index
    // into (i.e. whatever was passed to PartitionMeshIntoClusters to produce `clusters`).
    std::vector<ClusterGroup> GroupAdjacentClusters(
        const std::vector<MeshCluster>& clusters,
        const std::vector<renderer::Vertex>& allVertices);

}
