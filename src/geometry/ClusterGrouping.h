#pragma once
// Cluster grouping (Nanite-style "cluster group" construction): groups spatially/topologically
// adjacent clusters produced by ClusterPartitioner.h (up to 4 at a time, see GroupItemsIntoQuads)
// and merges each group into one local mesh ready for DAG-parent simplification (see
// MeshSimplifier.h). Grouping 4 at a time instead of pairing 2 at a time mirrors Nanite's actual
// cluster-graph partitioning granularity: a bigger group frees proportionally more of its members'
// shared boundary (turned interior by the merge) relative to the group's true external boundary
// that must stay locked, so QEM has more room to simplify before hitting the per-cluster
// vertex/triangle cap -- see ClusterDAG.cpp's EmitSimplifiedGroup for what happens on that cap.
//
// Two clusters coming out of geometry::PartitionMeshIntoClusters are considered adjacent if they
// share at least one *global* vertex index of the original mesh -- since the partitioner never
// duplicates vertices, a shared global index means the two clusters are stitched together along
// an exact edge/vertex chain of the source mesh, not just spatially close.
//
// Merging a group lets every boundary that used to sit *between* two of its members become
// interior (used by two triangles instead of one) in the merged mesh, so it can be freely
// simplified away. Only the resulting group's true outer boundary -- edges still used by exactly
// one triangle after the merge, because they border a cluster that was NOT included in this group
// -- is marked locked. This is the crack-prevention contract MeshSimplifier.h enforces: a locked
// vertex never moves and is never removed, so neighboring, differently-grouped/simplified geometry
// can still stitch against it exactly.

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

    // Hierarchically groups `count` items into groups of up to 4, by running GreedyPairByWeight
    // twice: pass 1 pairs individual items (exactly like GreedyPairByWeight alone); pass 2 pairs
    // those pairs, using as each pair-to-pair weight the sum of every original `weights` entry
    // connecting a member of one pair to a member of the other (so the second pass still prefers
    // the most naturally connected quads, not an arbitrary combination). This is a recursive
    // bisection of the same adjacency graph GreedyPairByWeight already walks -- the practical
    // stand-in this engine uses for the graph-partitioning step real Nanite runs (e.g. METIS) to
    // reach the same ~4-cluster group granularity, without pulling in a heavyweight graph-
    // partitioning library the project's "no heavy frameworks" constraint rules out.
    //
    // A cross-classification edge (opaque vs. masked) never appears in `weights` to begin with
    // (see BuildClusterAdjacencyWeights/BuildLevelAdjacencyWeights), so no synthesized pair-to-pair
    // weight can connect an opaque-only pair to a masked-only pair either -- the purity invariant
    // survives both passes for free. Returns one entry per final group (size 1 to 4), each holding
    // the flattened original item indices.
    std::vector<std::vector<uint32_t>> GroupItemsIntoQuads(
        uint32_t count, const std::unordered_map<uint64_t, uint32_t>& weights);

    // One cluster group: up to 4 source clusters (fewer at the edges of the adjacency graph, e.g.
    // a topologically isolated island) merged into a single boundary-locked local mesh.
    struct ClusterGroup {
        // Indices into the `clusters` vector passed to GroupAdjacentClusters (size 1 to 4).
        std::vector<uint32_t> memberClusterIndices;

        // Merged, deduplicated local mesh: mesh.locked[v] is true for every vertex lying on an
        // edge used by exactly one triangle within this group (i.e. the group's outer boundary).
        // Ready to pass directly to SimplifyMeshQEM.
        SimplifiableMesh mesh;

        // Sum of the member clusters' triangle counts, i.e. mesh.triangles.size() / 3 before any
        // simplification is applied.
        uint32_t originalTriangleCount = 0;
    };

    // Groups adjacent clusters into groups of up to 4 (see file header for the adjacency
    // definition and GroupItemsIntoQuads for the grouping strategy) and builds one merged,
    // boundary-locked ClusterGroup per group.
    //
    // `allVertices` must be the same array the clusters' MeshCluster::globalVertexIndices index
    // into (i.e. whatever was passed to PartitionMeshIntoClusters to produce `clusters`).
    std::vector<ClusterGroup> GroupAdjacentClusters(
        const std::vector<MeshCluster>& clusters,
        const std::vector<renderer::Vertex>& allVertices);

}
