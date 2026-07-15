#pragma once
// Nanite-style cluster DAG: connects high-resolution leaf clusters (ClusterPartitioner.h) to
// progressively coarser, simplified parent clusters (ClusterGrouping.h + MeshSimplifier.h), all
// the way up to one or more roots, so a renderer can pick a view-dependent LOD cut through the
// hierarchy instead of a single fixed mesh resolution.
//
// Level 0 is the set of leaf clusters produced by PartitionMeshIntoClusters -- exact geometry,
// e_local (clusterError) == 0. Every level above that is built by:
//   1. pairing adjacent same-level nodes (ClusterGrouping::GroupAdjacentClusters for level 0->1,
//      or this file's own position-based pairing for level 1->2, 2->3, ... since nodes above
//      level 0 no longer carry indices into the original mesh -- only their locked boundary
//      vertices, whose positions are preserved bit-exact from the original mesh, survive as a
//      reliable adjacency signal);
//   2. simplifying each merged pair with SimplifyMeshQEM, honoring the group's boundary lock so
//      the pair's outer boundary (shared with un-grouped neighbors) never moves;
//   3. recording that new, coarser node's error.
//
// Error bookkeeping (e_local / e_parent): each node stores clusterError (e_local), the geometric
// error introduced by using this node's mesh instead of its children's exact combined geometry,
// and parentError (e_parent), its own parent's clusterError (+infinity for a root). Monotonicity
// is enforced constructively, not just checked after the fact: a parent's clusterError is always
// set to strictly more than the largest clusterError among its children (see kMinimumErrorStep
// below), so e_parent > e_local holds for every non-root node by construction. A runtime LOD cut
// can then walk the DAG top-down and, for a node whose parentError is already below the view's
// error threshold, stop there; otherwise it must recurse into that node's children.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "core/maths/Maths.h"
#include "geometry/MeshSimplifier.h"
#include "renderer/RenderTypes.h"

namespace geometry {

    constexpr uint32_t kInvalidDAGNodeIndex = 0xFFFFFFFFu;

    // Every parent's clusterError is bumped above the largest of its children's clusterError by
    // at least max(kMinimumErrorStepAbsolute, childrenMaxError * kMinimumErrorStepRelative),
    // even when the QEM simplification pass itself introduced (numerically) zero measurable
    // error -- e.g. simplifying an already-flat patch. This guarantees strictly increasing error
    // everywhere in the DAG, which ValidateClusterDAG treats as a hard invariant (see its doc
    // comment). A purely absolute floor is not enough on its own: a float's precision (ULP) grows
    // with its magnitude, so adding a fixed 1e-6 to an error value already around 50-100 (a
    // realistic magnitude a few levels up a real mesh's DAG) silently rounds back down to the
    // exact same float, defeating the guarantee it was meant to provide -- hence the relative term.
    constexpr float kMinimumErrorStepAbsolute = 1e-4f;
    constexpr float kMinimumErrorStepRelative = 1e-3f; // 0.1% of the largest child error.

    // One node of a cluster DAG.
    struct ClusterDAGNode {
        SimplifiableMesh mesh;

        std::vector<uint32_t> childIndices;          // Indices into ClusterDAG::nodes; empty for leaves.
        uint32_t parentIndex = kInvalidDAGNodeIndex;  // Index into ClusterDAG::nodes; kInvalidDAGNodeIndex for a root.

        maths::vec3 boundsMin{};
        maths::vec3 boundsMax{};
        maths::vec3 sphereCenter{};
        float sphereRadius = 0.0f;

        // e_local: geometric error introduced by using this node's mesh in place of its
        // children's exact, pre-simplification geometry. Always 0 for a leaf (level 0): a leaf
        // *is* the exact original geometry, no simplification has been applied to it yet.
        float clusterError = 0.0f;

        // e_parent: this node's own parent's clusterError. +infinity for a root, matching
        // ClusterFormat.h's existing DAGNodeEntry::parentError convention (there is no coarser
        // representation to fall back past a root).
        float parentError = std::numeric_limits<float>::infinity();

        uint32_t level = 0; // 0 = leaf; +1 for every grouping/simplification pass above that.
    };

    struct ClusterDAG {
        std::vector<ClusterDAGNode> nodes;
        std::vector<uint32_t> rootIndices; // Nodes with parentIndex == kInvalidDAGNodeIndex.
    };

    // Builds the full multi-level DAG for every triangle of `allIndices` whose vertices carry
    // Vertex::meshID == targetMeshID. Returns a DAG with an empty `nodes`/`rootIndices` if no
    // triangle matches targetMeshID.
    //
    // Grouping+simplification repeats, level by level, until a single root remains or no further
    // pairing is possible within a level (e.g. topologically disconnected mesh islands, which can
    // never become adjacent no matter how many passes run) -- in the latter case every
    // still-unpaired top node of its island becomes its own root, so `rootIndices` may contain
    // more than one entry.
    ClusterDAG BuildClusterDAG(
        uint32_t targetMeshID,
        const std::vector<renderer::Vertex>& allVertices,
        const std::vector<uint32_t>& allIndices);

    // Validates a whole DAG's structural and error-monotonicity invariants. Intended to run once
    // at startup, right after BuildClusterDAG, before the DAG is trusted for runtime LOD cuts.
    // Walks every node reachable from `dag.rootIndices` down through childIndices (equivalently:
    // every node's ancestor chain up through parentIndex) and checks:
    //   1. Every childIndices/parentIndex entry is in range.
    //   2. The child/parent relation is mutually consistent: if node A lists B in childIndices,
    //      B.parentIndex must equal A's own index, and vice versa.
    //   3. The parent relation is acyclic: no node is reachable from itself by repeatedly
    //      following parentIndex (equivalently, no node is its own descendant).
    //   4. Every node reachable from a root is in fact reached (no disconnected/orphaned node is
    //      silently missing from `dag.rootIndices`'s reachable set) and, conversely, every node
    //      in `dag.nodes` is reachable from some root (no dangling node outside the declared
    //      forest).
    //   5. Error selection is strictly monotonic toward the root: for every non-root node,
    //      parentError > clusterError (strict); for every root, parentError == +infinity.
    //
    // Returns true iff every check passes across the whole DAG. On failure, `outErrors` receives
    // one human-readable message per violation found (validation does not stop at the first
    // failure, so a single call reports everything wrong with the DAG).
    bool ValidateClusterDAG(const ClusterDAG& dag, std::vector<std::string>& outErrors);

}
