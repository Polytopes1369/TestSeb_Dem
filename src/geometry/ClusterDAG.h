#pragma once
// Nanite-style cluster DAG: connects high-resolution leaf clusters (ClusterPartitioner.h) to
// progressively coarser, simplified parent clusters (ClusterGrouping.h + MeshSimplifier.h), all
// the way up to one or more roots, so a renderer can pick a view-dependent LOD cut through the
// hierarchy instead of a single fixed mesh resolution.
//
// Level 0 is the set of leaf clusters produced by PartitionMeshIntoClusters -- exact geometry,
// e_local (clusterError) == 0. Every level above that is built by:
//   1. grouping up to 4 adjacent same-level nodes into one ClusterDAGGroup (ClusterGrouping::
//      GroupAdjacentClusters/GroupItemsIntoQuads for level 0->1, or this file's own position-based
//      grouping for level 1->2, 2->3, ... since nodes above level 0 no longer carry indices into
//      the original mesh -- only their locked boundary vertices, whose positions are preserved
//      bit-exact from the original mesh, survive as a reliable adjacency signal);
//   2. simplifying the group's merged mesh with SimplifyMeshQEM, honoring the group's boundary
//      lock so its outer boundary (shared with un-grouped neighbors) never moves;
//   3. if the simplified mesh still doesn't fit one cluster's vertex/triangle cap, spatially
//      re-splitting it (SplitSimplifiableMesh) into up to kMaxGroupOutputClusters renderable
//      output nodes that ALL replace the same group's members -- this is what makes the structure
//      a true DAG (a member can be summarized by more than one coarser node) rather than a strict
//      tree, and is why ClusterDAGGroup exists as its own entity instead of folding straight into
//      ClusterDAGNode::parentIndex (see that struct's own comment);
//   4. recording the group's error/bounding sphere once, shared by every member and every output.
//
// Error bookkeeping (e_local / e_parent): each node stores clusterError (e_local), the geometric
// error introduced by using this node's mesh instead of its children's exact combined geometry,
// and parentError (e_parent), its own parent GROUP's groupError (+infinity for a root). Monotonicity
// is enforced constructively, not just checked after the fact: a group's groupError is always set
// to strictly more than the largest clusterError among its members (see kMinimumErrorStep below),
// so e_parent > e_local holds for every non-root node by construction. A runtime LOD cut can then
// evaluate each node independently against its own baked parentError -- no traversal needed for
// that decision (see renderer::ClusterLODSelectionPass) -- and only the (separate, residency-driven)
// ancestor walk needs to actually follow a node's group's output cluster(s).

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h" // geometry::kMaxGroupOutputClusters -- one shared definition
                                     // between the CPU build (this file) and the on-disk format
                                     // DAGNodeEntry::parentClusterID is sized against.
#include "geometry/MeshSimplifier.h"
#include "renderer/RenderTypes.h"

namespace geometry {

    constexpr uint32_t kInvalidDAGGroupIndex = 0xFFFFFFFFu;

    // Every group's groupError is bumped above the largest of its members' clusterError by
    // at least max(kMinimumErrorStepAbsolute, membersMaxError * kMinimumErrorStepRelative),
    // even when the QEM simplification pass itself introduced (numerically) zero measurable
    // error -- e.g. simplifying an already-flat patch. This guarantees strictly increasing error
    // everywhere in the DAG, which ValidateClusterDAG treats as a hard invariant (see its doc
    // comment). A purely absolute floor is not enough on its own: a float's precision (ULP) grows
    // with its magnitude, so adding a fixed 1e-6 to an error value already around 50-100 (a
    // realistic magnitude a few levels up a real mesh's DAG) silently rounds back down to the
    // exact same float, defeating the guarantee it was meant to provide -- hence the relative term.
    constexpr float kMinimumErrorStepAbsolute = 1e-4f;
    constexpr float kMinimumErrorStepRelative = 1e-3f; // 0.1% of the largest member error.

    // One DAG "group": the intermediate unit one merge+simplification pass produces from up to 4
    // same-level member nodes (see ClusterGrouping::GroupItemsIntoQuads) -- NOT itself a renderable
    // cluster. Exists so a group whose simplified mesh doesn't fit the per-cluster vertex/triangle
    // cap can be re-split into up to kMaxGroupOutputClusters renderable output clusters that ALL
    // replace the SAME set of members, without ever giving an individual member more than one
    // parent reference: every member points at this one group (ClusterDAGNode::parentGroupIndex),
    // and only the group tracks however many (1 or kMaxGroupOutputClusters) concrete outputs
    // resulted. This indirection is what makes a member's coarser representation resolvable to
    // more than one node (a true DAG) while every individual ClusterDAGNode still needs to store
    // only a single upward reference.
    //
    // groupError/groupSphereCenter/groupSphereRadius are computed from the group's merged+
    // simplified mesh BEFORE any re-split -- deliberately one shared value every member (and every
    // output sibling, when re-split fires) uses as its "parent" projection, so which specific
    // output cluster a runtime LOD cut actually selects never matters for the screen-error test
    // (renderer::ClusterLODSelectionPass): it is a property of the whole group's simplification,
    // not of any one spatial fragment produced by the later re-split.
    struct ClusterDAGGroup {
        std::vector<uint32_t> memberClusterIndices; // Up to 4: this group's same-level input nodes.
        std::vector<uint32_t> outputClusterIndices; // 1 to kMaxGroupOutputClusters coarser nodes.

        float groupError = 0.0f;
        maths::vec3 groupSphereCenter{};
        float groupSphereRadius = 0.0f;
    };

    // One node of a cluster DAG.
    struct ClusterDAGNode {
        SimplifiableMesh mesh;

        // Which ClusterDAGGroup produced this node by merging+simplifying (and, if needed,
        // re-splitting) its members -- kInvalidDAGGroupIndex for a level-0 leaf. This node's
        // children are dag.groups[sourceGroupIndex].memberClusterIndices.
        uint32_t sourceGroupIndex = kInvalidDAGGroupIndex;

        // Which ClusterDAGGroup this node is itself a MEMBER of (i.e. was merged into) --
        // kInvalidDAGGroupIndex for a root. This node's parent-facing data (parentError below, and
        // the group's shared bounding sphere) comes from dag.groups[parentGroupIndex].
        uint32_t parentGroupIndex = kInvalidDAGGroupIndex;

        maths::vec3 boundsMin{};
        maths::vec3 boundsMax{};
        maths::vec3 sphereCenter{};
        float sphereRadius = 0.0f;

        // e_local: geometric error introduced by using this node's mesh in place of its
        // children's exact, pre-simplification geometry. Always 0 for a leaf (level 0): a leaf
        // *is* the exact original geometry, no simplification has been applied to it yet. For a
        // node produced by a re-split output, this equals its whole group's groupError (see
        // ClusterDAGGroup's own comment for why the error is shared, not per-fragment).
        float clusterError = 0.0f;

        // e_parent: this node's own parent GROUP's groupError. +infinity for a root, matching
        // ClusterFormat.h's existing DAGNodeEntry::parentError convention (there is no coarser
        // representation to fall back past a root). Baked as a plain copy at build time (instead
        // of requiring every reader to dereference parentGroupIndex) so existing consumers that
        // read node.parentError directly keep working unchanged.
        float parentError = std::numeric_limits<float>::infinity();

        uint32_t level = 0; // 0 = leaf; +1 for every grouping/simplification pass above that.

        // True if this node (and every leaf descendant it summarizes) should sample the entity's
        // opacity-cutout mask at render time -- see geometry::MeshCluster::isMasked, which this is
        // stamped from at level 0. Grouping (ClusterGrouping::BuildClusterAdjacencyWeights, this
        // file's own BuildLevelAdjacencyWeights) never merges nodes with different isMasked
        // values, so every non-root node's isMasked is guaranteed to equal its parent group's --
        // checked explicitly by ValidateClusterDAG.
        bool isMasked = false;
    };

    struct ClusterDAG {
        std::vector<ClusterDAGNode> nodes;
        std::vector<ClusterDAGGroup> groups;
        std::vector<uint32_t> rootIndices; // Nodes with parentGroupIndex == kInvalidDAGGroupIndex.
    };

    // Builds the full multi-level DAG for every triangle of `allIndices` whose vertices carry
    // Vertex::meshID == targetMeshID. Returns a DAG with an empty `nodes`/`rootIndices` if no
    // triangle matches targetMeshID.
    //
    // `maskTextureIndex` is forwarded to PartitionMeshIntoClusters (see its doc comment in
    // ClusterPartitioner.h) to classify and, if needed, split leaf clusters by opacity; pass
    // geometry::kInvalidMaskTextureIndex for an entity with no cutout material.
    //
    // Grouping+simplification repeats, level by level, until a single root remains or no further
    // pairing is possible within a level (e.g. topologically disconnected mesh islands, or two
    // differently-classified (opaque vs. masked) clusters that can never merge -- see this file's
    // BuildLevelAdjacencyWeights) -- in the latter case every still-unpaired top node becomes its
    // own root, so `rootIndices` may contain more than one entry.
    ClusterDAG BuildClusterDAG(
        uint32_t targetMeshID,
        const std::vector<renderer::Vertex>& allVertices,
        const std::vector<uint32_t>& allIndices,
        uint32_t maskTextureIndex);

    // Validates a whole DAG's structural and error-monotonicity invariants. Intended to run once
    // at startup, right after BuildClusterDAG, before the DAG is trusted for runtime LOD cuts.
    // Walks every node reachable from `dag.rootIndices` down through its sourceGroupIndex group's
    // memberClusterIndices (equivalently: every node's ancestor chain up through parentGroupIndex)
    // and checks:
    //   1. Every sourceGroupIndex/parentGroupIndex entry is in range, and every
    //      memberClusterIndices/outputClusterIndices entry of every group is in range.
    //   2. The member/group relation is mutually consistent: if group G lists node N in
    //      memberClusterIndices, N.parentGroupIndex must equal G's own index; if G lists node N in
    //      outputClusterIndices, N.sourceGroupIndex must equal G's own index; and vice versa both
    //      ways.
    //   3. The parent relation is acyclic: no node is reachable from itself by repeatedly
    //      following parentGroupIndex -> that group's outputClusterIndices (equivalently, no node
    //      is its own descendant via sourceGroupIndex -> memberClusterIndices).
    //   4. Every node reachable from a root is in fact reached (no disconnected/orphaned node is
    //      silently missing from `dag.rootIndices`'s reachable set) and, conversely, every node
    //      in `dag.nodes` is reachable from some root (no dangling node outside the declared
    //      forest).
    //   5. Error selection is strictly monotonic toward the root: for every non-root node,
    //      parentError > clusterError (strict); for every root, parentError == +infinity. Every
    //      node produced by the same group (its outputClusterIndices) shares an identical
    //      clusterError, since re-splitting a group's simplified mesh changes its spatial extent
    //      but not its approximation quality.
    //
    // Returns true iff every check passes across the whole DAG. On failure, `outErrors` receives
    // one human-readable message per violation found (validation does not stop at the first
    // failure, so a single call reports everything wrong with the DAG).
    bool ValidateClusterDAG(const ClusterDAG& dag, std::vector<std::string>& outErrors);

}
