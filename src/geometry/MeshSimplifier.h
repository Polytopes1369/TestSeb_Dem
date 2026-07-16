#pragma once
// Native, dependency-free triangle mesh simplifier based on Quadric Error Metrics (QEM,
// Garland & Heckbert 1997): iterative half-edge collapse, each candidate edge scored by the
// quadric (sum of squared distances to the incident triangle planes) of its two endpoints, always
// picking the globally cheapest available collapse first.
//
// This is used to build the coarser DAG-parent level of a cluster group (see ClusterGrouping.h):
// two adjacent ~128-triangle clusters are merged into one ~256-triangle group, then simplified
// back down to ~128 triangles by this simplifier. Vertices on the group's outer boundary are
// marked "locked" by the caller (ClusterGrouping::GroupAdjacentClusters) before simplification is
// invoked; this file enforces that lock as a hard constraint so a locked vertex's position never
// moves and it is never removed by a collapse. Since every neighboring cluster group that was NOT
// merged with this one still references those exact boundary positions, leaving them untouched is
// what keeps the simplified mesh watertight (crack-free) against its still-unsimplified or
// differently-simplified neighbors.
//
// No external mesh-processing library is used: quadric accumulation, the edge priority queue,
// and the fold-over/degenerate-triangle safeguards are all implemented here in native C++23.

#include <cstdint>
#include <vector>
#include "core/maths/Maths.h"

namespace geometry {

    // A local, self-contained triangle mesh operated on in place by SimplifyMeshQEM. Positions
    // are indexed locally to this mesh (not into any global/original vertex array), so the
    // simplifier is free to compact/renumber vertices as edges collapse.
    struct SimplifiableMesh {
        std::vector<maths::vec3> positions;

        // 3 indices per triangle, into positions.
        std::vector<uint32_t> triangles;

        // Per-vertex (parallel to positions): true means this vertex's position must never
        // change and it must never be removed by an edge collapse. Set by the caller before
        // simplification (typically: vertices on the mesh's/group's outer boundary).
        std::vector<bool> locked;

        // Per-vertex (parallel to positions): texture coordinates, carried through unchanged for
        // a locked/surviving vertex and averaged (midpoint) across the collapsed edge for a
        // non-locked one (see SimplifyMeshQEM's collapse-application step) -- QEM itself has no
        // attribute-aware error term for UV, so this is a simple, reviewable approximation, not a
        // placeholder. Populated from the source geometry (renderer::Vertex::uv) at every site
        // that builds a SimplifiableMesh (ClusterGrouping::GroupAdjacentClusters,
        // ClusterDAG.cpp's leaf-node construction and coarser-level merge).
        std::vector<maths::vec2> uvs;
    };

    // Simplifies `mesh` in place via iterative QEM edge collapses until BOTH the triangle count
    // is at or below targetTriangleCount AND the vertex count is at or below targetVertexCount
    // (default: unbounded, i.e. only the triangle target applies), or until:
    //   - no further collapse is possible without violating a lock (both endpoints locked),
    //     producing an inverted/degenerate triangle, or the candidate queue is exhausted.
    // The second case is expected, legitimate behavior for a topology-constrained mesh (e.g. a
    // group whose entire boundary is locked and whose interior is too sparse to reach either
    // target) -- it is not treated as an error. Note that every collapse reduces the triangle
    // count by at least one AND the vertex count by exactly one, so a caller with a tight
    // targetVertexCount (e.g. a fixed-capacity on-disk cluster format) may see simplification
    // continue well past targetTriangleCount purely to satisfy the vertex cap -- this is by
    // design, not a bug.
    //
    // Every locked vertex present in `mesh.locked` at call time keeps its exact original
    // position for the entire call, and is guaranteed to still be referenced by at least one
    // triangle in the result (locked vertices are the surviving endpoint of any collapse they
    // take part in, never the removed one).
    //
    // If `outMaxError` is non-null, it receives the worst-case (maximum, not summed) geometric
    // error introduced by any single collapse actually applied during this call, in the same
    // units as `mesh.positions` -- the square root of the largest quadric cost (v^T Q v, a sum of
    // squared plane distances) accepted into the mesh. This is 0 if no collapse was applied
    // (target already met, or every candidate was rejected/locked out). Callers building a DAG
    // level (see ClusterDAG.h) use this as that level's e_local contribution: the geometric error
    // of representing this simplified mesh in place of its pre-simplification children.
    //
    // Returns the number of triangles remaining after simplification (mesh.triangles.size() / 3).
    uint32_t SimplifyMeshQEM(
        SimplifiableMesh& mesh, uint32_t targetTriangleCount,
        uint32_t targetVertexCount = 0xFFFFFFFFu, float* outMaxError = nullptr);

}
