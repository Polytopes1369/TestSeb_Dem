#pragma once
// Builds one coarse "Fallback Mesh" per entity: a single, low-triangle-count proxy mesh continuing
// the exact same QEM simplification used to build every other DAG level (see MeshSimplifier.h),
// but applied to the DAG's root geometry with every vertex UNLOCKED -- the key difference from an
// interior DAG level. An interior level's boundary vertices stay locked because a still-unmerged
// sibling cluster references those exact positions and must stitch against them crack-free; the
// Fallback Mesh has no such sibling (it merges the entity's ENTIRE root geometry into one mesh), so
// nothing needs to stay watertight and QEM can compact far more aggressively than any interior
// level ever could.
//
// This mirrors Unreal Engine 5 Nanite's "Auto" fallback mode: driven by the same error metric as
// every other level, not by a fixed target triangle percentage (a fixed percentage produces wildly
// inconsistent results across meshes of different complexity). The only safety net is a 1%-of-
// original-leaf-triangle-count floor, so a topologically simple mesh cannot degenerate to a
// near-empty proxy and a highly detailed one is still bounded.
//
// Used as input geometry for hardware ray tracing acceleration structures (VkAccelerationStructure*),
// not for rasterization -- so, unlike every rasterization path, it deliberately ignores the
// opaque/masked cluster classification (geometry::ClusterDAGNode::isMasked): the fallback mesh
// merges every root regardless of classification into one single BVH proxy.

#include <cstdint>
#include <vector>
#include "core/maths/Maths.h"
#include "geometry/ClusterDAG.h"

namespace geometry {

    // A free-form (not fixed-capacity, unlike ClusterData) coarse proxy mesh: full 32-bit indices,
    // unquantized float positions/normals/UVs -- appropriate for a mesh that is built once, read
    // once at startup for BVH construction, and never paged/streamed.
    struct FallbackMesh {
        std::vector<maths::vec3> positions;
        std::vector<maths::vec3> normals; // Face-accumulated via geometry::ComputeFaceAccumulatedNormals.
        std::vector<maths::vec2> uvs;
        std::vector<uint32_t> triangles; // 3 indices per triangle, into the arrays above.
    };

    // Builds the Fallback Mesh for one entity's already-built ClusterDAG. Returns an empty
    // FallbackMesh (all vectors empty) if `dag` has no roots (i.e. BuildClusterDAG found no matching
    // geometry for that entity).
    FallbackMesh BuildFallbackMesh(const ClusterDAG& dag);

}
