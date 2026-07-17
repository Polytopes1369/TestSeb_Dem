#pragma once
// Small CPU-side hashing/keying primitives shared by the mesh-processing pipeline (ClusterDAG,
// ClusterGrouping, MeshSimplifier, MeshSDFGenerator, FallbackMeshBuilder). These were previously
// copy-pasted independently into each of those files; centralizing them here means a future fix
// (e.g. to the hash mixing constants) only needs to happen once.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "core/maths/Maths.h"

namespace geometry {

    // Packs an unordered pair of vertex/index indices into a single 64-bit key, used as the
    // dedup/lookup key for edge- or index-pair-keyed maps (e.g. "how many triangles use this
    // edge", "cross-cluster adjacency weight between these two clusters"). Order-independent:
    // PackOrderedPair(a, b) == PackOrderedPair(b, a).
    inline uint64_t PackOrderedPair(uint32_t a, uint32_t b) {
        uint32_t lo = (a < b) ? a : b;
        uint32_t hi = (a < b) ? b : a;
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    }

    // Epsilon-quantized (not bit-exact) position equality/hash, used to recognize two vertices
    // from DIFFERENT sources as "the same point" for stitching purposes (locked/boundary vertices
    // across sibling clusters in ClusterDAG.cpp's MergeLevelMeshes, or DAG-root seam vertices in
    // FallbackMeshBuilder's MergeRootsWeldingAllVertices).
    //
    // This USED to be bit-exact equality, on the documented assumption that "two genuinely-shared
    // vertices are bit-for-bit identical, not merely close" -- confirmed FALSE by the 2026-07-16
    // "clusters mis-generated"/"shattered geometry" investigation: two independently-authored
    // generator code paths that are meant to produce the exact same physical boundary point (e.g.
    // a pyramid's base-plane formula vs. its side-face-at-row-0 formula, or two adjacent side
    // faces' own edge formulas) can and do produce numerically-close-but-not-bit-identical floats
    // for what is conceptually one point, purely from floating-point non-associativity between the
    // two expressions. Under the old exact-match scheme, two such near-duplicate locked vertices
    // are treated as CONCEPTUALLY DIFFERENT points and never aliased together: at level 0 this
    // showed up as a rare, easy-to-miss zero-area "sliver" triangle (see MeshSimplifier.cpp's own
    // WeldResidualSliverTriangles, a narrower band-aid for that specific symptom); at a higher DAG
    // level, aggressive QEM simplification can move ONE of the two never-aliased "duplicate"
    // vertices far from the other while the un-aliased side of the seam stays put, tearing what
    // should be one continuous surface into two visibly disconnected pieces -- whole clusters
    // rendering as floating, detached shards, confirmed via the engine's own DEBUG_VIEW_NANITE_
    // CLUSTERS/INSTANCES views.
    //
    // Quantizing each coordinate to the nearest kPositionEpsilon (1e-4 = 0.1mm) before hashing/
    // comparing fixes this at the root: it is ~500x smaller than this engine's finest tessellation
    // spacing (config::VERTEX_SPACING = 0.05 = 5cm), so no two DISTINCT, intentionally-separate
    // vertices in any procedurally-generated mesh this engine produces can land in the same
    // quantization cell purely by chance, while comfortably absorbing the ~1e-6-1e-5-magnitude
    // discrepancies floating-point non-associativity between two different formulas actually
    // produces for the same conceptual point.
    constexpr float kPositionEpsilon = 1e-4f;

    struct PositionKey {
        int32_t qx, qy, qz;
        bool operator==(const PositionKey& o) const { return qx == o.qx && qy == o.qy && qz == o.qz; }
    };

    struct PositionKeyHash {
        size_t operator()(const PositionKey& p) const {
            uint64_t h = 1469598103934665603ull;
            auto mix = [&](int32_t v) { h ^= static_cast<uint32_t>(v); h *= 1099511628211ull; };
            mix(p.qx); mix(p.qy); mix(p.qz);
            return static_cast<size_t>(h);
        }
    };

    inline PositionKey MakePositionKey(const maths::vec3& p) {
        // std::lround (round-half-away-from-zero) rather than a plain truncating cast: without it,
        // two near-identical positions straddling an integer quantization boundary (e.g. 0.99999997
        // and 1.00000002, both "meant" to be 1.0) would truncate to DIFFERENT cells (0 and 1) and
        // still fail to alias -- exactly the failure mode this whole scheme exists to fix.
        return PositionKey{
            static_cast<int32_t>(std::lround(p.x / kPositionEpsilon)),
            static_cast<int32_t>(std::lround(p.y / kPositionEpsilon)),
            static_cast<int32_t>(std::lround(p.z / kPositionEpsilon))
        };
    }

}
