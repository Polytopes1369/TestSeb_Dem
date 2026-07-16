#pragma once
// Small CPU-side hashing/keying primitives shared by the mesh-processing pipeline (ClusterDAG,
// ClusterGrouping, MeshSimplifier, MeshSDFGenerator, FallbackMeshBuilder). These were previously
// copy-pasted independently into each of those files; centralizing them here means a future fix
// (e.g. to the hash mixing constants) only needs to happen once.

#include <cstdint>
#include <cstring>
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

    // Exact (bit-value, not epsilon) position equality/hash. Valid only where positions are never
    // arithmetically touched between the point two "identical" vertices are compared (e.g. locked
    // vertices that descend unmodified from the same original mesh vertex, or DAG-root seam
    // vertices never perturbed by anything downstream of the simplification pass that produced
    // them) -- in those cases two genuinely-shared vertices are bit-for-bit identical, not merely
    // close, so an exact hash/equality is correct and far cheaper than an epsilon-tolerant one.
    struct PositionKey {
        float x, y, z;
        bool operator==(const PositionKey& o) const { return x == o.x && y == o.y && z == o.z; }
    };

    struct PositionKeyHash {
        size_t operator()(const PositionKey& p) const {
            auto normalizeZero = [](float f) { return f == 0.0f ? 0.0f : f; }; // -0.0 == +0.0
            float nx = normalizeZero(p.x), ny = normalizeZero(p.y), nz = normalizeZero(p.z);
            uint32_t bx, by, bz;
            std::memcpy(&bx, &nx, sizeof(bx));
            std::memcpy(&by, &ny, sizeof(by));
            std::memcpy(&bz, &nz, sizeof(bz));
            uint64_t h = 1469598103934665603ull;
            auto mix = [&](uint32_t v) { h ^= v; h *= 1099511628211ull; };
            mix(bx); mix(by); mix(bz);
            return static_cast<size_t>(h);
        }
    };

    inline PositionKey MakePositionKey(const maths::vec3& p) { return PositionKey{ p.x, p.y, p.z }; }

}
