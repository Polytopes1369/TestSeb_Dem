#pragma once
// Shared value types for the World Partition offline toolset (tools/WorldPartition/): the AABB,
// streaming-flag bitmask and transform every OFPA actor file (OfpaActor.h) and scene index entry
// (SceneIndex.h) is built from. Kept in one header so the two file formats -- which must agree
// bit-for-bit on ActorStreamingFlags's layout, since a SceneIndexEntry's flags are a verbatim
// copy of its source ActorRecord's -- can never drift apart.
//
// This is offline, editor-side tooling (asset authoring / cook-time pipeline), never linked into
// the shipping demo executable: it lives under tools/, outside src/, so the top-level
// CMakeLists.txt's `file(GLOB_RECURSE ALL_SOURCES "src/*.cpp")` never picks it up (see
// CMakeLists.txt's WorldPartitionTools section for how it IS built: as standalone test
// executables only, matching the project's existing framework-free test pattern).

#include <cstdint>
#include "core/maths/Maths.h"

namespace worldpartition {

    // Axis-aligned bounding box, world or local space depending on context (ActorRecord carries
    // both explicitly, see OfpaActor.h).
    struct AABB {
        maths::vec3 boundsMin{};
        maths::vec3 boundsMax{};

        maths::vec3 Center() const { return maths::AABBCenter(boundsMin, boundsMax); }

        // Standard slab-vs-slab overlap test, inclusive on shared boundaries -- deliberately so:
        // SpatialHashGrid::Build relies on an actor exactly touching a cell boundary being
        // considered present in both neighboring cells (see that file's straddling-actor note).
        constexpr bool Overlaps(const AABB& other) const {
            return boundsMin.x <= other.boundsMax.x && boundsMax.x >= other.boundsMin.x &&
                boundsMin.y <= other.boundsMax.y && boundsMax.y >= other.boundsMin.y &&
                boundsMin.z <= other.boundsMax.z && boundsMax.z >= other.boundsMin.z;
        }
    };

    // Mirrors UE5.8's UWorldPartition streaming source/actor descriptor flags, trimmed to the
    // subset this offline toolset actually needs to make streaming/HLOD decisions from.
    enum class ActorStreamingFlags : uint32_t {
        None = 0,
        AlwaysLoaded = 1u << 0,    // Never streamed out, regardless of the currently-loaded cell set (player start, sky dome, ...).
        SpatiallyLoaded = 1u << 1, // Default: loaded/unloaded purely by its owning cell's streaming state.
        RuntimeOnly = 1u << 2,     // Never listed in editor-only tooling views; only ever loaded at runtime.
        EditorOnly = 1u << 3,      // Never cooked/streamed at runtime (guides, helpers, editor-only gizmo actors).
        HLODGenerated = 1u << 4,   // This actor IS a generated HLOD proxy, not authored content -- excluded from further HLOD passes so HLODs are never recursively HLOD-ed.
        Hidden = 1u << 5,          // Editor-authored visibility toggle, independent of streaming state.
    };

    constexpr ActorStreamingFlags operator|(ActorStreamingFlags a, ActorStreamingFlags b) {
        return static_cast<ActorStreamingFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    constexpr ActorStreamingFlags operator&(ActorStreamingFlags a, ActorStreamingFlags b) {
        return static_cast<ActorStreamingFlags>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    }
    constexpr ActorStreamingFlags& operator|=(ActorStreamingFlags& a, ActorStreamingFlags b) {
        a = a | b;
        return a;
    }
    constexpr bool HasFlag(ActorStreamingFlags flags, ActorStreamingFlags test) {
        return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(test)) != 0;
    }

    // World-space placement. Kept separate from maths::mat4 (rather than storing a pre-composed
    // matrix) because per-field editing/serialization -- exactly what an OFPA actor file's
    // consumer (an editor property panel, a diff tool) needs -- is lossy to do against a matrix
    // (decomposition is not unique under non-uniform scale + rotation).
    struct ActorTransform {
        maths::vec3 position{};
        maths::quat rotation{};
        maths::vec3 scale{ 1.0f, 1.0f, 1.0f };
    };

}
