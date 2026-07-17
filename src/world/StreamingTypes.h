#pragma once
// Shared value types for the runtime World Partition streaming evaluator (StreamingManager.h).
//
// Deliberately independent from tools/WorldPartition/ (the OFFLINE OFPA/HLOD authoring toolset):
// that module lives outside src/ specifically so it never links into the shipping executable (see
// its own WorldPartitionTypes.h header comment), so runtime code under src/ must never depend on
// it -- doing so would either drag editor-only code into the game .exe or make the runtime build
// depend on a directory that a stripped-down production checkout might not even have. CellCoord
// here is therefore its own small, 2-axis (ground-plane) POD, not a reuse of the offline tool's
// 3-axis worldpartition::CellCoord; and actor identity here is a lightweight runtime handle
// (uint64_t), not the offline tool's 128-bit worldpartition::Uuid -- exactly the same split real
// engines draw between an editor-only GUID and a runtime instance handle.

#include <cstdint>
#include <functional>

#include "core/maths/Maths.h"

namespace world {

    // Ground-plane cell coordinate: the runtime streaming grid is always 2D (matching UE5.8's own
    // World Partition runtime grid -- a cell's world-space column has no height limit), unlike the
    // offline authoring grid which optionally buckets a 3rd (Y) axis for volumetric HLOD authoring.
    struct CellCoord {
        int32_t x = 0;
        int32_t z = 0;

        constexpr bool operator==(const CellCoord& other) const { return x == other.x && z == other.z; }
    };

    struct CellCoordHash {
        size_t operator()(const CellCoord& c) const noexcept {
            size_t h = std::hash<int32_t>{}(c.x);
            h ^= std::hash<int32_t>{}(c.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };

    // A camera or arbitrary point-of-interest driving streaming decisions. Multiple sources may
    // be active at once (split-screen, a remote spectator camera, a gameplay "streaming trigger"
    // volume, ...); StreamingManager::UpdateStreamingSources takes the full list and unions their
    // demands per cell (see that function's own comment).
    struct StreamingSource {
        maths::vec3 position{};
        float detailLoadRadius = 0.0f; // Within this radius: full per-actor detail must be resident.
        float hlodLoadRadius = 0.0f;   // Between detailLoadRadius and this radius: the cell's HLOD proxy must be resident (must be >= detailLoadRadius; if smaller, detailLoadRadius silently wins since it is always tested first -- see StreamingManager.cpp).
        uint32_t priority = 0;         // Reserved for future multi-source arbitration (e.g. a spectator camera's requests never preempting the player camera's); not yet consumed -- every source currently contributes equally to a cell's unioned desired representation.
    };

    // Coarsest -> finest representation a cell can be streamed in. Ordered so a numeric comparison
    // ( static_cast<uint8_t> ) tells you which of two representations is "more detailed" -- used
    // when multiple sources disagree about one cell (see StreamingManager::UpdateStreamingSources).
    enum class CellRepresentation : uint8_t {
        None = 0,
        HLOD = 1,
        FullDetail = 2,
    };

    // Exactly 4 states, matching the spec this class implements 1:1 -- deliberately NOT extended
    // with a 5th "swapping representation" state for an HLOD<->FullDetail transition; see
    // StreamingManager.cpp's sweep-pass comment for why that is handled as two sequential
    // Unload-then-Load convergence steps instead.
    enum class CellStreamingState : uint8_t {
        Unloaded,
        LoadingPending,
        LoadedActive,
        UnloadingPending,
    };

    // Caller-provided hook that actually performs a cell's load/unload work -- StreamingManager
    // only decides WHEN and WHICH cell, never HOW (dependency inversion, the same pattern
    // tools/WorldPartition/HlodPipeline.h's ActorMeshFetchFn uses to stay decoupled from any one
    // concrete asset/streaming backend). A real implementation is expected to eventually be backed
    // by an async disk-read + decompression pipeline; StreamingManager does not assume or require
    // one -- these calls may equally well be synchronous for a smaller scene.
    //
    // THREADING CONTRACT: every method below is called from a LoadingManager worker thread (see
    // core/LoadingManager.h), NEVER the main thread, and potentially from several worker threads
    // concurrently for different cells (up to StreamingManager's configured max-concurrent-loads
    // budget) -- an implementation must be safe to call concurrently with itself for different
    // `coord` values. It must NOT touch Vulkan command recording directly (Vulkan command buffers
    // are single-thread-recorded in this codebase) -- stage the load's result (e.g. into a
    // pre-allocated upload buffer or CPU-side actor list) and let the caller's own main-thread pump
    // step (this codebase's established core::LoadingManager::PumpCompletions discipline) perform
    // any actual GPU resource creation.
    class IWorldCellLoader {
    public:
        virtual ~IWorldCellLoader() = default;

        virtual void LoadCellFullDetail(const CellCoord& coord) = 0;
        virtual void LoadCellHlod(const CellCoord& coord) = 0;
        virtual void UnloadCell(const CellCoord& coord) = 0;
    };

}
