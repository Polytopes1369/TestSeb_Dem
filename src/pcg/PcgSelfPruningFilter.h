#pragma once

// PCG framework roadmap, Phase 3.2 ("Self-Pruning Filter"): UE5.8 PCG "Self Pruning" node parity --
// removes points from a point set that sit too close to another point already kept, so densely
// overlapping scatter output (e.g. thousands of points from the Surface Sampler, Phase 2.1) can be
// thinned into a set with a real minimum-separation guarantee before a later Spawner phase turns
// each surviving point into actual geometry (you don't want two rocks spawned inside each other).
//
// This phase is deliberately scoped to the filter itself: no Spawner/graph-engine-node wiring
// (later phases) lives here, matching every prior Phase 2 sampler's own "this layer only" scoping
// precedent (see e.g. PcgVolumeSampler.h's header comment).
//
// --- Spatial acceleration: uniform hash grid -------------------------------------------------
// A naive all-pairs O(n^2) distance check does not scale to realistic PCG point counts (a Surface
// or Terrain sampler can easily emit tens of thousands of points), so pruning instead buckets
// every point into a 3D uniform grid (PcgSpatialHashGrid, below) with a cell size of
// 2 * minDistance. Sizing the cell at exactly twice the pruning radius guarantees any OTHER point
// within `minDistance` of a given point can only ever land in that point's own cell or one of its
// 26 face/edge/corner-adjacent neighbor cells (a point up to `minDistance` away can cross at most
// one cell boundary along each axis, since a cell spans 2*minDistance) -- so PruneByDistance only
// ever needs to test a point against its own 3x3x3 cell neighborhood, never the full point set.
// This is the standard, well-known "uniform spatial hash" broad-phase technique (as used for
// neighbor queries in e.g. SPH fluid solvers and broad-phase collision detection) implemented
// directly here rather than pulling in a third-party spatial-structure library, matching this
// project's "no heavy frameworks" mandate (CLAUDE.md).
//
// --- Determinism & the keep/discard rule ------------------------------------------------------
// PruneByDistance processes points strictly in INPUT order (index 0, 1, 2, ...). Every point is
// tested against ONLY the points that came before it and already survived (i.e. were inserted into
// the grid) -- never against later points, and never against points already discarded. The very
// first point is always kept (nothing has been inserted yet, so it can never collide with
// anything). For any pair (A, B) with A appearing before B in the input and the two closer than
// the pruning distance, A is ALWAYS kept and B is ALWAYS discarded -- a stable, purely
// index-order-dependent rule with no dependency on point seed, position, density, or any other
// data-dependent tie-break. This is intentional: given the identical input vector (same points,
// same order) and the identical params, PruneByDistance always returns a byte-identical result
// (same surviving points, same relative order) on every run/platform -- this codebase's hard
// "the show must play back identically every run" requirement (see PcgSeededRandom.h's own header
// comment for the same principle applied to randomness). Shuffling the INPUT order is explicitly
// allowed to change WHICH points survive (a later point that used to be discarded may now come
// first and survive instead) -- that is a deliberate consequence of the order-dependent rule, not
// a bug -- but the OUTPUT of any single run is still fully self-consistent: no two points in the
// returned set are ever closer than their pair's effective minimum distance (see
// PcgPruningMode below), by construction (a candidate is only kept once every already-kept point
// within range has been checked).

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"

namespace pcg {

    // Selects how the effective minimum separation distance between a given PAIR of points is
    // derived. Both modes still use the SAME uniform hash grid / neighbor-cell search below --
    // only the actual distance comparison changes.
    enum class PcgPruningMode : uint8_t {
        // Every pair of points must be at least `minDistance` apart, full stop -- the point's own
        // `boundsMax`/`scale` are ignored entirely. Matches UE5.8's Self-Pruning node in its
        // simplest "uniform radius" configuration.
        Uniform,

        // The effective minimum distance for a given PAIR is derived from each point's own
        // extent, so a large scattered object (e.g. a boulder with a big `boundsMax`) pushes
        // nearby smaller/later points away more than a uniform radius would, while two tiny
        // points can still sit closer together than `minDistance` alone would allow. See
        // ComputePairMinDistance's own comment (PcgSelfPruningFilter.cpp) for the exact formula.
        ScaledByBounds,
    };

    // Parameters controlling PruneByDistance's output.
    struct PcgSelfPruningParams {
        // Base minimum separation distance, in world-space units. In `Uniform` mode this is the
        // ONLY distance used for every pair. In `ScaledByBounds` mode this is a FLOOR -- the
        // per-pair effective distance is never smaller than this value, only ever larger (see
        // ComputePairMinDistance's comment for why a floor, not a pure bounds-derived value, is
        // used: a pair of literally zero-extent points must still respect SOME minimum spacing).
        float minDistance = 1.0f;

        PcgPruningMode mode = PcgPruningMode::Uniform;
    };

    // Prunes `points` so that no two SURVIVING points are closer than their pair's effective
    // minimum distance (see PcgPruningMode above) apart, using the deterministic
    // earlier-index-wins rule documented at the top of this file. Returns a new vector containing
    // only the surviving points, in their original relative order (pruning never reorders
    // survivors, it only ever removes elements). `points` itself is not modified.
    //
    // Internally builds a PcgSpatialHashGrid sized off `params.minDistance` (see this file's own
    // "Spatial acceleration" comment above) and only ever tests each candidate point against the
    // already-kept points in its own cell + its 26 neighbor cells -- never the full input set --
    // so this scales to the thousands-of-points inputs a Surface/Terrain sampler realistically
    // produces (see tests/PcgSelfPruningFilterTests.cpp's own performance sanity check).
    std::vector<PcgPoint> PruneByDistance(const std::vector<PcgPoint>& points, float minDistance, PcgPruningMode mode);

    // Convenience overload taking a fully-populated params struct directly, for call sites that
    // already have one assembled (e.g. a future PCG graph node evaluating this filter from
    // serialized node parameters). Forwards directly to the 3-argument overload above.
    inline std::vector<PcgPoint> PruneByDistance(const std::vector<PcgPoint>& points, const PcgSelfPruningParams& params) {
        return PruneByDistance(points, params.minDistance, params.mode);
    }

    // ------------------------------------------------------------------------------------------
    // Uniform 3D spatial hash grid -- the acceleration structure PruneByDistance uses internally.
    // Exposed here (rather than kept file-private in the .cpp) since it is a small, generically
    // useful "bucket points into cells, query a cell's own 3x3x3 neighborhood" primitive that a
    // later PCG phase (e.g. a different proximity-based filter) may want to reuse directly rather
    // than reimplementing -- matching PcgSeededRandom's own precedent of exposing its hash
    // primitives (PcgHash32/PcgHashCombine) alongside the higher-level PruneByDistance-equivalent
    // convenience API.
    // ------------------------------------------------------------------------------------------
    class PcgSpatialHashGrid {
    public:
        // `cellSize` must be > 0; a degenerate/negative value is clamped internally to a tiny
        // positive minimum (see PcgSelfPruningFilter.cpp's kMinCellSize) rather than dividing by
        // zero when hashing a cell coordinate.
        explicit PcgSpatialHashGrid(float cellSize);

        // Inserts `position` under `pointIndex` (an index into whatever caller-side point array
        // `pointIndex` refers to -- this grid stores indices only, never a copy of the point
        // itself, keeping it a pure spatial-lookup structure with no PcgPoint dependency beyond
        // the vec3 position it was built to bucket).
        void Insert(const maths::vec3& position, uint32_t pointIndex);

        // Appends the point-index of every entry inserted into `position`'s own cell and all 26
        // face/edge/corner-adjacent neighbor cells (27 cells total, including `position`'s own)
        // to `outCandidates`. Does NOT clear `outCandidates` first -- callers reuse one scratch
        // vector across many queries to avoid a per-call heap allocation (see
        // PruneByDistance's own implementation).
        void QueryNeighborCells(const maths::vec3& position, std::vector<uint32_t>& outCandidates) const;

        float CellSize() const { return m_CellSize; }

    private:
        // Integer cell coordinate a world-space position falls into along one axis.
        int32_t CellCoord(float value) const;

        // Combines 3 signed cell coordinates into one 64-bit key for m_Cells. Packing 3
        // independent 21-bit-ish signed ranges into a 64-bit integer (rather than e.g. hashing
        // with XOR, which risks collisions between genuinely distinct cells) keeps every distinct
        // (cx,cy,cz) triple mapped to a distinct key, at the cost of an implicit (generous, +/-
        // ~1,000,000 cells per axis) world-space range limit -- more than sufficient for this
        // engine's LWC-rebased (see world/LwcOrigin.h) camera-relative world coordinates, which
        // never need a single grid to span more than a few thousand cells in any one axis.
        static int64_t MakeCellKey(int32_t cx, int32_t cy, int32_t cz);

        float m_CellSize;
        std::unordered_map<int64_t, std::vector<uint32_t>> m_Cells;
    };

}
