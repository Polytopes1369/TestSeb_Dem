// PCG framework roadmap, Phase 3.2 ("Self-Pruning Filter") -- implementation. See
// PcgSelfPruningFilter.h for the full design rationale (uniform hash grid sizing, the
// deterministic earlier-index-wins keep/discard rule, and the ScaledByBounds distance formula
// summary). This file fleshes out PcgSpatialHashGrid's cell math and PruneByDistance's actual
// sweep.

#include "pcg/PcgSelfPruningFilter.h"

#include <algorithm>
#include <cmath>

namespace pcg {

    namespace {

        // Guards a caller-supplied cell size of <= 0 from producing a division-by-zero (or a
        // degenerate single giant cell that defeats the whole point of bucketing) when hashing a
        // world-space position into a cell coordinate -- same defensive-floor pattern as
        // PcgVolumeSampler.cpp's own kMinGridSpacing.
        constexpr float kMinCellSize = 1.0e-4f;

        // Squared world-space distance between two positions -- used everywhere below instead of
        // the actual (sqrt-taking) distance, since every comparison this file makes is against
        // another squared distance; avoids one sqrt() per candidate pair for zero behavioral
        // difference.
        float DistanceSquared(const maths::vec3& a, const maths::vec3& b) {
            const maths::vec3 delta = a - b;
            return delta.Dot(delta);
        }

        // This point's own world-space bounding-sphere radius, for `ScaledByBounds` mode: the
        // local-space AABB's half-diagonal length (maths::AABBRadius -- the distance from the
        // bounds center to any corner, i.e. a sphere that fully encloses the local AABB)
        // conservatively scaled by the LARGEST of the point's 3 scale components. Using the
        // largest component (rather than e.g. averaging, or a proper per-axis-scaled AABB
        // recompute) keeps this a cheap, always-safe OVER-estimate for non-uniform scale -- a
        // point squashed thin on one axis but stretched long on another still gets a radius at
        // least as large as its true longest extent, so ScaledByBounds pruning never UNDER-prunes
        // (never lets two objects visually clip) at the cost of occasionally pruning slightly more
        // conservatively than a tighter (and more expensive) exact OBB-vs-OBB test would.
        // std::abs guards a negative scale component (a deliberate mirror-flip transform, which
        // this codebase's PcgPoint allows) from flipping the sign of the resulting radius.
        float ComputePointBoundingRadius(const PcgPoint& point) {
            const float localRadius = maths::AABBRadius(point.boundsMin, point.boundsMax);
            const float maxScale = std::max({ std::abs(point.scale.x), std::abs(point.scale.y), std::abs(point.scale.z) });
            return localRadius * maxScale;
        }

        // The effective minimum separation distance for one SPECIFIC pair of points (A, B).
        //
        //   Uniform:       pairMinDistance = baseMinDistance                       (unconditionally)
        //   ScaledByBounds: pairMinDistance = max(baseMinDistance, radius(A) + radius(B))
        //
        // The ScaledByBounds formula is a bounding-SPHERE non-overlap test: two spheres of radius
        // radius(A) and radius(B) centered at A.position/B.position do not overlap exactly when
        // the distance between their centers exceeds the SUM of their radii -- the same test
        // broad-phase collision systems use for "could these two objects possibly be touching".
        // `max(baseMinDistance, ...)` additionally guarantees `baseMinDistance` always acts as a
        // FLOOR even for a pair of points with a tiny/zero bounds extent (two point-like objects
        // with no meaningful size still get at least the caller's requested base spacing, not
        // zero) -- see PcgSelfPruningParams::minDistance's own header comment.
        float ComputePairMinDistance(const PcgPoint& a, const PcgPoint& b, float baseMinDistance, PcgPruningMode mode) {
            if (mode == PcgPruningMode::Uniform) {
                return baseMinDistance;
            }
            const float radiusSum = ComputePointBoundingRadius(a) + ComputePointBoundingRadius(b);
            return std::max(baseMinDistance, radiusSum);
        }

        // Upper bound on the effective pruning distance ANY pair drawn from `points` could
        // possibly need, used solely to size the acceleration grid's cells (see
        // PcgSpatialHashGrid's own "cellSize = 2 * queryRadius" correctness invariant, documented
        // in PcgSelfPruningFilter.h's header comment). In Uniform mode this is trivially
        // `baseMinDistance` itself (every pair uses exactly that distance). In ScaledByBounds mode
        // the worst case is two points that BOTH happen to have the single largest bounding
        // radius found anywhere in the input, so `2 * maxRadius` (clamped up to `baseMinDistance`,
        // which still acts as ScaledByBounds's own floor) is a safe, always-sufficient bound --
        // never an underestimate, which is the only direction that would break correctness (an
        // OVERestimate merely makes the grid's cells a bit larger / neighbor buckets a bit more
        // populated than the theoretical minimum, a performance-only cost).
        float ComputeMaxPossiblePairDistance(const std::vector<PcgPoint>& points, float baseMinDistance, PcgPruningMode mode) {
            if (mode == PcgPruningMode::Uniform) {
                return baseMinDistance;
            }
            float maxRadius = 0.0f;
            for (const PcgPoint& point : points) {
                maxRadius = std::max(maxRadius, ComputePointBoundingRadius(point));
            }
            return std::max(baseMinDistance, 2.0f * maxRadius);
        }

    } // namespace

    // ------------------------------------------------------------------------------------------
    // PcgSpatialHashGrid
    // ------------------------------------------------------------------------------------------
    PcgSpatialHashGrid::PcgSpatialHashGrid(float cellSize)
        : m_CellSize(std::max(cellSize, kMinCellSize)) {
    }

    int32_t PcgSpatialHashGrid::CellCoord(float value) const {
        // floor(), not truncation towards zero: -0.3 with a cell size of 1.0 must land in cell
        // -1 (not cell 0), so that negative-and-positive world-space positions bucket into a
        // contiguous, correctly-adjacent sequence of cells straddling the origin.
        return static_cast<int32_t>(std::floor(value / m_CellSize));
    }

    int64_t PcgSpatialHashGrid::MakeCellKey(int32_t cx, int32_t cy, int32_t cz) {
        // Biases each signed cell coordinate into an unbiased (non-negative) range before packing,
        // then packs 3 x 21-bit fields into one 64-bit key -- 21 bits gives a per-axis cell
        // coordinate range of [-2^20, 2^20 - 1] (~ +/-1,048,576 cells), which at a typical
        // pruning-radius-derived cell size (well under 100 world-space units for any realistic PCG
        // scatter density) covers a many-tens-of-millions-of-units world extent per axis --
        // comfortably more than this engine's LWC-rebased (world/LwcOrigin.h) camera-relative
        // coordinates ever need a single grid to span.
        constexpr int64_t kBias = int64_t{ 1 } << 20;
        constexpr int64_t kBits = 21;
        constexpr int64_t kMask = (int64_t{ 1 } << kBits) - 1;
        const int64_t ux = (static_cast<int64_t>(cx) + kBias) & kMask;
        const int64_t uy = (static_cast<int64_t>(cy) + kBias) & kMask;
        const int64_t uz = (static_cast<int64_t>(cz) + kBias) & kMask;
        return (ux << (2 * kBits)) | (uy << kBits) | uz;
    }

    void PcgSpatialHashGrid::Insert(const maths::vec3& position, uint32_t pointIndex) {
        const int32_t cx = CellCoord(position.x);
        const int32_t cy = CellCoord(position.y);
        const int32_t cz = CellCoord(position.z);
        m_Cells[MakeCellKey(cx, cy, cz)].push_back(pointIndex);
    }

    void PcgSpatialHashGrid::QueryNeighborCells(const maths::vec3& position, std::vector<uint32_t>& outCandidates) const {
        const int32_t cx = CellCoord(position.x);
        const int32_t cy = CellCoord(position.y);
        const int32_t cz = CellCoord(position.z);

        // The full 3x3x3 block (27 cells, including `position`'s own cell at dx=dy=dz=0) -- see
        // this file's header comment for why a cell size of 2*queryRadius makes exactly this
        // neighborhood sufficient to find every point within queryRadius.
        for (int32_t dz = -1; dz <= 1; ++dz) {
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    const auto it = m_Cells.find(MakeCellKey(cx + dx, cy + dy, cz + dz));
                    if (it == m_Cells.end()) {
                        continue;
                    }
                    outCandidates.insert(outCandidates.end(), it->second.begin(), it->second.end());
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // PruneByDistance
    // ------------------------------------------------------------------------------------------
    std::vector<PcgPoint> PruneByDistance(const std::vector<PcgPoint>& points, float minDistance, PcgPruningMode mode) {
        if (points.empty()) {
            return {};
        }

        const float safeMinDistance = std::max(minDistance, 0.0f);

        // Size the grid's cells at 2x the worst-case pair distance ANY pair in this input could
        // require (see ComputeMaxPossiblePairDistance's own comment) -- this is what guarantees
        // QueryNeighborCells' fixed 3x3x3 search always finds every already-kept point close
        // enough to matter, for every pair this call will ever actually test, in either mode.
        const float maxPossiblePairDistance = ComputeMaxPossiblePairDistance(points, safeMinDistance, mode);
        const float cellSize = std::max(2.0f * maxPossiblePairDistance, kMinCellSize);
        PcgSpatialHashGrid grid(cellSize);

        std::vector<PcgPoint> kept;
        kept.reserve(points.size());

        // Reused across every QueryNeighborCells call below (cleared, not reallocated, each
        // iteration) so this sweep performs at most one scratch-vector growth spike total rather
        // than one heap allocation per candidate point.
        std::vector<uint32_t> candidateScratch;

        // Deterministic, strictly-input-order sweep (see this file's header comment for the full
        // "earlier-indexed point always wins" determinism rationale): point `i` is tested only
        // against points with INDEX < i that already survived and were inserted into `grid` --
        // never against points that come later in `points`, and never against points already
        // discarded (those were never inserted, so QueryNeighborCells can't return them).
        for (size_t i = 0; i < points.size(); ++i) {
            const PcgPoint& candidate = points[i];

            candidateScratch.clear();
            grid.QueryNeighborCells(candidate.position, candidateScratch);

            bool tooCloseToSurvivor = false;
            for (uint32_t survivorIndex : candidateScratch) {
                const PcgPoint& survivor = points[survivorIndex];
                const float pairMinDistance = ComputePairMinDistance(candidate, survivor, safeMinDistance, mode);
                // Strict less-than: a candidate EXACTLY at the pair's minimum distance from a
                // survivor is NOT "too close" -- `minDistance` is the smallest ALLOWED separation,
                // not the largest forbidden one, matching this header's own "no point closer than
                // minDistance to a surviving point should itself survive" contract verbatim (a
                // point AT exactly minDistance is not closer than it).
                if (DistanceSquared(candidate.position, survivor.position) < pairMinDistance * pairMinDistance) {
                    tooCloseToSurvivor = true;
                    break;
                }
            }

            if (!tooCloseToSurvivor) {
                // Insert BEFORE pushing to `kept` using `i` (the ORIGINAL input index, not
                // `kept`'s own size) as the stored index -- QueryNeighborCells' returned indices
                // are looked up against `points` (the original input), not `kept`, throughout this
                // loop, so the two must stay consistent.
                grid.Insert(candidate.position, static_cast<uint32_t>(i));
                kept.push_back(candidate);
            }
        }

        return kept;
    }

}
