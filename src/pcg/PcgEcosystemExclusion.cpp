// PCG framework roadmap, Phase 8.2 ("Ecosystem Exclusion") -- implementation. See
// PcgEcosystemExclusion.h for the full design rationale (flat-vs-derived radius, the
// exactly-at-the-radius boundary convention, and why this file reuses PcgSpatialHashGrid --
// Phase 3.2, PcgSelfPruningFilter.h/.cpp -- unmodified rather than reimplementing spatial hashing).

#include "pcg/PcgEcosystemExclusion.h"

#include "pcg/PcgSelfPruningFilter.h"

#include "core/Logger.h"

#include <algorithm>
#include <cmath>
#include <format>

namespace pcg {

    namespace {

        // Guards a degenerate (zero/negative) cell size the same way
        // PcgSelfPruningFilter.cpp's own (file-private) kMinCellSize does -- see that file's
        // comment for why a tiny positive floor is used instead of dividing by zero when hashing a
        // world-space position into a cell coordinate.
        constexpr float kMinCellSize = 1.0e-4f;

        // Squared world-space distance -- avoids one sqrt() per candidate pair, exactly like
        // PcgSelfPruningFilter.cpp's own DistanceSquared helper (every comparison this file makes
        // is against another squared distance, so the plain squared form is sufficient throughout).
        float DistanceSquared(const maths::vec3& a, const maths::vec3& b) {
            const maths::vec3 delta = a - b;
            return delta.Dot(delta);
        }

        // This SUPPRESSING point's own world-space bounding-sphere radius, for `useDerivedRadius`
        // mode: the local-space AABB's half-diagonal length (maths::AABBRadius) conservatively
        // scaled by the LARGEST of the point's 3 scale components, then by the rule's own
        // `derivedRadiusMultiplier`. Deliberately the SAME "half-diagonal x largest abs scale
        // component" formula PcgSelfPruningFilter.cpp's own file-private ComputePointBoundingRadius
        // helper uses (see that file's own comment for why the largest scale component is a safe,
        // cheap over-estimate under non-uniform scale) -- reimplemented here rather than called
        // into, since that helper is anonymous-namespace-private to PcgSelfPruningFilter.cpp (see
        // PcgEcosystemExclusion.h's own header comment, "Which point's bounds, exactly", for why
        // this file only depends on that file's PUBLIC PcgSpatialHashGrid type, nothing else).
        // std::abs guards a negative (mirror-flip) scale component from flipping the resulting
        // radius's sign; the multiplier is floored at 0 so a negative/misauthored multiplier cannot
        // produce a negative "radius" that would silently disable suppression via always-false
        // comparisons further down.
        float ComputeSuppressingPointRadius(const PcgPoint& suppressingPoint, const PcgEcosystemExclusionRule& rule) {
            const float localRadius = maths::AABBRadius(suppressingPoint.boundsMin, suppressingPoint.boundsMax);
            const float maxScale = std::max({ std::abs(suppressingPoint.scale.x), std::abs(suppressingPoint.scale.y), std::abs(suppressingPoint.scale.z) });
            const float safeMultiplier = std::max(rule.derivedRadiusMultiplier, 0.0f);
            return localRadius * maxScale * safeMultiplier;
        }

        // The effective suppression radius for ONE suppressing point, per `rule`'s mode -- see
        // PcgEcosystemExclusion.h's own "Two ways to specify how far" header comment.
        float ComputeEffectiveRadius(const PcgPoint& suppressingPoint, const PcgEcosystemExclusionRule& rule) {
            if (rule.useDerivedRadius) {
                return ComputeSuppressingPointRadius(suppressingPoint, rule);
            }
            return std::max(rule.exclusionRadius, 0.0f);
        }

    } // namespace

    std::vector<PcgPoint> ApplyEcosystemExclusion(const std::vector<PcgPoint>& suppressedLayerPoints, const std::vector<PcgPoint>& suppressingLayerPoints, const PcgEcosystemExclusionRule& rule) {
        // Nothing CAN be suppressed with an empty suppressed or suppressing set -- return a plain
        // copy of the input (ApplyEcosystemExclusion's own contract is "return a new vector, never
        // mutate the input", matching every Phase 3 filter's convention, so even this early-out
        // still returns a genuinely new vector rather than aliasing the caller's own).
        if (suppressedLayerPoints.empty() || suppressingLayerPoints.empty()) {
            return suppressedLayerPoints;
        }

        // Determine the maximum possible effective radius across every suppressing point, to size
        // the acceleration grid's cells at 2x that value -- see PcgSpatialHashGrid's own
        // correctness invariant, documented in PcgSelfPruningFilter.h's header comment
        // ("Spatial acceleration: uniform hash grid"): a cell size of 2x the worst-case query
        // radius guarantees a suppressing point within any individual pair's own effective radius
        // of a given suppressed point can only ever land in that suppressed point's own cell or one
        // of its 26 face/edge/corner-adjacent neighbor cells, so QueryNeighborCells' fixed 3x3x3
        // search below always finds every suppressing point close enough to matter.
        float maxPossibleRadius = 0.0f;
        for (const PcgPoint& suppressingPoint : suppressingLayerPoints) {
            maxPossibleRadius = std::max(maxPossibleRadius, ComputeEffectiveRadius(suppressingPoint, rule));
        }

        if (maxPossibleRadius <= 0.0f) {
            // No suppressing point carries a positive effective radius at all (flat radius left
            // at/below its 0.0 default, or derived mode with every suppressing point's own bounds
            // degenerate) -- nothing can possibly be suppressed; return a plain copy rather than
            // paying for a grid build/query pass that would find nothing.
            return suppressedLayerPoints;
        }

        const float cellSize = std::max(2.0f * maxPossibleRadius, kMinCellSize);
        PcgSpatialHashGrid grid(cellSize);
        for (uint32_t suppressingIndex = 0; suppressingIndex < suppressingLayerPoints.size(); ++suppressingIndex) {
            grid.Insert(suppressingLayerPoints[suppressingIndex].position, suppressingIndex);
        }

        std::vector<PcgPoint> survivors;
        survivors.reserve(suppressedLayerPoints.size());

        // Reused across every QueryNeighborCells call below (cleared, not reallocated, each
        // iteration) -- same single-scratch-vector convention PruneByDistance's own sweep uses
        // (PcgSelfPruningFilter.cpp), to avoid one heap allocation per suppressed-layer point.
        std::vector<uint32_t> candidateScratch;

        // Strict input-order sweep -- every suppressed-layer point is tested independently against
        // the (fixed, fully-built-up-front) suppressing grid; unlike PruneByDistance, there is no
        // "already-kept survivors so far" dependency here (this is a two-set test, not a
        // self-pruning one), so the loop body's outcome for point i never depends on any other
        // suppressed-layer point's own outcome -- a purely per-point, order-STABLE (never
        // order-DEPENDENT) filter, matching this file's own "Determinism" header comment.
        for (const PcgPoint& candidate : suppressedLayerPoints) {
            candidateScratch.clear();
            grid.QueryNeighborCells(candidate.position, candidateScratch);

            bool suppressed = false;
            for (uint32_t suppressingIndex : candidateScratch) {
                const PcgPoint& suppressingPoint = suppressingLayerPoints[suppressingIndex];
                const float effectiveRadius = ComputeEffectiveRadius(suppressingPoint, rule);
                if (effectiveRadius <= 0.0f) {
                    continue;
                }
                // Inclusive (`<=`) comparison against the radius -- see this file's own header
                // comment ("Boundary convention: exactly-at-the-radius counts as SUPPRESSED") for
                // why this is the deliberately OPPOSITE convention to PruneByDistance's own `<`.
                if (DistanceSquared(candidate.position, suppressingPoint.position) <= effectiveRadius * effectiveRadius) {
                    suppressed = true;
                    break;
                }
            }

            if (!suppressed) {
                survivors.push_back(candidate);
            }
        }

        return survivors;
    }

    std::vector<PcgBiomeLayerResult> ApplyEcosystemRules(std::vector<PcgBiomeLayerResult> biomeResults, const std::vector<PcgEcosystemExclusionRule>& rules) {
        // Finds a MUTABLE pointer to the PcgBiomeLayerResult named `layerName` within
        // `biomeResults`'s CURRENT state, or nullptr if no such layer exists right now. Pointers
        // returned by this lambda stay valid across the whole loop below: `biomeResults` itself is
        // never resized (no push_back/erase) after this function receives it, only individual
        // `.points` members are reassigned in place, so no reallocation can ever invalidate a
        // pointer obtained here.
        const auto findLayer = [&biomeResults](const std::string& layerName) -> PcgBiomeLayerResult* {
            for (PcgBiomeLayerResult& result : biomeResults) {
                if (result.layerName == layerName) {
                    return &result;
                }
            }
            return nullptr;
        };

        // Applied strictly in `rules`' own list order -- see this file's own header comment
        // ("Rules compose when they target the SAME suppressed layer") for the full composition
        // contract this loop implements.
        for (const PcgEcosystemExclusionRule& rule : rules) {
            PcgBiomeLayerResult* suppressingLayer = findLayer(rule.suppressingLayerName);
            if (suppressingLayer == nullptr) {
                LOG_WARNING(std::format(
                    "[PcgEcosystemExclusion] ApplyEcosystemRules(): rule's suppressingLayerName '{}' "
                    "does not match any current layer -- rule skipped, not fatal.",
                    rule.suppressingLayerName));
                continue;
            }

            PcgBiomeLayerResult* suppressedLayer = findLayer(rule.suppressedLayerName);
            if (suppressedLayer == nullptr) {
                LOG_WARNING(std::format(
                    "[PcgEcosystemExclusion] ApplyEcosystemRules(): rule's suppressedLayerName '{}' "
                    "does not match any current layer -- rule skipped, not fatal.",
                    rule.suppressedLayerName));
                continue;
            }

            // Snapshot the suppressing layer's CURRENT points BEFORE reassigning suppressedLayer's
            // own points below, in case `suppressingLayer` and `suppressedLayer` alias the SAME
            // PcgBiomeLayerResult entry (a rule degenerately naming the same layer as both its own
            // suppressing and suppressed layer). ApplyEcosystemExclusion itself never mutates
            // either of its inputs, but without this snapshot the assignment to
            // `suppressedLayer->points` below would, in the aliased case, invalidate the very
            // reference ApplyEcosystemExclusion's `suppressingLayerPoints` parameter was bound to
            // mid-call; taking a copy up front sidesteps that entirely.
            const std::vector<PcgPoint> suppressingPointsSnapshot = suppressingLayer->points;
            suppressedLayer->points = ApplyEcosystemExclusion(suppressedLayer->points, suppressingPointsSnapshot, rule);
        }

        return biomeResults;
    }

}
