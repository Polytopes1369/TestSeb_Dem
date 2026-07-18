// PCG framework roadmap, Phase 8.1 ("Biome Rule Asset") -- implementation. See PcgBiomeRules.h for
// the full design rationale (layer independence model, pipeline order, determinism). This file is
// pure composition/orchestration over the four existing Phase 3 filter files
// (PcgDensityTransformFilter.h, PcgSlopeHeightFilter.h, PcgBooleanSetOps.h,
// PcgSelfPruningFilter.h) -- it contains exactly one small piece of NEW logic
// (ApplyBaseDensityWeight, a flat per-point density multiply with no existing primitive to call
// into) and otherwise only sequences existing calls.

#include "pcg/PcgBiomeRules.h"

#include "pcg/PcgBooleanSetOps.h"
#include "pcg/PcgDensityTransformFilter.h"
#include "pcg/PcgSeededRandom.h"
#include "pcg/PcgSelfPruningFilter.h"
#include "pcg/PcgSlopeHeightFilter.h"

#include <algorithm>

namespace pcg {

    namespace {

        // Applies `baseDensity` as a flat per-point density MULTIPLIER -- NOT a range remap the way
        // RemapDensity performs (that remaps the batch's ACTUAL observed [min,max] density span into
        // a NEW target range, a different operation entirely), and not itself one of the existing
        // Phase 3 filter primitives: PcgDensityTransformFilter.h/.cpp provides no "scale every
        // point's density by a constant weight" function to call into, so this one small, clearly-
        // contained multiply is this file's own minimal composition glue, not a reimplementation of
        // any Phase 3 filter's internal algorithm. Mutates `points` in place (a fixed-size,
        // per-point-field perturbation with no element added/removed -- the same in-place rationale
        // ApplyTransformJitter/ApplyNoiseToDensity already use, see PcgDensityTransformFilter.h's own
        // header comment). Result is clamped to [0, 1], matching PcgPoint::density's own documented
        // valid range (PcgPointData.h) and ApplyNoiseToDensity's own clamping convention.
        void ApplyBaseDensityWeight(std::vector<PcgPoint>& points, float baseDensity) {
            for (PcgPoint& point : points) {
                point.density = std::clamp(point.density * baseDensity, 0.0f, 1.0f);
            }
        }

    } // namespace

    std::vector<PcgPoint> ApplyBiomeLayer(const std::vector<PcgPoint>& inputPoints, const PcgBiomeLayerRule& layer, uint32_t seed) {
        // Step 1: density weight. Copies `inputPoints` here (rather than mutating a caller-owned
        // vector) since ApplyBiomeLayer's own contract, like every Phase 3 filter function it
        // composes, is "return a new, filtered/modulated copy, never mutate the input" -- callers
        // (including ApplyBiome, applying this same layer rule to the SAME shared `basePoints` for
        // every other layer too) must be able to rely on `inputPoints` staying untouched.
        std::vector<PcgPoint> points = inputPoints;
        ApplyBaseDensityWeight(points, layer.baseDensity);

        // Step 2: density filter (FilterByDensity parity). With baseDensity left at its 1.0 default
        // and minDensity/maxDensity at their own [0,1] "no filtering" default, this call is a
        // pass-through (every point's density already lies within [0,1], so nothing is excluded);
        // when only minDensity/maxDensity are configured away from their defaults, this step alone
        // determines the output, making ApplyBiomeLayer's result identical to calling
        // FilterByDensity(inputPoints, layer.minDensity, layer.maxDensity) directly -- the
        // "single-layer biome with just a density threshold" equivalence this phase's test suite
        // (tests/PcgBiomeRulesTests.cpp) verifies.
        points = FilterByDensity(points, layer.minDensity, layer.maxDensity);

        // Step 3: height filter (FilterByHeight parity). Defaults to the full float range, so this
        // is a true no-op unless the author narrows minHeight/maxHeight on purpose.
        points = FilterByHeight(points, layer.minHeight, layer.maxHeight);

        // Step 4: slope filter (FilterBySlope parity, generic std::vector<PcgPoint> overload --
        // see PcgSlopeHeightFilter.h's own header comment on why this overload's slope is only an
        // APPROXIMATION derived from each point's rotation, not a real terrain re-query). Skipped
        // entirely (not even called) when the layer's slope filtering is disabled via the
        // kBiomeSlopeFilterDisabled sentinel, so a layer that never wants slope filtering pays
        // nothing for it.
        if (layer.maxSlopeRadians != kBiomeSlopeFilterDisabled) {
            points = FilterBySlope(points, layer.maxSlopeRadians);
        }

        // Step 5: exclusion volumes (DifferenceFromVolume parity, composed once per volume, in list
        // order). This layer is suppressed wherever ANY of layer.exclusionVolumes contains a
        // surviving point -- an empty list is a no-op (the loop simply never executes).
        for (const PcgVolumeData& volume : layer.exclusionVolumes) {
            points = DifferenceFromVolume(points, volume);
        }

        // Step 6: noise-driven density modulation (ApplyNoiseToDensity parity), only if requested.
        // Uses `seed` directly (no additional internal salt) -- this is the ONLY randomness-
        // consuming step in this entire per-layer pipeline (self-pruning below is a pure,
        // deterministic index-order rule with no RNG, see PcgSelfPruningFilter.h's own header
        // comment), so there is no other draw within this function it could collide with. See this
        // file's header comment ("Determinism") for the guidance a future SECOND seeded step here
        // should follow (salt via PcgHashCombine(seed, someDistinctConstant) instead of reusing
        // `seed` raw).
        if (layer.useNoiseVariation) {
            ApplyNoiseToDensity(points, layer.noiseFrequency, seed);
        }

        // Step 7: self-pruning (PruneByDistance parity, Uniform mode -- ScaledByBounds is not
        // exposed at the biome-layer level in this phase; a layer wanting per-point bounds-scaled
        // spacing can still call PruneByDistance directly with its own params after ApplyBiomeLayer
        // returns), only if minSpacing is a real positive distance.
        if (layer.minSpacing > 0.0f) {
            points = PruneByDistance(points, layer.minSpacing, PcgPruningMode::Uniform);
        }

        return points;
    }

    std::vector<PcgBiomeLayerResult> ApplyBiome(const std::vector<PcgPoint>& basePoints, const PcgBiomeRuleSet& ruleSet, uint32_t seed) {
        std::vector<PcgBiomeLayerResult> results;
        results.reserve(ruleSet.layers.size());

        for (size_t layerIndex = 0; layerIndex < ruleSet.layers.size(); ++layerIndex) {
            const PcgBiomeLayerRule& layer = ruleSet.layers[layerIndex];

            // Per-layer seed derivation (see PcgBiomeRules.h's own "Determinism" header comment):
            // combining the biome's overall seed with this layer's own 0-based index guarantees no
            // two layers ever share a random stream, and that a layer's derived seed depends ONLY on
            // its own index -- reordering/adding/removing OTHER layers never perturbs this one's
            // output.
            const uint32_t layerSeed = PcgHashCombine(seed, static_cast<uint32_t>(layerIndex));

            // Every layer independently filters the SAME shared `basePoints` input -- this file's
            // documented "independent layers, not a hierarchical stack" composition model (see
            // PcgBiomeRules.h's own header comment) -- rather than chaining one layer's filtered
            // output into the next.
            PcgBiomeLayerResult result;
            result.layerName = layer.layerName;
            result.points = ApplyBiomeLayer(basePoints, layer, layerSeed);
            results.push_back(std::move(result));
        }

        return results;
    }

}
