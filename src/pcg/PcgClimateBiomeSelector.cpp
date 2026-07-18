// PCG framework roadmap, Phase 8.3 ("Climate-Driven Biome Selection") -- implementation. See
// PcgClimateBiomeSelector.h for the full design rationale (ground-truth climate-data investigation,
// weighting/falloff model, return-shape choice). This file is pure composition/orchestration over
// Phase 8.1's own pcg::ApplyBiome (PcgBiomeRules.h) -- it implements exactly one small piece of NEW
// logic (the per-axis soft-falloff fit factor, ComputeAxisFit below; no existing primitive performs
// this "distance past a [min,max] range, converted to a [0,1] suitability score" computation) and
// otherwise only sequences existing calls plus a plain highest-weight scan.

#include "pcg/PcgClimateBiomeSelector.h"

#include "renderer/passes/AtmosClimatePass.h"

#include <algorithm>
#include <cmath>

namespace pcg {

    namespace {

        // Per-axis "fit factor" in [0,1] -- see PcgClimateBiomeSelector.h's own "Weighting model"
        // header comment for the full rationale. Exactly 1.0 anywhere inside [minValue,maxValue]
        // (inclusive of both edges), then a linear ramp down to exactly 0.0 by the time `value` is
        // `margin` past whichever edge it is on the wrong side of. `margin` is derived from the
        // axis' own [minValue,maxValue] range width (kClimateFalloffMarginFraction of it), floored at
        // kClimateFalloffMinMargin so a degenerate zero-width (or very narrow) authored range never
        // divides by zero and always has SOME non-zero falloff band, however thin.
        float ComputeAxisFit(float value, float minValue, float maxValue) {
            if (value >= minValue && value <= maxValue) {
                return 1.0f;
            }

            const float rangeWidth = std::max(maxValue - minValue, 0.0f);
            const float margin = std::max(rangeWidth * kClimateFalloffMarginFraction, kClimateFalloffMinMargin);

            const float distancePastEdge = (value < minValue) ? (minValue - value) : (value - maxValue);
            if (distancePastEdge >= margin) {
                return 0.0f;
            }

            return 1.0f - (distancePastEdge / margin);
        }

    } // namespace

    float ComputeBiomeWeight(const PcgClimateSample& climate, const PcgClimateBiomeWeight& rule) {
        const float temperatureFit = ComputeAxisFit(climate.temperature, rule.minTemperature, rule.maxTemperature);
        const float moistureFit = ComputeAxisFit(climate.moisture, rule.minMoisture, rule.maxMoisture);
        return rule.baseWeight * temperatureFit * moistureFit;
    }

    std::vector<PcgBiomeLayerResult> SelectAndApplyClimateBiome(
        const std::vector<PcgPoint>& basePoints,
        const PcgClimateSample& climate,
        const std::vector<std::pair<PcgBiomeRuleSet, PcgClimateBiomeWeight>>& candidateBiomes,
        uint32_t seed,
        std::string* outSelectedBiomeName) {

        // Highest-weight scan, ties broken by input order: `bestWeight` starts at 0.0f (not
        // -infinity) so that a candidate must clear a STRICTLY POSITIVE weight to ever become the
        // best -- this is exactly what makes "every candidate scored exactly 0.0f" fall out of this
        // same loop as bestIndex staying -1, rather than needing a separate post-loop check. Using
        // strictly-greater-than (never >=) for the update means the FIRST candidate to reach a given
        // weight value keeps that spot; a later candidate with the identical weight never displaces
        // it -- the documented "ties broken by input order" contract.
        float bestWeight = 0.0f;
        int64_t bestIndex = -1;
        for (size_t i = 0; i < candidateBiomes.size(); ++i) {
            const float weight = ComputeBiomeWeight(climate, candidateBiomes[i].second);
            if (weight > bestWeight) {
                bestWeight = weight;
                bestIndex = static_cast<int64_t>(i);
            }
        }

        if (bestIndex < 0) {
            // No viable biome for this climate (or `candidateBiomes` was empty) -- see this
            // function's own header comment for why an arbitrary fallback pick is deliberately NOT
            // made here.
            if (outSelectedBiomeName != nullptr) {
                *outSelectedBiomeName = std::string{};
            }
            return {};
        }

        const PcgBiomeRuleSet& winningRuleSet = candidateBiomes[static_cast<size_t>(bestIndex)].first;
        if (outSelectedBiomeName != nullptr) {
            *outSelectedBiomeName = winningRuleSet.biomeName;
        }

        // Same seed passed straight through, unmodified -- see this function's own header comment
        // for why (selection itself consumes no randomness).
        return ApplyBiome(basePoints, winningRuleSet, seed);
    }

    PcgClimateSample SampleCurrentClimate(const renderer::AtmosClimatePass& atmosClimate) {
        PcgClimateSample sample;
        sample.temperature = atmosClimate.GetEffectiveTemperatureCelsius();
        sample.moisture = atmosClimate.GetEffectiveRelativeHumidity();
        return sample;
    }

}
