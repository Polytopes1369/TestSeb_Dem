#pragma once

// PCG framework roadmap, Phase 8.1 ("Biome Rule Asset" -- first/root subtask of Phase 8,
// Biome/Ecosystem): UE5.8 PCG Biome Core parity for the "biome" concept -- a named, ORDERED stack
// of declarative layer rules (e.g. "Grass", "Rocks", "Trees") that each independently filter/thin a
// shared base point set down to their own final point set, using ONLY the already-existing Phase 3
// filter primitives (PcgDensityTransformFilter.h, PcgSlopeHeightFilter.h, PcgBooleanSetOps.h,
// PcgSelfPruningFilter.h) as building blocks. This file is pure composition/orchestration -- it
// implements NO new low-level filtering algorithm; every actual point-set transformation below is a
// direct call into one of those four existing files.
//
// --- Layer independence & composition model (a deliberate design choice, documented here because
// UE5.8's own PCG Biome nodes support several composition modes and this phase picks ONE) ---------
// Every layer in a PcgBiomeRuleSet is evaluated against the SAME `basePoints` input (e.g. one dense
// scatter from a Surface/Terrain sampler) and produces its OWN independent output point set -- later
// layers do NOT consume earlier layers' filtered-down output, and a layer's own `exclusionVolumes`
// only suppress points WITHIN THAT SAME LAYER, never another layer's point set. This mirrors UE5.8's
// real, commonly-used PCG Biome authoring pattern: biome layers ("this is where grass grows", "this
// is where rocks spawn") are typically INDEPENDENT point sets that get separately weighted-mesh-
// spawned and composited together in the final render, not a strict "layer N is masked by whatever
// survived layer N-1" pipeline (that stacking IS expressible in UE5.8 PCG, but requires the author
// to explicitly wire one layer's output into the next -- it is not the default/implicit behavior of
// simply adding several layers to a biome). The simpler independent model is also what this phase's
// own task scope calls for: Phase 8.2 ("ecosystem exclusion", a LATER subtask building on this file)
// is explicitly where cross-layer suppression rules (e.g. "no grass under a tree's canopy") belong --
// baking any cross-layer interaction into Phase 8.1 would be scope creep on a subtask that has not
// started yet. Each layer's `exclusionVolumes` therefore models authored, PER-LAYER exclusion regions
// (e.g. "no rocks inside this authored no-spawn volume"), not layer-to-layer interaction.
//
// --- Determinism -------------------------------------------------------------------------------
// A biome's overall `seed` (ApplyBiome's own parameter) is combined with each layer's 0-based INDEX
// in `ruleSet.layers` via PcgHashCombine(seed, layerIndex) (see PcgSeededRandom.h) to derive that
// layer's OWN independent seed, passed to ApplyBiomeLayer. This guarantees:
//   - Two layers never share a random stream (no cross-layer correlation/collision), the same
//     per-stream-independence guarantee PcgHashCombine already provides for individual points
//     (see PcgSeededRandom.h's own header comment).
//   - REORDERING layers in `ruleSet.layers` does NOT perturb any OTHER layer's output: a layer's
//     seed depends only on (biome seed, that layer's own index), so appending, removing, or
//     reordering OTHER layers never changes an unrelated layer's derived seed unless that specific
//     layer's own index actually changes.
//   - Calling ApplyBiome twice with the identical `basePoints`/`ruleSet`/`seed` always reproduces a
//     byte-identical result (same per-layer point sets, same order) -- this codebase's hard
//     "same input -> byte-identical output, every run" requirement (see PcgSeededRandom.h's own
//     header comment for why that requirement exists project-wide).
// Within a single layer, ApplyBiomeLayer's own noise-driven density modulation step (the only
// randomness-consuming step in the whole per-layer pipeline -- self-pruning is a pure index-order
// rule with no RNG, see PcgSelfPruningFilter.h's own header comment) uses the layer's seed directly
// (equivalent to salting with a constant zero offset): there is currently only one seeded operation
// per layer, so no further internal salting is needed; a future addition of a SECOND
// randomness-consuming step to this file's per-layer pipeline should salt its own draw via
// PcgHashCombine(seed, someDistinctConstant) rather than reusing the same raw seed value, to avoid
// two steps silently sharing a stream.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

namespace pcg {

    // Sentinel value for PcgBiomeLayerRule::maxSlopeRadians meaning "slope filtering is disabled for
    // this layer" -- deliberately a NEGATIVE value (no real slope angle is ever negative) rather than
    // e.g. pi/2 (90 degrees), so "disabled" cannot be confused with "the author explicitly chose a
    // 90-degree cutoff" and ApplyBiomeLayer can skip the slope-filter step (and its underlying
    // rotation-derived slope approximation, see PcgSlopeHeightFilter.h's own header comment on why
    // that generic overload's slope is only approximate) entirely rather than paying for a filter
    // pass whose threshold happens to reject nothing.
    inline constexpr float kBiomeSlopeFilterDisabled = -1.0f;

    // One layer in a biome's ordered rule stack -- a purely DATA-DRIVEN description of which of the
    // existing Phase 3 filter primitives to run, and with what parameters, composed by
    // ApplyBiomeLayer below. Every field has a documented "no filtering" default so that a
    // freshly-default-constructed layer applied via ApplyBiomeLayer is a complete no-op (returns its
    // input unchanged, module point order) -- exactly the "a single-layer biome with just a density
    // threshold behaves identically to calling FilterByDensity directly" behavior this phase's own
    // test suite verifies (tests/PcgBiomeRulesTests.cpp), and the same "every field off is a real
    // no-op" convention PcgTransformJitterParams already established (PcgDensityTransformFilter.h).
    struct PcgBiomeLayerRule {
        // Purely descriptive -- carries no filtering behavior of its own. Echoed back verbatim into
        // this layer's PcgBiomeLayerResult::layerName by ApplyBiome, so a caller can tell which
        // layer a given result came from without needing to track index -> name mapping themselves.
        std::string layerName;

        // Starting density WEIGHT for this layer, applied as a flat per-point MULTIPLIER against
        // each point's existing `density` field (see ApplyBiomeLayer's own pipeline-order comment
        // for exactly where this runs and why it is NOT one of the Phase 3 filter primitives --
        // there is no existing "scale every point's density by a constant" function to call into).
        // Default 1.0 is a true no-op (every point's density is left byte-identical).
        float baseDensity = 1.0f;

        // Density-filter bounds (FilterByDensity parity, PcgDensityTransformFilter.h), applied
        // AFTER baseDensity's multiply. Defaults to [0, 1] -- PcgPoint::density's own full valid
        // range (PcgPointData.h) -- so with baseDensity left at its own 1.0 default, this step is a
        // true no-op (no point's density can ever fall outside [0, 1] to begin with).
        float minDensity = 0.0f;
        float maxDensity = 1.0f;

        // If true, this layer's per-point density is modulated by coherent value noise
        // (SampleAttributeNoise/ApplyNoiseToDensity, PcgDensityTransformFilter.h) sampled at each
        // point's own world-space position -- UE5.8's "noise-driven density variation for natural
        // clumping" pattern (e.g. grass reads sparser in some patches, denser in others, without an
        // artificial hard edge). False (the default) is a complete no-op: the noise sampler is never
        // even called.
        bool useNoiseVariation = false;
        float noiseFrequency = 1.0f;

        // Height-filter bounds (FilterByHeight parity, PcgSlopeHeightFilter.h) on world-space Y
        // (this codebase's up axis). Defaults span the full float range, so no point is ever
        // rejected by height unless the author narrows this on purpose (e.g. "no rocks below sea
        // level" -> minHeight = seaLevelY).
        float minHeight = -std::numeric_limits<float>::max();
        float maxHeight = std::numeric_limits<float>::max();

        // Slope-filter threshold (FilterBySlope parity, PcgSlopeHeightFilter.h's generic
        // std::vector<PcgPoint> overload -- an APPROXIMATE, rotation-derived slope, see that
        // header's own caveat) in radians; kBiomeSlopeFilterDisabled (the default) skips slope
        // filtering for this layer entirely. Set to e.g. maths::ToRadians(60.0f) for "no trees above
        // a 60-degree cliff face".
        float maxSlopeRadians = kBiomeSlopeFilterDisabled;

        // This layer is suppressed (points removed) wherever ANY of these volumes contains a point
        // -- e.g. "don't spawn grass under a building's footprint". Composed as a sequence of
        // DifferenceFromVolume calls (PcgBooleanSetOps.h), one per volume, in list order; an empty
        // list (the default) is a no-op -- the loop simply never executes.
        std::vector<PcgVolumeData> exclusionVolumes;

        // Minimum world-space separation this layer's surviving points must respect
        // (PruneByDistance/PcgPruningMode::Uniform parity, PcgSelfPruningFilter.h). 0.0 (the
        // default) disables self-pruning entirely for this layer (the filter is never invoked) --
        // any positive value thins this layer's own point set so no two of ITS OWN survivors sit
        // closer than `minSpacing` apart (pruning never looks at another layer's points, matching
        // this file's independent-layers composition model, see this file's own header comment).
        float minSpacing = 0.0f;
    };

    // A named, ordered stack of biome layer rules -- UE5.8 PCG Biome asset parity. `layers` are
    // evaluated in list order by ApplyBiome (order determines each layer's derived seed via its
    // index, see this file's own "Determinism" header comment -- and gives a stable, predictable
    // ordering to the returned PcgBiomeLayerResult list -- but, per this file's independent-layers
    // composition model, does NOT make a later layer depend on an earlier layer's filtered output).
    struct PcgBiomeRuleSet {
        std::string biomeName;
        std::vector<PcgBiomeLayerRule> layers;
    };

    // One layer's final, named point-set output -- what ApplyBiome returns one of, per layer.
    struct PcgBiomeLayerResult {
        std::string layerName;
        std::vector<PcgPoint> points;
    };

    // Applies ONE layer's full rule chain to `inputPoints`, returning a NEW filtered/modulated point
    // vector (does not mutate `inputPoints`). Pure composition over the existing Phase 3 filter
    // functions -- no filtering ALGORITHM is reimplemented here, only their ORDER and parameters are
    // chosen. Fixed, documented pipeline order (determinism depends on never reordering these steps):
    //   1. Density weight: every point's `density` *= layer.baseDensity (clamped to [0,1]) -- see
    //      PcgBiomeLayerRule::baseDensity's own comment for why this one small step is NOT a call
    //      into PcgDensityTransformFilter.h (no existing primitive performs a flat multiply).
    //   2. Density filter: FilterByDensity(points, layer.minDensity, layer.maxDensity).
    //   3. Height filter: FilterByHeight(points, layer.minHeight, layer.maxHeight).
    //   4. Slope filter: FilterBySlope(points, layer.maxSlopeRadians), ONLY IF
    //      layer.maxSlopeRadians != kBiomeSlopeFilterDisabled.
    //   5. Exclusion volumes: DifferenceFromVolume(points, volume) for each volume in
    //      layer.exclusionVolumes, in list order.
    //   6. Noise-driven density modulation: ApplyNoiseToDensity(points, layer.noiseFrequency, seed),
    //      ONLY IF layer.useNoiseVariation.
    //   7. Self-pruning: PruneByDistance(points, layer.minSpacing, PcgPruningMode::Uniform), ONLY IF
    //      layer.minSpacing > 0.
    // With every PcgBiomeLayerRule field left at its default EXCEPT minDensity/maxDensity, this
    // reduces exactly to FilterByDensity(inputPoints, layer.minDensity, layer.maxDensity) -- steps
    // 1 (baseDensity=1, a true no-op multiply) and 3-7 (each individually a documented no-op at its
    // own default) contribute nothing, leaving only step 2's call visible in the output.
    std::vector<PcgPoint> ApplyBiomeLayer(const std::vector<PcgPoint>& inputPoints, const PcgBiomeLayerRule& layer, uint32_t seed);

    // Applies every layer in `ruleSet` to the SAME `basePoints` input (see this file's own
    // "independence & composition model" header comment) and returns one PcgBiomeLayerResult per
    // layer, in `ruleSet.layers`' own order. Each layer's seed is derived as
    // PcgHashCombine(seed, static_cast<uint32_t>(layerIndex)) (see this file's own "Determinism"
    // header comment) before being forwarded to ApplyBiomeLayer -- callers wanting to invoke a
    // single layer's rule chain directly (bypassing the rest of the biome) can call ApplyBiomeLayer
    // themselves with that identical derivation to reproduce exactly what ApplyBiome would have
    // produced for that one layer.
    //
    // Wiring each layer's resulting point set into an actual weighted-mesh-spawner list (e.g. the
    // "Grass" layer's points spawning grass meshes, "Rocks" spawning rocks) is explicitly OUT OF
    // SCOPE for this Phase 8.1 file -- this function's contract ends at returning clean, independent
    // per-layer point sets; that spawner wiring is a later integration phase's job.
    std::vector<PcgBiomeLayerResult> ApplyBiome(const std::vector<PcgPoint>& basePoints, const PcgBiomeRuleSet& ruleSet, uint32_t seed);

}
