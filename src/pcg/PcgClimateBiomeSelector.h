#pragma once

// PCG framework roadmap, Phase 8.3 ("Climate-Driven Biome Selection" -- third subtask of Phase 8,
// Biome/Ecosystem, sequential after Phase 8.1's "Biome Rule Asset", PcgBiomeRules.h/.cpp, ALREADY
// MERGED; unrelated to the concurrently-developed Phase 6.2 Cell Generator / Phase 8.2 Ecosystem
// Exclusion work). UE5.8 PCG climate integration parity: rather than a fixed, hand-picked
// PcgBiomeRuleSet, an author supplies SEVERAL candidate biomes, each tagged with the
// [temperature, moisture] climate range it is eligible in (UE5.8's own PCG Biome Core climate-rule
// concept: "this biome variant grows in warm/humid areas", "this one in cold/dry areas", ...). This
// file's entire job is the climate -> biome SELECTION/WEIGHTING layer -- it implements NO new
// point-set sampling/filtering algorithm of its own; once a biome is selected, the actual point-set
// work is a single delegated call into Phase 8.1's own pcg::ApplyBiome (PcgBiomeRules.h).
//
// --- What climate data is actually CPU-readable in this engine (ground-truth investigation) --------
// This project's climate/weather simulation is renderer::AtmosClimatePass (src/renderer/passes/
// AtmosClimatePass.h/.cpp), backed by CPU-authored baseline knobs in config::atmos::* (src/core/
// EngineConfig.h). Of everything AtmosClimatePass tracks, exactly two scalars map onto UE5.8 PCG's
// "Temperature"/"Moisture" biome-climate axes and are genuinely CPU-readable:
//   - Temperature (Celsius): AtmosClimatePass::GetEffectiveTemperatureCelsius() -- already existed
//     before this phase (added for the Volumetric ImGui tab / precipitation rain-vs-snow
//     selection). Returns the Dynamic-Weather-Simulation-smoothed/seasonally-offset value when
//     config::atmos::DYNAMIC_WEATHER_ENABLED is true, or the literal config::atmos::TEMPERATURE_CELSIUS
//     slider otherwise.
//   - Moisture (mapped 1:1 to this engine's "relative humidity", a [0,1] fraction -- UE5.8 PCG has
//     no separate "moisture" concept beyond ambient humidity either): AtmosClimatePass already
//     tracked an equivalent internally-smoothed value (m_CurrentRelativeHumidity) but, UNLIKE
//     temperature, had NO public getter for it before this phase -- only the raw
//     config::atmos::RELATIVE_HUMIDITY baseline was reachable from outside the class. This phase
//     adds AtmosClimatePass::GetEffectiveRelativeHumidity(), a minimal, additive one-line getter
//     mirroring GetEffectiveTemperatureCelsius()'s own exact ternary pattern byte-for-byte (see
//     AtmosClimatePass.h's own comment on that new getter) -- the ONLY production-code change this
//     phase makes outside of src/pcg/ and tests/.
// UE5.8-conceptually-relevant fields that were CONSIDERED and deliberately left OUT of
// PcgClimateSample below (documented here rather than silently omitted):
//   - "Season": UE5.8's own PCG framework has no first-class season axis either, but -- contrary to
//     what a naive port might assume -- this engine's Dynamic Weather Simulation DOES already track
//     one CPU-readable value (AtmosClimatePass::GetSeasonPhase01(), [0,1), 0/1=winter, 0.5=summer).
//     It is deliberately NOT folded into this phase's 2-axis temperature/moisture model, to stay in
//     scope with PcgClimateBiomeWeight's own [minTemperature,maxTemperature] x
//     [minMoisture,maxMoisture] rectangle contract (matching UE5.8 PCG's own primary Biome
//     climate-rule shape, which this phase's own task description asked for). A future phase wanting
//     season-gated biomes (e.g. "only in summer") can extend PcgClimateBiomeWeight with a third
//     [minSeason,maxSeason] axis and PcgClimateSample with a `seasonPhase01` field, reusing
//     ComputeAxisFit's exact pattern (see the .cpp) -- no AtmosClimatePass change would even be
//     needed, since the accessor already exists.
//   - Wind speed/direction, cloud/fog/rain density targets, surface wetness/snow coverage: all real,
//     CPU-readable AtmosClimatePass state, but none of them map onto a UE5.8 PCG "biome climate
//     range" concept the way temperature/moisture do -- they are precipitation/VFX-driving signals,
//     not biome-selection axes, so folding them in here would be scope creep, not fidelity.
//
// --- Weighting model: soft suitability falloff (a deliberate choice between the two this phase's
// own task scope documents as acceptable) -------------------------------------------------------
// ComputeBiomeWeight below returns `rule.baseWeight` scaled by a per-axis "fit factor" in [0,1]:
// exactly 1.0 anywhere inside [minTemperature,maxTemperature] (respectively [minMoisture,maxMoisture],
// inclusive of both edges), then a LINEAR ramp down to exactly 0.0 over a margin equal to
// kClimateFalloffMarginFraction of that axis' own configured range width (floored at
// kClimateFalloffMinMargin so a zero-width/degenerate range never divides by zero) -- UE5.8's own PCG
// "climate suitability curve" behavior: a biome does not hard-cut at its authored range edge, it fades
// out just past it. Preferred here over a hard 0-outside-the-range cutoff per this phase's own task
// guidance ("a soft falloff is more true to UE5.8's typical... behavior and is preferred if it's not
// much more work" -- it wasn't). A hard cutoff is a strictly simpler special case of this (shrink the
// margin toward 0), so nothing is lost by picking the soft version. The two axes combine
// multiplicatively (temperatureFit * moistureFit), matching how a rectangle-shaped 2D eligibility
// region naturally composes from two independent 1D range tests.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pcg/PcgBiomeRules.h"
#include "pcg/PcgPointData.h"

// Forward declaration only -- SampleCurrentClimate below takes a `const renderer::AtmosClimatePass&`,
// which a reference parameter can be declared against without the class' full definition (Vulkan/VMA
// headers, see AtmosClimatePass.h's own #includes). This keeps every OTHER declaration in this file --
// PcgClimateSample/PcgClimateBiomeWeight/ComputeBiomeWeight/SelectAndApplyClimateBiome, the pure
// climate x biome selection logic most callers actually want -- entirely free of any Vulkan/renderer::
// dependency, matching every other src/pcg/*.h file's own "pure CPU logic, no renderer:: coupling"
// convention (the two documented exceptions in this codebase, PcgGpuDensityNoiseNode.h and
// PcgInstanceSpawnManager.h, are both explicitly about DISPATCHING GPU compute work -- a fundamentally
// different concern from this file's climate -> biome-selection orchestration). Only
// PcgClimateBiomeSelector.cpp's own SampleCurrentClimate definition needs AtmosClimatePass.h's full
// declaration, via its own #include.
namespace renderer { class AtmosClimatePass; }

namespace pcg {

    // Fraction of a PcgClimateBiomeWeight rule's own configured axis range width used as the soft
    // falloff margin (see this file's own "Weighting model" header comment). 0.25 means a value just
    // past the authored range edge still carries meaningful weight, fading linearly to 0 by the time
    // it is a further 25% of the range's own width away -- a gentle, UE5.8-style suitability taper,
    // not an abrupt cliff.
    inline constexpr float kClimateFalloffMarginFraction = 0.25f;

    // Absolute floor on the falloff margin (same units as the axis: Celsius for temperature, a [0,1]
    // fraction for moisture) -- guards a degenerate zero-width (or very narrow) authored range from
    // producing a zero-width margin, which would otherwise divide by zero in the falloff ramp and/or
    // make the rule's eligibility an infinitesimally thin, unreachable-in-practice spike.
    inline constexpr float kClimateFalloffMinMargin = 0.01f;

    // CPU-readable climate state this phase's biome selection reads -- see this file's own header
    // comment for exactly which AtmosClimatePass/config::atmos fields these mirror, and which
    // UE5.8-conceptually-relevant fields (season, ...) were deliberately left out.
    struct PcgClimateSample {
        // Degrees Celsius. Mirrors AtmosClimatePass::GetEffectiveTemperatureCelsius() /
        // config::atmos::TEMPERATURE_CELSIUS's own units exactly.
        float temperature = 20.0f;

        // [0,1] fraction. Mirrors AtmosClimatePass::GetEffectiveRelativeHumidity() (added by this
        // phase) / config::atmos::RELATIVE_HUMIDITY's own units exactly -- see this file's header
        // comment for why "moisture" and "relative humidity" are treated as the same concept here.
        float moisture = 0.5f;
    };

    // One declarative "this biome is eligible/weighted in this climate range" rule -- references a
    // PcgBiomeRuleSet by NAME only (does not own or embed one; see SelectAndApplyClimateBiome's own
    // comment for how a rule is actually paired with its concrete PcgBiomeRuleSet at the call site).
    struct PcgClimateBiomeWeight {
        std::string biomeName;

        // Inclusive eligibility range on the temperature axis, degrees Celsius, before the soft
        // falloff margin (see this file's "Weighting model" header comment) extends it slightly
        // further before reaching zero weight. Authoring minTemperature > maxTemperature is invalid
        // (undefined which end is which) -- callers must keep min <= max, same convention every
        // other Pcg*Filter min/max-range field in this codebase already assumes (e.g.
        // PcgBiomeLayerRule::minDensity/maxDensity).
        float minTemperature = -1000.0f;
        float maxTemperature = 1000.0f;

        // Inclusive eligibility range on the moisture axis, [0,1] fraction, same falloff/ordering
        // convention as temperature above.
        float minMoisture = 0.0f;
        float maxMoisture = 1.0f;

        // Flat weight multiplier applied on top of the [0,1] per-axis fit factors (see
        // ComputeBiomeWeight) -- lets an author bias one biome over another even when both are
        // equally climate-eligible (e.g. two biomes both fully eligible at [20C, 0.5 moisture], one
        // preferred 2x over the other). Expected non-negative; a negative baseWeight is not a
        // supported authoring case (ComputeBiomeWeight does not clamp it away).
        float baseWeight = 1.0f;
    };

    // Returns rule.baseWeight scaled by how well `climate` fits inside
    // [rule.minTemperature,rule.maxTemperature] x [rule.minMoisture,rule.maxMoisture] -- see this
    // file's own "Weighting model" header comment for the exact soft-falloff formula. Returns exactly
    // 0.0f (never a tiny non-zero residual) once EITHER axis' sample value is kClimateFalloffMarginFraction
    // (of that axis' own range width, floored at kClimateFalloffMinMargin) or further past that axis'
    // own [min,max] edge -- this exact "reaches a hard, provable zero" property is what lets
    // SelectAndApplyClimateBiome's "no viable biome" case below be well-defined rather than a fuzzy
    // near-zero threshold judgment call.
    float ComputeBiomeWeight(const PcgClimateSample& climate, const PcgClimateBiomeWeight& rule);

    // Evaluates every (PcgBiomeRuleSet, PcgClimateBiomeWeight) candidate pair's weight via
    // ComputeBiomeWeight(climate, candidate.second), picks the single highest-weight candidate
    // (ties -- an EXACT equal-float weight between two or more candidates -- are broken by INPUT
    // ORDER: the first candidate in `candidateBiomes` to reach the current-best weight keeps that
    // spot; a later candidate must have a STRICTLY greater weight to displace it), and returns
    // pcg::ApplyBiome(basePoints, winningRuleSet, seed) -- the exact same seed passed straight
    // through unmodified (selection itself consumes no randomness, so there is nothing to derive a
    // distinct sub-seed from; ApplyBiome's own per-layer PcgHashCombine(seed, layerIndex) derivation,
    // see PcgBiomeRules.h, is unaffected either way).
    //
    // Return shape: the FULL std::vector<PcgBiomeLayerResult> the winning biome's own ApplyBiome call
    // produces (one entry per layer, each carrying its own layerName), NOT a single flattened
    // std::vector<PcgPoint> -- deliberately matching ApplyBiome's own native output shape rather than
    // discarding its per-layer structure. This is the more consistent choice with how ApplyBiome's
    // result is normally consumed downstream: every existing/expected consumer (a future weighted-
    // mesh-spawner wiring stage, per Phase 8.1's own "out of scope for now" note in PcgBiomeRules.h)
    // needs to know WHICH layer ("Grass" vs "Rocks" vs ...) a given point came from to pick the right
    // mesh set; flattening here would destroy that information one layer earlier than necessary, for
    // no benefit (a caller that genuinely just wants every point in one flat list can trivially
    // concatenate the layers itself; the reverse -- recovering per-layer structure after a premature
    // flatten -- is not possible).
    //
    // "No viable biome for this climate" case: if EVERY candidate's ComputeBiomeWeight is exactly
    // 0.0f (the climate sample fits none of them, even accounting for each rule's own soft falloff
    // margin), this function returns an EMPTY vector rather than arbitrarily picking one -- picking
    // an arbitrary "best of a bad lot" biome when nothing is actually climate-eligible would silently
    // scatter e.g. lush jungle vegetation into a climate no author ever intended it to appear in. An
    // empty `candidateBiomes` list hits this same empty-result path (there is trivially no viable
    // biome among zero candidates).
    //
    // `outSelectedBiomeName`, if non-null, is set to the winning candidate's OWN PcgBiomeRuleSet::
    // biomeName (the identity of the ruleSet actually passed to ApplyBiome -- not
    // PcgClimateBiomeWeight::biomeName, though callers are expected to keep the two in sync; this
    // function does not cross-validate them) on a successful selection, or to an empty string on the
    // "no viable biome" case above (left unmodified only when the pointer itself is null).
    std::vector<PcgBiomeLayerResult> SelectAndApplyClimateBiome(
        const std::vector<PcgPoint>& basePoints,
        const PcgClimateSample& climate,
        const std::vector<std::pair<PcgBiomeRuleSet, PcgClimateBiomeWeight>>& candidateBiomes,
        uint32_t seed,
        std::string* outSelectedBiomeName = nullptr);

    // Pulls the ACTUAL current climate state from a live renderer::AtmosClimatePass instance --
    // GetEffectiveTemperatureCelsius() and GetEffectiveRelativeHumidity() (see this file's own header
    // comment for the ground-truth investigation behind exactly these two fields), which is what
    // makes this phase's climate integration real rather than a synthetic-data-only exercise. Reads
    // only -- never calls Init()/RecordUpdate()/Shutdown() on `atmosClimate` (this function has no
    // opinion on that object's lifecycle; the caller owns it, typically
    // renderer::ClusterRenderPipeline::GetAtmosClimate()). Both underlying getters are pure,
    // side-effect-free reads of already-computed state (this frame's, or -- before the very first
    // RecordUpdate() call -- the static config::atmos::* baseline; see both getters' own comments),
    // so this function is itself pure and safe to call any number of times per frame.
    PcgClimateSample SampleCurrentClimate(const renderer::AtmosClimatePass& atmosClimate);

}
