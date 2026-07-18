#pragma once

// PCG framework roadmap, Phase 8.2 ("Ecosystem Exclusion" -- second subtask of Phase 8,
// Biome/Ecosystem): UE5.8 PCG Ecosystem Rules parity -- CROSS-layer suppression on top of Phase
// 8.1's PcgBiomeRuleSet (PcgBiomeRules.h), whose own header comment explicitly scopes this exact
// feature ("no bushes directly under tree canopies") OUT of that phase and INTO this one. Phase 8.1
// models every biome layer as an INDEPENDENT filter over the same shared base point set -- a
// layer's own `exclusionVolumes` only ever suppress points WITHIN THAT SAME LAYER (see
// PcgBiomeRules.h's own "Layer independence & composition model" header comment). This file adds
// the missing piece: an explicit, opt-in, ORDERED list of rules, applied strictly AFTER ApplyBiome
// has already produced its independent per-layer results, each one removing points from one named
// layer's output ("suppressed") that fall too close to another named layer's own surviving points
// ("suppressing") -- e.g. "no Bushes within a Tree's own canopy radius".
//
// --- Two ways to specify "how far" ---------------------------------------------------------------
// PcgEcosystemExclusionRule::exclusionRadius is a flat, uniform radius (same for every suppressing
// point) -- the simple case ("no bushes within 3m of any tree"). Setting `useDerivedRadius = true`
// instead derives a PER-SUPPRESSING-POINT radius from THAT point's own local-space bounds
// (boundsMin/boundsMax, PcgPointData.h) scaled by its own transform's largest scale component --
// i.e. a big tree's own authored/scattered bounding radius (its canopy) suppresses a wider area
// than a small tree's, without the caller needing to pre-compute anything per point. In derived
// mode `exclusionRadius` itself is ignored entirely (0/unused is the documented "disabled" value
// for the flat radius when derived mode is active); `derivedRadiusMultiplier` (default 1.0, a true
// no-op scale) lets a caller grow/shrink the derived footprint uniformly (e.g. 1.5x a tree's own
// mesh bounds to account for canopy overhang the authored bounds don't fully capture).
//
// --- Which point's bounds, exactly -----------------------------------------------------------
// The derived radius is ALWAYS computed from the SUPPRESSING point (e.g. the tree), never the
// suppressed point (e.g. the bush) -- this models "how far does THIS tree's canopy reach", not any
// property of what it might be suppressing. This mirrors PcgSelfPruningFilter.h's own
// ComputePointBoundingRadius helper (PcgSelfPruningFilter.cpp, file-private) for HOW a single
// point's radius is computed (local AABB half-diagonal x largest abs scale component -- see that
// file's own comment for why the largest scale component is a safe, conservative choice for
// non-uniform scale), but this file does NOT sum two radii the way PruneByDistance's
// ScaledByBounds mode does (that models two SAME-layer objects mutually avoiding each other's own
// extent) -- ecosystem exclusion is a one-directional relationship (the suppressing point's
// footprint reaches out; the suppressed point's own size plays no part in how far that footprint
// extends), so this file reimplements the small radius helper locally (PcgEcosystemExclusion.cpp)
// rather than depending on PcgSelfPruningFilter.cpp's own anonymous-namespace-private version.
//
// --- Boundary convention: exactly-at-the-radius counts as SUPPRESSED --------------------------
// A suppressed-layer point at EXACTLY the effective exclusion radius from a suppressing point is
// treated as INSIDE the exclusion footprint and IS removed (a `<=` comparison against radius, not
// `<`) -- this models a filled-disk canopy footprint (its own boundary belongs to the disk), the
// natural reading of "falls within radius R of a tree". This is a deliberately OPPOSITE convention
// to PruneByDistance's own `<` comparison (PcgSelfPruningFilter.h/.cpp): there, `minDistance` is
// the smallest ALLOWED separation between two same-layer survivors, so a pair sitting exactly at
// that distance is specifically NOT too close -- a different semantic question ("is this pair far
// enough apart") from this file's own ("is this point inside that footprint").
//
// --- Spatial acceleration: reuses Phase 3.2's PcgSpatialHashGrid, unmodified ------------------
// This is the same spatial-proximity-rejection problem PruneByDistance (Phase 3.2,
// PcgSelfPruningFilter.h/.cpp) already solves efficiently via a uniform hash grid -- except
// PruneByDistance prunes ONE point set against ITSELF (a candidate is tested against
// already-accepted survivors drawn from that SAME set), whereas this file needs a TWO-SET query
// (every suppressed-layer point tested against a DIFFERENT set's points, the suppressing layer).
// Rather than reimplementing spatial hashing, this file reuses PcgSpatialHashGrid directly: it is
// already a public, standalone, generically-useful type (PcgSelfPruningFilter.h's own header
// comment documents it as exposed specifically for reuse by "a later PCG phase" wanting a "bucket
// points into cells, query a cell's own 3x3x3 neighborhood" primitive) whose existing API
// (`Insert`, `QueryNeighborCells`) already supports the "build a grid from one set, query it with
// positions from another set entirely" pattern with ZERO changes needed to
// PcgSelfPruningFilter.h/.cpp -- ApplyEcosystemExclusion (PcgEcosystemExclusion.cpp) builds the
// grid from `suppressingLayerPoints`, then queries it once per `suppressedLayerPoints` entry.
//
// --- Determinism ---------------------------------------------------------------------------------
// This is a pure geometric proximity test -- no RNG, no seed parameter anywhere in this file.
// Identical inputs (same points, same order, same rule(s)) always produce a byte-identical, order-
// stable output: ApplyEcosystemExclusion processes `suppressedLayerPoints` in strict input order
// and never reorders survivors (only ever removes elements), the same "filter, never reorder"
// contract every existing Phase 3 filter (FilterByDensity, FilterByHeight, PruneByDistance, ...)
// and Phase 8.1's own ApplyBiomeLayer already guarantee.

#include <string>
#include <vector>

#include "pcg/PcgBiomeRules.h"
#include "pcg/PcgPointData.h"

namespace pcg {

    // One cross-layer suppression rule: named layer `suppressingLayerName`'s own surviving points
    // (e.g. "Trees") suppress points from named layer `suppressedLayerName` (e.g. "Bushes") that
    // fall within its effective radius -- see this file's own header comment for the flat-vs-
    // derived radius choice and the exactly-at-the-radius boundary convention.
    struct PcgEcosystemExclusionRule {
        std::string suppressingLayerName;
        std::string suppressedLayerName;

        // Flat, uniform suppression radius (world-space units), used when `useDerivedRadius` is
        // false (the default). IGNORED entirely when `useDerivedRadius` is true -- leave this at
        // its documented-unused 0.0 default in that mode rather than setting a value that would
        // never be read, to avoid a misleading-looking rule at authoring time.
        float exclusionRadius = 0.0f;

        // When true, each suppressing point's OWN effective radius is derived from its local-space
        // bounds (boundsMin/boundsMax) scaled by its own transform's largest scale component,
        // instead of using the flat `exclusionRadius` above -- see this file's own header comment
        // ("Which point's bounds, exactly") for the exact formula and rationale. False (the
        // default) uses the flat `exclusionRadius` for every suppressing point uniformly.
        bool useDerivedRadius = false;

        // Uniform multiplier applied to every derived per-point radius (see `useDerivedRadius`
        // above); unused when `useDerivedRadius` is false. Default 1.0 is a true no-op scale.
        float derivedRadiusMultiplier = 1.0f;
    };

    // Removes points from `suppressedLayerPoints` that fall within `rule`'s effective radius (see
    // this file's own header comment) of ANY point in `suppressingLayerPoints`, returning a NEW
    // vector (does not mutate either input) containing only the surviving suppressed-layer points,
    // in their original relative order. `rule.suppressingLayerName`/`suppressedLayerName` are not
    // consulted here (this function operates purely on the two point-vector arguments already
    // handed to it) -- name-matching against a PcgBiomeLayerResult list is ApplyEcosystemRules'
    // job, below; this lower-level overload exists for callers that already have two concrete point
    // sets in hand and don't want to go through the named-layer-result machinery at all.
    std::vector<PcgPoint> ApplyEcosystemExclusion(const std::vector<PcgPoint>& suppressedLayerPoints, const std::vector<PcgPoint>& suppressingLayerPoints, const PcgEcosystemExclusionRule& rule);

    // Takes `biomeResults` -- typically Phase 8.1's ApplyBiome(...) output directly -- and applies
    // every rule in `rules`, IN LIST ORDER, by name-matching each rule's suppressingLayerName/
    // suppressedLayerName against `biomeResults`' own PcgBiomeLayerResult::layerName entries.
    // Returns a NEW vector (taken by value/moved through internally, `biomeResults` itself is
    // consumed -- callers wanting to keep the pre-exclusion result should copy it first) with the
    // same layer names/order as the input, but with suppressed layers' point sets thinned according
    // to whichever rules targeted them.
    //
    // Rules compose when they target the SAME suppressed layer: they are applied strictly in
    // `rules`' own list order, and each rule's surviving output becomes the NEXT rule's starting
    // point for that same layer (e.g. two separate rules, "no Bushes under Trees" then "no Bushes
    // under Rocks", both narrow the SAME "Bushes" entry, cumulatively). A rule's OWN
    // suppressing-layer point set is always read from `biomeResults`' CURRENT state at the moment
    // that rule runs -- i.e. if an earlier rule in the list already thinned the suppressing layer
    // itself (because it was ALSO some OTHER rule's suppressed layer), a later rule using it as ITS
    // suppressing layer sees the already-thinned set, not the original ApplyBiome output. This is a
    // deliberate, simple "rules run as a single ordered pipeline over one shared, mutating-as-it-
    // goes result table" model -- the same left-to-right, order-matters composability every other
    // ordered-list primitive in this PCG framework already uses (biome layers' own per-index seed
    // derivation, exclusion-volume lists within one layer, ...).
    //
    // A rule naming a `suppressingLayerName` or `suppressedLayerName` that does not match any
    // `PcgBiomeLayerResult::layerName` in the CURRENT result table is logged as a warning (Debug-
    // only, via core/Logger.h's own no-op-in-Release LOG_WARNING -- CLAUDE.md's build-time debug-
    // tooling exclusion) and SKIPPED entirely (that one rule contributes nothing; every other rule
    // in the list still runs normally) -- never a crash/throw, since an authored biome/ecosystem
    // rule asset referencing a since-renamed/removed layer is a plausible, recoverable authoring
    // mistake, not the kind of programmer error this codebase's "crash explicit on Vulkan/VkResult
    // failures" policy (CLAUDE.md) is about.
    std::vector<PcgBiomeLayerResult> ApplyEcosystemRules(std::vector<PcgBiomeLayerResult> biomeResults, const std::vector<PcgEcosystemExclusionRule>& rules);

}
