#pragma once

// PCG framework roadmap, Phase 4.1 ("Weighted Mesh Spawner"): UE5.8 PCG's (Static) Mesh Spawner
// node parity -- takes a std::vector<pcg::PcgPoint> (produced upstream by a Phase 2 sampler /
// Phase 3 filter chain) and, for EACH point, picks exactly one mesh from a caller-supplied weighted
// candidate list (e.g. "70% grass tuft A, 20% grass tuft B, 10% rock"), producing one resolved
// PcgSpawnRequest per surviving point. This is a PURE CPU/data-transform phase: it does not touch
// the GPU, does not stream any mesh, and does not itself feed renderer::PcgInstanceDrawPass -- that
// glue is explicitly Phase 4.2 (a later, separate step). PcgSpawnRequest's field shape below is
// deliberately a near-verbatim match of PcgInstanceDrawPass::AcquireInstance's own parameter list
// (src/renderer/passes/PcgInstanceDrawPass.h) specifically so Phase 4.2's future glue code is a
// trivial "for each request, call AcquireInstance(request.meshID, request.materialID,
// request.position, request.rotation, request.scale)" loop, with no field reordering/renaming
// required at that call site.
//
// --- Why this header carries NO dependency on pcg/PcgGraph.h (layering) -------------------------
// PcgSpawnRequest is also registered as a NEW PcgPinDataType/PcgPinData variant alternative
// (PcgPinDataType::SpawnRequests, see PcgGraph.h's own enum + variant) so this node type's output
// pin can be typed as "a list of spawn requests" rather than silently degrading to a generic
// AttributeSet. That means PcgGraph.h itself needs a COMPLETE definition of PcgSpawnRequest at the
// point it declares `using PcgPinData = std::variant<..., std::vector<PcgSpawnRequest>>` (a
// std::variant's alternatives must be complete types wherever the variant specialization itself is
// instantiated). If this header pulled in PcgGraph.h (e.g. to reach PCG_REGISTER_NODE_TYPE's own
// pin-schema types), PcgGraph.h including THIS header back would be a circular #include -- so this
// header stays a pure LEAF data-type header, exactly like PcgPointData.h/PcgAttributeSet.h/
// PcgSpatialData.h already are for PcgGraph.h's own existing variant alternatives (PcgGraph.h
// already includes PcgPointData.h/PcgAttributeSet.h for the identical reason -- Points/AttributeSet
// are also variant alternatives). PcgGraph.h therefore includes THIS file too (additively), and
// this file only ever depends on PcgPointData.h/PcgAttributeSet.h/PcgSeededRandom.h -- never on
// PcgGraph.h, PcgGraphEvaluator.h, or PcgNodePlugin.h. The actual PCG_REGISTER_NODE_TYPE graph-node
// registration (which DOES need PcgNodePlugin.h) lives exclusively in PcgMeshSpawner.cpp, a
// translation unit, where an #include cycle is a non-issue (a .cpp file is never itself #included).
//
// --- Density-threshold culling: raw `density`, not GetEffectiveDensity() ------------------------
// SpawnFromPoints (below) filters against each point's raw, intrinsic `density` field, mirroring
// PcgDensityTransformFilter.h's own FilterByDensity precedent EXACTLY (see that header's own field
// comment: "Filters against each point's raw, intrinsic density field, NOT GetEffectiveDensity() --
// a density-filter graph node has no per-sample local-space position to evaluate the steepness/
// bounds-edge falloff against... exactly matching UE5.8's own Density Filter node"). The same
// reasoning applies here, for the same structural reason: PcgPoint::GetEffectiveDensity(localPos)
// evaluates the steepness/bounds-edge falloff at an ARBITRARY local-space sample position relative
// to the point's own bounds -- it answers "how dense is the field near THIS position inside the
// point's footprint", not "should this already-placed point spawn something". A mesh spawner has no
// such arbitrary sample position to offer; the only "local" position available is the point's own
// origin, which only coincides with GetEffectiveDensity's meaningful bounds-CENTER case when
// boundsMin/boundsMax happen to still be symmetric around zero (PcgPoint's own default) -- for any
// point whose bounds were later shifted off-center by an upstream node, silently probing local
// {0,0,0} would evaluate the falloff at an arbitrary (and possibly clipped/zero) location instead of
// the point's own density, giving a wrong and non-obvious result. Raw `density` has no such hidden
// failure mode and is the documented, precedented choice in this codebase for exactly this class of
// node.
//
// --- Determinism ----------------------------------------------------------------------------------
// Per-point mesh selection draws from a PcgSeededRandom constructed from
// PcgHashCombine(point.seed, seed) -- the SAME "combine this point's own seed with a call-level
// salt, then construct a fresh independent stream" idiom PcgDensityTransformFilter.h's own
// ApplyTransformJitter already established (see that header's own comment) -- so a given point's
// mesh pick is reproducible regardless of vector order, batch size, or what other points are also
// being spawned in the same call (order-independent per-point determinism). Re-running
// SpawnFromPoints on the exact same points + weightedMeshes + seed always reproduces byte-identical
// output.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgPointData.h"

namespace pcg {

    // One weighted candidate mesh in a Mesh Spawner's palette. `weight` is a RELATIVE weight, not a
    // normalized probability -- SpawnFromPoints normalizes internally via a cumulative-sum /
    // total-weight ratio, so callers can use any convenient scale (e.g. plain percentages summing to
    // 100, or arbitrary relative counts). A non-positive `weight` (zero or negative) contributes
    // ZERO selection probability -- see SpawnFromPoints' own comment for exactly how that is
    // guaranteed, not just approximately true.
    struct PcgMeshSpawnEntry {
        uint32_t meshID = 0;
        uint32_t materialID = 0;
        float weight = 0.0f;
    };

    // One resolved spawn request -- ready to feed renderer::PcgInstanceDrawPass::AcquireInstance
    // almost verbatim (see this file's own top-of-file comment for the exact field-order rationale):
    //   instanceDrawPass.AcquireInstance(request.meshID, request.materialID,
    //       request.position, request.rotation, request.scale);
    struct PcgSpawnRequest {
        uint32_t meshID = 0;
        uint32_t materialID = 0;
        maths::vec3 position{ 0.0f, 0.0f, 0.0f };
        maths::quat rotation{}; // Identity by default (maths::quat's own default: w=1, x=y=z=0).
        maths::vec3 scale{ 1.0f, 1.0f, 1.0f };
    };

    // Resolves each surviving point in `points` into exactly one PcgSpawnRequest, choosing a mesh
    // from `weightedMeshes` via area-weighted-style cumulative-weight selection (the identical
    // cumulative-prefix-sum + std::upper_bound binary-search technique PcgSurfaceSampler.cpp's own
    // SampleSurfacePoints() already uses for area-weighted triangle selection -- see that file's own
    // comment for the full rationale; this function reuses the SAME pattern rather than inventing a
    // different one, per this phase's own design brief).
    //
    // Per-point processing, in order:
    //   1. Density cull: if point.density < densityThreshold, the point is SKIPPED entirely (no
    //      PcgSpawnRequest is produced for it, and -- important for determinism of LATER points in
    //      the SAME call -- no random draw is consumed for a skipped point, since each point's draw
    //      derives independently from its own seed, never from a shared running stream). See this
    //      file's own top-of-file comment for why `density` (not GetEffectiveDensity()) is the field
    //      tested. densityThreshold defaults to 0.0f (PcgPoint::density's own documented minimum),
    //      which is a complete no-op cull -- every point with a valid (see PcgPointData.h,
    //      documented range [0,1]) density passes through unfiltered by default; pass a higher
    //      threshold to explicitly cull sparse points.
    //   2. Mesh selection: a fresh PcgSeededRandom is constructed from
    //      PcgHashCombine(point.seed, seed) (see this file's own top-of-file Determinism comment);
    //      exactly ONE draw (NextFloatRange(0, totalWeight)) selects a cumulative-weight bucket via
    //      std::upper_bound, exactly like PcgSurfaceSampler's own triangle-selector draw.
    //   3. Transform: the resolved request's position/rotation/scale are copied DIRECTLY from the
    //      point's own fields (PcgPoint::position/rotation/scale) -- this phase does not apply any
    //      additional per-mesh scale multiplier (kept simple, see this phase's own design brief: "a
    //      per-mesh scale multiplier if you want... keep it simple if not clearly needed" -- a
    //      spawned instance's size is already fully controlled by whatever upstream sampler/filter
    //      authored the point's own `scale` field, e.g. Phase 3.1's Transform Filter jitter, so a
    //      second multiplier here would be redundant, not a missing feature).
    //
    // Edge cases (never crashes, never divides by zero):
    //   - `points` empty -> returns an empty vector.
    //   - `weightedMeshes` empty -> returns an empty vector (nothing to select FROM).
    //   - `weightedMeshes` entirely zero/negative weight -> returns an empty vector (a degenerate,
    //     zero-total-weight distribution has nothing meaningful to select; mirrors
    //     PcgSurfaceSampler's own "all-degenerate input -> empty result" convention for an
    //     all-zero-area triangle list).
    //   - A zero (or negative, clamped to zero) `weight` entry NEVER gets selected: it contributes a
    //     zero-WIDTH slice to the cumulative-weight distribution, so std::upper_bound has exactly
    //     zero probability of landing inside it for a continuous random draw -- the identical
    //     "excluded for free, no separate filtering pass" guarantee PcgSurfaceSampler.cpp's own
    //     degenerate-triangle handling documents.
    std::vector<PcgSpawnRequest> SpawnFromPoints(const std::vector<PcgPoint>& points,
        const std::vector<PcgMeshSpawnEntry>& weightedMeshes, uint32_t seed, float densityThreshold = 0.0f);

    // ---------------------------------------------------------------------------------------------
    // PcgAttributeSet encoding for a weighted mesh list -- used by this phase's graph-node
    // registration (PcgMeshSpawner.cpp) to read a node's `params` (a plain pcg::PcgAttributeSet,
    // reused per PcgGraph.h's own "reuse Phase 1's attribute bag rather than inventing a second
    // generic typed key/value type" design brief) back into a std::vector<PcgMeshSpawnEntry>.
    //
    // Encoding scheme: PcgAttributeSet::AttributeValue (PcgAttributeSet.h) is a deliberately CLOSED
    // std::variant<bool, int32_t, float, maths::vec3, std::string> with NO array/list alternative --
    // so a variable-length list of {meshID, materialID, weight} triples cannot be stored as one
    // single attribute value. Instead: one "entryCount" (int32_t) key gives the list length N,
    // followed by N groups of 3 INDEXED keys, "mesh<i>_id"/"mesh<i>_material"/"mesh<i>_weight" (i in
    // [0, N)), storing meshID/materialID/weight respectively. meshID/materialID are stored as
    // int32_t (AttributeValue has no uint32_t alternative) -- every realistic meshID/materialID in
    // this codebase's actual content pool (the 4 fixed streaming archetype shapes, see
    // tools/WorldPartition/ArchetypeMeshLibrary.h) is a small non-negative index that fits comfortably
    // inside int32_t's positive range, so this is a safe, deliberate simplification, not a silent
    // truncation risk for this project's real ID ranges; DecodeWeightedMeshList defensively clamps a
    // decoded negative value to 0 rather than reinterpreting its bit pattern as a large uint32_t.
    // This "flat indexed-key run" scheme was chosen over inventing a "three parallel arrays" struct
    // type specifically because PcgAttributeSet's own backing storage IS ALREADY a flat,
    // insertion-order-preserving key/value list (see PcgAttributeSet.h's own class comment) --
    // indexed keys reuse that existing shape directly, with zero new container/serialization
    // machinery needed.
    inline constexpr const char* kMeshSpawnEntryCountKey = "entryCount";

    // Node-param keys read directly by this file's PCG_REGISTER_NODE_TYPE registration
    // (PcgMeshSpawner.cpp) -- exposed here (not left as magic strings buried in the .cpp) so a test
    // or a future graph-authoring tool can construct a matching PcgAttributeSet without having to
    // duplicate/guess the exact key spelling the node's execute callback reads.
    inline constexpr const char* kSpawnerDensityThresholdParamKey = "densityThreshold";
    inline constexpr const char* kSpawnerSeedParamKey = "seed";

    // Appends `weightedMeshes` to `outParams` using the encoding documented above. Does NOT clear
    // `outParams` first (callers building a node's full param set -- entry list plus
    // densityThreshold/seed -- call this alongside plain outParams.Set() calls for the other keys,
    // in any order); re-encoding into a `outParams` that already has a DIFFERENT (e.g. larger)
    // "entryCount" from a previous call correctly overwrites every key this call touches (via
    // PcgAttributeSet::Set's own upsert semantics) but does NOT retroactively erase stale
    // mesh<i>_* keys the previous, longer encoding left behind for i >= the new entryCount -- callers
    // that need a clean re-encode of a shrinking list should start from a fresh PcgAttributeSet.
    void EncodeWeightedMeshList(PcgAttributeSet& outParams, const std::vector<PcgMeshSpawnEntry>& weightedMeshes);

    // Inverse of EncodeWeightedMeshList. Reads "entryCount" then that many "mesh<i>_id"/
    // "mesh<i>_material"/"mesh<i>_weight" triples back into a fresh vector, in index order. Lenient,
    // never throws: an absent or non-int32 "entryCount" yields an empty result; a missing or
    // wrong-typed field WITHIN an otherwise-valid entry range defaults that one field to 0 (via
    // PcgAttributeSet::GetOr) rather than aborting the whole decode -- consistent with
    // PcgAttributeSet's own general "value or fallback, never throw" accessor philosophy.
    std::vector<PcgMeshSpawnEntry> DecodeWeightedMeshList(const PcgAttributeSet& params);

}
