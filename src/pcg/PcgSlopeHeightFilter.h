#pragma once

// PCG framework roadmap, Phase 3.4 ("Slope/Height Filter + Projection"): UE5.8 PCG's Slope Filter /
// Height Filter / Projection node equivalents -- pure, deterministic post-processing steps that run
// AFTER a point set already exists (from a Phase 2 sampler, or from an upstream Phase 3 filter like
// Self-Pruning/Density-Transform), never generating new points themselves.
//
//   - Slope Filter: excludes points whose underlying terrain is too steep (e.g. "no trees above 60
//     degrees" for a cliff face). Two overloads are provided:
//       1. A PcgTerrainPointBatch overload that reads the REAL per-point terrain slope the Terrain
//          Sampler (Phase 2.2, PcgTerrainSampler.h/.cpp) already computed at scatter time (parallel
//          `slopeRadians` array) -- this is the exact, ground-truth slope value.
//       2. A plain std::vector<PcgPoint> overload for points that did NOT come from the Terrain
//          Sampler (e.g. Surface/Volume/Spline sampler output, or a point set already re-oriented by
//          an upstream filter) and therefore carry no parallel slope array. This overload derives an
//          APPROXIMATE slope from each point's own `rotation`: it rotates local +Y by that rotation
//          to recover the point's "up" direction, then measures the angle between that direction and
//          world-up the exact same way ComputeSlopeRadians already does for a real terrain normal.
//          This is only as good as the point's rotation already being aligned to whatever surface it
//          was sampled from (true for every existing Phase 2 sampler's output, which all call
//          QuatFromNormal or an equivalent "align to normal" step) -- it is NOT a real terrain query,
//          just a proxy derived from data the point already carries. Documented here so a future
//          caller does not mistake this for an actual re-query of the ground.
//
//   - Height Filter: excludes points whose world-space Y (this codebase's up-axis convention --
//     PcgPoint::position uses maths::vec3 with Y as the up component, matching PcgTerrainSampler's
//     own `worldY`/`point.position = maths::vec3{ worldX, worldY, worldZ }` construction and
//     PcgLandscapeData's `worldOffset.y` "vertical anchor" field) falls outside a [minHeight,
//     maxHeight] range (inclusive both ends) -- e.g. "no rocks below sea level".
//
//   - Projection: re-snaps a point set onto the terrain surface after some upstream operation (a
//     jitter, Self-Pruning's horizontal repositioning, ...) has moved points horizontally, leaving
//     their Y/rotation stale. For each point, re-samples the terrain height/normal at the point's
//     CURRENT (x, z) via PcgTerrainSampler's existing CPU query (SampleHeightCPU/
//     ComputeTerrainNormalCPU -- reused directly, not reimplemented), overwrites position.y with the
//     freshly sampled height, and re-aligns rotation to the freshly sampled normal via
//     PcgTerrainSampler's own QuatFromNormal (also reused directly: it is a plain, header-exposed
//     free function, not hidden in an anonymous namespace, so there is nothing to re-derive here).
//
// All three operations are pure functions of their inputs -- no PcgSeededRandom stream is touched
// anywhere in this file -- so running the same filter/projection on the same input always produces
// the exact same output, matching this codebase's "same input -> byte-identical output" determinism
// requirement (see PcgSeededRandom.h's own header comment for why that requirement exists project-wide).

#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"
#include "pcg/PcgTerrainSampler.h"

namespace pcg {

    // Slope Filter, terrain-batch overload: keeps only the points in `batch` whose corresponding
    // REAL terrain slope (`batch.slopeRadians[i]`, computed by the Terrain Sampler from the actual
    // analytic terrain normal at that point's sample position -- see PcgTerrainSampler.h's own
    // PcgTerrainPointBatch comment) is <= `maxSlopeRadians`. `batch.points`/`batch.slopeRadians` are
    // documented as exactly parallel arrays; this function only iterates the common prefix length in
    // the (should-never-happen) case a caller hands it a batch built by hand with mismatched sizes,
    // rather than reading past either array's end.
    std::vector<PcgPoint> FilterBySlope(const PcgTerrainPointBatch& batch, float maxSlopeRadians);

    // Slope Filter, generic overload (APPROXIMATE -- see this file's own header comment for the full
    // caveat): for a plain point set with no parallel terrain-slope array, derives a slope-like value
    // from each point's own `rotation` (angle between rotation-applied local +Y and world-up) and
    // keeps only the points whose derived value is <= `maxSlopeRadians`.
    std::vector<PcgPoint> FilterBySlope(const std::vector<PcgPoint>& points, float maxSlopeRadians);

    // Height Filter: keeps only the points whose `position.y` falls within [minHeight, maxHeight]
    // (both ends inclusive -- a point sitting exactly on either boundary passes).
    std::vector<PcgPoint> FilterByHeight(const std::vector<PcgPoint>& points, float minHeight, float maxHeight);

    // Projection: re-samples `terrain`'s height/normal at each point's CURRENT (x, z) and overwrites
    // that point's position.y (to the freshly sampled height) and rotation (re-aligned to the freshly
    // sampled normal via QuatFromNormal) in place. Leaves position.x/position.z and every other
    // PcgPoint field (density, color, seed, bounds, steepness, scale) untouched. Intended to run
    // immediately AFTER any upstream step that moved points horizontally without keeping their
    // vertical placement in sync with the ground (e.g. Self-Pruning's jitter, a future Density-
    // Transform horizontal-offset attribute) -- calling this on points that are already correctly
    // placed is a (cheap) no-op, not an error.
    void ProjectOntoTerrain(std::vector<PcgPoint>& points, const PcgLandscapeData& terrain,
        float normalEpsilon = kDefaultTerrainNormalEpsilon);

}
