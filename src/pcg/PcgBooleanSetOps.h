#pragma once

// PCG framework roadmap, Phase 3.3 ("Boolean Set Operations"): UE5.8 PCG parity for the
// Union/Intersect/Difference graph nodes -- but, matching UE5.8's own PCG framework, these operate
// on POINT SETS, not on raw geometry booleans. Two independently-sampled point sets (e.g. one
// scatter pass for trees, another for rocks) almost never share exact positions, so a literal
// "point-for-point set intersection by identical position" is rarely the useful operation; the
// practical UE5.8 PCG pattern instead filters ONE point set against a SPATIAL region (a volume, or
// a distance-from-spline test) -- e.g. "no trees within N meters of the road spline", or "only keep
// points inside region A and outside region B". This file implements exactly that pattern:
//   - Union: plain point-set concatenation (see Union's own comment below for why no dedup happens
//     here -- that is Phase 3.2's Self-Pruning filter's job, a separate later step).
//   - IntersectWithVolume / DifferenceFromVolume: keep-inside / keep-outside filtering against a
//     pcg::PcgVolumeData (PcgSpatialData.h), reusing its own ContainsWorldPoint test.
//   - DifferenceFromSpline: keep-outside-exclusion-radius filtering against a pcg::PcgSplineData
//     (PcgSpatialData.h) curve, reusing Phase 2.4's arc-length utilities (PcgSplineSampler.h/.cpp)
//     for the candidate sample-point set rather than re-deriving a fresh dense spline re-sample.
//   - IntersectMultipleVolumes: a practical "union of inclusion regions" helper (keep points inside
//     ANY of several volumes) that composes with IntersectWithVolume rather than requiring the
//     caller to loop and re-concatenate results themselves.
//
// This phase deliberately stops at these 5 operations -- Phase 3.1 (Density/Transform filter),
// Phase 3.2 (Self-Pruning filter), and Phase 3.4 (Slope/Height filter) are separate, independently
// built files (PcgDensityTransformFilter.*/PcgSelfPruningFilter.*/PcgSlopeHeightFilter.*, not
// touched here), and later PCG roadmap phases (spawners, the graph engine wiring these nodes
// together) are out of scope for this file entirely.
//
// Determinism: every function here is a PURE function of its inputs -- no randomness, no hidden
// global/static state, no iteration-order dependence (output point ORDER always follows input
// order, filtered-down but never reshuffled) -- so identical inputs always produce byte-identical
// outputs. Unlike the sampler phases (which derive per-point randomness from a seed), boolean set
// operations have no seed parameter at all: there is nothing here for a seed to drive.

#include <vector>

#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

namespace pcg {

    // Union of two point sets: simple concatenation, `a`'s points followed by `b`'s points, in
    // their own original relative order. UE5.8's PCG Union node on point data is exactly this --
    // it does NOT deduplicate overlapping/coincident points, since PCG point data has no inherent
    // notion of point identity/equality to dedupe by (two points at the same position but different
    // density/seed/color are not "the same point"). A caller wanting deduplication should run
    // Phase 3.2's Self-Pruning filter on the result as a separate, explicit later step -- that
    // pruning logic is intentionally NOT reimplemented here.
    //
    // Result size is always exactly a.size() + b.size() -- the defining, trivially-verifiable
    // property of a non-deduplicating concatenation.
    std::vector<PcgPoint> Union(const std::vector<PcgPoint>& a, const std::vector<PcgPoint>& b);

    // Keeps only the points from `points` whose `position` is INSIDE `volume` (per `volume`'s own
    // ContainsWorldPoint test, PcgSpatialData.h). Relative order of the surviving points is
    // preserved from the input.
    //
    // Boundary convention: a point exactly ON the volume's surface is treated as INSIDE (kept) --
    // this simply inherits PcgVolumeData::ContainsWorldPoint's own convention, which compares each
    // local axis with `<=` against halfExtents, not `<`. IntersectWithVolume and
    // DifferenceFromVolume are therefore exact complements of each other over `points`: every input
    // point ends up in exactly one of the two results, never both, never neither.
    std::vector<PcgPoint> IntersectWithVolume(const std::vector<PcgPoint>& points, const PcgVolumeData& volume);

    // Keeps only the points from `points` whose `position` is OUTSIDE `volume` -- the complement of
    // IntersectWithVolume over the same `points`/`volume` pair (see IntersectWithVolume's own
    // comment for the shared boundary convention: a boundary point counts as inside, so it is
    // EXCLUDED here, matching the "no trees inside this no-build volume" use case). Relative order
    // of the surviving points is preserved from the input.
    std::vector<PcgPoint> DifferenceFromVolume(const std::vector<PcgPoint>& points, const PcgVolumeData& volume);

    // Keeps only the points from `points` whose closest distance to `spline`'s curve exceeds
    // `exclusionRadius` -- the "no trees within N meters of the road spline" use case. Relative
    // order of the surviving points is preserved from the input.
    //
    // Closest-point-on-spline approximation: rather than an exact closest-point-on-cubic-Hermite-
    // curve solve (a quintic root-find per query point, with no closed-form solution), this reuses
    // Phase 2.4's own arc-length lookup table (PcgSplineSampler.h's BuildSplineArcLengthTable,
    // built here at its own default 256-subdivisions-per-segment resolution, matching
    // PcgSplineSamplerParams::arcLengthSubdivisionsPerSegment's own default) as the candidate
    // sample-point set: the curve's world-space position is evaluated once at every table entry's
    // `t`, and each input point's distance-to-curve is approximated as the MINIMUM straight-line
    // distance to any of those sampled positions. This is a coarse (piecewise-nearest-sample, not a
    // true nearest-point-on-the-connecting-chord) but standard and entirely sufficient
    // approximation for this use case: with 256 samples per segment the worst-case error is a small
    // fraction of the local sample spacing (itself already sub-percent of a typical authored
    // segment length, per BuildSplineArcLengthTable's own accuracy discussion in
    // PcgSplineSampler.cpp) -- i.e. utterly negligible next to a `exclusionRadius` that is always
    // measured in whole meters for this feature's actual use cases (tree/road clearance). Reusing
    // the SAME table-building function as the Phase 2.4 sampler (rather than a fresh dense
    // re-sample of its own) also keeps this exclusion test numerically consistent with wherever
    // that spline's points were originally scattered from.
    //
    // Boundary convention: a point at EXACTLY `exclusionRadius` distance is treated as inside the
    // exclusion zone (excluded, not kept) -- the kept condition is a strict `>`, matching
    // IntersectWithVolume/DifferenceFromVolume's shared "boundary counts as the region interior"
    // convention above (the exclusion disc around the spline is the "volume" being subtracted here,
    // and its own boundary is inside it).
    //
    // Degenerate splines (0 or 1 control points -- BuildSplineArcLengthTable's own single-entry
    // table for those cases) still produce a defined answer: distance is measured to that single
    // sampled position (the spline's sole point, or the origin for a 0-control-point spline),
    // exactly as a "spline" that is really just a point would be expected to behave.
    std::vector<PcgPoint> DifferenceFromSpline(const std::vector<PcgPoint>& points, const PcgSplineData& spline, float exclusionRadius);

    // Keeps only the points from `points` whose `position` is inside AT LEAST ONE of `volumes` --
    // a "union of inclusion regions" composition of IntersectWithVolume over multiple volumes.
    // Relative order of the surviving points is preserved from the input; a point inside two or
    // more overlapping volumes is still emitted exactly ONCE (this tests each input point against
    // every volume and keeps it as soon as any one volume contains it, rather than looping
    // per-volume and concatenating IntersectWithVolume's own per-volume results together, which
    // would duplicate points lying in an overlap region).
    //
    // An empty `volumes` list contains no points by definition (there is no volume for any point to
    // be inside), so this returns an empty result -- never `points` itself.
    std::vector<PcgPoint> IntersectMultipleVolumes(const std::vector<PcgPoint>& points, const std::vector<PcgVolumeData>& volumes);

}
