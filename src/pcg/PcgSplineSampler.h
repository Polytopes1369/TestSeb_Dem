#pragma once

// PCG framework roadmap, Phase 2.4 ("Spline Sampler"): UE5.8 PCG parity for the "Spline Sampler"
// graph node -- places pcg::PcgPoint instances along an authored pcg::PcgSplineData curve
// (PcgSpatialData.h) at regular REAL-WORLD-DISTANCE (arc-length) intervals, not regular steps of
// the curve's own Hermite parameter `t`. This distinction is the entire point of this file: `t`
// steps are NOT uniform-speed (the Hermite curve's instantaneous speed |dP/dt| varies along its
// length -- it slows through tight turns and speeds up along straight, widely-spaced control
// points), so naively calling EvaluatePosition(t) at t = 0, dt, 2*dt, ... would bunch points
// together in fast segments and spread them apart in slow ones. UE5.8's Spline Sampler node
// instead scatters points at a fixed real-world spacing along the curve (e.g. fence posts every
// 2m, lamp posts every 8m) -- see this file's own PcgSplineSampler.cpp for the arc-length
// remapping technique that makes that possible on top of PcgSplineData's t-parametrized API.
//
// This phase deliberately stops at THIS ONE sampler -- Phase 2.1/2.2/2.3 (Surface/Terrain/Volume
// samplers) are separate, independently-built files (PcgSurfaceSampler.*/PcgTerrainSampler.*/
// PcgVolumeSampler.*, not touched here) and later PCG roadmap phases (filters/spawners/the graph
// engine that wires samplers together) are out of scope for this file entirely.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

namespace pcg {

    // One entry of the arc-length lookup table PcgSplineSampler.cpp builds internally to remap
    // "distance travelled along the curve" back to the Hermite parameter `t` PcgSplineData's own
    // EvaluatePosition/EvaluateTangent expect. Exposed publicly (rather than kept file-private)
    // because "distance -> t" is a generically useful query for any FUTURE consumer that wants to
    // walk a PcgSplineData curve at a controlled real-world pace without re-deriving this exact
    // remapping technique -- e.g. a future debug-gizmo curve visualizer, or a future PCG node that
    // needs "the point 5 meters into this spline" without wanting a full point-array sample pass.
    struct PcgArcLengthSample {
        float t = 0.0f;        // PcgSplineData global parameter, in [0, ControlPointCount()-1].
        float distance = 0.0f; // Cumulative arc length from the curve's start (t=0) up to this sample's `t`.
    };

    // Builds the distance->t lookup table for `spline` by walking its Hermite parameter `t` in
    // `subdivisionsPerSegment` fine, uniform steps PER SEGMENT (a "segment" being the span between
    // two consecutive control points, i.e. one integer step of `t`) and accumulating the piecewise-
    // linear (straight-line-between-samples) distance between consecutive sampled positions. See
    // PcgSplineSampler.cpp's own comment on this function for the accuracy/cost tradeoff behind the
    // chosen default resolution.
    //
    // Degenerate spline handling (0 or 1 control points, see PcgSplineData::EvaluateSegment): the
    // curve has no extent to walk, so the returned table is a single {t=0, distance=0} entry --
    // never empty, so callers (including ResolveSplineTAtArcLength below) never need a
    // table.empty() special case.
    std::vector<PcgArcLengthSample> BuildSplineArcLengthTable(const PcgSplineData& spline, int subdivisionsPerSegment);

    // Resolves the Hermite parameter `t` corresponding to `targetDistance` (arc length from the
    // curve's start), via a binary search over `table` (assumed sorted by ascending `distance`,
    // i.e. exactly what BuildSplineArcLengthTable produces) followed by a linear interpolation
    // between the two bracketing samples. `targetDistance` is clamped into [0, table.back().distance]
    // first, so out-of-range inputs return the curve's start/end `t` rather than extrapolating.
    float ResolveSplineTAtArcLength(const std::vector<PcgArcLengthSample>& table, float targetDistance);

    // Configuration for SampleSplineByArcLength below. Every field has a sensible, inert default
    // (spacing aside, which has no meaningful "off" value and so has no default a caller should
    // rely on -- always set it explicitly) so a caller wanting plain "evenly spaced points on the
    // centerline" only needs to set `spacing` and `seed`.
    struct PcgSplineSamplerParams {
        // World-space distance between consecutive points along the curve's TRUE arc length (not
        // `t` steps). Must be > 0 -- SampleSplineByArcLength returns an empty point list for a
        // non-positive spacing rather than looping forever or dividing by zero.
        float spacing = 1.0f;

        // Deterministic seed for this sampler's derived per-point PcgSeededRandom stream (distance
        // jitter, perpendicular jitter, and each output point's own PcgPoint::seed). Same spline +
        // spacing + seed + every other field below -> byte-identical output, always.
        uint32_t seed = 0;

        // Signed perpendicular offset (world units), applied along `tangent x world-up` (world-up
        // = {0,1,0}, matching PcgSplineControlPoint's own default tangent {0,0,1} "Z-forward,
        // Y-up" convention) at every sampled point -- UE5.8 Spline Sampler's "offset to either side
        // of the centerline" use case (fence posts/lamp posts set back from a path's exact
        // centerline). 0 (the default) places points exactly on the centerline. Negative values
        // offset to the opposite side from positive ones.
        float perpendicularOffset = 0.0f;

        // Optional along-curve jitter: each sample's base arc-length distance (k * spacing) is
        // perturbed by a uniform random value in [-jitterDistance, +jitterDistance] before being
        // resolved back to `t` (then clamped back into the curve's valid [0, totalArcLength]
        // range). 0 (the default) disables this jitter -- every point lands EXACTLY on its
        // k * spacing mark.
        float jitterDistance = 0.0f;

        // Optional additional perpendicular jitter, on top of `perpendicularOffset`: a uniform
        // random value in [-jitterPerpendicular, +jitterPerpendicular] is added to the fixed
        // `perpendicularOffset` at every point independently. 0 (the default) disables this jitter.
        float jitterPerpendicular = 0.0f;

        // Resolution of the internal arc-length lookup table -- see BuildSplineArcLengthTable's own
        // comment and PcgSplineSampler.cpp for the accuracy/cost discussion behind this default.
        // Must be >= 1; SampleSplineByArcLength clamps a caller-supplied value below 1 up to 1
        // rather than producing a degenerate/empty table.
        int arcLengthSubdivisionsPerSegment = 256;
    };

    // The Spline Sampler itself: walks `spline`'s TRUE arc length in `params.spacing`-sized steps
    // (at k * params.spacing for k = 1, 2, 3, ... while k * params.spacing <= the curve's total arc
    // length -- see PcgSplineSampler.cpp's own comment for why the first point is NOT placed at
    // distance 0) and emits one pcg::PcgPoint per step:
    //   - `position`: the curve's world-space position at that arc-length distance, optionally
    //     shifted by `params.perpendicularOffset` (+ its own jitter) perpendicular to the tangent.
    //   - `rotation`: a quaternion whose forward axis (world-space {0,0,1}, this codebase's
    //     forward convention -- see PcgSplineControlPoint's own default tangent) is aligned to the
    //     curve's LOCAL tangent direction at that point (normalized; degenerate zero-length
    //     tangents fall back to identity rotation).
    //   - `seed`: a value drawn from this call's own seeded stream (derived from `params.seed`),
    //     unique per output point and fully deterministic given identical inputs.
    //   - every other PcgPoint field (`density`, `scale`, `color`, `boundsMin`/`boundsMax`,
    //     `steepness`) is left at PcgPoint's own default -- this sampler only ever writes
    //     position/rotation/seed, exactly as this phase's task scope specifies.
    //
    // Returns an empty vector (never crashes) for: non-positive `params.spacing`, a degenerate
    // spline with 0 or 1 control points (zero arc length -- nothing to space points along), or any
    // spline whose total arc length is shorter than one `params.spacing` step.
    std::vector<PcgPoint> SampleSplineByArcLength(const PcgSplineData& spline, const PcgSplineSamplerParams& params);

}
