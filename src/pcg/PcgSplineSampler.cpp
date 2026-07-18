#include "pcg/PcgSplineSampler.h"

#include "pcg/PcgSeededRandom.h"

#include <algorithm>
#include <cmath>

namespace pcg {

    namespace {

        // This codebase's world-space forward/up/right convention, matching
        // PcgSplineControlPoint's own default tangent {0,0,1} (PcgSpatialData.h) -- Z-forward,
        // Y-up, X-right (a standard right-handed convention already implicit in that default).
        constexpr maths::vec3 kWorldForward{ 0.0f, 0.0f, 1.0f };
        constexpr maths::vec3 kWorldUp{ 0.0f, 1.0f, 0.0f };
        constexpr maths::vec3 kWorldRight{ 1.0f, 0.0f, 0.0f };

        // General "small number" guards against divide-by-zero / normalizing a near-zero vector --
        // matching PcgPointData.h's own kMinExtent/kMinDenom convention (1.0e-6f) for the same
        // purpose.
        constexpr float kEpsilon = 1.0e-6f;

        // Standard "shortest-arc" quaternion construction: the unique rotation of angle
        // acos(dot(from,to)) about the axis (from x to) that carries the unit vector `from` onto
        // the unit vector `to`, matching maths::quat::FromAxisAngle's own axis-angle convention.
        // `from`/`to` must already be normalized by the caller (both call sites below pass
        // known-unit vectors, so no redundant re-normalization here).
        //
        // Two degenerate cases handled explicitly, since (from x to) itself degenerates exactly
        // when `from`/`to` are parallel or anti-parallel:
        //   - Already aligned (dot ~= +1): the identity rotation (default-constructed maths::quat)
        //     is correct and the cross-product axis would be a zero vector anyway.
        //   - Exactly opposite (dot ~= -1): infinitely many 180-degree rotations carry `from` onto
        //     `to` (any axis perpendicular to `from` works) -- picks world-up crossed with `from`
        //     as that perpendicular axis, falling back to world-right crossed with `from` for the
        //     rare case `from` itself is parallel to world-up (both cross products cannot be
        //     degenerate simultaneously, since world-up and world-right are themselves
        //     perpendicular).
        maths::quat QuatAlignVector(const maths::vec3& from, const maths::vec3& to) {
            const float d = std::clamp(from.Dot(to), -1.0f, 1.0f);

            if (d > 1.0f - kEpsilon) {
                return maths::quat{}; // Already aligned (or close enough) -- identity rotation.
            }

            if (d < -1.0f + kEpsilon) {
                // Exactly (or nearly) opposite: pick any axis perpendicular to `from` for a
                // 180-degree flip.
                maths::vec3 axis = kWorldUp.Cross(from);
                if (axis.Length() < kEpsilon) {
                    axis = kWorldRight.Cross(from);
                }
                axis = axis.Normalize();
                return maths::quat::FromAxisAngle(axis, maths::PI);
            }

            const maths::vec3 axis = from.Cross(to).Normalize();
            const float angle = std::acos(d);
            return maths::quat::FromAxisAngle(axis, angle);
        }

    } // namespace

    // Arc-length table resolution choice: `subdivisionsPerSegment` fine, uniform `t` steps PER
    // SEGMENT (a "segment" = the Hermite span between two consecutive control points -- i.e. one
    // unit step of PcgSplineData's global `t`), and the default of 256 (see the header's
    // PcgSplineSamplerParams::arcLengthSubdivisionsPerSegment default) sits comfortably in this
    // task's suggested 100-500 range.
    //
    // Accuracy/cost tradeoff: a cubic Hermite segment's curvature is bounded by its two endpoint
    // tangents, so the piecewise-linear chord approximation this function builds converges to the
    // true arc length quadratically as subdivision count increases (the classic "polygon
    // approximates a smooth curve" error bound). 256 samples per segment keeps the worst-case chord
    // error at a small fraction of a percent of segment length even for a segment whose tangents
    // make it curve sharply (a segment gentle enough to be authored as a nearly-straight path,
    // which is the overwhelmingly common demoscene-path/river-spline case this codebase's existing
    // spline authoring targets, needs far fewer samples than that to be near-exact) -- while the
    // resulting table build cost stays trivially cheap: a spline with C control points costs
    // O(256 * (C-1)) position evaluations, done ONCE per SampleSplineByArcLength call (not per
    // frame, not per point), so even a lavishly long 50-control-point authored path is only ~12800
    // evaluations, microseconds of CPU time.
    std::vector<PcgArcLengthSample> BuildSplineArcLengthTable(const PcgSplineData& spline, int subdivisionsPerSegment) {
        std::vector<PcgArcLengthSample> table;

        const size_t controlPointCount = spline.ControlPointCount();
        if (controlPointCount < 2) {
            // Degenerate (0 or 1 control points, see PcgSplineData::EvaluateSegment): no extent to
            // walk. A single {t=0, distance=0} entry keeps this table non-empty so
            // ResolveSplineTAtArcLength never needs an empty-table special case.
            table.push_back(PcgArcLengthSample{ 0.0f, 0.0f });
            return table;
        }

        const int clampedSubdivisions = std::max(subdivisionsPerSegment, 1);
        const float maxT = static_cast<float>(controlPointCount - 1);
        const int totalSteps = clampedSubdivisions * static_cast<int>(controlPointCount - 1);

        table.reserve(static_cast<size_t>(totalSteps) + 1);
        table.push_back(PcgArcLengthSample{ 0.0f, 0.0f });

        maths::vec3 previousPosition = spline.EvaluatePosition(0.0f);
        float accumulatedDistance = 0.0f;
        for (int step = 1; step <= totalSteps; ++step) {
            const float t = maxT * (static_cast<float>(step) / static_cast<float>(totalSteps));
            const maths::vec3 position = spline.EvaluatePosition(t);
            accumulatedDistance += (position - previousPosition).Length();
            table.push_back(PcgArcLengthSample{ t, accumulatedDistance });
            previousPosition = position;
        }

        return table;
    }

    float ResolveSplineTAtArcLength(const std::vector<PcgArcLengthSample>& table, float targetDistance) {
        // BuildSplineArcLengthTable always returns at least one entry -- see its own comment.
        if (table.size() == 1) {
            return table.front().t;
        }

        const float clampedDistance = std::clamp(targetDistance, table.front().distance, table.back().distance);

        // Binary search (std::lower_bound over a sorted-by-distance table, exactly what
        // BuildSplineArcLengthTable produces since `accumulatedDistance` is monotonically
        // non-decreasing) for the first sample whose distance is >= clampedDistance.
        const auto it = std::lower_bound(table.begin(), table.end(), clampedDistance,
            [](const PcgArcLengthSample& sample, float distance) {
                return sample.distance < distance;
            });

        if (it == table.begin()) {
            return table.front().t;
        }
        if (it == table.end()) {
            return table.back().t;
        }

        const PcgArcLengthSample& hi = *it;
        const PcgArcLengthSample& lo = *(it - 1);
        const float span = hi.distance - lo.distance;
        if (span < kEpsilon) {
            // Degenerate zero-length span (e.g. a cusp/reversal where the curve briefly doubles
            // back on itself within one subdivision step) -- fall back to the lower sample's `t`
            // rather than dividing by ~0.
            return lo.t;
        }

        const float frac = (clampedDistance - lo.distance) / span;
        return lo.t + (hi.t - lo.t) * frac;
    }

    std::vector<PcgPoint> SampleSplineByArcLength(const PcgSplineData& spline, const PcgSplineSamplerParams& params) {
        std::vector<PcgPoint> points;

        if (params.spacing <= kEpsilon) {
            return points; // Non-positive spacing: nothing meaningful to place, and looping would never terminate.
        }

        const std::vector<PcgArcLengthSample> table = BuildSplineArcLengthTable(spline, params.arcLengthSubdivisionsPerSegment);
        const float totalArcLength = table.back().distance;
        if (totalArcLength <= kEpsilon) {
            return points; // Degenerate spline (0 or 1 control points, or a zero-length curve): no extent to sample.
        }

        // Points are placed at k * spacing for k = 1, 2, 3, ... (NOT starting at distance 0) --
        // this sampler treats `spacing` purely as a spacing/pitch value, the same way UE5.8's
        // Spline Sampler's "Distance" mode advances a running distance counter by `spacing` before
        // placing each point rather than placing an initial point at the spline's start regardless
        // of spacing; a caller that specifically wants a point pinned to the curve's start can
        // always prepend spline.EvaluatePosition(0.0f) itself.
        //
        // Small epsilon added before the floor() guards against float rounding shorting the count
        // by one when totalArcLength is (up to rounding noise) an exact multiple of spacing --
        // e.g. totalArcLength == 10.0f and spacing == 2.0f should yield k=1..5, not k=1..4 because
        // totalArcLength/spacing computed as 4.999999f.
        const float ratio = totalArcLength / params.spacing;
        const int pointCount = static_cast<int>(std::floor(ratio + 1.0e-4f));
        if (pointCount <= 0) {
            return points; // The curve is shorter than a single spacing step.
        }
        points.reserve(static_cast<size_t>(pointCount));

        // All per-point randomness (distance jitter, perpendicular jitter, and each output point's
        // own PcgPoint::seed) is drawn from ONE seeded stream derived from params.seed, consumed in
        // a fixed, parameter-independent order (jitter draws happen every iteration even when the
        // corresponding jitter amount is 0, since NextFloatRange(-0,0) deterministically returns
        // 0) -- so toggling a jitter amount on/off never reshuffles which stream index feeds any
        // other point's seed or jitter.
        PcgSeededRandom stream(params.seed);

        for (int k = 1; k <= pointCount; ++k) {
            const float baseDistance = static_cast<float>(k) * params.spacing;
            const float distanceJitter = stream.NextFloatRange(-params.jitterDistance, params.jitterDistance);
            const float perpendicularJitter = stream.NextFloatRange(-params.jitterPerpendicular, params.jitterPerpendicular);
            const uint32_t pointSeed = stream.NextUint32();

            const float jitteredDistance = std::clamp(baseDistance + distanceJitter, 0.0f, totalArcLength);
            const float t = ResolveSplineTAtArcLength(table, jitteredDistance);

            const maths::vec3 position = spline.EvaluatePosition(t);
            const maths::vec3 rawTangent = spline.EvaluateTangent(t);
            const float tangentLength = rawTangent.Length();
            const maths::vec3 tangent = (tangentLength > kEpsilon) ? (rawTangent * (1.0f / tangentLength)) : kWorldForward;

            // Perpendicular offset direction: tangent x world-up, matching this file's own
            // "fence/lamppost scatter to either side of the centerline" use case (see
            // PcgSplineSamplerParams::perpendicularOffset's own comment). Falls back to
            // world-right when the tangent is itself (near-)parallel to world-up (a near-vertical
            // curve segment, e.g. a spline authored as a waterfall/cliff path).
            maths::vec3 perpendicularDir = tangent.Cross(kWorldUp);
            const float perpendicularDirLength = perpendicularDir.Length();
            perpendicularDir = (perpendicularDirLength > kEpsilon) ? (perpendicularDir * (1.0f / perpendicularDirLength)) : kWorldRight;

            const float totalPerpendicularOffset = params.perpendicularOffset + perpendicularJitter;

            PcgPoint point; // Every field besides position/rotation/seed keeps PcgPoint's own default.
            point.position = position + perpendicularDir * totalPerpendicularOffset;
            point.rotation = QuatAlignVector(kWorldForward, tangent);
            point.seed = pointSeed;

            points.push_back(point);
        }

        return points;
    }

}
