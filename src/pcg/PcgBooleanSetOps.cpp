#include "pcg/PcgBooleanSetOps.h"

#include "pcg/PcgSplineSampler.h"

#include <algorithm>
#include <limits>

namespace pcg {

    namespace {

        // Matches PcgSplineSamplerParams::arcLengthSubdivisionsPerSegment's own default (see that
        // struct's comment, PcgSplineSampler.h) -- kept as its own named constant here rather than
        // exposing a resolution parameter on DifferenceFromSpline's public signature, since this
        // phase's task scope fixes that signature to exactly (points, spline, exclusionRadius) and
        // 256 subdivisions/segment is already established as this codebase's accepted default
        // arc-length table resolution for spline queries of this kind.
        constexpr int kSplineDistanceSubdivisionsPerSegment = 256;

    } // namespace

    std::vector<PcgPoint> Union(const std::vector<PcgPoint>& a, const std::vector<PcgPoint>& b) {
        std::vector<PcgPoint> result;
        result.reserve(a.size() + b.size());
        result.insert(result.end(), a.begin(), a.end());
        result.insert(result.end(), b.begin(), b.end());
        return result;
    }

    std::vector<PcgPoint> IntersectWithVolume(const std::vector<PcgPoint>& points, const PcgVolumeData& volume) {
        std::vector<PcgPoint> result;
        result.reserve(points.size());
        for (const PcgPoint& point : points) {
            if (volume.ContainsWorldPoint(point.position)) {
                result.push_back(point);
            }
        }
        return result;
    }

    std::vector<PcgPoint> DifferenceFromVolume(const std::vector<PcgPoint>& points, const PcgVolumeData& volume) {
        std::vector<PcgPoint> result;
        result.reserve(points.size());
        for (const PcgPoint& point : points) {
            if (!volume.ContainsWorldPoint(point.position)) {
                result.push_back(point);
            }
        }
        return result;
    }

    std::vector<PcgPoint> DifferenceFromSpline(const std::vector<PcgPoint>& points, const PcgSplineData& spline, float exclusionRadius) {
        std::vector<PcgPoint> result;
        result.reserve(points.size());

        // Candidate sample-point set: reuse Phase 2.4's own arc-length table builder rather than
        // re-deriving a fresh dense spline re-sample (see this function's own header comment for
        // the accuracy/consistency rationale). The table always has at least one entry (even for a
        // degenerate 0/1-control-point spline, see BuildSplineArcLengthTable's own comment), so the
        // sampled-position list built below is never empty.
        const std::vector<PcgArcLengthSample> table = BuildSplineArcLengthTable(spline, kSplineDistanceSubdivisionsPerSegment);

        std::vector<maths::vec3> sampledPositions;
        sampledPositions.reserve(table.size());
        for (const PcgArcLengthSample& sample : table) {
            sampledPositions.push_back(spline.EvaluatePosition(sample.t));
        }

        for (const PcgPoint& point : points) {
            float minDistance = std::numeric_limits<float>::max();
            for (const maths::vec3& sampledPosition : sampledPositions) {
                const float distance = (point.position - sampledPosition).Length();
                minDistance = std::min(minDistance, distance);
            }

            // Strictly greater than: a point at exactly `exclusionRadius` counts as inside the
            // exclusion disc (excluded, not kept) -- see this function's own header comment for the
            // shared "boundary counts as the region interior" convention with
            // IntersectWithVolume/DifferenceFromVolume.
            if (minDistance > exclusionRadius) {
                result.push_back(point);
            }
        }

        return result;
    }

    std::vector<PcgPoint> IntersectMultipleVolumes(const std::vector<PcgPoint>& points, const std::vector<PcgVolumeData>& volumes) {
        std::vector<PcgPoint> result;
        result.reserve(points.size());

        for (const PcgPoint& point : points) {
            for (const PcgVolumeData& volume : volumes) {
                if (volume.ContainsWorldPoint(point.position)) {
                    result.push_back(point);
                    break; // Already kept -- stop checking the remaining volumes for THIS point, so
                           // an overlap region never emits the same point twice.
                }
            }
        }

        return result;
    }

}
