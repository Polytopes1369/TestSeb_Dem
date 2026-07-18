#include "pcg/PcgSlopeHeightFilter.h"

#include <algorithm>

// PCG framework roadmap, Phase 3.4 ("Slope/Height Filter + Projection"): implementation. See
// PcgSlopeHeightFilter.h's own header comment for the overall design/reuse rationale -- every
// terrain query below (SampleHeightCPU, ComputeTerrainNormalCPU, ComputeSlopeRadians, QuatFromNormal)
// is reused directly from PcgTerrainSampler.h/.cpp (Phase 2.2), not reimplemented here.

namespace pcg {

    std::vector<PcgPoint> FilterBySlope(const PcgTerrainPointBatch& batch, float maxSlopeRadians) {
        std::vector<PcgPoint> result;
        result.reserve(batch.points.size());

        // batch.points/batch.slopeRadians are documented as exactly parallel (PcgTerrainSampler.h's
        // own PcgTerrainPointBatch comment); only iterate the common prefix so a hand-built batch
        // with mismatched array sizes can never read past either array's end.
        const size_t count = std::min(batch.points.size(), batch.slopeRadians.size());
        for (size_t i = 0; i < count; ++i) {
            if (batch.slopeRadians[i] <= maxSlopeRadians) {
                result.push_back(batch.points[i]);
            }
        }
        return result;
    }

    std::vector<PcgPoint> FilterBySlope(const std::vector<PcgPoint>& points, float maxSlopeRadians) {
        std::vector<PcgPoint> result;
        result.reserve(points.size());

        // Local +Y is this codebase's "up" convention (PcgPoint's own identity-rotation default,
        // matching mat4::FromQuat/RotateVector's existing basis -- see PcgTerrainSampler.h's
        // QuatFromNormal comment). Rotating it by the point's own `rotation` recovers whatever
        // direction that point's transform currently treats as "up" -- an approximation of the
        // point's underlying surface normal, valid only insofar as the point's rotation was already
        // aligned to some surface by an upstream sampler/filter (see this file's own header comment
        // for the full caveat: this is NOT a real terrain re-query).
        const maths::vec3 kLocalUp{ 0.0f, 1.0f, 0.0f };
        for (const PcgPoint& point : points) {
            const maths::vec3 approxUp = point.rotation.RotateVector(kLocalUp);
            // Reuses ComputeSlopeRadians verbatim -- it is already exactly "angle between a direction
            // and world-up", the same computation this approximation needs, just fed a
            // rotation-derived direction instead of a real terrain analytic normal.
            const float approxSlopeRadians = ComputeSlopeRadians(approxUp);
            if (approxSlopeRadians <= maxSlopeRadians) {
                result.push_back(point);
            }
        }
        return result;
    }

    std::vector<PcgPoint> FilterByHeight(const std::vector<PcgPoint>& points, float minHeight, float maxHeight) {
        std::vector<PcgPoint> result;
        result.reserve(points.size());
        for (const PcgPoint& point : points) {
            if (point.position.y >= minHeight && point.position.y <= maxHeight) {
                result.push_back(point);
            }
        }
        return result;
    }

    void ProjectOntoTerrain(std::vector<PcgPoint>& points, const PcgLandscapeData& terrain, float normalEpsilon) {
        for (PcgPoint& point : points) {
            // Re-sample at the point's CURRENT (x, z) -- x/z are left untouched by this function, so
            // this always queries the ground directly beneath (or above) wherever the point currently
            // sits horizontally, exactly the "snap back onto the surface after a horizontal move"
            // contract this function documents.
            const float freshHeight = SampleHeightCPU(terrain, point.position.x, point.position.z);
            const maths::vec3 freshNormal = ComputeTerrainNormalCPU(terrain, point.position.x, point.position.z, normalEpsilon);

            point.position.y = freshHeight;
            // QuatFromNormal is PcgTerrainSampler.h's own header-exposed "align local +Y to this
            // normal" utility (used by SampleTerrainPoints itself for exactly the same purpose) --
            // reused directly rather than re-derived, since it is already a plain free function, not
            // hidden inside an anonymous namespace.
            point.rotation = QuatFromNormal(freshNormal);
        }
    }

}
