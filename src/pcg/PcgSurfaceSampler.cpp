// PCG framework roadmap, Phase 2.1 ("Surface Sampler") implementation. See PcgSurfaceSampler.h's
// own top-of-file comment for the full algorithm description and scope boundary -- this file is
// deliberately just the mechanical implementation of what that header already documents.

#include "pcg/PcgSurfaceSampler.h"

#include "pcg/PcgSeededRandom.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pcg {

    float ComputeTriangleArea(const PcgSurfaceTriangle& triangle) {
        const maths::vec3 edge1 = triangle.positionB - triangle.positionA;
        const maths::vec3 edge2 = triangle.positionC - triangle.positionA;
        const maths::vec3 crossProduct = edge1.Cross(edge2);
        // 0.5 * |cross(edge1, edge2)| -- the standard triangle-area-from-two-edges formula.
        // vec3::Length() is a sqrt of a sum of squares, so this is always >= 0.0f and finite (never
        // NaN) for any finite input, including a fully degenerate (collinear or coincident-vertex)
        // triangle, which correctly yields exactly 0.0f.
        return 0.5f * crossProduct.Length();
    }

    maths::quat ComputeUpAlignedRotation(const maths::vec3& normal) {
        constexpr maths::vec3 kWorldUp{ 0.0f, 1.0f, 0.0f }; // This codebase's world-up convention (core/Camera.cpp's worldUp).

        const float lengthSq = normal.Dot(normal);
        if (lengthSq < 1.0e-12f) {
            return maths::quat{}; // Zero-length input: no meaningful direction, identity is the safe fallback.
        }
        const maths::vec3 n = normal.Normalize();
        const float d = kWorldUp.Dot(n);

        constexpr float kParallelEpsilon = 1.0e-6f;
        if (d > 1.0f - kParallelEpsilon) {
            return maths::quat{}; // Already aligned with +Y: identity, no rotation needed.
        }
        if (d < -1.0f + kParallelEpsilon) {
            // Exactly opposite +Y: the shortest-arc axis (worldUp x n) degenerates to zero, so any
            // axis perpendicular to worldUp works for the 180-degree flip. World +X is perpendicular
            // to worldUp by construction; it is never itself parallel to worldUp, so this cross
            // product can never degenerate.
            const maths::vec3 axis = kWorldUp.Cross(maths::vec3{ 1.0f, 0.0f, 0.0f }).Normalize();
            return maths::quat::FromAxisAngle(axis, maths::PI);
        }

        const maths::vec3 axis = kWorldUp.Cross(n).Normalize();
        const float angle = std::acos(std::clamp(d, -1.0f, 1.0f));
        return maths::quat::FromAxisAngle(axis, angle);
    }

    std::vector<PcgPoint> SampleSurfacePoints(const std::vector<PcgSurfaceTriangle>& triangles, const PcgSurfaceSamplerParams& params) {
        std::vector<PcgPoint> result;

        if (triangles.empty() || params.density <= 0.0f) {
            return result; // Nothing to sample from, or the caller explicitly asked for zero density.
        }

        // Running cumulative-area prefix sum, in input order -- the area-weighted triangle
        // selection distribution. A degenerate (zero-area) triangle contributes a zero-WIDTH slice
        // to this distribution, so std::upper_bound below has exactly zero probability of selecting
        // it for a continuous random draw -- degenerate triangles are excluded from selection "for
        // free", with no separate filtering pass needed.
        std::vector<float> cumulativeAreas;
        cumulativeAreas.reserve(triangles.size());
        float runningArea = 0.0f;
        for (const PcgSurfaceTriangle& triangle : triangles) {
            runningArea += ComputeTriangleArea(triangle);
            cumulativeAreas.push_back(runningArea);
        }
        const float totalArea = runningArea;

        constexpr float kMinTotalArea = 1.0e-9f; // Guards an all-degenerate (zero total area) triangle list.
        if (totalArea < kMinTotalArea) {
            return result;
        }

        // Every random draw this function makes descends from this ONE stream, constructed fresh
        // from params.seed -- this is what makes SampleSurfacePoints() deterministic: same
        // triangles + same params (this seed included) always replays the exact same draw sequence,
        // in the exact same order, producing byte-identical output.
        PcgSeededRandom stream(params.seed);

        // Expected point count = totalArea * density. The integer part is generated
        // unconditionally; the fractional remainder is resolved with ONE extra seeded coin-flip
        // (stream draw #1) so the point count's EXPECTATION over many different seeds matches
        // totalArea * density exactly, rather than always silently truncating downward.
        const float expectedCount = totalArea * params.density;
        size_t pointCount = static_cast<size_t>(std::floor(expectedCount));
        const float fractionalRemainder = expectedCount - static_cast<float>(pointCount);
        if (fractionalRemainder > 0.0f && stream.NextFloat01() < fractionalRemainder) {
            pointCount += 1;
        }

        result.reserve(pointCount);

        for (size_t i = 0; i < pointCount; ++i) {
            // Fixed, documented per-point draw order (determinism depends on never reordering
            // these six draws):
            //   1. triangleSelector in [0, totalArea)  -> area-weighted triangle pick (upper_bound).
            //   2. u in [0, 1)                          -> barycentric coordinate draw #1.
            //   3. v in [0, 1)                          -> barycentric coordinate draw #2.
            //   4. jitterU in [-jitterMagnitude, jitterMagnitude) -> tangent-plane jitter, axis 1.
            //   5. jitterV in [-jitterMagnitude, jitterMagnitude) -> tangent-plane jitter, axis 2.
            //   6. pointSeed via NextUint32()            -> this point's own output PcgPoint::seed
            //      (a FRESH stream-derived value, never the raw params.seed reused).
            const float triangleSelector = stream.NextFloatRange(0.0f, totalArea);
            auto upperBoundIt = std::upper_bound(cumulativeAreas.begin(), cumulativeAreas.end(), triangleSelector);
            size_t triangleIndex = static_cast<size_t>(std::distance(cumulativeAreas.begin(), upperBoundIt));
            if (triangleIndex >= triangles.size()) {
                // Extremely rare float-rounding edge: NextFloatRange's t is drawn from [0,1), so
                // triangleSelector should always land strictly below totalArea, but guard the exact
                // boundary case defensively rather than risk an out-of-bounds access.
                triangleIndex = triangles.size() - 1;
            }
            const PcgSurfaceTriangle& triangle = triangles[triangleIndex];

            float u = stream.NextFloat01();
            float v = stream.NextFloat01();
            if (u + v > 1.0f) {
                // Standard parallelogram-to-triangle fold: reflects a sample that landed in the
                // "wrong half" of the (u,v) unit square back into the triangle, preserving
                // uniformity across the triangle's area (not just its bounding parallelogram).
                u = 1.0f - u;
                v = 1.0f - v;
            }
            const float barycentricA = 1.0f - u - v;
            const float barycentricB = u;
            const float barycentricC = v;

            const maths::vec3 position = triangle.positionA * barycentricA + triangle.positionB * barycentricB + triangle.positionC * barycentricC;

            maths::vec3 normal;
            const maths::vec3 interpolatedNormal = triangle.normalA * barycentricA + triangle.normalB * barycentricB + triangle.normalC * barycentricC;
            const float normalLengthSq = interpolatedNormal.Dot(interpolatedNormal);
            if (normalLengthSq > 1.0e-12f) {
                normal = interpolatedNormal.Normalize();
            } else {
                // Degenerate/cancelling per-vertex normals (e.g. a caller passing all-zero normals):
                // fall back to the triangle's own geometric face normal so the output normal is
                // never a zero vector. Falls back further, to world +Y, only if the triangle itself
                // is degenerate too (should be unreachable here since such a triangle would have
                // contributed zero weight to selection above, but kept as a defensive final resort
                // so this function can never produce a NaN/zero-length normal).
                const maths::vec3 edge1 = triangle.positionB - triangle.positionA;
                const maths::vec3 edge2 = triangle.positionC - triangle.positionA;
                const maths::vec3 faceNormal = edge1.Cross(edge2);
                const float faceLengthSq = faceNormal.Dot(faceNormal);
                normal = (faceLengthSq > 1.0e-12f) ? faceNormal.Normalize() : maths::vec3{ 0.0f, 1.0f, 0.0f };
            }

            const float jitterMagnitude = std::abs(params.positionJitter);
            const float jitterU = stream.NextFloatRange(-jitterMagnitude, jitterMagnitude);
            const float jitterV = stream.NextFloatRange(-jitterMagnitude, jitterMagnitude);

            // Orthonormal tangent basis (tangentU, tangentV, normal) so the jitter offset stays IN
            // the surface's local tangent plane rather than lifting the point off the surface along
            // its own normal. Reference-axis switch (world +Y, or world +X when `normal` is itself
            // near-parallel to +Y) guards the cross product below from ever degenerating to
            // near-zero. When jitterMagnitude is 0.0f (the default, jitter disabled), jitterU and
            // jitterV are both exactly 0.0f, so this unconditionally reproduces `position` exactly
            // -- no separate "is jitter enabled" branch is needed.
            maths::vec3 referenceAxis{ 0.0f, 1.0f, 0.0f };
            if (std::abs(normal.Dot(referenceAxis)) > 0.999f) {
                referenceAxis = maths::vec3{ 1.0f, 0.0f, 0.0f };
            }
            const maths::vec3 tangentU = referenceAxis.Cross(normal).Normalize();
            const maths::vec3 tangentV = normal.Cross(tangentU);
            const maths::vec3 jitteredPosition = position + tangentU * jitterU + tangentV * jitterV;

            const uint32_t pointSeed = stream.NextUint32();

            PcgPoint point;
            point.position = jitteredPosition;
            point.rotation = ComputeUpAlignedRotation(normal);
            point.seed = pointSeed;
            // scale, density, color, boundsMin/boundsMax, steepness all keep PcgPoint's own default
            // field initializers (unit scale, density=1.0f, opaque white, unit bounds cube,
            // steepness=0.0f) -- this sampler does not author any of those, matching this phase's
            // documented scope (position/rotation/seed only).
            result.push_back(point);
        }

        return result;
    }

}
