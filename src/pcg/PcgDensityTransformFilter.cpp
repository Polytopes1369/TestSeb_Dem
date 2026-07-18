// PCG framework roadmap, Phase 3.1 ("Density Filter, Transform Filter, Attribute Noise/Remap")
// implementation. See PcgDensityTransformFilter.h's own top-of-file comment for the full algorithm
// descriptions, determinism guarantees, and per-function scope -- this file is deliberately just
// the mechanical implementation of what that header already documents.

#include "pcg/PcgDensityTransformFilter.h"

#include "pcg/PcgSeededRandom.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pcg {

    namespace {

        // Hamilton (quaternion) product a*b: rotating a vector by the result is equivalent to
        // rotating it by `b` FIRST, then by `a` (i.e. ComposeQuaternions(a,b).RotateVector(v) ==
        // a.RotateVector(b.RotateVector(v))) -- the standard quaternion-composition convention,
        // consistent with quat::RotateVector's own "v + 2w(qv x v) + 2(qv x (qv x v))" sandwich
        // formula in core/maths/Maths.h. Kept local to this file (a private helper) rather than
        // added to the shared Maths.h: no other Phase 3 filter needs a general quaternion multiply
        // yet, and this codebase's quat type has otherwise only ever needed FromAxisAngle/
        // RotateVector (see Maths.h's own quat-forward-declaration comment) -- adding a shared
        // operator* is a reasonable FUTURE addition once a second caller actually needs it, not
        // something this scoped phase should take on unilaterally.
        maths::quat ComposeQuaternions(const maths::quat& a, const maths::quat& b) {
            return maths::quat{
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
            };
        }

        // Hermite smoothstep easing curve (3t^2 - 2t^3), the standard "value noise" interpolant --
        // zero first-derivative at both t=0 and t=1, so adjacent lattice cells blend with no visible
        // gradient discontinuity at cell boundaries (a plain linear `t` interpolant would produce a
        // visible faceted/creased look at every integer lattice line). `t` is always fed a value in
        // [0,1) by SampleAttributeNoise below (a fractional lattice-cell coordinate), so no
        // clamping is needed here.
        constexpr float SmoothStep(float t) {
            return t * t * (3.0f - 2.0f * t);
        }

        // Hashes one integer lattice corner (ix, iy, iz) plus `seed` into a single [0,1) value.
        // Chains PcgHashCombine three times (once per axis) rather than trying to pack all three
        // integers into a single 32-bit combine call -- each chained call re-mixes the FULL 32-bit
        // state through PcgHash32's own avalanche step (see PcgSeededRandom.h), so this reuses this
        // codebase's EXISTING, already-proven hash primitives exactly as-is with no new mixing
        // constants of its own to reason about. The final PcgRandomFloat01(h, 0u) call performs the
        // identical "hash then divide by 2^32" conversion PcgSeededRandom's own NextFloat01() uses.
        // Casting the (possibly negative) lattice coordinates through uint32_t is well-defined
        // (2's-complement wraparound is mandatory as of C++20) and simply folds negative lattice
        // indices into a different-but-still-well-mixed hash input, so there is no discontinuity or
        // special-case needed at worldPos == 0.
        float HashLatticeCorner01(int32_t ix, int32_t iy, int32_t iz, uint32_t seed) {
            uint32_t h = PcgHashCombine(seed, static_cast<uint32_t>(ix));
            h = PcgHashCombine(h, static_cast<uint32_t>(iy));
            h = PcgHashCombine(h, static_cast<uint32_t>(iz));
            return PcgRandomFloat01(h, 0u);
        }

    } // namespace

    // -------------------------------------------------------------------------------------------
    // 1. Density Filter.
    // -------------------------------------------------------------------------------------------

    std::vector<PcgPoint> FilterByDensity(const std::vector<PcgPoint>& points, float minDensity, float maxDensity) {
        std::vector<PcgPoint> result;
        result.reserve(points.size()); // Worst case (every point kept) -- avoids repeated reallocation.

        for (const PcgPoint& point : points) {
            if (point.density >= minDensity && point.density <= maxDensity) {
                result.push_back(point);
            }
        }
        return result;
    }

    std::vector<PcgPoint> RemapDensity(const std::vector<PcgPoint>& points, float newMin, float newMax) {
        std::vector<PcgPoint> result = points; // Start from a full copy -- every field except density is untouched.

        if (result.empty()) {
            return result;
        }

        // First pass: find the ACTUAL min/max density across the input (the "old range" being
        // remapped FROM) -- this is a genuine data-driven range, not assumed to already be [0,1],
        // since an upstream filter/noise pass may have already shifted it elsewhere.
        float oldMin = result.front().density;
        float oldMax = result.front().density;
        for (const PcgPoint& point : result) {
            oldMin = std::min(oldMin, point.density);
            oldMax = std::max(oldMax, point.density);
        }

        const float oldRange = oldMax - oldMin;
        constexpr float kMinRange = 1.0e-8f; // Guards a degenerate (all-identical-density) input from a divide-by-zero.
        if (oldRange < kMinRange) {
            // Every input density is (effectively) identical -- there is no meaningful "position
            // within the range" to remap, so every output point collapses to newMin rather than
            // dividing by ~zero (an arbitrary but well-defined and documented choice, see this
            // function's own header comment).
            for (PcgPoint& point : result) {
                point.density = newMin;
            }
            return result;
        }

        // Second pass: linear remap t = (density - oldMin) / oldRange, then density' = newMin + t *
        // (newMax - newMin) -- standard range-to-range linear interpolation.
        const float newRange = newMax - newMin;
        for (PcgPoint& point : result) {
            const float t = (point.density - oldMin) / oldRange;
            point.density = newMin + t * newRange;
        }
        return result;
    }

    // -------------------------------------------------------------------------------------------
    // 2. Transform Filter.
    // -------------------------------------------------------------------------------------------

    void ApplyTransformJitter(std::vector<PcgPoint>& points, const PcgTransformJitterParams& params, uint32_t seed) {
        if (!params.jitterPosition && !params.jitterRotation && !params.jitterScale) {
            return; // Fast no-op path -- matches this function's own documented "every axis off means zero mutation" contract without even constructing a stream per point.
        }

        for (PcgPoint& point : points) {
            // Per-point stream: this point's OWN seed combined with the call-level `seed` salt (see
            // this function's own header comment for the determinism rationale) -- NEVER a single
            // stream shared across the whole `points` vector, so jitter output for a given point
            // does not depend on the vector's size or this point's index within it.
            PcgSeededRandom stream(PcgHashCombine(point.seed, seed));

            if (params.jitterPosition) {
                const float rangeX = std::abs(params.positionRange.x);
                const float rangeY = std::abs(params.positionRange.y);
                const float rangeZ = std::abs(params.positionRange.z);
                const float dx = stream.NextFloatRange(-rangeX, rangeX);
                const float dy = stream.NextFloatRange(-rangeY, rangeY);
                const float dz = stream.NextFloatRange(-rangeZ, rangeZ);
                point.position = point.position + maths::vec3{ dx, dy, dz };
            }

            if (params.jitterRotation) {
                const float axisLengthSq = params.rotationAxis.Dot(params.rotationAxis);
                if (axisLengthSq > 1.0e-12f) {
                    const maths::vec3 axis = params.rotationAxis.Normalize();
                    const float range = std::abs(params.rotationRangeRadians);
                    const float angle = stream.NextFloatRange(-range, range);
                    const maths::quat jitterRotation = maths::quat::FromAxisAngle(axis, angle);
                    // Composed AFTER the point's existing rotation (jitterRotation is the OUTER/left
                    // operand) -- see this file's own ComposeQuaternions comment for the exact
                    // convention: the point's base orientation is applied first, then the random
                    // jitter rotates that already-oriented point further, both expressed in WORLD
                    // space (rotationAxis is a world-space axis, not a point-local one).
                    point.rotation = ComposeQuaternions(jitterRotation, point.rotation);
                }
                // Zero-length rotationAxis: intentionally consumes NO random draw and mutates
                // nothing (see this function's header comment -- "no meaningful axis" is treated
                // identically to jitterRotation=false).
            }

            if (params.jitterScale) {
                if (params.uniformScale) {
                    const float multiplier = stream.NextFloatRange(params.uniformScaleRange.x, params.uniformScaleRange.y);
                    point.scale = point.scale * multiplier;
                } else {
                    const float mx = stream.NextFloatRange(params.perAxisScaleRangeMin.x, params.perAxisScaleRangeMax.x);
                    const float my = stream.NextFloatRange(params.perAxisScaleRangeMin.y, params.perAxisScaleRangeMax.y);
                    const float mz = stream.NextFloatRange(params.perAxisScaleRangeMin.z, params.perAxisScaleRangeMax.z);
                    point.scale = maths::vec3{ point.scale.x * mx, point.scale.y * my, point.scale.z * mz };
                }
            }
        }
    }

    // -------------------------------------------------------------------------------------------
    // 3. Attribute Noise / Remap.
    // -------------------------------------------------------------------------------------------

    float SampleAttributeNoise(const maths::vec3& worldPos, float frequency, uint32_t seed) {
        const maths::vec3 samplePos = worldPos * frequency;

        // Integer lattice cell containing samplePos (the cell's "min corner"), plus the fractional
        // offset within that cell -- std::floor (not a plain int truncation) is required so negative
        // coordinates floor DOWNWARD (e.g. floor(-0.3) == -1.0, not 0.0), giving a continuous lattice
        // with no discontinuity or mirrored cell straddling worldPos == 0.
        const float floorX = std::floor(samplePos.x);
        const float floorY = std::floor(samplePos.y);
        const float floorZ = std::floor(samplePos.z);
        const int32_t ix0 = static_cast<int32_t>(floorX);
        const int32_t iy0 = static_cast<int32_t>(floorY);
        const int32_t iz0 = static_cast<int32_t>(floorZ);
        const int32_t ix1 = ix0 + 1;
        const int32_t iy1 = iy0 + 1;
        const int32_t iz1 = iz0 + 1;

        const float tx = SmoothStep(samplePos.x - floorX);
        const float ty = SmoothStep(samplePos.y - floorY);
        const float tz = SmoothStep(samplePos.z - floorZ);

        // The 8 corners of the unit lattice cell containing samplePos, each an independent [0,1)
        // hashed value (naming: c<xBit><yBit><zBit>, 0 = the cell's min corner on that axis, 1 = its
        // max corner).
        const float c000 = HashLatticeCorner01(ix0, iy0, iz0, seed);
        const float c100 = HashLatticeCorner01(ix1, iy0, iz0, seed);
        const float c010 = HashLatticeCorner01(ix0, iy1, iz0, seed);
        const float c110 = HashLatticeCorner01(ix1, iy1, iz0, seed);
        const float c001 = HashLatticeCorner01(ix0, iy0, iz1, seed);
        const float c101 = HashLatticeCorner01(ix1, iy0, iz1, seed);
        const float c011 = HashLatticeCorner01(ix0, iy1, iz1, seed);
        const float c111 = HashLatticeCorner01(ix1, iy1, iz1, seed);

        // Standard trilinear blend: interpolate along X first (4 edges -> 4 intermediate values),
        // then Y (those 4 -> 2), then Z (those 2 -> the final scalar).
        const float x00 = c000 + (c100 - c000) * tx;
        const float x10 = c010 + (c110 - c010) * tx;
        const float x01 = c001 + (c101 - c001) * tx;
        const float x11 = c011 + (c111 - c011) * tx;

        const float y0 = x00 + (x10 - x00) * ty;
        const float y1 = x01 + (x11 - x01) * ty;

        const float value = y0 + (y1 - y0) * tz;
        // Each corner is already in [0,1) and trilinear interpolation is a convex combination of
        // those 8 corners, so `value` is mathematically already within [0,1) -- the clamp below is
        // a defensive belt-and-braces guard against float rounding only, not a sign that the math
        // can otherwise escape that range.
        return std::clamp(value, 0.0f, 1.0f);
    }

    void ApplyNoiseToDensity(std::vector<PcgPoint>& points, float frequency, uint32_t seed) {
        for (PcgPoint& point : points) {
            const float noise = SampleAttributeNoise(point.position, frequency, seed);
            point.density = std::clamp(point.density * noise, 0.0f, 1.0f);
        }
    }

}
