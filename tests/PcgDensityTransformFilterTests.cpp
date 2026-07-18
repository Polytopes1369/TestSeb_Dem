// Standalone, framework-free unit test for the PCG framework roadmap's Phase 3.1 ("Density Filter,
// Transform Filter, Attribute Noise/Remap") types: src/pcg/PcgDensityTransformFilter.h/.cpp.
// Exercises density-range filtering (including exact boundary inclusion), density remapping
// (including the degenerate all-identical-density case), transform jitter (position/rotation/scale,
// each independently toggleable, per-point-seed determinism), and the coherent value-noise function
// (determinism, [0,1] range, and "different positions differ / same position repeats" sanity).
// Exits 0 if every check passes, non-zero otherwise -- registered with CTest (see the top-level
// CMakeLists.txt), matching this project's existing tests/*.cpp convention (mirrors
// tests/PcgDataModelTests.cpp's own framework-free structure).

#include "core/maths/Maths.h"
#include "pcg/PcgDensityTransformFilter.h"
#include "pcg/PcgPointData.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    bool NearlyEqual(float a, float b, float epsilon = 1.0e-4f) {
        return std::abs(a - b) <= epsilon;
    }

    pcg::PcgPoint MakePoint(uint32_t seed, float density) {
        pcg::PcgPoint point;
        point.seed = seed;
        point.density = density;
        return point;
    }

    // -------------------------------------------------------------------------------------------
    // 1. Density Filter.
    // -------------------------------------------------------------------------------------------

    void TestFilterByDensity() {
        std::vector<pcg::PcgPoint> points;
        points.push_back(MakePoint(1u, 0.0f));
        points.push_back(MakePoint(2u, 0.3f));
        points.push_back(MakePoint(3u, 0.5f));
        points.push_back(MakePoint(4u, 0.7f));
        points.push_back(MakePoint(5u, 1.0f));

        const std::vector<pcg::PcgPoint> midRange = pcg::FilterByDensity(points, 0.3f, 0.7f);
        Check(midRange.size() == 3, "FilterByDensity: [0.3, 0.7] keeps exactly 3 of 5 points");
        if (midRange.size() == 3) {
            Check(midRange[0].seed == 2u && midRange[1].seed == 3u && midRange[2].seed == 4u,
                "FilterByDensity: kept points retain their original relative order");
        }

        // Exact boundary inclusion: minDensity/maxDensity are INCLUSIVE on both ends.
        const std::vector<pcg::PcgPoint> lowerBoundary = pcg::FilterByDensity(points, 0.3f, 0.3f);
        Check(lowerBoundary.size() == 1 && lowerBoundary[0].seed == 2u,
            "FilterByDensity: a point exactly AT minDensity==maxDensity is kept (inclusive boundary)");

        const std::vector<pcg::PcgPoint> exactEdges = pcg::FilterByDensity(points, 0.0f, 1.0f);
        Check(exactEdges.size() == 5, "FilterByDensity: [0.0, 1.0] keeps points exactly at both extremes (density 0.0 and 1.0)");

        const std::vector<pcg::PcgPoint> excludeAll = pcg::FilterByDensity(points, 0.31f, 0.49f);
        Check(excludeAll.empty(), "FilterByDensity: a range containing no point's density returns an empty vector");

        const std::vector<pcg::PcgPoint> emptyInput = pcg::FilterByDensity({}, 0.0f, 1.0f);
        Check(emptyInput.empty(), "FilterByDensity: an empty input returns an empty vector");
    }

    void TestRemapDensity() {
        std::vector<pcg::PcgPoint> points;
        points.push_back(MakePoint(1u, 0.2f));
        points.push_back(MakePoint(2u, 0.5f));
        points.push_back(MakePoint(3u, 0.8f));

        const std::vector<pcg::PcgPoint> remappedToUnit = pcg::RemapDensity(points, 0.0f, 1.0f);
        Check(remappedToUnit.size() == 3, "RemapDensity: output size matches input size");
        Check(NearlyEqual(remappedToUnit[0].density, 0.0f), "RemapDensity: old-range minimum maps exactly to newMin");
        Check(NearlyEqual(remappedToUnit[1].density, 0.5f), "RemapDensity: old-range midpoint maps to the new-range midpoint");
        Check(NearlyEqual(remappedToUnit[2].density, 1.0f), "RemapDensity: old-range maximum maps exactly to newMax");

        const std::vector<pcg::PcgPoint> remappedToTenTwenty = pcg::RemapDensity(points, 10.0f, 20.0f);
        Check(NearlyEqual(remappedToTenTwenty[0].density, 10.0f) && NearlyEqual(remappedToTenTwenty[1].density, 15.0f) && NearlyEqual(remappedToTenTwenty[2].density, 20.0f),
            "RemapDensity: an arbitrary [10, 20] target range is honored exactly");

        // Input `points` must be untouched (RemapDensity returns a copy, per its own documented convention).
        Check(NearlyEqual(points[0].density, 0.2f) && NearlyEqual(points[1].density, 0.5f) && NearlyEqual(points[2].density, 0.8f),
            "RemapDensity: does not mutate its input vector");

        // Degenerate case: every input density identical -> collapses to newMin, no divide-by-zero/NaN.
        std::vector<pcg::PcgPoint> identicalDensity;
        identicalDensity.push_back(MakePoint(10u, 0.5f));
        identicalDensity.push_back(MakePoint(11u, 0.5f));
        identicalDensity.push_back(MakePoint(12u, 0.5f));
        const std::vector<pcg::PcgPoint> collapsed = pcg::RemapDensity(identicalDensity, 2.0f, 5.0f);
        for (const pcg::PcgPoint& point : collapsed) {
            Check(NearlyEqual(point.density, 2.0f), "RemapDensity: degenerate all-identical-density input collapses to newMin (no NaN/divide-by-zero)");
        }

        const std::vector<pcg::PcgPoint> emptyResult = pcg::RemapDensity({}, 0.0f, 1.0f);
        Check(emptyResult.empty(), "RemapDensity: an empty input returns an empty vector");
    }

    // -------------------------------------------------------------------------------------------
    // 2. Transform Filter.
    // -------------------------------------------------------------------------------------------

    void TestTransformJitterNoOp() {
        pcg::PcgPoint original;
        original.seed = 777u;
        original.position = maths::vec3{ 1.0f, 2.0f, 3.0f };
        original.rotation = maths::quat{};
        original.scale = maths::vec3{ 1.0f, 1.0f, 1.0f };

        std::vector<pcg::PcgPoint> points{ original };
        pcg::PcgTransformJitterParams noOpParams; // All jitter* bools default to false.
        pcg::ApplyTransformJitter(points, noOpParams, 12345u);

        Check(NearlyEqual(points[0].position.x, original.position.x) && NearlyEqual(points[0].position.y, original.position.y) && NearlyEqual(points[0].position.z, original.position.z),
            "ApplyTransformJitter: default (all-disabled) params leave position exactly unchanged");
        Check(NearlyEqual(points[0].rotation.w, 1.0f) && NearlyEqual(points[0].rotation.x, 0.0f),
            "ApplyTransformJitter: default (all-disabled) params leave rotation exactly unchanged");
        Check(NearlyEqual(points[0].scale.x, 1.0f) && NearlyEqual(points[0].scale.y, 1.0f) && NearlyEqual(points[0].scale.z, 1.0f),
            "ApplyTransformJitter: default (all-disabled) params leave scale exactly unchanged");
    }

    void TestTransformJitterPositionOnly() {
        pcg::PcgPoint point;
        point.seed = 42u;
        point.position = maths::vec3{ 0.0f, 0.0f, 0.0f };
        point.rotation = maths::quat{};
        point.scale = maths::vec3{ 1.0f, 1.0f, 1.0f };

        pcg::PcgTransformJitterParams params;
        params.jitterPosition = true;
        params.positionRange = maths::vec3{ 1.0f, 2.0f, 3.0f };

        std::vector<pcg::PcgPoint> points{ point };
        pcg::ApplyTransformJitter(points, params, 999u);

        Check(std::abs(points[0].position.x) <= 1.0f + 1.0e-4f, "ApplyTransformJitter (position only): X offset stays within +/-positionRange.x");
        Check(std::abs(points[0].position.y) <= 2.0f + 1.0e-4f, "ApplyTransformJitter (position only): Y offset stays within +/-positionRange.y");
        Check(std::abs(points[0].position.z) <= 3.0f + 1.0e-4f, "ApplyTransformJitter (position only): Z offset stays within +/-positionRange.z");

        // Rotation/scale must be untouched -- ONLY jitterPosition was enabled.
        Check(NearlyEqual(points[0].rotation.w, 1.0f), "ApplyTransformJitter (position only): rotation is untouched when jitterRotation is disabled");
        Check(NearlyEqual(points[0].scale.x, 1.0f) && NearlyEqual(points[0].scale.y, 1.0f) && NearlyEqual(points[0].scale.z, 1.0f),
            "ApplyTransformJitter (position only): scale is untouched when jitterScale is disabled");

        // A genuinely nonzero range with a nonzero seed should actually move the point (not land on
        // exactly zero by chance -- vanishingly unlikely with this hash, treated as a real failure
        // if it ever happens, same convention PcgDataModelTests.cpp's own RNG tests already use).
        const bool actuallyMoved = std::abs(points[0].position.x) > 1.0e-6f || std::abs(points[0].position.y) > 1.0e-6f || std::abs(points[0].position.z) > 1.0e-6f;
        Check(actuallyMoved, "ApplyTransformJitter (position only): position actually changes from its original (0,0,0)");
    }

    void TestTransformJitterRotationOnly() {
        pcg::PcgPoint point;
        point.seed = 123u;
        point.position = maths::vec3{ 5.0f, 5.0f, 5.0f };
        point.rotation = maths::quat{}; // Identity.
        point.scale = maths::vec3{ 1.0f, 1.0f, 1.0f };

        pcg::PcgTransformJitterParams params;
        params.jitterRotation = true;
        params.rotationRangeRadians = 0.6f;
        params.rotationAxis = maths::vec3{ 0.0f, 1.0f, 0.0f };

        std::vector<pcg::PcgPoint> points{ point };
        pcg::ApplyTransformJitter(points, params, 555u);

        // Position/scale untouched -- ONLY jitterRotation was enabled.
        Check(NearlyEqual(points[0].position.x, 5.0f) && NearlyEqual(points[0].position.y, 5.0f) && NearlyEqual(points[0].position.z, 5.0f),
            "ApplyTransformJitter (rotation only): position is untouched when jitterPosition is disabled");
        Check(NearlyEqual(points[0].scale.x, 1.0f), "ApplyTransformJitter (rotation only): scale is untouched when jitterScale is disabled");

        // Recover the rotation angle magnitude from the resulting quaternion (w = cos(angle/2)) and
        // check it stays within the requested +/-rotationRangeRadians bound. Starting rotation was
        // identity, so the composed result IS exactly the jitter rotation itself.
        const float recoveredHalfAngle = std::acos(std::clamp(points[0].rotation.w, -1.0f, 1.0f));
        const float recoveredAngle = 2.0f * recoveredHalfAngle;
        Check(recoveredAngle <= params.rotationRangeRadians + 1.0e-3f, "ApplyTransformJitter (rotation only): rotated angle magnitude stays within +/-rotationRangeRadians");

        const bool actuallyRotated = std::abs(points[0].rotation.w - 1.0f) > 1.0e-6f;
        Check(actuallyRotated, "ApplyTransformJitter (rotation only): rotation actually changes from identity");
    }

    void TestTransformJitterRotationZeroAxisIsNoOp() {
        pcg::PcgPoint point;
        point.seed = 8u;
        point.rotation = maths::quat{};

        pcg::PcgTransformJitterParams params;
        params.jitterRotation = true;
        params.rotationRangeRadians = 1.0f;
        params.rotationAxis = maths::vec3{ 0.0f, 0.0f, 0.0f }; // Zero-length axis.

        std::vector<pcg::PcgPoint> points{ point };
        pcg::ApplyTransformJitter(points, params, 1u);

        Check(NearlyEqual(points[0].rotation.w, 1.0f) && NearlyEqual(points[0].rotation.x, 0.0f) && NearlyEqual(points[0].rotation.y, 0.0f) && NearlyEqual(points[0].rotation.z, 0.0f),
            "ApplyTransformJitter: a zero-length rotationAxis leaves rotation exactly unchanged (treated as jitterRotation=false)");
    }

    void TestTransformJitterScaleUniform() {
        pcg::PcgPoint point;
        point.seed = 2024u;
        point.scale = maths::vec3{ 1.0f, 1.0f, 1.0f };

        pcg::PcgTransformJitterParams params;
        params.jitterScale = true;
        params.uniformScale = true;
        params.uniformScaleRange = maths::vec2{ 0.5f, 2.0f };

        std::vector<pcg::PcgPoint> points{ point };
        pcg::ApplyTransformJitter(points, params, 3u);

        Check(points[0].scale.x >= 0.5f - 1.0e-4f && points[0].scale.x <= 2.0f + 1.0e-4f, "ApplyTransformJitter (uniform scale): result stays within [0.5, 2.0]");
        Check(NearlyEqual(points[0].scale.x, points[0].scale.y) && NearlyEqual(points[0].scale.y, points[0].scale.z),
            "ApplyTransformJitter (uniform scale): the SAME multiplier is applied to all 3 axes");
    }

    void TestTransformJitterScalePerAxis() {
        pcg::PcgPoint point;
        point.seed = 31337u;
        point.scale = maths::vec3{ 1.0f, 1.0f, 1.0f };

        pcg::PcgTransformJitterParams params;
        params.jitterScale = true;
        params.uniformScale = false;
        params.perAxisScaleRangeMin = maths::vec3{ 0.5f, 1.0f, 2.0f };
        params.perAxisScaleRangeMax = maths::vec3{ 0.5f, 1.0f, 2.0f }; // Fixed (zero-width) ranges -> exact expected values.

        std::vector<pcg::PcgPoint> points{ point };
        pcg::ApplyTransformJitter(points, params, 4u);

        Check(NearlyEqual(points[0].scale.x, 0.5f), "ApplyTransformJitter (per-axis scale): X multiplier matches its own fixed range");
        Check(NearlyEqual(points[0].scale.y, 1.0f), "ApplyTransformJitter (per-axis scale): Y multiplier matches its own fixed range");
        Check(NearlyEqual(points[0].scale.z, 2.0f), "ApplyTransformJitter (per-axis scale): Z multiplier matches its own fixed range");
    }

    void TestTransformJitterDeterminism() {
        pcg::PcgTransformJitterParams params;
        params.jitterPosition = true;
        params.positionRange = maths::vec3{ 2.0f, 2.0f, 2.0f };
        params.jitterRotation = true;
        params.rotationRangeRadians = 0.5f;
        params.jitterScale = true;
        params.uniformScaleRange = maths::vec2{ 0.8f, 1.2f };

        pcg::PcgPoint basePoint;
        basePoint.seed = 0xBEEFu;
        basePoint.position = maths::vec3{ 10.0f, 20.0f, 30.0f };

        std::vector<pcg::PcgPoint> runA{ basePoint };
        std::vector<pcg::PcgPoint> runB{ basePoint };
        pcg::ApplyTransformJitter(runA, params, 0xC0FFEEu);
        pcg::ApplyTransformJitter(runB, params, 0xC0FFEEu);

        Check(NearlyEqual(runA[0].position.x, runB[0].position.x) && NearlyEqual(runA[0].position.y, runB[0].position.y) && NearlyEqual(runA[0].position.z, runB[0].position.z),
            "ApplyTransformJitter: identical point.seed + identical call seed reproduces byte-identical position");
        Check(NearlyEqual(runA[0].rotation.w, runB[0].rotation.w) && NearlyEqual(runA[0].rotation.x, runB[0].rotation.x),
            "ApplyTransformJitter: identical point.seed + identical call seed reproduces byte-identical rotation");
        Check(NearlyEqual(runA[0].scale.x, runB[0].scale.x), "ApplyTransformJitter: identical point.seed + identical call seed reproduces byte-identical scale");

        // A DIFFERENT point.seed (everything else identical) must produce different jitter output --
        // proves the per-point seed genuinely drives the result (not the call-level `seed` alone).
        pcg::PcgPoint otherPoint = basePoint;
        otherPoint.seed = 0xDEAD0001u;
        std::vector<pcg::PcgPoint> runC{ otherPoint };
        pcg::ApplyTransformJitter(runC, params, 0xC0FFEEu);
        const bool differsFromA = !NearlyEqual(runC[0].position.x, runA[0].position.x, 1.0e-6f) || !NearlyEqual(runC[0].position.y, runA[0].position.y, 1.0e-6f) || !NearlyEqual(runC[0].position.z, runA[0].position.z, 1.0e-6f);
        Check(differsFromA, "ApplyTransformJitter: a different point.seed (same call seed) produces different jitter output");

        // A DIFFERENT call-level seed (same point.seed) must also produce different jitter output --
        // proves the `seed` parameter is actually mixed in, not ignored.
        std::vector<pcg::PcgPoint> runD{ basePoint };
        pcg::ApplyTransformJitter(runD, params, 0x1234u);
        const bool differsWithDifferentCallSeed = !NearlyEqual(runD[0].position.x, runA[0].position.x, 1.0e-6f) || !NearlyEqual(runD[0].position.y, runA[0].position.y, 1.0e-6f) || !NearlyEqual(runD[0].position.z, runA[0].position.z, 1.0e-6f);
        Check(differsWithDifferentCallSeed, "ApplyTransformJitter: a different call-level seed (same point.seed) produces different jitter output");
    }

    // -------------------------------------------------------------------------------------------
    // 3. Attribute Noise / Remap.
    // -------------------------------------------------------------------------------------------

    void TestSampleAttributeNoiseDeterminism() {
        const maths::vec3 samplePos{ 3.25f, -1.5f, 7.0f };
        const float valueA = pcg::SampleAttributeNoise(samplePos, 0.1f, 42u);
        const float valueB = pcg::SampleAttributeNoise(samplePos, 0.1f, 42u);
        Check(valueA == valueB, "SampleAttributeNoise: identical (worldPos, frequency, seed) reproduces the EXACT same output on repeated calls");

        Check(valueA >= 0.0f && valueA <= 1.0f, "SampleAttributeNoise: output stays within [0, 1]");

        // Different world positions -> different values (checked across several widely-separated
        // sample points to make an accidental collision vanishingly unlikely).
        const float atOrigin = pcg::SampleAttributeNoise(maths::vec3{ 0.0f, 0.0f, 0.0f }, 0.2f, 7u);
        const float farAway = pcg::SampleAttributeNoise(maths::vec3{ 137.0f, -42.0f, 91.0f }, 0.2f, 7u);
        Check(atOrigin != farAway, "SampleAttributeNoise: sufficiently different world positions produce different noise values");

        // Repeating the SAME position gives the SAME value again (the other half of the "same
        // position always gets the same value" sanity check).
        const float atOriginAgain = pcg::SampleAttributeNoise(maths::vec3{ 0.0f, 0.0f, 0.0f }, 0.2f, 7u);
        Check(atOrigin == atOriginAgain, "SampleAttributeNoise: re-sampling the exact same world position always returns the exact same value");

        // A different seed at the SAME position should (in practice, with this hash) produce a
        // different value too -- proves `seed` genuinely participates in the hash, not just worldPos.
        const float differentSeed = pcg::SampleAttributeNoise(maths::vec3{ 0.0f, 0.0f, 0.0f }, 0.2f, 999u);
        Check(atOrigin != differentSeed, "SampleAttributeNoise: a different seed at the same world position produces a different noise value");

        // Sample many scattered points and confirm every single one stays in [0, 1] -- a broader
        // range sweep than the single spot-check above.
        bool allInRange = true;
        for (int i = 0; i < 200; ++i) {
            const maths::vec3 pos{ static_cast<float>(i) * 0.37f, static_cast<float>(i) * -1.13f, static_cast<float>(i) * 2.71f };
            const float v = pcg::SampleAttributeNoise(pos, 0.05f, 100u + static_cast<uint32_t>(i));
            if (v < 0.0f || v > 1.0f) {
                allInRange = false;
                break;
            }
        }
        Check(allInRange, "SampleAttributeNoise: stays within [0, 1] across 200 scattered (worldPos, seed) samples");
    }

    void TestApplyNoiseToDensity() {
        std::vector<pcg::PcgPoint> points;
        pcg::PcgPoint a; a.seed = 1u; a.position = maths::vec3{ 0.0f, 0.0f, 0.0f }; a.density = 1.0f;
        pcg::PcgPoint b; b.seed = 2u; b.position = maths::vec3{ 50.0f, 12.0f, -8.0f }; b.density = 1.0f;
        points.push_back(a);
        points.push_back(b);

        std::vector<pcg::PcgPoint> runA = points;
        std::vector<pcg::PcgPoint> runB = points;
        pcg::ApplyNoiseToDensity(runA, 0.15f, 2024u);
        pcg::ApplyNoiseToDensity(runB, 0.15f, 2024u);

        Check(NearlyEqual(runA[0].density, runB[0].density) && NearlyEqual(runA[1].density, runB[1].density),
            "ApplyNoiseToDensity: identical (points, frequency, seed) reproduces byte-identical density output");

        for (const pcg::PcgPoint& point : runA) {
            Check(point.density >= 0.0f && point.density <= 1.0f, "ApplyNoiseToDensity: resulting density stays clamped within [0, 1]");
            Check(point.density <= 1.0f + 1.0e-6f, "ApplyNoiseToDensity: resulting density never exceeds the original density (noise multiplier is <= 1)");
        }
    }

} // namespace

int main() {
    TestFilterByDensity();
    TestRemapDensity();
    TestTransformJitterNoOp();
    TestTransformJitterPositionOnly();
    TestTransformJitterRotationOnly();
    TestTransformJitterRotationZeroAxisIsNoOp();
    TestTransformJitterScaleUniform();
    TestTransformJitterScalePerAxis();
    TestTransformJitterDeterminism();
    TestSampleAttributeNoiseDeterminism();
    TestApplyNoiseToDensity();

    if (g_failCount == 0) {
        std::cout << "PcgDensityTransformFilterTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgDensityTransformFilterTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
