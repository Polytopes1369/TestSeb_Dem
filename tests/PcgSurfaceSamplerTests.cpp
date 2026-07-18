// Standalone, framework-free unit test for the PCG framework roadmap's Phase 2.1 ("Surface
// Sampler") code: src/pcg/PcgSurfaceSampler.h/.cpp. Exercises triangle-area computation, the
// up-aligned rotation helper, the core SampleSurfacePoints() algorithm's determinism (the hard
// "same triangles + same params -> byte-identical output" requirement), its area-weighted triangle
// selection (statistical check), and its degenerate/empty/zero-density edge cases. Exits 0 if every
// check passes, non-zero otherwise -- registered with CTest (see the top-level CMakeLists.txt),
// mirroring tests/PcgDataModelTests.cpp's own convention exactly (same Check()/NearlyEqual()
// harness, same self-contained no-Logger/no-Vulkan-dependency style).

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSurfaceSampler.h"

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

    bool IsFinite(float v) {
        return std::isfinite(v);
    }

    // Builds a single flat, unit right triangle in the XY plane (z=0), A=(0,0,0), B=(1,0,0),
    // C=(0,1,0) -- area exactly 0.5, face normal exactly +Z. Per-vertex normals all set to the same
    // face normal (no authored smooth-shading variation) since these tests care about the sampler's
    // own algorithm, not about interpolated-normal shading fidelity.
    pcg::PcgSurfaceTriangle MakeUnitRightTriangle() {
        pcg::PcgSurfaceTriangle triangle;
        triangle.positionA = maths::vec3{ 0.0f, 0.0f, 0.0f };
        triangle.positionB = maths::vec3{ 1.0f, 0.0f, 0.0f };
        triangle.positionC = maths::vec3{ 0.0f, 1.0f, 0.0f };
        const maths::vec3 faceNormal{ 0.0f, 0.0f, 1.0f };
        triangle.normalA = faceNormal;
        triangle.normalB = faceNormal;
        triangle.normalC = faceNormal;
        return triangle;
    }

    void TestComputeTriangleArea() {
        const pcg::PcgSurfaceTriangle unitTriangle = MakeUnitRightTriangle();
        Check(NearlyEqual(pcg::ComputeTriangleArea(unitTriangle), 0.5f), "ComputeTriangleArea: unit right triangle has area 0.5");

        pcg::PcgSurfaceTriangle bigTriangle;
        bigTriangle.positionA = maths::vec3{ 10.0f, 0.0f, 0.0f };
        bigTriangle.positionB = maths::vec3{ 20.0f, 0.0f, 0.0f };
        bigTriangle.positionC = maths::vec3{ 10.0f, 1.0f, 0.0f };
        Check(NearlyEqual(pcg::ComputeTriangleArea(bigTriangle), 5.0f), "ComputeTriangleArea: 10x1 right triangle has area 5.0 (10x the unit triangle)");

        pcg::PcgSurfaceTriangle degenerateCollinear;
        degenerateCollinear.positionA = maths::vec3{ 0.0f, 0.0f, 0.0f };
        degenerateCollinear.positionB = maths::vec3{ 1.0f, 0.0f, 0.0f };
        degenerateCollinear.positionC = maths::vec3{ 2.0f, 0.0f, 0.0f };
        Check(NearlyEqual(pcg::ComputeTriangleArea(degenerateCollinear), 0.0f), "ComputeTriangleArea: collinear points -> exactly zero area");

        pcg::PcgSurfaceTriangle degenerateCoincident;
        degenerateCoincident.positionA = maths::vec3{ 3.0f, 3.0f, 3.0f };
        degenerateCoincident.positionB = maths::vec3{ 3.0f, 3.0f, 3.0f };
        degenerateCoincident.positionC = maths::vec3{ 3.0f, 3.0f, 3.0f };
        Check(NearlyEqual(pcg::ComputeTriangleArea(degenerateCoincident), 0.0f), "ComputeTriangleArea: coincident points -> exactly zero area");
        Check(IsFinite(pcg::ComputeTriangleArea(degenerateCoincident)), "ComputeTriangleArea: coincident points never produce NaN/Inf");
    }

    void TestComputeUpAlignedRotation() {
        const maths::quat identityForUp = pcg::ComputeUpAlignedRotation(maths::vec3{ 0.0f, 1.0f, 0.0f });
        const maths::vec3 rotatedUp = identityForUp.RotateVector(maths::vec3{ 0.0f, 1.0f, 0.0f });
        Check(NearlyEqual(rotatedUp.x, 0.0f) && NearlyEqual(rotatedUp.y, 1.0f) && NearlyEqual(rotatedUp.z, 0.0f),
            "ComputeUpAlignedRotation: normal == +Y maps local +Y back onto +Y (identity)");

        const maths::quat rotForSideways = pcg::ComputeUpAlignedRotation(maths::vec3{ 1.0f, 0.0f, 0.0f });
        const maths::vec3 rotatedSideways = rotForSideways.RotateVector(maths::vec3{ 0.0f, 1.0f, 0.0f });
        Check(NearlyEqual(rotatedSideways.x, 1.0f, 1.0e-3f) && NearlyEqual(rotatedSideways.y, 0.0f, 1.0e-3f) && NearlyEqual(rotatedSideways.z, 0.0f, 1.0e-3f),
            "ComputeUpAlignedRotation: normal == +X rotates local +Y onto +X");

        const maths::quat rotForDown = pcg::ComputeUpAlignedRotation(maths::vec3{ 0.0f, -1.0f, 0.0f });
        const maths::vec3 rotatedDown = rotForDown.RotateVector(maths::vec3{ 0.0f, 1.0f, 0.0f });
        Check(NearlyEqual(rotatedDown.x, 0.0f, 1.0e-3f) && NearlyEqual(rotatedDown.y, -1.0f, 1.0e-3f) && NearlyEqual(rotatedDown.z, 0.0f, 1.0e-3f),
            "ComputeUpAlignedRotation: normal == -Y (the 180-degree edge case) rotates local +Y onto -Y");

        const maths::quat rotForZero = pcg::ComputeUpAlignedRotation(maths::vec3{ 0.0f, 0.0f, 0.0f });
        Check(IsFinite(rotForZero.x) && IsFinite(rotForZero.y) && IsFinite(rotForZero.z) && IsFinite(rotForZero.w),
            "ComputeUpAlignedRotation: zero-length normal input never produces a NaN quaternion");
    }

    void TestDeterminism() {
        std::vector<pcg::PcgSurfaceTriangle> triangles;
        triangles.push_back(MakeUnitRightTriangle());
        pcg::PcgSurfaceTriangle secondTriangle = MakeUnitRightTriangle();
        secondTriangle.positionA = secondTriangle.positionA + maths::vec3{ 5.0f, 0.0f, 0.0f };
        secondTriangle.positionB = secondTriangle.positionB + maths::vec3{ 5.0f, 0.0f, 0.0f };
        secondTriangle.positionC = secondTriangle.positionC + maths::vec3{ 5.0f, 0.0f, 0.0f };
        triangles.push_back(secondTriangle);

        pcg::PcgSurfaceSamplerParams params;
        params.density = 50.0f;
        params.seed = 0xC0FFEEu;
        params.positionJitter = 0.05f;

        const std::vector<pcg::PcgPoint> runA = pcg::SampleSurfacePoints(triangles, params);
        const std::vector<pcg::PcgPoint> runB = pcg::SampleSurfacePoints(triangles, params);

        Check(!runA.empty(), "SampleSurfacePoints: a reasonable density over non-degenerate triangles produces at least one point");
        Check(runA.size() == runB.size(), "SampleSurfacePoints: two runs with the same triangles/params produce the same point COUNT");

        bool allFieldsMatch = (runA.size() == runB.size());
        if (allFieldsMatch) {
            for (size_t i = 0; i < runA.size(); ++i) {
                const pcg::PcgPoint& a = runA[i];
                const pcg::PcgPoint& b = runB[i];
                const bool positionMatches = (a.position.x == b.position.x) && (a.position.y == b.position.y) && (a.position.z == b.position.z);
                const bool rotationMatches = (a.rotation.x == b.rotation.x) && (a.rotation.y == b.rotation.y)
                    && (a.rotation.z == b.rotation.z) && (a.rotation.w == b.rotation.w);
                const bool seedMatches = (a.seed == b.seed);
                if (!positionMatches || !rotationMatches || !seedMatches) {
                    allFieldsMatch = false;
                    break;
                }
            }
        }
        Check(allFieldsMatch, "SampleSurfacePoints: two runs with the same triangles/params produce BYTE-IDENTICAL position/rotation/seed for every point");

        // A different seed must (overwhelmingly likely, for a nontrivial point count) produce a
        // different point set -- proves the params.seed is actually driving the stream, not being
        // ignored.
        pcg::PcgSurfaceSamplerParams differentSeedParams = params;
        differentSeedParams.seed = 0x12345678u;
        const std::vector<pcg::PcgPoint> runC = pcg::SampleSurfacePoints(triangles, differentSeedParams);
        bool anyDifferent = (runC.size() != runA.size());
        if (!anyDifferent) {
            for (size_t i = 0; i < runA.size(); ++i) {
                if (runA[i].position.x != runC[i].position.x || runA[i].position.y != runC[i].position.y || runA[i].position.z != runC[i].position.z) {
                    anyDifferent = true;
                    break;
                }
            }
        }
        Check(anyDifferent, "SampleSurfacePoints: a different seed produces a different point set");
    }

    void TestAreaWeighting() {
        // Two triangles, exact area ratio 10:1, placed far apart on the X axis so every sampled
        // point can be unambiguously attributed to its source triangle by position.x alone (small
        // triangle spans x in [0,1], big triangle spans x in [10,20]).
        pcg::PcgSurfaceTriangle smallTriangle = MakeUnitRightTriangle(); // area 0.5, x in [0,1]
        pcg::PcgSurfaceTriangle bigTriangle;
        bigTriangle.positionA = maths::vec3{ 10.0f, 0.0f, 0.0f };
        bigTriangle.positionB = maths::vec3{ 20.0f, 0.0f, 0.0f };
        bigTriangle.positionC = maths::vec3{ 10.0f, 1.0f, 0.0f }; // area 5.0 -- exactly 10x smallTriangle
        const maths::vec3 faceNormal{ 0.0f, 0.0f, 1.0f };
        bigTriangle.normalA = bigTriangle.normalB = bigTriangle.normalC = faceNormal;

        std::vector<pcg::PcgSurfaceTriangle> triangles{ smallTriangle, bigTriangle };

        pcg::PcgSurfaceSamplerParams params;
        params.density = 2000.0f; // total area 5.5 -> expected count ~11000, large enough to keep sampling noise small.
        params.seed = 0x5EED1234u;

        const std::vector<pcg::PcgPoint> points = pcg::SampleSurfacePoints(triangles, params);
        Check(points.size() > 5000, "SampleSurfacePoints (area-weighting setup): a high-density sample over 5.5 total area yields a large point count");

        size_t smallCount = 0, bigCount = 0;
        for (const pcg::PcgPoint& point : points) {
            if (point.position.x < 5.0f) {
                ++smallCount;
            } else {
                ++bigCount;
            }
        }

        Check(smallCount > 0 && bigCount > 0, "SampleSurfacePoints (area-weighting): both triangles receive at least some points");
        if (smallCount > 0) {
            const float ratio = static_cast<float>(bigCount) / static_cast<float>(smallCount);
            // Expected ratio is exactly 10.0 (area-proportional); allow generous +/-20% slack for
            // sampling noise at this point count (binomial std-dev at p=1/11, n~11000 is well under 1%,
            // so a 20% band is comfortably conservative, not a tuned-to-pass hack).
            Check(ratio > 8.0f && ratio < 12.0f,
                "SampleSurfacePoints (area-weighting): the 10x-area triangle receives roughly 10x the points of the 1x-area triangle (ratio="
                + std::to_string(ratio) + ")");
        }
    }

    void TestEdgeCases() {
        std::vector<pcg::PcgSurfaceTriangle> empty;
        pcg::PcgSurfaceSamplerParams defaultParams;
        defaultParams.density = 10.0f;
        defaultParams.seed = 1u;
        Check(pcg::SampleSurfacePoints(empty, defaultParams).empty(), "SampleSurfacePoints: an empty triangle list produces zero points");

        std::vector<pcg::PcgSurfaceTriangle> single{ MakeUnitRightTriangle() };

        pcg::PcgSurfaceSamplerParams zeroDensity = defaultParams;
        zeroDensity.density = 0.0f;
        Check(pcg::SampleSurfacePoints(single, zeroDensity).empty(), "SampleSurfacePoints: density == 0.0f produces zero points");

        pcg::PcgSurfaceSamplerParams negativeDensity = defaultParams;
        negativeDensity.density = -5.0f;
        Check(pcg::SampleSurfacePoints(single, negativeDensity).empty(), "SampleSurfacePoints: a negative density produces zero points (not a crash/UB)");

        // All-degenerate triangle list (every triangle has zero area): must not crash and must not
        // produce NaN points -- exercises the kMinTotalArea early-return guard.
        std::vector<pcg::PcgSurfaceTriangle> allDegenerate;
        pcg::PcgSurfaceTriangle degenerate;
        degenerate.positionA = degenerate.positionB = degenerate.positionC = maths::vec3{ 1.0f, 1.0f, 1.0f };
        allDegenerate.push_back(degenerate);
        allDegenerate.push_back(degenerate);
        Check(pcg::SampleSurfacePoints(allDegenerate, defaultParams).empty(), "SampleSurfacePoints: an all-degenerate (zero-area) triangle list produces zero points");

        // A degenerate triangle MIXED IN with a valid one must not crash/NaN, and (since the
        // degenerate triangle contributes zero weight to area-based selection) every produced point
        // should still land on the valid triangle.
        std::vector<pcg::PcgSurfaceTriangle> mixed;
        mixed.push_back(degenerate);
        mixed.push_back(MakeUnitRightTriangle());
        pcg::PcgSurfaceSamplerParams mixedParams = defaultParams;
        mixedParams.density = 200.0f;
        const std::vector<pcg::PcgPoint> mixedPoints = pcg::SampleSurfacePoints(mixed, mixedParams);
        Check(!mixedPoints.empty(), "SampleSurfacePoints: a valid triangle mixed with a degenerate one still produces points");
        bool anyNaNOrOutOfTriangle = false;
        for (const pcg::PcgPoint& point : mixedPoints) {
            if (!IsFinite(point.position.x) || !IsFinite(point.position.y) || !IsFinite(point.position.z)
                || !IsFinite(point.rotation.x) || !IsFinite(point.rotation.y) || !IsFinite(point.rotation.z) || !IsFinite(point.rotation.w)) {
                anyNaNOrOutOfTriangle = true;
                break;
            }
            // The unit right triangle (A=(0,0,0), B=(1,0,0), C=(0,1,0)) satisfies x>=0, y>=0,
            // x+y<=1, z==0 for every point strictly inside/on it.
            if (point.position.x < -1.0e-4f || point.position.y < -1.0e-4f || (point.position.x + point.position.y) > 1.0f + 1.0e-4f
                || std::abs(point.position.z) > 1.0e-4f) {
                anyNaNOrOutOfTriangle = true;
                break;
            }
        }
        Check(!anyNaNOrOutOfTriangle, "SampleSurfacePoints: degenerate triangle in the mix never produces a NaN point or a point off the valid triangle");

        // Single-triangle input works on its own (not just as part of a multi-triangle list).
        pcg::PcgSurfaceSamplerParams singleParams = defaultParams;
        singleParams.density = 100.0f;
        const std::vector<pcg::PcgPoint> singlePoints = pcg::SampleSurfacePoints(single, singleParams);
        Check(!singlePoints.empty(), "SampleSurfacePoints: single-triangle input (area 0.5, density 100) produces points");
        bool allOnTriangle = true;
        for (const pcg::PcgPoint& point : singlePoints) {
            if (point.position.x < -1.0e-4f || point.position.y < -1.0e-4f || (point.position.x + point.position.y) > 1.0f + 1.0e-4f) {
                allOnTriangle = false;
                break;
            }
        }
        Check(allOnTriangle, "SampleSurfacePoints: every point from a single-triangle sample lies within that triangle's barycentric bounds");

        // Every point's rotation should map local +Y onto the triangle's own face normal (+Z here).
        bool allNormalsAligned = true;
        for (const pcg::PcgPoint& point : singlePoints) {
            const maths::vec3 rotatedUp = point.rotation.RotateVector(maths::vec3{ 0.0f, 1.0f, 0.0f });
            if (!NearlyEqual(rotatedUp.x, 0.0f, 1.0e-3f) || !NearlyEqual(rotatedUp.y, 0.0f, 1.0e-3f) || !NearlyEqual(rotatedUp.z, 1.0f, 1.0e-3f)) {
                allNormalsAligned = false;
                break;
            }
        }
        Check(allNormalsAligned, "SampleSurfacePoints: every point's rotation aligns local +Y with the triangle's face normal (+Z)");

        // Default-constructed output fields (this sampler only authors position/rotation/seed).
        Check(NearlyEqual(singlePoints[0].scale.x, 1.0f) && NearlyEqual(singlePoints[0].scale.y, 1.0f) && NearlyEqual(singlePoints[0].scale.z, 1.0f),
            "SampleSurfacePoints: output points keep PcgPoint's default unit scale");
        Check(NearlyEqual(singlePoints[0].density, 1.0f), "SampleSurfacePoints: output points keep PcgPoint's default density (1.0f)");
    }

} // namespace

int main() {
    TestComputeTriangleArea();
    TestComputeUpAlignedRotation();
    TestDeterminism();
    TestAreaWeighting();
    TestEdgeCases();

    if (g_failCount == 0) {
        std::cout << "PcgSurfaceSamplerTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgSurfaceSamplerTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
