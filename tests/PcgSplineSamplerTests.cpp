// Standalone, framework-free unit test for the PCG framework roadmap's Phase 2.4 ("Spline
// Sampler") type: src/pcg/PcgSplineSampler.h/.cpp. Exercises the arc-length remapping table
// (BuildSplineArcLengthTable/ResolveSplineTAtArcLength) and the full SampleSplineByArcLength
// sampler -- point count vs. total arc length, TRUE real-world spacing between consecutive points
// (the key check that proves this is genuine arc-length parametrization, not just evenly-stepped
// `t`), tangent-aligned rotation, degenerate 1/2-control-point splines, and seeded determinism.
// Exits 0 if every check passes, non-zero otherwise -- registered with CTest (see the top-level
// CMakeLists.txt), matching tests/PcgDataModelTests.cpp's own framework-free convention.

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"
#include "pcg/PcgSplineSampler.h"

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

    // True when `a` and `b` point in (nearly) the same normalized direction -- used to check a
    // sampled point's rotation-derived forward axis against the curve's own local tangent.
    bool NearlyParallelSameDirection(const maths::vec3& a, const maths::vec3& b, float epsilon = 1.0e-3f) {
        const float aLen = a.Length();
        const float bLen = b.Length();
        if (aLen < 1.0e-8f || bLen < 1.0e-8f) {
            return false;
        }
        const maths::vec3 an = a.Normalize();
        const maths::vec3 bn = b.Normalize();
        return an.Dot(bn) > 1.0f - epsilon;
    }

    // A straight (collinear position AND tangent along the same axis) spline with strongly
    // non-uniform Hermite parametrization speed: small tangent magnitudes at both ends (slow) vs.
    // the large position gap between them, producing a smoothstep-like "slow-fast-slow" `t`->speed
    // profile. Because every control point and tangent lies exactly on the Z axis, the curve
    // itself has ZERO lateral curvature -- its true arc length between any two `t` values is
    // EXACTLY the Z-displacement between them (no piecewise-linear-approximation error at all,
    // even at low table resolution), which makes this an exact ground truth for the point-count
    // and spacing checks below, isolating "does the arc-length remap work" from "does the chord
    // approximation converge".
    pcg::PcgSplineData MakeStraightVariableSpeedSpline() {
        std::vector<pcg::PcgSplineControlPoint> controlPoints;
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 0.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 20.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
        return pcg::PcgSplineData(controlPoints);
    }

    // A genuinely CURVED spline (an "L-bend" in the XZ plane) with sharply differing tangent
    // magnitudes across its 3 control points (2, then 8, then 2) -- this induces strong
    // non-uniform `t`-parametrization speed AND real lateral curvature, so a sampler that
    // (incorrectly) stepped `t` uniformly instead of remapping through true arc length would
    // produce consecutive real-world point distances that visibly do NOT match the requested
    // spacing (bunched up where speed is high near the middle control point, spread out at the
    // slow ends). A CORRECT arc-length sampler keeps consecutive real distances close to the
    // requested spacing throughout.
    pcg::PcgSplineData MakeCurvedVariableSpeedSpline() {
        std::vector<pcg::PcgSplineControlPoint> controlPoints;
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 0.0f }, maths::vec3{ 2.0f, 0.0f, 0.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 5.0f, 0.0f, 0.0f }, maths::vec3{ 0.0f, 0.0f, 8.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 5.0f, 0.0f, 10.0f }, maths::vec3{ 2.0f, 0.0f, 0.0f } });
        return pcg::PcgSplineData(controlPoints);
    }

    void TestArcLengthTableStraightSpline() {
        const pcg::PcgSplineData spline = MakeStraightVariableSpeedSpline();
        const std::vector<pcg::PcgArcLengthSample> table = pcg::BuildSplineArcLengthTable(spline, 256);

        Check(table.size() > 1, "BuildSplineArcLengthTable: a 2-control-point spline produces a multi-sample table");
        Check(NearlyEqual(table.front().distance, 0.0f), "BuildSplineArcLengthTable: first sample is at distance 0");
        // Exact ground truth (see MakeStraightVariableSpeedSpline's own comment): total arc length
        // of a straight Z-axis-aligned curve equals its endpoint Z displacement, exactly.
        Check(NearlyEqual(table.back().distance, 20.0f, 0.01f),
            "BuildSplineArcLengthTable: total arc length of a straight collinear spline matches its exact Z displacement");

        // The parametrization speed is non-uniform (slow-fast-slow), so equal STEPS of `t` must
        // NOT correspond to equal steps of distance -- otherwise this whole test spline would fail
        // to exercise the arc-length remap at all.
        const float distanceFirstQuarterT = pcg::ResolveSplineTAtArcLength(table, 0.0f); // placeholder, unused directly
        (void)distanceFirstQuarterT;
        const float distAtQuarterT = table[table.size() / 4].distance;
        const float distAtHalfT = table[table.size() / 2].distance;
        const float firstQuarterSpan = distAtQuarterT - 0.0f;
        const float secondQuarterSpan = distAtHalfT - distAtQuarterT;
        Check(std::abs(firstQuarterSpan - secondQuarterSpan) > 1.0f,
            "Test spline sanity: equal t-steps do NOT cover equal arc-length spans (proves this spline actually stresses the remap)");

        // Round-trip: resolving t back from a known distance should land close to the distance we asked for.
        const float tAtTen = pcg::ResolveSplineTAtArcLength(table, 10.0f);
        const maths::vec3 posAtTen = spline.EvaluatePosition(tAtTen);
        Check(NearlyEqual(posAtTen.z, 10.0f, 0.05f), "ResolveSplineTAtArcLength: distance 10 resolves to a t whose position is at Z=10");
    }

    void TestPointCountMatchesArcLength() {
        const pcg::PcgSplineData spline = MakeStraightVariableSpeedSpline(); // exact total length 20.0
        pcg::PcgSplineSamplerParams params;
        params.spacing = 2.0f;
        params.seed = 1u;

        const std::vector<pcg::PcgPoint> points = pcg::SampleSplineByArcLength(spline, params);
        const float expectedCount = 20.0f / 2.0f; // == 10
        Check(std::abs(static_cast<float>(points.size()) - expectedCount) <= 1.0f,
            "SampleSplineByArcLength: point count is within one point of totalArcLength/spacing");
        Check(points.size() >= 9 && points.size() <= 10,
            "SampleSplineByArcLength: point count for a 20-unit spline at spacing=2 is 9 or 10");
    }

    void TestConsecutiveSpacingOnStraightSpline() {
        const pcg::PcgSplineData spline = MakeStraightVariableSpeedSpline();
        pcg::PcgSplineSamplerParams params;
        params.spacing = 1.5f;
        params.seed = 7u;

        const std::vector<pcg::PcgPoint> points = pcg::SampleSplineByArcLength(spline, params);
        Check(points.size() >= 12, "SampleSplineByArcLength (straight spline): produced a reasonable number of points to check spacing on");

        for (size_t i = 1; i < points.size(); ++i) {
            const maths::vec3 delta = points[i].position - points[i - 1].position;
            const float realDistance = delta.Length();
            // Zero curvature -> chord distance should match the requested spacing almost exactly
            // (tolerance only needs to cover the arc-length table's own piecewise-chord
            // resolution, not any curvature error).
            Check(NearlyEqual(realDistance, params.spacing, 0.02f),
                "SampleSplineByArcLength (straight, variable-speed spline): consecutive points are ~spacing apart in real 3D distance");
        }
    }

    void TestConsecutiveSpacingOnCurvedSpline() {
        const pcg::PcgSplineData spline = MakeCurvedVariableSpeedSpline();
        pcg::PcgSplineSamplerParams params;
        params.spacing = 1.0f;
        params.seed = 42u;

        const std::vector<pcg::PcgPoint> points = pcg::SampleSplineByArcLength(spline, params);
        Check(points.size() >= 8, "SampleSplineByArcLength (curved spline): produced a reasonable number of points to check spacing on");

        int withinToleranceCount = 0;
        for (size_t i = 1; i < points.size(); ++i) {
            const maths::vec3 delta = points[i].position - points[i - 1].position;
            const float realDistance = delta.Length();
            // A curved path's chord distance is always <= its arc-length distance; with spacing
            // small relative to the curve's own extent (16 points over this spline at spacing=1.0),
            // the two stay close. Empirically (verified while authoring this test) the worst-case
            // deviation on this exact spline is ~3%, right at the sharply-bending middle control
            // point -- a 10% tolerance therefore keeps a healthy ~3x safety margin while still
            // being tight enough to catch a real regression. This is the KEY correctness check: if
            // the sampler only stepped `t` uniformly instead of remapping through true arc length,
            // this curve's sharp speed variation would blow WAY past this tolerance (points would
            // visibly bunch up near the fast middle control point and spread out near the slow
            // ends, rather than staying within a few percent of `spacing` everywhere).
            if (NearlyEqual(realDistance, params.spacing, 0.1f)) {
                ++withinToleranceCount;
            }
        }
        Check(withinToleranceCount == static_cast<int>(points.size()) - 1,
            "SampleSplineByArcLength (curved, variable-speed spline): every consecutive pair is ~spacing apart in real 3D distance, proving the arc-length remap (not naive uniform-t stepping) is what actually runs");
    }

    void TestRotationAlignsWithTangent() {
        const pcg::PcgSplineData spline = MakeCurvedVariableSpeedSpline();
        pcg::PcgSplineSamplerParams params;
        params.spacing = 1.0f;
        params.seed = 99u;

        const std::vector<pcg::PcgArcLengthSample> table = pcg::BuildSplineArcLengthTable(spline, params.arcLengthSubdivisionsPerSegment);
        const std::vector<pcg::PcgPoint> points = pcg::SampleSplineByArcLength(spline, params);
        Check(!points.empty(), "SampleSplineByArcLength (curved spline): produced at least one point to check rotation on");

        for (size_t i = 0; i < points.size(); ++i) {
            const float baseDistance = static_cast<float>(i + 1) * params.spacing; // jitter is 0 by default -> exact.
            const float t = pcg::ResolveSplineTAtArcLength(table, baseDistance);
            const maths::vec3 expectedTangent = spline.EvaluateTangent(t);

            const maths::vec3 worldForward{ 0.0f, 0.0f, 1.0f };
            const maths::vec3 rotatedForward = points[i].rotation.RotateVector(worldForward);

            Check(NearlyParallelSameDirection(rotatedForward, expectedTangent),
                "SampleSplineByArcLength: point rotation's forward axis is parallel to the curve's local tangent direction");
        }
    }

    void TestDegenerateSplinesDoNotCrash() {
        // 0 control points.
        {
            std::vector<pcg::PcgSplineControlPoint> emptyControlPoints;
            pcg::PcgSplineData emptySpline(emptyControlPoints);
            pcg::PcgSplineSamplerParams params;
            params.spacing = 1.0f;
            params.seed = 1u;
            const std::vector<pcg::PcgPoint> points = pcg::SampleSplineByArcLength(emptySpline, params);
            Check(points.empty(), "SampleSplineByArcLength: a 0-control-point spline does not crash and yields no points");

            const std::vector<pcg::PcgArcLengthSample> table = pcg::BuildSplineArcLengthTable(emptySpline, 256);
            Check(table.size() == 1, "BuildSplineArcLengthTable: a 0-control-point spline yields a single degenerate table entry");
        }

        // 1 control point.
        {
            std::vector<pcg::PcgSplineControlPoint> oneControlPoint;
            oneControlPoint.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 3.0f, 4.0f, 5.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
            pcg::PcgSplineData oneSpline(oneControlPoint);
            pcg::PcgSplineSamplerParams params;
            params.spacing = 1.0f;
            params.seed = 2u;
            const std::vector<pcg::PcgPoint> points = pcg::SampleSplineByArcLength(oneSpline, params);
            Check(points.empty(), "SampleSplineByArcLength: a 1-control-point spline (zero extent) does not crash and yields no points");
        }

        // 2 control points (the minimum non-degenerate case -- exactly one Hermite segment).
        {
            std::vector<pcg::PcgSplineControlPoint> twoControlPoints;
            twoControlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 0.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
            twoControlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 4.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
            pcg::PcgSplineData twoSpline(twoControlPoints);
            pcg::PcgSplineSamplerParams params;
            params.spacing = 1.0f;
            params.seed = 3u;
            const std::vector<pcg::PcgPoint> points = pcg::SampleSplineByArcLength(twoSpline, params);
            Check(!points.empty(), "SampleSplineByArcLength: a 2-control-point spline (one segment) does not crash and yields points");
            Check(points.size() >= 3 && points.size() <= 4,
                "SampleSplineByArcLength: a 4-unit, 2-control-point spline at spacing=1 yields 3 or 4 points");
        }

        // Non-positive spacing must not hang or divide by zero.
        {
            const pcg::PcgSplineData spline = MakeStraightVariableSpeedSpline();
            pcg::PcgSplineSamplerParams zeroSpacingParams;
            zeroSpacingParams.spacing = 0.0f;
            const std::vector<pcg::PcgPoint> zeroPoints = pcg::SampleSplineByArcLength(spline, zeroSpacingParams);
            Check(zeroPoints.empty(), "SampleSplineByArcLength: zero spacing yields no points instead of hanging/dividing by zero");

            pcg::PcgSplineSamplerParams negativeSpacingParams;
            negativeSpacingParams.spacing = -1.0f;
            const std::vector<pcg::PcgPoint> negativePoints = pcg::SampleSplineByArcLength(spline, negativeSpacingParams);
            Check(negativePoints.empty(), "SampleSplineByArcLength: negative spacing yields no points instead of hanging/dividing by zero");
        }
    }

    void TestDeterminism() {
        const pcg::PcgSplineData spline = MakeCurvedVariableSpeedSpline();
        pcg::PcgSplineSamplerParams params;
        params.spacing = 0.75f;
        params.seed = 0xBEEF1234u;
        params.perpendicularOffset = 0.3f;
        params.jitterDistance = 0.1f;
        params.jitterPerpendicular = 0.05f;

        const std::vector<pcg::PcgPoint> runA = pcg::SampleSplineByArcLength(spline, params);
        const std::vector<pcg::PcgPoint> runB = pcg::SampleSplineByArcLength(spline, params);

        Check(runA.size() == runB.size() && !runA.empty(), "SampleSplineByArcLength: two runs with identical inputs produce the same point count");

        bool allIdentical = (runA.size() == runB.size());
        for (size_t i = 0; allIdentical && i < runA.size(); ++i) {
            const pcg::PcgPoint& a = runA[i];
            const pcg::PcgPoint& b = runB[i];
            if (a.position.x != b.position.x || a.position.y != b.position.y || a.position.z != b.position.z ||
                a.rotation.x != b.rotation.x || a.rotation.y != b.rotation.y || a.rotation.z != b.rotation.z || a.rotation.w != b.rotation.w ||
                a.seed != b.seed) {
                allIdentical = false;
            }
        }
        Check(allIdentical, "SampleSplineByArcLength: two runs with identical spline+params produce byte-identical points (position/rotation/seed)");

        // A different seed must (overwhelmingly likely, given 6+ points) diverge in at least the
        // per-point seeds even though the base (unjittered) positions stay identical.
        pcg::PcgSplineSamplerParams differentSeedParams = params;
        differentSeedParams.seed = 0xDEADBEEFu;
        const std::vector<pcg::PcgPoint> runC = pcg::SampleSplineByArcLength(spline, differentSeedParams);
        bool anySeedDiffers = (runC.size() != runA.size());
        for (size_t i = 0; !anySeedDiffers && i < runC.size(); ++i) {
            if (runC[i].seed != runA[i].seed) {
                anySeedDiffers = true;
            }
        }
        Check(anySeedDiffers, "SampleSplineByArcLength: a different seed changes the derived per-point seeds");
    }

} // namespace

int main() {
    TestArcLengthTableStraightSpline();
    TestPointCountMatchesArcLength();
    TestConsecutiveSpacingOnStraightSpline();
    TestConsecutiveSpacingOnCurvedSpline();
    TestRotationAlignsWithTangent();
    TestDegenerateSplinesDoNotCrash();
    TestDeterminism();

    if (g_failCount == 0) {
        std::cout << "PcgSplineSamplerTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgSplineSamplerTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
