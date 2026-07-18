// Standalone, framework-free unit test for the PCG framework roadmap's Phase 3.3 ("Boolean Set
// Operations") type: src/pcg/PcgBooleanSetOps.h/.cpp. Exercises Union's plain concatenation,
// IntersectWithVolume/DifferenceFromVolume's inside/outside filtering (including the documented
// "boundary counts as inside" convention), DifferenceFromSpline's arc-length-table-based
// closest-distance exclusion against both a straight and a curved spline segment, and
// IntersectMultipleVolumes' no-double-count behavior for a point inside two overlapping volumes.
// Exits 0 if every check passes, non-zero otherwise -- registered with CTest (see the top-level
// CMakeLists.txt), matching tests/PcgDataModelTests.cpp's own framework-free convention.

#include "core/maths/Maths.h"
#include "pcg/PcgBooleanSetOps.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

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

    pcg::PcgPoint MakePointAt(const maths::vec3& position, uint32_t seed = 0) {
        pcg::PcgPoint point;
        point.position = position;
        point.seed = seed;
        return point;
    }

    // ------------------------------------------------------------------------------------------
    // Union
    // ------------------------------------------------------------------------------------------

    void TestUnionIsPlainConcatenation() {
        std::vector<pcg::PcgPoint> a;
        a.push_back(MakePointAt(maths::vec3{ 0.0f, 0.0f, 0.0f }, 1));
        a.push_back(MakePointAt(maths::vec3{ 1.0f, 0.0f, 0.0f }, 2));
        a.push_back(MakePointAt(maths::vec3{ 2.0f, 0.0f, 0.0f }, 3));

        std::vector<pcg::PcgPoint> b;
        b.push_back(MakePointAt(maths::vec3{ 0.0f, 0.0f, 0.0f }, 1)); // Deliberately coincident with a[0].
        b.push_back(MakePointAt(maths::vec3{ 5.0f, 0.0f, 0.0f }, 4));

        const std::vector<pcg::PcgPoint> result = pcg::Union(a, b);

        Check(result.size() == a.size() + b.size(), "Union: result point count is exactly the sum of the two inputs' counts");
        Check(result.size() == 5, "Union: 3-point set + 2-point set (including one coincident position) yields 5 points, NOT deduplicated");

        // Order preserved: a's points first (in their own order), then b's points (in their own order).
        Check(result[0].seed == 1 && result[1].seed == 2 && result[2].seed == 3 && result[3].seed == 1 && result[4].seed == 4,
            "Union: result preserves a's points (in order) followed by b's points (in order)");

        // Empty-input edge cases.
        const std::vector<pcg::PcgPoint> emptyA;
        const std::vector<pcg::PcgPoint> unionWithEmpty = pcg::Union(emptyA, b);
        Check(unionWithEmpty.size() == b.size(), "Union: empty `a` yields exactly `b`'s points");
        const std::vector<pcg::PcgPoint> bothEmpty = pcg::Union(emptyA, emptyA);
        Check(bothEmpty.empty(), "Union: two empty inputs yield an empty result");
    }

    // ------------------------------------------------------------------------------------------
    // IntersectWithVolume / DifferenceFromVolume
    // ------------------------------------------------------------------------------------------

    void TestVolumeIntersectionAndDifference() {
        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        volume.halfExtents = maths::vec3{ 2.0f, 2.0f, 2.0f };
        // Identity orientation -> a plain AABB spanning [-2,2] on every axis.

        std::vector<pcg::PcgPoint> points;
        points.push_back(MakePointAt(maths::vec3{ 0.0f, 0.0f, 0.0f }, 1));   // Deep interior.
        points.push_back(MakePointAt(maths::vec3{ 1.5f, 0.0f, 0.0f }, 2));   // Interior.
        points.push_back(MakePointAt(maths::vec3{ 2.0f, 0.0f, 0.0f }, 3));   // EXACTLY on the boundary (x == halfExtents.x).
        points.push_back(MakePointAt(maths::vec3{ 2.5f, 0.0f, 0.0f }, 4));   // Just outside.
        points.push_back(MakePointAt(maths::vec3{ 10.0f, 10.0f, 10.0f }, 5)); // Far outside.

        const std::vector<pcg::PcgPoint> inside = pcg::IntersectWithVolume(points, volume);
        const std::vector<pcg::PcgPoint> outside = pcg::DifferenceFromVolume(points, volume);

        Check(inside.size() + outside.size() == points.size(), "IntersectWithVolume + DifferenceFromVolume: every input point lands in exactly one of the two results");

        // Documented convention: boundary point (seed 3, exactly at x == halfExtents.x) counts as INSIDE.
        bool boundaryInInside = false;
        for (const pcg::PcgPoint& p : inside) if (p.seed == 3) boundaryInInside = true;
        bool boundaryInOutside = false;
        for (const pcg::PcgPoint& p : outside) if (p.seed == 3) boundaryInOutside = true;
        Check(boundaryInInside && !boundaryInOutside, "IntersectWithVolume/DifferenceFromVolume: a point exactly on the volume boundary is treated as INSIDE (kept by Intersect, excluded by Difference)");

        Check(inside.size() == 3, "IntersectWithVolume: interior points (deep, near-edge, and boundary) are kept -- 3 of 5");
        Check(outside.size() == 2, "DifferenceFromVolume: exterior points are kept -- 2 of 5");

        // Order preservation.
        Check(inside[0].seed == 1 && inside[1].seed == 2 && inside[2].seed == 3, "IntersectWithVolume: result preserves input relative order");
        Check(outside[0].seed == 4 && outside[1].seed == 5, "DifferenceFromVolume: result preserves input relative order");
    }

    // ------------------------------------------------------------------------------------------
    // DifferenceFromSpline
    // ------------------------------------------------------------------------------------------

    pcg::PcgSplineData MakeStraightSplineAlongZ() {
        std::vector<pcg::PcgSplineControlPoint> controlPoints;
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 0.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 20.0f }, maths::vec3{ 0.0f, 0.0f, 1.0f } });
        return pcg::PcgSplineData(controlPoints);
    }

    // An "L-bend" curved spline in the XZ plane, same shape as the Phase 2.4 spline sampler test's
    // own MakeCurvedVariableSpeedSpline -- reused here for the same reason: sharply differing
    // tangent magnitudes across control points give real lateral curvature, so this is a genuine
    // (not just straight-line) test of the closest-distance-to-curve query.
    pcg::PcgSplineData MakeCurvedSpline() {
        std::vector<pcg::PcgSplineControlPoint> controlPoints;
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 0.0f, 0.0f, 0.0f }, maths::vec3{ 2.0f, 0.0f, 0.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 5.0f, 0.0f, 0.0f }, maths::vec3{ 0.0f, 0.0f, 8.0f } });
        controlPoints.push_back(pcg::PcgSplineControlPoint{ maths::vec3{ 5.0f, 0.0f, 10.0f }, maths::vec3{ 2.0f, 0.0f, 0.0f } });
        return pcg::PcgSplineData(controlPoints);
    }

    void TestDifferenceFromStraightSpline() {
        const pcg::PcgSplineData spline = MakeStraightSplineAlongZ(); // Straight, lies exactly on the Z axis (x=0,y=0) from z=0 to z=20.

        std::vector<pcg::PcgPoint> points;
        points.push_back(MakePointAt(maths::vec3{ 0.0f, 0.0f, 10.0f }, 1));  // ON the curve: distance 0.
        points.push_back(MakePointAt(maths::vec3{ 3.0f, 0.0f, 10.0f }, 2));  // Perpendicular distance exactly 3.
        points.push_back(MakePointAt(maths::vec3{ 6.0f, 0.0f, 10.0f }, 3));  // Perpendicular distance exactly 6.
        points.push_back(MakePointAt(maths::vec3{ 100.0f, 0.0f, 100.0f }, 4)); // Far away in every direction.

        const float exclusionRadius = 5.0f;
        const std::vector<pcg::PcgPoint> result = pcg::DifferenceFromSpline(points, spline, exclusionRadius);

        // Points 1 (dist 0) and 2 (dist 3) are within the 5-unit exclusion radius -> excluded.
        // Points 3 (dist 6) and 4 (far) are outside it -> kept.
        Check(result.size() == 2, "DifferenceFromSpline (straight spline): exactly the 2 far-enough points survive a 5-unit exclusion radius");
        bool has3 = false, has4 = false, has1 = false, has2 = false;
        for (const pcg::PcgPoint& p : result) {
            if (p.seed == 1) has1 = true;
            if (p.seed == 2) has2 = true;
            if (p.seed == 3) has3 = true;
            if (p.seed == 4) has4 = true;
        }
        Check(!has1 && !has2 && has3 && has4, "DifferenceFromSpline (straight spline): near points (dist 0, dist 3) excluded; far points (dist 6, dist 100+) kept");

        // Boundary convention: a point at (very nearly) EXACTLY the exclusion radius. The
        // arc-length-table candidate-sampling approach is a coarse approximation (see
        // PcgBooleanSetOps.h's own comment), so this checks the DOCUMENTED convention (strict `>`
        // required to survive) via two points bracketing the radius rather than expecting exact
        // floating-point equality to resolve one specific way.
        std::vector<pcg::PcgPoint> bracketPoints;
        bracketPoints.push_back(MakePointAt(maths::vec3{ 4.99f, 0.0f, 10.0f }, 10)); // Just inside the exclusion radius.
        bracketPoints.push_back(MakePointAt(maths::vec3{ 5.01f, 0.0f, 10.0f }, 11)); // Just outside the exclusion radius.
        const std::vector<pcg::PcgPoint> bracketResult = pcg::DifferenceFromSpline(bracketPoints, spline, exclusionRadius);
        Check(bracketResult.size() == 1 && bracketResult[0].seed == 11,
            "DifferenceFromSpline: a point just inside the exclusion radius is excluded, a point just outside it is kept (strict > convention)");
    }

    void TestDifferenceFromCurvedSpline() {
        const pcg::PcgSplineData spline = MakeCurvedSpline();

        // A point sitting exactly on one of the spline's own control points (seed 1) has a true
        // closest distance of 0 (or extremely close to 0, bounded by the arc-length table's own
        // sampling resolution) -- must be excluded by any positive exclusion radius.
        std::vector<pcg::PcgPoint> points;
        points.push_back(MakePointAt(maths::vec3{ 5.0f, 0.0f, 0.0f }, 1)); // The curved spline's own middle control point.
        points.push_back(MakePointAt(maths::vec3{ -50.0f, 0.0f, -50.0f }, 2)); // Far outside the curve's entire extent.

        const float exclusionRadius = 2.0f;
        const std::vector<pcg::PcgPoint> result = pcg::DifferenceFromSpline(points, spline, exclusionRadius);

        Check(result.size() == 1 && result[0].seed == 2,
            "DifferenceFromSpline (curved spline): a point ON the curve is excluded, a point far from the curve's entire extent is kept");
    }

    void TestDifferenceFromSplineDoesNotCrashOnDegenerateSpline() {
        // 0 control points -- BuildSplineArcLengthTable's own single degenerate {t=0,distance=0}
        // entry, EvaluatePosition(0) on an empty spline returns the origin (PcgSplineData's own
        // EvaluateSegment degenerate-case fallback).
        std::vector<pcg::PcgSplineControlPoint> emptyControlPoints;
        pcg::PcgSplineData emptySpline(emptyControlPoints);

        std::vector<pcg::PcgPoint> points;
        points.push_back(MakePointAt(maths::vec3{ 0.0f, 0.0f, 0.0f }, 1)); // Exactly at the origin fallback position.
        points.push_back(MakePointAt(maths::vec3{ 100.0f, 0.0f, 0.0f }, 2));

        const std::vector<pcg::PcgPoint> result = pcg::DifferenceFromSpline(points, emptySpline, 5.0f);
        Check(result.size() == 1 && result[0].seed == 2, "DifferenceFromSpline: a degenerate 0-control-point spline does not crash and behaves as a single exclusion point at the origin");
    }

    // ------------------------------------------------------------------------------------------
    // IntersectMultipleVolumes
    // ------------------------------------------------------------------------------------------

    void TestIntersectMultipleVolumesNoDoubleCounting() {
        pcg::PcgVolumeData volumeA;
        volumeA.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        volumeA.halfExtents = maths::vec3{ 3.0f, 3.0f, 3.0f };

        pcg::PcgVolumeData volumeB;
        volumeB.center = maths::vec3{ 2.0f, 0.0f, 0.0f }; // Overlaps volumeA over roughly x in [-1,3].
        volumeB.halfExtents = maths::vec3{ 3.0f, 3.0f, 3.0f };

        std::vector<pcg::PcgVolumeData> volumes;
        volumes.push_back(volumeA);
        volumes.push_back(volumeB);

        std::vector<pcg::PcgPoint> points;
        points.push_back(MakePointAt(maths::vec3{ -2.0f, 0.0f, 0.0f }, 1)); // Inside A only.
        points.push_back(MakePointAt(maths::vec3{ 4.0f, 0.0f, 0.0f }, 2));  // Inside B only.
        points.push_back(MakePointAt(maths::vec3{ 1.0f, 0.0f, 0.0f }, 3));  // Inside BOTH A and B (the overlap region).
        points.push_back(MakePointAt(maths::vec3{ 50.0f, 50.0f, 50.0f }, 4)); // Inside neither.

        const std::vector<pcg::PcgPoint> result = pcg::IntersectMultipleVolumes(points, volumes);

        Check(result.size() == 3, "IntersectMultipleVolumes: 3 of 4 points are inside at least one of the two overlapping volumes");

        int seed3Count = 0;
        for (const pcg::PcgPoint& p : result) {
            if (p.seed == 3) ++seed3Count;
        }
        Check(seed3Count == 1, "IntersectMultipleVolumes: a point inside BOTH overlapping volumes is emitted exactly once, not duplicated");

        bool has1 = false, has2 = false, has4 = false;
        for (const pcg::PcgPoint& p : result) {
            if (p.seed == 1) has1 = true;
            if (p.seed == 2) has2 = true;
            if (p.seed == 4) has4 = true;
        }
        Check(has1 && has2 && !has4, "IntersectMultipleVolumes: single-volume-membership points are kept, points outside every volume are excluded");

        // Empty volumes list -> empty result (no volume for any point to be inside).
        const std::vector<pcg::PcgVolumeData> noVolumes;
        const std::vector<pcg::PcgPoint> emptyResult = pcg::IntersectMultipleVolumes(points, noVolumes);
        Check(emptyResult.empty(), "IntersectMultipleVolumes: an empty volumes list yields an empty result");
    }

    // ------------------------------------------------------------------------------------------
    // Determinism smoke test (pure functions -- included for consistency with the other PCG
    // phases' testing convention, even though purity makes this trivially true by construction).
    // ------------------------------------------------------------------------------------------

    void TestDeterminism() {
        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        volume.halfExtents = maths::vec3{ 5.0f, 5.0f, 5.0f };

        std::vector<pcg::PcgPoint> points;
        for (int i = 0; i < 20; ++i) {
            points.push_back(MakePointAt(maths::vec3{ static_cast<float>(i) - 10.0f, 0.0f, 0.0f }, static_cast<uint32_t>(i)));
        }

        const std::vector<pcg::PcgPoint> runA = pcg::IntersectWithVolume(points, volume);
        const std::vector<pcg::PcgPoint> runB = pcg::IntersectWithVolume(points, volume);
        Check(runA.size() == runB.size(), "Determinism: IntersectWithVolume produces the same result size across two identical calls");
        bool identical = (runA.size() == runB.size());
        for (size_t i = 0; identical && i < runA.size(); ++i) {
            if (runA[i].seed != runB[i].seed || runA[i].position.x != runB[i].position.x) {
                identical = false;
            }
        }
        Check(identical, "Determinism: IntersectWithVolume is a pure function -- identical input always yields byte-identical output");

        const pcg::PcgSplineData spline = MakeCurvedSpline();
        const std::vector<pcg::PcgPoint> diffA = pcg::DifferenceFromSpline(points, spline, 3.0f);
        const std::vector<pcg::PcgPoint> diffB = pcg::DifferenceFromSpline(points, spline, 3.0f);
        Check(diffA.size() == diffB.size() && diffA.size() > 0, "Determinism: DifferenceFromSpline produces the same non-trivial result size across two identical calls");
        bool diffIdentical = (diffA.size() == diffB.size());
        for (size_t i = 0; diffIdentical && i < diffA.size(); ++i) {
            if (diffA[i].seed != diffB[i].seed) {
                diffIdentical = false;
            }
        }
        Check(diffIdentical, "Determinism: DifferenceFromSpline is a pure function -- identical input always yields byte-identical output");
    }

} // namespace

int main() {
    TestUnionIsPlainConcatenation();
    TestVolumeIntersectionAndDifference();
    TestDifferenceFromStraightSpline();
    TestDifferenceFromCurvedSpline();
    TestDifferenceFromSplineDoesNotCrashOnDegenerateSpline();
    TestIntersectMultipleVolumesNoDoubleCounting();
    TestDeterminism();

    if (g_failCount == 0) {
        std::cout << "PcgBooleanSetOpsTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgBooleanSetOpsTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
