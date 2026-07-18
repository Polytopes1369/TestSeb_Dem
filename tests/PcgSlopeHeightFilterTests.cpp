// Standalone, framework-free unit test for the PCG framework roadmap's Phase 3.4 ("Slope/Height
// Filter + Projection") types: src/pcg/PcgSlopeHeightFilter.h/.cpp. Mirrors
// PcgDataModelTests.cpp/PcgTerrainSamplerTests.cpp's own convention exactly (plain std::cerr on
// failure, exits 0/1, no Logger.cpp/Vulkan dependency) so it builds/runs identically in either config
// and needs no engine bring-up.
//
// Exercises: the Slope Filter's terrain-batch overload (exact, synthetic-known-slope cases plus a
// real-terrain integration cross-check against an independently-computed reference count), the Slope
// Filter's approximate rotation-derived overload, the Height Filter's inclusive-boundary semantics,
// and Projection's actual re-snap onto a freshly re-queried terrain height/normal (verified against
// PcgTerrainSampler's own SampleHeightCPU/ComputeTerrainNormalCPU directly, the same
// "re-query and confirm agreement" technique PcgTerrainSamplerTests.cpp already uses for
// QuatFromNormal).

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSlopeHeightFilter.h"
#include "pcg/PcgSpatialData.h"
#include "pcg/PcgTerrainSampler.h"

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

    // ------------------------------------------------------------------------------------------
    // Slope Filter -- terrain-batch overload (reads batch.slopeRadians directly).
    // ------------------------------------------------------------------------------------------

    void TestSlopeFilterTerrainBatchExactness() {
        // Hand-built batch with fully known slope values (NOT from SampleTerrainPoints), so the
        // exact expected keep/exclude set is known in advance without depending on noise output.
        pcg::PcgTerrainPointBatch batch;
        const float slopes[] = { 0.0f, 0.2f, 0.5f, 0.7853981634f /* PI/4 */, 1.0f, 1.5707963268f /* PI/2 */ };
        for (size_t i = 0; i < 6; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ static_cast<float>(i), 0.0f, 0.0f };
            p.seed = static_cast<uint32_t>(i);
            batch.points.push_back(p);
            batch.slopeRadians.push_back(slopes[i]);
        }

        const std::vector<pcg::PcgPoint> filtered = pcg::FilterBySlope(batch, 0.5f);
        Check(filtered.size() == 3, "FilterBySlope(batch): keeps exactly the points at/under the threshold (0.0, 0.2, 0.5)");
        if (filtered.size() == 3) {
            Check(NearlyEqual(filtered[0].position.x, 0.0f) && NearlyEqual(filtered[1].position.x, 1.0f) && NearlyEqual(filtered[2].position.x, 2.0f),
                "FilterBySlope(batch): kept points preserve original relative order/identity");
        }

        // Boundary: threshold exactly equal to a point's own slope keeps that point (<=, not <).
        const std::vector<pcg::PcgPoint> boundary = pcg::FilterBySlope(batch, slopes[2]);
        Check(boundary.size() == 3, "FilterBySlope(batch): a threshold exactly equal to a point's slope keeps that point (inclusive)");

        // A threshold no real ComputeSlopeRadians output (range [0, PI]) can ever exceed keeps everything.
        const std::vector<pcg::PcgPoint> keepAll = pcg::FilterBySlope(batch, maths::PI);
        Check(keepAll.size() == batch.points.size(), "FilterBySlope(batch): threshold=PI keeps every point");

        // A threshold below the minimum possible slope (0) keeps nothing.
        const std::vector<pcg::PcgPoint> keepNone = pcg::FilterBySlope(batch, -0.001f);
        Check(keepNone.empty(), "FilterBySlope(batch): a threshold below the minimum possible slope keeps nothing");

        // Mismatched-size safety: a hand-built batch whose slopeRadians array is shorter than points
        // must not read past the shorter array's end (only the common prefix is considered).
        pcg::PcgTerrainPointBatch mismatched;
        mismatched.points.push_back(pcg::PcgPoint{});
        mismatched.points.push_back(pcg::PcgPoint{});
        mismatched.slopeRadians.push_back(0.0f); // Only one slope entry for two points.
        const std::vector<pcg::PcgPoint> safeResult = pcg::FilterBySlope(mismatched, maths::PI);
        Check(safeResult.size() == 1, "FilterBySlope(batch): a mismatched-size batch only considers the common prefix, never reads out of bounds");
    }

    void TestSlopeFilterTerrainBatchIntegration() {
        pcg::PcgLandscapeData terrain;
        terrain.width = 64.0f;
        terrain.length = 64.0f;
        terrain.worldOffset = maths::vec3{ 0.0f, 0.0f, 0.0f };
        const pcg::PcgTerrainPointBatch batch = pcg::SampleTerrainPoints(terrain, 0.25f, 777u);
        Check(!batch.points.empty(), "SampleTerrainPoints (slope filter integration setup): produces a non-empty batch");

        // A threshold no real slope value can ever exceed keeps every real-terrain point -- reaches
        // the same "a maximally permissive filter is a no-op" invariant a literal flat-terrain case
        // would, without requiring the noise-driven terrain to be literally flat (kTerrainAmplitude is
        // a compile-time GPU-matching constant, not a test-adjustable knob -- see
        // PcgTerrainSampler.h's own header comment).
        const std::vector<pcg::PcgPoint> keepAll = pcg::FilterBySlope(batch, maths::PI);
        Check(keepAll.size() == batch.points.size(), "FilterBySlope(batch) integration: threshold=PI keeps every real-terrain point");

        // Cross-check against an independently computed reference count (a plain loop over the same
        // parallel slopeRadians array), so this is not just re-testing the implementation against
        // itself.
        constexpr float kThreshold = 0.3f;
        size_t expectedCount = 0;
        for (float s : batch.slopeRadians) {
            if (s <= kThreshold) { ++expectedCount; }
        }
        const std::vector<pcg::PcgPoint> filtered = pcg::FilterBySlope(batch, kThreshold);
        Check(filtered.size() == expectedCount, "FilterBySlope(batch) integration: filtered count matches an independently-computed reference count");
    }

    // ------------------------------------------------------------------------------------------
    // Slope Filter -- generic (approximate, rotation-derived) overload.
    // ------------------------------------------------------------------------------------------

    void TestSlopeFilterApproximateFromRotation() {
        std::vector<pcg::PcgPoint> points;

        pcg::PcgPoint flatPoint; // Identity rotation -> local up IS world up -> ~0 slope.
        points.push_back(flatPoint);

        pcg::PcgPoint tiltedPoint;
        // QuatFromNormal is reused here purely as a convenient way to build a KNOWN-angle rotation for
        // this test (not because FilterBySlope itself calls it) -- aligns local +Y to world +X, a
        // perfectly vertical "cliff-face" direction, i.e. an exact PI/2 approximate slope.
        tiltedPoint.rotation = pcg::QuatFromNormal(maths::vec3{ 1.0f, 0.0f, 0.0f });
        points.push_back(tiltedPoint);

        pcg::PcgPoint diagonalPoint;
        diagonalPoint.rotation = pcg::QuatFromNormal(maths::vec3{ 1.0f, 1.0f, 0.0f }.Normalize()); // Exact 45-degree tilt.
        points.push_back(diagonalPoint);

        const std::vector<pcg::PcgPoint> tight = pcg::FilterBySlope(points, maths::ToRadians(10.0f));
        Check(tight.size() == 1, "FilterBySlope(points, approximate): a tight (10-degree) threshold keeps only the flat, identity-rotation point");

        const std::vector<pcg::PcgPoint> medium = pcg::FilterBySlope(points, maths::ToRadians(50.0f));
        Check(medium.size() == 2, "FilterBySlope(points, approximate): a 50-degree threshold keeps the flat and 45-degree points, excludes the 90-degree one");

        const std::vector<pcg::PcgPoint> all = pcg::FilterBySlope(points, maths::PI);
        Check(all.size() == points.size(), "FilterBySlope(points, approximate): a generous (PI) threshold keeps every point");

        const std::vector<pcg::PcgPoint> none = pcg::FilterBySlope(points, -0.001f);
        Check(none.empty(), "FilterBySlope(points, approximate): a threshold below the minimum possible slope keeps nothing");
    }

    // ------------------------------------------------------------------------------------------
    // Height Filter.
    // ------------------------------------------------------------------------------------------

    void TestHeightFilterBoundaries() {
        std::vector<pcg::PcgPoint> points;
        const float heights[] = { -10.0f, 0.0f, 5.0f, 10.0f, 10.0001f, 25.0f };
        for (float h : heights) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ 0.0f, h, 0.0f };
            points.push_back(p);
        }

        // Range [0, 10], inclusive at both ends.
        const std::vector<pcg::PcgPoint> filtered = pcg::FilterByHeight(points, 0.0f, 10.0f);
        Check(filtered.size() == 3, "FilterByHeight: keeps exactly the points within [min, max] inclusive");
        if (filtered.size() == 3) {
            Check(NearlyEqual(filtered[0].position.y, 0.0f) && NearlyEqual(filtered[1].position.y, 5.0f) && NearlyEqual(filtered[2].position.y, 10.0f),
                "FilterByHeight: kept points are exactly the ones at the min boundary (0.0), the interior (5.0), and the max boundary (10.0)");
        }

        // A degenerate [h, h] range keeps only points at exactly that height.
        const std::vector<pcg::PcgPoint> exact = pcg::FilterByHeight(points, 5.0f, 5.0f);
        Check(exact.size() == 1 && NearlyEqual(exact[0].position.y, 5.0f), "FilterByHeight: a degenerate [h, h] range keeps only the point at exactly that height");

        // A range excluding everything.
        const std::vector<pcg::PcgPoint> none = pcg::FilterByHeight(points, 100.0f, 200.0f);
        Check(none.empty(), "FilterByHeight: a range with no points inside it returns an empty result");

        // A range covering everything.
        const std::vector<pcg::PcgPoint> allPoints = pcg::FilterByHeight(points, -1000.0f, 1000.0f);
        Check(allPoints.size() == points.size(), "FilterByHeight: a range covering every point's height returns the full set");
    }

    // ------------------------------------------------------------------------------------------
    // Projection.
    // ------------------------------------------------------------------------------------------

    void TestProjectionSnapsToTerrain() {
        pcg::PcgLandscapeData terrain;
        terrain.meshID = 5;
        terrain.width = 128.0f;
        terrain.length = 128.0f;
        terrain.worldOffset = maths::vec3{ 20.0f, 4.0f, -30.0f };

        std::vector<pcg::PcgPoint> points;
        const maths::vec3 testPositions[] = {
            maths::vec3{ 20.0f, 999.0f, -30.0f }, // Deliberately floating far above the surface.
            maths::vec3{ 5.0f, -50.0f, 10.0f },   // Deliberately buried far below.
            maths::vec3{ -40.0f, 0.0f, 60.0f },
        };
        for (const maths::vec3& pos : testPositions) {
            pcg::PcgPoint p;
            p.position = pos;
            p.rotation = maths::quat{}; // Identity -- deliberately NOT pre-aligned to the terrain.
            p.seed = 123u;
            p.density = 0.75f; // Non-default, to verify Projection leaves unrelated fields untouched.
            points.push_back(p);
        }

        const float originalX[] = { points[0].position.x, points[1].position.x, points[2].position.x };
        const float originalZ[] = { points[0].position.z, points[1].position.z, points[2].position.z };
        const float originalDensity[] = { points[0].density, points[1].density, points[2].density };
        const uint32_t originalSeed[] = { points[0].seed, points[1].seed, points[2].seed };
        const float originalY0 = points[0].position.y;
        const float originalY1 = points[1].position.y;

        pcg::ProjectOntoTerrain(points, terrain);

        for (size_t i = 0; i < points.size(); ++i) {
            Check(NearlyEqual(points[i].position.x, originalX[i]) && NearlyEqual(points[i].position.z, originalZ[i]),
                "ProjectOntoTerrain: position.x/z are left untouched");
            Check(NearlyEqual(points[i].density, originalDensity[i]) && points[i].seed == originalSeed[i],
                "ProjectOntoTerrain: unrelated fields (density, seed) are left untouched");

            // Re-query the terrain directly (independent of ProjectOntoTerrain's own internal call)
            // and confirm position.y now agrees exactly with a fresh SampleHeightCPU query at the
            // point's (possibly-still-original) (x, z).
            const float expectedHeight = pcg::SampleHeightCPU(terrain, points[i].position.x, points[i].position.z);
            Check(points[i].position.y == expectedHeight,
                "ProjectOntoTerrain: position.y exactly matches a fresh, independent SampleHeightCPU query at the same (x, z)");

            // Re-query the terrain normal directly and confirm rotation reproduces it: rotating local
            // +Y by the point's NEW rotation must reproduce the freshly-queried normal -- the same
            // "rotate up, compare to target" technique PcgTerrainSamplerTests.cpp already uses to
            // verify QuatFromNormal itself.
            const maths::vec3 expectedNormal = pcg::ComputeTerrainNormalCPU(terrain, points[i].position.x, points[i].position.z);
            const maths::vec3 rotatedUp = points[i].rotation.RotateVector(maths::vec3{ 0.0f, 1.0f, 0.0f });
            Check(NearlyEqual(rotatedUp.x, expectedNormal.x, 1.0e-3f) && NearlyEqual(rotatedUp.y, expectedNormal.y, 1.0e-3f) && NearlyEqual(rotatedUp.z, expectedNormal.z, 1.0e-3f),
                "ProjectOntoTerrain: rotation is re-aligned to the freshly sampled terrain normal at the point's (x, z)");
        }

        // A point that started floating far above/below the surface must have actually MOVED (this
        // gentle terrain's height is always within (-0.4, 0.4) world units of worldOffset.y, so
        // neither y=999 nor y=-50 can possibly still hold after projection).
        Check(!NearlyEqual(points[0].position.y, originalY0, 1.0f), "ProjectOntoTerrain: a floating point's y is actually snapped down onto the surface, not left untouched");
        Check(!NearlyEqual(points[1].position.y, originalY1, 1.0f), "ProjectOntoTerrain: a buried point's y is actually snapped up onto the surface, not left untouched");
    }

    void TestProjectionIsIdempotentOnAlreadyProjectedPoints() {
        // Projecting a point that is ALREADY exactly on the surface (e.g. the direct output of
        // SampleTerrainPoints, or a point that was already projected once) must be a no-op: same
        // (x, z) always resamples to the same height/normal (SampleTerrainHeightLocalCPU's own
        // determinism guarantee -- see PcgTerrainSamplerTests.cpp's own
        // TestHeightDeterminismAndWorldOffset), so a second projection must not move anything further.
        pcg::PcgLandscapeData terrain;
        terrain.width = 64.0f;
        terrain.length = 64.0f;
        terrain.worldOffset = maths::vec3{ 0.0f, 0.0f, 0.0f };

        pcg::PcgTerrainPointBatch batch = pcg::SampleTerrainPoints(terrain, 0.25f, 42u);
        Check(!batch.points.empty(), "SampleTerrainPoints (idempotency test setup): produces a non-empty batch");

        std::vector<pcg::PcgPoint> points = batch.points; // Copy: already correctly placed by the sampler.
        const std::vector<pcg::PcgPoint> beforeSecondProjection = points;

        pcg::ProjectOntoTerrain(points, terrain);

        bool allUnchanged = (points.size() == beforeSecondProjection.size());
        if (allUnchanged) {
            for (size_t i = 0; i < points.size(); ++i) {
                if (points[i].position.y != beforeSecondProjection[i].position.y) {
                    allUnchanged = false;
                    break;
                }
            }
        }
        Check(allUnchanged, "ProjectOntoTerrain: re-projecting already-correctly-placed points is a no-op on position.y");
    }

} // namespace

int main() {
    TestSlopeFilterTerrainBatchExactness();
    TestSlopeFilterTerrainBatchIntegration();
    TestSlopeFilterApproximateFromRotation();
    TestHeightFilterBoundaries();
    TestProjectionSnapsToTerrain();
    TestProjectionIsIdempotentOnAlreadyProjectedPoints();

    if (g_failCount == 0) {
        std::cout << "PcgSlopeHeightFilterTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgSlopeHeightFilterTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
