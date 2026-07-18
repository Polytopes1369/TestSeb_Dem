// Standalone, framework-free unit test for the PCG framework roadmap's Phase 2.3 ("Volume
// Sampler") code: src/pcg/PcgVolumeSampler.h/.cpp. Exercises Grid-mode lattice point count and
// jitter, Random-mode density-driven point count and statistical spread, a rotated-OBB case that
// specifically proves the sampler's local-space-then-transform pipeline honors
// PcgVolumeData::orientation (not just the volume's world-space AABB), and cross-call
// determinism. Exits 0 if every check passes, non-zero otherwise -- registered with CTest (see
// the top-level CMakeLists.txt), matching tests/PcgDataModelTests.cpp's own established
// framework-free convention for this project's PCG tests.

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"
#include "pcg/PcgVolumeSampler.h"

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

    bool NearlyEqual(float a, float b, float epsilon = 1.0e-3f) {
        return std::abs(a - b) <= epsilon;
    }

    bool PointsNearlyEqual(const maths::vec3& a, const maths::vec3& b, float epsilon = 1.0e-4f) {
        return NearlyEqual(a.x, b.x, epsilon) && NearlyEqual(a.y, b.y, epsilon) && NearlyEqual(a.z, b.z, epsilon);
    }

    // Checks that every point in `points` satisfies volume.ContainsWorldPoint() -- the core
    // correctness contract SampleVolume() promises for both modes. Returns the number of
    // violations found (0 == fully correct) rather than failing fast, so a single bad point
    // doesn't hide how widespread a regression is.
    size_t CountContainmentViolations(const pcg::PcgVolumeData& volume, const std::vector<pcg::PcgPoint>& points) {
        size_t violations = 0;
        for (const pcg::PcgPoint& p : points) {
            if (!volume.ContainsWorldPoint(p.position)) {
                ++violations;
            }
        }
        return violations;
    }

    // -----------------------------------------------------------------------------------------
    // Grid mode: axis-aligned volume, no jitter -- verifies the lattice point count matches the
    // documented floor((2*halfExtent)/spacing)+1-per-axis formula (using spacing values that
    // evenly divide the extent, so there is no floor-rounding ambiguity to worry about in the
    // test's own expected-count math), and that every lattice point lands inside the volume.
    // -----------------------------------------------------------------------------------------
    void TestGridModeBasicCountAndContainment() {
        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        volume.halfExtents = maths::vec3{ 2.0f, 1.0f, 2.0f };
        // Identity orientation (default) -> a plain AABB.

        pcg::PcgVolumeSamplerParams params;
        params.mode = pcg::PcgVolumeSamplingMode::Grid;
        params.gridSpacing = maths::vec3{ 1.0f, 1.0f, 1.0f };
        params.jitterFraction = 0.0f;

        const std::vector<pcg::PcgPoint> points = pcg::SampleVolume(volume, params, 1234u);

        // Expected per-axis counts: floor(2*halfExtent/spacing) + 1.
        const int expectedCountX = 5; // floor(4/1)+1
        const int expectedCountY = 3; // floor(2/1)+1
        const int expectedCountZ = 5; // floor(4/1)+1
        const size_t expectedTotal = static_cast<size_t>(expectedCountX) * static_cast<size_t>(expectedCountY) * static_cast<size_t>(expectedCountZ);

        Check(points.size() == expectedTotal,
            "Grid mode: point count matches floor(extent/spacing)+1 per axis (5*3*5 = 75)");

        const size_t violations = CountContainmentViolations(volume, points);
        Check(violations == 0, "Grid mode (no jitter): every lattice point satisfies volume.ContainsWorldPoint()");

        // The un-jittered corner point (first in iteration order: ix=0, iy=0, iz=0) must sit at
        // the volume's own local-space corner (center - halfExtents), transformed by identity
        // orientation -- within PcgVolumeSampler.cpp's own documented kContainmentEpsilon inset
        // (1e-4, applied to EVERY emitted point, jittered or not, as a defensive containment
        // safety clamp -- see that constant's comment), hence the epsilon here being a couple of
        // multiples of that inset rather than an exact-equality check.
        Check(!points.empty() && PointsNearlyEqual(points.front().position, maths::vec3{ -2.0f, -1.0f, -2.0f }, 5.0e-4f),
            "Grid mode: first lattice point is at the volume's local-space corner (identity orientation), within the containment-safety inset");

        // Every point's rotation defaults to the volume's own orientation (identity here).
        bool allRotationsIdentity = true;
        for (const pcg::PcgPoint& p : points) {
            if (!NearlyEqual(p.rotation.w, 1.0f) || !NearlyEqual(p.rotation.x, 0.0f) || !NearlyEqual(p.rotation.y, 0.0f) || !NearlyEqual(p.rotation.z, 0.0f)) {
                allRotationsIdentity = false;
                break;
            }
        }
        Check(allRotationsIdentity, "Grid mode: every point's rotation defaults to the volume's own (identity) orientation");
    }

    // -----------------------------------------------------------------------------------------
    // Grid mode with jitter: same volume/spacing as above but jitterFraction > 0. Point count
    // must stay identical to the un-jittered case (jitter perturbs POSITIONS, never adds/removes
    // lattice points), every jittered point must still satisfy containment (proves the
    // containment safety-clamp actually does its job at the boundary rows), and at least some
    // points must have actually moved off their exact lattice position (proves jitter isn't
    // silently a no-op).
    // -----------------------------------------------------------------------------------------
    void TestGridModeJitter() {
        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        volume.halfExtents = maths::vec3{ 2.0f, 1.0f, 2.0f };

        pcg::PcgVolumeSamplerParams baseParams;
        baseParams.mode = pcg::PcgVolumeSamplingMode::Grid;
        baseParams.gridSpacing = maths::vec3{ 1.0f, 1.0f, 1.0f };
        baseParams.jitterFraction = 0.0f;

        pcg::PcgVolumeSamplerParams jitteredParams = baseParams;
        jitteredParams.jitterFraction = 0.9f; // Aggressive jitter -- close to the max allowed.

        const std::vector<pcg::PcgPoint> basePoints = pcg::SampleVolume(volume, baseParams, 777u);
        const std::vector<pcg::PcgPoint> jitteredPoints = pcg::SampleVolume(volume, jitteredParams, 777u);

        Check(basePoints.size() == jitteredPoints.size(), "Grid mode with jitter: point count is unchanged by jitter");

        const size_t violations = CountContainmentViolations(volume, jitteredPoints);
        Check(violations == 0, "Grid mode with jitter: every jittered point still satisfies volume.ContainsWorldPoint(), including boundary rows");

        bool anyPointMoved = false;
        const size_t compareCount = std::min(basePoints.size(), jitteredPoints.size());
        for (size_t i = 0; i < compareCount; ++i) {
            if (!PointsNearlyEqual(basePoints[i].position, jitteredPoints[i].position, 1.0e-5f)) {
                anyPointMoved = true;
                break;
            }
        }
        Check(anyPointMoved, "Grid mode with jitter: at least one point's position actually differs from its un-jittered lattice position");
    }

    // -----------------------------------------------------------------------------------------
    // Rotated OBB: the load-bearing correctness check. Uses ASYMMETRIC half-extents (x != z) so
    // that an implementation which forgot to apply `orientation` (e.g. one that built the lattice
    // directly in world space, or transformed with an identity rotation) would produce points
    // that fail volume.ContainsWorldPoint() -- a symmetric box could accidentally still "look"
    // correct under a 90-degree rotation even with a broken transform, an asymmetric one cannot.
    // Also checks one point's exact expected world position, hand-derived from the same
    // quaternion rotation formula PcgVolumeData::ContainsWorldPoint/PcgVolumeSampler both use.
    // -----------------------------------------------------------------------------------------
    void TestGridModeRotatedOBB() {
        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 5.0f, 10.0f, -3.0f }; // Nonzero center, to also prove the center offset composes correctly with rotation.
        volume.halfExtents = maths::vec3{ 2.0f, 1.0f, 3.0f }; // Asymmetric X vs Z on purpose (see comment above).
        volume.orientation = maths::quat::FromAxisAngle(maths::vec3{ 0.0f, 1.0f, 0.0f }, maths::ToRadians(90.0f));

        pcg::PcgVolumeSamplerParams params;
        params.mode = pcg::PcgVolumeSamplingMode::Grid;
        params.gridSpacing = maths::vec3{ 1.0f, 1.0f, 1.0f };
        params.jitterFraction = 0.0f; // No jitter -- keeps the first point's position exact for the hand-derived check below.

        const std::vector<pcg::PcgPoint> points = pcg::SampleVolume(volume, params, 4242u);

        const int expectedCountX = 5; // floor(4/1)+1
        const int expectedCountY = 3; // floor(2/1)+1
        const int expectedCountZ = 7; // floor(6/1)+1
        const size_t expectedTotal = static_cast<size_t>(expectedCountX) * static_cast<size_t>(expectedCountY) * static_cast<size_t>(expectedCountZ);
        Check(points.size() == expectedTotal, "Rotated OBB, grid mode: point count is unaffected by orientation (105 = 5*3*7)");

        const size_t violations = CountContainmentViolations(volume, points);
        Check(violations == 0,
            "Rotated OBB, grid mode: every lattice point satisfies volume.ContainsWorldPoint() -- proves orientation is honored, not ignored");

        // Hand-derived exact expected position for the first point (local corner
        // (-2,-1,-3), no jitter): for this specific 90-degree-about-Y rotation,
        // quat::RotateVector(x,y,z) == (z, y, -x) (verified by direct expansion of the
        // "v + 2w(qv x v) + 2(qv x (qv x v))" formula for this exact quaternion). So local
        // (-2,-1,-3) -> rotated offset (-3,-1,2) -> world = center + offset = (5-3, 10-1, -3+2) = (2, 9, -1).
        Check(!points.empty() && PointsNearlyEqual(points.front().position, maths::vec3{ 2.0f, 9.0f, -1.0f }, 1.0e-3f),
            "Rotated OBB, grid mode: first lattice point's world position matches the hand-derived rotated-and-translated corner exactly");

        // Every point's rotation still defaults to the volume's own (now non-identity) orientation.
        bool allRotationsMatchVolume = true;
        for (const pcg::PcgPoint& p : points) {
            if (!NearlyEqual(p.rotation.x, volume.orientation.x) || !NearlyEqual(p.rotation.y, volume.orientation.y)
                || !NearlyEqual(p.rotation.z, volume.orientation.z) || !NearlyEqual(p.rotation.w, volume.orientation.w)) {
                allRotationsMatchVolume = false;
                break;
            }
        }
        Check(allRotationsMatchVolume, "Rotated OBB, grid mode: every point's rotation defaults to the volume's own non-identity orientation");
    }

    // -----------------------------------------------------------------------------------------
    // Random mode: point count is deterministically density * volume (this sampler's design
    // makes the COUNT exact/non-random -- only the individual positions are randomized -- so
    // "statistically close to the expected density*volume value" holds trivially/exactly here);
    // every sampled position must satisfy containment; and, over a reasonably large sample, the
    // mean position should land close to the volume's own center (a basic sanity check that
    // NextFloatRange's per-axis sampling isn't systematically biased toward one side).
    // -----------------------------------------------------------------------------------------
    void TestRandomModeCountAndContainment() {
        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        volume.halfExtents = maths::vec3{ 3.0f, 2.0f, 1.0f }; // Volume = 8*3*2*1 = 48 m^3.

        pcg::PcgVolumeSamplerParams params;
        params.mode = pcg::PcgVolumeSamplingMode::Random;
        params.density = 50.0f; // Expected point count = 50 * 48 = 2400.

        const std::vector<pcg::PcgPoint> points = pcg::SampleVolume(volume, params, 98765u);

        const size_t expectedCount = 2400;
        Check(points.size() == expectedCount, "Random mode: point count matches density * volume exactly (50 * 48 = 2400)");

        const size_t violations = CountContainmentViolations(volume, points);
        Check(violations == 0, "Random mode: every sampled point satisfies volume.ContainsWorldPoint()");

        // Mean position should be close to the volume's center for a large, unbiased uniform
        // sample. Tolerance (1.0 on X, well above the ~0.08 expected standard error of the mean
        // for 2400 uniform(-3,3) draws) is generous enough to not flake on RNG variance while
        // still catching an axis-swap or a systematic one-sided sampling bug.
        maths::vec3 sum{ 0.0f, 0.0f, 0.0f };
        for (const pcg::PcgPoint& p : points) {
            sum = sum + p.position;
        }
        const maths::vec3 mean = points.empty() ? maths::vec3{} : sum * (1.0f / static_cast<float>(points.size()));
        Check(NearlyEqual(mean.x, 0.0f, 1.0f) && NearlyEqual(mean.y, 0.0f, 1.0f) && NearlyEqual(mean.z, 0.0f, 1.0f),
            "Random mode: mean sampled position is statistically close to the volume's own center (unbiased per-axis sampling)");

        // Sanity: density <= 0 must yield zero points, not garbage/negative sizes.
        pcg::PcgVolumeSamplerParams zeroParams = params;
        zeroParams.density = 0.0f;
        const std::vector<pcg::PcgPoint> zeroPoints = pcg::SampleVolume(volume, zeroParams, 1u);
        Check(zeroPoints.empty(), "Random mode: zero density yields zero points");
    }

    // -----------------------------------------------------------------------------------------
    // Determinism: the hard project-wide requirement (PcgSeededRandom.h) that the same
    // (volume, params, seed) always reproduces a byte-identical point set. Checked for both
    // modes, including full per-field comparison (position, rotation, seed), not just size.
    // -----------------------------------------------------------------------------------------
    void TestDeterminism() {
        pcg::PcgVolumeData volume;
        volume.center = maths::vec3{ 1.0f, -2.0f, 3.0f };
        volume.halfExtents = maths::vec3{ 2.5f, 1.5f, 2.0f };
        volume.orientation = maths::quat::FromAxisAngle(maths::vec3{ 0.0f, 0.0f, 1.0f }, maths::ToRadians(37.0f));

        pcg::PcgVolumeSamplerParams gridParams;
        gridParams.mode = pcg::PcgVolumeSamplingMode::Grid;
        gridParams.gridSpacing = maths::vec3{ 0.7f, 0.6f, 0.5f };
        gridParams.jitterFraction = 0.6f;

        const std::vector<pcg::PcgPoint> gridA = pcg::SampleVolume(volume, gridParams, 555111u);
        const std::vector<pcg::PcgPoint> gridB = pcg::SampleVolume(volume, gridParams, 555111u);

        bool gridIdentical = gridA.size() == gridB.size();
        for (size_t i = 0; gridIdentical && i < gridA.size(); ++i) {
            gridIdentical = PointsNearlyEqual(gridA[i].position, gridB[i].position, 0.0f)
                && gridA[i].seed == gridB[i].seed
                && NearlyEqual(gridA[i].rotation.x, gridB[i].rotation.x, 0.0f)
                && NearlyEqual(gridA[i].rotation.y, gridB[i].rotation.y, 0.0f)
                && NearlyEqual(gridA[i].rotation.z, gridB[i].rotation.z, 0.0f)
                && NearlyEqual(gridA[i].rotation.w, gridB[i].rotation.w, 0.0f);
        }
        Check(gridIdentical, "Determinism (Grid mode, with jitter): repeated SampleVolume() calls produce a byte-identical point set");

        pcg::PcgVolumeSamplerParams randomParams;
        randomParams.mode = pcg::PcgVolumeSamplingMode::Random;
        randomParams.density = 20.0f;

        const std::vector<pcg::PcgPoint> randomA = pcg::SampleVolume(volume, randomParams, 909090u);
        const std::vector<pcg::PcgPoint> randomB = pcg::SampleVolume(volume, randomParams, 909090u);

        bool randomIdentical = randomA.size() == randomB.size();
        for (size_t i = 0; randomIdentical && i < randomA.size(); ++i) {
            randomIdentical = PointsNearlyEqual(randomA[i].position, randomB[i].position, 0.0f) && randomA[i].seed == randomB[i].seed;
        }
        Check(randomIdentical, "Determinism (Random mode): repeated SampleVolume() calls produce a byte-identical point set");

        // A different seed must (overwhelmingly likely) produce a different point set -- guards
        // against a broken RNG wiring that silently ignores `seed` entirely.
        const std::vector<pcg::PcgPoint> randomC = pcg::SampleVolume(volume, randomParams, 111222u);
        bool anyDifferent = randomC.size() != randomA.size();
        for (size_t i = 0; !anyDifferent && i < randomC.size(); ++i) {
            if (!PointsNearlyEqual(randomC[i].position, randomA[i].position, 1.0e-5f)) {
                anyDifferent = true;
            }
        }
        Check(anyDifferent, "Determinism: a different seed produces a different point set");
    }

} // namespace

int main() {
    TestGridModeBasicCountAndContainment();
    TestGridModeJitter();
    TestGridModeRotatedOBB();
    TestRandomModeCountAndContainment();
    TestDeterminism();

    if (g_failCount == 0) {
        std::cout << "PcgVolumeSamplerTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgVolumeSamplerTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
