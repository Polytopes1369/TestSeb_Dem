// Standalone, framework-free unit test for the PCG framework roadmap's Phase 3.2
// ("Self-Pruning Filter") type: src/pcg/PcgSelfPruningFilter.h/.cpp. Exercises the uniform hash
// grid's correctness (a dense cluster prunes down to a set with a real minimum separation, a
// widely-spaced input is left untouched), the ScaledByBounds distance formula actually changing
// results vs Uniform, and a performance sanity check proving the grid acceleration -- not an
// accidental O(n^2) fallback -- is what actually runs. Exits 0 if every check passes, non-zero
// otherwise -- registered with CTest (see the top-level CMakeLists.txt), matching this project's
// existing tests/*.cpp convention (mirrors tests/PcgDataModelTests.cpp's own framework-free,
// Logger/Vulkan-independent style).

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSeededRandom.h"
#include "pcg/PcgSelfPruningFilter.h"

#include <chrono>
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

    float DistanceBetween(const maths::vec3& a, const maths::vec3& b) {
        return (a - b).Length();
    }

    // O(k^2) self-consistency check over a (presumably already small, post-pruning) point set:
    // for every distinct pair of SURVIVORS, verifies neither one is closer than the pair's own
    // effective minimum distance (the exact same ComputePairMinDistance formula PruneByDistance
    // itself uses -- reproduced locally here since that helper is file-private to
    // PcgSelfPruningFilter.cpp) to the other. This is the "internally consistent" guarantee the
    // task calls out explicitly: regardless of what order pruning happened to process points in,
    // NO two points left in its output may violate the minimum-separation contract.
    float LocalPointBoundingRadius(const pcg::PcgPoint& point) {
        const float localRadius = maths::AABBRadius(point.boundsMin, point.boundsMax);
        const float maxScale = std::max({ std::abs(point.scale.x), std::abs(point.scale.y), std::abs(point.scale.z) });
        return localRadius * maxScale;
    }

    float LocalPairMinDistance(const pcg::PcgPoint& a, const pcg::PcgPoint& b, float baseMinDistance, pcg::PcgPruningMode mode) {
        if (mode == pcg::PcgPruningMode::Uniform) {
            return baseMinDistance;
        }
        return std::max(baseMinDistance, LocalPointBoundingRadius(a) + LocalPointBoundingRadius(b));
    }

    bool NoSurvivorsTooClose(const std::vector<pcg::PcgPoint>& survivors, float baseMinDistance, pcg::PcgPruningMode mode) {
        for (size_t i = 0; i < survivors.size(); ++i) {
            for (size_t j = i + 1; j < survivors.size(); ++j) {
                const float pairMinDistance = LocalPairMinDistance(survivors[i], survivors[j], baseMinDistance, mode);
                const float dist = DistanceBetween(survivors[i].position, survivors[j].position);
                if (dist < pairMinDistance - 1.0e-3f) { // Small epsilon guards float-rounding false positives.
                    return false;
                }
            }
        }
        return true;
    }

    // ------------------------------------------------------------------------------------------
    // A dense cluster of points -- deliberately packed far closer together than `minDistance` --
    // must reduce to a set of isolated survivors, each at least `minDistance` from every other.
    // ------------------------------------------------------------------------------------------
    void TestDenseClusterPruning() {
        std::vector<pcg::PcgPoint> points;
        // A 6x6x6 lattice at 0.2 spacing (125... actually 216 points) packed into a 1.0-unit cube
        // -- every pair of adjacent lattice points is only 0.2 apart, far tighter than the 1.5
        // minDistance used below, so heavy pruning is expected.
        constexpr int kLatticeDim = 6;
        constexpr float kSpacing = 0.2f;
        uint32_t seed = 0;
        for (int ix = 0; ix < kLatticeDim; ++ix) {
            for (int iy = 0; iy < kLatticeDim; ++iy) {
                for (int iz = 0; iz < kLatticeDim; ++iz) {
                    pcg::PcgPoint p;
                    p.position = maths::vec3{ static_cast<float>(ix) * kSpacing, static_cast<float>(iy) * kSpacing, static_cast<float>(iz) * kSpacing };
                    p.seed = seed++;
                    points.push_back(p);
                }
            }
        }

        constexpr float kMinDistance = 1.5f;
        const std::vector<pcg::PcgPoint> pruned = pcg::PruneByDistance(points, kMinDistance, pcg::PcgPruningMode::Uniform);

        Check(pruned.size() < points.size(), "Dense cluster: pruning strictly reduces the point count");
        Check(!pruned.empty(), "Dense cluster: at least the first point always survives");
        Check(NoSurvivorsTooClose(pruned, kMinDistance, pcg::PcgPruningMode::Uniform),
            "Dense cluster: every surviving pair is at least minDistance apart");

        // The very first input point (index 0, position at the lattice origin) can never be
        // discarded -- nothing has been inserted into the grid yet when it is processed -- so it
        // must be present in the output (this is the documented earlier-index-always-wins rule's
        // most basic consequence).
        bool firstPointSurvived = false;
        for (const pcg::PcgPoint& p : pruned) {
            if (p.seed == 0u) {
                firstPointSurvived = true;
                break;
            }
        }
        Check(firstPointSurvived, "Dense cluster: the very first input point always survives (nothing precedes it)");
    }

    // ------------------------------------------------------------------------------------------
    // A widely-spaced input (every pair already far beyond minDistance) must pass through
    // pruning completely untouched -- same count, same points, same order.
    // ------------------------------------------------------------------------------------------
    void TestWidelySpacedNoPruning() {
        std::vector<pcg::PcgPoint> points;
        constexpr int kCount = 20;
        constexpr float kSpacing = 100.0f; // Far larger than the 1.0 minDistance used below.
        for (int i = 0; i < kCount; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ static_cast<float>(i) * kSpacing, 0.0f, 0.0f };
            p.seed = static_cast<uint32_t>(i);
            points.push_back(p);
        }

        constexpr float kMinDistance = 1.0f;
        const std::vector<pcg::PcgPoint> pruned = pcg::PruneByDistance(points, kMinDistance, pcg::PcgPruningMode::Uniform);

        Check(pruned.size() == points.size(), "Widely-spaced input: no points are pruned");
        bool orderAndValuesPreserved = true;
        for (size_t i = 0; i < pruned.size(); ++i) {
            if (pruned[i].seed != points[i].seed || !NearlyEqual(DistanceBetween(pruned[i].position, points[i].position), 0.0f)) {
                orderAndValuesPreserved = false;
                break;
            }
        }
        Check(orderAndValuesPreserved, "Widely-spaced input: surviving points keep their original values and relative order");
    }

    // ------------------------------------------------------------------------------------------
    // ScaledByBounds must actually change the result vs Uniform for points whose bounds/scale
    // differ: a large-radius point should push a nearby point out even when a plain Uniform
    // minDistance would have let both survive.
    // ------------------------------------------------------------------------------------------
    void TestScaledByBoundsChangesResult() {
        pcg::PcgPoint bigPoint;
        bigPoint.position = maths::vec3{ 0.0f, 0.0f, 0.0f };
        bigPoint.seed = 0u;
        bigPoint.scale = maths::vec3{ 10.0f, 10.0f, 10.0f }; // Default bounds (+/-0.5) * scale 10 -> a large effective radius.

        pcg::PcgPoint smallPoint;
        smallPoint.position = maths::vec3{ 5.0f, 0.0f, 0.0f }; // 5 units away from bigPoint.
        smallPoint.seed = 1u;
        // Default scale (1,1,1) and default bounds -- a small effective radius.

        const std::vector<pcg::PcgPoint> input{ bigPoint, smallPoint };
        constexpr float kBaseMinDistance = 1.0f; // Smaller than the 5-unit gap, so Uniform mode keeps both.

        const std::vector<pcg::PcgPoint> uniformResult = pcg::PruneByDistance(input, kBaseMinDistance, pcg::PcgPruningMode::Uniform);
        const std::vector<pcg::PcgPoint> scaledResult = pcg::PruneByDistance(input, kBaseMinDistance, pcg::PcgPruningMode::ScaledByBounds);

        Check(uniformResult.size() == 2, "ScaledByBounds vs Uniform: Uniform mode keeps both points (5-unit gap exceeds the 1.0 base minDistance)");
        Check(scaledResult.size() == 1, "ScaledByBounds vs Uniform: ScaledByBounds mode prunes the smaller point (bigPoint's bounding-sphere radius alone exceeds the 5-unit gap)");
        Check(uniformResult.size() != scaledResult.size(), "ScaledByBounds mode actually changes the pruning result vs Uniform mode for identical input/baseMinDistance");

        if (!scaledResult.empty()) {
            Check(scaledResult[0].seed == 0u, "ScaledByBounds: the earlier-indexed (bigger) point is the one that survives, per the documented earlier-index-wins rule");
        }
    }

    // ------------------------------------------------------------------------------------------
    // Determinism: identical input (including input order) + identical params must always
    // produce a byte-identical (same count, same values, same order) output.
    // ------------------------------------------------------------------------------------------
    void TestDeterminism() {
        pcg::PcgSeededRandom rng(0xF00DBEEFu);
        std::vector<pcg::PcgPoint> points;
        constexpr int kCount = 300;
        for (int i = 0; i < kCount; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ rng.NextFloatRange(-10.0f, 10.0f), rng.NextFloatRange(-10.0f, 10.0f), rng.NextFloatRange(-10.0f, 10.0f) };
            p.seed = static_cast<uint32_t>(i);
            points.push_back(p);
        }

        constexpr float kMinDistance = 0.75f;
        const std::vector<pcg::PcgPoint> resultA = pcg::PruneByDistance(points, kMinDistance, pcg::PcgPruningMode::Uniform);
        const std::vector<pcg::PcgPoint> resultB = pcg::PruneByDistance(points, kMinDistance, pcg::PcgPruningMode::Uniform);

        Check(resultA.size() == resultB.size(), "Determinism: two runs on identical input/params produce the same survivor count");
        bool identical = (resultA.size() == resultB.size());
        if (identical) {
            for (size_t i = 0; i < resultA.size(); ++i) {
                if (resultA[i].seed != resultB[i].seed || !NearlyEqual(DistanceBetween(resultA[i].position, resultB[i].position), 0.0f)) {
                    identical = false;
                    break;
                }
            }
        }
        Check(identical, "Determinism: two runs on identical input/params produce byte-identical (same values, same order) output");

        // Shuffling the INPUT order is explicitly allowed to change WHICH points survive (the
        // keep/discard rule is index-order-dependent by design -- see PcgSelfPruningFilter.h's
        // header comment) -- but the shuffled run's own output must still be internally
        // consistent: no surviving point may be closer than minDistance to another surviving
        // point. Uses a fixed (not randomized) reversal as the "shuffle" so this test itself stays
        // fully deterministic.
        std::vector<pcg::PcgPoint> reversedPoints(points.rbegin(), points.rend());
        const std::vector<pcg::PcgPoint> reversedResult = pcg::PruneByDistance(reversedPoints, kMinDistance, pcg::PcgPruningMode::Uniform);
        Check(NoSurvivorsTooClose(reversedResult, kMinDistance, pcg::PcgPruningMode::Uniform),
            "Determinism: reordering the input is allowed to change which points survive, but the reordered run's own output is still internally consistent");
    }

    // ------------------------------------------------------------------------------------------
    // Performance sanity check: pruning a few thousand random points must complete quickly,
    // proving the uniform hash grid is actually accelerating the search rather than an
    // accidental O(n^2) all-pairs fallback (which would be dramatically slower at this count).
    // ------------------------------------------------------------------------------------------
    void TestPerformanceSanityCheck() {
        pcg::PcgSeededRandom rng(0xABAD1DEAu);
        std::vector<pcg::PcgPoint> points;
        constexpr int kCount = 8000;
        // Scattered through a 200x200x200 volume -- large enough relative to the 1.0 minDistance
        // used below that most points are NOT clustered right on top of each other (a
        // representative "sparse-ish scatter with local pockets of density" case, similar to a
        // real Surface Sampler's output), which is exactly the case a broad-phase grid is meant
        // to handle efficiently.
        for (int i = 0; i < kCount; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ rng.NextFloatRange(-100.0f, 100.0f), rng.NextFloatRange(-100.0f, 100.0f), rng.NextFloatRange(-100.0f, 100.0f) };
            p.seed = static_cast<uint32_t>(i);
            points.push_back(p);
        }

        const auto start = std::chrono::steady_clock::now();
        const std::vector<pcg::PcgPoint> pruned = pcg::PruneByDistance(points, 1.0f, pcg::PcgPruningMode::Uniform);
        const auto end = std::chrono::steady_clock::now();
        const double elapsedSeconds = std::chrono::duration<double>(end - start).count();

        std::cout << "PcgSelfPruningFilterTests: pruned " << points.size() << " points down to " << pruned.size()
            << " survivors in " << elapsedSeconds << " seconds.\n";

        Check(pruned.size() <= points.size() && !pruned.empty(), "Performance check: pruning a large random point set still yields a sane (non-empty, not larger than input) result");
        Check(NoSurvivorsTooClose(pruned, 1.0f, pcg::PcgPruningMode::Uniform),
            "Performance check: the large random point set's survivors still respect the minimum-separation contract");
        // A generous ceiling: an O(n^2) fallback over 8000 points (64,000,000 pair checks) would
        // take multiple seconds even on fast hardware in a debug build; the grid-accelerated
        // version should finish in a small fraction of a second. 3 seconds leaves wide headroom
        // for slow/loaded CI machines while still failing hard if the grid path were silently
        // bypassed in favor of an all-pairs scan.
        Check(elapsedSeconds < 3.0, "Performance check: pruning 8000 points completes well within budget (grid acceleration is actually in effect, not an O(n^2) fallback)");
    }

} // namespace

int main() {
    TestDenseClusterPruning();
    TestWidelySpacedNoPruning();
    TestScaledByBoundsChangesResult();
    TestDeterminism();
    TestPerformanceSanityCheck();

    if (g_failCount == 0) {
        std::cout << "PcgSelfPruningFilterTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgSelfPruningFilterTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
