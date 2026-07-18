// Standalone, framework-free unit test for the PCG framework roadmap's Phase 8.1 ("Biome Rule
// Asset") types: src/pcg/PcgBiomeRules.h/.cpp. Mirrors tests/PcgDataModelTests.cpp's own convention
// exactly (plain std::cerr on failure, exits 0/1, no Logger.cpp/Vulkan dependency) so it builds/runs
// identically in either config and needs no engine bring-up.
//
// Exercises: the "single-layer biome with just a density threshold is identical to calling
// FilterByDensity directly" equivalence (ApplyBiomeLayer's step-1/step-3..7 no-op-at-default
// guarantee), a multi-layer (Grass + Rocks) biome's independent-per-layer composition model (an
// exclusion volume on one layer never affects another layer's own output, even though both layers
// filter the SAME shared base point set), height/slope filtering actually excluding
// out-of-range points (cross-checked against direct sequential FilterByHeight/FilterBySlope calls,
// matching ApplyBiomeLayer's own documented step order), exclusion volumes actually suppressing
// points inside them (including composing TWO volumes in sequence), self-pruning within a layer
// actually thinning a dense cluster (contrasted against an unpruned sibling layer on the identical
// input), and determinism (repeated calls reproduce byte-identical output, a different seed changes
// noise-driven results, and the documented PcgHashCombine(seed, layerIndex) per-layer seed
// derivation is exactly what ApplyBiome uses internally).

#include "core/maths/Maths.h"
#include "pcg/PcgBiomeRules.h"
#include "pcg/PcgDensityTransformFilter.h"
#include "pcg/PcgSeededRandom.h"
#include "pcg/PcgSlopeHeightFilter.h"
#include "pcg/PcgSpatialData.h"
#include "pcg/PcgTerrainSampler.h"

#include <algorithm>
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

    bool NoSurvivorsTooClose(const std::vector<pcg::PcgPoint>& survivors, float minDistance) {
        for (size_t i = 0; i < survivors.size(); ++i) {
            for (size_t j = i + 1; j < survivors.size(); ++j) {
                if (DistanceBetween(survivors[i].position, survivors[j].position) < minDistance - 1.0e-3f) {
                    return false;
                }
            }
        }
        return true;
    }

    // Compares two point vectors for exact (seed + density + position) equality, in order --
    // the "byte-identical output" check used by several tests below.
    bool PointsMatchExactly(const std::vector<pcg::PcgPoint>& a, const std::vector<pcg::PcgPoint>& b) {
        if (a.size() != b.size()) { return false; }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].seed != b[i].seed || !NearlyEqual(a[i].density, b[i].density) || !NearlyEqual(DistanceBetween(a[i].position, b[i].position), 0.0f)) {
                return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------------------------------
    // A single-layer biome whose rule only sets minDensity/maxDensity (everything else left at its
    // documented no-op default) must produce EXACTLY the same result as calling FilterByDensity
    // directly on the same input.
    // ------------------------------------------------------------------------------------------
    void TestSingleLayerDensityThresholdEquivalence() {
        pcg::PcgSeededRandom rng(0x1234ABCDu);
        std::vector<pcg::PcgPoint> points;
        constexpr int kCount = 50;
        for (int i = 0; i < kCount; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ rng.NextFloatRange(-10.0f, 10.0f), rng.NextFloatRange(-10.0f, 10.0f), rng.NextFloatRange(-10.0f, 10.0f) };
            p.density = rng.NextFloat01();
            p.seed = static_cast<uint32_t>(i);
            points.push_back(p);
        }

        pcg::PcgBiomeLayerRule layer;
        layer.layerName = "DensityOnly";
        layer.minDensity = 0.3f;
        layer.maxDensity = 0.8f;
        // baseDensity (1.0), useNoiseVariation (false), minHeight/maxHeight (full range),
        // maxSlopeRadians (disabled), exclusionVolumes (empty), minSpacing (0.0) are all left at
        // their documented no-op defaults.

        const std::vector<pcg::PcgPoint> biomeResult = pcg::ApplyBiomeLayer(points, layer, 777u);
        const std::vector<pcg::PcgPoint> directResult = pcg::FilterByDensity(points, 0.3f, 0.8f);

        Check(!directResult.empty() && directResult.size() < points.size(),
            "Density-threshold equivalence setup: the [0.3, 0.8] range excludes at least one point but keeps at least one, so this test is not vacuous");
        Check(PointsMatchExactly(biomeResult, directResult),
            "ApplyBiomeLayer with only minDensity/maxDensity set produces a byte-identical result to calling FilterByDensity directly");
    }

    // ------------------------------------------------------------------------------------------
    // A multi-layer biome (Grass + Rocks, Rocks carrying an exclusion volume) must apply each
    // layer independently to the SAME shared base point set: Grass's own output must be entirely
    // unaffected by Rocks' exclusion volume, even though both layers process the identical input.
    // ------------------------------------------------------------------------------------------
    void TestMultiLayerIndependentComposition() {
        std::vector<pcg::PcgPoint> basePoints;
        for (int i = -10; i <= 10; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ static_cast<float>(i), 0.0f, 0.0f };
            p.seed = static_cast<uint32_t>(i + 10);
            basePoints.push_back(p);
        }

        pcg::PcgVolumeData exclusion;
        exclusion.center = maths::vec3{ 0.0f, 0.0f, 0.0f };
        exclusion.halfExtents = maths::vec3{ 5.0f, 5.0f, 5.0f }; // Excludes x in [-5, 5].

        pcg::PcgBiomeRuleSet ruleSet;
        ruleSet.biomeName = "GrassRocksTestBiome";

        pcg::PcgBiomeLayerRule grass;
        grass.layerName = "Grass";
        ruleSet.layers.push_back(grass);

        pcg::PcgBiomeLayerRule rocks;
        rocks.layerName = "Rocks";
        rocks.exclusionVolumes.push_back(exclusion);
        ruleSet.layers.push_back(rocks);

        const std::vector<pcg::PcgBiomeLayerResult> results = pcg::ApplyBiome(basePoints, ruleSet, 42u);

        Check(results.size() == 2, "ApplyBiome: returns exactly one PcgBiomeLayerResult per layer");
        Check(results[0].layerName == "Grass" && results[1].layerName == "Rocks",
            "ApplyBiome: results are returned in the same order as ruleSet.layers, with names echoed back verbatim");

        Check(results[0].points.size() == basePoints.size(),
            "Independence: Grass (no filtering rules of its own) keeps every base point, UNAFFECTED by Rocks' exclusion volume");

        size_t expectedRocksCount = 0;
        for (const pcg::PcgPoint& p : basePoints) {
            if (!exclusion.ContainsWorldPoint(p.position)) { ++expectedRocksCount; }
        }
        Check(expectedRocksCount > 0 && expectedRocksCount < basePoints.size(),
            "Multi-layer setup: the exclusion volume excludes some but not all base points, so this test is not vacuous");
        Check(results[1].points.size() == expectedRocksCount,
            "Rocks layer: surviving point count matches an independently-computed reference count of points outside the exclusion volume");
        for (const pcg::PcgPoint& p : results[1].points) {
            Check(!exclusion.ContainsWorldPoint(p.position), "Rocks layer: no surviving point lies inside its own exclusion volume");
        }
    }

    // ------------------------------------------------------------------------------------------
    // Height and slope filtering, composed together in one layer, must actually exclude points
    // outside their configured ranges -- cross-checked against direct sequential
    // FilterByHeight -> FilterBySlope calls (ApplyBiomeLayer's own documented step order).
    // ------------------------------------------------------------------------------------------
    void TestHeightAndSlopeFiltering() {
        std::vector<pcg::PcgPoint> points;

        pcg::PcgPoint flatInRange; // height 2 (in [0,10]), identity rotation -> 0 slope (in range).
        flatInRange.position = maths::vec3{ 0.0f, 2.0f, 0.0f };
        flatInRange.seed = 0u;
        points.push_back(flatInRange);

        pcg::PcgPoint tooHigh; // height 20, outside [0,10].
        tooHigh.position = maths::vec3{ 0.0f, 20.0f, 0.0f };
        tooHigh.seed = 1u;
        points.push_back(tooHigh);

        pcg::PcgPoint tooSteep; // height 5 (in range), but a 90-degree tilt exceeds the 30-degree slope cap.
        tooSteep.position = maths::vec3{ 0.0f, 5.0f, 0.0f };
        tooSteep.rotation = pcg::QuatFromNormal(maths::vec3{ 1.0f, 0.0f, 0.0f });
        tooSteep.seed = 2u;
        points.push_back(tooSteep);

        pcg::PcgPoint tooLow; // height -5, below the 0 minimum.
        tooLow.position = maths::vec3{ 0.0f, -5.0f, 0.0f };
        tooLow.seed = 3u;
        points.push_back(tooLow);

        pcg::PcgBiomeLayerRule layer;
        layer.layerName = "HeightSlope";
        layer.minHeight = 0.0f;
        layer.maxHeight = 10.0f;
        layer.maxSlopeRadians = maths::ToRadians(30.0f);

        const std::vector<pcg::PcgPoint> result = pcg::ApplyBiomeLayer(points, layer, 5u);

        Check(result.size() == 1 && result[0].seed == 0u,
            "Height+slope filtering: only the in-range, flat point survives (too-high, too-steep, and too-low points are all excluded)");

        // Cross-check against a direct, manually sequenced FilterByHeight -> FilterBySlope call
        // pair, matching ApplyBiomeLayer's own documented step order (3 then 4).
        const std::vector<pcg::PcgPoint> expected = pcg::FilterBySlope(pcg::FilterByHeight(points, 0.0f, 10.0f), maths::ToRadians(30.0f));
        Check(PointsMatchExactly(result, expected),
            "Height+slope filtering: ApplyBiomeLayer's result is byte-identical to a direct FilterByHeight -> FilterBySlope call pair");
    }

    // ------------------------------------------------------------------------------------------
    // Exclusion volumes must actually suppress points that fall inside them, and multiple
    // exclusion volumes on the same layer compose (points inside EITHER volume are excluded).
    // ------------------------------------------------------------------------------------------
    void TestExclusionVolumesSuppressPoints() {
        std::vector<pcg::PcgPoint> points;
        constexpr int kCount = 30;
        for (int i = 0; i < kCount; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ static_cast<float>(i), 0.0f, 0.0f };
            p.seed = static_cast<uint32_t>(i);
            points.push_back(p);
        }

        pcg::PcgVolumeData volumeA;
        volumeA.center = maths::vec3{ 5.0f, 0.0f, 0.0f };
        volumeA.halfExtents = maths::vec3{ 2.0f, 2.0f, 2.0f }; // Excludes x in [3, 7] -> 5 points (3,4,5,6,7).

        pcg::PcgVolumeData volumeB;
        volumeB.center = maths::vec3{ 20.0f, 0.0f, 0.0f };
        volumeB.halfExtents = maths::vec3{ 1.0f, 2.0f, 2.0f }; // Excludes x in [19, 21] -> 3 points (19,20,21).

        pcg::PcgBiomeLayerRule layer;
        layer.layerName = "Exclusion";
        layer.exclusionVolumes = { volumeA, volumeB };

        const std::vector<pcg::PcgPoint> result = pcg::ApplyBiomeLayer(points, layer, 9u);

        for (const pcg::PcgPoint& p : result) {
            Check(!volumeA.ContainsWorldPoint(p.position) && !volumeB.ContainsWorldPoint(p.position),
                "Exclusion volumes: no surviving point lies inside EITHER configured exclusion volume");
        }
        Check(result.size() == static_cast<size_t>(kCount - 5 - 3),
            "Exclusion volumes: two non-overlapping exclusion volumes compose additively (both sets of points are removed)");
    }

    // ------------------------------------------------------------------------------------------
    // Self-pruning within a layer must actually thin a densely packed cluster down to a set
    // respecting the configured minimum spacing, while a sibling layer with minSpacing left at
    // its 0.0 default (no pruning) passes the identical dense input straight through untouched.
    // ------------------------------------------------------------------------------------------
    void TestSelfPruningThinsDenseCluster() {
        std::vector<pcg::PcgPoint> points;
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

        pcg::PcgBiomeLayerRule prunedLayer;
        prunedLayer.layerName = "Pruned";
        prunedLayer.minSpacing = 1.5f;

        pcg::PcgBiomeLayerRule unprunedLayer;
        unprunedLayer.layerName = "Unpruned"; // minSpacing left at its 0.0 default -- pruning disabled.

        const std::vector<pcg::PcgPoint> prunedResult = pcg::ApplyBiomeLayer(points, prunedLayer, 3u);
        const std::vector<pcg::PcgPoint> unprunedResult = pcg::ApplyBiomeLayer(points, unprunedLayer, 3u);

        Check(prunedResult.size() < points.size(), "Self-pruning: a dense cluster is strictly thinned when minSpacing > 0");
        Check(NoSurvivorsTooClose(prunedResult, 1.5f), "Self-pruning: every surviving pair in the pruned layer respects the configured minSpacing");
        Check(unprunedResult.size() == points.size(), "Self-pruning: minSpacing left at its 0.0 default is a true no-op (the identical dense input passes straight through)");
    }

    // ------------------------------------------------------------------------------------------
    // Determinism: repeated ApplyBiome calls on identical input/ruleSet/seed reproduce a
    // byte-identical result; a different seed changes a noise-driven layer's density values; and
    // the documented per-layer seed derivation (PcgHashCombine(seed, layerIndex)) is exactly what
    // ApplyBiome uses internally (a direct ApplyBiomeLayer call with that derived seed reproduces
    // ApplyBiome's own per-layer result exactly).
    // ------------------------------------------------------------------------------------------
    void TestDeterminismAcrossRepeatedCalls() {
        pcg::PcgSeededRandom rng(0xDEADBEEFu);
        std::vector<pcg::PcgPoint> basePoints;
        constexpr int kCount = 40;
        for (int i = 0; i < kCount; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ rng.NextFloatRange(-20.0f, 20.0f), rng.NextFloatRange(0.0f, 20.0f), rng.NextFloatRange(-20.0f, 20.0f) };
            p.density = rng.NextFloat01();
            p.seed = static_cast<uint32_t>(i);
            basePoints.push_back(p);
        }

        pcg::PcgBiomeRuleSet ruleSet;
        ruleSet.biomeName = "DeterminismBiome";

        pcg::PcgBiomeLayerRule grass;
        grass.layerName = "Grass";
        grass.useNoiseVariation = true;
        grass.noiseFrequency = 0.1f;
        ruleSet.layers.push_back(grass);

        pcg::PcgBiomeLayerRule rocks;
        rocks.layerName = "Rocks";
        rocks.minSpacing = 3.0f;
        ruleSet.layers.push_back(rocks);

        const std::vector<pcg::PcgBiomeLayerResult> resultA = pcg::ApplyBiome(basePoints, ruleSet, 555u);
        const std::vector<pcg::PcgBiomeLayerResult> resultB = pcg::ApplyBiome(basePoints, ruleSet, 555u);

        Check(resultA.size() == 2 && resultB.size() == 2, "Determinism setup: both runs produce one result per layer");
        bool allLayersIdentical = (resultA.size() == resultB.size());
        for (size_t li = 0; allLayersIdentical && li < resultA.size(); ++li) {
            allLayersIdentical = (resultA[li].layerName == resultB[li].layerName) && PointsMatchExactly(resultA[li].points, resultB[li].points);
        }
        Check(allLayersIdentical, "Determinism: two ApplyBiome calls on identical basePoints/ruleSet/seed reproduce a byte-identical result (every layer, every point)");

        // A different biome seed must change the noise-driven Grass layer's density values --
        // sanity that the seed genuinely reaches (and is actually consumed by) the noise step.
        const std::vector<pcg::PcgBiomeLayerResult> resultDiffSeed = pcg::ApplyBiome(basePoints, ruleSet, 999u);
        bool anyDensityDiffers = false;
        const size_t grassCount = std::min(resultA[0].points.size(), resultDiffSeed[0].points.size());
        for (size_t i = 0; i < grassCount; ++i) {
            if (!NearlyEqual(resultA[0].points[i].density, resultDiffSeed[0].points[i].density, 1.0e-5f)) {
                anyDensityDiffers = true;
                break;
            }
        }
        Check(anyDensityDiffers, "Determinism: a different biome seed changes the noise-driven Grass layer's density values");

        // Layer-seed derivation contract: PcgHashCombine(seed, layerIndex), called directly and
        // fed straight into ApplyBiomeLayer, must reproduce ApplyBiome's own per-layer result.
        const uint32_t grassSeed = pcg::PcgHashCombine(555u, 0u);
        const uint32_t rocksSeed = pcg::PcgHashCombine(555u, 1u);
        const std::vector<pcg::PcgPoint> directGrass = pcg::ApplyBiomeLayer(basePoints, ruleSet.layers[0], grassSeed);
        const std::vector<pcg::PcgPoint> directRocks = pcg::ApplyBiomeLayer(basePoints, ruleSet.layers[1], rocksSeed);
        Check(PointsMatchExactly(directGrass, resultA[0].points),
            "Determinism: a direct ApplyBiomeLayer call using PcgHashCombine(seed, 0) reproduces ApplyBiome's own Grass (index 0) result exactly");
        Check(PointsMatchExactly(directRocks, resultA[1].points),
            "Determinism: a direct ApplyBiomeLayer call using PcgHashCombine(seed, 1) reproduces ApplyBiome's own Rocks (index 1) result exactly");
    }

} // namespace

int main() {
    TestSingleLayerDensityThresholdEquivalence();
    TestMultiLayerIndependentComposition();
    TestHeightAndSlopeFiltering();
    TestExclusionVolumesSuppressPoints();
    TestSelfPruningThinsDenseCluster();
    TestDeterminismAcrossRepeatedCalls();

    if (g_failCount == 0) {
        std::cout << "PcgBiomeRulesTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgBiomeRulesTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
