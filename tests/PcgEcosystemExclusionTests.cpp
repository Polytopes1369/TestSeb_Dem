// Standalone, framework-free unit test for the PCG framework roadmap's Phase 8.2 ("Ecosystem
// Exclusion") types: src/pcg/PcgEcosystemExclusion.h/.cpp. Mirrors tests/PcgBiomeRulesTests.cpp's
// (and, further back, tests/PcgDataModelTests.cpp's) own convention exactly (plain std::cerr on
// failure, exits 0/1, no Logger.cpp/Vulkan dependency) so it builds/runs identically in either
// config and needs no engine bring-up.
//
// Exercises: a suppressed-layer point directly co-located with (and one clearly far from) a
// suppressing-layer point, the exactly-at-the-radius boundary convention (inclusive -- see
// PcgEcosystemExclusion.h's own header comment), derived-radius mode actually deriving a DIFFERENT
// effective radius per suppressing point from that point's own bounds (not the flat radius),
// ApplyEcosystemRules composing TWO rules against a real ApplyBiome(...) output (Phase 8.1) and
// progressively shrinking the same suppressed layer, an unknown layer-name rule being skipped
// without crashing (and without perturbing any other layer), and determinism (repeated calls on
// identical input reproduce a byte-identical, order-stable result).

#include "core/maths/Maths.h"
#include "pcg/PcgBiomeRules.h"
#include "pcg/PcgEcosystemExclusion.h"

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

    float DistanceBetween(const maths::vec3& a, const maths::vec3& b) {
        return (a - b).Length();
    }

    // Compares two point vectors for exact (seed + position) equality, in order -- the
    // "byte-identical, order-stable output" check used by this file's determinism tests.
    bool PointsMatchExactly(const std::vector<pcg::PcgPoint>& a, const std::vector<pcg::PcgPoint>& b) {
        if (a.size() != b.size()) { return false; }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].seed != b[i].seed || DistanceBetween(a[i].position, b[i].position) > 1.0e-4f) {
                return false;
            }
        }
        return true;
    }

    pcg::PcgPoint MakePoint(const maths::vec3& position, uint32_t seed, float density = 1.0f) {
        pcg::PcgPoint p;
        p.position = position;
        p.seed = seed;
        p.density = density;
        // boundsMin/boundsMax/scale left at PcgPoint's own documented defaults (unit cube, scale
        // 1.0) unless a specific test below overrides them for derived-radius purposes.
        return p;
    }

    // ------------------------------------------------------------------------------------------
    // A suppressed-layer point exactly co-located with a suppressing-layer point (distance 0, well
    // inside any positive radius) must be removed; a suppressed-layer point far outside the radius
    // must survive untouched.
    // ------------------------------------------------------------------------------------------
    void TestDirectlyUnderIsRemovedFarAwaySurvives() {
        std::vector<pcg::PcgPoint> suppressing = { MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 100u) };

        std::vector<pcg::PcgPoint> suppressed = {
            MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 1u),   // Directly under -- distance 0.
            MakePoint(maths::vec3{ 1000.0f, 0.0f, 0.0f }, 2u), // Far away.
        };

        pcg::PcgEcosystemExclusionRule rule;
        rule.suppressingLayerName = "Trees";
        rule.suppressedLayerName = "Bushes";
        rule.exclusionRadius = 3.0f;

        const std::vector<pcg::PcgPoint> result = pcg::ApplyEcosystemExclusion(suppressed, suppressing, rule);

        Check(result.size() == 1 && result[0].seed == 2u,
            "Direct-under/far-away: the co-located point is removed, the far point survives untouched");
    }

    // ------------------------------------------------------------------------------------------
    // Boundary convention: a suppressed point at EXACTLY the exclusion radius from a suppressing
    // point is treated as INSIDE the footprint (removed, inclusive <=); a point a hair beyond the
    // radius survives.
    // ------------------------------------------------------------------------------------------
    void TestBoundaryAtExactRadius() {
        std::vector<pcg::PcgPoint> suppressing = { MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 100u) };

        constexpr float kRadius = 5.0f;
        std::vector<pcg::PcgPoint> suppressed = {
            MakePoint(maths::vec3{ kRadius, 0.0f, 0.0f }, 1u),          // Exactly at the radius.
            MakePoint(maths::vec3{ kRadius + 0.01f, 0.0f, 0.0f }, 2u),  // Just beyond the radius.
        };

        pcg::PcgEcosystemExclusionRule rule;
        rule.suppressingLayerName = "Trees";
        rule.suppressedLayerName = "Bushes";
        rule.exclusionRadius = kRadius;

        const std::vector<pcg::PcgPoint> result = pcg::ApplyEcosystemExclusion(suppressed, suppressing, rule);

        Check(result.size() == 1 && result[0].seed == 2u,
            "Boundary: a point EXACTLY at the exclusion radius is removed (inclusive convention), a point just beyond it survives");
    }

    // ------------------------------------------------------------------------------------------
    // Derived-radius mode must actually derive a DIFFERENT effective radius per suppressing point
    // from THAT point's own local-space bounds -- a point at distance 4 from a big-canopy
    // suppressing point (derived radius > 4) is suppressed, while an identically-distanced-4 point
    // near a small-canopy suppressing point (derived radius < 4) survives.
    // ------------------------------------------------------------------------------------------
    void TestDerivedRadiusVariesByBounds() {
        pcg::PcgPoint bigTree = MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 10u);
        bigTree.boundsMin = maths::vec3{ -3.0f, -3.0f, -3.0f };
        bigTree.boundsMax = maths::vec3{ 3.0f, 3.0f, 3.0f };
        // AABBRadius = length(boundsMax - boundsMin) * 0.5 = length(6,6,6) * 0.5 = 3*sqrt(3) ~= 5.196.

        pcg::PcgPoint smallTree = MakePoint(maths::vec3{ 100.0f, 0.0f, 0.0f }, 11u);
        // Left at PcgPoint's own default unit-cube bounds: AABBRadius = length(1,1,1) * 0.5 ~= 0.866.

        std::vector<pcg::PcgPoint> suppressing = { bigTree, smallTree };

        std::vector<pcg::PcgPoint> suppressed = {
            MakePoint(maths::vec3{ 4.0f, 0.0f, 0.0f }, 1u),   // Distance 4 from bigTree (radius ~5.196 > 4 -> suppressed).
            MakePoint(maths::vec3{ 96.0f, 0.0f, 0.0f }, 2u),  // Distance 4 from smallTree (radius ~0.866 < 4 -> survives).
        };

        pcg::PcgEcosystemExclusionRule rule;
        rule.suppressingLayerName = "Trees";
        rule.suppressedLayerName = "Bushes";
        rule.useDerivedRadius = true;
        rule.derivedRadiusMultiplier = 1.0f;
        // rule.exclusionRadius intentionally left at its documented-unused 0.0 default in derived mode.

        const std::vector<pcg::PcgPoint> result = pcg::ApplyEcosystemExclusion(suppressed, suppressing, rule);

        Check(result.size() == 1 && result[0].seed == 2u,
            "Derived radius: the point near the BIG canopy (distance 4 < ~5.196) is suppressed, "
            "the identically-distanced point near the SMALL canopy (distance 4 > ~0.866) survives -- "
            "proving the effective radius genuinely varies per suppressing point's own bounds, not a flat value");

        // Sanity: the SAME two suppressed points under a flat (non-derived) radius smaller than
        // both distances would suppress NEITHER -- confirming the difference above is really
        // attributable to derived-radius mode, not some other effect.
        pcg::PcgEcosystemExclusionRule flatRule;
        flatRule.suppressingLayerName = "Trees";
        flatRule.suppressedLayerName = "Bushes";
        flatRule.useDerivedRadius = false;
        flatRule.exclusionRadius = 1.0f;
        const std::vector<pcg::PcgPoint> flatResult = pcg::ApplyEcosystemExclusion(suppressed, suppressing, flatRule);
        Check(flatResult.size() == 2, "Derived-radius control: a flat radius (1.0) smaller than both test distances (4.0) suppresses neither point");
    }

    // ------------------------------------------------------------------------------------------
    // End-to-end: build a real 2-layer PcgBiomeRuleSet, run Phase 8.1's ApplyBiome, then apply TWO
    // composed PcgEcosystemExclusionRules against its output, confirming the suppressed layer
    // shrinks progressively (each rule's surviving output feeds the next rule targeting the same
    // layer) and unrelated layers are left untouched.
    //
    // Setup: 20 base points at x = 0..19 (y = z = 0), density alternating 1.0 (even x) / 0.0 (odd
    // x). "Trees" layer keeps only density in [0.5, 1.0] -> the 10 even-x points. "Rocks" layer
    // keeps only density in [0.0, 0.0] -> the 10 odd-x points. "Bushes" layer keeps density in
    // [0.0, 1.0] (everything) -> all 20 points, independently of Trees/Rocks (Phase 8.1's
    // independent-layers model).
    //
    // Rule 1: Trees suppress Bushes within radius 0.5 -> since spacing between adjacent base points
    // is 1.0, this only removes Bushes points EXACTLY co-located with a Tree (the 10 even-x
    // points), leaving the 10 odd-x Bushes points (20 -> 10).
    // Rule 2: Rocks suppress Bushes within radius 0.5 -> Rocks are exactly the 10 odd-x points,
    // which are now EXACTLY the remaining Bushes points from rule 1 -- every one is co-located with
    // a Rock (distance 0), so this second rule empties Bushes entirely (10 -> 0), demonstrating that
    // rule 2 operates on rule 1's ALREADY-thinned Bushes output, not the original ApplyBiome result.
    // ------------------------------------------------------------------------------------------
    void TestApplyEcosystemRulesEndToEndWithApplyBiome() {
        std::vector<pcg::PcgPoint> basePoints;
        constexpr int kCount = 20;
        for (int i = 0; i < kCount; ++i) {
            const float density = (i % 2 == 0) ? 1.0f : 0.0f;
            basePoints.push_back(MakePoint(maths::vec3{ static_cast<float>(i), 0.0f, 0.0f }, static_cast<uint32_t>(i), density));
        }

        pcg::PcgBiomeRuleSet ruleSet;
        ruleSet.biomeName = "EcosystemTestBiome";

        pcg::PcgBiomeLayerRule trees;
        trees.layerName = "Trees";
        trees.minDensity = 0.5f;
        trees.maxDensity = 1.0f;
        ruleSet.layers.push_back(trees);

        pcg::PcgBiomeLayerRule rocks;
        rocks.layerName = "Rocks";
        rocks.minDensity = 0.0f;
        rocks.maxDensity = 0.0f;
        ruleSet.layers.push_back(rocks);

        pcg::PcgBiomeLayerRule bushes;
        bushes.layerName = "Bushes";
        bushes.minDensity = 0.0f;
        bushes.maxDensity = 1.0f;
        ruleSet.layers.push_back(bushes);

        std::vector<pcg::PcgBiomeLayerResult> biomeResult = pcg::ApplyBiome(basePoints, ruleSet, 1234u);

        Check(biomeResult.size() == 3, "End-to-end setup: ApplyBiome returns exactly 3 layer results");
        auto findByName = [&biomeResult](const std::string& name) -> const pcg::PcgBiomeLayerResult* {
            for (const auto& r : biomeResult) { if (r.layerName == name) { return &r; } }
            return nullptr;
        };
        const pcg::PcgBiomeLayerResult* treesBefore = findByName("Trees");
        const pcg::PcgBiomeLayerResult* rocksBefore = findByName("Rocks");
        const pcg::PcgBiomeLayerResult* bushesBefore = findByName("Bushes");
        Check(treesBefore != nullptr && treesBefore->points.size() == 10, "End-to-end setup: Trees keeps exactly the 10 even-x (density 1.0) base points");
        Check(rocksBefore != nullptr && rocksBefore->points.size() == 10, "End-to-end setup: Rocks keeps exactly the 10 odd-x (density 0.0) base points");
        Check(bushesBefore != nullptr && bushesBefore->points.size() == 20, "End-to-end setup: Bushes (full [0,1] density range) independently keeps all 20 base points");

        std::vector<pcg::PcgEcosystemExclusionRule> rules;
        pcg::PcgEcosystemExclusionRule ruleTreesVsBushes;
        ruleTreesVsBushes.suppressingLayerName = "Trees";
        ruleTreesVsBushes.suppressedLayerName = "Bushes";
        ruleTreesVsBushes.exclusionRadius = 0.5f;
        rules.push_back(ruleTreesVsBushes);

        pcg::PcgEcosystemExclusionRule ruleRocksVsBushes;
        ruleRocksVsBushes.suppressingLayerName = "Rocks";
        ruleRocksVsBushes.suppressedLayerName = "Bushes";
        ruleRocksVsBushes.exclusionRadius = 0.5f;
        rules.push_back(ruleRocksVsBushes);

        const std::vector<pcg::PcgBiomeLayerResult> afterExclusion = pcg::ApplyEcosystemRules(biomeResult, rules);

        Check(afterExclusion.size() == 3, "End-to-end: ApplyEcosystemRules preserves the layer count/order contract");
        auto findByNameAfter = [&afterExclusion](const std::string& name) -> const pcg::PcgBiomeLayerResult* {
            for (const auto& r : afterExclusion) { if (r.layerName == name) { return &r; } }
            return nullptr;
        };
        const pcg::PcgBiomeLayerResult* treesAfter = findByNameAfter("Trees");
        const pcg::PcgBiomeLayerResult* rocksAfter = findByNameAfter("Rocks");
        const pcg::PcgBiomeLayerResult* bushesAfter = findByNameAfter("Bushes");

        Check(treesAfter != nullptr && treesAfter->points.size() == 10, "End-to-end: Trees (never a suppressed layer in this rule list) is completely untouched");
        Check(rocksAfter != nullptr && rocksAfter->points.size() == 10, "End-to-end: Rocks (never a suppressed layer in this rule list) is completely untouched");
        Check(bushesAfter != nullptr && bushesAfter->points.empty(),
            "End-to-end composition: Bushes shrinks 20 -> 10 (rule 1, suppressed by co-located Trees) -> 0 (rule 2, suppressed by co-located Rocks, "
            "which are exactly rule 1's own 10 survivors) -- proving rule 2 reads rule 1's ALREADY-thinned Bushes output, not the original ApplyBiome result");
    }

    // ------------------------------------------------------------------------------------------
    // A rule naming an unknown layer (either side) must be skipped without crashing, and must not
    // perturb any OTHER layer's points -- including a valid rule elsewhere in the same rules list.
    // ------------------------------------------------------------------------------------------
    void TestUnknownLayerNameRuleSkippedWithoutCrash() {
        std::vector<pcg::PcgBiomeLayerResult> results;
        pcg::PcgBiomeLayerResult trees;
        trees.layerName = "Trees";
        trees.points = { MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 1u) };
        results.push_back(trees);

        pcg::PcgBiomeLayerResult bushes;
        bushes.layerName = "Bushes";
        bushes.points = { MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 2u), MakePoint(maths::vec3{ 1000.0f, 0.0f, 0.0f }, 3u) };
        results.push_back(bushes);

        std::vector<pcg::PcgEcosystemExclusionRule> rules;

        pcg::PcgEcosystemExclusionRule unknownSuppressing;
        unknownSuppressing.suppressingLayerName = "DoesNotExist";
        unknownSuppressing.suppressedLayerName = "Bushes";
        unknownSuppressing.exclusionRadius = 5.0f;
        rules.push_back(unknownSuppressing);

        pcg::PcgEcosystemExclusionRule unknownSuppressed;
        unknownSuppressed.suppressingLayerName = "Trees";
        unknownSuppressed.suppressedLayerName = "AlsoDoesNotExist";
        unknownSuppressed.exclusionRadius = 5.0f;
        rules.push_back(unknownSuppressed);

        // A valid rule sandwiched between the two invalid ones -- must still run normally.
        pcg::PcgEcosystemExclusionRule validRule;
        validRule.suppressingLayerName = "Trees";
        validRule.suppressedLayerName = "Bushes";
        validRule.exclusionRadius = 5.0f;
        rules.push_back(validRule);

        const std::vector<pcg::PcgBiomeLayerResult> after = pcg::ApplyEcosystemRules(results, rules);

        Check(after.size() == 2, "Unknown layer names: no layer is added or removed from the result table");
        const pcg::PcgBiomeLayerResult* bushesAfter = nullptr;
        for (const auto& r : after) { if (r.layerName == "Bushes") { bushesAfter = &r; } }
        Check(bushesAfter != nullptr && bushesAfter->points.size() == 1 && bushesAfter->points[0].seed == 3u,
            "Unknown layer names: the two invalid rules are skipped (no crash), while the valid rule sandwiched between them still runs "
            "and correctly suppresses the co-located Bushes point, leaving only the far one");
    }

    // ------------------------------------------------------------------------------------------
    // Determinism: repeated ApplyEcosystemExclusion/ApplyEcosystemRules calls on identical input
    // reproduce a byte-identical, order-stable result.
    // ------------------------------------------------------------------------------------------
    void TestDeterminismAcrossRepeatedCalls() {
        std::vector<pcg::PcgPoint> suppressing = {
            MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 10u),
            MakePoint(maths::vec3{ 20.0f, 0.0f, 0.0f }, 11u),
        };
        std::vector<pcg::PcgPoint> suppressed;
        for (int i = 0; i < 15; ++i) {
            suppressed.push_back(MakePoint(maths::vec3{ static_cast<float>(i) * 1.7f, static_cast<float>(i) * 0.3f, 0.0f }, static_cast<uint32_t>(i)));
        }

        pcg::PcgEcosystemExclusionRule rule;
        rule.suppressingLayerName = "Trees";
        rule.suppressedLayerName = "Bushes";
        rule.exclusionRadius = 2.5f;

        const std::vector<pcg::PcgPoint> resultA = pcg::ApplyEcosystemExclusion(suppressed, suppressing, rule);
        const std::vector<pcg::PcgPoint> resultB = pcg::ApplyEcosystemExclusion(suppressed, suppressing, rule);

        Check(!resultA.empty() && resultA.size() < suppressed.size(), "Determinism setup: this configuration suppresses at least one but not all points (not vacuous)");
        Check(PointsMatchExactly(resultA, resultB), "Determinism: two ApplyEcosystemExclusion calls on identical input reproduce a byte-identical, order-stable result");

        // Order stability: survivors must appear in the SAME relative order as the original input.
        size_t searchStart = 0;
        bool orderPreserved = true;
        for (const pcg::PcgPoint& survivor : resultA) {
            bool found = false;
            for (size_t i = searchStart; i < suppressed.size(); ++i) {
                if (suppressed[i].seed == survivor.seed) {
                    searchStart = i + 1;
                    found = true;
                    break;
                }
            }
            if (!found) { orderPreserved = false; break; }
        }
        Check(orderPreserved, "Determinism: surviving points appear in the same relative order as the original suppressed-layer input");
    }

} // namespace

int main() {
    TestDirectlyUnderIsRemovedFarAwaySurvives();
    TestBoundaryAtExactRadius();
    TestDerivedRadiusVariesByBounds();
    TestApplyEcosystemRulesEndToEndWithApplyBiome();
    TestUnknownLayerNameRuleSkippedWithoutCrash();
    TestDeterminismAcrossRepeatedCalls();

    if (g_failCount == 0) {
        std::cout << "PcgEcosystemExclusionTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgEcosystemExclusionTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
