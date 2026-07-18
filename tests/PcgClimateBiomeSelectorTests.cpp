// Standalone, framework-free unit test for the PCG framework roadmap's Phase 8.3 ("Climate-Driven
// Biome Selection") types: src/pcg/PcgClimateBiomeSelector.h/.cpp. Mirrors tests/PcgDataModelTests.cpp
// / tests/PcgBiomeRulesTests.cpp's own convention (plain std::cerr on failure, exits 0/1, no test
// framework dependency) for every check EXCEPT TestSampleCurrentClimateSanity, which -- because
// SampleCurrentClimate's whole point is to read a real renderer::AtmosClimatePass -- constructs one
// real (never-Init()'d) instance; see this file's own CMakeLists.txt entry for exactly why that is
// safe without a live VkDevice/VkQueue/VmaAllocator (GpuBuffer's destructor no-ops on a
// never-Create()'d buffer) and what extra link-time dependencies it costs (GpuBuffer.cpp/VmaUsage.cpp/
// VulkanUtils.cpp/Logger.cpp/AtmosClimatePass.cpp, on top of every other Pcg*Tests target's own Phase
// 3 filter .cpp set).
//
// Exercises: ComputeBiomeWeight's soft-falloff formula (inside a range, exactly at an edge, partially
// past an edge within the falloff margin, and fully past the margin -> exactly 0), highest-weight
// biome selection among 2-3 climate-tagged candidates across a few different climate samples
// (cross-checked against a direct pcg::ApplyBiome call on the expected winning ruleSet), tie-breaking
// by input order, the "no viable biome for this climate" empty-result case (including an empty
// candidate list), determinism (repeated calls reproduce byte-identical output, same seed passed
// straight through to ApplyBiome), and -- when buildable, see above -- a sanity check that
// SampleCurrentClimate returns plausible, deterministic values from a real (default, never-Init()'d)
// renderer::AtmosClimatePass instance without crashing.

#include "core/EngineConfig.h"
#include "core/maths/Maths.h"
#include "pcg/PcgClimateBiomeSelector.h"
#include "renderer/passes/AtmosClimatePass.h"

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

    // Same "exact seed + density + position equality, in order" comparison
    // tests/PcgBiomeRulesTests.cpp's own PointsMatchExactly uses -- reimplemented locally since that
    // helper lives in another translation unit's anonymous namespace, not a shared header.
    bool PointsMatchExactly(const std::vector<pcg::PcgPoint>& a, const std::vector<pcg::PcgPoint>& b) {
        if (a.size() != b.size()) { return false; }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].seed != b[i].seed || !NearlyEqual(a[i].density, b[i].density) || !NearlyEqual(DistanceBetween(a[i].position, b[i].position), 0.0f)) {
                return false;
            }
        }
        return true;
    }

    bool LayerResultsMatchExactly(const std::vector<pcg::PcgBiomeLayerResult>& a, const std::vector<pcg::PcgBiomeLayerResult>& b) {
        if (a.size() != b.size()) { return false; }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].layerName != b[i].layerName || !PointsMatchExactly(a[i].points, b[i].points)) {
                return false;
            }
        }
        return true;
    }

    std::vector<pcg::PcgPoint> MakeGridPoints(int count) {
        std::vector<pcg::PcgPoint> points;
        points.reserve(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            pcg::PcgPoint p;
            p.position = maths::vec3{ static_cast<float>(i), 0.0f, 0.0f };
            p.seed = static_cast<uint32_t>(i);
            points.push_back(p);
        }
        return points;
    }

    // ------------------------------------------------------------------------------------------
    // ComputeBiomeWeight: inside a range -> full baseWeight; exactly at an edge (inclusive) -> full
    // baseWeight; partially past an edge within the falloff margin -> a value strictly between 0 and
    // baseWeight, matching the documented linear-ramp formula exactly; fully past the margin -> 0.
    // ------------------------------------------------------------------------------------------
    void TestComputeBiomeWeightInsideOutsideEdge() {
        pcg::PcgClimateBiomeWeight rule;
        rule.biomeName = "TemperateRule";
        rule.minTemperature = 10.0f;
        rule.maxTemperature = 20.0f; // range width 10 -> falloff margin = 10 * 0.25 = 2.5.
        rule.minMoisture = 0.3f;
        rule.maxMoisture = 0.7f;
        rule.baseWeight = 2.0f;

        pcg::PcgClimateSample inside{ 15.0f, 0.5f };
        Check(NearlyEqual(pcg::ComputeBiomeWeight(inside, rule), 2.0f),
            "ComputeBiomeWeight: climate strictly inside both ranges yields the full baseWeight");

        pcg::PcgClimateSample atEdge{ 20.0f, 0.5f }; // temperature exactly at maxTemperature.
        Check(NearlyEqual(pcg::ComputeBiomeWeight(atEdge, rule), 2.0f),
            "ComputeBiomeWeight: climate exactly at a range edge (inclusive) yields the full baseWeight");

        pcg::PcgClimateSample atOtherEdge{ 10.0f, 0.3f }; // both at their min edges.
        Check(NearlyEqual(pcg::ComputeBiomeWeight(atOtherEdge, rule), 2.0f),
            "ComputeBiomeWeight: climate exactly at BOTH min edges simultaneously yields the full baseWeight");

        // 1 degree past maxTemperature (21), margin is 2.5 -> temperatureFit = 1 - 1/2.5 = 0.6.
        pcg::PcgClimateSample justOutside{ 21.0f, 0.5f };
        const float expectedPartial = 2.0f * 0.6f;
        Check(NearlyEqual(pcg::ComputeBiomeWeight(justOutside, rule), expectedPartial, 1.0e-3f),
            "ComputeBiomeWeight: climate partially past an edge (within the falloff margin) yields the documented linear-ramp value");
        const float partialWeight = pcg::ComputeBiomeWeight(justOutside, rule);
        Check(partialWeight > 0.0f && partialWeight < rule.baseWeight,
            "ComputeBiomeWeight: a partial-falloff weight is strictly between 0 and baseWeight");

        // 5 degrees past maxTemperature -- well beyond the 2.5 margin -> exactly 0.
        pcg::PcgClimateSample farOutside{ 25.0f, 0.5f };
        Check(NearlyEqual(pcg::ComputeBiomeWeight(farOutside, rule), 0.0f),
            "ComputeBiomeWeight: climate far past the falloff margin on one axis yields exactly 0, regardless of the other axis' fit");

        // Moisture axis pushed out of range too (both axes fail) -- still exactly 0 (not negative,
        // not double-penalized below 0).
        pcg::PcgClimateSample bothOutside{ 25.0f, 0.95f };
        Check(NearlyEqual(pcg::ComputeBiomeWeight(bothOutside, rule), 0.0f),
            "ComputeBiomeWeight: climate far outside BOTH axes still yields exactly 0 (not negative)");
    }

    // ------------------------------------------------------------------------------------------
    // SelectAndApplyClimateBiome: among 3 candidate biomes tagged for cold/temperate/hot climates,
    // the correct highest-weight biome is picked for a few different climate samples, and its
    // returned per-layer results are byte-identical to a direct pcg::ApplyBiome call on that same
    // winning ruleSet.
    // ------------------------------------------------------------------------------------------
    void TestSelectAndApplyClimateBiomePicksHighestWeight() {
        const std::vector<pcg::PcgPoint> basePoints = MakeGridPoints(20);

        pcg::PcgBiomeRuleSet coldRuleSet;
        coldRuleSet.biomeName = "ColdTundra";
        pcg::PcgBiomeLayerRule coldLayer;
        coldLayer.layerName = "Scrub";
        coldLayer.minDensity = 0.0f;
        coldLayer.maxDensity = 1.0f;
        coldRuleSet.layers.push_back(coldLayer);
        pcg::PcgClimateBiomeWeight coldWeight;
        coldWeight.biomeName = "ColdTundra";
        coldWeight.minTemperature = -20.0f; coldWeight.maxTemperature = 0.0f;
        coldWeight.minMoisture = 0.0f; coldWeight.maxMoisture = 1.0f;
        coldWeight.baseWeight = 1.0f;

        pcg::PcgBiomeRuleSet temperateRuleSet;
        temperateRuleSet.biomeName = "TemperateForest";
        pcg::PcgBiomeLayerRule temperateLayer;
        temperateLayer.layerName = "Grass";
        temperateRuleSet.layers.push_back(temperateLayer);
        pcg::PcgClimateBiomeWeight temperateWeight;
        temperateWeight.biomeName = "TemperateForest";
        temperateWeight.minTemperature = 5.0f; temperateWeight.maxTemperature = 25.0f;
        temperateWeight.minMoisture = 0.3f; temperateWeight.maxMoisture = 0.8f;
        temperateWeight.baseWeight = 1.0f;

        pcg::PcgBiomeRuleSet desertRuleSet;
        desertRuleSet.biomeName = "HotDesert";
        pcg::PcgBiomeLayerRule desertLayer;
        desertLayer.layerName = "Sand";
        desertRuleSet.layers.push_back(desertLayer);
        pcg::PcgClimateBiomeWeight desertWeight;
        desertWeight.biomeName = "HotDesert";
        desertWeight.minTemperature = 25.0f; desertWeight.maxTemperature = 45.0f;
        desertWeight.minMoisture = 0.0f; desertWeight.maxMoisture = 0.2f;
        desertWeight.baseWeight = 1.0f;

        const std::vector<std::pair<pcg::PcgBiomeRuleSet, pcg::PcgClimateBiomeWeight>> candidates = {
            { coldRuleSet, coldWeight }, { temperateRuleSet, temperateWeight }, { desertRuleSet, desertWeight }
        };

        // Case 1: clearly cold climate.
        {
            std::string selected;
            const pcg::PcgClimateSample cold{ -10.0f, 0.5f };
            const std::vector<pcg::PcgBiomeLayerResult> result = pcg::SelectAndApplyClimateBiome(basePoints, cold, candidates, 123u, &selected);
            Check(selected == "ColdTundra", "SelectAndApplyClimateBiome: a clearly cold climate selects the ColdTundra biome");
            const std::vector<pcg::PcgBiomeLayerResult> expected = pcg::ApplyBiome(basePoints, coldRuleSet, 123u);
            Check(LayerResultsMatchExactly(result, expected),
                "SelectAndApplyClimateBiome: ColdTundra selection's returned points are byte-identical to a direct ApplyBiome(basePoints, coldRuleSet, seed) call");
        }

        // Case 2: clearly temperate/humid climate.
        {
            std::string selected;
            const pcg::PcgClimateSample temperate{ 15.0f, 0.5f };
            const std::vector<pcg::PcgBiomeLayerResult> result = pcg::SelectAndApplyClimateBiome(basePoints, temperate, candidates, 456u, &selected);
            Check(selected == "TemperateForest", "SelectAndApplyClimateBiome: a clearly temperate/humid climate selects the TemperateForest biome");
            const std::vector<pcg::PcgBiomeLayerResult> expected = pcg::ApplyBiome(basePoints, temperateRuleSet, 456u);
            Check(LayerResultsMatchExactly(result, expected),
                "SelectAndApplyClimateBiome: TemperateForest selection's returned points are byte-identical to a direct ApplyBiome call");
        }

        // Case 3: clearly hot/dry climate.
        {
            std::string selected;
            const pcg::PcgClimateSample hotDry{ 40.0f, 0.05f };
            const std::vector<pcg::PcgBiomeLayerResult> result = pcg::SelectAndApplyClimateBiome(basePoints, hotDry, candidates, 789u, &selected);
            Check(selected == "HotDesert", "SelectAndApplyClimateBiome: a clearly hot/dry climate selects the HotDesert biome");
            const std::vector<pcg::PcgBiomeLayerResult> expected = pcg::ApplyBiome(basePoints, desertRuleSet, 789u);
            Check(LayerResultsMatchExactly(result, expected),
                "SelectAndApplyClimateBiome: HotDesert selection's returned points are byte-identical to a direct ApplyBiome call");
        }
    }

    // ------------------------------------------------------------------------------------------
    // Tie-breaking: two candidates crafted to score an EXACTLY equal weight for the same climate
    // sample -- the first one in `candidateBiomes`' own order must win.
    // ------------------------------------------------------------------------------------------
    void TestTieBreakByInputOrder() {
        const std::vector<pcg::PcgPoint> basePoints = MakeGridPoints(10);

        pcg::PcgBiomeRuleSet ruleSetA;
        ruleSetA.biomeName = "FirstInOrder";
        pcg::PcgBiomeRuleSet ruleSetB;
        ruleSetB.biomeName = "SecondInOrder";

        // Identical eligibility ranges and baseWeight -- both fully eligible (fit = 1.0 on both
        // axes) for the same climate sample, so both score EXACTLY baseWeight (1.0), a genuine tie.
        pcg::PcgClimateBiomeWeight weightA;
        weightA.biomeName = "FirstInOrder";
        weightA.minTemperature = 0.0f; weightA.maxTemperature = 30.0f;
        weightA.minMoisture = 0.0f; weightA.maxMoisture = 1.0f;
        weightA.baseWeight = 1.0f;

        pcg::PcgClimateBiomeWeight weightB = weightA;
        weightB.biomeName = "SecondInOrder";

        const pcg::PcgClimateSample climate{ 15.0f, 0.5f };
        Check(NearlyEqual(pcg::ComputeBiomeWeight(climate, weightA), pcg::ComputeBiomeWeight(climate, weightB)),
            "Tie-break setup: both candidates genuinely score an identical weight for this climate sample");

        std::string selected;
        const std::vector<std::pair<pcg::PcgBiomeRuleSet, pcg::PcgClimateBiomeWeight>> candidates = {
            { ruleSetA, weightA }, { ruleSetB, weightB }
        };
        pcg::SelectAndApplyClimateBiome(basePoints, climate, candidates, 1u, &selected);
        Check(selected == "FirstInOrder", "SelectAndApplyClimateBiome: an exact weight tie is broken by input order (the first candidate wins)");

        // Reversing the input order must flip which biome wins -- proves the tie-break really is
        // order-driven, not some other hidden bias (e.g. alphabetical, or always-first-declared).
        std::string selectedReversed;
        const std::vector<std::pair<pcg::PcgBiomeRuleSet, pcg::PcgClimateBiomeWeight>> reversedCandidates = {
            { ruleSetB, weightB }, { ruleSetA, weightA }
        };
        pcg::SelectAndApplyClimateBiome(basePoints, climate, reversedCandidates, 1u, &selectedReversed);
        Check(selectedReversed == "SecondInOrder", "SelectAndApplyClimateBiome: reversing candidate input order flips which tied biome wins");
    }

    // ------------------------------------------------------------------------------------------
    // "No viable biome for this climate": every candidate scores exactly 0 -> empty result, empty
    // selected name, no arbitrary fallback pick. Also covers an entirely empty candidate list.
    // ------------------------------------------------------------------------------------------
    void TestNoViableBiomeReturnsEmpty() {
        const std::vector<pcg::PcgPoint> basePoints = MakeGridPoints(10);

        pcg::PcgBiomeRuleSet ruleSet;
        ruleSet.biomeName = "OnlyCandidate";
        pcg::PcgClimateBiomeWeight weight;
        weight.biomeName = "OnlyCandidate";
        weight.minTemperature = 0.0f; weight.maxTemperature = 10.0f; // range width 10 -> margin 2.5.
        weight.minMoisture = 0.0f; weight.maxMoisture = 1.0f;
        weight.baseWeight = 1.0f;

        const std::vector<std::pair<pcg::PcgBiomeRuleSet, pcg::PcgClimateBiomeWeight>> candidates = { { ruleSet, weight } };

        // 100C is far past maxTemperature=10 + its 2.5 margin -> weight is exactly 0.
        std::string selected = "untouched-sentinel";
        const pcg::PcgClimateSample impossibleClimate{ 100.0f, 0.5f };
        const std::vector<pcg::PcgBiomeLayerResult> result = pcg::SelectAndApplyClimateBiome(basePoints, impossibleClimate, candidates, 42u, &selected);
        Check(result.empty(), "SelectAndApplyClimateBiome: a climate matching no candidate (even accounting for falloff) returns an empty result");
        Check(selected.empty(), "SelectAndApplyClimateBiome: outSelectedBiomeName is set to an empty string on the no-viable-biome case");

        // An entirely empty candidate list must hit the same empty-result path.
        const std::vector<std::pair<pcg::PcgBiomeRuleSet, pcg::PcgClimateBiomeWeight>> noCandidates;
        const pcg::PcgClimateSample anyClimate{ 20.0f, 0.5f };
        const std::vector<pcg::PcgBiomeLayerResult> emptyCandidatesResult = pcg::SelectAndApplyClimateBiome(basePoints, anyClimate, noCandidates, 42u, nullptr);
        Check(emptyCandidatesResult.empty(), "SelectAndApplyClimateBiome: an empty candidate list returns an empty result (outSelectedBiomeName == nullptr is also handled without crashing)");
    }

    // ------------------------------------------------------------------------------------------
    // Determinism: repeated SelectAndApplyClimateBiome calls on identical basePoints/climate/
    // candidates/seed reproduce a byte-identical selection and result.
    // ------------------------------------------------------------------------------------------
    void TestDeterminism() {
        const std::vector<pcg::PcgPoint> basePoints = MakeGridPoints(30);

        pcg::PcgBiomeRuleSet ruleSetA;
        ruleSetA.biomeName = "BiomeA";
        pcg::PcgBiomeLayerRule layerA;
        layerA.layerName = "LayerA";
        layerA.useNoiseVariation = true;
        layerA.noiseFrequency = 0.2f;
        ruleSetA.layers.push_back(layerA);
        pcg::PcgClimateBiomeWeight weightA;
        weightA.biomeName = "BiomeA";
        weightA.minTemperature = 0.0f; weightA.maxTemperature = 30.0f;
        weightA.minMoisture = 0.0f; weightA.maxMoisture = 1.0f;
        weightA.baseWeight = 2.0f;

        pcg::PcgBiomeRuleSet ruleSetB;
        ruleSetB.biomeName = "BiomeB";
        pcg::PcgBiomeLayerRule layerB;
        layerB.layerName = "LayerB";
        ruleSetB.layers.push_back(layerB);
        pcg::PcgClimateBiomeWeight weightB;
        weightB.biomeName = "BiomeB";
        weightB.minTemperature = 0.0f; weightB.maxTemperature = 30.0f;
        weightB.minMoisture = 0.0f; weightB.maxMoisture = 1.0f;
        weightB.baseWeight = 1.0f; // Deliberately lower than BiomeA -- BiomeA should always win.

        const std::vector<std::pair<pcg::PcgBiomeRuleSet, pcg::PcgClimateBiomeWeight>> candidates = { { ruleSetA, weightA }, { ruleSetB, weightB } };
        const pcg::PcgClimateSample climate{ 18.0f, 0.6f };

        std::string selectedFirst, selectedSecond;
        const std::vector<pcg::PcgBiomeLayerResult> resultFirst = pcg::SelectAndApplyClimateBiome(basePoints, climate, candidates, 999u, &selectedFirst);
        const std::vector<pcg::PcgBiomeLayerResult> resultSecond = pcg::SelectAndApplyClimateBiome(basePoints, climate, candidates, 999u, &selectedSecond);

        Check(selectedFirst == "BiomeA" && selectedSecond == "BiomeA", "Determinism setup: both runs select the same (higher-baseWeight) biome");
        Check(LayerResultsMatchExactly(resultFirst, resultSecond),
            "Determinism: two SelectAndApplyClimateBiome calls on identical basePoints/climate/candidates/seed reproduce a byte-identical result");

        // A different seed must still select the SAME biome (selection has nothing to do with the
        // seed) but CAN change the noise-driven layer's density values.
        std::string selectedDiffSeed;
        const std::vector<pcg::PcgBiomeLayerResult> resultDiffSeed = pcg::SelectAndApplyClimateBiome(basePoints, climate, candidates, 111u, &selectedDiffSeed);
        Check(selectedDiffSeed == "BiomeA", "Determinism: a different seed does not change which biome is selected (selection is seed-independent)");
        bool anyDensityDiffers = false;
        const size_t count = std::min(resultFirst[0].points.size(), resultDiffSeed[0].points.size());
        for (size_t i = 0; i < count; ++i) {
            if (!NearlyEqual(resultFirst[0].points[i].density, resultDiffSeed[0].points[i].density, 1.0e-5f)) {
                anyDensityDiffers = true;
                break;
            }
        }
        Check(anyDensityDiffers, "Determinism: a different seed changes the noise-driven layer's density values (the seed IS threaded through to ApplyBiome)");
    }

    // ------------------------------------------------------------------------------------------
    // SampleCurrentClimate: constructs a real (default, never-Init()'d) renderer::AtmosClimatePass
    // -- safe with no live VkDevice/VmaAllocator because GpuBuffer's destructor no-ops on a
    // never-Create()'d buffer (see this file's own header comment) -- and checks the returned
    // PcgClimateSample is both plausible and, since Init()/RecordUpdate() are never called (so
    // AtmosClimatePass's own m_HasLastFrameTime stays false), deterministically equal to the static
    // config::atmos::TEMPERATURE_CELSIUS/RELATIVE_HUMIDITY baseline (see both GetEffective*()
    // getters' own comments for exactly why that is the guaranteed pre-RecordUpdate() value).
    // ------------------------------------------------------------------------------------------
    void TestSampleCurrentClimateSanity() {
        renderer::AtmosClimatePass atmosClimate; // Never Init()'d/RecordUpdate()'d -- see comment above.
        const pcg::PcgClimateSample sample = pcg::SampleCurrentClimate(atmosClimate);

        Check(sample.temperature > -100.0f && sample.temperature < 100.0f,
            "SampleCurrentClimate: temperature is within a plausible real-world Celsius range");
        Check(sample.moisture >= 0.0f && sample.moisture <= 1.0f,
            "SampleCurrentClimate: moisture is within its documented [0,1] fraction range");

        Check(NearlyEqual(sample.temperature, config::atmos::TEMPERATURE_CELSIUS),
            "SampleCurrentClimate: before any RecordUpdate() call, temperature exactly matches the static config::atmos::TEMPERATURE_CELSIUS baseline");
        Check(NearlyEqual(sample.moisture, config::atmos::RELATIVE_HUMIDITY),
            "SampleCurrentClimate: before any RecordUpdate() call, moisture exactly matches the static config::atmos::RELATIVE_HUMIDITY baseline");
    }

} // namespace

int main() {
    TestComputeBiomeWeightInsideOutsideEdge();
    TestSelectAndApplyClimateBiomePicksHighestWeight();
    TestTieBreakByInputOrder();
    TestNoViableBiomeReturnsEmpty();
    TestDeterminism();
    TestSampleCurrentClimateSanity();

    if (g_failCount == 0) {
        std::cout << "PcgClimateBiomeSelectorTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgClimateBiomeSelectorTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
