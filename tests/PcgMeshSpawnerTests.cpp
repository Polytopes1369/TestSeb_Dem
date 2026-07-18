// Standalone, framework-free unit test for the PCG framework roadmap's Phase 4.1 ("Weighted Mesh
// Spawner") types: src/pcg/PcgMeshSpawner.h/.cpp -- weighted mesh selection (SpawnFromPoints),
// density-threshold culling, determinism, empty/degenerate edge cases, the
// PcgAttributeSet encode/decode round-trip for a weighted mesh list, and the
// "pcg.spawner.weighted_mesh" graph-node registration (built+evaluated through the real
// PcgGraphEvaluator, exactly mirroring tests/PcgNodePluginTests.cpp's own registration-round-trip
// pattern). Exits 0 if every check passes, non-zero otherwise -- registered with CTest (see the
// top-level CMakeLists.txt), matching this project's existing tests/*.cpp convention.

#include "pcg/PcgMeshSpawner.h"
#include "pcg/PcgNodePlugin.h"

#include "core/maths/Maths.h"
#include "pcg/PcgAttributeSet.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSeededRandom.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

// ------------------------------------------------------------------------------------------------
// A small test-only Points SOURCE node type, registered via Phase 5.4's PCG_REGISTER_NODE_TYPE at
// file scope -- needed purely so this file's own TestGraphNodeRegistrationRoundTrip() can feed a
// real "Points" input pin into "pcg.spawner.weighted_mesh" through an actual PcgGraph link, the same
// way tests/PcgNodePluginTests.cpp's own "pcg.plugin.constant_points" example node type exists
// purely to build a testable multi-node chain. Distinct "pcg.test.*" typeId prefix (matching
// PcgGraphEngineTests.cpp's own convention for synthetic test-only node types) keeps this from ever
// being confused with a real content-authoring node type; this test binary is its own separate
// process with its own separate registry/catalog instances, so there is no runtime collision with
// any other test file's own "pcg.test.*"/"pcg.plugin.*" node types either way.
//
// MakeSequentialTestPoints() is deliberately a plain free function (not a lambda-local closure) so
// TestGraphNodeRegistrationRoundTrip() below can call the EXACT SAME point-generation logic outside
// the graph, to build an independent expected-output baseline via a direct pcg::SpawnFromPoints()
// call -- proving the graph-node path and the direct-call path agree byte-for-byte, not just that
// "the graph evaluation didn't crash".
static std::vector<pcg::PcgPoint> MakeSequentialTestPoints(int32_t count) {
    std::vector<pcg::PcgPoint> points;
    points.reserve(static_cast<size_t>(count > 0 ? count : 0));
    for (int32_t i = 0; i < count; ++i) {
        pcg::PcgPoint point;
        point.position.x = static_cast<float>(i);
        point.position.y = 0.0f;
        point.position.z = 0.0f;
        point.seed = static_cast<uint32_t>(i);
        // density/rotation/scale/bounds/steepness all keep PcgPoint's own default field
        // initializers (density = 1.0f, identity rotation, unit scale) -- this source node type
        // does not author any of those, it only needs to prove Points-pin plumbing.
        points.push_back(point);
    }
    return points;
}

PCG_REGISTER_NODE_TYPE("pcg.test.mesh_spawner_points_source", "Mesh Spawner Test Points Source",
    .Output("Points", pcg::PcgPinDataType::Points),
    [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
        (void)inputs; // This node type declares no input pins.
        const int32_t count = params.GetOr<int32_t>("pointCount", 0);
        std::vector<pcg::PcgPoint> points = MakeSequentialTestPoints(count);
        pcg::PcgNodePinDataMap outputs;
        outputs.emplace("Points", std::move(points));
        return pcg::PcgNodeExecuteResult::Ok(std::move(outputs));
    });

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

    bool Vec3NearlyEqual(const maths::vec3& a, const maths::vec3& b, float epsilon = 1.0e-4f) {
        return NearlyEqual(a.x, b.x, epsilon) && NearlyEqual(a.y, b.y, epsilon) && NearlyEqual(a.z, b.z, epsilon);
    }

    bool QuatNearlyEqual(const maths::quat& a, const maths::quat& b, float epsilon = 1.0e-4f) {
        return NearlyEqual(a.x, b.x, epsilon) && NearlyEqual(a.y, b.y, epsilon) &&
            NearlyEqual(a.z, b.z, epsilon) && NearlyEqual(a.w, b.w, epsilon);
    }

    bool RequestsEqual(const pcg::PcgSpawnRequest& a, const pcg::PcgSpawnRequest& b) {
        return a.meshID == b.meshID && a.materialID == b.materialID &&
            Vec3NearlyEqual(a.position, b.position) && QuatNearlyEqual(a.rotation, b.rotation) && Vec3NearlyEqual(a.scale, b.scale);
    }

    // Builds N points with distinct seeds/positions -- the standard input fixture most tests below
    // share. `density` is applied uniformly to every point (default 1.0f, i.e. no point is at risk
    // of being culled by a densityThreshold of 0.0f).
    std::vector<pcg::PcgPoint> MakePoints(int32_t count, float density = 1.0f) {
        std::vector<pcg::PcgPoint> points = MakeSequentialTestPoints(count);
        for (pcg::PcgPoint& point : points) {
            point.density = density;
        }
        return points;
    }

    // ---------------------------------------------------------------------------------------------
    // Test 1: weighted selection roughly matches the expected proportions over many points --
    // mirrors tests/PcgSurfaceSamplerTests.cpp's own TestAreaWeighting() statistical-check
    // convention exactly (generous, conservatively-wide tolerance band, not a tuned-to-pass hack).
    // ---------------------------------------------------------------------------------------------
    void TestWeightedSelectionStatistics() {
        const std::vector<pcg::PcgMeshSpawnEntry> weightedMeshes{
            pcg::PcgMeshSpawnEntry{ /*meshID=*/1, /*materialID=*/10, /*weight=*/3.0f },
            pcg::PcgMeshSpawnEntry{ /*meshID=*/2, /*materialID=*/20, /*weight=*/1.0f },
        };
        const std::vector<pcg::PcgPoint> points = MakePoints(8000);

        const std::vector<pcg::PcgSpawnRequest> requests = pcg::SpawnFromPoints(points, weightedMeshes, /*seed=*/0xABCDu);
        Check(requests.size() == points.size(), "SpawnFromPoints (statistics): every point produces exactly one request when density passes and weights are all positive");

        size_t countMesh1 = 0, countMesh2 = 0, countOther = 0;
        for (const pcg::PcgSpawnRequest& request : requests) {
            if (request.meshID == 1) ++countMesh1;
            else if (request.meshID == 2) ++countMesh2;
            else ++countOther;
        }
        Check(countOther == 0, "SpawnFromPoints (statistics): every request resolves to one of the two declared meshIDs");
        Check(countMesh1 > 0 && countMesh2 > 0, "SpawnFromPoints (statistics): both weighted candidates receive at least some picks");

        if (countMesh2 > 0) {
            const float ratio = static_cast<float>(countMesh1) / static_cast<float>(countMesh2);
            // Expected ratio is exactly 3.0 (weight-proportional, 3.0 : 1.0); a 20% band is
            // comfortably conservative at n=8000 (binomial std-dev at p=0.75 is well under 1% of
            // the count), same reasoning PcgSurfaceSamplerTests.cpp's own TestAreaWeighting uses.
            Check(ratio > 2.4f && ratio < 3.6f,
                "SpawnFromPoints (statistics): the weight-3.0 mesh is picked roughly 3x as often as the weight-1.0 mesh (ratio=" + std::to_string(ratio) + ")");
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Test 2: a zero-weight (or negative-weight, clamped to zero) entry is NEVER selected.
    // ---------------------------------------------------------------------------------------------
    void TestZeroAndNegativeWeightNeverPicked() {
        const std::vector<pcg::PcgMeshSpawnEntry> weightedMeshes{
            pcg::PcgMeshSpawnEntry{ /*meshID=*/100, /*materialID=*/1, /*weight=*/1.0f },
            pcg::PcgMeshSpawnEntry{ /*meshID=*/200, /*materialID=*/2, /*weight=*/0.0f },
            pcg::PcgMeshSpawnEntry{ /*meshID=*/300, /*materialID=*/3, /*weight=*/-5.0f },
        };
        const std::vector<pcg::PcgPoint> points = MakePoints(3000);

        const std::vector<pcg::PcgSpawnRequest> requests = pcg::SpawnFromPoints(points, weightedMeshes, /*seed=*/7u);
        Check(requests.size() == points.size(), "SpawnFromPoints (zero-weight): every point still produces a request (a zero/negative-weight entry does not shrink the candidate list's TOTAL positive weight)");

        const bool anyZeroWeightPicked = std::any_of(requests.begin(), requests.end(), [](const pcg::PcgSpawnRequest& r) { return r.meshID == 200; });
        const bool anyNegativeWeightPicked = std::any_of(requests.begin(), requests.end(), [](const pcg::PcgSpawnRequest& r) { return r.meshID == 300; });
        Check(!anyZeroWeightPicked, "SpawnFromPoints (zero-weight): the zero-weight entry (meshID=200) is never selected across 3000 draws");
        Check(!anyNegativeWeightPicked, "SpawnFromPoints (zero-weight): the negative-weight entry (meshID=300) is never selected across 3000 draws");

        const bool allPositiveWeightPicked = std::all_of(requests.begin(), requests.end(), [](const pcg::PcgSpawnRequest& r) { return r.meshID == 100; });
        Check(allPositiveWeightPicked, "SpawnFromPoints (zero-weight): with only one positive-weight candidate, every request resolves to it");
    }

    // ---------------------------------------------------------------------------------------------
    // Test 3: densityThreshold correctly filters points by their raw `density` field.
    // ---------------------------------------------------------------------------------------------
    void TestDensityThresholdFiltering() {
        std::vector<pcg::PcgPoint> points = MakePoints(2000);
        // Alternate density: even index -> 1.0 (passes a 0.5 threshold), odd index -> 0.2 (culled).
        for (size_t i = 0; i < points.size(); ++i) {
            points[i].density = (i % 2 == 0) ? 1.0f : 0.2f;
        }
        const size_t expectedSurvivorCount = points.size() / 2; // every even index.

        const std::vector<pcg::PcgMeshSpawnEntry> weightedMeshes{ pcg::PcgMeshSpawnEntry{ /*meshID=*/7, /*materialID=*/1, /*weight=*/1.0f } };
        const std::vector<pcg::PcgSpawnRequest> requests = pcg::SpawnFromPoints(points, weightedMeshes, /*seed=*/42u, /*densityThreshold=*/0.5f);

        Check(requests.size() == expectedSurvivorCount,
            "SpawnFromPoints (density threshold): exactly the points with density >= threshold survive (got " +
            std::to_string(requests.size()) + ", expected " + std::to_string(expectedSurvivorCount) + ")");

        // Every surviving request's position.x must be an EVEN integer (this fixture's own
        // "high-density point" identity marker, via MakeSequentialTestPoints' position.x == seed
        // convention) -- proves the CORRECT points survived, not merely the correct COUNT.
        bool allSurvivorsEven = true;
        for (const pcg::PcgSpawnRequest& request : requests) {
            const int32_t xAsInt = static_cast<int32_t>(request.position.x + 0.5f);
            if (xAsInt % 2 != 0) {
                allSurvivorsEven = false;
                break;
            }
        }
        Check(allSurvivorsEven, "SpawnFromPoints (density threshold): every surviving request comes from an even-indexed (density=1.0) source point");

        // densityThreshold defaults to 0.0f (no culling) -- omitting the argument entirely must
        // reproduce every point, even the density=0.2f ones from this same fixture.
        const std::vector<pcg::PcgSpawnRequest> requestsNoThreshold = pcg::SpawnFromPoints(points, weightedMeshes, /*seed=*/42u);
        Check(requestsNoThreshold.size() == points.size(), "SpawnFromPoints (density threshold): the default densityThreshold (0.0f) culls nothing");
    }

    // ---------------------------------------------------------------------------------------------
    // Test 4: determinism -- byte-identical output for identical input, AND order-independent
    // per-point determinism (a point's own outcome never depends on which other points share the
    // same call).
    // ---------------------------------------------------------------------------------------------
    void TestDeterminism() {
        const std::vector<pcg::PcgMeshSpawnEntry> weightedMeshes{
            pcg::PcgMeshSpawnEntry{ /*meshID=*/1, /*materialID=*/10, /*weight=*/2.0f },
            pcg::PcgMeshSpawnEntry{ /*meshID=*/2, /*materialID=*/20, /*weight=*/1.0f },
            pcg::PcgMeshSpawnEntry{ /*meshID=*/3, /*materialID=*/30, /*weight=*/1.0f },
        };
        const std::vector<pcg::PcgPoint> points = MakePoints(500);

        const std::vector<pcg::PcgSpawnRequest> runA = pcg::SpawnFromPoints(points, weightedMeshes, /*seed=*/999u);
        const std::vector<pcg::PcgSpawnRequest> runB = pcg::SpawnFromPoints(points, weightedMeshes, /*seed=*/999u);
        Check(runA.size() == runB.size(), "SpawnFromPoints (determinism): repeated calls with identical input produce the same request count");
        bool allMatch = runA.size() == runB.size();
        for (size_t i = 0; allMatch && i < runA.size(); ++i) {
            if (!RequestsEqual(runA[i], runB[i])) {
                allMatch = false;
            }
        }
        Check(allMatch, "SpawnFromPoints (determinism): repeated calls with identical input produce byte-identical requests, element-for-element");

        // Order-independence: re-run against a REVERSED and TRUNCATED copy of `points` (drop the
        // last 100), and confirm every point that appears in BOTH runs (matched by its own `seed`
        // field, since seed == original index here) resolves to the exact same meshID/materialID --
        // i.e. removing/reordering OTHER points never perturbs a given point's own outcome.
        std::vector<pcg::PcgPoint> subset(points.begin(), points.end() - 100);
        std::reverse(subset.begin(), subset.end());
        const std::vector<pcg::PcgSpawnRequest> runSubset = pcg::SpawnFromPoints(subset, weightedMeshes, /*seed=*/999u);

        // Index runA by seed (== original point index here) for O(1) lookup against runSubset.
        std::vector<uint32_t> meshByPointSeed(points.size(), 0xFFFFFFFFu);
        for (size_t i = 0; i < points.size(); ++i) {
            meshByPointSeed[points[i].seed] = runA[i].meshID;
        }
        bool orderIndependent = true;
        for (size_t i = 0; i < subset.size(); ++i) {
            const uint32_t expectedMeshID = meshByPointSeed[subset[i].seed];
            if (runSubset[i].meshID != expectedMeshID) {
                orderIndependent = false;
                break;
            }
        }
        Check(orderIndependent, "SpawnFromPoints (determinism): a point's own resolved meshID is unaffected by reordering/removing OTHER points in the same call");
    }

    // ---------------------------------------------------------------------------------------------
    // Test 5: empty-list / degenerate edge cases -- never crashes, never divides by zero.
    // ---------------------------------------------------------------------------------------------
    void TestEmptyEdgeCases() {
        const std::vector<pcg::PcgMeshSpawnEntry> oneEntry{ pcg::PcgMeshSpawnEntry{ 1, 1, 1.0f } };
        const std::vector<pcg::PcgPoint> somePoints = MakePoints(10);

        Check(pcg::SpawnFromPoints({}, oneEntry, 1u).empty(), "SpawnFromPoints (edge cases): empty points list -> empty result");
        Check(pcg::SpawnFromPoints(somePoints, {}, 1u).empty(), "SpawnFromPoints (edge cases): empty weightedMeshes list -> empty result");

        const std::vector<pcg::PcgMeshSpawnEntry> allZero{ pcg::PcgMeshSpawnEntry{ 1, 1, 0.0f }, pcg::PcgMeshSpawnEntry{ 2, 2, 0.0f } };
        Check(pcg::SpawnFromPoints(somePoints, allZero, 1u).empty(), "SpawnFromPoints (edge cases): all-zero-weight list -> empty result, not a crash");

        const std::vector<pcg::PcgMeshSpawnEntry> allNegative{ pcg::PcgMeshSpawnEntry{ 1, 1, -1.0f }, pcg::PcgMeshSpawnEntry{ 2, 2, -2.0f } };
        Check(pcg::SpawnFromPoints(somePoints, allNegative, 1u).empty(), "SpawnFromPoints (edge cases): all-negative-weight list -> empty result, not a crash");

        Check(pcg::SpawnFromPoints({}, {}, 1u).empty(), "SpawnFromPoints (edge cases): both lists empty -> empty result");
    }

    // ---------------------------------------------------------------------------------------------
    // Test 6: PcgAttributeSet encode/decode round-trip for a weighted mesh list.
    // ---------------------------------------------------------------------------------------------
    void TestEncodeDecodeRoundTrip() {
        const std::vector<pcg::PcgMeshSpawnEntry> original{
            pcg::PcgMeshSpawnEntry{ /*meshID=*/0, /*materialID=*/5, /*weight=*/2.5f },
            pcg::PcgMeshSpawnEntry{ /*meshID=*/3, /*materialID=*/0, /*weight=*/1.0f },
            pcg::PcgMeshSpawnEntry{ /*meshID=*/42, /*materialID=*/17, /*weight=*/0.75f },
        };

        pcg::PcgAttributeSet params;
        pcg::EncodeWeightedMeshList(params, original);

        const std::vector<pcg::PcgMeshSpawnEntry> decoded = pcg::DecodeWeightedMeshList(params);
        Check(decoded.size() == original.size(), "EncodeWeightedMeshList/DecodeWeightedMeshList: round-trip preserves entry count");
        if (decoded.size() == original.size()) {
            for (size_t i = 0; i < original.size(); ++i) {
                Check(decoded[i].meshID == original[i].meshID, "encode/decode round-trip: entry[" + std::to_string(i) + "].meshID matches");
                Check(decoded[i].materialID == original[i].materialID, "encode/decode round-trip: entry[" + std::to_string(i) + "].materialID matches");
                Check(NearlyEqual(decoded[i].weight, original[i].weight), "encode/decode round-trip: entry[" + std::to_string(i) + "].weight matches");
            }
        }

        pcg::PcgAttributeSet emptyParams;
        Check(pcg::DecodeWeightedMeshList(emptyParams).empty(), "DecodeWeightedMeshList: a params set with no 'entryCount' key decodes to an empty list");

        pcg::PcgAttributeSet zeroCountParams;
        zeroCountParams.Set(pcg::kMeshSpawnEntryCountKey, static_cast<int32_t>(0));
        Check(pcg::DecodeWeightedMeshList(zeroCountParams).empty(), "DecodeWeightedMeshList: entryCount == 0 decodes to an empty list");

        pcg::PcgAttributeSet negativeCountParams;
        negativeCountParams.Set(pcg::kMeshSpawnEntryCountKey, static_cast<int32_t>(-3));
        Check(pcg::DecodeWeightedMeshList(negativeCountParams).empty(), "DecodeWeightedMeshList: a negative entryCount defensively decodes to an empty list, not a crash");
    }

    // ---------------------------------------------------------------------------------------------
    // Test 7: the "pcg.spawner.weighted_mesh" graph-node registration round-trip -- mirrors
    // tests/PcgNodePluginTests.cpp's own TestPluginRegistrationPopulatesRegistryAndCatalog +
    // TestEvaluateGraphBuiltFromCatalogAndValidatesClean pattern: registry/catalog population,
    // descriptor introspection, then a real 2-node graph built via AddNodeFromCatalog and evaluated
    // through the unmodified PcgGraphEvaluator.
    // ---------------------------------------------------------------------------------------------
    void TestGraphNodeRegistrationRoundTrip() {
        pcg::PcgNodeTypeRegistry registry;
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PopulateNativeNodeTypePlugins(registry, catalog);

        Check(registry.IsRegistered("pcg.spawner.weighted_mesh"), "registry: 'pcg.spawner.weighted_mesh' is registered (execution half)");
        Check(catalog.Find("pcg.spawner.weighted_mesh") != nullptr, "catalog: 'pcg.spawner.weighted_mesh' is registered (introspection half)");

        const pcg::PcgNodeTypeDescriptor* descriptor = catalog.Find("pcg.spawner.weighted_mesh");
        if (descriptor) {
            Check(descriptor->displayName == "Weighted Mesh Spawner", "catalog: weighted_mesh displayName matches registration call");
            Check(descriptor->inputPins.size() == 1, "catalog: weighted_mesh declares exactly 1 input pin");
            if (descriptor->inputPins.size() == 1) {
                Check(descriptor->inputPins[0].name == "Points", "catalog: weighted_mesh input pin is named 'Points'");
                Check(descriptor->inputPins[0].type == pcg::PcgPinDataType::Points, "catalog: weighted_mesh input pin type is Points");
                Check(descriptor->inputPins[0].required, "catalog: weighted_mesh input pin is required");
            }
            Check(descriptor->outputPins.size() == 1, "catalog: weighted_mesh declares exactly 1 output pin");
            if (descriptor->outputPins.size() == 1) {
                Check(descriptor->outputPins[0].name == "SpawnRequests", "catalog: weighted_mesh output pin is named 'SpawnRequests'");
                Check(descriptor->outputPins[0].type == pcg::PcgPinDataType::SpawnRequests, "catalog: weighted_mesh output pin type is the new SpawnRequests PcgPinDataType");
            }
        }

        // --- Build a real 2-node graph: [test points source] --Points--> [weighted mesh spawner] ---
        constexpr int32_t kPointCount = 200;
        constexpr float kDensityThreshold = 0.0f;
        constexpr uint32_t kSeed = 0x1234u;
        const std::vector<pcg::PcgMeshSpawnEntry> weightedMeshes{
            pcg::PcgMeshSpawnEntry{ /*meshID=*/2, /*materialID=*/0, /*weight=*/1.0f }, // Rock archetype, per this phase's own env notes.
            pcg::PcgMeshSpawnEntry{ /*meshID=*/1, /*materialID=*/0, /*weight=*/1.0f }, // Bush archetype.
        };

        pcg::PcgAttributeSet sourceParams;
        sourceParams.Set("pointCount", kPointCount);

        pcg::PcgAttributeSet spawnerParams;
        pcg::EncodeWeightedMeshList(spawnerParams, weightedMeshes);
        spawnerParams.Set(pcg::kSpawnerDensityThresholdParamKey, kDensityThreshold);
        spawnerParams.Set(pcg::kSpawnerSeedParamKey, static_cast<int32_t>(kSeed));

        pcg::PcgGraph graph;
        std::string addError;
        const uint32_t sourceNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.test.mesh_spawner_points_source", sourceParams, "Source", &addError);
        Check(sourceNode != pcg::PcgNode::kInvalidId, "AddNodeFromCatalog: points-source node added (" + addError + ")");
        const uint32_t spawnerNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.spawner.weighted_mesh", spawnerParams, "Spawner", &addError);
        Check(spawnerNode != pcg::PcgNode::kInvalidId, "AddNodeFromCatalog: weighted_mesh spawner node added (" + addError + ")");

        std::string linkMsg;
        Check(graph.AddLink(sourceNode, "Points", spawnerNode, "Points", &linkMsg) == pcg::PcgGraph::AddLinkStatus::Ok,
            "chain: source->spawner Points link (" + linkMsg + ")");

        pcg::PcgGraphEvaluator evaluator(registry);
        const pcg::PcgGraphEvaluator::EvalResult result = evaluator.Evaluate(graph);
        Check(result.success, "evaluate: source->spawner 2-node graph succeeds (" + result.errorMessage + ")");

        if (result.success) {
            const auto nodeOutputsIt = result.nodeOutputs.find(spawnerNode);
            Check(nodeOutputsIt != result.nodeOutputs.end(), "evaluate: spawner node has cached outputs");
            if (nodeOutputsIt != result.nodeOutputs.end()) {
                const auto spawnPinIt = nodeOutputsIt->second.find("SpawnRequests");
                Check(spawnPinIt != nodeOutputsIt->second.end(), "evaluate: spawner node produced a 'SpawnRequests' output pin");
                if (spawnPinIt != nodeOutputsIt->second.end()) {
                    const std::vector<pcg::PcgSpawnRequest>* graphRequests = std::get_if<std::vector<pcg::PcgSpawnRequest>>(&spawnPinIt->second);
                    Check(graphRequests != nullptr, "evaluate: 'SpawnRequests' output pin holds std::vector<pcg::PcgSpawnRequest> (the new PcgPinData variant alternative)");
                    if (graphRequests) {
                        // The decisive check: build the IDENTICAL point set OUTSIDE the graph and
                        // call pcg::SpawnFromPoints() directly -- the graph-node execute callback
                        // must produce EXACTLY the same output, proving it correctly decodes
                        // params (EncodeWeightedMeshList's own scheme) and delegates to
                        // SpawnFromPoints without any transcription error.
                        const std::vector<pcg::PcgPoint> expectedPoints = MakeSequentialTestPoints(kPointCount);
                        const std::vector<pcg::PcgSpawnRequest> expectedRequests =
                            pcg::SpawnFromPoints(expectedPoints, weightedMeshes, kSeed, kDensityThreshold);

                        Check(graphRequests->size() == expectedRequests.size(),
                            "evaluate: graph-produced SpawnRequests count matches a direct SpawnFromPoints() call (" +
                            std::to_string(graphRequests->size()) + " vs " + std::to_string(expectedRequests.size()) + ")");
                        bool allMatch = graphRequests->size() == expectedRequests.size();
                        for (size_t i = 0; allMatch && i < graphRequests->size(); ++i) {
                            if (!RequestsEqual((*graphRequests)[i], expectedRequests[i])) {
                                allMatch = false;
                            }
                        }
                        Check(allMatch, "evaluate: graph-produced SpawnRequests are element-for-element identical to a direct SpawnFromPoints() call on the same inputs");
                    }
                }
            }
        }
    }

} // namespace

int main() {
    TestWeightedSelectionStatistics();
    TestZeroAndNegativeWeightNeverPicked();
    TestDensityThresholdFiltering();
    TestDeterminism();
    TestEmptyEdgeCases();
    TestEncodeDecodeRoundTrip();
    TestGraphNodeRegistrationRoundTrip();

    if (g_failCount == 0) {
        std::cout << "PcgMeshSpawnerTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgMeshSpawnerTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
