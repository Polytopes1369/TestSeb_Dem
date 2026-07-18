// Standalone, framework-free unit test for the PCG framework roadmap's World Partition runtime
// generation Phase 6.2 ("Partitioned PCG Cell Execution"): src/pcg/PcgCellGenerator.h/.cpp. Mirrors
// tests/PcgVolumeActorTests.cpp / tests/PcgNodePluginTests.cpp's own framework-free convention: exits
// 0 if every check passes, non-zero otherwise, registered with CTest (see the top-level
// CMakeLists.txt).
//
// A "pcg.test.cellgen_grid_points" node type is registered below via PCG_REGISTER_NODE_TYPE at
// namespace (file) scope -- a deterministic, purely-authoring-time "constant points on a grid"
// source node (positions are computed directly from its own params, never randomized), standing in
// for a real Phase 2 sampler (none of which are yet wired into the graph-node registry -- see
// PcgCellGenerator.h's own top-of-file comment, point 2). Distinct "pcg.test.*" prefix keeps this
// from ever being confused with the ONE real native node type this phase actually exercises,
// "pcg.spawner.weighted_mesh" (PcgMeshSpawner.cpp, Phase 4.1/5.4) -- every graph built below chains
// this file's own synthetic source node into that REAL registered spawner node, exactly the
// authoring pattern PcgCellGenerator.h's own top-of-file comment documents as "the real expected
// pattern".

#include "pcg/PcgCellGenerator.h"

#include "pcg/PcgGraph.h"
#include "pcg/PcgMeshSpawner.h"
#include "pcg/PcgNodePlugin.h"
#include "pcg/PcgPointData.h"
#include "core/maths/Maths.h"
#include "WorldPartition/PcgVolumeActor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// --- Synthetic test source node: "pcg.test.cellgen_grid_points" ---------------------------------
// No input pins; one Points output. Places countX * countZ points on a deterministic, purely
// param-driven X/Z grid (Y fixed at 0), starting at (originX, 0, originZ) with `spacing` between
// consecutive samples on each axis -- reused across this file's test graphs to place points
// straddling a chosen cell boundary in an exactly-predictable way. Each point's own `seed` is its
// flat grid index, matching tests/PcgNodePluginTests.cpp's own "pcg.plugin.constant_points" per-
// point seeding convention.
PCG_REGISTER_NODE_TYPE("pcg.test.cellgen_grid_points", "CellGen Test Grid Points",
    .Output("Points", pcg::PcgPinDataType::Points),
    [](const pcg::PcgNodePinDataMap& inputs, const pcg::PcgAttributeSet& params) -> pcg::PcgNodeExecuteResult {
        (void)inputs; // This node type declares no input pins.
        const int32_t countX = params.GetOr<int32_t>("countX", 1);
        const int32_t countZ = params.GetOr<int32_t>("countZ", 1);
        const float spacing = params.GetOr<float>("spacing", 1.0f);
        const float originX = params.GetOr<float>("originX", 0.0f);
        const float originZ = params.GetOr<float>("originZ", 0.0f);

        std::vector<pcg::PcgPoint> points;
        points.reserve(static_cast<size_t>(std::max(countX, 0)) * static_cast<size_t>(std::max(countZ, 0)));
        uint32_t index = 0;
        for (int32_t iz = 0; iz < countZ; ++iz) {
            for (int32_t ix = 0; ix < countX; ++ix) {
                pcg::PcgPoint point;
                point.position.x = originX + static_cast<float>(ix) * spacing;
                point.position.y = 0.0f;
                point.position.z = originZ + static_cast<float>(iz) * spacing;
                point.seed = index++;
                points.push_back(point);
            }
        }

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
        return std::fabs(a - b) <= epsilon;
    }

    // --- Test-graph authoring helpers ------------------------------------------------------------

    pcg::PcgAttributeSet MakeGridPointsParams(int32_t countX, int32_t countZ, float spacing, float originX, float originZ) {
        pcg::PcgAttributeSet params;
        params.Set("countX", countX);
        params.Set("countZ", countZ);
        params.Set("spacing", spacing);
        params.Set("originX", originX);
        params.Set("originZ", originZ);
        return params;
    }

    pcg::PcgAttributeSet MakeWeightedMeshSpawnerParams(uint32_t meshID, uint32_t materialID, uint32_t nodeSeed) {
        pcg::PcgAttributeSet params;
        std::vector<pcg::PcgMeshSpawnEntry> palette = { pcg::PcgMeshSpawnEntry{ meshID, materialID, 1.0f } };
        pcg::EncodeWeightedMeshList(params, palette);
        params.Set(pcg::kSpawnerDensityThresholdParamKey, 0.0f);
        params.Set(pcg::kSpawnerSeedParamKey, static_cast<int32_t>(nodeSeed));
        return params;
    }

    // Builds "grid points -> weighted mesh spawner" -- the REAL expected authoring pattern (see this
    // file's own top-of-file comment) -- and returns the fully-linked graph. Fails the calling test
    // (via Check) and returns an empty graph on any construction problem.
    pcg::PcgGraph BuildSpawnerGraph(const pcg::PcgAttributeSet& gridParams, uint32_t meshID, uint32_t materialID, uint32_t nodeSeed) {
        pcg::PcgNodeTypeRegistry registryUnused;
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PopulateNativeNodeTypePlugins(registryUnused, catalog);

        pcg::PcgGraph graph;
        std::string error;
        const uint32_t sourceNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.test.cellgen_grid_points", gridParams, "GridPoints", &error);
        Check(sourceNode != pcg::PcgNode::kInvalidId, "BuildSpawnerGraph: grid points node added (" + error + ")");

        const uint32_t spawnerNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.spawner.weighted_mesh",
            MakeWeightedMeshSpawnerParams(meshID, materialID, nodeSeed), "Spawner", &error);
        Check(spawnerNode != pcg::PcgNode::kInvalidId, "BuildSpawnerGraph: weighted_mesh spawner node added (" + error + ")");

        std::string linkError;
        Check(graph.AddLink(sourceNode, "Points", spawnerNode, "Points", &linkError) == pcg::PcgGraph::AddLinkStatus::Ok,
            "BuildSpawnerGraph: GridPoints -> Spawner link (" + linkError + ")");

        return graph;
    }

    // Builds a graph that stops at the raw Points output (no spawner attached) -- exercises
    // PcgCellGenerator's trivial 1:1 fallback path.
    pcg::PcgGraph BuildPointsOnlyGraph(const pcg::PcgAttributeSet& gridParams) {
        pcg::PcgNodeTypeRegistry registryUnused;
        pcg::PcgNodeTypeCatalog catalog;
        pcg::PopulateNativeNodeTypePlugins(registryUnused, catalog);

        pcg::PcgGraph graph;
        std::string error;
        const uint32_t sourceNode = pcg::AddNodeFromCatalog(graph, catalog, "pcg.test.cellgen_grid_points", gridParams, "GridPoints", &error);
        Check(sourceNode != pcg::PcgNode::kInvalidId, "BuildPointsOnlyGraph: grid points node added (" + error + ")");
        return graph;
    }

    // Writes `graph`'s serialized JSON to a fresh temp file under `scratchDir`, named `fileName`.
    // Returns the full path as a plain (forward-slash-friendly) string, ready to drop straight into
    // worldpartition::PcgVolumeDesc::graphAssetPath.
    std::string WriteGraphAssetToDisk(const std::filesystem::path& scratchDir, const std::string& fileName, const pcg::PcgGraph& graph) {
        std::filesystem::path filePath = scratchDir / fileName;
        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        out << graph.SerializeToJson();
        out.close();
        return filePath.string();
    }

    worldpartition::PcgVolumeDesc MakeVolumeDesc(const worldpartition::AABB& bounds, const std::string& graphAssetPath, uint32_t seed) {
        worldpartition::PcgVolumeDesc desc;
        desc.bounds = bounds;
        desc.graphAssetPath = graphAssetPath;
        desc.seed = seed;
        return desc;
    }

    // --- Test 1: a volume fully contained within one cell produces spawn requests entirely inside
    // that cell's own world-space bounds, via the REAL "grid points -> weighted_mesh spawner"
    // authoring pattern (this doubles as this file's "real end-to-end" case: a real PcgGraph, built
    // through PcgGraph's own API, ending in the real registered "pcg.spawner.weighted_mesh" node,
    // serialized to a real temp file, referenced by a real PcgVolumeDesc). ------------------------
    void TestSingleCellVolumeSpawnsWithinBounds(const std::filesystem::path& scratchDir) {
        constexpr float kCellSize = 20.0f;

        // 5x5 grid of points at integer offsets starting at (2,0,2) -- fully inside cell (0,0,0)'s
        // own [0,20)x[0,20) footprint with margin on every side.
        const pcg::PcgAttributeSet gridParams = MakeGridPointsParams(5, 5, 3.0f, 2.0f, 2.0f);
        const pcg::PcgGraph graph = BuildSpawnerGraph(gridParams, /*meshID=*/7u, /*materialID=*/3u, /*nodeSeed=*/999u);
        const std::string assetPath = WriteGraphAssetToDisk(scratchDir, "SingleCellSpawner.pcggraph.json", graph);

        worldpartition::AABB volumeBounds;
        volumeBounds.boundsMin = { 0.0f, 0.0f, 0.0f };
        volumeBounds.boundsMax = { 20.0f, 5.0f, 20.0f };
        const worldpartition::PcgVolumeDesc volumeDesc = MakeVolumeDesc(volumeBounds, assetPath, 123u);

        pcg::PcgCellGenerationInput input;
        input.cellCoord = worldpartition::CellCoord{ 0, 0, 0 };
        input.cellSize = kCellSize;
        input.overlappingVolumes = { volumeDesc };

        const pcg::PcgCellGenerationResult result = pcg::GeneratePcgContentForCell(input);
        Check(result.success, "single-cell: GeneratePcgContentForCell reports success (" + result.errorMessage + ")");
        Check(result.cellCoord == input.cellCoord, "single-cell: result.cellCoord echoes input.cellCoord");
        Check(result.spawnRequests.size() == 25, "single-cell: expected 25 spawn requests (5x5 grid, all inside cell), got " + std::to_string(result.spawnRequests.size()));

        for (const pcg::PcgSpawnRequest& request : result.spawnRequests) {
            Check(request.meshID == 7u, "single-cell: every spawn request uses the palette's meshID (7)");
            Check(request.materialID == 3u, "single-cell: every spawn request uses the palette's materialID (3)");
            Check(request.position.x >= 0.0f && request.position.x < 20.0f, "single-cell: spawn request X within cell (0,0) bounds");
            Check(request.position.z >= 0.0f && request.position.z < 20.0f, "single-cell: spawn request Z within cell (0,0) bounds");
        }
    }

    // --- Test 2: a volume spanning two cells (cellCoord{-1,0,0} and cellCoord{0,0,0} at cellSize 20)
    // generates DIFFERENT, DISJOINT content for each cell -- the defining property of genuinely
    // PARTITIONED generation (not duplicating a volume's full content into every cell it spans). ---
    void TestMultiCellSpanningVolumeProducesDifferentContentPerCell(const std::filesystem::path& scratchDir) {
        constexpr float kCellSize = 20.0f;

        // 10 points along X at spacing 4, starting at x=-18: -18,-14,-10,-6,-2 (5 points, cell x=-1,
        // i.e. [-20,0)) then 2,6,10,14,18 (5 points, cell x=0, i.e. [0,20)). Z fixed at 5 (inside
        // either cell's own [0,20) Z band).
        const pcg::PcgAttributeSet gridParams = MakeGridPointsParams(10, 1, 4.0f, -18.0f, 5.0f);
        const pcg::PcgGraph graph = BuildSpawnerGraph(gridParams, /*meshID=*/11u, /*materialID=*/2u, /*nodeSeed=*/555u);
        const std::string assetPath = WriteGraphAssetToDisk(scratchDir, "MultiCellSpawner.pcggraph.json", graph);

        worldpartition::AABB volumeBounds;
        volumeBounds.boundsMin = { -20.0f, 0.0f, 0.0f };
        volumeBounds.boundsMax = { 20.0f, 10.0f, 20.0f }; // Spans cell x in {-1, 0}.
        const worldpartition::PcgVolumeDesc volumeDesc = MakeVolumeDesc(volumeBounds, assetPath, 777u);

        pcg::PcgCellGenerationInput inputA;
        inputA.cellCoord = worldpartition::CellCoord{ -1, 0, 0 };
        inputA.cellSize = kCellSize;
        inputA.overlappingVolumes = { volumeDesc };
        const pcg::PcgCellGenerationResult resultA = pcg::GeneratePcgContentForCell(inputA);

        pcg::PcgCellGenerationInput inputB;
        inputB.cellCoord = worldpartition::CellCoord{ 0, 0, 0 };
        inputB.cellSize = kCellSize;
        inputB.overlappingVolumes = { volumeDesc };
        const pcg::PcgCellGenerationResult resultB = pcg::GeneratePcgContentForCell(inputB);

        Check(resultA.success && resultB.success, "multi-cell: both per-cell generations report success");
        Check(resultA.spawnRequests.size() == 5, "multi-cell: cell(-1,0,0) gets exactly the 5 points with x < 0, got " + std::to_string(resultA.spawnRequests.size()));
        Check(resultB.spawnRequests.size() == 5, "multi-cell: cell(0,0,0) gets exactly the 5 points with x >= 0, got " + std::to_string(resultB.spawnRequests.size()));

        std::set<float> positionsX_A, positionsX_B;
        for (const pcg::PcgSpawnRequest& request : resultA.spawnRequests) {
            Check(request.position.x >= -20.0f && request.position.x < 0.0f, "multi-cell: cell(-1,0,0) spawn request X within its own bounds");
            positionsX_A.insert(request.position.x);
        }
        for (const pcg::PcgSpawnRequest& request : resultB.spawnRequests) {
            Check(request.position.x >= 0.0f && request.position.x < 20.0f, "multi-cell: cell(0,0,0) spawn request X within its own bounds");
            positionsX_B.insert(request.position.x);
        }

        // DISJOINT, not duplicated: no X position appears in both cells' surviving sets.
        bool anyOverlap = false;
        for (float x : positionsX_A) {
            if (positionsX_B.contains(x)) anyOverlap = true;
        }
        Check(!anyOverlap, "multi-cell: cell(-1,0,0) and cell(0,0,0) content is disjoint (no duplicated point between them)");
        Check(positionsX_A.size() == 5 && positionsX_B.size() == 5, "multi-cell: both cells' surviving X positions are themselves distinct (no intra-cell duplication)");
    }

    // --- Test 3: determinism -- generating the SAME (input) twice produces byte-identical output. -
    void TestDeterminism(const std::filesystem::path& scratchDir) {
        constexpr float kCellSize = 20.0f;

        // Deliberately exercises the TRIVIAL POINTS FALLBACK path (no spawner node), since that is
        // the path this phase's own per-(volume,cell) derived seed (PcgHashCombine(volumeDesc.seed,
        // cellCoord)) actually feeds into -- see PcgCellGenerator.h's own "Determinism" comment.
        const pcg::PcgAttributeSet gridParams = MakeGridPointsParams(4, 4, 2.0f, 1.0f, 1.0f);
        const pcg::PcgGraph graph = BuildPointsOnlyGraph(gridParams);
        const std::string assetPath = WriteGraphAssetToDisk(scratchDir, "DeterminismPointsOnly.pcggraph.json", graph);

        worldpartition::AABB volumeBounds;
        volumeBounds.boundsMin = { 0.0f, 0.0f, 0.0f };
        volumeBounds.boundsMax = { 20.0f, 5.0f, 20.0f };
        const worldpartition::PcgVolumeDesc volumeDesc = MakeVolumeDesc(volumeBounds, assetPath, 42424242u);

        pcg::PcgCellGenerationInput input;
        input.cellCoord = worldpartition::CellCoord{ 0, 0, 0 };
        input.cellSize = kCellSize;
        input.overlappingVolumes = { volumeDesc };

        const pcg::PcgCellGenerationResult first = pcg::GeneratePcgContentForCell(input);
        const pcg::PcgCellGenerationResult second = pcg::GeneratePcgContentForCell(input);

        Check(first.success && second.success, "determinism: both runs report success");
        Check(first.spawnRequests.size() == 16, "determinism: expected 16 spawn requests (4x4 grid, all inside cell), got " + std::to_string(first.spawnRequests.size()));
        Check(first.spawnRequests.size() == second.spawnRequests.size(), "determinism: both runs produce the same spawn request count");

        const size_t n = std::min(first.spawnRequests.size(), second.spawnRequests.size());
        for (size_t i = 0; i < n; ++i) {
            const pcg::PcgSpawnRequest& a = first.spawnRequests[i];
            const pcg::PcgSpawnRequest& b = second.spawnRequests[i];
            Check(a.meshID == b.meshID && a.materialID == b.materialID, "determinism: request[" + std::to_string(i) + "] meshID/materialID match across runs");
            Check(NearlyEqual(a.position.x, b.position.x) && NearlyEqual(a.position.y, b.position.y) && NearlyEqual(a.position.z, b.position.z),
                "determinism: request[" + std::to_string(i) + "] position matches across runs");
        }
    }

    // --- Test 4: a volume whose graph asset does not exist on disk is skipped gracefully -- the
    // cell as a whole still reports success, just with zero spawn requests contributed by that
    // volume, and a warning is logged (Debug-only; not independently observable from this
    // framework-free test beyond the fact that execution does not fail or crash). -----------------
    void TestMissingGraphAssetSkippedGracefully() {
        worldpartition::AABB volumeBounds;
        volumeBounds.boundsMin = { 0.0f, 0.0f, 0.0f };
        volumeBounds.boundsMax = { 20.0f, 5.0f, 20.0f };
        const worldpartition::PcgVolumeDesc volumeDesc = MakeVolumeDesc(volumeBounds, "this/path/definitely_does_not_exist_12345.pcggraph.json", 1u);

        pcg::PcgCellGenerationInput input;
        input.cellCoord = worldpartition::CellCoord{ 0, 0, 0 };
        input.cellSize = 20.0f;
        input.overlappingVolumes = { volumeDesc };

        const pcg::PcgCellGenerationResult result = pcg::GeneratePcgContentForCell(input);
        Check(result.success, "missing graph asset: cell generation still reports success");
        Check(result.spawnRequests.empty(), "missing graph asset: zero spawn requests contributed");
    }

    // --- Test 4b (bonus): a graph asset that DOES exist but fails to parse (malformed JSON) is
    // skipped exactly as gracefully as a missing one -- the other half of this phase's own task
    // brief ("doesn't exist OR fails to parse"). ----------------------------------------------------
    void TestMalformedGraphAssetSkippedGracefully(const std::filesystem::path& scratchDir) {
        std::filesystem::path filePath = scratchDir / "Malformed.pcggraph.json";
        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        out << "{ this is not valid JSON at all ]]]";
        out.close();

        worldpartition::AABB volumeBounds;
        volumeBounds.boundsMin = { 0.0f, 0.0f, 0.0f };
        volumeBounds.boundsMax = { 20.0f, 5.0f, 20.0f };
        const worldpartition::PcgVolumeDesc volumeDesc = MakeVolumeDesc(volumeBounds, filePath.string(), 1u);

        pcg::PcgCellGenerationInput input;
        input.cellCoord = worldpartition::CellCoord{ 0, 0, 0 };
        input.cellSize = 20.0f;
        input.overlappingVolumes = { volumeDesc };

        const pcg::PcgCellGenerationResult result = pcg::GeneratePcgContentForCell(input);
        Check(result.success, "malformed graph asset: cell generation still reports success");
        Check(result.spawnRequests.empty(), "malformed graph asset: zero spawn requests contributed");
    }

    // --- Test 5 (bonus): a structurally invalid request (non-positive cellSize) is the one case
    // that DOES set success = false at the cell level (see PcgCellGenerationResult's own comment). -
    void TestInvalidCellSizeFails() {
        pcg::PcgCellGenerationInput input;
        input.cellCoord = worldpartition::CellCoord{ 0, 0, 0 };
        input.cellSize = 0.0f; // Invalid.
        input.overlappingVolumes = {};

        const pcg::PcgCellGenerationResult result = pcg::GeneratePcgContentForCell(input);
        Check(!result.success, "invalid cellSize: GeneratePcgContentForCell reports failure");
        Check(!result.errorMessage.empty(), "invalid cellSize: a non-empty errorMessage is set");
    }

    // --- Test 6 (bonus): an empty overlappingVolumes list is NOT an error -- a cell with no PCG
    // Volumes touching it simply generates nothing. --------------------------------------------------
    void TestNoOverlappingVolumesProducesEmptySuccess() {
        pcg::PcgCellGenerationInput input;
        input.cellCoord = worldpartition::CellCoord{ 5, 0, -3 };
        input.cellSize = 20.0f;
        input.overlappingVolumes = {};

        const pcg::PcgCellGenerationResult result = pcg::GeneratePcgContentForCell(input);
        Check(result.success, "no overlapping volumes: still reports success");
        Check(result.spawnRequests.empty(), "no overlapping volumes: zero spawn requests");
    }

} // namespace

int main() {
    std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "PcgCellGeneratorTests";
    std::error_code ec;
    std::filesystem::create_directories(scratchDir, ec);

    TestSingleCellVolumeSpawnsWithinBounds(scratchDir);
    TestMultiCellSpanningVolumeProducesDifferentContentPerCell(scratchDir);
    TestDeterminism(scratchDir);
    TestMissingGraphAssetSkippedGracefully();
    TestMalformedGraphAssetSkippedGracefully(scratchDir);
    TestInvalidCellSizeFails();
    TestNoOverlappingVolumesProducesEmptySuccess();

    if (g_failCount == 0) {
        std::cout << "PcgCellGeneratorTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgCellGeneratorTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
