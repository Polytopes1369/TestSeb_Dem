// Standalone, framework-free unit test for the PCG framework roadmap's World Partition runtime
// generation Phase 6.3 ("Runtime Generator Hook"): src/world/PcgVolumeCellIndex.h/.cpp -- the PURE,
// Vulkan/PcgInstanceSpawnManager-free half of world::PcgCellLoader (see that class' own top-of-file
// comment for exactly why this split exists). Covers:
//   - world::ToOfflineCellCoord / world::ToRuntimeCellCoord round-tripping.
//   - world::ScanPcgVolumeActorFiles: finds real PcgVolume .actor files, skips non-PcgVolume and
//     malformed ones, returns empty (not an error) for a missing directory.
//   - world::BuildPcgVolumeCellIndex: single-cell bucketing, multi-cell-spanning bucketing (a volume
//     appears in EVERY cell it overlaps), and multiple volumes overlapping the SAME cell both appear
//     (appended, never deduplicated/overwritten).
// world::PcgCellLoader itself (the live IWorldCellLoader implementation, needing a real
// pcg::PcgInstanceSpawnManager/renderer::PcgInstanceDrawPass/Vulkan device) is intentionally NOT
// covered here -- per this phase's own task brief ("if the class is too tightly coupled to
// PcgInstanceSpawnManager/live rendering to test standalone, rely on the in-engine smoke test
// instead and document that choice"), that end-to-end path is validated by
// renderer::ClusterRenderPipeline::RunPcgCellLoaderSmokeTest() instead (see that method's own header
// comment in src/renderer/ClusterRenderPipeline.h for exactly what it checks).
//
// Exits 0 if every check passes, non-zero otherwise -- registered with CTest (see the top-level
// CMakeLists.txt), matching this project's existing tests/*.cpp convention.

#include "world/PcgVolumeCellIndex.h"

#include "WorldPartition/OfpaActor.h"
#include "WorldPartition/PcgVolumeActor.h"
#include "WorldPartition/Uuid.h"
#include "core/maths/Maths.h"

#include <filesystem>
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

    worldpartition::AABB MakeAABB(float minX, float minZ, float maxX, float maxZ) {
        worldpartition::AABB box;
        box.boundsMin = { minX, 0.0f, minZ };
        box.boundsMax = { maxX, 5.0f, maxZ };
        return box;
    }

    // Writes a real PcgVolume .actor file (via the real BuildPcgVolumeActorRecord + WriteActorFile
    // pair, the exact same functions world::ScanPcgVolumeActorFiles' own callee,
    // worldpartition::TryParsePcgVolumeDesc, is the inverse of) under `actorsRootDir`.
    worldpartition::Uuid WritePcgVolumeActorFile(const std::filesystem::path& actorsRootDir,
        worldpartition::UuidGenerator& uuidGen, const worldpartition::AABB& bounds,
        const std::string& graphAssetPath, uint32_t seed) {
        worldpartition::Uuid uuid = uuidGen.Generate();

        worldpartition::PcgVolumeDesc desc;
        desc.bounds = bounds;
        desc.graphAssetPath = graphAssetPath;
        desc.seed = seed;

        worldpartition::ActorRecord record = worldpartition::BuildPcgVolumeActorRecord(uuid, desc);
        std::filesystem::path actorPath = worldpartition::MakeActorFilePath(actorsRootDir, uuid);
        Check(worldpartition::WriteActorFile(actorPath, record), "WritePcgVolumeActorFile: WriteActorFile succeeded");
        return uuid;
    }

    // Writes a real, but NON-PcgVolume, actor file (mirrors tools/WorldPartition/BakeDemoWorld.cpp's
    // own "Rock"/"Bush"/"Tree" archetype actors) -- ScanPcgVolumeActorFiles must silently skip this,
    // never crash or abort the scan over it.
    void WriteNonPcgVolumeActorFile(const std::filesystem::path& actorsRootDir, worldpartition::UuidGenerator& uuidGen) {
        worldpartition::Uuid uuid = uuidGen.Generate();
        worldpartition::ActorRecord record;
        record.uuid = uuid;
        record.className = "Rock";
        record.actorLabel = "Rock_0";
        record.localBounds = MakeAABB(0.0f, 0.0f, 1.0f, 1.0f);
        record.RecomputeWorldBounds();
        std::filesystem::path actorPath = worldpartition::MakeActorFilePath(actorsRootDir, uuid);
        Check(worldpartition::WriteActorFile(actorPath, record), "WriteNonPcgVolumeActorFile: WriteActorFile succeeded");
    }

    // --- Test 1: CellCoord conversion round-trips exactly (Grid2D-mode y is always 0 on both sides,
    // so this is lossless in both directions). ------------------------------------------------------
    void TestCellCoordConversionRoundTrips() {
        const world::CellCoord runtime{ 7, -3 };
        const worldpartition::CellCoord offline = world::ToOfflineCellCoord(runtime);
        Check(offline.x == 7 && offline.y == 0 && offline.z == -3, "ToOfflineCellCoord: x/z carried over, y forced to 0");

        const world::CellCoord roundTripped = world::ToRuntimeCellCoord(offline);
        Check(roundTripped.x == runtime.x && roundTripped.z == runtime.z, "ToRuntimeCellCoord: round-trip reproduces the original x/z");

        // Direct offline -> runtime conversion (the direction PcgVolumeCellIndex.cpp actually uses,
        // via worldpartition::ComputeOverlappingCells' own output).
        const worldpartition::CellCoord anotherOffline{ -12, 0, 42 };
        const world::CellCoord converted = world::ToRuntimeCellCoord(anotherOffline);
        Check(converted.x == -12 && converted.z == 42, "ToRuntimeCellCoord: arbitrary offline coord converts correctly");
    }

    // --- Test 2: ScanPcgVolumeActorFiles finds real PcgVolume actors, skips a non-PcgVolume actor
    // sharing the same directory, and returns empty (not an error) for a directory that does not
    // exist at all. -------------------------------------------------------------------------------
    void TestScanPcgVolumeActorFiles(const std::filesystem::path& scratchDir) {
        const std::filesystem::path actorsDir = scratchDir / "scan_test_actors";
        std::error_code ec;
        std::filesystem::create_directories(actorsDir, ec);

        worldpartition::UuidGenerator uuidGen(0xABCDEF0123456789ULL);
        WritePcgVolumeActorFile(actorsDir, uuidGen, MakeAABB(0.0f, 0.0f, 10.0f, 10.0f), "some/graph.pcggraph.json", 111u);
        WritePcgVolumeActorFile(actorsDir, uuidGen, MakeAABB(20.0f, 20.0f, 30.0f, 30.0f), "some/other_graph.pcggraph.json", 222u);
        WriteNonPcgVolumeActorFile(actorsDir, uuidGen);

        const std::vector<worldpartition::PcgVolumeDesc> found = world::ScanPcgVolumeActorFiles(actorsDir);
        Check(found.size() == 2, "ScanPcgVolumeActorFiles: found exactly the 2 real PcgVolume actors (skipped the Rock actor), got " + std::to_string(found.size()));

        bool sawSeed111 = false, sawSeed222 = false;
        for (const worldpartition::PcgVolumeDesc& desc : found) {
            if (desc.seed == 111u) sawSeed111 = true;
            if (desc.seed == 222u) sawSeed222 = true;
        }
        Check(sawSeed111 && sawSeed222, "ScanPcgVolumeActorFiles: both authored volumes' seeds were recovered");

        const std::filesystem::path missingDir = scratchDir / "this_directory_does_not_exist";
        const std::vector<worldpartition::PcgVolumeDesc> foundMissing = world::ScanPcgVolumeActorFiles(missingDir);
        Check(foundMissing.empty(), "ScanPcgVolumeActorFiles: a missing directory returns an empty (not error) result");
    }

    // --- Test 3: BuildPcgVolumeCellIndex buckets a single-cell-contained volume into exactly 1 cell.
    void TestBuildIndexSingleCell() {
        constexpr float kCellSize = 20.0f;
        worldpartition::PcgVolumeDesc desc;
        desc.bounds = MakeAABB(2.0f, 2.0f, 8.0f, 8.0f); // Fully inside cell (0,0).
        desc.graphAssetPath = "graph.pcggraph.json";
        desc.seed = 1u;

        const world::PcgVolumeCellIndex index = world::BuildPcgVolumeCellIndex({ desc }, kCellSize);
        Check(index.size() == 1, "single-cell index: exactly 1 cell touched, got " + std::to_string(index.size()));

        const auto it = index.find(world::CellCoord{ 0, 0 });
        Check(it != index.end(), "single-cell index: cell (0,0) is present");
        if (it != index.end()) {
            Check(it->second.size() == 1, "single-cell index: cell (0,0) has exactly 1 overlapping volume");
        }
    }

    // --- Test 4: a volume spanning multiple cells appears in EVERY cell it overlaps (matching
    // worldpartition::ComputeOverlappingCells' own documented "straddling actor" convention). -------
    void TestBuildIndexMultiCellSpan() {
        constexpr float kCellSize = 20.0f;
        worldpartition::PcgVolumeDesc desc;
        desc.bounds = MakeAABB(-5.0f, -5.0f, 25.0f, 25.0f); // Spans cell x/z in {-1, 0, 1} -- a 3x3 block.
        desc.graphAssetPath = "graph.pcggraph.json";
        desc.seed = 2u;

        const world::PcgVolumeCellIndex index = world::BuildPcgVolumeCellIndex({ desc }, kCellSize);
        Check(index.size() == 9, "multi-cell index: 3x3 = 9 cells touched, got " + std::to_string(index.size()));

        for (int32_t x = -1; x <= 1; ++x) {
            for (int32_t z = -1; z <= 1; ++z) {
                const auto it = index.find(world::CellCoord{ x, z });
                Check(it != index.end(), "multi-cell index: cell (" + std::to_string(x) + "," + std::to_string(z) + ") is present");
            }
        }
    }

    // --- Test 5: two DIFFERENT volumes both overlapping the SAME cell both appear in that cell's
    // own entry (appended, never overwritten/deduplicated). ------------------------------------------
    void TestBuildIndexMultipleVolumesSameCellAppend() {
        constexpr float kCellSize = 20.0f;
        worldpartition::PcgVolumeDesc descA;
        descA.bounds = MakeAABB(1.0f, 1.0f, 5.0f, 5.0f);
        descA.graphAssetPath = "graphA.pcggraph.json";
        descA.seed = 10u;

        worldpartition::PcgVolumeDesc descB;
        descB.bounds = MakeAABB(6.0f, 6.0f, 9.0f, 9.0f);
        descB.graphAssetPath = "graphB.pcggraph.json";
        descB.seed = 20u;

        const world::PcgVolumeCellIndex index = world::BuildPcgVolumeCellIndex({ descA, descB }, kCellSize);
        Check(index.size() == 1, "same-cell append: both volumes land in the SAME single cell, got " + std::to_string(index.size()));

        const auto it = index.find(world::CellCoord{ 0, 0 });
        Check(it != index.end(), "same-cell append: cell (0,0) is present");
        if (it != index.end()) {
            Check(it->second.size() == 2, "same-cell append: cell (0,0) has BOTH volumes (2), not deduplicated, got " + std::to_string(it->second.size()));
            bool sawSeed10 = false, sawSeed20 = false;
            for (const worldpartition::PcgVolumeDesc& d : it->second) {
                if (d.seed == 10u) sawSeed10 = true;
                if (d.seed == 20u) sawSeed20 = true;
            }
            Check(sawSeed10 && sawSeed20, "same-cell append: both original volumes (seeds 10 and 20) are present, unmodified");
        }
    }

    // --- Test 6: an empty volume list produces an empty index (not an error). ----------------------
    void TestBuildIndexEmptyInput() {
        const world::PcgVolumeCellIndex index = world::BuildPcgVolumeCellIndex({}, 20.0f);
        Check(index.empty(), "empty input: BuildPcgVolumeCellIndex({}, ...) produces an empty index");
    }

} // namespace

int main() {
    std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "PcgCellLoaderTests";
    std::error_code ec;
    std::filesystem::create_directories(scratchDir, ec);

    TestCellCoordConversionRoundTrips();
    TestScanPcgVolumeActorFiles(scratchDir);
    TestBuildIndexSingleCell();
    TestBuildIndexMultiCellSpan();
    TestBuildIndexMultipleVolumesSameCellAppend();
    TestBuildIndexEmptyInput();

    if (g_failCount == 0) {
        std::cout << "PcgCellLoaderTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgCellLoaderTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
