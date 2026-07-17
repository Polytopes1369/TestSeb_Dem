// Standalone, framework-free unit test for worldpartition::SpatialHashGrid
// (tools/WorldPartition/SpatialHashGrid.h). Exits 0 if every check passes, non-zero otherwise --
// registered with CTest (see the top-level CMakeLists.txt), matching this project's existing
// tests/*.cpp convention.

#include "WorldPartition/SpatialHashGrid.h"
#include "WorldPartition/Uuid.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    worldpartition::SceneIndexEntry MakeEntry(worldpartition::UuidGenerator& gen, const maths::vec3& boundsMin, const maths::vec3& boundsMax) {
        worldpartition::SceneIndexEntry entry;
        entry.uuid = gen.Generate();
        entry.bounds.boundsMin = boundsMin;
        entry.bounds.boundsMax = boundsMax;
        entry.streamingFlags = worldpartition::ActorStreamingFlags::SpatiallyLoaded;
        return entry;
    }

    void TestSingleCellActor() {
        worldpartition::UuidGenerator gen(1u);
        worldpartition::SceneIndexEntry actor = MakeEntry(gen, { 5.0f, 0.0f, 5.0f }, { 8.0f, 2.0f, 8.0f }); // Fully inside cell (0,0) at cellSize 10.

        worldpartition::SpatialHashGrid grid(10.0f, worldpartition::GridDimension::Grid2D);
        grid.Build({ actor });

        Check(grid.CellCount() == 1, "single interior actor should occupy exactly 1 cell");

        auto it = grid.Cells().find(worldpartition::CellCoord{ 0, 0, 0 });
        Check(it != grid.Cells().end(), "expected actor bucketed into cell (0,0,0)");
        if (it != grid.Cells().end()) {
            Check(it->second.actorUuids.size() == 1, "cell (0,0,0) should list exactly 1 actor");
            Check(!it->second.actorUuids.empty() && it->second.actorUuids[0] == actor.uuid, "cell (0,0,0) should list the inserted actor's uuid");
        }
    }

    void TestStraddlingActorDuplicatedAcrossCells() {
        worldpartition::UuidGenerator gen(2u);
        // Spans x in [-1, 1] at cellSize 10: touches cell x=-1 (covers [-10,0)) and cell x=0 (covers [0,10)).
        worldpartition::SceneIndexEntry actor = MakeEntry(gen, { -1.0f, 0.0f, 5.0f }, { 1.0f, 2.0f, 5.0f });

        worldpartition::SpatialHashGrid grid(10.0f, worldpartition::GridDimension::Grid2D);
        grid.Build({ actor });

        Check(grid.CellCount() == 2, "an actor straddling a cell boundary should occupy exactly 2 cells");

        for (int32_t cx : { -1, 0 }) {
            auto it = grid.Cells().find(worldpartition::CellCoord{ cx, 0, 0 });
            Check(it != grid.Cells().end(), "expected straddling actor bucketed into cell x=" + std::to_string(cx));
            if (it != grid.Cells().end()) {
                Check(it->second.actorUuids.size() == 1 && it->second.actorUuids[0] == actor.uuid,
                    "straddling actor should be duplicated (not split) into cell x=" + std::to_string(cx));
            }
        }
    }

    void TestGrid2DIgnoresHeight() {
        worldpartition::UuidGenerator gen(3u);
        worldpartition::SceneIndexEntry low = MakeEntry(gen, { 1.0f, 0.0f, 1.0f }, { 2.0f, 1.0f, 2.0f });
        worldpartition::SceneIndexEntry high = MakeEntry(gen, { 1.0f, 500.0f, 1.0f }, { 2.0f, 501.0f, 2.0f });

        worldpartition::SpatialHashGrid grid2D(10.0f, worldpartition::GridDimension::Grid2D);
        grid2D.Build({ low, high });
        Check(grid2D.CellCount() == 1, "Grid2D must collapse actors at different heights into the same column cell");

        worldpartition::SpatialHashGrid grid3D(10.0f, worldpartition::GridDimension::Grid3D);
        grid3D.Build({ low, high });
        Check(grid3D.CellCount() == 2, "Grid3D must bucket actors at very different heights into different cells");
    }

    void TestCellBoundsRoundTrip() {
        worldpartition::SpatialHashGrid grid(10.0f, worldpartition::GridDimension::Grid2D);
        worldpartition::CellCoord coord = grid.WorldToCell({ 23.5f, 0.0f, -7.0f });
        Check(coord.x == 2, "WorldToCell: expected x cell index 2 for world x=23.5 at cellSize 10");
        Check(coord.z == -1, "WorldToCell: expected z cell index -1 for world z=-7 at cellSize 10");

        worldpartition::AABB bounds = grid.CellBounds(coord);
        Check(bounds.boundsMin.x <= 23.5f && bounds.boundsMax.x > 23.5f, "CellBounds: computed cell should contain the original world x");
        Check(bounds.boundsMin.z <= -7.0f && bounds.boundsMax.z > -7.0f, "CellBounds: computed cell should contain the original world z");
    }

    void TestWriteSpatialHashGrid() {
        worldpartition::UuidGenerator gen(4u);
        std::vector<worldpartition::SceneIndexEntry> actors;
        for (int i = 0; i < 10; ++i) {
            float base = static_cast<float>(i) * 3.0f;
            actors.push_back(MakeEntry(gen, { base, 0.0f, base }, { base + 1.0f, 1.0f, base + 1.0f }));
        }

        worldpartition::SpatialHashGrid grid(10.0f, worldpartition::GridDimension::Grid2D);
        grid.Build(actors);

        std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "SpatialHashGridTests";
        std::error_code ec;
        std::filesystem::create_directories(scratchDir, ec);
        std::filesystem::path gridPath = scratchDir / "grid.bin";

        Check(worldpartition::WriteSpatialHashGrid(gridPath, grid), "WriteSpatialHashGrid should succeed for a non-empty built grid");
        Check(std::filesystem::exists(gridPath) && std::filesystem::file_size(gridPath) > 0, "WriteSpatialHashGrid should produce a non-empty file");
    }

}

int main() {
    TestSingleCellActor();
    TestStraddlingActorDuplicatedAcrossCells();
    TestGrid2DIgnoresHeight();
    TestCellBoundsRoundTrip();
    TestWriteSpatialHashGrid();

    if (g_failCount == 0) {
        std::cout << "[PASS] All SpatialHashGrid checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
