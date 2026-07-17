// Standalone, framework-free unit test for the World Partition HLOD pipeline architecture
// (tools/WorldPartition/HlodPipeline.h): mesh gathering, merging, the native QEM simplification
// backend, the HLOD level-chain builder, and the uniform-grid atlas packer. Exits 0 if every
// check passes, non-zero otherwise -- registered with CTest (see the top-level CMakeLists.txt),
// matching this project's existing tests/*.cpp convention.

#include "WorldPartition/HlodPipeline.h"
#include "WorldPartition/Uuid.h"
#include "geometry/MeshSimplifier.h"

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

    // A simple 2-triangle quad, offset on X so two instances never share a vertex position.
    geometry::SimplifiableMesh MakeQuad(float offsetX) {
        geometry::SimplifiableMesh mesh;
        mesh.positions = {
            { offsetX + 0.0f, 0.0f, 0.0f },
            { offsetX + 1.0f, 0.0f, 0.0f },
            { offsetX + 1.0f, 1.0f, 0.0f },
            { offsetX + 0.0f, 1.0f, 0.0f },
        };
        mesh.uvs = { {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f} };
        mesh.locked = { false, false, false, false };
        mesh.triangles = { 0, 1, 2, 0, 2, 3 };
        return mesh;
    }

    void TestGatherAndMerge() {
        worldpartition::UuidGenerator gen(42u);
        worldpartition::Uuid uuidA = gen.Generate();
        worldpartition::Uuid uuidB = gen.Generate();
        worldpartition::Uuid uuidMissing = gen.Generate();

        worldpartition::SpatialHashCell cell;
        cell.coord = { 0, 0, 0 };
        cell.actorUuids = { uuidA, uuidB, uuidMissing };

        worldpartition::ActorMeshFetchFn fetch = [&](const worldpartition::Uuid& id, geometry::SimplifiableMesh& out) -> bool {
            if (id == uuidA) { out = MakeQuad(0.0f); return true; }
            if (id == uuidB) { out = MakeQuad(10.0f); return true; }
            return false; // uuidMissing: no mesh contributed (e.g. a non-visual actor) -- must be skipped, not an error.
            };

        std::vector<geometry::SimplifiableMesh> gathered = worldpartition::GatherCellMeshes(cell, fetch);
        Check(gathered.size() == 2, "GatherCellMeshes should skip the actor its fetch callback rejects");

        geometry::SimplifiableMesh merged = worldpartition::MergeCellMeshes(gathered);
        Check(merged.positions.size() == 8, "MergeCellMeshes should concatenate both quads' 4 vertices each");
        Check(merged.triangles.size() == 12, "MergeCellMeshes should concatenate both quads' 2 triangles each (6 indices each)");

        // Second source mesh's triangle indices must be offset by the first mesh's vertex count (4).
        bool secondMeshOffsetCorrect = merged.triangles.size() >= 12 &&
            merged.triangles[6] == 4 && merged.triangles[7] == 5 && merged.triangles[8] == 6;
        Check(secondMeshOffsetCorrect, "MergeCellMeshes: second source mesh's triangle indices were not correctly offset");
    }

    void TestBuildHlodForCell() {
        worldpartition::UuidGenerator gen(43u);
        worldpartition::Uuid uuidA = gen.Generate();
        worldpartition::Uuid uuidB = gen.Generate();

        worldpartition::SpatialHashCell cell;
        cell.coord = { 0, 0, 0 };
        cell.actorUuids = { uuidA, uuidB };

        worldpartition::ActorMeshFetchFn fetch = [&](const worldpartition::Uuid& id, geometry::SimplifiableMesh& out) -> bool {
            if (id == uuidA) { out = MakeQuad(0.0f); return true; }
            if (id == uuidB) { out = MakeQuad(10.0f); return true; }
            return false;
            };

        worldpartition::NativeQEMSimplificationBackend backend;
        worldpartition::HlodLevel level;
        level.levelIndex = 0;
        level.cellSize = 100.0f;
        level.triangleBudget = 2;

        worldpartition::HlodProxyMesh proxy = worldpartition::BuildHlodForCell(cell, fetch, level, backend);

        uint32_t triangleCount = static_cast<uint32_t>(proxy.mesh.triangles.size() / 3);
        Check(triangleCount <= 4, "BuildHlodForCell: simplification must never increase the original triangle count");

        if (!proxy.mesh.positions.empty()) {
            Check(proxy.bounds.boundsMin.x <= proxy.bounds.boundsMax.x &&
                proxy.bounds.boundsMin.y <= proxy.bounds.boundsMax.y &&
                proxy.bounds.boundsMin.z <= proxy.bounds.boundsMax.z,
                "BuildHlodForCell: resulting bounds must be a valid (min <= max) AABB");

            // Both source quads span x in [0,1] and [10,11]; the merged/simplified proxy's bounds
            // must still cover that full combined extent (simplification moves geometry, but
            // never discards a whole disjoint component when unconstrained by any lock).
            Check(proxy.bounds.boundsMin.x <= 0.5f && proxy.bounds.boundsMax.x >= 10.5f,
                "BuildHlodForCell: proxy bounds should still span both source quads' X extent");
        }
    }

    void TestBuildHlodLevelChain() {
        std::vector<worldpartition::HlodLevel> levels = worldpartition::BuildHlodLevelChain(100.0f, 500u, 3u);
        Check(levels.size() == 3, "BuildHlodLevelChain: expected exactly numLevels entries");

        if (levels.size() == 3) {
            Check(levels[0].levelIndex == 0 && levels[1].levelIndex == 1 && levels[2].levelIndex == 2, "BuildHlodLevelChain: levelIndex should be sequential");
            Check(levels[0].cellSize == 100.0f && levels[1].cellSize == 200.0f && levels[2].cellSize == 400.0f, "BuildHlodLevelChain: cellSize should double each level");
            Check(levels[0].triangleBudget == 500u && levels[1].triangleBudget == 500u && levels[2].triangleBudget == 500u, "BuildHlodLevelChain: triangleBudget should stay constant across levels");
        }
    }

    void TestShelfPackAtlasBaker() {
        worldpartition::ShelfPackAtlasBaker baker;

        std::vector<uint32_t> materials = { 1, 2, 3, 4 };
        std::vector<worldpartition::HlodAtlasTile> tiles;
        bool ok = baker.PackMaterialsIntoAtlas(materials, 64u, 128u, tiles);

        Check(ok, "ShelfPackAtlasBaker: 4 tiles of 64 should fit a 128x128 atlas (2x2 grid)");
        Check(tiles.size() == 4, "ShelfPackAtlasBaker: expected one tile per material");

        if (tiles.size() == 4) {
            Check(tiles[0].atlasOffsetX == 0 && tiles[0].atlasOffsetY == 0, "ShelfPackAtlasBaker: tile 0 should be at origin");
            Check(tiles[1].atlasOffsetX == 64 && tiles[1].atlasOffsetY == 0, "ShelfPackAtlasBaker: tile 1 should be to the right of tile 0");
            Check(tiles[2].atlasOffsetX == 0 && tiles[2].atlasOffsetY == 64, "ShelfPackAtlasBaker: tile 2 should start the second row");
            Check(tiles[3].atlasOffsetX == 64 && tiles[3].atlasOffsetY == 64, "ShelfPackAtlasBaker: tile 3 should be the second row's second column");
        }

        std::vector<uint32_t> tooMany = { 1, 2, 3, 4, 5 };
        std::vector<worldpartition::HlodAtlasTile> overflowTiles;
        bool overflowOk = baker.PackMaterialsIntoAtlas(tooMany, 64u, 128u, overflowTiles);
        Check(!overflowOk, "ShelfPackAtlasBaker: 5 tiles of 64 must NOT fit a 128x128 atlas (capacity is 4)");
    }

}

int main() {
    TestGatherAndMerge();
    TestBuildHlodForCell();
    TestBuildHlodLevelChain();
    TestShelfPackAtlasBaker();

    if (g_failCount == 0) {
        std::cout << "[PASS] All HLOD pipeline checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
