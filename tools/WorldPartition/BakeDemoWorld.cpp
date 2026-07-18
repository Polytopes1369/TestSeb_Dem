// Offline content-authoring tool for the runtime World Partition streaming demo: deterministically
// scatters one procedural-prop actor per cell across a small grid, writes real OFPA .actor files
// (OfpaActor.h), rebuilds the SceneIndex (SceneIndex.h) and SpatialHashGrid (SpatialHashGrid.h)
// from them exactly the way a real editor cook step would, then exports the runtime-readable
// RuntimeCellManifest.h consumed by world::CellManifest at startup.
//
// This is a standalone dev-time tool, run manually before launching the shipping app -- it links
// tools/WorldPartition/ (which never links into DemoSceneVK.exe, see WorldPartitionTypes.h) and
// writes its output under world_data/, a plain generated-content directory the app reads the same
// way it already reads scene.cache (see main.cpp's own "build if missing/stale" comment for
// scene.cache -- world_data/ is the same idea, just produced by a separate offline step instead of
// the shipping exe itself, since the exe cannot link this authoring code).
//
// Deliberately fully deterministic (seeded UuidGenerator, no wall-clock/OS entropy in placement) --
// same reasoning as renderer::GenerateShowcaseMaterialTable's own "a demoscene demo is a fixed
// procedural performance" comment: re-running this tool must always reproduce the exact same world.

#include <iostream>
#include <string>
#include <vector>

#include "OfpaActor.h"
#include "SceneIndex.h"
#include "SpatialHashGrid.h"
#include "RuntimeCellManifest.h"

namespace {

    // Must exactly match the cellSize world::StreamingManager is constructed with in main.cpp --
    // see that call site's own cross-reference comment.
    constexpr float kDemoWorldCellSize = 20.0f;

    // Grid of cells around a center deliberately well clear of the fixed showcase gallery (whose
    // 12 primitives live within a few meters of the origin, see VulkanContext::GridSlot) so flying
    // the camera between the two areas visibly demonstrates cells streaming in/out.
    constexpr float kWorldCenterX = 80.0f;
    constexpr float kWorldCenterZ = 0.0f;
    constexpr int32_t kGridRadiusCells = 3; // (2*3+1)^2 = 49 cells/actors.

    // Mirrors WorldCellStreamingLoader.h's own archetype shape table -- kept as a plain string
    // list here (not a shared enum) since this tool has zero compile-time dependency on src/world/
    // by design (tools/ never depends on src/, only the other direction is ever true elsewhere in
    // this codebase, and here neither direction exists at all -- the two sides only agree on the
    // RuntimeCellManifest byte format).
    const std::vector<std::string> kArchetypeClassNames = { "Rock", "Bush", "Tree", "Debris" };

    uint32_t ClassNameToArchetypeShape(const std::string& className) {
        for (size_t i = 0; i < kArchetypeClassNames.size(); ++i) {
            if (kArchetypeClassNames[i] == className) return static_cast<uint32_t>(i);
        }
        return 0u; // Unknown className: fall back to shape 0, never a hard failure (offline tooling convention throughout this module).
    }

}

int main() {
    const std::filesystem::path worldDataRoot = "world_data";
    const std::filesystem::path actorsRoot = worldDataRoot / "actors";
    const std::filesystem::path sceneIndexPath = worldDataRoot / "scene.index";
    const std::filesystem::path manifestPath = worldDataRoot / "cellmanifest.bin";

    worldpartition::UuidGenerator uuidGen(0xD3D5CE7E5EEDULL); // Fixed seed: see file header comment on determinism.

    std::cout << "[BakeDemoWorld] Authoring " << ((2 * kGridRadiusCells + 1) * (2 * kGridRadiusCells + 1))
              << " actors across a " << (2 * kGridRadiusCells + 1) << "x" << (2 * kGridRadiusCells + 1)
              << " cell grid (cellSize=" << kDemoWorldCellSize << ")...\n";

    uint32_t actorsWritten = 0;
    for (int32_t cz = -kGridRadiusCells; cz <= kGridRadiusCells; ++cz) {
        for (int32_t cx = -kGridRadiusCells; cx <= kGridRadiusCells; ++cx) {
            // Deterministic per-cell jitter (not random): spreads actors off the exact cell-center
            // grid without breaking reproducibility -- a simple hash of the cell coordinate, folded
            // into [-0.3, 0.3] of a cell width.
            uint32_t h = static_cast<uint32_t>(cx * 73856093) ^ static_cast<uint32_t>(cz * 19349663);
            float jitterX = (static_cast<float>(h % 1000u) / 1000.0f - 0.5f) * 0.6f * kDemoWorldCellSize;
            h = h * 2654435761u + 1u;
            float jitterZ = (static_cast<float>(h % 1000u) / 1000.0f - 0.5f) * 0.6f * kDemoWorldCellSize;

            float worldX = kWorldCenterX + (static_cast<float>(cx) + 0.5f) * kDemoWorldCellSize + jitterX;
            float worldZ = kWorldCenterZ + (static_cast<float>(cz) + 0.5f) * kDemoWorldCellSize + jitterZ;

            uint32_t shapeIndex = static_cast<uint32_t>((cx + cz + 2 * kGridRadiusCells)) % static_cast<uint32_t>(kArchetypeClassNames.size());

            worldpartition::ActorRecord record;
            record.uuid = uuidGen.Generate();
            record.className = kArchetypeClassNames[shapeIndex];
            record.actorLabel = record.className + "_" + std::to_string(cx) + "_" + std::to_string(cz);
            record.transform.position = maths::vec3{ worldX, 0.0f, worldZ };
            record.localBounds = worldpartition::AABB{ maths::vec3{-0.6f, -0.6f, -0.6f}, maths::vec3{0.6f, 0.6f, 0.6f} };
            record.streamingFlags = worldpartition::ActorStreamingFlags::SpatiallyLoaded;
            record.RecomputeWorldBounds();

            std::filesystem::path actorPath = worldpartition::MakeActorFilePath(actorsRoot, record.uuid);
            if (!worldpartition::WriteActorFile(actorPath, record)) {
                std::cerr << "[BakeDemoWorld] Failed to write actor file: " << actorPath << "\n";
                return 1;
            }
            ++actorsWritten;
        }
    }
    std::cout << "[BakeDemoWorld] Wrote " << actorsWritten << " OFPA actor files under " << actorsRoot << "\n";

    std::vector<worldpartition::SceneIndexEntry> sceneEntries =
        worldpartition::RebuildSceneIndexFromActorFiles(actorsRoot);
    if (!worldpartition::WriteSceneIndex(sceneIndexPath, sceneEntries)) {
        std::cerr << "[BakeDemoWorld] Failed to write scene index: " << sceneIndexPath << "\n";
        return 1;
    }
    std::cout << "[BakeDemoWorld] Rebuilt scene index (" << sceneEntries.size() << " entries) at " << sceneIndexPath << "\n";

    worldpartition::SpatialHashGrid grid(kDemoWorldCellSize, worldpartition::GridDimension::Grid2D);
    grid.Build(sceneEntries);
    std::cout << "[BakeDemoWorld] Built spatial hash grid: " << grid.CellCount() << " occupied cells.\n";

    auto fetchActor = [&actorsRoot](const worldpartition::Uuid& uuid, std::string& outClassName,
                                     worldpartition::ActorTransform& outTransform) -> bool {
        worldpartition::ActorRecord record;
        if (!worldpartition::ReadActorFile(worldpartition::MakeActorFilePath(actorsRoot, uuid), record)) return false;
        outClassName = record.className;
        outTransform = record.transform;
        return true;
    };

    std::vector<worldpartition::RuntimeCellManifestRecord> manifest =
        worldpartition::BuildRuntimeCellManifest(grid, fetchActor, ClassNameToArchetypeShape);

    if (!worldpartition::WriteRuntimeCellManifest(manifestPath, kDemoWorldCellSize, manifest)) {
        std::cerr << "[BakeDemoWorld] Failed to write runtime cell manifest: " << manifestPath << "\n";
        return 1;
    }
    std::cout << "[BakeDemoWorld] Exported runtime cell manifest (" << manifest.size() << " records) to "
              << manifestPath << "\n";
    std::cout << "[BakeDemoWorld] Done. Copy/keep world_data/ next to DemoSceneVK.exe for streaming to activate.\n";

    return 0;
}
