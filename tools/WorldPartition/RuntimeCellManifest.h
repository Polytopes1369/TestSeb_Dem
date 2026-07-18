#pragma once
// Offline export of a built SpatialHashGrid + its actors' authored placements into a small, flat,
// POD binary format the SHIPPING executable can read without linking any of tools/WorldPartition/
// (see WorldPartitionTypes.h's own header comment for why tools/ never links into the .exe). This
// is the bridge between the offline OFPA/SceneIndex/SpatialHashGrid authoring chain and the
// runtime world::CellManifest reader (src/world/CellManifest.h) -- the two sides agree only on
// this file's plain byte layout, never on any worldpartition:: C++ type.
//
// One record per occupied cell (not per actor): this demo's streaming pool is a small bounded set
// of GPU entity slots (see VulkanContext::kStreamingUnitCount), so rather than pretend to stream
// every individual actor in a busy cell, BakeDemoWorld.cpp deliberately authors exactly one actor
// per cell and this format mirrors that -- one representative placement per cell is what the
// runtime loader actually instantiates. A denser multi-actor-per-cell scene is future scope, not
// silently approximated here: see RuntimeCellManifest.cpp's own comment.

#include <cstdint>
#include <filesystem>
#include <vector>

#include "SpatialHashGrid.h"

namespace worldpartition {

    inline constexpr uint32_t kRuntimeCellManifestMagic = 0x4D434C57u; // 'WLCM' little-endian.
    inline constexpr uint32_t kRuntimeCellManifestVersion = 1u;

    struct RuntimeCellManifestHeader {
        uint32_t magic = kRuntimeCellManifestMagic;
        uint32_t version = kRuntimeCellManifestVersion;
        float cellSize = 0.0f;   // Must match the world::StreamingManager cellSize the runtime constructs with.
        uint32_t recordCount = 0;
    };

    // One representative placement for one occupied ground-plane cell. Deliberately plain int32/
    // float fields only -- no worldpartition::Uuid, no worldpartition::CellCoord -- so the runtime
    // reader (src/world/CellManifest.h) never needs any tools/WorldPartition/ type or header.
    struct RuntimeCellManifestRecord {
        int32_t cellX = 0;
        int32_t cellZ = 0;
        uint32_t archetypeShape = 0;  // Index into the runtime's small fixed archetype shape table (see WorldCellStreamingLoader.h).
        float localOffsetX = 0.0f;    // World-space placement, already resolved (cell center + authored jitter) -- the runtime applies this verbatim as the entity's translation.
        float localOffsetY = 0.0f;
        float localOffsetZ = 0.0f;
    };

    // Derives one RuntimeCellManifestRecord per occupied cell in `grid`: takes the first actor
    // UUID in each SpatialHashCell (by construction there is ever exactly one, see BakeDemoWorld.cpp),
    // looks it up via `fetchActor`, and maps its className to an archetype shape index via
    // `classNameToShape`. Cells whose actor fails to resolve are skipped (never aborts the whole
    // export), matching RebuildSceneIndexFromActorFiles' own "one bad record must never block
    // every other cell" convention.
    using ActorClassNameFetchFn = std::function<bool(const Uuid&, std::string& outClassName, ActorTransform& outTransform)>;
    using ClassNameToArchetypeShapeFn = std::function<uint32_t(const std::string&)>;

    std::vector<RuntimeCellManifestRecord> BuildRuntimeCellManifest(
        const SpatialHashGrid& grid,
        const ActorClassNameFetchFn& fetchActor,
        const ClassNameToArchetypeShapeFn& classNameToShape);

    // Writes a flat [RuntimeCellManifestHeader][RuntimeCellManifestRecord * recordCount] file,
    // overwriting any existing file. Returns false on any I/O failure.
    bool WriteRuntimeCellManifest(const std::filesystem::path& filePath, float cellSize,
                                   const std::vector<RuntimeCellManifestRecord>& records);

}
