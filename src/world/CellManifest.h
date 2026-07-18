#pragma once
// Runtime reader for the flat binary manifest tools/WorldPartition/BakeDemoWorld.cpp exports (see
// tools/WorldPartition/RuntimeCellManifest.h for the shared byte layout both sides agree on).
// Deliberately depends on nothing under tools/WorldPartition/ -- only the plain POD record layout,
// duplicated here by convention (see StreamingTypes.h's own header comment on why src/ and tools/
// never share C++ types even when they describe the same on-disk format).
//
// Small enough to read wholesale at streaming-system startup (this demo's world_data/cellmanifest.bin
// holds on the order of dozens of records) -- exactly the same "index is cheap, read it all up
// front" reasoning SceneIndex.h's own header comment gives for its own (much larger, per-actor)
// index.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <unordered_map>

#include "StreamingTypes.h"
#include "core/maths/Maths.h"

namespace world {

    // One representative prop placement for one occupied cell -- see RuntimeCellManifest.h's own
    // comment for why this is one-per-cell rather than a full actor list.
    struct CellPlacement {
        uint32_t archetypeShape = 0;
        maths::vec3 worldPosition{};
    };

    class CellManifest {
    public:
        // Reads `filePath` (the file BakeDemoWorld.cpp's WriteRuntimeCellManifest wrote). Returns
        // false on any I/O failure, magic/version mismatch, or truncated record table -- exactly
        // the same failure contract as worldpartition::ReadSceneIndex, whose format this mirrors.
        bool Load(const std::filesystem::path& filePath);

        bool IsLoaded() const { return m_Loaded; }
        float CellSize() const { return m_CellSize; }
        size_t RecordCount() const { return m_Placements.size(); }

        // Returns the cell's placement, or std::nullopt if this cell has no authored content (most
        // cells, outside the small baked demo grid, will not).
        std::optional<CellPlacement> GetPlacement(const CellCoord& coord) const;

    private:
        bool m_Loaded = false;
        float m_CellSize = 0.0f;
        std::unordered_map<CellCoord, CellPlacement, CellCoordHash> m_Placements;
    };

}
