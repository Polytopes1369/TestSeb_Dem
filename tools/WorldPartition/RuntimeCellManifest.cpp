#include "RuntimeCellManifest.h"

#include <fstream>

namespace worldpartition {

    std::vector<RuntimeCellManifestRecord> BuildRuntimeCellManifest(
        const SpatialHashGrid& grid,
        const ActorClassNameFetchFn& fetchActor,
        const ClassNameToArchetypeShapeFn& classNameToShape) {

        std::vector<RuntimeCellManifestRecord> records;
        records.reserve(grid.CellCount());

        for (const auto& [coord, cell] : grid.Cells()) {
            if (cell.actorUuids.empty()) continue;

            // One representative actor per cell -- see this header's own comment for why. Always
            // the first (and, by BakeDemoWorld.cpp's own authoring convention, only) UUID.
            std::string className;
            ActorTransform transform;
            if (!fetchActor(cell.actorUuids[0], className, transform)) continue; // Skip, don't abort -- same convention as RebuildSceneIndexFromActorFiles.

            RuntimeCellManifestRecord record;
            record.cellX = coord.x;
            record.cellZ = coord.z;
            record.archetypeShape = classNameToShape(className);
            record.localOffsetX = transform.position.x;
            record.localOffsetY = transform.position.y;
            record.localOffsetZ = transform.position.z;
            records.push_back(record);
        }

        return records;
    }

    bool WriteRuntimeCellManifest(const std::filesystem::path& filePath, float cellSize,
                                   const std::vector<RuntimeCellManifestRecord>& records) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);

        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) return false;

        RuntimeCellManifestHeader header;
        header.cellSize = cellSize;
        header.recordCount = static_cast<uint32_t>(records.size());
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        if (!records.empty()) {
            out.write(reinterpret_cast<const char*>(records.data()),
                static_cast<std::streamsize>(records.size() * sizeof(RuntimeCellManifestRecord)));
        }

        return out.good();
    }

}
