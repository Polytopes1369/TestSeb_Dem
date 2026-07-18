#include "PcgVolumeCellIndex.h"
#ifndef NDEBUG

#include "core/Logger.h"
#include "WorldPartition/OfpaActor.h"

#include <format>

namespace world {

    std::vector<worldpartition::PcgVolumeDesc> ScanPcgVolumeActorFiles(const std::filesystem::path& actorsRootDir) {
        std::vector<worldpartition::PcgVolumeDesc> result;

        if (!std::filesystem::exists(actorsRootDir)) return result; // No authored content at all -- not an error, see this header's own comment.

        // Walks the 2-hex-char shard subfolders worldpartition::MakeActorFilePath creates, exactly
        // the way renderer::debug::PcgVolumeInspector::ScanActorsDirectory and
        // RebuildSceneIndexFromActorFiles already do (see this file's own header comment).
        for (const std::filesystem::directory_entry& dirEntry :
            std::filesystem::recursive_directory_iterator(actorsRootDir)) {

            if (!dirEntry.is_regular_file() || dirEntry.path().extension() != ".actor") continue;

            worldpartition::ActorRecord record;
            if (!worldpartition::ReadActorFile(dirEntry.path(), record)) {
                LOG_WARNING(std::format("[PcgVolumeCellIndex] Failed to read actor file '{}' -- skipping.", dirEntry.path().string()));
                continue;
            }

            worldpartition::PcgVolumeDesc desc;
            if (!worldpartition::TryParsePcgVolumeDesc(record, desc)) continue; // Not a PcgVolume actor (or malformed) -- silently skip.

            result.push_back(std::move(desc));
        }

        return result;
    }

    PcgVolumeCellIndex BuildPcgVolumeCellIndex(const std::vector<worldpartition::PcgVolumeDesc>& volumes, float cellSize) {
        PcgVolumeCellIndex index;

        for (const worldpartition::PcgVolumeDesc& desc : volumes) {
            const std::vector<worldpartition::CellCoord> overlappingCells =
                worldpartition::ComputeOverlappingCells(desc.bounds, cellSize);

            for (const worldpartition::CellCoord& offlineCoord : overlappingCells) {
                index[ToRuntimeCellCoord(offlineCoord)].push_back(desc);
            }
        }

        return index;
    }

}
#endif
