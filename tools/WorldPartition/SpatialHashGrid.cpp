#include "SpatialHashGrid.h"

#include <cmath>
#include <fstream>
#include <limits>

namespace worldpartition {

    namespace {
        inline constexpr uint32_t kGridFileMagic = 0x44524753u; // 'SGRD' little-endian.
        inline constexpr uint32_t kGridFileVersion = 1u;

        int32_t FloorToCell(float worldCoord, float cellSize) {
            return static_cast<int32_t>(std::floor(worldCoord / cellSize));
        }
    }

    SpatialHashGrid::SpatialHashGrid(float cellSize, GridDimension dimension)
        : cellSize_(cellSize), dimension_(dimension) {
    }

    CellCoord SpatialHashGrid::WorldToCell(const maths::vec3& worldPos) const {
        CellCoord coord;
        coord.x = FloorToCell(worldPos.x, cellSize_);
        coord.z = FloorToCell(worldPos.z, cellSize_);
        coord.y = (dimension_ == GridDimension::Grid3D) ? FloorToCell(worldPos.y, cellSize_) : 0;
        return coord;
    }

    AABB SpatialHashGrid::CellBounds(const CellCoord& coord) const {
        AABB box;
        box.boundsMin.x = static_cast<float>(coord.x) * cellSize_;
        box.boundsMax.x = box.boundsMin.x + cellSize_;
        box.boundsMin.z = static_cast<float>(coord.z) * cellSize_;
        box.boundsMax.z = box.boundsMin.z + cellSize_;

        if (dimension_ == GridDimension::Grid3D) {
            box.boundsMin.y = static_cast<float>(coord.y) * cellSize_;
            box.boundsMax.y = box.boundsMin.y + cellSize_;
        } else {
            box.boundsMin.y = std::numeric_limits<float>::lowest();
            box.boundsMax.y = std::numeric_limits<float>::max();
        }

        return box;
    }

    void SpatialHashGrid::Build(const std::vector<SceneIndexEntry>& actors) {
        cells_.clear();

        for (const SceneIndexEntry& actor : actors) {
            CellCoord cellMin = WorldToCell(actor.bounds.boundsMin);
            CellCoord cellMax = WorldToCell(actor.bounds.boundsMax);

            int32_t yBegin = (dimension_ == GridDimension::Grid3D) ? cellMin.y : 0;
            int32_t yEnd = (dimension_ == GridDimension::Grid3D) ? cellMax.y : 0;

            for (int32_t cy = yBegin; cy <= yEnd; ++cy) {
                for (int32_t cz = cellMin.z; cz <= cellMax.z; ++cz) {
                    for (int32_t cx = cellMin.x; cx <= cellMax.x; ++cx) {
                        CellCoord coord{ cx, cy, cz };
                        SpatialHashCell& cell = cells_[coord];
                        if (cell.actorUuids.empty()) {
                            // First insertion into this bucket (operator[] just default-constructed it):
                            // stamp its coord and reset its running AABB before accumulating into it.
                            cell.coord = coord;
                            maths::ResetAABB(cell.actorBounds.boundsMin, cell.actorBounds.boundsMax);
                        }
                        cell.actorUuids.push_back(actor.uuid);
                        maths::ExpandAABB(cell.actorBounds.boundsMin, cell.actorBounds.boundsMax, actor.bounds.boundsMin);
                        maths::ExpandAABB(cell.actorBounds.boundsMin, cell.actorBounds.boundsMax, actor.bounds.boundsMax);
                    }
                }
            }
        }
    }

    bool WriteSpatialHashGrid(const std::filesystem::path& filePath, const SpatialHashGrid& grid) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);

        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) return false;

        uint32_t magic = kGridFileMagic;
        uint32_t version = kGridFileVersion;
        float cellSize = grid.CellSize();
        uint32_t dimension = static_cast<uint32_t>(grid.Dimension());
        uint32_t cellCount = static_cast<uint32_t>(grid.CellCount());

        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&cellSize), sizeof(cellSize));
        out.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        out.write(reinterpret_cast<const char*>(&cellCount), sizeof(cellCount));

        for (const auto& [coord, cell] : grid.Cells()) {
            out.write(reinterpret_cast<const char*>(&coord), sizeof(coord));
            uint32_t actorCount = static_cast<uint32_t>(cell.actorUuids.size());
            out.write(reinterpret_cast<const char*>(&actorCount), sizeof(actorCount));
            if (actorCount > 0) {
                out.write(reinterpret_cast<const char*>(cell.actorUuids.data()),
                    static_cast<std::streamsize>(actorCount * sizeof(Uuid)));
            }
        }

        return out.good();
    }

}
