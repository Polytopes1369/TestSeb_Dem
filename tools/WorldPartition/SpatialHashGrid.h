#pragma once
// Offline spatial hash grid over a scene index (SceneIndex.h): buckets every actor's AABB into
// the fixed-size world-space cell(s) it overlaps. This is the structure both the HLOD generation
// pipeline (HlodPipeline.h -- "which actors belong to this HLOD cell?") and, at runtime, the
// streaming source evaluation ("which cells are within my load radius?") key off of; this file
// only builds and (de)serializes it offline, it does not itself decide what streams.
//
// Cell coordinates are integer, computed as floor(worldPos / cellSize) per axis. In Grid2D mode
// (the default, matching UE5.8's own World Partition runtime grid) the Y axis is collapsed: every
// cell spans the full height column (its CellBounds's Y extent is [-inf, +inf]), matching how a
// 2D open-world grid actually partitions a level -- an actor's height above/below the ground
// plane never changes which cell it belongs to. Grid3D mode buckets all three axes, useful for a
// volumetric HLOD chain (e.g. deep cave systems, space scenes) where Y-collapsing would put
// wildly different-altitude actors in the same cell.

#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include "SceneIndex.h"
#include "Uuid.h"
#include "WorldPartitionTypes.h"

namespace worldpartition {

    enum class GridDimension { Grid2D, Grid3D };

    struct CellCoord {
        int32_t x = 0;
        int32_t y = 0; // Always 0 in Grid2D mode.
        int32_t z = 0;

        constexpr bool operator==(const CellCoord& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct CellCoordHash {
        size_t operator()(const CellCoord& c) const noexcept {
            // 32-bit hash-combine chain (same constant/shift pattern as std::hash<Uuid>, see
            // Uuid.h) so all three axes contribute to the bucket even at small grid extents where
            // x/y/z individually only span a handful of values.
            size_t h = std::hash<int32_t>{}(c.x);
            h ^= std::hash<int32_t>{}(c.y) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>{}(c.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct SpatialHashCell {
        CellCoord coord;
        std::vector<Uuid> actorUuids;

        // Union of every placed actor's AABB in this cell (NOT clipped to the cell's own
        // geometric bounds -- an actor straddling a boundary contributes its full extent here).
        // This is what HlodPipeline.h uses to size a cell's merged proxy mesh's bounds without
        // re-scanning the scene index.
        AABB actorBounds;
    };

    // Builds and owns the cell -> actor-list mapping for one fixed cellSize/dimension pair. Not
    // copyable-cheap by design (cells_ can hold references to every actor in the level) -- move
    // it, don't copy it.
    class SpatialHashGrid {
    public:
        SpatialHashGrid(float cellSize, GridDimension dimension = GridDimension::Grid2D);

        // Clears any previously built state and re-buckets every entry in `actors`. An actor
        // whose AABB spans multiple cells is inserted into ALL of them (never split or
        // deduplicated to a single "owning" cell) -- HLOD cell grouping needs each cell's actor
        // list to be geometrically complete on its own, and a split actor would produce a visible
        // seam at the cell boundary in the merged proxy mesh.
        void Build(const std::vector<SceneIndexEntry>& actors);

        CellCoord WorldToCell(const maths::vec3& worldPos) const;

        // World-space bounds of the cell at `coord`. In Grid2D mode, boundsMin.y/boundsMax.y are
        // -infinity/+infinity (the cell's column has no height limit); in Grid3D mode all 3 axes
        // are bounded to exactly cellSize_.
        AABB CellBounds(const CellCoord& coord) const;

        const std::unordered_map<CellCoord, SpatialHashCell, CellCoordHash>& Cells() const { return cells_; }
        size_t CellCount() const { return cells_.size(); }

        float CellSize() const { return cellSize_; }
        GridDimension Dimension() const { return dimension_; }

    private:
        float cellSize_;
        GridDimension dimension_;
        std::unordered_map<CellCoord, SpatialHashCell, CellCoordHash> cells_;
    };

    // On-disk export of a built grid: [header][cellCount * (CellCoord, actorCount, actorUuids)].
    // Consumed by the runtime streaming source (not part of this offline toolset) so it never has
    // to re-run Build() over the full scene index at load time. Returns false on any I/O failure.
    bool WriteSpatialHashGrid(const std::filesystem::path& filePath, const SpatialHashGrid& grid);

}
