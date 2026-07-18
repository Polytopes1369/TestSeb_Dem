#pragma once
// PCG framework roadmap, Phase 6.3 ("Runtime Generator Hook"): the PURE, Vulkan/PcgInstanceSpawnManager-
// free half of world::PcgCellLoader's own job -- discovering authored PCG Volume actor files on disk
// and bucketing them into a world::CellCoord (runtime, 2-axis) -> overlapping-volumes index, so
// PcgCellLoader itself never has to re-scan disk on every single LoadCellFullDetail()/LoadCellHlod()
// call (see PcgCellLoader.h's own header comment for the full rationale). Split into its own
// header/.cpp specifically so this logic -- and ONLY this logic -- can be exercised by a standalone,
// framework-free CTest (tests/PcgCellLoaderTests.cpp) with ZERO Vulkan/renderer::PcgInstanceDrawPass
// dependency: PcgCellLoader.h itself pulls in pcg::PcgInstanceSpawnManager (a live-rendering type),
// which would drag the full renderer pass machinery into any test that merely wants to check "did the
// index bucket this volume into the right cell" -- see PcgCellLoader.h's own top-of-file comment for
// why that split matters architecturally, not just for test convenience.
//
// --- Debug-only, whole file (see the #ifndef NDEBUG guard below) -------------------------------
// ScanPcgVolumeActorFiles()/BuildPcgVolumeCellIndex() call worldpartition::ReadActorFile /
// worldpartition::TryParsePcgVolumeDesc / worldpartition::ComputeOverlappingCells -- all THREE are
// OUT-OF-LINE, COMPILED functions defined in tools/WorldPartition/OfpaActor.cpp / PcgVolumeActor.cpp,
// which the top-level CMakeLists.txt (see its own "WORLD PARTITION OFFLINE TOOLSET" section, and the
// WORLDPARTITION_PCG_VOLUME_SOURCES variable specifically) links into the shipping DemoSceneVK target
// ONLY for the Debug configuration -- the Release configuration links ZERO of tools/WorldPartition/'s
// own compiled object code, matching WorldPartitionTypes.h's own header comment ("never linked into
// the shipping demo executable") and PcgVolumeInspector.h's own precedent (the only OTHER place in
// this codebase that reaches these same three functions from a file under src/, also whole-file
// Debug-only for exactly this reason). This is a real, current scope limitation this phase inherits
// rather than resolves (re-implementing OFPA's binary ActorRecord parser independently under src/,
// the way src/world/CellManifest.cpp independently re-implements ITS OWN much simpler flat-record
// format rather than linking tools/WorldPartition/RuntimeCellManifest.cpp, would be a much larger,
// separate undertaking -- out of scope for "Phase 6.3: Runtime Generator Hook", and unnecessary today
// since nothing in this codebase yet AUTHORS a real PcgVolume actor file for a Release-relevant
// scenario either, see PcgVolumeInspector.h's own "Demo dataset fallback" comment). A future phase
// that actually needs Release-shipping PCG Volume discovery is where that larger call belongs.
#ifndef NDEBUG

#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

#include "StreamingTypes.h"
#include "WorldPartition/PcgVolumeActor.h" // worldpartition::PcgVolumeDesc, worldpartition::CellCoord (via SpatialHashGrid.h), worldpartition::AABB, ComputeOverlappingCells

namespace world {

    // world::CellCoord (runtime, 2-axis) -> worldpartition::CellCoord (offline authoring grid,
    // 3-axis, y always 0 -- see StreamingTypes.h's own header comment for why these are two
    // deliberately independent types, and PcgCellGenerator.h's own "Two-namespace CellCoord
    // situation" comment for the fuller cross-reference). A one-line, always-lossless round trip in
    // BOTH directions as long as the offline side's y stays 0, which SpatialHashGrid's own
    // ComputeOverlappingCells (Grid2D mode, the only mode this codebase's runtime streaming grid
    // ever uses) already guarantees by construction -- see that function's own header comment.
    constexpr worldpartition::CellCoord ToOfflineCellCoord(const CellCoord& c) {
        return worldpartition::CellCoord{ c.x, 0, c.z };
    }
    constexpr CellCoord ToRuntimeCellCoord(const worldpartition::CellCoord& c) {
        return CellCoord{ c.x, c.z };
    }

    // cellCoord -> every PcgVolumeDesc whose authored bounds overlap it, per
    // worldpartition::ComputeOverlappingCells' own Grid2D convention (matching PcgCellLoader's own
    // cellSize, see that class' own constructor comment for why cellSize is always an explicit,
    // caller-supplied parameter here rather than a hardcoded global).
    using PcgVolumeCellIndex = std::unordered_map<CellCoord, std::vector<worldpartition::PcgVolumeDesc>, CellCoordHash>;

    // Scans `actorsRootDir` recursively for "*.actor" files, keeping only the ones
    // worldpartition::TryParsePcgVolumeDesc actually recognizes as a PcgVolume -- every other
    // className present is silently skipped (matches RebuildSceneIndexFromActorFiles' own "one
    // bad/unrecognized record must never block the rest" convention, see SceneIndex.h). Mirrors
    // renderer::debug::PcgVolumeInspector::ScanActorsDirectory's own scan approach (Phase 7.4) --
    // reused as a PATTERN here, not as shared code, since that class returns its own richer
    // PcgVolumeInspectorEntry (UUID, on-disk path, ImGui edit-buffer state) this purely-logic caller
    // has no use for. Returns an empty vector (never an error) if `actorsRootDir` does not exist --
    // matches this codebase's "missing authored content is additive-not-fatal" convention (see e.g.
    // world::CellManifest::Load's own call site in main.cpp).
    std::vector<worldpartition::PcgVolumeDesc> ScanPcgVolumeActorFiles(const std::filesystem::path& actorsRootDir);

    // Buckets every entry of `volumes` into the cell(s) it overlaps (worldpartition::
    // ComputeOverlappingCells(desc.bounds, cellSize), converted to world::CellCoord via
    // ToRuntimeCellCoord above), appending -- NOT deduplicating -- into each overlapped cell's own
    // vector. A volume spanning N cells therefore appears once in each of those N cells' own entry,
    // exactly the "an actor straddling a boundary contributes its full extent to every cell it
    // touches" convention worldpartition::SpatialHashGrid::Build already documents for ordinary
    // actors. `cellSize` must match whatever cell size the runtime streaming grid actually uses
    // (world::CellManifest::CellSize() when streaming is enabled) -- see PcgCellLoader's own
    // constructor comment for why this is always an explicit parameter, never a hardcoded literal.
    PcgVolumeCellIndex BuildPcgVolumeCellIndex(const std::vector<worldpartition::PcgVolumeDesc>& volumes, float cellSize);

}
#endif
