#pragma once
// PCG roadmap, Phase 6.1 ("PCG Volume Authoring"): an authorable bounding region that references a
// PCG Graph asset (pcg::PcgGraph, see src/pcg/PcgGraph.h) and defines the world-space bounds within
// which that graph should generate content once actually executed. This is the bridge between
// "a graph exists" (Phase 5, already merged) and "content actually gets generated somewhere in the
// world, tied to World Partition cell streaming" (Phase 6.2+, later work -- NOT implemented here).
//
// Deliberately NOT a new parallel file format: a PCG Volume is just an ordinary
// worldpartition::ActorRecord (see OfpaActor.h) whose className is kPcgVolumeClassName and whose
// PCG-specific data is packed into a handful of ordinary PropertyEntry items -- exactly the same
// generic-record convention BakeDemoWorld.cpp's "ProceduralTree"/"Rock"/"Bush" actors already use.
// This means a PCG Volume authors, serializes, and round-trips through the exact same
// WriteActorFile/ReadActorFile pipeline as every other actor in this codebase, with zero changes to
// OfpaActor.h/.cpp -- BuildPcgVolumeActorRecord/TryParsePcgVolumeDesc below are purely a typed
// pack/unpack convenience layer on top of that generic record, nothing more.
//
// Reference vs. embed (a design choice this phase's task brief explicitly calls out): the graph a
// volume executes is stored as a REFERENCE (a relative path string to a separate PcgGraph JSON file
// on disk, see pcg::PcgGraph::SerializeToJson/DeserializeFromJson), NOT as the graph's serialized
// JSON embedded inline in the actor's properties. Chosen over embedding because:
//   1. Reusability: a real content pipeline places many PCG Volumes referencing the SAME graph
//      asset (e.g. a hundred "scatter rocks along this shoreline" volumes all pointing at one
//      "RockScatter.pcggraph.json"). Embedding would duplicate that graph's full JSON in every
//      single actor file, and editing the graph would require re-authoring every volume that uses
//      it. A path reference keeps the graph a single source of truth, exactly the same
//      asset-reference relationship BakeDemoWorld.cpp's className already has to its interpreting
//      importer/spawner tool.
//   2. Actor file size / OFPA scalability: OFPA's whole reason for existing (see OfpaActor.h's own
//      header comment) is that touching one actor should only ever touch that actor's small file --
//      an embedded graph JSON (which can grow arbitrarily large as a graph gains nodes) would make
//      every PCG Volume actor file's size proportional to graph complexity instead of O(1).
// The one real cost -- a volume's file alone doesn't fully describe its own generated content,
// a graph file must also be resolved -- is accepted as out of scope for this phase; a later phase
// (6.3's runtime generator hook) is exactly where that resolution happens.
//
// Cell-grid convention reused (per this phase's task brief: "don't invent a new grid size"): this
// file reuses worldpartition::CellCoord/SpatialHashGrid (SpatialHashGrid.h), the SAME 3-axis,
// Grid2D-by-default cell coordinate and floor(worldPos / cellSize) conversion BakeDemoWorld.cpp's
// SpatialHashGrid already buckets every other actor into (kDemoWorldCellSize == 20.0f there, also
// cross-referenced by src/main.cpp and RuntimeCellManifestHeader::cellSize) -- NOT
// world::CellCoord (src/world/StreamingTypes.h), the separate RUNTIME cell type. This file lives
// under tools/WorldPartition/, this codebase's offline/editor toolset, which by established
// convention (see StreamingTypes.h's and BakeDemoWorld.cpp's own header comments: "tools/ never
// depends on src/, only the other direction is ever true elsewhere in this codebase") has ZERO
// compile-time dependency on anything under src/world/. ComputeOverlappingCells() below therefore
// takes `cellSize` as an explicit parameter -- exactly how SpatialHashGrid's own constructor and
// world::StreamingManager's own constructor already treat it (a caller-supplied value, never a
// hardcoded global) -- rather than baking in a copy of BakeDemoWorld.cpp's anonymous-namespace-local
// kDemoWorldCellSize, which is not reachable from a header anyway.

#include <cstdint>
#include <string>
#include <vector>

#include "OfpaActor.h"
#include "SpatialHashGrid.h"

namespace worldpartition {

    // ActorRecord::className value that marks a record as a PCG Volume -- see this file's own
    // header comment for the generic-record convention this follows.
    inline constexpr const char* kPcgVolumeClassName = "PcgVolume";

    // Typed authoring view of a PCG Volume's PCG-specific data. Deliberately minimal -- exactly the
    // 3 fields this phase's task brief calls "clearly needed": the region to generate within, which
    // graph asset drives that generation, and a seed so multiple volumes referencing the same graph
    // still produce distinct (but still fully deterministic, matching this codebase's established
    // "a demoscene demo is a fixed procedural performance" convention -- see BakeDemoWorld.cpp's own
    // header comment) content. Anything beyond this (priority/generation radius/regeneration
    // triggers/per-cell overrides) belongs to a later Phase 6 subtask that actually executes a
    // volume's graph, not to this phase's pure authoring concern.
    struct PcgVolumeDesc {
        AABB bounds;                // World-space region the referenced graph should generate content within.
        std::string graphAssetPath; // Relative path to a separate PcgGraph JSON file on disk (pcg::PcgGraph::SerializeToJson output) -- a REFERENCE, not the embedded JSON itself, see this file's own header comment for why.
        uint32_t seed = 0;          // Per-volume seed feeding the referenced graph's execution (a later phase's generator hook) -- lets many volumes share one graph asset while still producing distinct, deterministic content.
    };

    // Packs `desc` into a generic ActorRecord: className == kPcgVolumeClassName, an identity
    // transform (localBounds == worldBounds == desc.bounds exactly, no floating-point drift --
    // RecomputeWorldBounds() under an identity transform is an exact identity map), and
    // graphAssetPath/seed encoded as ordinary PropertyEntry items. The returned record is otherwise
    // a completely normal ActorRecord: callers pass it to WriteActorFile exactly like any other
    // actor (see OfpaActor.h).
    ActorRecord BuildPcgVolumeActorRecord(const Uuid& uuid, const PcgVolumeDesc& desc);

    // Inverse of BuildPcgVolumeActorRecord: returns false (leaving `outDesc` untouched) if
    // `record.className != kPcgVolumeClassName`, or if either expected property is missing or holds
    // the wrong PropertyValue alternative (a malformed/foreign-tool-authored record) -- never
    // crashes or asserts on a record this function does not recognize, matching this offline
    // toolset's established "one bad record must never block the rest" convention (see
    // RebuildSceneIndexFromActorFiles' own comment in SceneIndex.h). On success, `outDesc.bounds` is
    // taken from `record.worldBounds` (the same canonical, transform-resolved field
    // SceneIndexEntry::bounds and SpatialHashGrid::Build already key every other actor's spatial
    // queries off of).
    bool TryParsePcgVolumeDesc(const ActorRecord& record, PcgVolumeDesc& outDesc);

    // Every worldpartition::CellCoord `worldBounds` overlaps, in the SAME Grid2D (Y-collapsed)
    // convention SpatialHashGrid.h's own header comment documents and BakeDemoWorld.cpp's grid
    // already uses -- ties a PCG Volume to the cell(s) that must trigger its generation once a later
    // Phase 6 subtask actually wires that up. `cellSize` must match whatever cell size the rest of
    // the authored world/streaming setup uses (e.g. BakeDemoWorld.cpp's kDemoWorldCellSize == 20.0f
    // for this demo's own baked world) -- see this file's own header comment for why that value is
    // an explicit parameter here rather than a hardcoded constant. Degenerate (zero-size, or even
    // inverted) bounds still return exactly the 1 cell containing boundsMin/boundsMax's shared
    // point; a bounds spanning N x M cells returns all N*M coordinates, none deduplicated (there is
    // nothing to deduplicate -- each coordinate in the returned range is distinct by construction).
    std::vector<CellCoord> ComputeOverlappingCells(const AABB& worldBounds, float cellSize);

}
