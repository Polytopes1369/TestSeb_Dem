#pragma once

// PCG framework roadmap, World Partition runtime generation, Phase 6.2 ("Partitioned PCG Cell
// Execution"): the pure, engine-integration-free evaluation logic that turns "a PCG Volume
// (Phase 6.1, tools/WorldPartition/PcgVolumeActor.h) overlaps this World Partition cell" into "here
// are the concrete PcgSpawnRequest instances that cell should render". This file deliberately does
// NOT touch threading, the LoadingManager worker-thread contract, or world::IWorldCellLoader
// (src/world/StreamingTypes.h) -- wiring THIS module into that live streaming loop is explicitly
// Phase 6.3's job, a later, separate subtask. GeneratePcgContentForCell() below is a plain,
// synchronous, side-effect-free (beyond Debug-only diagnostic logging) function: safe to call from
// ANY thread (a future Phase 6.3 caller running it on a LoadingManager worker, or this phase's own
// CTest, or a hypothetical future offline bake tool), with no shared mutable state of its own.
//
// --- Why this lives under src/pcg/, not tools/WorldPartition/ -----------------------------------
// This is CORE, ALWAYS-ON runtime generation logic -- the whole point of "100% procedural GPU
// driven" World Partition content (see this project's CLAUDE.md) is that it must exist and run in a
// shipping RELEASE build, not just an editor/Debug tool. It is therefore placed alongside the rest
// of the PCG graph engine (src/pcg/), which already ships in every configuration, rather than under
// tools/WorldPartition/ (the OFPA/HLOD OFFLINE authoring toolset, which WorldPartitionTypes.h's own
// header comment documents as "never linked into the shipping demo executable" -- true for that
// toolset's own COMPILED .cpp logic, see this file's own "Two-namespace CellCoord" section below for
// exactly how this file avoids inheriting that restriction while still reusing Phase 6.1's types).
//
// --- Two-namespace CellCoord situation (per this phase's own task brief) ------------------------
// Two separate, independently-motivated CellCoord types exist in this codebase:
//   - worldpartition::CellCoord (tools/WorldPartition/SpatialHashGrid.h): the OFFLINE authoring
//     grid's 3-axis (x/y/z, y always 0 in the default Grid2D mode) coordinate, produced by Phase
//     6.1's worldpartition::ComputeOverlappingCells(). This is the type a PCG Volume's own
//     "which cells do I overlap" query already speaks.
//   - world::CellCoord (src/world/StreamingTypes.h): the RUNTIME streaming grid's own 2-axis (x/z)
//     coordinate, consumed by world::IWorldCellLoader -- deliberately independent of the offline
//     tool's type (see that header's own comment: "runtime code under src/ must never depend on
//     [tools/WorldPartition/]").
// This file's own PcgCellGenerationInput/Result deliberately use worldpartition::CellCoord (NOT
// world::CellCoord): GeneratePcgContentForCell's whole job is "resolve a PCG Volume, which is
// authored and queried in the offline tool's own coordinate space, against ONE of the cells
// worldpartition::ComputeOverlappingCells() already said it overlaps" -- staying in that same
// coordinate space end-to-end avoids inventing a THIRD, redundant cell-coordinate type here just to
// re-describe the same 20m-grid concept Phase 6.1 already established. A future Phase 6.3 caller
// (which DOES need to bridge into world::IWorldCellLoader's world::CellCoord to actually drive
// StreamingManager) is where that coordinate-space conversion belongs -- it is a one-line, lossy-only-
// if-cell-sizes-differ (x/z carry over directly, y is simply dropped) conversion, deliberately left
// to that later phase rather than speculatively added here.
//
// This DOES mean this file (src/pcg/, a shipping-in-every-config module) includes
// "WorldPartition/PcgVolumeActor.h" (tools/WorldPartition/) -- a real, one-directional compile-time
// dependency from src/ onto tools/. This is NOT a violation of "tools/ never depends on src/, only
// the other direction is ever true" (see PcgVolumeActor.h's own header comment: that rule is about
// which direction crosses the boundary, and src/ -> tools/ is exactly the permitted direction).
// What IS preserved is WorldPartitionTypes.h's narrower "never linked into the shipping executable"
// promise for the offline toolset's own COMPILED BEHAVIOR: this file only ever reaches
// worldpartition::PcgVolumeDesc / worldpartition::CellCoord / worldpartition::AABB, three plain,
// header-only POD types with no out-of-line method bodies this translation unit actually calls (see
// PcgCellGenerator.cpp's own "CellWorldBounds" comment for the one place this is worth spelling out
// explicitly: it deliberately RE-DERIVES SpatialHashGrid::CellBounds's own documented formula rather
// than calling that compiled method, specifically so tools/WorldPartition/SpatialHashGrid.cpp's own
// .cpp never has to be linked into the shipping DemoSceneVK target either). The only actual build-
// system change this required was adding the `tools` include directory to the main target
// UNCONDITIONALLY (see the top-level CMakeLists.txt's own comment at that change) -- an include path
// costs zero bytes in the shipping binary; only compiled-and-linked object code does.
//
// --- What "generation" means here (pure CPU evaluation, no GPU/render-pass involvement) ---------
// GeneratePcgContentForCell(), for each worldpartition::PcgVolumeDesc overlapping this cell:
//   1. Loads and parses `graphAssetPath` from disk (pcg::PcgGraph::DeserializeFromJson, Phase 5.1).
//      A missing or unparseable file is NEVER a fatal error for the whole cell -- it is logged
//      (Debug-only LOG_WARNING) and that ONE volume is skipped, matching this codebase's established
//      "one bad record must never block the rest" convention (see e.g.
//      tools/WorldPartition/SceneIndex.h's RebuildSceneIndexFromActorFiles comment, or this same
//      file's own PcgVolumeDesc::TryParsePcgVolumeDesc comment).
//   2. Evaluates the parsed graph (pcg::PcgGraphEvaluator, Phase 5.2) against a registry populated
//      from Phase 5.4's pcg::PopulateNativeNodeTypePlugins() -- the SAME native node-type set (today:
//      only "pcg.spawner.weighted_mesh", see PcgMeshSpawner.cpp's own registration comment; Phase 2's
//      samplers/Phase 3's filters are not yet wired into the graph-node registry, only usable as
//      plain C++ functions -- a real content author's graph can therefore only actually reach this
//      generator's spawn-request output through either a graph ending directly in
//      "pcg.spawner.weighted_mesh", or (this phase's own trivial fallback, see point 4 below) a graph
//      that stops at a raw Points output) any future caller (a real editor session, a bake tool)
//      would also see -- built FRESH per call (see PcgCellGenerator.cpp's own comment on why this is
//      deliberately NOT cached/hoisted out of the per-volume loop in this phase: that is explicitly
//      Phase 6.4's "Generation Caching" concern, out of scope here).
//   3. Identifies the graph's TERMINAL output: a node that is never the SOURCE of any link in the
//      graph (i.e. nothing downstream consumes its output) -- this is the same "terminal/leaf node"
//      concept PcgGraphEvaluator.h's own EvalResult::nodeOutputs comment already names ("exposed (not
//      just the graph's terminal/leaf nodes)..."), just made concrete here as an actual search,
//      since no existing code in this codebase needed to programmatically FIND one before this phase
//      (every existing test/tool that reads a specific node's output already knows that node's id by
//      construction). Preference order among a graph's terminal node(s), in `graph.Nodes()`'s own
//      insertion-order (deterministic, since PcgGraph::AddNode ids are monotonically increasing and
//      never reused):
//        a) A terminal node with a PcgPinDataType::SpawnRequests output pin -- the REAL, expected
//           authoring pattern (a well-formed PCG Volume graph ends in "pcg.spawner.weighted_mesh",
//           see PcgMeshSpawner.cpp's own registration comment): that node's output is used AS-IS
//           (after cell-bounds clipping, point 4), a straight pass-through.
//        b) Otherwise, a terminal node with a PcgPinDataType::Points output pin -- a graph that stops
//           short of spawning gets a TRIVIAL 1:1 fallback (point 4) rather than producing nothing.
//        c) Neither found: this volume contributes zero spawn requests for this cell (logged, not an
//           error -- e.g. a graph authored purely for its side-visible-in-the-node-inspector
//           intermediate outputs, or a work-in-progress asset).
//   4. CLIPS the terminal output to THIS cell's own world-space bounds (X/Z only, per this
//      codebase's established Grid2D convention -- see CellWorldBounds's own comment in the .cpp for
//      exactly why Y stays unconstrained, matching worldpartition::SpatialHashGrid::CellBounds's own
//      documented Grid2D behavior) -- THE step that makes generation genuinely PARTITIONED rather
//      than duplicating a volume's full content into every cell it spans:
//        a) SpawnRequests case: filtered directly by each request's own `position` against the
//           cell's PcgVolumeData (pcg::PcgVolumeData::ContainsWorldPoint) -- pcg::IntersectWithVolume
//           (Phase 3.3) only operates on std::vector<PcgPoint>, not PcgSpawnRequest, so this file
//           reimplements the identical "keep iff ContainsWorldPoint" test directly over
//           PcgSpawnRequest's own position field.
//        b) Points case: pcg::IntersectWithVolume(points, cellVolume) (Phase 3.3, reused verbatim --
//           the cell's own AABB IS treated as a PcgVolumeData, exactly per this phase's task brief),
//           then the trivial fallback below.
//   5. Trivial Points fallback (case 3b/4b only): since worldpartition::PcgVolumeDesc carries no
//      weighted-mesh palette of its own (only bounds/graphAssetPath/seed -- a weighted-mesh list only
//      exists as a "pcg.spawner.weighted_mesh" NODE's own params, per PcgMeshSpawner.h's own design),
//      each surviving point becomes exactly one PcgSpawnRequest via pcg::SpawnFromPoints (Phase 4.1)
//      with a single-entry {meshID=0, materialID=0, weight=1.0} palette -- reusing the SAME tested
//      density-cull + per-point-seeded-selection code path a real spawner node would run, rather than
//      hand-rolling a parallel "just copy the transform" loop. A future phase wanting a richer
//      fallback palette (e.g. read from a DIFFERENT PcgVolumeDesc field) can extend this without
//      touching the SpawnRequests pass-through path at all.
//
// --- Determinism -----------------------------------------------------------------------------------
// Every volume's contribution to this cell is additionally salted by a per-(volume,cell) evaluation
// seed -- PcgHashCombine(volumeDesc.seed, cellCoord.x/y/z folded in) -- fed into the Points-fallback's
// SpawnFromPoints() call (point 5 above). This guarantees re-running GeneratePcgContentForCell for the
// exact same (volumeDesc, cellCoord) pair always reproduces byte-identical output (this codebase's
// hard "the show must play back identically every run" requirement, see PcgSeededRandom.h's own
// top-of-file comment), AND that two DIFFERENT cells a volume spans never share a copy-pasted-
// identical PcgSeededRandom stream even in a degenerate case where their surviving point sets
// happened to be the same size. Note the SpawnRequests pass-through path (point 4a) does NOT re-seed
// anything -- a "pcg.spawner.weighted_mesh" node's own mesh selection already ran, using ITS OWN
// baked-at-authoring-time params seed, once, during graph evaluation (point 2); genuinely different
// content per cell for THAT path comes entirely from the geometric clip (point 4a) -- two different
// cells clip the SAME evaluated point cloud down to two DISJOINT surviving subsets by construction,
// which is already "different, not duplicated" without needing (and this phase does not attempt) any
// per-cell re-evaluation of the graph's own internal node params.

#include "pcg/PcgMeshSpawner.h" // pcg::PcgSpawnRequest
#include "WorldPartition/PcgVolumeActor.h" // worldpartition::PcgVolumeDesc, worldpartition::CellCoord (via SpatialHashGrid.h), worldpartition::AABB (via WorldPartitionTypes.h)

#include <string>
#include <vector>

namespace pcg {

    // Everything GeneratePcgContentForCell needs to generate ONE cell's worth of PCG content: which
    // cell (in the SAME worldpartition::CellCoord space Phase 6.1's ComputeOverlappingCells() already
    // produces), which PCG Volumes a caller has already determined overlap it (see this file's own
    // top-of-file comment -- a Phase 6.3 caller is expected to have already called
    // worldpartition::ComputeOverlappingCells(volumeDesc.bounds, cellSize) per authored volume and
    // filtered down to the ones whose result contains THIS cellCoord; this struct does not repeat
    // that overlap test), and the grid's own cellSize (must match whatever cellSize the caller used
    // to compute `overlappingVolumes` in the first place -- see worldpartition::PcgVolumeActor.h's
    // own header comment for why this is always an explicit, caller-supplied value in this codebase,
    // never a hardcoded global).
    struct PcgCellGenerationInput {
        worldpartition::CellCoord cellCoord;
        std::vector<worldpartition::PcgVolumeDesc> overlappingVolumes;
        float cellSize = 0.0f;
    };

    // The concrete, ready-to-spawn result of generating one cell's worth of PCG content. `success`/
    // `errorMessage` are reserved for a STRUCTURAL problem with the request itself (e.g. a
    // non-positive cellSize) -- a per-volume problem (missing/unparseable graph asset, an evaluation
    // error, a graph with no recognizable terminal output) NEVER sets success to false; it is logged
    // and that one volume simply contributes zero spawn requests, exactly per this file's own
    // top-of-file comment. A caller should therefore treat `success == false` as "this cell's
    // manifest could not be generated at all, do not retry with the same input", and `success == true`
    // with a shorter-than-expected `spawnRequests` as "generation ran, some volumes may have been
    // individually skipped -- see the Debug log for which".
    struct PcgCellGenerationResult {
        worldpartition::CellCoord cellCoord;
        std::vector<PcgSpawnRequest> spawnRequests;
        bool success = true;
        std::string errorMessage;
    };

    // Pure, synchronous, thread-safe (no shared mutable state) evaluation of every
    // `input.overlappingVolumes` entry against `input.cellCoord`, clipped to that cell's own bounds --
    // see this file's own top-of-file comment for the full per-volume algorithm. Never throws.
    PcgCellGenerationResult GeneratePcgContentForCell(const PcgCellGenerationInput& input);

}
