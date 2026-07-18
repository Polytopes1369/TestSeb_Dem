#pragma once
// PCG framework roadmap, World Partition runtime generation, Phase 6.3 ("Runtime Generator Hook"):
// the world::IWorldCellLoader implementation that finally wires Phase 6.1's authored PCG Volumes
// (tools/WorldPartition/PcgVolumeActor.h) and Phase 6.2's pure per-cell generation logic
// (pcg::GeneratePcgContentForCell, src/pcg/PcgCellGenerator.h) into the LIVE world::StreamingManager
// streaming loop -- the piece Phase 6.2's own top-of-file comment explicitly deferred ("wiring THIS
// module into that live streaming loop is explicitly Phase 6.3's job, a later, separate subtask").
//
// --- Structural template: world::WorldCellStreamingLoader ---------------------------------------
// This class mirrors WorldCellStreamingLoader.h's own producer(worker-thread)/consumer(main-thread-
// pump) split almost exactly: LoadCellFullDetail()/LoadCellHlod()/UnloadCell() (called from a
// core::LoadingManager worker thread, per IWorldCellLoader's own threading contract, see
// StreamingTypes.h) only ever stage a small event into a mutex-guarded queue; Pump() (main-thread-
// only, called once per frame) drains it and performs the actual side effects. It deliberately does
// NOT reuse WorldCellStreamingLoader's own StreamingPlacementEvent/DrainEvents() shape, for two
// reasons:
//   1. PCG-driven cells and pre-baked-archetype cells are conceptually different content sources
//      (a small fixed pool of hand-authored shapes vs. arbitrarily many procedurally-generated
//      spawn requests) that may need to coexist -- see this class' own "coexistence" comment below.
//      Keeping them as two SEPARATE world::IWorldCellLoader implementations (rather than folding
//      PCG generation into WorldCellStreamingLoader itself) matches this phase's own task brief,
//      which explicitly prefers this over modifying WorldCellStreamingLoader.
//   2. Unlike WorldCellStreamingLoader's own DrainEvents() (which hands raw StreamingPlacementEvent
//      structs back to main.cpp because claiming a physical GPU streaming-unit slot needs
//      main.cpp's own cellToStreamingUnit/freeStreamingUnits bookkeeping), PCG spawn/despawn
//      integration is fully self-contained: pcg::PcgInstanceSpawnManager::SpawnInstances()/
//      DespawnInstances() are the entire main-thread-side contract. Pump() below therefore performs
//      those calls ITSELF (taking a pcg::PcgInstanceSpawnManager& at construction, borrowed non-
//      owning exactly like PcgInstanceSpawnManager's own borrowed renderer::PcgInstanceDrawPass&
//      reference -- see that class' own "Ownership model" comment for the precedent this follows),
//      rather than exposing raw events for a caller to interpret -- there is no cell-specific
//      GPU-slot-arbitration decision left for a caller to make.
//
// --- Coexistence with WorldCellStreamingLoader ---------------------------------------------------
// world::StreamingManager holds exactly ONE `IWorldCellLoader&` (see StreamingManager.h's own
// constructor) -- so a PcgCellLoader instance and a WorldCellStreamingLoader instance cannot BOTH
// drive the SAME StreamingManager simultaneously without a composite/dispatcher loader (out of scope
// for this phase; not built here). What this phase DOES deliver is a fully self-contained, correctly
// implemented world::IWorldCellLoader that a caller can wire into its OWN StreamingManager instance
// (a second, PCG-dedicated one, exactly the pattern this codebase already uses for "concurrent LWC
// origin instances", see main.cpp's own s_DebugShadowLwcOrigin precedent) whenever a future phase
// decides to actually run it against live camera-driven streaming. This phase's OWN validation
// (RunPcgCellLoaderSmokeTest, renderer::ClusterRenderPipeline.cpp) instead calls
// LoadCellFullDetail()/LoadCellHlod()/UnloadCell() directly, simulating exactly what a
// StreamingManager would trigger, without needing a live StreamingManager/CellManifest/camera at
// all -- see that method's own header comment for why this is still genuine, real validation.
//
// --- Debug-only, whole file (see the #ifndef NDEBUG guard below) --------------------------------
// Two independent reasons converge on the same conclusion:
//   1. This class' constructor builds its volume index via PcgVolumeCellIndex.h's
//      ScanPcgVolumeActorFiles()/BuildPcgVolumeCellIndex() -- itself whole-file Debug-only for the
//      tools/WorldPartition/ Release-link-boundary reason documented in THAT header's own comment.
//      A PcgCellLoader that could never actually populate its own index in a Release build would be
//      a hollow, misleading Release-shipping type.
//   2. Nothing in this codebase yet AUTHORS a real PcgVolume actor file for a Release-relevant
//      scenario either (tools/WorldPartition/BakeDemoWorld.cpp only ever writes "Rock"/"Bush"/
//      "Tree"/"Debris" actors -- see renderer::debug::PcgVolumeInspector.h's own "Demo dataset
//      fallback" comment, the only other place in this codebase to have hit this exact same gap).
// A future phase that decides to promote real, disk-authored PCG Volume content (and therefore the
// tools/WorldPartition/PcgVolumeActor.cpp/OfpaActor.cpp linkage it requires) to Release is where
// that larger, precedent-changing call belongs -- explicitly out of scope for "Phase 6.3: Runtime
// Generator Hook" (see this phase's own task brief, "Stay scoped to 6.3").
#ifndef NDEBUG

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "PcgVolumeCellIndex.h"
#include "StreamingTypes.h"
#include "pcg/PcgCellGenerator.h" // pcg::PcgCellGenerationInput/Result, pcg::GeneratePcgContentForCell
#include "pcg/PcgInstanceSpawnManager.h" // pcg::PcgInstanceSpawnManager, pcg::PcgSpawnRequest

namespace world {

    // One staged cross-thread event: either "this cell generated real content, spawn it" (isLoad ==
    // true, spawnRequests populated) or "this cell is going away, despawn whatever it previously
    // spawned" (isLoad == false, spawnRequests left empty). A single event type (rather than two
    // separate queues) keeps Pump()'s drain loop processing events in the exact order they were
    // staged -- relevant if a worker thread races a load and an unload for the same cell close
    // together (StreamingManager's own documented "Unload-then-Load convergence" ordering, see
    // StreamingTypes.h's own CellStreamingState comment, means this SHOULD never actually happen out
    // of order, but Pump() below does not rely on that guarantee to stay correct either way).
    struct PcgCellLoadEvent {
        CellCoord coord;
        bool isLoad = false;
        std::vector<pcg::PcgSpawnRequest> spawnRequests; // Only meaningful when isLoad == true.
    };

    class PcgCellLoader : public IWorldCellLoader {
    public:
        // `actorsRootDir` is scanned ONCE, here, at construction (main thread, before any streaming
        // starts -- per this phase's own task brief) via PcgVolumeCellIndex.h's
        // ScanPcgVolumeActorFiles()/BuildPcgVolumeCellIndex(), so LoadCellFullDetail()/LoadCellHlod()
        // (worker-thread, potentially high-frequency as a camera moves) never touch disk themselves
        // -- an O(1) unordered_map lookup against the already-built m_VolumeIndex instead. `cellSize`
        // must match whatever cell size the runtime streaming grid actually uses (see
        // PcgVolumeCellIndex.h's own BuildPcgVolumeCellIndex comment) -- an explicit, caller-supplied
        // value, never a hardcoded literal (matches worldpartition::ComputeOverlappingCells' own
        // established convention). `spawnManager` is borrowed, never owned (must outlive this
        // PcgCellLoader instance) -- see this file's own top-of-file "Structural template" comment
        // for why Pump() itself, not a caller, drives it.
        PcgCellLoader(const std::filesystem::path& actorsRootDir, float cellSize, pcg::PcgInstanceSpawnManager& spawnManager);

        PcgCellLoader(const PcgCellLoader&) = delete;
        PcgCellLoader& operator=(const PcgCellLoader&) = delete;

        // --- IWorldCellLoader: called from a core::LoadingManager worker thread, never the main
        // thread, and potentially concurrently for different cells (see the interface's own
        // threading contract, StreamingTypes.h). m_VolumeIndex is built once at construction and
        // never mutated afterward, so concurrent lookups against it need no locking of their own --
        // only m_PendingEvents (appended to here) is guarded by m_EventsMutex. ---

        // Looks up `coord` in m_VolumeIndex; if any volumes overlap it, calls
        // pcg::GeneratePcgContentForCell() for each (worker-thread-safe, no Vulkan -- see that
        // function's own comment) and stages the COMBINED PcgSpawnRequest list (across every
        // overlapping volume) as one load event tagged with `coord`. A cell with no overlapping
        // volumes, or whose generation produced zero spawn requests, stages nothing -- exactly
        // WorldCellStreamingLoader::StageActivate's own "no authored content -> no-op" convention.
        void LoadCellFullDetail(const CellCoord& coord) override;

        // PCG has no HLOD tier of its own yet (Phase 4.3's HLOD integration operates on the OFFLINE
        // bake path, tools/WorldPartition/HlodPipeline.h, not this live streaming path -- see this
        // phase's own task brief). Rather than faking an HLOD behavior that does not exist (e.g.
        // silently generating the SAME full-detail content and mislabeling it "HLOD"), this is a
        // documented no-op: a cell that only ever reaches the HLOD band (inside a StreamingSource's
        // hlodLoadRadius but outside its detailLoadRadius, see StreamingTypes.h's own
        // StreamingSource comment) never gets any PCG content until it is promoted to FullDetail. A
        // Debug-only diagnostic log fires (throttled to once per distinct coord, so a camera
        // orbiting the HLOD band's own boundary does not spam the log) documenting exactly this.
        void LoadCellHlod(const CellCoord& coord) override;

        // Stages an unload event for `coord` IF (and only if) m_VolumeIndex has any entry for it --
        // a cell that never had any overlapping PCG Volume could not possibly have acquired any
        // instance slots, so staging an event for it would be pure queue churn Pump() would just
        // no-op on anyway (see Pump()'s own comment). Never touches m_CellToAcquiredSlots directly
        // (main-thread-only state) -- that lookup/release happens in Pump().
        void UnloadCell(const CellCoord& coord) override;

        // Main-thread-only equivalent of WorldCellStreamingLoader::DrainEvents(), but -- unlike that
        // method -- this ALSO performs the actual pcg::PcgInstanceSpawnManager::SpawnInstances()/
        // DespawnInstances() calls itself rather than handing raw events back to a caller (see this
        // file's own top-of-file "Structural template" comment, point 2, for why). For each staged
        // load event: calls SpawnInstances(event.spawnRequests) and records the returned slot list
        // against event.coord in m_CellToAcquiredSlots (appended, not overwritten -- see .cpp for
        // why). For each staged unload event: looks up event.coord in m_CellToAcquiredSlots; if
        // found, calls DespawnInstances() on that cell's full acquired-slot list and erases the
        // entry. Does NOT call renderer::PcgInstanceDrawPass::UploadInstances() -- exactly like
        // PcgInstanceSpawnManager::SpawnInstances()/DespawnInstances() themselves, that remains the
        // caller's own per-frame (or per-rebuild) responsibility (see PcgInstanceSpawnManager.h's
        // own "What this class does NOT do" comment). Must be called once per frame from the main
        // thread, after StreamingManager::Update() and core::LoadingManager::PumpCompletions() have
        // had a chance to actually invoke LoadCellFullDetail()/LoadCellHlod()/UnloadCell() above --
        // exactly WorldCellStreamingLoader::DrainEvents()'s own call-site convention (see main.cpp).
        void Pump();

        // --- Read-only accessors, mainly for RunPcgCellLoaderSmokeTest's/PcgCellLoaderTests' own
        // verification (mirrors renderer::debug::PcgVolumeInspector::GetVolumeCount()'s own "expose
        // a live count for verification" convention). ---
        size_t GetVolumeCount() const { return m_VolumeCount; }
        size_t GetIndexedCellCount() const { return m_VolumeIndex.size(); }
        size_t GetLoadedCellCount() const { return m_CellToAcquiredSlots.size(); } // Main-thread-only -- only meaningful after at least one Pump() call.

    private:
        // Shared staging helper for LoadCellFullDetail(): looks up `coord`, runs
        // pcg::GeneratePcgContentForCell() for every overlapping volume, and stages a combined load
        // event if the combined result is non-empty. LoadCellHlod() deliberately does NOT call this
        // (see that method's own comment) -- kept as its own private helper rather than a shared
        // "activate" entry point specifically so that distinction cannot silently blur in a future
        // edit.
        void StageFullDetailGeneration(const CellCoord& coord);

        float m_CellSize;
        pcg::PcgInstanceSpawnManager& m_SpawnManager;

        size_t m_VolumeCount = 0; // Total authored volumes found at construction (before cell-bucketing) -- see GetVolumeCount().
        PcgVolumeCellIndex m_VolumeIndex; // Built once at construction, read-only afterward -- see this class' own constructor comment.

        std::mutex m_EventsMutex;
        std::vector<PcgCellLoadEvent> m_PendingEvents; // Guarded by m_EventsMutex.

        // Main-thread-only (touched only from Pump()): which instance slots each currently-loaded
        // cell's PCG generation acquired, so a later UnloadCell()'s own Pump()-side DespawnInstances()
        // call knows exactly what to release. Never accessed from a worker thread -- worker threads
        // only ever stage events (m_PendingEvents above), never touch this map directly.
        std::unordered_map<CellCoord, std::vector<uint32_t>, CellCoordHash> m_CellToAcquiredSlots;
    };

}
#endif
