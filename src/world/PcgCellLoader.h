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
// --- Phase 6.4 ("Generation Caching"): per-cell generation-result cache -------------------------
// The gap this phase closes: world::StreamingManager's own streaming loop can (and, for a camera
// oscillating near a cell boundary, WILL) call LoadCellFullDetail() for the SAME coord many times
// across a session, each preceded by an UnloadCell() for that same coord as the cell briefly drops
// out of range. Before this phase, EVERY one of those reloads re-ran the full
// pcg::GeneratePcgContentForCell() evaluation (disk read + JSON parse + graph evaluation per
// overlapping volume) from scratch -- pure waste, since that function is a documented pure,
// deterministic function of its input (see PcgCellGenerator.h's own "Determinism" comment: the
// exact same (volumeDesc, cellCoord) pair always reproduces byte-identical output).
//
// Cache key: `CellCoord` ALONE (m_GenerationResultCache below), not a hash of coord+volumes+seed.
// This is deliberate, not an oversight: m_VolumeIndex is built ONCE at construction and NEVER
// mutated afterward (see this class' own constructor comment), and m_CellSize is likewise fixed for
// this instance's entire lifetime. So, for a GIVEN PcgCellLoader instance, `coord` alone already
// fully determines the PcgCellGenerationInput (both overlappingVolumes AND cellSize) that
// StageFullDetailGeneration() would otherwise rebuild -- which, since GeneratePcgContentForCell is
// pure, fully determines its output too. Hashing volume ids/seeds into the key as well would just
// re-derive information m_VolumeIndex already pins down per-coord; it would never change the
// key-to-result mapping for THIS instance.
//
// Critically, UnloadCell() does NOT evict this cache -- only m_CellToAcquiredSlots (main-thread,
// Pump()-side bookkeeping) is touched on unload. A generation result, once computed, survives for
// this loader instance's entire session lifetime regardless of how many times its cell gets
// unloaded and reloaded -- this is exactly what makes the "camera oscillates across a cell
// boundary" scenario above a guaranteed cache HIT on every reload after the first.
//
// Eviction policy: none -- an unbounded (for this instance's lifetime) unordered_map. Two
// independent reasons converge on this being safe rather than a leak risk:
//   1. This cache can only ever grow via StageFullDetailGeneration(), which returns BEFORE ever
//      touching the cache for any coord absent from m_VolumeIndex (see that method's own early-out).
//      Its size is therefore hard-bounded by m_VolumeIndex.size() (== GetIndexedCellCount()) --
//      an AUTHORING-TIME quantity fixed at construction, which does NOT grow with how far or how
//      long a camera travels/streams. For this codebase's own current authored content
//      (tools/WorldPartition/BakeDemoWorld.cpp's kGridRadiusCells == 3, a 7x7 == 49-cell patch),
//      that is a firm upper bound of 49 entries, each holding at most a handful of small
//      pcg::PcgSpawnRequest structs -- kilobytes, not a concern.
//   2. Precedent: world::StreamingManager::m_Cells (StreamingManager.h) is ALREADY an unbounded,
//      never-erased unordered_map<CellCoord, ...> in this exact codebase, and that one has a WEAKER
//      bound (it grows with every distinct cell a camera ever visits, not just ones with authored
//      PCG content) -- this cache's own bound is strictly tighter.
// This is a Debug-only dev/streaming aid (see this class' own Debug-only rationale below), not a
// shipping persistence system -- an LRU or explicit size cap would be solving a problem this class
// provably cannot have, so none is built.
//
// Thread-safety: m_GenerationResultCache/m_CacheHitCount/m_CacheMissCount are guarded by their OWN
// mutex (m_CacheMutex), deliberately separate from m_EventsMutex -- an unrelated concern (staging a
// cross-thread load/unload event vs. memoizing a pure function's result), kept as two independent
// locks so a worker thread's cache lookup for one cell never contends with another worker thread's
// event-staging for a different cell. A cache MISS races a fresh pcg::GeneratePcgContentForCell()
// call OUTSIDE the lock (never hold m_CacheMutex across that potentially-slow call), then inserts
// via unordered_map::emplace (insert-only-if-absent) -- if two worker threads race a miss for the
// SAME coord, both compute and offer up a byte-identical result (determinism, again), so whichever
// one's emplace() actually wins the slot is immaterial; the loser's own local copy is simply
// discarded after use.
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
#include <optional>
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

        // --- Phase 6.4 ("Generation Caching") accessors -- worker-thread-safe (lock m_CacheMutex),
        // unlike GetLoadedCellCount() above, since m_GenerationResultCache/m_CacheHitCount/
        // m_CacheMissCount ARE touched from worker threads (see this class' own top-of-file "Phase
        // 6.4" comment). Mainly for RunPcgCellLoaderSmokeTest's own cache-hit verification: a test
        // can snapshot GetCacheHitCount()/GetCacheMissCount() before and after a repeat
        // LoadCellFullDetail() call for the same coord to prove a reload actually skipped
        // pcg::GeneratePcgContentForCell() rather than merely re-deriving the same deterministic
        // answer the slow way. ---
        size_t GetCacheHitCount() const { std::lock_guard<std::mutex> lock(m_CacheMutex); return m_CacheHitCount; }
        size_t GetCacheMissCount() const { std::lock_guard<std::mutex> lock(m_CacheMutex); return m_CacheMissCount; }
        size_t GetCachedCellCount() const { std::lock_guard<std::mutex> lock(m_CacheMutex); return m_GenerationResultCache.size(); }

        // Phase 6.5 ("Bake-vs-Runtime Determinism Validation") accessor -- worker-thread-safe (locks
        // m_CacheMutex), same convention as the 3 accessors just above. Returns a COPY of the cached
        // pcg::PcgCellGenerationResult for `coord` (std::nullopt if nothing is cached for it yet --
        // e.g. LoadCellFullDetail() was never called for this coord, or that coord has no overlapping
        // PCG volume at all, so StageFullDetailGeneration() never reached its cache-populating path).
        // This is the ONE thing GetCacheHitCount()/GetCacheMissCount()/GetLoadedCellCount() above
        // cannot expose: they only ever prove a hit/miss/load HAPPENED, never what the live runtime
        // path's own generated content actually WAS. RunPcgCellLoaderSmokeTest's own Phase 6.5 step
        // uses this to compare the exact spawn-request list the LIVE world::PcgCellLoader path
        // produced/cached against a separately, directly-computed pcg::GeneratePcgContentForCell()
        // call simulating a hypothetical offline bake tool -- proving the live path never silently
        // diverges (wrong order, corrupted transform, dropped/duplicated request) from what generation
        // itself actually produced, a property no existing count-only accessor could ever catch. A
        // copy (not a reference) is returned for the exact same reason the cache-hit path in
        // StageFullDetailGeneration() (.cpp) copies out while still holding the lock: cheap (a
        // handful of small PcgSpawnRequest structs at most, see this cache's own bound comment), and
        // avoids handing out a reference a concurrent worker thread's insert for a DIFFERENT coord
        // could invalidate via unordered_map rehashing.
        std::optional<pcg::PcgCellGenerationResult> GetCachedResultForTest(const CellCoord& coord) const {
            std::lock_guard<std::mutex> lock(m_CacheMutex);
            auto it = m_GenerationResultCache.find(coord);
            if (it == m_GenerationResultCache.end()) return std::nullopt;
            return it->second;
        }

    private:
        // Shared staging helper for LoadCellFullDetail(): looks up `coord`, runs
        // pcg::GeneratePcgContentForCell() for every overlapping volume, and stages a combined load
        // event if the combined result is non-empty. LoadCellHlod() deliberately does NOT call this
        // (see that method's own comment) -- kept as its own private helper rather than a shared
        // "activate" entry point specifically so that distinction cannot silently blur in a future
        // edit. Since Phase 6.4, this ALSO consults/populates m_GenerationResultCache before/after
        // calling pcg::GeneratePcgContentForCell -- see this class' own top-of-file "Phase 6.4"
        // comment for the full caching rationale, and the .cpp for the exact lock-scoping.
        void StageFullDetailGeneration(const CellCoord& coord);

        float m_CellSize;
        pcg::PcgInstanceSpawnManager& m_SpawnManager;

        size_t m_VolumeCount = 0; // Total authored volumes found at construction (before cell-bucketing) -- see GetVolumeCount().
        PcgVolumeCellIndex m_VolumeIndex; // Built once at construction, read-only afterward -- see this class' own constructor comment.

        // Phase 6.4 ("Generation Caching"): coord -> the pcg::PcgCellGenerationResult
        // pcg::GeneratePcgContentForCell() previously produced for that coord, memoized for this
        // loader instance's entire lifetime (NEVER evicted by UnloadCell() -- see this class' own
        // top-of-file "Phase 6.4" comment for the full key/eviction-policy/thread-safety rationale).
        // `mutable` on the mutex only (not the map/counters) -- the map/counters are only ever
        // mutated from StageFullDetailGeneration() (a non-const method); the mutex itself must be
        // lockable from the const accessors above too, hence `mutable`.
        mutable std::mutex m_CacheMutex;
        std::unordered_map<CellCoord, pcg::PcgCellGenerationResult, CellCoordHash> m_GenerationResultCache; // Guarded by m_CacheMutex.
        size_t m_CacheHitCount = 0;  // Guarded by m_CacheMutex.
        size_t m_CacheMissCount = 0; // Guarded by m_CacheMutex.

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
