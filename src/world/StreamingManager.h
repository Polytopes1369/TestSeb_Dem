#pragma once
// Real-time CPU-side spatial streaming evaluator for the World Partition runtime grid: each frame,
// decides which ground-plane cells need their full-detail actors resident, which need only their
// HLOD proxy, and which should be torn down -- then dispatches the resulting load/unload work onto
// core::LoadingManager's worker pool through a priority queue ordered by distance to the nearest
// StreamingSource, so a burst of newly-revealed cells (e.g. a fast-traveling camera) never floods
// the I/O/worker budget: only maxConcurrentLoads_ requests are ever in flight at once, closest
// (most urgent) first.
//
// --- Threading model ---
// UpdateStreamingSources() and Update() are main-thread-only (they mutate the priority queue and
// per-cell `desiredRepresentation`, which are not internally synchronized against each other --
// only ONE thread is ever expected to drive streaming decisions). GetCellState() /
// GetCellRepresentation() are safe to call from ANY thread at ANY time, including concurrently
// with Update() and with the worker threads that complete loads -- every per-cell field they read
// is either std::atomic (state, currentRepresentation) or guarded by a std::shared_mutex used in
// reader-many/writer-rare mode (the cell registry itself, which only ever grows -- see
// GetOrCreateCellRecord's comment on why erasing entries is never needed).

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "StreamingTypes.h"
#include "core/LoadingManager.h"
#include "core/maths/Maths.h"

namespace world {

    class StreamingManager {
    public:
        // `cellSize` is the runtime grid's fixed world-space cell size (ground-plane, matching
        // CellCoord's 2-axis convention). `loader` and `workerPool` are owned by the caller and
        // must outlive this StreamingManager. `maxConcurrentLoads` bounds how many
        // load/unload requests may be in flight on the worker pool at once -- the actual I/O/CPU
        // load-balancing knob (set it well below workerPool's own thread count to leave headroom
        // for other background work sharing that same pool, e.g. procedural generation bakes).
        StreamingManager(float cellSize, IWorldCellLoader& loader, core::LoadingManager& workerPool, uint32_t maxConcurrentLoads = 4);

        StreamingManager(const StreamingManager&) = delete;
        StreamingManager& operator=(const StreamingManager&) = delete;

        // Main-thread-only. Recomputes every tracked cell's desired representation from `sources`
        // (unioned across all of them -- if ANY source wants a cell at FullDetail, it is desired
        // at FullDetail even if another, farther source would only need HLOD for it) and enqueues
        // load/unload requests for cells whose desired representation differs from what is
        // currently loaded or already in flight. Never blocks on I/O -- actual loading happens on
        // worker threads once Update() dispatches a queued request.
        void UpdateStreamingSources(const std::vector<StreamingSource>& sources);

        // Main-thread-only. Dispatches queued requests (highest priority / nearest-to-a-source
        // first) onto the worker pool until either the queue is empty or maxConcurrentLoads_
        // in-flight requests are reached. Call once per frame, after UpdateStreamingSources().
        void Update();

        // Thread-safe: returns CellStreamingState::Unloaded for a cell this manager has never
        // tracked (equivalent to its real, un-materialized state -- there is no "unknown" state).
        CellStreamingState GetCellState(const CellCoord& coord) const;
        CellRepresentation GetCellRepresentation(const CellCoord& coord) const;

        size_t GetTrackedCellCount() const;
        uint32_t GetInFlightCount() const { return m_InFlightCount.load(std::memory_order_acquire); }
        size_t GetPendingQueueLength() const;

        CellCoord WorldToCell(const maths::vec3& worldPos) const;
        maths::vec3 CellCenter(const CellCoord& coord) const;

    private:
        struct CellRecord {
            std::atomic<CellStreamingState> state{ CellStreamingState::Unloaded };
            std::atomic<CellRepresentation> currentRepresentation{ CellRepresentation::None };
            CellRepresentation desiredRepresentation = CellRepresentation::None; // Main-thread-only: only ever read/written from UpdateStreamingSources.
        };

        struct PendingRequest {
            CellCoord coord;
            CellRepresentation targetRepresentation; // CellRepresentation::None means "unload".
            float priorityDistance; // Lower = more urgent. Unload requests always use 0.0f (see .cpp): freeing a cell must never be starved behind farther-away load requests.
            uint64_t sequence;      // Tie-breaker for equal priorityDistance, so equally-urgent requests dispatch in the order they were queued rather than in std::priority_queue's unspecified tie order.
        };

        struct PendingRequestGreater {
            bool operator()(const PendingRequest& a, const PendingRequest& b) const {
                if (a.priorityDistance != b.priorityDistance) return a.priorityDistance > b.priorityDistance;
                return a.sequence > b.sequence;
            }
        };

        // Returns the existing record for `coord`, or creates one (Unloaded / None / None) if this
        // is the first time it has ever been touched. Cell records are NEVER erased once created
        // -- a bounded open-world's total cell count is small enough (tens of thousands at most)
        // that keeping every ever-visited cell's bookkeeping alive for the process's lifetime is
        // cheap, and it is what makes capturing a raw CellRecord* into a worker-thread lambda (see
        // Update()'s dispatch loop in the .cpp) safe without a separate ownership/lifetime dance.
        CellRecord& GetOrCreateCellRecord(const CellCoord& coord);

        void EnqueueRequest(const CellCoord& coord, CellRepresentation target, float priorityDistance);

        float m_CellSize;
        IWorldCellLoader& m_Loader;
        core::LoadingManager& m_WorkerPool;
        uint32_t m_MaxConcurrentLoads;
        std::atomic<uint32_t> m_InFlightCount{ 0 };

        mutable std::shared_mutex m_CellsMutex;
        std::unordered_map<CellCoord, std::unique_ptr<CellRecord>, CellCoordHash> m_Cells; // Guarded by m_CellsMutex.

        mutable std::mutex m_QueueMutex; // mutable: GetPendingQueueLength() is const but must lock it.
        std::priority_queue<PendingRequest, std::vector<PendingRequest>, PendingRequestGreater> m_PendingQueue; // Guarded by m_QueueMutex.

        uint64_t m_NextSequence = 0; // Main-thread-only (only touched from EnqueueRequest, itself only ever called from UpdateStreamingSources).
    };

}
