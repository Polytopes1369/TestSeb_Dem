#include "StreamingManager.h"

#include <algorithm>
#include <cmath>

namespace world {

    namespace {
        // Numeric rank so "which of two desired representations is more detailed" is a plain
        // comparison -- see UpdateStreamingSources' union-across-sources step.
        int RepresentationRank(CellRepresentation r) { return static_cast<int>(r); }
    }

    StreamingManager::StreamingManager(float cellSize, IWorldCellLoader& loader, core::LoadingManager& workerPool, uint32_t maxConcurrentLoads)
        : m_CellSize(cellSize), m_Loader(loader), m_WorkerPool(workerPool), m_MaxConcurrentLoads(maxConcurrentLoads) {
    }

    CellCoord StreamingManager::WorldToCell(const maths::vec3& worldPos) const {
        return CellCoord{
            static_cast<int32_t>(std::floor(worldPos.x / m_CellSize)),
            static_cast<int32_t>(std::floor(worldPos.z / m_CellSize)),
        };
    }

    maths::vec3 StreamingManager::CellCenter(const CellCoord& coord) const {
        return maths::vec3{
            (static_cast<float>(coord.x) + 0.5f) * m_CellSize,
            0.0f,
            (static_cast<float>(coord.z) + 0.5f) * m_CellSize,
        };
    }

    StreamingManager::CellRecord& StreamingManager::GetOrCreateCellRecord(const CellCoord& coord) {
        {
            std::shared_lock read(m_CellsMutex);
            auto it = m_Cells.find(coord);
            if (it != m_Cells.end()) return *it->second;
        }
        // Not found under a shared (read) lock -- upgrade to exclusive and insert. try_emplace is
        // idempotent, so if another thread won the race and inserted `coord` between the two
        // locks, this simply returns its record instead of creating a second one.
        std::unique_lock write(m_CellsMutex);
        auto [it, inserted] = m_Cells.try_emplace(coord, std::make_unique<CellRecord>());
        return *it->second;
    }

    void StreamingManager::EnqueueRequest(const CellCoord& coord, CellRepresentation target, float priorityDistance) {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        m_PendingQueue.push(PendingRequest{ coord, target, priorityDistance, m_NextSequence++ });
    }

    void StreamingManager::UpdateStreamingSources(const std::vector<StreamingSource>& sources) {
        struct TouchedCell {
            CellRepresentation desired = CellRepresentation::None;
            float nearestDistance = std::numeric_limits<float>::max();
        };
        std::unordered_map<CellCoord, TouchedCell, CellCoordHash> touched;

        // Pass 1: union every source's demand per cell. A square bounding box over-selects a few
        // corner cells outside the true circular radius; the per-cell `dist > hlodLoadRadius`
        // check below discards those, so the only cost is a few wasted distance checks, never a
        // correctness issue.
        for (const StreamingSource& source : sources) {
            CellCoord cellMin = WorldToCell({ source.position.x - source.hlodLoadRadius, 0.0f, source.position.z - source.hlodLoadRadius });
            CellCoord cellMax = WorldToCell({ source.position.x + source.hlodLoadRadius, 0.0f, source.position.z + source.hlodLoadRadius });

            for (int32_t cz = cellMin.z; cz <= cellMax.z; ++cz) {
                for (int32_t cx = cellMin.x; cx <= cellMax.x; ++cx) {
                    CellCoord coord{ cx, cz };
                    maths::vec3 center = CellCenter(coord);
                    float dx = center.x - source.position.x;
                    float dz = center.z - source.position.z;
                    float dist = std::sqrt(dx * dx + dz * dz);
                    if (dist > source.hlodLoadRadius) continue;

                    CellRepresentation desired = (dist <= source.detailLoadRadius) ? CellRepresentation::FullDetail : CellRepresentation::HLOD;

                    TouchedCell& entry = touched[coord];
                    if (RepresentationRank(desired) > RepresentationRank(entry.desired)) entry.desired = desired;
                    entry.nearestDistance = std::min(entry.nearestDistance, dist);
                }
            }
        }

        // Pass 2: for every touched cell, request whatever load its desired representation needs
        // if it isn't already loaded, loading, or unloading.
        for (const auto& [coord, info] : touched) {
            CellRecord& record = GetOrCreateCellRecord(coord);
            record.desiredRepresentation = info.desired;

            CellStreamingState state = record.state.load(std::memory_order_acquire);
            CellRepresentation current = record.currentRepresentation.load(std::memory_order_acquire);

            if (state == CellStreamingState::Unloaded && info.desired != current) {
                // CAS (not a plain store) guards against a race with a worker thread that is, in
                // this exact instant, finishing a previous request for this same cell and about to
                // flip its state away from Unloaded -- see Update()'s dispatch lambda.
                if (record.state.compare_exchange_strong(state, CellStreamingState::LoadingPending)) {
                    EnqueueRequest(coord, info.desired, info.nearestDistance);
                }
            }
            // state == LoadingPending: already in flight; whatever it loads as is picked up as
            // `current` on a later frame, and re-evaluated then if it turns out not to match.
            // state == LoadedActive with current == desired: stable, nothing to do.
            // state == LoadedActive with current != desired, and state == UnloadingPending: both
            // handled by the sweep below.
        }

        // Pass 3: sweep every tracked cell (not just the ones touched this frame) to catch cells
        // no source wants any more (unload) and cells whose loaded representation no longer
        // matches their desired one (an HLOD<->FullDetail swap).
        std::shared_lock read(m_CellsMutex);
        for (auto& [coord, recordPtr] : m_Cells) {
            CellRecord& record = *recordPtr;
            auto touchedIt = touched.find(coord);
            if (touchedIt == touched.end()) {
                record.desiredRepresentation = CellRepresentation::None;
            }

            CellRepresentation desired = record.desiredRepresentation;
            CellStreamingState state = record.state.load(std::memory_order_acquire);
            CellRepresentation current = record.currentRepresentation.load(std::memory_order_acquire);

            if (state == CellStreamingState::LoadedActive && desired != current) {
                // Deliberately unload-then-reload rather than an atomic representation swap: this
                // keeps the state machine at exactly the 4 states the spec calls for (see
                // CellStreamingState's doc comment) instead of adding a 5th state for what a real
                // disk-streaming backend would perform as two separate I/O operations anyway (free
                // the FullDetail actor set, then separately fetch the HLOD proxy, or vice versa).
                // The resulting Unloaded state is picked up and re-requested at the (by-then
                // possibly further-updated) desired representation on a subsequent frame's pass 2.
                if (record.state.compare_exchange_strong(state, CellStreamingState::UnloadingPending)) {
                    // Unload requests always dispatch at maximum priority (distance 0): reclaiming
                    // a cell's budget must never be starved behind farther-away load requests
                    // already queued ahead of it.
                    EnqueueRequest(coord, CellRepresentation::None, 0.0f);
                }
            }
        }
    }

    void StreamingManager::Update() {
        for (;;) {
            if (m_InFlightCount.load(std::memory_order_relaxed) >= m_MaxConcurrentLoads) break;

            PendingRequest request;
            {
                std::lock_guard<std::mutex> lock(m_QueueMutex);
                if (m_PendingQueue.empty()) break;
                request = m_PendingQueue.top();
                m_PendingQueue.pop();
            }

            CellRecord* record = nullptr;
            {
                std::shared_lock read(m_CellsMutex);
                auto it = m_Cells.find(request.coord);
                if (it != m_Cells.end()) record = it->second.get();
            }
            if (!record) continue; // Cannot happen in practice: EnqueueRequest is only ever called right after GetOrCreateCellRecord for the same coord. Defensive only.

            m_InFlightCount.fetch_add(1, std::memory_order_acq_rel);

            CellRepresentation target = request.targetRepresentation;
            CellCoord coord = request.coord;

            // record is a raw pointer into m_Cells, safe to capture: cell records are never erased
            // (see GetOrCreateCellRecord's comment), so it outlives this StreamingManager instance
            // exactly as long as this lambda could possibly run.
            m_WorkerPool.Submit([this, record, coord, target]() {
                // Runs on a core::LoadingManager worker thread -- see IWorldCellLoader's own
                // threading contract (StreamingTypes.h) for what m_Loader's methods may and may not
                // do from here.
                if (target == CellRepresentation::None) {
                    m_Loader.UnloadCell(coord);
                    record->currentRepresentation.store(CellRepresentation::None, std::memory_order_release);
                    record->state.store(CellStreamingState::Unloaded, std::memory_order_release);
                } else if (target == CellRepresentation::HLOD) {
                    m_Loader.LoadCellHlod(coord);
                    record->currentRepresentation.store(CellRepresentation::HLOD, std::memory_order_release);
                    record->state.store(CellStreamingState::LoadedActive, std::memory_order_release);
                } else {
                    m_Loader.LoadCellFullDetail(coord);
                    record->currentRepresentation.store(CellRepresentation::FullDetail, std::memory_order_release);
                    record->state.store(CellStreamingState::LoadedActive, std::memory_order_release);
                }
                m_InFlightCount.fetch_sub(1, std::memory_order_acq_rel);
                });
        }
    }

    CellStreamingState StreamingManager::GetCellState(const CellCoord& coord) const {
        std::shared_lock read(m_CellsMutex);
        auto it = m_Cells.find(coord);
        return (it != m_Cells.end()) ? it->second->state.load(std::memory_order_acquire) : CellStreamingState::Unloaded;
    }

    CellRepresentation StreamingManager::GetCellRepresentation(const CellCoord& coord) const {
        std::shared_lock read(m_CellsMutex);
        auto it = m_Cells.find(coord);
        return (it != m_Cells.end()) ? it->second->currentRepresentation.load(std::memory_order_acquire) : CellRepresentation::None;
    }

    size_t StreamingManager::GetTrackedCellCount() const {
        std::shared_lock read(m_CellsMutex);
        return m_Cells.size();
    }

    size_t StreamingManager::GetPendingQueueLength() const {
        std::lock_guard<std::mutex> lock(m_QueueMutex);
        return m_PendingQueue.size();
    }

}
