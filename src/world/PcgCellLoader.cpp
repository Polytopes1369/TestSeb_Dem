#include "PcgCellLoader.h"
#ifndef NDEBUG

#include "core/Logger.h"

#include <format>
#include <mutex>
#include <unordered_set>

namespace world {

    namespace {

        // Worker-thread-safe, process-lifetime throttle for LoadCellHlod()'s own "not implemented"
        // diagnostic (see that method's own header comment) -- a camera orbiting the HLOD band's own
        // boundary would otherwise call LoadCellHlod() for the SAME coord repeatedly, spamming the
        // log with an identical line every time. Deliberately a free function + file-local statics
        // (not a PcgCellLoader member) since this is a pure logging convenience with no bearing on
        // this class' own generation/spawn state -- keeping it out of PcgCellLoader itself avoids
        // growing that class' own locking surface for a diagnostic-only concern.
        bool ShouldLogHlodNotImplemented(const CellCoord& coord) {
            static std::mutex s_Mutex;
            static std::unordered_set<CellCoord, CellCoordHash> s_AlreadyLogged;
            std::lock_guard<std::mutex> lock(s_Mutex);
            return s_AlreadyLogged.insert(coord).second; // .second == true only on first insertion for this coord.
        }

    } // namespace

    PcgCellLoader::PcgCellLoader(const std::filesystem::path& actorsRootDir, float cellSize, pcg::PcgInstanceSpawnManager& spawnManager)
        : m_CellSize(cellSize), m_SpawnManager(spawnManager) {

        const std::vector<worldpartition::PcgVolumeDesc> volumes = ScanPcgVolumeActorFiles(actorsRootDir);
        m_VolumeCount = volumes.size();
        m_VolumeIndex = BuildPcgVolumeCellIndex(volumes, cellSize);

        LOG_INFO(std::format(
            "[PcgCellLoader] Indexed {} authored PCG Volume(s) under '{}' into {} cell(s) (cellSize={:.1f}).",
            m_VolumeCount, actorsRootDir.string(), m_VolumeIndex.size(), cellSize));
    }

    void PcgCellLoader::LoadCellFullDetail(const CellCoord& coord) {
        StageFullDetailGeneration(coord);
    }

    void PcgCellLoader::LoadCellHlod(const CellCoord& coord) {
        // Deliberate no-op -- see this method's own declaration comment (PcgCellLoader.h) for the
        // full rationale: PCG has no HLOD tier of its own yet, and generating full-detail content
        // here would silently mislabel it as an HLOD proxy, which this class refuses to fake.
        if (ShouldLogHlodNotImplemented(coord)) {
            LOG_INFO(std::format(
                "[PcgCellLoader] LoadCellHlod({}, {}): HLOD-tier PCG generation is not implemented "
                "(Phase 6.3 scope) -- no content will be generated for this cell unless/until it is "
                "promoted to FullDetail.", coord.x, coord.z));
        }
    }

    void PcgCellLoader::UnloadCell(const CellCoord& coord) {
        auto it = m_VolumeIndex.find(coord);
        if (it == m_VolumeIndex.end() || it->second.empty()) return; // Never had any overlapping PCG volume -- nothing could have been spawned for it.

        std::lock_guard<std::mutex> lock(m_EventsMutex);
        m_PendingEvents.push_back(PcgCellLoadEvent{ coord, /*isLoad=*/false, {} });
    }

    void PcgCellLoader::StageFullDetailGeneration(const CellCoord& coord) {
        auto it = m_VolumeIndex.find(coord);
        if (it == m_VolumeIndex.end() || it->second.empty()) return; // No PCG volumes overlap this cell -- nothing to generate.

        // --- Phase 6.4 ("Generation Caching"): consult m_GenerationResultCache BEFORE calling the
        // (potentially slow -- disk read + JSON parse + graph evaluation per overlapping volume)
        // pcg::GeneratePcgContentForCell() below. See PcgCellLoader.h's own top-of-file "Phase 6.4"
        // comment for the full key-choice/eviction-policy/thread-safety rationale; this is just that
        // policy's actual implementation. `result` ends up populated either way (from the cache on a
        // hit, freshly generated on a miss) and is always a value THIS function fully owns from this
        // point on -- std::move-ing out of it below (building the PcgCellLoadEvent) is therefore
        // always safe, regardless of which path filled it in. ---
        pcg::PcgCellGenerationResult result;
        bool cacheHit = false;
        {
            std::lock_guard<std::mutex> cacheLock(m_CacheMutex);
            auto cacheIt = m_GenerationResultCache.find(coord);
            if (cacheIt != m_GenerationResultCache.end()) {
                result = cacheIt->second; // Copy out while still holding the lock: cheap (a handful of small PcgSpawnRequest structs at most, see this cache's own bound comment), and avoids handing out a reference/pointer that a concurrent worker thread's insert for a DIFFERENT coord could invalidate via unordered_map rehashing.
                cacheHit = true;
                ++m_CacheHitCount;
            } else {
                ++m_CacheMissCount;
            }
        }

        if (cacheHit) {
            LOG_INFO(std::format(
                "[PcgCellLoader] Cache HIT for cell ({}, {}): reusing {} previously-generated spawn "
                "request(s), pcg::GeneratePcgContentForCell was NOT called.", coord.x, coord.z, result.spawnRequests.size()));
        } else {
            // worldpartition::CellCoord (3-axis, y always 0 in this codebase's Grid2D-only runtime
            // streaming grid) -- see PcgVolumeCellIndex.h's own ToOfflineCellCoord comment.
            pcg::PcgCellGenerationInput input;
            input.cellCoord = ToOfflineCellCoord(coord);
            input.overlappingVolumes = it->second;
            input.cellSize = m_CellSize;

            LOG_INFO(std::format(
                "[PcgCellLoader] Cache MISS for cell ({}, {}): running pcg::GeneratePcgContentForCell "
                "(first load of this cell since construction, or since the last cache-clearing event -- "
                "this class never clears its own cache once populated).", coord.x, coord.z));

            result = pcg::GeneratePcgContentForCell(input);

            // Cache every STRUCTURALLY successful result, including a legitimately empty one (an
            // empty spawnRequests list with success == true is still a valid, deterministic, reusable
            // answer -- e.g. every overlapping volume's graph clipped to zero surviving points inside
            // THIS exact cell; see PcgCellGenerationResult's own field comment, PcgCellGenerator.h).
            // A STRUCTURAL failure (success == false, e.g. a non-positive cellSize) is deliberately
            // NOT cached: m_CellSize is fixed for this instance's whole lifetime, so caching it would
            // not meaningfully save more than the trivial check GeneratePcgContentForCell already
            // performs first, and leaving it uncached keeps the LOG_WARNING below firing on every
            // call for that already-rare, not-recoverable-by-retry case -- exactly this class'
            // pre-6.4 behavior for it, unchanged.
            if (result.success) {
                std::lock_guard<std::mutex> cacheLock(m_CacheMutex);
                // emplace (insert-only-if-absent), not insert_or_assign: if another worker thread's
                // concurrent miss for this SAME coord already won this race, its own result is
                // byte-identical to ours (pcg::GeneratePcgContentForCell is pure/deterministic -- see
                // PcgCellGenerator.h's own "Determinism" comment), so whichever copy actually ends up
                // cached is immaterial.
                m_GenerationResultCache.emplace(coord, result);
            }
        }

        if (!result.success) {
            LOG_WARNING(std::format(
                "[PcgCellLoader] GeneratePcgContentForCell({}, {}) FAILED: {}", coord.x, coord.z, result.errorMessage));
            return;
        }
        if (result.spawnRequests.empty()) {
            // Not an error -- every overlapping volume may have individually skipped (missing/
            // unparseable graph asset, or a graph with no recognizable terminal output -- see
            // PcgCellGenerator.h's own comment). Nothing to stage.
            return;
        }

        PcgCellLoadEvent event;
        event.coord = coord;
        event.isLoad = true;
        event.spawnRequests = std::move(result.spawnRequests);

        std::lock_guard<std::mutex> lock(m_EventsMutex);
        m_PendingEvents.push_back(std::move(event));
    }

    void PcgCellLoader::Pump() {
        std::vector<PcgCellLoadEvent> events;
        {
            std::lock_guard<std::mutex> lock(m_EventsMutex);
            events.swap(m_PendingEvents);
        }

        for (PcgCellLoadEvent& event : events) {
            if (event.isLoad) {
                std::vector<uint32_t> acquiredSlots = m_SpawnManager.SpawnInstances(event.spawnRequests);

                LOG_INFO(std::format(
                    "[PcgCellLoader] Pump: cell ({}, {}) spawned {} of {} requested PCG instance(s).",
                    event.coord.x, event.coord.z, acquiredSlots.size(), event.spawnRequests.size()));

                if (!acquiredSlots.empty()) {
                    // Appended, not overwritten: if LoadCellFullDetail() is ever re-triggered for a
                    // cell before a paired UnloadCell() reaches Pump() (StreamingManager's own
                    // documented Unload-then-Load convergence ordering means this should not
                    // normally happen, but this class does not depend on that guarantee to stay
                    // correct -- see PcgCellLoadEvent's own comment), appending guarantees a later
                    // UnloadCell()'s DespawnInstances() call below releases EVERY slot ever acquired
                    // for this cell, never leaking a subset.
                    std::vector<uint32_t>& existingSlots = m_CellToAcquiredSlots[event.coord];
                    existingSlots.insert(existingSlots.end(), acquiredSlots.begin(), acquiredSlots.end());
                }
            } else {
                auto it = m_CellToAcquiredSlots.find(event.coord);
                if (it == m_CellToAcquiredSlots.end()) {
                    // Nothing acquired for this cell yet (e.g. its own load event is still pending
                    // in a FUTURE Pump() call, or generation produced zero spawn requests) -- no-op,
                    // matches WorldCellStreamingLoader's own "unload of a never-activated cell" case.
                    continue;
                }

                LOG_INFO(std::format(
                    "[PcgCellLoader] Pump: cell ({}, {}) despawning {} previously-acquired PCG instance(s).",
                    event.coord.x, event.coord.z, it->second.size()));

                m_SpawnManager.DespawnInstances(it->second);
                m_CellToAcquiredSlots.erase(it);
            }
        }
    }

}
#endif
