#pragma once
// Concrete world::IWorldCellLoader backing StreamingManager with this demo's bounded pool of
// pre-baked procedural "archetype" GPU entity slots (see renderer::VulkanContext's own
// kStreamingUnitCount/SetStreamingUnitState comment for the full rationale: live per-cell Nanite
// cluster DAG builds are not feasible on a streaming budget in this engine, see that class's
// header for the DAG-build-patience precedent, so streamed-in content is drawn from a small fixed
// set of pre-baked shapes rather than unique per-cell geometry).
//
// Deliberately has ZERO dependency on renderer::VulkanContext or any Vulkan type, per
// IWorldCellLoader's own threading contract ("must NOT touch Vulkan command recording directly").
// LoadCellFullDetail/LoadCellHlod/UnloadCell run on a core::LoadingManager worker thread and only
// ever stage a StreamingPlacementEvent into a small thread-safe queue; the caller's main thread
// drains it once per frame via DrainEvents() and performs the actual GPU entity-slot mutation --
// same producer(worker)/consumer(main-thread-pump) split this codebase already uses for
// io::AsyncDecompressingLoader's ChunkLoadCallback.

#include <mutex>
#include <string>
#include <vector>

#include "CellManifest.h"
#include "StreamingTypes.h"

namespace world {

    // Mirrors renderer::VulkanContext's own archetype shape table 1:1 (Rock/Bush/Tree/Debris) --
    // see that class's kStreamingArchetypeShapeCount comment. Kept as a plain index (not shared
    // enum) since world:: has no dependency on renderer::, matching this file's own header comment.
    inline constexpr uint32_t kArchetypeShapeCount = 4;

    struct StreamingPlacementEvent {
        CellCoord coord;
        bool activate = false; // true: claim/reposition a unit for this cell. false: release the cell's claimed unit.
        bool useFineVariant = false; // Only meaningful when activate == true: FullDetail -> fine mesh, HLOD -> coarse mesh.
        uint32_t archetypeShape = 0;
        maths::vec3 worldPosition{};
    };

    class WorldCellStreamingLoader : public IWorldCellLoader {
    public:
        explicit WorldCellStreamingLoader(const CellManifest& manifest) : m_Manifest(manifest) {}

        // --- IWorldCellLoader: called from a core::LoadingManager worker thread, never the main
        // thread, and potentially concurrently for different cells (see the interface's own
        // threading contract). ---
        void LoadCellFullDetail(const CellCoord& coord) override { StageActivate(coord, /*useFineVariant=*/true); }
        void LoadCellHlod(const CellCoord& coord) override { StageActivate(coord, /*useFineVariant=*/false); }
        void UnloadCell(const CellCoord& coord) override {
            std::lock_guard<std::mutex> lock(m_EventsMutex);
            m_PendingEvents.push_back(StreamingPlacementEvent{ coord, /*activate=*/false, false, 0, {} });
        }

        // Main-thread-only: returns and clears every event staged since the last call.
        std::vector<StreamingPlacementEvent> DrainEvents() {
            std::lock_guard<std::mutex> lock(m_EventsMutex);
            std::vector<StreamingPlacementEvent> events;
            events.swap(m_PendingEvents);
            return events;
        }

    private:
        void StageActivate(const CellCoord& coord, bool useFineVariant) {
            std::optional<CellPlacement> placement = m_Manifest.GetPlacement(coord);
            if (!placement.has_value()) return; // No authored content for this cell -- nothing to stream in.

            std::lock_guard<std::mutex> lock(m_EventsMutex);
            m_PendingEvents.push_back(StreamingPlacementEvent{
                coord, /*activate=*/true, useFineVariant, placement->archetypeShape, placement->worldPosition });
        }

        const CellManifest& m_Manifest;
        std::mutex m_EventsMutex;
        std::vector<StreamingPlacementEvent> m_PendingEvents; // Guarded by m_EventsMutex.
    };

}
