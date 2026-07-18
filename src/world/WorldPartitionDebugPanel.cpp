#ifndef NDEBUG

#include "world/WorldPartitionDebugPanel.h"
#include "world/StreamingManager.h"
#include <imgui.h>

namespace world::debug {

void WorldPartitionDebugPanel::RenderImGui(const world::StreamingManager& streaming) {
    ImGui::BeginChild("World Partition Streaming", ImVec2(0, 0), true);
    {
        ImGui::Text("World Streaming & Partition Grid");
        ImGui::Separator();

        size_t trackedCells = streaming.GetTrackedCellCount();
        uint32_t inFlightCount = streaming.GetInFlightCount();
        size_t pendingQueueLength = streaming.GetPendingQueueLength();

        ImGui::Text("Tracked grid cells: %zu", trackedCells);
        ImGui::Text("In-flight load/unload requests: %u", inFlightCount);
        ImGui::Text("Pending queue depth: %zu", pendingQueueLength);
        ImGui::ProgressBar(inFlightCount / 4.0f, ImVec2(-1, 0), "Load Budget Used");

        ImGui::Separator();
        ImGui::Text("Cell Grid Status:");
        ImGui::Indent();
        {
            ImGui::BulletText("Awaiting debug accessors");
            ImGui::BulletText("Would display grid coordinates (min/max X/Z)");
            ImGui::BulletText("Would show per-cell entity counts");
            ImGui::BulletText("Would indicate residency state (loaded/pending/unloaded)");
        }
        ImGui::Unindent();

        ImGui::Separator();
        ImGui::Text("To enable full debug output:");
        ImGui::BulletText("Add accessor GetGridCellCount()");
        ImGui::BulletText("Add accessor GetActiveCellCount()");
        ImGui::BulletText("Add accessor GetEntityCount()");
        ImGui::BulletText("Add accessor GetCellEntityCount(cellCoord)");
        ImGui::BulletText("Add accessor GetCellResidencyState(cellCoord)");
    }
    ImGui::EndChild();
}

} // namespace world::debug

#endif
