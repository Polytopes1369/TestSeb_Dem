#ifndef NDEBUG

#include "world/WorldPartitionDebugPanel.h"
#include "world/WorldPartition.h"
#include <imgui.h>

namespace world::debug {

void WorldPartitionDebugPanel::RenderImGui(const world::WorldPartition& partition) {
    ImGui::BeginChild("World Partition Streaming", ImVec2(0, 0), true);
    {
        ImGui::Text("World Partition & Streaming Grid");
        ImGui::Separator();

        // Get partition statistics (these methods would need to be added to WorldPartition).
        // For now, we provide a placeholder structure that can be expanded once the partition
        // is fully instrumented with debug accessors.

        ImGui::Text("Total grid cells: (awaiting WorldPartition debug API)");
        ImGui::Text("Active loaded cells: (awaiting WorldPartition debug API)");
        ImGui::Text("Total entities in memory: (awaiting WorldPartition debug API)");
        ImGui::Text("Streaming throughput: (awaiting WorldPartition debug API)");

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
