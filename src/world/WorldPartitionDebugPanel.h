#pragma once
// Debug-only (whole file compiled out in Release) world partition streaming diagnostics via ImGui.
// Displays streaming cell grid status, entity counts per cell, residency state, and distance-to-
// camera metrics. Integrated into main.cpp's Debug-only ImGui interface.
//
// No GPU work: pure CPU-side parameter readback from WorldPartition, displayed via ImGui.
// Safe to call every frame even when streaming is disabled.

#ifndef NDEBUG

#include <cstdint>

namespace world {
class StreamingManager;
}

namespace world::debug {

class WorldPartitionDebugPanel {
public:
    WorldPartitionDebugPanel() = default;

    WorldPartitionDebugPanel(const WorldPartitionDebugPanel&) = delete;
    WorldPartitionDebugPanel& operator=(const WorldPartitionDebugPanel&) = delete;

    // Render ImGui diagnostics panel for world streaming and partition grid.
    // Shows active cells, entity distribution, memory usage, and streaming throughput.
    static void RenderImGui(const world::StreamingManager& streaming);
};

} // namespace world::debug

#else
// Release-mode stub: compile to nothing
namespace world::debug {
class WorldPartitionDebugPanel {
public:
    static void RenderImGui(const world::WorldPartition&) {}
};
}
#endif
