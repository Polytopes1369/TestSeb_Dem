#pragma once
// Debug-only (whole file compiled out in Release) skeletal animation diagnostics via ImGui.
// Displays active animations per entity, joint transform hierarchy, playback time, and frame
// index. Integrated into main.cpp's Debug-only ImGui interface.
//
// No GPU work: pure CPU-side parameter readback from SkeletalAnimator, displayed via ImGui.
// Safe to call every frame even when no animations are playing.

#ifndef NDEBUG

#include <cstdint>

namespace animation {
class SkeletalAnimator;
}

namespace animation::debug {

class AnimationDebugPanel {
public:
    AnimationDebugPanel() = default;

    AnimationDebugPanel(const AnimationDebugPanel&) = delete;
    AnimationDebugPanel& operator=(const AnimationDebugPanel&) = delete;

    // Render ImGui diagnostics panel for all active skeletal animations.
    // Lists entity animations, joint counts, playback time, and frame index.
    static void RenderImGui(const animation::SkeletalAnimator& animator);
};

} // namespace animation::debug

#else
// Release-mode stub: compile to nothing
namespace animation::debug {
class AnimationDebugPanel {
public:
    static void RenderImGui(const animation::SkeletalAnimator&) {}
};
}
#endif
