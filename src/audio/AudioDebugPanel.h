#pragma once
// Debug-only (whole file compiled out in Release) real-time audio engine diagnostics via ImGui.
// Displays active synth notes, generative composer step/chord index, positional source pan/gain,
// and voice buffer queue depth. Integrated into main.cpp's Debug-only ImGui "Audio" tab.
//
// No GPU work: pure CPU-side parameter readback from AudioEngine, displayed via ImGui's own
// rendering. Safe to call every frame even when audio is uninitialized (renders nothing).

#ifndef NDEBUG

#include <cstdint>

namespace audio {
class AudioEngine;
}

namespace audio::debug {

class AudioDebugPanel {
public:
    AudioDebugPanel() = default;

    AudioDebugPanel(const AudioDebugPanel&) = delete;
    AudioDebugPanel& operator=(const AudioDebugPanel&) = delete;

    // Render ImGui diagnostics panel for the given AudioEngine.
    // Safe to call every frame; renders nothing if engine is not initialized.
    static void RenderImGui(const audio::AudioEngine& engine);
};

} // namespace audio::debug

#else
// Release-mode stub: compile to nothing
namespace audio::debug {
class AudioDebugPanel {
public:
    static void RenderImGui(const audio::AudioEngine&) {}
};
}
#endif
