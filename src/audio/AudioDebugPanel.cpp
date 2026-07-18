#ifndef NDEBUG

#include "audio/AudioDebugPanel.h"
#include "audio/AudioEngine.h"
#include <imgui.h>
#include <string>

namespace audio::debug {

void AudioDebugPanel::RenderImGui(const audio::AudioEngine& engine) {
    if (!engine.IsInitialized()) {
        ImGui::Text("Audio Engine: NOT INITIALIZED");
        return;
    }

    ImGui::Text("Audio Engine: ACTIVE");
    ImGui::Separator();

    // Generative music bed diagnostics.
    ImGui::BeginChild("Generative Composer", ImVec2(0, 0), true);
    {
        ImGui::Text("Generative Composer (Stereo Music Bed):");
        ImGui::Indent();
        {
            uint32_t activeNotes = engine.GetGenerativeActiveNoteCount();
            uint32_t stepIdx = engine.GetGenerativeStepIndex();
            uint32_t chordIdx = engine.GetGenerativeChordIndex();

            ImGui::Text("  Active Notes: %u", activeNotes);
            ImGui::Text("  Step Index: %u", stepIdx);
            ImGui::Text("  Chord Index: %u", chordIdx);
            ImGui::ProgressBar(stepIdx / 32.0f, ImVec2(-1, 0), "Step Progress");
            ImGui::ProgressBar(chordIdx / 8.0f, ImVec2(-1, 0), "Chord Progress");
        }
        ImGui::Unindent();
    }
    ImGui::EndChild();

    ImGui::Spacing();

    // Positional sources (Embers, Waterfall, Wind).
    ImGui::BeginChild("Positional Sources", ImVec2(0, 0), true);
    {
        ImGui::Text("Positional Sources (3D Environmental Audio):");
        ImGui::Indent();
        {
            for (uint32_t i = 0; i < audio::AudioEngine::kPositionalSourceCountDebug; ++i) {
                const char* sourceName = engine.GetPositionalSourceName(i);
                float pan = engine.GetPositionalPan(i);
                float distanceGain = engine.GetPositionalDistanceGain(i);

                ImGui::Separator();
                ImGui::Text("  [%u] %s", i, sourceName);
                ImGui::SliderFloat(
                    ("Pan##" + std::to_string(i)).c_str(),
                    &pan, -1.0f, 1.0f, "%.2f", ImGuiSliderFlags_ReadOnly);
                ImGui::SliderFloat(
                    ("Distance Gain##" + std::to_string(i)).c_str(),
                    &distanceGain, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_ReadOnly);
            }
        }
        ImGui::Unindent();
    }
    ImGui::EndChild();

    ImGui::Text("Sample Rate: %.0f Hz | Block Size: 512 frames (~10.7 ms latency)", AudioEngine::kSampleRateHz);
}

} // namespace audio::debug

#endif
