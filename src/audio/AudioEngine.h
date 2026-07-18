#pragma once
// English comments only.
//
// Owns the XAudio2 device + mastering voice and every source voice this engine ever plays: 3
// positional environmental sources (PositionalSource.h -- Embers/Waterfall/Wind) and 1 non-
// positional generative music bed (GenerativeComposer.h). This is a REAL, always-on feature (per
// CLAUDE.md's build-separation rule: audio synthesis/playback/positional mixing is not diagnostic
// tooling) -- it compiles and runs identically in Debug and Release, exactly like this codebase's
// particle system / Atmos weather. Only its ImGui diagnostic panel (main.cpp's Debug-only "Audio"
// tab) is #ifndef NDEBUG-gated -- see the accessors below.
//
// Fully procedural, real-time STREAMING synthesis -- no .wav/.ogg files are ever loaded (matches
// this project's own "aucune data dans mon .exe" constraint): every sample is generated block-by-
// block (see kBlockFrames below, ~10.7 ms of audio per block) by Synth.h/GenerativeComposer.h/
// PositionalSource.h and streamed into XAudio2 through a small ring of pre-allocated float buffers
// per voice, refilled every Update() call as XAudio2 finishes consuming the oldest one -- never
// "render N seconds once and loop". A live parameter change (e.g. an ImGui slider, or
// config::atmos::WIND_SPEED_MPS changing) is audible within roughly one block's worth of audio
// (~10-40 ms depending on queue depth), not only after a full pattern restart.
//
// Backend: XAudio2 (Windows SDK's xaudio2.h, linked against xaudio2.lib -- see CMakeLists.txt's own
// audio-linking comment for exactly what was verified to link cleanly). On this SDK/OS target,
// XAudio2Create resolves to xaudio2_9.dll, which ships as part of Windows 10/11 itself -- zero
// redistributable, zero vendoring weight, matching this project's explicit "no heavy frameworks"
// rule. No X3DAudio/HRTF: positional mixing is plain, fully-owned distance-attenuation + equal-
// power stereo pan math (see PositionalSource.h's own comment for why that is the right amount of
// complexity for this engine's "distance attenuation + stereo pan" positional model).
//
// Per-frame contract mirrors this codebase's own renderer::*Pass Init()/Shutdown()/RecordUpdate()-
// style lifecycle (e.g. renderer::AtmosClimatePass) even though this is not a Vulkan pass, for
// consistency: Init() once at startup, Update(dt, camera) once per frame -- entirely decoupled from
// Vulkan command-buffer recording, since XAudio2 mixes on its own internal thread and this call
// only tops up ring buffers + updates 3D pan/attenuation parameters, never blocks on audio hardware
// -- and Shutdown() once at teardown.
//
// IXAudio2/IXAudio2MasteringVoice/IXAudio2SourceVoice are only forward-declared here (never
// dereferenced outside AudioEngine.cpp) so that including THIS header (e.g. from main.cpp) never
// pulls in <Windows.h>/<xaudio2.h> or their macro pollution (min/max, etc.) -- only AudioEngine.cpp
// itself needs the real XAudio2 declarations.

#include "PositionalSource.h"
#include "GenerativeComposer.h"
#include "core/Camera.h" // CameraFrameInfo
#include <cstdint>
#include <vector>
#include <array>
#include <memory>
#include <functional>

struct IXAudio2;
struct IXAudio2MasteringVoice;
struct IXAudio2SourceVoice;

namespace audio {

class AudioEngine {
public:
    AudioEngine() = default;
    ~AudioEngine();

    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    // Creates the IXAudio2 device + mastering voice, the 3 positional source voices, and the
    // stereo music-bed source voice, then Start()s all 4 (each begins with an empty queue --
    // Update()'s first call submits the first real synthesized buffers, at most kBlockFrames/
    // kSampleRateHz seconds of latency before audio is actually heard). Returns false (logged,
    // never throws) on any XAudio2/COM failure -- callers should treat a false return as "run
    // without audio", not fatal, matching this codebase's own convention for additive-not-load-
    // bearing subsystems (e.g. World Partition streaming's own missing-manifest fallback in
    // main.cpp).
    bool Init();

    void Shutdown();

    // Advances every voice's DSP state by dt (simulation seconds, NOT wall-clock-jitter-prone --
    // caller derives it from consecutive glfwGetTime() calls, same idiom as main.cpp's own fly-
    // camera dt block), recomputes each PositionalSource's pan/attenuation against `camera`, and
    // tops up every voice's ring buffer with freshly-synthesized blocks as XAudio2 finishes
    // consuming old ones. Safe to call even if Init() failed or was never called (silently a
    // no-op).
    void Update(float dt, const CameraFrameInfo& camera);

#ifndef NDEBUG
    // Debug-only diagnostics (ImGui "Audio" tab, main.cpp) -- purely informational read-back, never
    // fed back into Update()'s own logic. Compiled out entirely in Release per CLAUDE.md's build-
    // separation rule (the underlying engine itself is NOT gated -- only this readout is).
    bool IsInitialized() const { return m_Initialized; }
    uint32_t GetGenerativeActiveNoteCount() const;
    uint32_t GetGenerativeStepIndex() const;
    uint32_t GetGenerativeChordIndex() const;
    float GetPositionalPan(uint32_t sourceIndex) const;         // sourceIndex in [0, kPositionalSourceCount).
    float GetPositionalDistanceGain(uint32_t sourceIndex) const;
    const char* GetPositionalSourceName(uint32_t sourceIndex) const;
    static constexpr uint32_t kPositionalSourceCountDebug = 3;
#endif

    static constexpr float kSampleRateHz = 48000.0f;

private:
    static constexpr uint32_t kBlockFrames = 512;      // ~10.7 ms per synthesized block @ 48 kHz.
    static constexpr uint32_t kRingBufferCount = 4;    // ~42.7 ms of buffered lookahead per voice.
    static constexpr uint32_t kPositionalSourceCount = 3;

    // Per-voice XAudio2 handle + its own ring of pre-allocated interleaved float buffers, owned
    // HERE (not by XAudio2 -- SubmitSourceBuffer only borrows the pointer, so this memory must
    // outlive every submission until XAudio2 finishes consuming it, hence a fixed ring rather than
    // transient heap allocations every refill).
    struct VoiceStream {
        IXAudio2SourceVoice* voice = nullptr;
        uint32_t channelCount = 1;
        std::array<std::vector<float>, kRingBufferCount> ringBuffers;
        uint32_t nextRingIndex = 0;
    };

    bool CreateVoice(VoiceStream& stream, uint32_t channelCount);
    void RefillIfNeeded(VoiceStream& stream, uint32_t frameCount,
                         const std::function<void(float*, uint32_t)>& renderFn);

    IXAudio2* m_XAudio2 = nullptr;
    IXAudio2MasteringVoice* m_MasteringVoice = nullptr;
    bool m_ComInitialized = false; // Only true if THIS class' Init() call is what initialized COM (see .cpp) -- only then does Shutdown() call CoUninitialize().
    bool m_Initialized = false;

    std::array<std::unique_ptr<PositionalSource>, kPositionalSourceCount> m_PositionalSources;
    std::array<VoiceStream, kPositionalSourceCount> m_PositionalVoiceStreams;

    std::unique_ptr<GenerativeComposer> m_Composer;
    VoiceStream m_MusicVoiceStream;
};

} // namespace audio
