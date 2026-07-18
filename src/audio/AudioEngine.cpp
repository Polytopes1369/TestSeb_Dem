// English comments only.
#include "AudioEngine.h"
#include "core/EngineConfig.h"
#include "core/Logger.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <objbase.h> // CoInitializeEx / CoUninitialize (WIN32_LEAN_AND_MEAN strips these from Windows.h itself).
#include <xaudio2.h>

#include <format>
#include <cmath>
#include <algorithm>

namespace audio {

AudioEngine::~AudioEngine() {
    Shutdown();
}

bool AudioEngine::Init() {
    // On this SDK/OS target (Windows 10+), XAudio2Create (xaudio2.h's own inline definition)
    // resolves to a plain LoadLibraryEx("xaudio2_9.dll")/GetProcAddress call, NOT COM activation
    // (that CoCreateInstance-based path only applied to the legacy XAudio2 2.7 DirectX-SDK
    // redistributable) -- see xaudio2.h's own XAudio2Create implementation. CoInitializeEx is still
    // called defensively here (some XAudio2/WASAPI internals and this app's own GLFW/ImGui Win32
    // backends may expect the calling thread's COM apartment state to already be set), tolerating
    // "already initialized" (S_FALSE, or RPC_E_CHANGED_MODE if some other component already chose
    // a different concurrency model) rather than treating either as fatal -- only a genuine
    // negative HRESULT other than RPC_E_CHANGED_MODE aborts Init().
    HRESULT comHr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (SUCCEEDED(comHr)) {
        // Only take ownership (and later CoUninitialize) if THIS call is what actually initialized
        // COM on this thread (S_OK) -- S_FALSE means it was already initialized by someone else
        // (GLFW's Win32 backend, most likely), whose own lifetime we must not interfere with.
        m_ComInitialized = (comHr == S_OK);
    } else if (comHr != RPC_E_CHANGED_MODE) {
        LOG_CRITICAL(std::format("[AudioEngine] CoInitializeEx failed: 0x{:08X}", static_cast<uint32_t>(comHr)));
        return false;
    }

    HRESULT hr = XAudio2Create(&m_XAudio2, 0, XAUDIO2_DEFAULT_PROCESSOR);
    if (FAILED(hr)) {
        LOG_CRITICAL(std::format("[AudioEngine] XAudio2Create failed: 0x{:08X}", static_cast<uint32_t>(hr)));
        return false;
    }

    hr = m_XAudio2->CreateMasteringVoice(&m_MasteringVoice, 2u, static_cast<UINT32>(kSampleRateHz));
    if (FAILED(hr)) {
        LOG_CRITICAL(std::format("[AudioEngine] CreateMasteringVoice failed: 0x{:08X}", static_cast<uint32_t>(hr)));
        return false;
    }
    m_MasteringVoice->SetVolume(config::audio::MASTER_VOLUME);

    // --- 3 positional environmental sources, wired to real, existing scene elements -------------
    // Embers: config::particles::EMITTERS[0] -- the existing "Embers" fire particle emitter.
    // Waterfall: config::particles::EMITTERS[3] -- the existing waterfall mist/foam particle
    //   emitter (itself already kept in sync with river_spline.glsl's own kRiverControlXZ[3]/
    //   kRiverControlHeight[3], see that EmitterConfig's own comment in EngineConfig.h -- reusing
    //   its position here rather than re-deriving/duplicating the river spline's own literals).
    // Wind: has no fixed real-world position -- PositionalSource itself recomputes it every frame
    //   relative to the camera (see PositionalSource.h's own class comment); the position passed
    //   here is only ever used before the very first Update() call.
    const config::particles::EmitterConfig& embersEmitter = config::particles::EMITTERS[0];
    const config::particles::EmitterConfig& waterfallEmitter = config::particles::EMITTERS[3];

    m_PositionalSources[0] = std::make_unique<PositionalSource>(PositionalSource::Kind::Embers, 0x1001u,
        maths::vec3{ embersEmitter.positionX, embersEmitter.positionY, embersEmitter.positionZ });
    m_PositionalSources[1] = std::make_unique<PositionalSource>(PositionalSource::Kind::Waterfall, 0x1002u,
        maths::vec3{ waterfallEmitter.positionX, waterfallEmitter.positionY, waterfallEmitter.positionZ });
    m_PositionalSources[2] = std::make_unique<PositionalSource>(PositionalSource::Kind::Wind, 0x1003u,
        maths::vec3{ 0.0f, 2.0f, 0.0f });

    for (uint32_t i = 0; i < kPositionalSourceCount; ++i) {
        if (!CreateVoice(m_PositionalVoiceStreams[i], /*channelCount*/ 1u)) {
            LOG_CRITICAL(std::format("[AudioEngine] Failed to create positional source voice {}.", i));
            return false;
        }
    }

    // --- Generative music bed (non-positional, stereo) -------------------------------------
    m_Composer = std::make_unique<GenerativeComposer>(config::audio::GENERATIVE_SEED);
    if (!CreateVoice(m_MusicVoiceStream, /*channelCount*/ 2u)) {
        LOG_CRITICAL("[AudioEngine] Failed to create generative music source voice.");
        return false;
    }

    m_Initialized = true;
    LOG_INFO("[AudioEngine] Initialized: XAudio2 device + mastering voice + 3 positional sources "
              "(Embers/Waterfall/Wind) + 1 generative music bed, all real-time streaming synthesis.");
    return true;
}

bool AudioEngine::CreateVoice(VoiceStream& stream, uint32_t channelCount) {
    WAVEFORMATEX format{};
    format.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    format.nChannels = static_cast<WORD>(channelCount);
    format.nSamplesPerSec = static_cast<DWORD>(kSampleRateHz);
    format.wBitsPerSample = 32;
    format.nBlockAlign = static_cast<WORD>(format.nChannels * (format.wBitsPerSample / 8));
    format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
    format.cbSize = 0;

    // pSendList = nullptr: XAudio2 defaults a source voice's output to the first mastering voice
    // created (m_MasteringVoice, already created above), so no explicit XAUDIO2_VOICE_SENDS is
    // needed here -- see IXAudio2::CreateSourceVoice's own documented default-routing behavior.
    HRESULT hr = m_XAudio2->CreateSourceVoice(&stream.voice, &format);
    if (FAILED(hr)) {
        LOG_CRITICAL(std::format("[AudioEngine] CreateSourceVoice ({} ch) failed: 0x{:08X}",
            channelCount, static_cast<uint32_t>(hr)));
        return false;
    }

    stream.channelCount = channelCount;
    for (std::vector<float>& buffer : stream.ringBuffers) {
        buffer.assign(static_cast<size_t>(kBlockFrames) * channelCount, 0.0f);
    }
    stream.nextRingIndex = 0;

    hr = stream.voice->Start(0);
    if (FAILED(hr)) {
        LOG_CRITICAL(std::format("[AudioEngine] IXAudio2SourceVoice::Start failed: 0x{:08X}", static_cast<uint32_t>(hr)));
        return false;
    }
    return true;
}

void AudioEngine::RefillIfNeeded(VoiceStream& stream, uint32_t frameCount,
                                  const std::function<void(float*, uint32_t)>& renderFn) {
    if (!stream.voice) return;

    XAUDIO2_VOICE_STATE state{};
    stream.voice->GetState(&state, XAUDIO2_VOICE_NOSAMPLESPLAYED);

    // Keep at most kRingBufferCount buffers queued on the voice at any time. XAudio2 processes
    // submitted buffers in strict FIFO order, so as soon as BuffersQueued drops below the ring
    // size, the ring slot exactly kRingBufferCount submissions behind `nextRingIndex` (i.e. the
    // very slot `nextRingIndex` is about to (re)write) is GUARANTEED to have already finished
    // playing -- this is the standard XAudio2 streaming-ring idiom, and the reason the ring must be
    // sized kRingBufferCount with no smaller: submitting into a slot XAudio2 might still be reading
    // from would corrupt in-flight audio.
    while (state.BuffersQueued < kRingBufferCount) {
        std::vector<float>& buffer = stream.ringBuffers[stream.nextRingIndex];
        renderFn(buffer.data(), frameCount);

        XAUDIO2_BUFFER xaBuffer{};
        xaBuffer.AudioBytes = static_cast<UINT32>(buffer.size() * sizeof(float));
        xaBuffer.pAudioData = reinterpret_cast<const BYTE*>(buffer.data());
        xaBuffer.Flags = 0; // Not XAUDIO2_END_OF_STREAM -- this is a continuous, never-ending stream.

        HRESULT hr = stream.voice->SubmitSourceBuffer(&xaBuffer);
        if (FAILED(hr)) {
            LOG_WARNING(std::format("[AudioEngine] SubmitSourceBuffer failed: 0x{:08X}", static_cast<uint32_t>(hr)));
            break;
        }

        stream.nextRingIndex = (stream.nextRingIndex + 1u) % kRingBufferCount;
        // Reflects the buffer we just queued without an extra GetState() round-trip -- BuffersQueued
        // is a plain counter XAudio2 increments on submit/decrements on completion, so tracking it
        // locally across this loop's own iterations is exact, not an approximation.
        state.BuffersQueued++;
    }
}

void AudioEngine::Update(float dt, const CameraFrameInfo& camera) {
    if (!m_Initialized) return;

    m_MasteringVoice->SetVolume(config::audio::MASTER_VOLUME);

    for (uint32_t i = 0; i < kPositionalSourceCount; ++i) {
        PositionalSource& source = *m_PositionalSources[i];
        source.Update(dt, camera, kSampleRateHz);

        VoiceStream& stream = m_PositionalVoiceStreams[i];
        if (config::audio::POSITIONAL_AUDIO_ENABLED) {
            float userVolume = 1.0f;
            switch (source.GetKind()) {
            case PositionalSource::Kind::Embers:    userVolume = config::audio::EMBERS_VOLUME; break;
            case PositionalSource::Kind::Waterfall: userVolume = config::audio::WATERFALL_VOLUME; break;
            case PositionalSource::Kind::Wind:      userVolume = config::audio::WIND_VOLUME; break;
            }
            float finalVolume = std::clamp(userVolume * source.GetDistanceAttenuationGain(), 0.0f, 1.0f);
            stream.voice->SetVolume(finalVolume);

            // Mono source -> stereo mastering voice: a [1 x 2] output matrix
            // (pLevelMatrix[S + SourceChannels*D], SourceChannels==1 so this is simply {left, right}).
            float leftGain, rightGain;
            ComputeEqualPowerPan(source.GetPan(), leftGain, rightGain);
            float matrix[2] = { leftGain, rightGain };
            stream.voice->SetOutputMatrix(m_MasteringVoice, 1u, 2u, matrix);
        } else {
            // Positional audio disabled: these are continuous ambience beds (no "note" to let ring
            // out), so a hard mute is the correct, click-free behavior here -- unlike the
            // generative music bed below, which fades its already-triggered notes out naturally
            // instead of being volume-gated.
            stream.voice->SetVolume(0.0f);
        }

        RefillIfNeeded(stream, kBlockFrames, [&source](float* out, uint32_t frameCount) {
            source.RenderBlock(out, frameCount, kSampleRateHz);
        });
    }

    if (m_MusicVoiceStream.voice) {
        // Always audible at its configured volume regardless of config::audio::
        // GENERATIVE_MUSIC_ENABLED -- GenerativeComposer::RenderBlock itself stops triggering NEW
        // notes when disabled but keeps rendering already-triggered ones through their own release
        // tail (see that method's own comment), so the piece fades out gracefully rather than
        // being cut off mid-note by a volume gate here.
        m_MusicVoiceStream.voice->SetVolume(config::audio::GENERATIVE_MUSIC_VOLUME);
        RefillIfNeeded(m_MusicVoiceStream, kBlockFrames, [this](float* out, uint32_t frameCount) {
            m_Composer->RenderBlock(out, frameCount, kSampleRateHz);
        });
    }
}

void AudioEngine::Shutdown() {
    if (!m_Initialized && !m_XAudio2) return; // Never initialized, or already shut down -- idempotent.

    for (VoiceStream& stream : m_PositionalVoiceStreams) {
        if (stream.voice) {
            stream.voice->Stop(0);
            stream.voice->DestroyVoice();
            stream.voice = nullptr;
        }
    }
    if (m_MusicVoiceStream.voice) {
        m_MusicVoiceStream.voice->Stop(0);
        m_MusicVoiceStream.voice->DestroyVoice();
        m_MusicVoiceStream.voice = nullptr;
    }
    if (m_MasteringVoice) {
        m_MasteringVoice->DestroyVoice();
        m_MasteringVoice = nullptr;
    }
    if (m_XAudio2) {
        m_XAudio2->Release();
        m_XAudio2 = nullptr;
    }
    if (m_ComInitialized) {
        CoUninitialize();
        m_ComInitialized = false;
    }

    m_Composer.reset();
    for (std::unique_ptr<PositionalSource>& source : m_PositionalSources) source.reset();

    if (m_Initialized) LOG_INFO("[AudioEngine] Shutdown complete.");
    m_Initialized = false;
}

#ifndef NDEBUG
uint32_t AudioEngine::GetGenerativeActiveNoteCount() const {
    return m_Composer ? m_Composer->GetActiveVoiceCount() : 0u;
}
uint32_t AudioEngine::GetGenerativeStepIndex() const {
    return m_Composer ? m_Composer->GetCurrentStepIndex() : 0u;
}
uint32_t AudioEngine::GetGenerativeChordIndex() const {
    return m_Composer ? m_Composer->GetCurrentChordIndex() : 0u;
}
float AudioEngine::GetPositionalPan(uint32_t sourceIndex) const {
    return (sourceIndex < kPositionalSourceCount && m_PositionalSources[sourceIndex])
        ? m_PositionalSources[sourceIndex]->GetPan() : 0.0f;
}
float AudioEngine::GetPositionalDistanceGain(uint32_t sourceIndex) const {
    return (sourceIndex < kPositionalSourceCount && m_PositionalSources[sourceIndex])
        ? m_PositionalSources[sourceIndex]->GetDistanceAttenuationGain() : 0.0f;
}
const char* AudioEngine::GetPositionalSourceName(uint32_t sourceIndex) const {
    static const char* kNames[kPositionalSourceCount] = { "Embers", "Waterfall", "Wind" };
    return (sourceIndex < kPositionalSourceCount) ? kNames[sourceIndex] : "?";
}
#endif

} // namespace audio
