#include "PositionalSource.h"
#include "core/EngineConfig.h"
#include <cmath>
#include <algorithm>

namespace audio {

PositionalSource::PositionalSource(Kind kind, uint32_t seed, const maths::vec3& fixedWorldPosition)
    : m_Kind(kind)
    , m_WorldPosition(fixedWorldPosition)
    , m_NoiseBed(seed)
    , m_CrackleVoice(seed ^ 0xC0FFEEu)
    , m_Random(seed ^ 0xBADF00Du) {
    // Filter/LFO parameters are deliberately NOT set here (left at ResonantLowPassFilter's
    // passthrough default): Update() sets them every frame using the CALLER's actual configured
    // sample rate (AudioEngine::kSampleRateHz), avoiding a second, easy-to-desync copy of that
    // constant in this file. Update() is guaranteed to run at least once before RenderBlock() ever
    // does (AudioEngine's own per-frame contract), so this is never audible as "one silent frame".
}

void PositionalSource::Update(float dt, const CameraFrameInfo& camera, float sampleRateHz) {
    switch (m_Kind) {
    case Kind::Embers:
        // Soft, warm rumble bed (fire's low-frequency roar) plus randomized crackle transients
        // (below) -- position is fixed (config::particles::EMITTERS[0], see AudioEngine.cpp).
        m_NoiseBed.SetFilter(500.0f, 0.7f, sampleRateHz);
        m_NoiseBed.SetLFO(0.6f, 0.3f);
        m_BaseGain = 0.35f;
        break;

    case Kind::Waterfall:
        // Brighter, denser broadband rush -- position is fixed (config::particles::EMITTERS[3]).
        m_NoiseBed.SetFilter(2200.0f, 0.5f, sampleRateHz);
        m_NoiseBed.SetLFO(0.25f, 0.15f);
        m_BaseGain = 0.8f;
        break;

    case Kind::Wind: {
        // See PositionalSource.h's own class comment: wind has no single fixed emission point, so
        // this source's world position tracks a fixed distance from the camera along the current
        // wind bearing every frame instead. dirX = sin(bearing), dirZ = cos(bearing) mirrors
        // config::atmos::WIND_DIRECTION_DEGREES' own documented convention exactly (0 = +Z north,
        // 90 = +X east, XZ-plane compass bearing -- see that field's own comment in EngineConfig.h).
        constexpr float kWindSourceDistanceMeters = 15.0f;
        float bearingRad = maths::ToRadians(config::atmos::WIND_DIRECTION_DEGREES);
        maths::vec3 windDir{ std::sin(bearingRad), 0.0f, std::cos(bearingRad) };
        m_WorldPosition = camera.position + windDir * kWindSourceDistanceMeters;

        float windSpeed = std::max(config::atmos::WIND_SPEED_MPS, 0.0f);
        float speedNorm = std::clamp(windSpeed / 15.0f, 0.0f, 1.0f); // 15 m/s ~ strong gale, a subjective ceiling for this mapping.
        float cutoffHz = 300.0f + speedNorm * 2500.0f;   // Stronger wind -> brighter/hissier.
        m_NoiseBed.SetFilter(cutoffHz, 0.9f, sampleRateHz);
        // Ties the "gustiness" LFO rate to the Atmos wind-turbulence simulation's own frequency
        // knob (config::atmos::WIND_TURBULENCE_FREQUENCY) so the audible gusts and the (separately
        // rendered) visual wind turbulence used by the terrain/foliage systems read as the same
        // underlying weather, not two unrelated randomizations.
        m_NoiseBed.SetLFO(std::max(config::atmos::WIND_TURBULENCE_FREQUENCY, 0.01f) * 2.0f, 0.4f);
        m_BaseGain = 0.15f + speedNorm * 0.85f; // Stronger wind -> louder, never fully silent (ambient breeze floor).
        break;
    }
    }

    // --- Shared pan/distance-attenuation math (all 3 kinds) ---------------------------------
    maths::vec3 toSource = m_WorldPosition - camera.position;
    float distance = toSource.Length();
    maths::vec3 dirToSource = (distance > 1e-4f) ? (toSource * (1.0f / distance)) : camera.forward;

    // Right = forward x worldUp -- mirrors Camera::GetRightVector()'s own exact convention (see
    // Camera.cpp) since CameraFrameInfo only carries `forward`, not a precomputed right vector.
    maths::vec3 worldUp{ 0.0f, 1.0f, 0.0f };
    maths::vec3 right = camera.forward.Cross(worldUp).Normalize();

    m_Pan = std::clamp(dirToSource.Dot(right), -1.0f, 1.0f);
    m_DistanceAttenuationGain = ComputeDistanceAttenuation(distance);

    // --- Embers-only: randomized crackle-transient scheduling -------------------------------
    if (m_Kind == Kind::Embers) {
        m_TimeToNextCrackleSeconds -= dt;
        if (m_TimeToNextCrackleSeconds <= 0.0f) {
            TriggerCrackle(sampleRateHz);
            m_TimeToNextCrackleSeconds = 0.15f + m_Random.NextUnipolar() * 1.1f; // 0.15 - 1.25s between pops.
        }
    }
}

void PositionalSource::TriggerCrackle(float sampleRateHz) {
    SynthVoice::VoiceParams params;
    params.waveform = Waveform::Noise;
    params.envelope.attackSeconds = 0.002f;               // Near-instant snap.
    params.envelope.decaySeconds = 0.03f + m_Random.NextUnipolar() * 0.08f;
    params.envelope.sustainLevel = 0.0f;                   // One-shot AD envelope -- see Envelope.h's own comment.
    params.envelope.releaseSeconds = 0.02f;                // Unused in practice (Decay lands directly on Idle at sustainLevel 0).
    params.filterCutoffHz = 1500.0f + m_Random.NextUnipolar() * 3500.0f; // Bright, "snap"-like timbre.
    params.filterResonanceQ = 1.2f;
    params.gain = 0.5f + m_Random.NextUnipolar() * 0.5f;
    m_CrackleVoice.Trigger(220.0f, params, sampleRateHz); // frequencyHz is irrelevant for Waveform::Noise.
}

float PositionalSource::ComputeDistanceAttenuation(float distance) const {
    // OpenAL-style "inverse clamped distance" model: gain == 1 at/inside refDist, falls off toward
    // 0 as distance grows past it, hard-zeroed beyond maxDist.
    float refDist = std::max(config::audio::ATTENUATION_REFERENCE_DISTANCE_METERS, 0.01f);
    float maxDist = std::max(config::audio::ATTENUATION_MAX_DISTANCE_METERS, refDist);
    if (distance >= maxDist) return 0.0f;
    float rolloff = std::max(config::audio::ATTENUATION_ROLLOFF, 0.0f);
    float gain = refDist / (refDist + rolloff * std::max(0.0f, distance - refDist));
    return std::clamp(gain, 0.0f, 1.0f);
}

void PositionalSource::RenderBlock(float* outMono, uint32_t frameCount, float sampleRateHz) {
    for (uint32_t i = 0; i < frameCount; ++i) {
        float sample = m_NoiseBed.NextSample(sampleRateHz) * m_BaseGain;
        if (m_Kind == Kind::Embers && m_CrackleVoice.IsActive()) {
            sample += m_CrackleVoice.NextSample(sampleRateHz);
        }
        outMono[i] = std::clamp(sample, -1.0f, 1.0f);
    }
}

} // namespace audio
