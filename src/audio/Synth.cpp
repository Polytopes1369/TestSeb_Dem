#include "Synth.h"
#include <cmath>
#include <algorithm>

namespace audio {

// ============================================================================================
// SynthVoice
// ============================================================================================

SynthVoice::SynthVoice(uint32_t noiseSeed) : m_Oscillator(noiseSeed) {}

void SynthVoice::Trigger(float frequencyHz, const VoiceParams& params, float sampleRateHz) {
    m_Oscillator.SetWaveform(params.waveform);
    m_Oscillator.SetFrequencyHz(frequencyHz);
    m_Filter.SetParams(params.filterCutoffHz, params.filterResonanceQ, sampleRateHz);
    m_Envelope.SetParams(params.envelope);
    m_Envelope.NoteOn();
    m_Gain = params.gain;
}

void SynthVoice::Release() { m_Envelope.NoteOff(); }

bool SynthVoice::IsActive() const { return m_Envelope.IsActive(); }

float SynthVoice::NextSample(float sampleRateHz) {
    float raw = m_Oscillator.NextSample(sampleRateHz);
    float filtered = m_Filter.Process(raw);
    float envAmount = m_Envelope.NextSample(sampleRateHz);
    return filtered * envAmount * m_Gain;
}

// ============================================================================================
// NoiseBed
// ============================================================================================

NoiseBed::NoiseBed(uint32_t seed) : m_Noise(seed) {}

void NoiseBed::SetFilter(float cutoffHz, float resonanceQ, float sampleRateHz) {
    m_Filter.SetParams(cutoffHz, resonanceQ, sampleRateHz);
}

void NoiseBed::SetLFO(float lfoRateHz, float lfoDepth01) {
    m_LFORateHz = lfoRateHz;
    m_LFODepth = lfoDepth01;
}

float NoiseBed::NextSample(float sampleRateHz) {
    float raw = m_Noise.NextBipolar();
    float filtered = m_Filter.Process(raw);

    m_LFOPhase += m_LFORateHz / sampleRateHz;
    if (m_LFOPhase >= 1.0f) m_LFOPhase -= std::floor(m_LFOPhase);
    // Raised-cosine LFO in [1 - lfoDepth, 1] -- never fully mutes the bed even at full depth,
    // just breathes its amplitude, matching real wind/water gustiness rather than tremolo-gating.
    float lfo = 1.0f - m_LFODepth * (0.5f - 0.5f * std::cos(m_LFOPhase * 2.0f * 3.14159265358979323846f));

    return filtered * lfo;
}

// ============================================================================================
// ComputeEqualPowerPan
// ============================================================================================

void ComputeEqualPowerPan(float pan, float& outLeft, float& outRight) {
    float clampedPan = std::clamp(pan, -1.0f, 1.0f);
    float angle = (clampedPan * 0.5f + 0.5f) * (3.14159265358979323846f * 0.5f);
    outLeft = std::cos(angle);
    outRight = std::sin(angle);
}

} // namespace audio
