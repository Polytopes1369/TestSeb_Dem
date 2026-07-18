#pragma once
// English comments only.
//
// Minimal procedural oscillator + fast PRNG for the audio synthesis core (see AudioEngine.h's own
// class comment for the overall src/audio/ architecture). Header-only: every method here is a
// handful of scalar float-math instructions, called once per OUTPUT SAMPLE from AudioEngine's
// real-time buffer-fill path (up to 48000 times/second per active voice), so inlining matters more
// here than almost anywhere else in this codebase -- the same "hot, tiny, header-only" convention
// core/maths/Maths.h already establishes for vec3's own Dot/Cross/Normalize.

#include <cstdint>
#include <cmath>

namespace audio {

enum class Waveform : uint32_t {
    Sine = 0,
    Saw = 1,
    Square = 2,
    Triangle = 3,
    Noise = 4,
};

// Fast, deterministic-from-seed pseudo-random generator (xorshift32) for the Noise waveform and
// every other DSP block in src/audio/ that needs cheap real-time randomness (PositionalSource's
// filtered-noise environmental generators, GenerativeComposer's note picker). Deliberately NOT
// std::mt19937: that is the right choice for this codebase's offline/CPU-time-insensitive
// procedural generation elsewhere (terrain, cluster partitioning), but this generator is stepped up
// to 48000 times per second per active voice, where std::mt19937's much larger state and per-step
// cost buys no audible benefit.
class Xorshift32 {
public:
    explicit Xorshift32(uint32_t seed) : m_State(seed != 0 ? seed : 0x9E3779B9u) {}

    // Returns a uniform pseudo-random float in [-1, 1).
    float NextBipolar() {
        m_State ^= m_State << 13;
        m_State ^= m_State >> 17;
        m_State ^= m_State << 5;
        // Top 23 bits -> [0, 1), then remap to [-1, 1).
        constexpr float inv = 1.0f / 8388608.0f; // 1 / 2^23
        return static_cast<float>(m_State >> 9) * inv - 1.0f;
    }

    // Returns a uniform pseudo-random float in [0, 1).
    float NextUnipolar() { return (NextBipolar() + 1.0f) * 0.5f; }

private:
    uint32_t m_State;
};

// Single band-limited-ish oscillator (naive/non-PolyBLEP synthesis for Saw/Square/Triangle --
// acceptable aliasing for this engine's actual register: every waveform played here is either a
// soft pad note well below Nyquist/8 (GenerativeComposer's ambient bed) or already broadband noise
// (PositionalSource's environmental textures), so the added complexity of PolyBLEP/BLIT
// anti-aliased synthesis a lead-synth-focused engine would need buys nothing audible here).
class Oscillator {
public:
    explicit Oscillator(uint32_t noiseSeed = 0x1234567u) : m_Noise(noiseSeed) {}

    void SetWaveform(Waveform waveform) { m_Waveform = waveform; }
    void SetFrequencyHz(float frequencyHz) { m_FrequencyHz = frequencyHz; }
    float GetFrequencyHz() const { return m_FrequencyHz; }

    // Advances the phase accumulator by one sample and returns the next output sample, in [-1, 1].
    float NextSample(float sampleRateHz) {
        float sample = 0.0f;
        switch (m_Waveform) {
        case Waveform::Sine:
            sample = std::sin(m_Phase * 2.0f * 3.14159265358979323846f);
            break;
        case Waveform::Saw:
            sample = 2.0f * m_Phase - 1.0f;
            break;
        case Waveform::Square:
            sample = (m_Phase < 0.5f) ? 1.0f : -1.0f;
            break;
        case Waveform::Triangle:
            sample = 4.0f * std::fabs(m_Phase - 0.5f) - 1.0f;
            break;
        case Waveform::Noise:
            sample = m_Noise.NextBipolar();
            break;
        }

        m_Phase += m_FrequencyHz / sampleRateHz;
        if (m_Phase >= 1.0f) m_Phase -= std::floor(m_Phase);

        return sample;
    }

    void ResetPhase(float phase01 = 0.0f) { m_Phase = phase01; }

private:
    Waveform m_Waveform = Waveform::Sine;
    float m_FrequencyHz = 440.0f;
    float m_Phase = 0.0f;
    Xorshift32 m_Noise;
};

} // namespace audio
