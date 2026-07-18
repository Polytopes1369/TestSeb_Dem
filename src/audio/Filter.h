#pragma once
// English comments only.
//
// Resonant low-pass filter -- a direct-form-I biquad using Robert Bristow-Johnson's "Audio EQ
// Cookbook" low-pass coefficient formulas (the reference-standard formulas real DAW/synth filter
// sections are built on), driven by a cutoff frequency in Hz and a resonance/Q factor. Coefficients
// are only recomputed when SetParams() is actually called (not every sample) since cutoff/
// resonance change far less often than the sample rate -- Process() itself is the pure per-sample
// hot path. Header-only, same "hot, tiny" reasoning as Oscillator.h/Envelope.h.

#include <cmath>
#include <algorithm>

namespace audio {

class ResonantLowPassFilter {
public:
    void SetParams(float cutoffHz, float resonanceQ, float sampleRateHz) {
        cutoffHz = std::clamp(cutoffHz, 20.0f, sampleRateHz * 0.45f);
        resonanceQ = std::max(resonanceQ, 0.1f);

        float omega = 2.0f * 3.14159265358979323846f * cutoffHz / sampleRateHz;
        float sinOmega = std::sin(omega);
        float cosOmega = std::cos(omega);
        float alpha = sinOmega / (2.0f * resonanceQ);

        // RBJ Cookbook low-pass (constant 0dB peak gain form).
        float b0 = (1.0f - cosOmega) * 0.5f;
        float b1 = 1.0f - cosOmega;
        float b2 = (1.0f - cosOmega) * 0.5f;
        float a0 = 1.0f + alpha;
        float a1 = -2.0f * cosOmega;
        float a2 = 1.0f - alpha;

        // Normalize so a0 == 1 (Process()'s own difference equation assumes this).
        m_B0 = b0 / a0; m_B1 = b1 / a0; m_B2 = b2 / a0;
        m_A1 = a1 / a0; m_A2 = a2 / a0;
    }

    // Direct-form-I biquad difference equation:
    //   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    float Process(float x) {
        float y = m_B0 * x + m_B1 * m_X1 + m_B2 * m_X2 - m_A1 * m_Y1 - m_A2 * m_Y2;
        m_X2 = m_X1; m_X1 = x;
        m_Y2 = m_Y1; m_Y1 = y;
        return y;
    }

    void Reset() { m_X1 = m_X2 = m_Y1 = m_Y2 = 0.0f; }

private:
    float m_B0 = 1.0f, m_B1 = 0.0f, m_B2 = 0.0f;
    float m_A1 = 0.0f, m_A2 = 0.0f;
    float m_X1 = 0.0f, m_X2 = 0.0f;
    float m_Y1 = 0.0f, m_Y2 = 0.0f;
};

} // namespace audio
