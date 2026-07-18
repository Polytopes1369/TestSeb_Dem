#pragma once
// English comments only.
//
// Two DSP "voice" building blocks, both composed from Oscillator.h/Envelope.h/Filter.h:
//
//  - SynthVoice: a triggerable, releasable NOTE (oscillator -> resonant low-pass filter -> ADSR
//    amplitude envelope). Used by GenerativeComposer's polyphonic pad pool for the generative
//    music bed, and by PositionalSource for the ember-crackle transient.
//  - NoiseBed: a CONTINUOUS (never-idle) filtered-noise texture generator with an optional slow
//    tremolo LFO. Used by PositionalSource for the wind/waterfall/ember-rumble environmental beds.
//
// Declarations only here; NextSample()/Trigger() bodies live in Synth.cpp (unlike the pure-leaf
// Oscillator/Envelope/Filter primitives, which stay header-only for per-sample inlining -- see
// those files' own comments) since these two classes compose several such primitives together and
// match this codebase's general .h/.cpp pairing convention for anything beyond a trivial leaf.

#include "Oscillator.h"
#include "Envelope.h"
#include "Filter.h"

namespace audio {

// One playable synthesizer voice: oscillator -> resonant low-pass filter -> ADSR amplitude
// envelope. GenerativeComposer's polyphonic pad pool triggers one of these per generated note;
// PositionalSource's ember generator retriggers a single instance per crackle transient. Trigger()
// re-arms the voice with a new frequency/timbre; Release() begins the envelope's Release stage;
// IsActive() reports false once that release has fully decayed to silence, at which point the
// owning pool is free to retarget this voice at a new note/transient.
class SynthVoice {
public:
    struct VoiceParams {
        Waveform waveform = Waveform::Sine;
        ADSREnvelope::Params envelope;
        float filterCutoffHz = 4000.0f;
        float filterResonanceQ = 0.707f;
        float gain = 1.0f; // Per-voice linear gain, e.g. to vary loudness across chord tones.
    };

    explicit SynthVoice(uint32_t noiseSeed = 0);

    void Trigger(float frequencyHz, const VoiceParams& params, float sampleRateHz);
    void Release();
    bool IsActive() const;
    float NextSample(float sampleRateHz);

private:
    Oscillator m_Oscillator;
    ResonantLowPassFilter m_Filter;
    ADSREnvelope m_Envelope;
    float m_Gain = 1.0f;
};

// Continuous filtered-noise texture generator: white noise through a resonant low-pass, with an
// optional slow secondary LFO that modulates output AMPLITUDE (a cheap post-filter tremolo, not a
// per-sample filter-coefficient recompute) for a "breathing"/gusting quality instead of a static
// hiss. Unlike SynthVoice above this has no envelope/note concept -- it always produces output,
// matching what a continuous environmental ambience (wind, waterfall, fire rumble) needs.
class NoiseBed {
public:
    explicit NoiseBed(uint32_t seed);

    void SetFilter(float cutoffHz, float resonanceQ, float sampleRateHz);
    void SetLFO(float lfoRateHz, float lfoDepth01);
    float NextSample(float sampleRateHz);

private:
    Xorshift32 m_Noise;
    ResonantLowPassFilter m_Filter;
    float m_LFORateHz = 0.1f;
    float m_LFODepth = 0.0f;
    float m_LFOPhase = 0.0f;
};

// Equal-power pan law: given pan in [-1 (hard left), 1 (hard right)], returns the L/R gain pair
// that sums to constant perceived loudness across the stereo field (sin/cos quarter-circle, the
// standard mixing-console pan law -- NOT a naive linear crossfade, which dips in perceived volume
// at center). Shared by GenerativeComposer (fixed per-voice-slot stereo spread across its pad pool)
// and AudioEngine (3D positional pan fed into IXAudio2SourceVoice::SetOutputMatrix).
void ComputeEqualPowerPan(float pan, float& outLeft, float& outRight);

} // namespace audio
