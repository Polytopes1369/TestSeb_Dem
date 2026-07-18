#include "GenerativeComposer.h"
#include "core/EngineConfig.h"
#include <algorithm>
#include <cmath>

namespace audio {

namespace {
    constexpr int kRootMidiNote = 45; // A2 (~110 Hz) -- see GenerativeComposer.h's own scale comment.
    constexpr int kScaleIntervals[5] = { 0, 3, 5, 7, 10 }; // A minor pentatonic, semitones from root.

    // Fixed 4-chord progression (i - VI - III - VII), one chord per 16-step bar, expressed as
    // INDICES into kScaleIntervals (not raw semitones) -- see GenerativeComposer.h's own comment.
    constexpr int kChordProgression[4][3] = {
        { 0, 2, 4 },
        { 3, 0, 2 },
        { 2, 4, 1 },
        { 4, 1, 3 },
    };

    float MidiNoteToFrequencyHz(int midiNote) {
        return 440.0f * std::pow(2.0f, (static_cast<float>(midiNote) - 69.0f) / 12.0f);
    }

    // Fixed per-voice-slot stereo placement (equal-power pan law) -- gives the pad pool natural
    // stereo width without any real positional 3D placement (this bed is intentionally non-
    // positional, see GenerativeComposer.h's own class comment).
    constexpr float kVoicePanPositions[GenerativeComposer::kMaxPolyphony] = {
        -0.7f, -0.35f, -0.1f, 0.1f, 0.35f, 0.7f
    };
} // namespace

GenerativeComposer::GenerativeComposer(uint32_t seed) : m_Random(seed) {
    m_VoiceNoteOffTime.fill(-1.0);
    // m_Voices is default-constructed (each SynthVoice's own Oscillator seeded 0, see Synth.h) --
    // harmless: TriggerNote() below only ever selects Waveform::Sine/Triangle for pad notes, never
    // Waveform::Noise, so each voice's internal noise generator is never actually consulted.
}

void GenerativeComposer::TriggerNote(float sampleRateHz) {
    // Find a free (inactive) voice slot; if the whole pool is busy (rare -- kMaxPolyphony=6 against
    // a sparse, long-release ambient texture), simply skip this step's note rather than stealing/
    // cutting off a still-ringing voice, which would produce an audible click. An occasionally
    // missed trigger is inaudible in this generator's own sparse, evolving texture.
    uint32_t slot = kMaxPolyphony;
    for (uint32_t v = 0; v < kMaxPolyphony; ++v) {
        if (!m_Voices[v].IsActive()) { slot = v; break; }
    }
    if (slot == kMaxPolyphony) return;

    const int* chordDegrees = kChordProgression[m_ChordIndex];
    uint32_t degreePick = static_cast<uint32_t>(m_Random.NextUnipolar() * 3.0f);
    if (degreePick > 2u) degreePick = 2u; // Defensive clamp against the [0,1) upper-edge case.
    int scaleDegreeIndex = chordDegrees[degreePick];

    // Octave offset weighted toward 0 (stay near the base register most of the time, occasionally
    // drop or lift an octave for variety).
    float octaveRoll = m_Random.NextUnipolar();
    int octaveOffset = (octaveRoll < 0.15f) ? -1 : (octaveRoll < 0.85f ? 0 : 1);

    // +24 semitones (two octaves) lifts the pentatonic root (A2) into a pleasant mid register
    // (effectively centered around A4) for an ambient pad texture -- a documented aesthetic choice,
    // not a derived constant.
    int midiNote = kRootMidiNote + kScaleIntervals[scaleDegreeIndex] + octaveOffset * 12 + 24;
    float freqHz = MidiNoteToFrequencyHz(midiNote);

    SynthVoice::VoiceParams params;
    params.waveform = (m_Random.NextUnipolar() < 0.5f) ? Waveform::Sine : Waveform::Triangle;
    params.envelope.attackSeconds = 0.8f + m_Random.NextUnipolar() * 1.5f;    // 0.8 - 2.3s soft swell.
    params.envelope.decaySeconds = 0.5f + m_Random.NextUnipolar() * 1.0f;
    params.envelope.sustainLevel = 0.55f + m_Random.NextUnipolar() * 0.2f;
    params.envelope.releaseSeconds = 1.5f + m_Random.NextUnipolar() * 2.5f;   // Long tail -- notes overlap.
    params.filterCutoffHz = 900.0f + m_Random.NextUnipolar() * 2200.0f;
    params.filterResonanceQ = 0.6f + m_Random.NextUnipolar() * 0.4f;
    params.gain = 0.5f + m_Random.NextUnipolar() * 0.2f;

    m_Voices[slot].Trigger(freqHz, params, sampleRateHz);

    // Hold duration is independent of the sequencer's own step grid -- pad notes deliberately
    // overlap several steps/bars (3-9 simulated seconds), giving the "evolving drone" texture
    // rather than one-note-per-step staccato playback.
    double holdSeconds = 3.0 + static_cast<double>(m_Random.NextUnipolar()) * 6.0;
    m_VoiceNoteOffTime[slot] = m_ClockSeconds + holdSeconds;
}

void GenerativeComposer::AdvanceStep(float sampleRateHz) {
    float bpm = std::max(config::audio::GENERATIVE_TEMPO_BPM, 1.0f);
    double secondsPerBeat = 60.0 / static_cast<double>(bpm);
    double secondsPerStep = secondsPerBeat / 4.0; // 16th-note subdivision.
    m_NextStepTimeSeconds = m_ClockSeconds + secondsPerStep;

    float density = std::clamp(config::audio::GENERATIVE_NOTE_DENSITY, 0.0f, 1.0f);
    if (m_Random.NextUnipolar() < density) {
        TriggerNote(sampleRateHz);
    }

    m_StepIndex = (m_StepIndex + 1) % 16;
    if (m_StepIndex == 0) {
        // Crossed a bar boundary (16 steps) -- advance to the next chord in the progression for
        // the upcoming bar's own 16 steps.
        m_ChordIndex = (m_ChordIndex + 1) % 4;
    }
}

void GenerativeComposer::RenderBlock(float* outStereoInterleaved, uint32_t frameCount, float sampleRateHz) {
    constexpr float kMixHeadroom = 0.6f; // Guards against clipping when several pad voices overlap.

    for (uint32_t i = 0; i < frameCount; ++i) {
        m_ClockSeconds += 1.0 / static_cast<double>(sampleRateHz);

        // Only trigger NEW notes while the feature is enabled; already-sounding notes still ring
        // out through their own release tail below, giving a graceful fade rather than a hard cut
        // when config::audio::GENERATIVE_MUSIC_ENABLED is toggled off live.
        if (config::audio::GENERATIVE_MUSIC_ENABLED && m_ClockSeconds >= m_NextStepTimeSeconds) {
            AdvanceStep(sampleRateHz);
        }

        for (uint32_t v = 0; v < kMaxPolyphony; ++v) {
            if (m_VoiceNoteOffTime[v] >= 0.0 && m_ClockSeconds >= m_VoiceNoteOffTime[v]) {
                m_Voices[v].Release();
                m_VoiceNoteOffTime[v] = -1.0;
            }
        }

        float left = 0.0f, right = 0.0f;
        for (uint32_t v = 0; v < kMaxPolyphony; ++v) {
            if (!m_Voices[v].IsActive()) continue;
            float sample = m_Voices[v].NextSample(sampleRateHz);
            float leftGain, rightGain;
            ComputeEqualPowerPan(kVoicePanPositions[v], leftGain, rightGain);
            left += sample * leftGain;
            right += sample * rightGain;
        }

        outStereoInterleaved[i * 2u + 0u] = std::clamp(left * kMixHeadroom, -1.0f, 1.0f);
        outStereoInterleaved[i * 2u + 1u] = std::clamp(right * kMixHeadroom, -1.0f, 1.0f);
    }
}

uint32_t GenerativeComposer::GetActiveVoiceCount() const {
    uint32_t count = 0;
    for (const SynthVoice& voice : m_Voices) {
        if (voice.IsActive()) ++count;
    }
    return count;
}

} // namespace audio
