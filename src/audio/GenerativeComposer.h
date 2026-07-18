#pragma once
// English comments only.
//
// The "FL Studio style" generative composition layer: NOT a DAW UI (there is none here), but an
// algorithmic step-sequencer/probabilistic note-generator that procedurally produces an evolving
// ambient/melodic score in real time, using SynthVoice (Synth.h) as its instrument. This is a
// non-positional, always-audible 2-channel bed -- AudioEngine mixes it independently of the 3D
// positional environmental sources (PositionalSource.h).
//
// Musical design (a documented judgment call, per this feature's own task brief -- "your call,
// document which you chose and why"):
//   - Scale: A minor PENTATONIC (intervals {0,3,5,7,10} semitones from the root), root = A2
//     (MIDI 45, ~110 Hz). Pentatonic scales have no interval that ever sounds dissonant against
//     any other member of the scale, so ANY random walk across scale degrees stays musically safe
//     -- exactly the property a fully-procedural, unsupervised generator needs (no voice-leading
//     rules, no dissonance-avoidance logic required to still sound intentional).
//   - Chord progression: a fixed 4-chord cycle (i - VI - III - VII in scale-degree terms), one
//     chord per 16-step bar, looping -- gives the piece a recognizable harmonic "shape" over time
//     instead of a pure random walk, while each individual note still comes from the current
//     chord's scale degrees (weighted) or a passing tone (unweighted scale degree), picked live.
//   - Rhythm: sparse, probabilistic note triggering (config::audio::GENERATIVE_NOTE_DENSITY gates
//     each 16th-note step) with long attack/release pad envelopes -- an Eno "Music for Airports"-
//     style overlapping-drone texture, not a rhythmic melody. This both suits an ambient demoscene
//     backdrop and sidesteps the much harder problem of generating a genuinely satisfying rhythmic
//     melody procedurally.
//   - Determinism: seeded from config::audio::GENERATIVE_SEED (audio::Xorshift32, see
//     Oscillator.h) -- matches this codebase's own established "same seed -> same output"
//     procedural-generation discipline (terrain, clusters, HLOD, ...) rather than an
//     unrepeatable live-evolving generator. The piece still never LOOPS in the traditional sense
//     (it is not a fixed-length sequence played back), it is a continuously-advancing generative
//     process -- reseeding only replays the exact same evolving process from its own start, it
//     does not introduce a discrete "chorus"/repeat.

#include "Synth.h"
#include <cstdint>
#include <array>

namespace audio {

class GenerativeComposer {
public:
    // kMaxPolyphony: simultaneously-sounding pad notes. 6 is enough headroom for a 2-3 note chord
    // plus 1-2 lingering long-release tails from the previous chord without ever needing voice
    // stealing (see TriggerNote()'s own comment on what happens if the pool is briefly exhausted).
    static constexpr uint32_t kMaxPolyphony = 6;

    explicit GenerativeComposer(uint32_t seed);

    // Advances the internal sample-accurate sequencer clock by frameCount samples, triggering/
    // releasing pad notes as step/bar boundaries are crossed, and additively mixes every active
    // voice into outStereoInterleaved (L,R,L,R,... already zero-initialized is NOT assumed -- this
    // OVERWRITES, it does not accumulate onto existing content, matching AudioEngine's own "render
    // straight into this ring-buffer slot" call convention). Reads config::audio::
    // GENERATIVE_MUSIC_ENABLED/GENERATIVE_NOTE_DENSITY/GENERATIVE_TEMPO_BPM live (every call), so
    // ImGui slider changes are audible within one call's worth of samples -- i.e. one buffer-refill
    // period, not only after a full pattern restart.
    void RenderBlock(float* outStereoInterleaved, uint32_t frameCount, float sampleRateHz);

    // Debug/diagnostic read-back (ImGui "Audio" tab) -- purely informational, never fed back into
    // RenderBlock's own logic.
    uint32_t GetCurrentStepIndex() const { return m_StepIndex; }
    uint32_t GetCurrentChordIndex() const { return m_ChordIndex; }
    uint32_t GetActiveVoiceCount() const;

private:
    void AdvanceStep(float sampleRateHz);
    void TriggerNote(float sampleRateHz);

    std::array<SynthVoice, kMaxPolyphony> m_Voices;
    // Wall-clock (well, simulation-clock -- see RenderBlock's own comment) seconds each currently-
    // sounding voice slot should have Release() called, indexed 1:1 with m_Voices; NaN/negative
    // sentinel (< 0) means "slot not tracked" (either never triggered or already released).
    std::array<double, kMaxPolyphony> m_VoiceNoteOffTime;

    Xorshift32 m_Random;

    double m_ClockSeconds = 0.0;      // Simulation-time accumulator RenderBlock advances by frameCount/sampleRateHz each call.
    double m_NextStepTimeSeconds = 0.0;
    uint32_t m_StepIndex = 0;         // [0, 16) within the current bar.
    uint32_t m_ChordIndex = 0;        // [0, 4) into the fixed progression, advances once per 16-step bar.
};

} // namespace audio
