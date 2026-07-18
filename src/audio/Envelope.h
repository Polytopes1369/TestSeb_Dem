#pragma once
// English comments only.
//
// Classic 4-stage ADSR envelope generator (Attack/Decay/Sustain/Release), linear ramps -- the
// standard amplitude-shaping building block every synthesizer voice in this engine uses
// (GenerativeComposer's pad notes, PositionalSource's ember-crackle transients). Header-only for
// the same "hot, tiny, evaluated once per output sample" reason as Oscillator.h.

#include <algorithm>

namespace audio {

class ADSREnvelope {
public:
    struct Params {
        float attackSeconds = 0.05f;
        float decaySeconds = 0.2f;
        float sustainLevel = 0.6f;   // [0, 1]
        float releaseSeconds = 0.5f;
    };

    void SetParams(const Params& params) { m_Params = params; }

    void NoteOn() {
        // Deliberately does NOT reset m_Level to 0 -- the attack ramp starts from whatever level
        // the envelope is currently at, so retriggering a voice mid-release (this engine's
        // polyphonic pad pool and ember-crackle generator both do this opportunistically, see
        // GenerativeComposer::TriggerNote's own comment) never produces an audible discontinuity
        // click, only a shorter effective attack.
        m_Stage = Stage::Attack;
    }

    void NoteOff() {
        if (m_Stage == Stage::Idle) return;
        // Capture the level release should ramp down FROM, so the release ramp's rate scales
        // correctly regardless of which stage NoteOff() was called during (e.g. released during
        // Attack, before ever reaching 1.0) -- a fixed "always ramp from 1.0" release would either
        // audibly speed up or slow down depending on the interrupted stage's own level.
        m_ReleaseStartLevel = m_Level;
        m_Stage = Stage::Release;
    }

    bool IsActive() const { return m_Stage != Stage::Idle; }

    // Advances the envelope by one sample and returns the current amplitude multiplier, [0, 1].
    float NextSample(float sampleRateHz) {
        float dt = 1.0f / sampleRateHz;
        switch (m_Stage) {
        case Stage::Idle:
            m_Level = 0.0f;
            break;
        case Stage::Attack: {
            float rate = (m_Params.attackSeconds > 0.0f) ? (dt / m_Params.attackSeconds) : 1.0f;
            m_Level += rate;
            if (m_Level >= 1.0f) { m_Level = 1.0f; m_Stage = Stage::Decay; }
            break;
        }
        case Stage::Decay: {
            float rate = (m_Params.decaySeconds > 0.0f) ? (dt / m_Params.decaySeconds) : 1.0f;
            m_Level -= rate * (1.0f - m_Params.sustainLevel);
            if (m_Level <= m_Params.sustainLevel) {
                m_Level = m_Params.sustainLevel;
                // sustainLevel == 0 means this is a one-shot AD (Attack-Decay, no hold) envelope --
                // standard synth envelope-generator convention for percussive/transient hits
                // (PositionalSource's ember-crackle transient uses exactly this): once decay
                // reaches silence there is nothing left to sustain, so go straight to Idle instead
                // of parking in Sustain at amplitude 0 forever (which would leave IsActive() true
                // indefinitely for a voice producing no audible output).
                m_Stage = (m_Params.sustainLevel > 0.0f) ? Stage::Sustain : Stage::Idle;
            }
            break;
        }
        case Stage::Sustain:
            m_Level = m_Params.sustainLevel;
            break;
        case Stage::Release: {
            // Linear ramp from m_ReleaseStartLevel (captured in NoteOff() above) down to 0 over
            // releaseSeconds -- NOT from 1.0, see NoteOff()'s own comment.
            float rate = (m_Params.releaseSeconds > 0.0f) ? (dt / m_Params.releaseSeconds) : 1.0f;
            m_Level -= rate * m_ReleaseStartLevel;
            if (m_Level <= 0.0f) { m_Level = 0.0f; m_Stage = Stage::Idle; }
            break;
        }
        }
        return m_Level;
    }

    float GetCurrentLevel() const { return m_Level; }

private:
    enum class Stage { Idle, Attack, Decay, Sustain, Release };
    Stage m_Stage = Stage::Idle;
    Params m_Params;
    float m_Level = 0.0f;
    float m_ReleaseStartLevel = 1.0f;
};

} // namespace audio
