#pragma once
// English comments only.
//
// World-space sound source with distance attenuation + stereo panning relative to the camera --
// the "3D positional mixer" half of this engine (see AudioEngine.h's own class comment for how
// this fits into the whole subsystem). Three kinds are wired to real, existing scene elements (see
// AudioEngine.cpp's construction site for exactly which):
//
//  - Embers: continuous low rumble bed + randomized bright noise-burst "crackle" transients,
//    positioned at config::particles::EMITTERS[0]'s own world position (the existing "Embers"
//    particle emitter).
//  - Waterfall: continuous filtered-noise rush, positioned at config::particles::EMITTERS[3]'s own
//    world position (the existing waterfall mist/foam particle emitter, itself kept in sync with
//    river_spline.glsl's kRiverControlXZ[3]/kRiverControlHeight[3] -- see that EmitterConfig's own
//    comment in EngineConfig.h).
//  - Wind: continuous filtered-noise "whoosh" whose cutoff/amplitude track
//    config::atmos::WIND_SPEED_MPS. Wind has no single real-world emission point, so -- a
//    documented judgment call -- this source's world position is NOT fixed: Update() re-anchors it
//    every frame a fixed distance from the CAMERA along the current
//    config::atmos::WIND_DIRECTION_DEGREES compass bearing, so its stereo pan genuinely tracks
//    "which way the wind is coming from" as the player turns, instead of sitting at an arbitrary
//    fixed point in space.
//
// Panning/attenuation is plain, fully-owned vector math (dot product of the source direction
// against a reconstructed camera-right vector, OpenAL-style inverse-clamped-distance gain) rather
// than X3DAudio/HRTF spatialization -- this engine's positional model is deliberately
// "distance attenuation + stereo pan", matching the task brief's own description, not full binaural
// 3D audio. RenderBlock() produces the RAW (pre-pan, pre-distance-gain) mono signal; AudioEngine
// applies GetPan()/GetDistanceAttenuationGain() via IXAudio2SourceVoice::SetOutputMatrix/SetVolume
// every frame independently of buffer refills, so pan/attenuation updates are smooth and immediate
// even though the underlying synthesized audio is only regenerated every ~10ms block.

#include "Synth.h"
#include "core/maths/Maths.h"
#include "core/Camera.h" // CameraFrameInfo

namespace audio {

class PositionalSource {
public:
    enum class Kind {
        Embers,
        Waterfall,
        Wind,
    };

    // fixedWorldPosition is ignored for Kind::Wind (see class comment -- Update() recomputes its
    // own position every frame instead).
    PositionalSource(Kind kind, uint32_t seed, const maths::vec3& fixedWorldPosition);

    // Recomputes this source's live DSP parameters (Wind's filter/LFO from current wind speed),
    // world position (Wind only), and this frame's pan/distance-attenuation gain against the
    // camera. Also drives Embers' randomized crackle-transient scheduling. Must be called once per
    // frame BEFORE RenderBlock() and before AudioEngine reads GetPan()/
    // GetDistanceAttenuationGain() for this frame's SetOutputMatrix/SetVolume calls.
    void Update(float dt, const CameraFrameInfo& camera, float sampleRateHz);

    // Fills outMono[0, frameCount) with this source's raw synthesized signal, in [-1, 1] --
    // NOT yet scaled by distance attenuation or panned (see class comment for why that split
    // exists).
    void RenderBlock(float* outMono, uint32_t frameCount, float sampleRateHz);

    Kind GetKind() const { return m_Kind; }
    const maths::vec3& GetWorldPosition() const { return m_WorldPosition; }
    float GetPan() const { return m_Pan; }                                     // [-1, 1].
    float GetDistanceAttenuationGain() const { return m_DistanceAttenuationGain; } // [0, 1].

private:
    float ComputeDistanceAttenuation(float distance) const;
    void TriggerCrackle(float sampleRateHz);

    Kind m_Kind;
    maths::vec3 m_WorldPosition;

    NoiseBed m_NoiseBed;
    float m_BaseGain = 1.0f; // Kind-specific intrinsic level (e.g. Wind's own speed-driven loudness), pre-attenuation.

    // Embers-only: one-shot crackle transient voice + its own scheduling countdown/PRNG. Harmless
    // to keep allocated (a few floats) even for the other two kinds, avoiding a Kind-specific
    // subclass hierarchy for what is otherwise near-identical logic.
    SynthVoice m_CrackleVoice;
    Xorshift32 m_Random;
    float m_TimeToNextCrackleSeconds = 0.3f;

    float m_Pan = 0.0f;
    float m_DistanceAttenuationGain = 1.0f;
};

} // namespace audio
