#ifndef PCG_COMMON_GLSL
#define PCG_COMMON_GLSL

// PCG framework roadmap, Phase 1 ("PCG Data Model"): shared GPU-side declarations for the future
// PCG points SSBO plus the deterministic hash-based RNG every future GPU-side PCG graph node
// (samplers/filters/spawners, later phases) must use for reproducible randomness. This phase does
// not yet dispatch any compute shader that includes this file -- it establishes the shared struct/
// function contract the CPU side (src/pcg/PcgPointData.h, src/pcg/PcgSeededRandom.h) already
// mirrors, ready for a later phase's actual GPU point-generation dispatch to include and use
// unmodified, matching this codebase's own "declare the shared contract before the first consumer
// exists" precedent (see e.g. ParticleCommon.glsl's own header comment, written the same way for
// Subtask 1 of the particle-system integration plan before Subtask 2's compute shader existed).

// ------------------------------------------------------------------------------------------------
// PcgGpuPoint: mirrors src/pcg/PcgPointData.h's `GpuPcgPoint` byte-for-byte -- 96 bytes, std430.
// Every vec3 below is immediately followed by a scalar that packs into the SAME 16-byte std430
// slot (vec3 members are 16-byte aligned in std430; a trailing scalar re-uses the 4 bytes std430
// would otherwise waste as padding) -- the identical "vec3 + trailing scalar" idiom this codebase's
// ParticleCommon.glsl already established for its own `Particle` struct.
// ------------------------------------------------------------------------------------------------
struct PcgGpuPoint {
    vec3 position;
    float density;      // Packs into position's 16-byte slot. Intrinsic density BEFORE steepness falloff -- see PcgComputeDensityFalloff/PcgGetEffectiveDensity below.
    vec4 rotation;       // Quaternion (x, y, z, w).
    vec3 scale;
    float steepness;     // Packs into scale's 16-byte slot. [0,1] -- see PcgComputeDensityFalloff's own comment for the exact semantics.
    vec4 color;           // RGBA tint.
    vec3 boundsMin;        // Local-space AABB min corner.
    uint seed;              // Packs into boundsMin's 16-byte slot. Deterministic per-point seed (feed into PcgHashCombine/PcgRandomFloat01 below).
    vec3 boundsMax;         // Local-space AABB max corner.
    float _pad0;             // Packs into boundsMax's 16-byte slot. Reserved for future per-point authoring data (e.g. a future custom-attribute-set index).
};

// ------------------------------------------------------------------------------------------------
// Deterministic hash-based RNG -- byte-for-byte port of src/pcg/PcgSeededRandom.h's CPU functions
// of the same name (that header's own top comment explains the full algorithm-choice rationale:
// "pcg_hash" from Jarzynski & Olano, "Hash Functions for GPU Rendering", JCGT 2020, used as a
// stateless one-shot hash rather than a running stream-PRNG permutation, so any GPU thread can
// compute any Nth value in O(1) with zero shared state). Every operation here is pure 32-bit
// unsigned integer multiply/shift/xor (GLSL `uint` arithmetic is defined modulo 2^32, exactly like
// C++'s uint32_t), so this function computes the IDENTICAL sequence of operations on the IDENTICAL
// integer inputs as the CPU-side PcgHash32/PcgHashCombine/PcgRandomFloat01 -- this is what gives
// CPU (bake-time) and GPU (runtime) bit-identical point generation for a shared seed, satisfying
// this project's "the show must play back identically every run" determinism requirement.
// ------------------------------------------------------------------------------------------------
uint PcgHash32(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Mirrors PcgSeededRandom.h's PcgHashCombine exactly -- same golden-ratio (0x9E3779B9) /
// MurmurHash3-finalizer (0x85EBCA6B) spreading constants, same combine order, so an identical
// (seed, index) pair hashes to an identical value on both CPU and GPU.
uint PcgHashCombine(uint seed, uint index) {
    return PcgHash32(seed ^ (index * 0x9E3779B9u + 0x85EBCA6Bu));
}

// [0, 1) float from a 32-bit hash -- mirrors PcgSeededRandom.h's PcgRandomFloat01 exactly (same
// exact power-of-two divide-by-2^32 constant, which has zero rounding ambiguity in IEEE-754
// single precision on either platform).
float PcgRandomFloat01(uint seed, uint index) {
    return float(PcgHashCombine(seed, index)) * (1.0 / 4294967296.0);
}

// ------------------------------------------------------------------------------------------------
// Steepness-driven density falloff -- mirrors src/pcg/PcgPointData.h's PcgPoint::ComputeEdgeFactor
// / ComputeDensityFalloff / GetEffectiveDensity exactly (see that header's own `steepness` field
// comment for the full semantics being implemented: steepness=0 is a soft gradient spanning the
// point's entire local-space bounds, steepness=1 is a hard box that only falls off in a thin band
// right at the edge). `localPos` must already be in the point's own LOCAL space (i.e. relative to
// its untransformed bounds, the same convention PcgPoint::ComputeEdgeFactor uses).
// ------------------------------------------------------------------------------------------------
float PcgComputeEdgeFactor(vec3 localPos, vec3 boundsMin, vec3 boundsMax) {
    vec3 halfExtent = (boundsMax - boundsMin) * 0.5;
    vec3 center = (boundsMin + boundsMax) * 0.5;
    vec3 offset = localPos - center;

    const float kMinExtent = 1.0e-6; // Guards a degenerate (zero-width) bounds axis from a divide-by-zero.
    float ex = abs(offset.x) / max(halfExtent.x, kMinExtent);
    float ey = abs(offset.y) / max(halfExtent.y, kMinExtent);
    float ez = abs(offset.z) / max(halfExtent.z, kMinExtent);
    return max(ex, max(ey, ez));
}

float PcgComputeDensityFalloff(vec3 localPos, vec3 boundsMin, vec3 boundsMax, float steepness) {
    float edgeFactor = PcgComputeEdgeFactor(localPos, boundsMin, boundsMax);
    float threshold = clamp(steepness, 0.0, 1.0);
    if (edgeFactor <= threshold) {
        return 1.0;
    }
    const float kMinDenom = 1.0e-6; // Guards threshold==1 (denom would otherwise be exactly 0) from a divide-by-zero.
    float denom = max(1.0 - threshold, kMinDenom);
    float falloff = 1.0 - (edgeFactor - threshold) / denom;
    return clamp(falloff, 0.0, 1.0);
}

float PcgGetEffectiveDensity(vec3 localPos, vec3 boundsMin, vec3 boundsMax, float steepness, float density) {
    return clamp(density, 0.0, 1.0) * PcgComputeDensityFalloff(localPos, boundsMin, boundsMax, steepness);
}

#endif // PCG_COMMON_GLSL
