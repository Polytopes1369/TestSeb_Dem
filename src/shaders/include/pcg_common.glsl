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

// ------------------------------------------------------------------------------------------------
// PCG framework roadmap, Phase 5.3 ("GPU-Resident Node Execution"): a small, deterministic 3D value
// noise, added here (rather than reusing AtmosNoiseCommon.glsl's existing curl/fractal noise, which
// is tuned for wind-field turbulence, a different visual target) specifically for the reference GPU
// PCG node type this phase proves the execute-callback registration mechanism with
// (PcgDensityNoise.comp / src/pcg/PcgGpuDensityNoiseNode.h, "pcg.gpu.density_noise"). No CPU-side
// PCG noise/filter node existed anywhere in this codebase at the time this was written (Phase 3's
// CPU filters, one of which is a density/noise transform, are a separate, concurrently-developed
// phase) -- this is therefore an INDEPENDENT implementation, not a GPU port of an existing CPU
// algorithm, built entirely from PcgHash32 above so it stays within this file's existing "one
// deterministic hash family" contract rather than introducing a second one. If/when Phase 3 lands
// its own CPU-side density-noise filter, a future cleanup pass should reconcile the two rather than
// carrying two independently-tuned noise functions forward indefinitely -- flagged here so that
// duplication is not accidentally overlooked.
// ------------------------------------------------------------------------------------------------

// Hashes one integer lattice cell corner to a [0, 1) scalar. Three axis coordinates are spread
// apart with distinct large odd multipliers (a standard "spatial hash" idiom) before being folded
// together with XOR and run through PcgHash32's own avalanche mix -- deterministic and collision-
// resistant enough for a visual noise field (this is explicitly NOT claimed to be a cryptographic
// or statistically rigorous hash, only "looks like noise, is 100% reproducible for the same input").
// `cell` uses signed ints (a lattice coordinate can legitimately be negative for a world-space
// position near/behind the origin); the int->uint cast below is a well-defined two's-complement
// bit-pattern reinterpretation in GLSL, exactly like C++'s equivalent cast, so negative cells hash
// just as deterministically as positive ones.
float PcgLatticeHash01(ivec3 cell) {
    uint h = PcgHash32(uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u);
    return float(h) * (1.0 / 4294967296.0);
}

// Trilinearly-interpolated 3D value noise over `p` (world-space, already pre-scaled by whatever
// frequency the caller wants -- this function itself has no notion of "frequency", it always
// samples at unit lattice spacing). Uses a smoothstep-style (3t^2 - 2t^3) fade curve on the
// fractional lattice position, matching Perlin's own "improved" interpolant, so the result has a
// continuous first derivative across lattice cell boundaries (a plain linear fade would produce
// visible faceting/creasing at cell edges). Returns a value in [0, 1); callers wanting a
// signed/centered noise (e.g. PcgDensityNoise.comp's own density offset) remap it themselves
// (`noise * 2.0 - 1.0`) rather than baking that convention into this shared helper.
float PcgValueNoise3D(vec3 p) {
    vec3 cellOrigin = floor(p);
    vec3 f = p - cellOrigin;
    vec3 u = f * f * (3.0 - 2.0 * f);
    ivec3 i = ivec3(cellOrigin);

    float n000 = PcgLatticeHash01(i + ivec3(0, 0, 0));
    float n100 = PcgLatticeHash01(i + ivec3(1, 0, 0));
    float n010 = PcgLatticeHash01(i + ivec3(0, 1, 0));
    float n110 = PcgLatticeHash01(i + ivec3(1, 1, 0));
    float n001 = PcgLatticeHash01(i + ivec3(0, 0, 1));
    float n101 = PcgLatticeHash01(i + ivec3(1, 0, 1));
    float n011 = PcgLatticeHash01(i + ivec3(0, 1, 1));
    float n111 = PcgLatticeHash01(i + ivec3(1, 1, 1));

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

#endif // PCG_COMMON_GLSL
