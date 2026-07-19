#ifndef HASH_UTILS_GLSL
#define HASH_UTILS_GLSL

// Cleanup pass (A5): single canonical copy of the Wang-style 32-bit integer hash that was
// previously duplicated byte-for-byte in src/shaders/src/Streaming/ProceduralMaskGenerate.comp and
// src/shaders/src/Renderer/ParticleSimulation.comp. Deliberately its own tiny, binding-free header
// (matching cluster_limits.glsl's own "split out so shaders that only need this don't pull in
// unrelated deps" precedent) rather than folded into math_utils.glsl (which pulls in
// GL_EXT_shader_explicit_arithmetic_types_int64 for its own unrelated 64-bit hash helpers) or
// pcg_common.glsl (a different hash family entirely -- PcgHash32's Jarzynski & Olano "pcg_hash"
// construction, not this Wang-style xorshift-multiply mix -- used for the deterministic CPU/GPU
// point-generation contract, not a drop-in equivalent for this one).
uint HashUint(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

#endif // HASH_UTILS_GLSL
