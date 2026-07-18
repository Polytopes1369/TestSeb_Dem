#pragma once

// PCG framework roadmap, Phase 1 ("PCG Data Model"): deterministic, hash-based per-seed random
// number stream. Every future PCG node (samplers/filters/spawners, later phases) that needs
// randomness -- jittered spawn position, random rotation, per-point variation -- must derive it
// from a PcgSeededRandom constructed from that point/node's own seed, NEVER from a global
// std::rand()/std::mt19937 instance: a demoscene "show" must play back bit-identically every run,
// so bake-time (offline tooling) and runtime (in-engine) point generation must reproduce the exact
// same sequence given the exact same seed.
//
// Algorithm choice: a STATELESS "hash(seed, index) -> value" construction, not a stateful stream
// PRNG (e.g. the classic PCG32 64-bit LCG + xorshift permutation this file's name otherwise
// evokes). This is deliberate: a stateful stream's Nth output depends on sequentially replaying
// N-1 prior steps, which is trivial to reproduce on the CPU but awkward/expensive to reproduce
// exactly on the GPU (a compute-shader thread wanting the Nth value for point #12345 would
// otherwise need to loop 12345 times just to get there, or the CPU would need to upload the whole
// running-state history). A stateless hash keyed directly by (seed, streamIndex) lets ANY consumer
// -- a CPU stream object below, or an independent GPU thread -- compute any Nth value in O(1) with
// zero shared state, which is exactly what a future GPU-driven graph-execution model (later PCG
// roadmap phases) needs, and it also sidesteps 64-bit-integer-arithmetic availability concerns in
// core GLSL (this hash uses ONLY 32-bit unsigned multiply/shift/xor).
//
// The one-shot mixing function itself is "pcg_hash" from Mark Jarzynski & Marc Olano, "Hash
// Functions for GPU Rendering" (Journal of Computer Graphics Techniques, 2020) -- a widely used,
// well-analyzed, single-pass 32-bit avalanche mix from the same PCG family this class borrows its
// name from (used here as a one-shot hash rather than as a running generator's permutation step).
// pcg_common.glsl's PcgHash32/PcgHashCombine/PcgRandomFloat01 are a byte-for-byte port of the
// functions in this file -- both sides perform the identical sequence of 32-bit integer operations
// on the identical integer inputs, which is what gives CPU and GPU bit-identical results for a
// shared seed (the only place approximate cross-platform float behavior could sneak in is the
// final uint32->float divide-by-2^32 conversion; see PcgRandomFloat01's own comment below for why
// that step is safe too).

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "core/maths/Maths.h"

namespace pcg {

    // Single-pass 32-bit avalanche hash (Jarzynski & Olano's "pcg_hash", see this file's own header
    // comment). Pure 32-bit unsigned integer multiply/shift/xor -- unsigned integer arithmetic in
    // C++ is well-defined modulo 2^32 (same wraparound semantics as GLSL's `uint`), so this
    // function's result is bit-identical to pcg_common.glsl's PcgHash32 for the same input on any
    // conforming C++ compiler and any conforming GLSL compiler.
    inline uint32_t PcgHash32(uint32_t v) {
        uint32_t state = v * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    // Combines a base seed with a stream index into one 32-bit hash input before mixing, so that
    // e.g. PcgHashCombine(seedA, 0) and PcgHashCombine(seedB, 0) do not collide just because index
    // 0 is common to both streams. The multiply/xor constants are the standard Knuth/Fibonacci
    // golden-ratio (0x9E3779B9) and a MurmurHash3-finalizer constant (0x85EBCA6B), used here only to
    // spread the two inputs apart before PcgHash32's own avalanche mix does the real work -- they
    // are not claimed to be a hash in their own right.
    inline uint32_t PcgHashCombine(uint32_t seed, uint32_t index) {
        return PcgHash32(seed ^ (index * 0x9E3779B9u + 0x85EBCA6Bu));
    }

    // [0, 1) float from a 32-bit hash, via the standard "divide by 2^32" construction. 1/2^32 has
    // an EXACT float32 representation (a power of two), so this performs exactly one IEEE-754
    // single-precision multiply with no intermediate rounding ambiguity -- the matching GLSL
    // PcgRandomFloat01 in pcg_common.glsl performs the identical operation (uint -> float convert,
    // multiply by the identical exact constant), which is what makes this safe to rely on for
    // CPU/GPU bit-identical output despite running on totally different hardware.
    inline float PcgRandomFloat01(uint32_t seed, uint32_t index) {
        return static_cast<float>(PcgHashCombine(seed, index)) * (1.0f / 4294967296.0f);
    }

    // A per-seed logical "stream": wraps PcgHashCombine/PcgRandomFloat01 behind a familiar
    // stateful-looking Next*() API (an internal monotonically increasing index counter) for CPU
    // call sites that want plain "construct once, call Next() repeatedly" stream-PRNG ergonomics.
    // Every individual Next() call is still just one O(1) stateless hash lookup (see PcgHash32's
    // own comment) -- m_Index is purely a bookkeeping convenience, not part of the actual random
    // state, so a GPU thread that independently computes PcgRandomFloat01(seed, 7) gets the exact
    // same value this stream's 8th NextFloat01() call would (0-based index), with no need to
    // replicate m_Index's incrementing on the GPU side.
    class PcgSeededRandom {
    public:
        explicit PcgSeededRandom(uint32_t seed) : m_Seed(seed), m_Index(0) {}

        uint32_t Seed() const { return m_Seed; }
        uint32_t IndexConsumed() const { return m_Index; }

        // Re-seeds this stream and resets its index counter to 0. Calling Reset() with the same
        // seed the stream was originally constructed with makes the NEXT Next*() call sequence
        // replay bit-identically from the start -- exactly the "seed twice, compare" determinism
        // check this phase's smoke test (PcgDataModelSmokeTest.cpp) performs.
        void Reset(uint32_t seed) {
            m_Seed = seed;
            m_Index = 0;
        }

        uint32_t NextUint32() {
            return PcgHashCombine(m_Seed, m_Index++);
        }

        float NextFloat01() {
            return PcgRandomFloat01(m_Seed, m_Index++);
        }

        float NextFloatRange(float lo, float hi) {
            return lo + NextFloat01() * (hi - lo);
        }

        // Inclusive on both ends: returns a value in [lo, hiInclusive]. Defensively clamps the
        // rounded offset into range in case NextFloat01()'s (extremely rare, but not impossible
        // given float rounding) exact 1.0 upper edge would otherwise push the result one past
        // hiInclusive.
        int32_t NextIntRange(int32_t lo, int32_t hiInclusive) {
            float t = NextFloat01();
            int32_t span = hiInclusive - lo + 1;
            int32_t offset = static_cast<int32_t>(t * static_cast<float>(span));
            offset = std::clamp(offset, 0, span - 1);
            return lo + offset;
        }

        // Uniform-random unit-length direction vector via the standard z/phi (Marsaglia-style)
        // sphere parametrization, using two hashed floats. A small, generic RNG primitive (like
        // NextFloatRange), not a spawner/sampler in its own right -- kept here because "give me a
        // random direction" is exactly the kind of basic stream operation a hash-based RNG utility
        // is expected to provide, the same way std::uniform_real_distribution would.
        maths::vec3 NextUnitVec3() {
            float z = NextFloatRange(-1.0f, 1.0f);
            float phi = NextFloatRange(0.0f, 2.0f * maths::PI);
            float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
            return maths::vec3{ r * std::cos(phi), r * std::sin(phi), z };
        }

    private:
        uint32_t m_Seed;
        uint32_t m_Index;
    };

}
