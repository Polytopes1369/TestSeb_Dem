#pragma once
// Bit-exact C++ mirror of src/shaders/src/Streaming/ProceduralMaskGenerate.comp's procedural cutout
// mask formula (two-octave hashed value noise, thresholded), so the CPU-side clusterizer can
// classify triangles as opaque/masked WITHOUT any GPU readback dependency -- ClusterPartitioner.cpp/
// ClusterDAG.cpp are deliberately Vulkan-free (unit-tested by device-less CTest executables), and
// using the same hash/noise math here guarantees the CPU classification and the GPU's actual
// runtime sample (mask_sampling.glsl's SampleMaskAlpha) can never drift apart -- same hash, same
// constants, same threshold. Every constant/operation below (0x7feb352du, 0x846ca68bu, the 1973/
// 9277/26699 mixing primes, the 6.0/13.0 octave frequencies, the 0.6/0.4 blend weights, the
// 0.35/0.55 smoothstep edges) is copied verbatim from that compute shader; changing one side without
// the other silently breaks the CPU/GPU classification guarantee.

#include <cstdint>
#include <algorithm>
#include <cmath>
#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"

namespace geometry {

    // Must stay in lockstep with ClusterRaster.frag's kMaskAlphaCutoff / ClusterSoftwareRaster.comp's
    // literal 0.5 alpha-test threshold.
    constexpr float kMaskAlphaCutoff = 0.5f;

    namespace detail {

        constexpr uint32_t HashUint(uint32_t x) {
            x ^= x >> 16u;
            x *= 0x7feb352du;
            x ^= x >> 15u;
            x *= 0x846ca68bu;
            x ^= x >> 16u;
            return x;
        }

        inline float SmoothStep(float edge0, float edge1, float x) {
            float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
            return t * t * (3.0f - 2.0f * t);
        }

        inline float Mix(float a, float b, float t) { return a + (b - a) * t; }

        // Value noise at integer cell corners, hashed by (cellX, cellY, slot), bilinearly
        // interpolated with a smoothstep-shaped blend -- mirrors ProceduralMaskGenerate.comp's
        // ValueNoise exactly, component-by-component since maths::vec2 has no floor()/component
        // division operator.
        inline float ValueNoiseCPU(float px, float py, uint32_t slot) {
            float cellX = std::floor(px);
            float cellY = std::floor(py);
            float fracX = px - cellX;
            float fracY = py - cellY;
            float smoothX = fracX * fracX * (3.0f - 2.0f * fracX);
            float smoothY = fracY * fracY * (3.0f - 2.0f * fracY);

            uint32_t ix = static_cast<uint32_t>(cellX);
            uint32_t iy = static_cast<uint32_t>(cellY);

            float h00 = static_cast<float>(HashUint(ix * 1973u + iy * 9277u + slot * 26699u) & 0xFFFFu) / 65535.0f;
            float h10 = static_cast<float>(HashUint((ix + 1u) * 1973u + iy * 9277u + slot * 26699u) & 0xFFFFu) / 65535.0f;
            float h01 = static_cast<float>(HashUint(ix * 1973u + (iy + 1u) * 9277u + slot * 26699u) & 0xFFFFu) / 65535.0f;
            float h11 = static_cast<float>(HashUint((ix + 1u) * 1973u + (iy + 1u) * 9277u + slot * 26699u) & 0xFFFFu) / 65535.0f;

            float top = Mix(h00, h10, smoothX);
            float bottom = Mix(h01, h11, smoothX);
            return Mix(top, bottom, smoothY);
        }

    } // namespace detail

    // CPU-side equivalent of mask_sampling.glsl's SampleMaskAlpha: returns 1.0 (fully opaque)
    // immediately for the "no cutout" sentinel without evaluating any noise, otherwise reproduces
    // ProceduralMaskGenerate.comp's two-octave blotchy cutout formula bit-for-bit.
    inline float SampleMaskAlphaCPU(uint32_t maskTextureIndex, const maths::vec2& uv) {
        if (maskTextureIndex == kInvalidMaskTextureIndex) {
            return 1.0f;
        }
        float n = detail::ValueNoiseCPU(uv.x * 6.0f, uv.y * 6.0f, maskTextureIndex) * 0.6f
                + detail::ValueNoiseCPU(uv.x * 13.0f, uv.y * 13.0f, maskTextureIndex) * 0.4f;
        return detail::SmoothStep(0.35f, 0.55f, n);
    }

}
