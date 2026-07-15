#pragma once
// Encode/decode helpers for the quantized vertex attributes stored in geometry::ClusterData
// (see ClusterFormat.h). Kept as free functions, header-only, so both CPU-side cluster building
// (VirtualGeometryCacheTest.cpp today, a real mesh simplifier tomorrow) and CPU-side validation
// code can share the exact same bit-for-bit encoding.

#include <algorithm>
#include <cmath>
#include <cstring>
#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"

namespace geometry {

    // -----------------------------------------------------------------------------------------
    // IEEE-754 binary32 <-> binary16 (half float) conversion, round-to-nearest-even. Handles
    // zero, subnormals, normals, infinity and NaN — no truncation shortcuts.
    // -----------------------------------------------------------------------------------------

    inline uint16_t FloatToHalf(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));

        uint32_t sign = (bits >> 16) & 0x8000u;
        uint32_t rawExponent = (bits >> 23) & 0xFFu;
        uint32_t mantissa = bits & 0x7FFFFFu;

        if (rawExponent == 0xFFu) {
            // Infinity or NaN: preserve a set mantissa bit so NaNs never collapse to Infinity.
            return static_cast<uint16_t>(sign | 0x7C00u | (mantissa != 0u ? 0x0200u : 0u));
        }

        int32_t exponent = static_cast<int32_t>(rawExponent) - 127 + 15;

        if (exponent >= 0x1F) {
            // Overflow: too large to represent as a finite half -> Infinity.
            return static_cast<uint16_t>(sign | 0x7C00u);
        }

        if (exponent <= 0) {
            if (exponent < -10) {
                // Too small even for a half subnormal: flushes to (signed) zero.
                return static_cast<uint16_t>(sign);
            }
            // Subnormal half: restore the implicit leading 1, then shift the 24-bit significand
            // down into the available subnormal mantissa bits, rounding to nearest-even.
            mantissa |= 0x800000u;
            uint32_t shift = static_cast<uint32_t>(14 - exponent);
            uint32_t halfMantissa = mantissa >> shift;
            uint32_t remainder = mantissa & ((1u << shift) - 1u);
            uint32_t halfway = 1u << (shift - 1u);
            if (remainder > halfway || (remainder == halfway && (halfMantissa & 1u) != 0u)) {
                ++halfMantissa;
            }
            return static_cast<uint16_t>(sign | halfMantissa);
        }

        // Normalized: round the 23-bit mantissa down to 10 bits, ties to even.
        uint32_t roundedMantissa = mantissa + 0x00000FFFu + ((mantissa >> 13) & 1u);
        if ((roundedMantissa & 0x00800000u) != 0u) {
            // Rounding carried into the exponent.
            roundedMantissa = 0u;
            ++exponent;
            if (exponent >= 0x1F) {
                return static_cast<uint16_t>(sign | 0x7C00u); // Overflow -> Infinity
            }
        }
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (roundedMantissa >> 13));
    }

    inline float HalfToFloat(uint16_t half) {
        uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
        uint32_t exponent = (half >> 10) & 0x1Fu;
        uint32_t mantissa = half & 0x3FFu;

        uint32_t bits;
        if (exponent == 0u) {
            if (mantissa == 0u) {
                bits = sign; // Signed zero
            }
            else {
                // Subnormal half -> normalize by shifting the mantissa's leading 1 into place.
                int32_t shiftCount = -1;
                do {
                    ++shiftCount;
                    mantissa <<= 1;
                } while ((mantissa & 0x400u) == 0u);
                mantissa &= 0x3FFu;
                uint32_t floatExponent = static_cast<uint32_t>(127 - 15 - shiftCount);
                bits = sign | (floatExponent << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 0x1Fu) {
            bits = sign | 0x7F800000u | (mantissa << 13); // Infinity or NaN
        }
        else {
            uint32_t floatExponent = exponent - 15u + 127u;
            bits = sign | (floatExponent << 23) | (mantissa << 13);
        }

        float result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    // -----------------------------------------------------------------------------------------
    // Vertex position quantization: maps a world/local-space position into 16-bit-per-channel
    // fixed point, normalized against a cluster's own [boundsMin, boundsMax] AABB.
    // -----------------------------------------------------------------------------------------

    inline ClusterVertexPosition QuantizePosition(const maths::vec3& position, const maths::vec3& boundsMin, const maths::vec3& boundsMax) {
        auto quantizeAxis = [](float value, float lo, float hi) -> uint16_t {
            float range = hi - lo;
            float t = (range > 0.0f) ? (value - lo) / range : 0.0f;
            t = std::clamp(t, 0.0f, 1.0f);
            return static_cast<uint16_t>(std::lround(t * 65535.0f));
            };
        return ClusterVertexPosition{
            quantizeAxis(position.x, boundsMin.x, boundsMax.x),
            quantizeAxis(position.y, boundsMin.y, boundsMax.y),
            quantizeAxis(position.z, boundsMin.z, boundsMax.z)
        };
    }

    inline maths::vec3 DequantizePosition(const ClusterVertexPosition& quantized, const maths::vec3& boundsMin, const maths::vec3& boundsMax) {
        auto dequantizeAxis = [](uint16_t value, float lo, float hi) -> float {
            return lo + (static_cast<float>(value) / 65535.0f) * (hi - lo);
            };
        return maths::vec3{
            dequantizeAxis(quantized.x, boundsMin.x, boundsMax.x),
            dequantizeAxis(quantized.y, boundsMin.y, boundsMax.y),
            dequantizeAxis(quantized.z, boundsMin.z, boundsMax.z)
        };
    }

    // -----------------------------------------------------------------------------------------
    // Normal encoding: standard octahedral mapping (Cigolle et al.) of a unit vector onto the
    // [-1,1]^2 square, then quantized to two 12-bit unsigned channels packed across 3 bytes.
    // -----------------------------------------------------------------------------------------

    inline ClusterVertexNormal EncodeOctNormal24(const maths::vec3& normal) {
        maths::vec3 n = normal.Normalize();
        float l1Norm = std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z);
        float invL1 = (l1Norm > 0.0f) ? (1.0f / l1Norm) : 0.0f;
        float px = n.x * invL1;
        float py = n.y * invL1;

        if (n.z < 0.0f) {
            float oldPx = px;
            px = (1.0f - std::fabs(py)) * ((oldPx >= 0.0f) ? 1.0f : -1.0f);
            py = (1.0f - std::fabs(oldPx)) * ((py >= 0.0f) ? 1.0f : -1.0f);
        }

        constexpr float kChannelMax = 4095.0f; // 12-bit unsigned per channel
        auto toChannel = [](float v) -> uint32_t {
            float t = std::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f);
            return static_cast<uint32_t>(std::lround(t * kChannelMax));
            };
        uint32_t u = toChannel(px);
        uint32_t v = toChannel(py);

        ClusterVertexNormal result{};
        result.data[0] = static_cast<uint8_t>(u & 0xFFu);
        result.data[1] = static_cast<uint8_t>(((u >> 8) & 0x0Fu) | ((v & 0x0Fu) << 4));
        result.data[2] = static_cast<uint8_t>((v >> 4) & 0xFFu);
        return result;
    }

    inline maths::vec3 DecodeOctNormal24(const ClusterVertexNormal& encoded) {
        uint32_t u = static_cast<uint32_t>(encoded.data[0]) | (static_cast<uint32_t>(encoded.data[1] & 0x0Fu) << 8);
        uint32_t v = static_cast<uint32_t>((encoded.data[1] >> 4) & 0x0Fu) | (static_cast<uint32_t>(encoded.data[2]) << 4);

        constexpr float kChannelMax = 4095.0f;
        float px = (static_cast<float>(u) / kChannelMax) * 2.0f - 1.0f;
        float py = (static_cast<float>(v) / kChannelMax) * 2.0f - 1.0f;

        maths::vec3 n{ px, py, 1.0f - std::fabs(px) - std::fabs(py) };
        if (n.z < 0.0f) {
            float oldX = n.x;
            n.x = (1.0f - std::fabs(n.y)) * ((oldX >= 0.0f) ? 1.0f : -1.0f);
            n.y = (1.0f - std::fabs(oldX)) * ((n.y >= 0.0f) ? 1.0f : -1.0f);
        }
        return n.Normalize();
    }

    // -----------------------------------------------------------------------------------------
    // UV encoding: plain half-float per channel.
    // -----------------------------------------------------------------------------------------

    inline ClusterVertexUV EncodeUV(const maths::vec2& uv) {
        return ClusterVertexUV{ FloatToHalf(uv.x), FloatToHalf(uv.y) };
    }

    inline maths::vec2 DecodeUV(const ClusterVertexUV& encoded) {
        return maths::vec2{ HalfToFloat(encoded.u), HalfToFloat(encoded.v) };
    }

}
