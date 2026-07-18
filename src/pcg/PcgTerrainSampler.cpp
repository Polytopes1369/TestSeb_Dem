#include "pcg/PcgTerrainSampler.h"

#include "pcg/PcgSeededRandom.h"

#include <algorithm>
#include <cmath>

// PCG framework roadmap, Phase 2.2 ("Landscape / Terrain Sampler"): implementation. See
// PcgTerrainSampler.h's own header comment for the overall file-pair rationale and verification
// summary.

namespace pcg {

    namespace {

        // ---------------------------------------------------------------------------------------
        // Byte-for-byte port of displacement_noise.glsl's HashUint3D/HashToUnitFloat/ValueNoise3D
        // (terrain_noise.glsl #includes displacement_noise.glsl and calls ValueNoise3D directly --
        // see that file's own #include line).
        // ---------------------------------------------------------------------------------------

        // Same 3-round avalanche mixer as displacement_noise.glsl's HashUint3D. Pure 32-bit unsigned
        // multiply/shift/xor -- unsigned arithmetic in C++ is well-defined modulo 2^32 (identical
        // wraparound semantics to GLSL's `uint`), so this is bit-identical to the GPU version given
        // identical uint32_t inputs.
        inline uint32_t HashUint3D(uint32_t x, uint32_t y, uint32_t z) {
            uint32_t h = x * 1973u + y * 9277u + z * 42013u;
            h ^= h >> 16u;
            h *= 0x7feb352du;
            h ^= h >> 15u;
            h *= 0x846ca68bu;
            h ^= h >> 16u;
            return h;
        }

        inline float HashToUnitFloat(uint32_t h) {
            return static_cast<float>(h & 0xFFFFu) / 65535.0f;
        }

        // Converts a floor()'d noise-cell coordinate (may be negative -- terrain world coordinates
        // span both signs) into the uint32_t HashUint3D expects, mirroring GLSL's `uint(cell.x)`
        // cast at each ValueNoise3D call site.
        //
        // This specific cast is a genuine platform-ambiguity risk on the GPU side itself: converting
        // a NEGATIVE float to an unsigned integer type is implementation-defined/undefined behavior
        // per both the GLSL spec (4.60, section 5.4.1) and the SPIR-V spec (OpConvertFToU: "results
        // are undefined if Float Value is outside the range of the result's underlying type"), so
        // different GPU vendors/drivers are not guaranteed to agree with EACH OTHER here, let alone
        // with a naive CPU port.
        //
        // This port does NOT guess -- it was empirically cross-checked against the real, unmodified
        // GPU shader (see PcgTerrainSampler.h's own header comment for the throwaway-shader dispatch
        // that did this). The first guess (truncate to int32_t, reinterpret the two's-complement bit
        // pattern as uint32_t -- the natural "portable C++" choice) turned out to be WRONG: it
        // disagreed with the real GPU output by a large margin (e.g. local (x,z) = (-0.0685868,
        // -0.0173766) produced ValueNoise3D ~= 0.0059 CPU-side vs. the GPU's actual 0.6804). The
        // correct model, confirmed by matching the GPU's actual output exactly, is a SATURATING
        // conversion (negative inputs clamp to 0, not wrap) -- this matches NVIDIA's documented PTX
        // ISA behavior for float-to-unsigned conversion (`cvt.rzi.u32.f32`: "if the value being
        // converted is ... outside the range of the destination type, the result is clamped to the
        // appropriate end of the range"), which is exactly what SPIR-V's OpConvertFToU lowers to on
        // this hardware. This IS a genuine, pre-existing quirk of the production shader (any negative
        // noise-cell coordinate collapses its hash-corner lookup toward cell 0 rather than
        // continuing to vary), not something this CPU port invents -- reproducing it faithfully is
        // this phase's job, not fixing GPU-shader noise quality (out of scope here).
        inline uint32_t CellCoordToUint(float cellCoord) {
            if (cellCoord <= 0.0f) {
                return 0u;
            }
            if (cellCoord >= 4294967295.0f) {
                return 0xFFFFFFFFu;
            }
            return static_cast<uint32_t>(cellCoord);
        }

        // Value noise at integer cell corners, trilinearly interpolated with a smoothstep blend --
        // exact port of displacement_noise.glsl's ValueNoise3D (8 corners, same smoothstep-weighted
        // trilinear mix).
        float ValueNoise3DCPU(const maths::vec3& p) {
            const maths::vec3 cell{ std::floor(p.x), std::floor(p.y), std::floor(p.z) };
            const maths::vec3 f = p - cell;
            const maths::vec3 s{
                f.x * f.x * (3.0f - 2.0f * f.x),
                f.y * f.y * (3.0f - 2.0f * f.y),
                f.z * f.z * (3.0f - 2.0f * f.z)
            };

            const uint32_t ix = CellCoordToUint(cell.x);
            const uint32_t iy = CellCoordToUint(cell.y);
            const uint32_t iz = CellCoordToUint(cell.z);

            const float h000 = HashToUnitFloat(HashUint3D(ix,      iy,      iz));
            const float h100 = HashToUnitFloat(HashUint3D(ix + 1u, iy,      iz));
            const float h010 = HashToUnitFloat(HashUint3D(ix,      iy + 1u, iz));
            const float h110 = HashToUnitFloat(HashUint3D(ix + 1u, iy + 1u, iz));
            const float h001 = HashToUnitFloat(HashUint3D(ix,      iy,      iz + 1u));
            const float h101 = HashToUnitFloat(HashUint3D(ix + 1u, iy,      iz + 1u));
            const float h011 = HashToUnitFloat(HashUint3D(ix,      iy + 1u, iz + 1u));
            const float h111 = HashToUnitFloat(HashUint3D(ix + 1u, iy + 1u, iz + 1u));

            // GLSL mix(a, b, t) == a + (b - a) * t -- used verbatim below, same associativity/order
            // of operations as the GPU version, to keep float rounding identical step-for-step.
            const float x00 = h000 + (h100 - h000) * s.x;
            const float x10 = h010 + (h110 - h010) * s.x;
            const float x01 = h001 + (h101 - h001) * s.x;
            const float x11 = h011 + (h111 - h011) * s.x;
            const float y0 = x00 + (x10 - x00) * s.y;
            const float y1 = x01 + (x11 - x01) * s.y;
            return y0 + (y1 - y0) * s.z; // [0, 1)
        }

        // ---------------------------------------------------------------------------------------
        // Byte-for-byte port of terrain_noise.glsl's own tuning constants and fbm combination.
        // Duplicated (not shared via a common header) because GLSL is not preprocessable by a C++
        // translation unit -- the same "two independently-maintained, comment-cross-referenced
        // copies" convention PcgSeededRandom.h's own header comment documents for PcgHash32 vs.
        // pcg_common.glsl's PcgHash32.
        // ---------------------------------------------------------------------------------------

        constexpr uint32_t kTerrainOctaves = 2u;
        constexpr float kTerrainBaseFrequency = 0.06f;
        constexpr float kTerrainRidgeWeight = 0.0f;
        constexpr float kTerrainAmplitude = 0.4f;
        constexpr float kWarpFrequency = 0.06f;
        constexpr float kWarpAmplitude = 1.8f;

        // Port of terrain_noise.glsl's RidgedNoiseOctave.
        float RidgedNoiseOctaveCPU(const maths::vec3& p) {
            return 1.0f - std::abs(ValueNoise3DCPU(p) * 2.0f - 1.0f);
        }

        // Port of terrain_noise.glsl's DomainWarpOffset. NOTE: the GPU version's
        // `ValueNoise3D(vec3(xz.x, 7.0, xz.y) * kWarpFrequency)` multiplies ALL THREE vec3
        // components (including the constant 7.0/13.0 y-plane selector) by kWarpFrequency, not just
        // the xz.x/xz.y ones -- reproduced exactly here via vec3::operator*(scalar) applied to the
        // whole 3-component value, not per-component.
        maths::vec2 DomainWarpOffsetCPU(const maths::vec2& xz) {
            const maths::vec3 warpInputX = maths::vec3{ xz.x, 7.0f, xz.y } * kWarpFrequency;
            const maths::vec3 warpInputZ = maths::vec3{ xz.x, 13.0f, xz.y } * kWarpFrequency;
            const float warpX = (ValueNoise3DCPU(warpInputX) * 2.0f - 1.0f) * kWarpAmplitude;
            const float warpZ = (ValueNoise3DCPU(warpInputZ) * 2.0f - 1.0f) * kWarpAmplitude;
            return maths::vec2{ warpX, warpZ };
        }

        // Port of terrain_noise.glsl's TerrainHeightFbm.
        float TerrainHeightFbmCPU(const maths::vec2& xz) {
            const maths::vec2 warpOffset = DomainWarpOffsetCPU(xz);
            const maths::vec2 warped = xz + warpOffset;

            float frequency = kTerrainBaseFrequency;
            float amplitude = 0.5f;
            float smoothSum = 0.0f;
            float ridgedSum = 0.0f;
            float maxAmplitude = 0.0f;
            for (uint32_t i = 0; i < kTerrainOctaves; ++i) {
                // GLSL: vec3 p = vec3(warped.x, 0.0, warped.y) * frequency; -- the middle 0.0
                // component scales to 0.0 * frequency == 0.0 either way, reproduced directly below.
                const maths::vec3 p{ warped.x * frequency, 0.0f, warped.y * frequency };
                smoothSum += (ValueNoise3DCPU(p) * 2.0f - 1.0f) * amplitude;
                ridgedSum += RidgedNoiseOctaveCPU(p) * amplitude;
                maxAmplitude += amplitude;
                frequency *= 2.0f;
                amplitude *= 0.5f;
            }
            const float smoothNorm = smoothSum / maxAmplitude;               // [-1, 1]
            const float ridgedNorm = (ridgedSum / maxAmplitude) * 2.0f - 1.0f; // [-1, 1]
            return smoothNorm + (ridgedNorm - smoothNorm) * kTerrainRidgeWeight; // mix(smoothNorm, ridgedNorm, kTerrainRidgeWeight)
        }

    } // namespace

    float SampleTerrainHeightLocalCPU(float localX, float localZ) {
        return TerrainHeightFbmCPU(maths::vec2{ localX, localZ }) * kTerrainAmplitude;
    }

    float SampleHeightCPU(const PcgLandscapeData& terrain, float worldX, float worldZ) {
        const float localX = worldX - terrain.worldOffset.x;
        const float localZ = worldZ - terrain.worldOffset.z;
        return terrain.worldOffset.y + SampleTerrainHeightLocalCPU(localX, localZ);
    }

    maths::vec3 ComputeTerrainNormalCPU(const PcgLandscapeData& terrain, float worldX, float worldZ, float epsilon) {
        // Analytic central-difference normal -- same technique as geom_terrain.comp's own: tangent
        // vectors built from height differences along local X/Z, then cross-producted (valid because
        // a heightfield's base normal is always +Y, so world X/Z already serve as tangent directions
        // directly -- no general ONB needed).
        const float hL = SampleHeightCPU(terrain, worldX - epsilon, worldZ);
        const float hR = SampleHeightCPU(terrain, worldX + epsilon, worldZ);
        const float hD = SampleHeightCPU(terrain, worldX, worldZ - epsilon);
        const float hU = SampleHeightCPU(terrain, worldX, worldZ + epsilon);
        const maths::vec3 tangentX{ 2.0f * epsilon, hR - hL, 0.0f };
        const maths::vec3 tangentZ{ 0.0f, hU - hD, 2.0f * epsilon };
        return tangentZ.Cross(tangentX).Normalize();
    }

    float ComputeSlopeRadians(const maths::vec3& terrainNormal) {
        const maths::vec3 n = terrainNormal.Normalize();
        const float cosAngle = std::clamp(n.y, -1.0f, 1.0f); // dot(n, (0,1,0)) == n.y
        return std::acos(cosAngle);
    }

    maths::quat QuatFromNormal(const maths::vec3& normal) {
        const maths::vec3 up{ 0.0f, 1.0f, 0.0f };
        const maths::vec3 n = normal.Normalize();
        const float d = std::clamp(up.Dot(n), -1.0f, 1.0f);

        constexpr float kParallelEpsilon = 1.0e-6f;
        if (d > 1.0f - kParallelEpsilon) {
            return maths::quat{}; // Already aligned with +Y: identity rotation.
        }
        if (d < -1.0f + kParallelEpsilon) {
            // Exactly opposite +Y (a fully overhanging/inverted normal): any axis perpendicular to
            // `up` gives a valid 180-degree flip. Cross with +X unless the normal happens to be
            // degenerate there too (would only happen if +X were somehow parallel to +Y, which it
            // never is), in which case fall back to +Z.
            maths::vec3 axis = maths::vec3{ 1.0f, 0.0f, 0.0f }.Cross(up);
            if (axis.Length() < kParallelEpsilon) {
                axis = maths::vec3{ 0.0f, 0.0f, 1.0f }.Cross(up);
            }
            return maths::quat::FromAxisAngle(axis.Normalize(), maths::PI);
        }

        const maths::vec3 axis = up.Cross(n).Normalize();
        const float angle = std::acos(d);
        return maths::quat::FromAxisAngle(axis, angle);
    }

    PcgTerrainPointBatch SampleTerrainPoints(const PcgLandscapeData& terrain, float pointsPerSquareMeter,
        uint32_t seed, float normalEpsilon) {
        PcgTerrainPointBatch batch;

        // Mirrors geom_terrain.comp's own degenerate-Params validation: reject and return nothing
        // rather than dividing by zero / looping forever below.
        if (terrain.width <= 0.0f || terrain.length <= 0.0f || pointsPerSquareMeter <= 0.0f) {
            return batch;
        }

        // Jittered grid: divide the extent into square cells sized so their count roughly matches
        // the requested density (one candidate point per cell), then place each point at a random
        // jittered offset within its own cell -- simple, deterministic, and sufficient for this
        // phase (true Poisson-disc pruning is a later phase's Self-Pruning filter's job).
        const float cellSize = 1.0f / std::sqrt(pointsPerSquareMeter);
        const uint32_t cellCountX = std::max<uint32_t>(1u, static_cast<uint32_t>(std::round(terrain.width / cellSize)));
        const uint32_t cellCountZ = std::max<uint32_t>(1u, static_cast<uint32_t>(std::round(terrain.length / cellSize)));
        const float cellSizeX = terrain.width / static_cast<float>(cellCountX);
        const float cellSizeZ = terrain.length / static_cast<float>(cellCountZ);

        // Matches geom_terrain.comp's own local-grid-before-worldOffset convention: the grid spans
        // [-width/2, +width/2] x [-length/2, +length/2] around worldOffset.x/z.
        const float originX = terrain.worldOffset.x - terrain.width * 0.5f;
        const float originZ = terrain.worldOffset.z - terrain.length * 0.5f;

        batch.points.reserve(static_cast<size_t>(cellCountX) * static_cast<size_t>(cellCountZ));
        batch.slopeRadians.reserve(batch.points.capacity());

        for (uint32_t j = 0; j < cellCountZ; ++j) {
            for (uint32_t i = 0; i < cellCountX; ++i) {
                // Per-cell seed derived directly from (seed, cellIndex) via PcgHashCombine, NOT from
                // one shared stream advanced cell-by-cell -- see this function's own header comment
                // in PcgTerrainSampler.h for why that makes the result iteration-order-independent
                // and reproducible.
                const uint32_t cellIndex = j * cellCountX + i;
                const uint32_t cellSeed = PcgHashCombine(seed, cellIndex);
                PcgSeededRandom rng(cellSeed);

                const float jitterX = rng.NextFloat01();
                const float jitterZ = rng.NextFloat01();
                const float worldX = originX + (static_cast<float>(i) + jitterX) * cellSizeX;
                const float worldZ = originZ + (static_cast<float>(j) + jitterZ) * cellSizeZ;
                const float worldY = SampleHeightCPU(terrain, worldX, worldZ);
                const maths::vec3 normal = ComputeTerrainNormalCPU(terrain, worldX, worldZ, normalEpsilon);

                PcgPoint point;
                point.position = maths::vec3{ worldX, worldY, worldZ };
                point.rotation = QuatFromNormal(normal);
                // Every future PCG node needing per-point randomness for THIS point (jitter,
                // variation, ...) derives its own stream from this seed, per PcgPoint's own
                // documented seed-field contract (PcgPointData.h) -- reusing the cell's own seed
                // here (rather than minting an unrelated one) keeps the whole pipeline traceable
                // back to a single (terrain, density, seed) input tuple.
                point.seed = cellSeed;

                batch.points.push_back(point);
                batch.slopeRadians.push_back(ComputeSlopeRadians(normal));
            }
        }

        return batch;
    }

}
