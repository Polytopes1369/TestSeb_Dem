// Standalone, framework-free unit test for the PCG framework roadmap's Phase 2.2 ("Landscape /
// Terrain Sampler") types: src/pcg/PcgTerrainSampler.h/.cpp. Mirrors PcgDataModelTests.cpp's own
// convention exactly (plain std::cerr on failure, exits 0/1, no Logger.cpp/Vulkan dependency) so it
// builds/runs identically in either config and needs no engine bring-up.
//
// Exercises: the terrain-noise CPU port's GPU parity (hardcoded reference values captured from the
// REAL, unmodified terrain_noise.glsl/displacement_noise.glsl shaders via a throwaway headless
// Vulkan compute dispatch -- see PcgTerrainSampler.h's own header comment for the full story,
// including a real bug this cross-check caught and fixed), height/normal/slope determinism and
// sanity, and the jittered-grid point-scatter sampler's determinism and basic invariants.

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"
#include "pcg/PcgTerrainSampler.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    bool NearlyEqual(float a, float b, float epsilon = 1.0e-4f) {
        return std::abs(a - b) <= epsilon;
    }

    // ------------------------------------------------------------------------------------------
    // GPU parity: reference values captured from the REAL terrain_noise.glsl SampleTerrainHeight,
    // dispatched via a throwaway headless Vulkan compute harness against an NVIDIA GeForce RTX 5080
    // Laptop GPU (see PcgTerrainSampler.h's own header comment for the full verification writeup).
    // localX/localZ below are exactly the (xz.x, xz.y) values the GPU probe shader's own
    // kSamplePoints array used, and expectedHeight is that same invocation's SampleTerrainHeight()
    // output, byte-for-byte as printed by the probe (float32 %.9g). Several entries are deliberately
    // negative-coordinate cases -- see PcgTerrainSampler.cpp's CellCoordToUint for why this matters.
    // ------------------------------------------------------------------------------------------
    struct GpuParityCase {
        float localX;
        float localZ;
        float expectedHeight;
    };

    const GpuParityCase kGpuParityCases[] = {
        { 0.0f,     0.0f,      0.140438199f },
        { 5.0f,     0.0f,     -0.300523907f },
        { -5.0f,    0.0f,     -0.15868935f  },
        { 0.0f,     5.0f,      0.0217571259f },
        { -37.25f,  12.75f,    0.0594239719f },
        { 100.0f,  -100.0f,    0.13168636f   },
        { -0.5f,   -0.5f,      0.134455115f  },
        { 1234.5f, -6789.25f, -0.126240328f  },
    };

    void TestGpuParity() {
        for (const GpuParityCase& c : kGpuParityCases) {
            const float actual = pcg::SampleTerrainHeightLocalCPU(c.localX, c.localZ);
            Check(NearlyEqual(actual, c.expectedHeight, 1.0e-5f),
                "SampleTerrainHeightLocalCPU GPU parity at (" + std::to_string(c.localX) + ", " + std::to_string(c.localZ) +
                "): expected " + std::to_string(c.expectedHeight) + ", got " + std::to_string(actual));
        }
    }

    void TestHeightDeterminismAndWorldOffset() {
        // Same local coordinate queried twice must be byte-identical (this port has no hidden state).
        const float h1 = pcg::SampleTerrainHeightLocalCPU(12.3f, -45.6f);
        const float h2 = pcg::SampleTerrainHeightLocalCPU(12.3f, -45.6f);
        Check(h1 == h2, "SampleTerrainHeightLocalCPU: identical inputs produce a byte-identical result");

        // SampleHeightCPU's world-space wrapper must subtract worldOffset.x/z before sampling (so
        // two landscape patches with different worldOffset.x but the same worldX query land on
        // DIFFERENT local noise samples) and must add worldOffset.y back in as a pure, exact
        // additive term (so changing ONLY worldOffset.y shifts every sampled height by exactly that
        // delta, with no interaction with the noise term itself) -- exactly mirroring
        // geom_terrain.comp's own "sample local xz, then translate by worldOffset afterward" order
        // of operations.
        pcg::PcgLandscapeData terrainA;
        terrainA.width = 200.0f;
        terrainA.length = 200.0f;
        terrainA.worldOffset = maths::vec3{ 10.0f, 3.0f, -20.0f };

        pcg::PcgLandscapeData terrainB = terrainA;
        terrainB.worldOffset.y = 3.0f + 7.5f; // Only the vertical anchor differs.

        const float worldX = 25.0f, worldZ = -5.0f;
        const float heightA = pcg::SampleHeightCPU(terrainA, worldX, worldZ);
        const float heightB = pcg::SampleHeightCPU(terrainB, worldX, worldZ);
        Check(NearlyEqual(heightB - heightA, 7.5f, 1.0e-4f),
            "SampleHeightCPU: changing ONLY worldOffset.y shifts the sampled height by exactly that delta");

        // Directly cross-checks the local/world decomposition: SampleHeightCPU(terrain, worldX,
        // worldZ) must equal worldOffset.y + SampleTerrainHeightLocalCPU(worldX - worldOffset.x,
        // worldZ - worldOffset.z), i.e. the wrapper performs exactly the documented subtraction.
        const float expected = terrainA.worldOffset.y +
            pcg::SampleTerrainHeightLocalCPU(worldX - terrainA.worldOffset.x, worldZ - terrainA.worldOffset.z);
        Check(NearlyEqual(heightA, expected, 1.0e-5f),
            "SampleHeightCPU: matches worldOffset.y + SampleTerrainHeightLocalCPU(worldX - offset.x, worldZ - offset.z) exactly");
    }

    void TestHeightBounds() {
        // TerrainHeightFbm's blended [-1,1]-range fbm sum, scaled by kTerrainAmplitude (0.4 in the
        // real shader), bounds every LOCAL height query to (-0.4, 0.4) regardless of input --
        // exercising a wide spread of coordinates (including large-magnitude and negative ones) as a
        // "the port didn't blow up / doesn't secretly diverge" sanity net, standing in for a literal
        // flat-terrain check (this port intentionally keeps the tuning constants as compile-time
        // constants, byte-for-byte matching the GPU shader, rather than exposing an "octave count"
        // testing knob that doesn't exist on the GPU side either -- see PcgTerrainSampler.h's own
        // header comment on this design choice).
        constexpr float kMaxPossibleAmplitude = 0.4f;
        bool allWithinBounds = true;
        for (int i = -10; i <= 10; ++i) {
            for (int j = -10; j <= 10; ++j) {
                const float x = static_cast<float>(i) * 137.0f;
                const float z = static_cast<float>(j) * 251.0f;
                const float h = pcg::SampleTerrainHeightLocalCPU(x, z);
                if (h < -kMaxPossibleAmplitude - 1.0e-4f || h > kMaxPossibleAmplitude + 1.0e-4f) {
                    allWithinBounds = false;
                }
            }
        }
        Check(allWithinBounds, "SampleTerrainHeightLocalCPU: every sampled height across a wide coordinate spread stays within (-kTerrainAmplitude, kTerrainAmplitude)");
    }

    void TestSlopeAndNormalSanity() {
        // ComputeSlopeRadians is a small, self-contained "angle between normal and world-up"
        // function -- exercised directly against synthetic (non-terrain) normals first, since these
        // give exactly known expected angles (unlike the real noise-driven terrain normal, whose
        // exact value isn't independently knowable without re-deriving the whole fbm by hand).
        Check(NearlyEqual(pcg::ComputeSlopeRadians(maths::vec3{ 0.0f, 1.0f, 0.0f }), 0.0f, 1.0e-4f),
            "ComputeSlopeRadians: a perfectly-up normal has zero slope");
        Check(NearlyEqual(pcg::ComputeSlopeRadians(maths::vec3{ 1.0f, 0.0f, 0.0f }), maths::PI * 0.5f, 1.0e-4f),
            "ComputeSlopeRadians: a horizontal (cliff-face) normal has a PI/2 slope");
        const maths::vec3 diagonal = maths::vec3{ 1.0f, 1.0f, 0.0f }.Normalize();
        Check(NearlyEqual(pcg::ComputeSlopeRadians(diagonal), maths::PI * 0.25f, 1.0e-3f),
            "ComputeSlopeRadians: a 45-degree normal has a PI/4 slope");

        // QuatFromNormal must actually align local +Y to the target normal: rotating (0,1,0) by the
        // returned quaternion must reproduce the target (within normalization tolerance), for a
        // normal set spanning "already aligned", "generic tilt", and the degenerate "exactly
        // opposite" 180-degree-flip case.
        const maths::vec3 testNormals[] = {
            maths::vec3{ 0.0f, 1.0f, 0.0f },
            maths::vec3{ 1.0f, 0.0f, 0.0f },
            maths::vec3{ 0.3f, 0.9f, -0.2f }.Normalize(),
            maths::vec3{ 0.0f, -1.0f, 0.0f }, // Exactly opposite +Y: the 180-degree-flip edge case.
        };
        for (const maths::vec3& n : testNormals) {
            const maths::quat q = pcg::QuatFromNormal(n);
            const maths::vec3 rotatedUp = q.RotateVector(maths::vec3{ 0.0f, 1.0f, 0.0f });
            Check(NearlyEqual(rotatedUp.x, n.x, 1.0e-3f) && NearlyEqual(rotatedUp.y, n.y, 1.0e-3f) && NearlyEqual(rotatedUp.z, n.z, 1.0e-3f),
                "QuatFromNormal: rotating local +Y by the returned quaternion reproduces the target normal (" +
                std::to_string(n.x) + ", " + std::to_string(n.y) + ", " + std::to_string(n.z) + ")");
        }

        // ComputeTerrainNormalCPU on the real (gentle-amplitude) terrain: at the origin, the normal
        // must be unit-length and must point predominantly upward (this terrain's own amplitude/
        // frequency tuning -- kTerrainAmplitude=0.4 over a kTerrainBaseFrequency=0.06 wavelength --
        // never produces a literal vertical cliff, so a strongly-upward normal is a valid invariant,
        // not a coincidence of one sample point).
        pcg::PcgLandscapeData terrain;
        terrain.width = 200.0f;
        terrain.length = 200.0f;
        terrain.worldOffset = maths::vec3{ 0.0f, 0.0f, 0.0f };
        const maths::vec3 normal = pcg::ComputeTerrainNormalCPU(terrain, 0.0f, 0.0f);
        Check(NearlyEqual(normal.Length(), 1.0f, 1.0e-3f), "ComputeTerrainNormalCPU: returns a unit-length normal");
        Check(normal.y > 0.5f, "ComputeTerrainNormalCPU: at the origin, the normal points predominantly upward (gentle terrain slope)");
    }

    void TestScatterDegenerateInputsReturnEmpty() {
        pcg::PcgLandscapeData zeroWidth;
        zeroWidth.width = 0.0f;
        zeroWidth.length = 100.0f;
        Check(pcg::SampleTerrainPoints(zeroWidth, 1.0f, 1u).points.empty(),
            "SampleTerrainPoints: zero width -> empty batch (mirrors geom_terrain.comp's own degenerate-Params validation)");

        pcg::PcgLandscapeData negativeLength;
        negativeLength.width = 100.0f;
        negativeLength.length = -5.0f;
        Check(pcg::SampleTerrainPoints(negativeLength, 1.0f, 1u).points.empty(),
            "SampleTerrainPoints: negative length -> empty batch");

        pcg::PcgLandscapeData validExtent;
        validExtent.width = 100.0f;
        validExtent.length = 100.0f;
        Check(pcg::SampleTerrainPoints(validExtent, 0.0f, 1u).points.empty(),
            "SampleTerrainPoints: zero pointsPerSquareMeter -> empty batch");
        Check(pcg::SampleTerrainPoints(validExtent, -1.0f, 1u).points.empty(),
            "SampleTerrainPoints: negative pointsPerSquareMeter -> empty batch");
    }

    void TestScatterDeterminismAndInvariants() {
        pcg::PcgLandscapeData terrain;
        terrain.meshID = 11;
        terrain.width = 64.0f;
        terrain.length = 64.0f;
        terrain.worldOffset = maths::vec3{ 100.0f, 5.0f, -50.0f };
        constexpr float kDensity = 0.25f; // 1 point per 4 square meters.
        constexpr uint32_t kSeed = 0xC0FFEEu;

        const pcg::PcgTerrainPointBatch batchA = pcg::SampleTerrainPoints(terrain, kDensity, kSeed);
        const pcg::PcgTerrainPointBatch batchB = pcg::SampleTerrainPoints(terrain, kDensity, kSeed);

        Check(!batchA.points.empty(), "SampleTerrainPoints: a valid (terrain, density, seed) input produces at least one point");
        Check(batchA.points.size() == batchB.points.size(),
            "SampleTerrainPoints: identical inputs produce the same point count across two independent calls");
        Check(batchA.slopeRadians.size() == batchA.points.size(),
            "SampleTerrainPoints: slopeRadians is exactly parallel to points (same size)");

        bool allPointsIdentical = (batchA.points.size() == batchB.points.size());
        if (allPointsIdentical) {
            for (size_t i = 0; i < batchA.points.size(); ++i) {
                const pcg::PcgPoint& a = batchA.points[i];
                const pcg::PcgPoint& b = batchB.points[i];
                if (!(a.position.x == b.position.x && a.position.y == b.position.y && a.position.z == b.position.z &&
                      a.seed == b.seed && batchA.slopeRadians[i] == batchB.slopeRadians[i])) {
                    allPointsIdentical = false;
                    break;
                }
            }
        }
        Check(allPointsIdentical, "SampleTerrainPoints: two calls with identical (terrain, density, seed) produce a byte-identical point sequence");

        // A different seed must (overwhelmingly likely) produce a different point set -- at minimum
        // the per-cell jitter and per-point seed differ, so an exact position match across the WHOLE
        // batch would require a hash collision in every single cell simultaneously.
        const pcg::PcgTerrainPointBatch batchDifferentSeed = pcg::SampleTerrainPoints(terrain, kDensity, kSeed + 1u);
        bool anyPositionDiffers = (batchDifferentSeed.points.size() != batchA.points.size());
        if (!anyPositionDiffers) {
            for (size_t i = 0; i < batchA.points.size(); ++i) {
                if (batchA.points[i].position.x != batchDifferentSeed.points[i].position.x ||
                    batchA.points[i].position.z != batchDifferentSeed.points[i].position.z) {
                    anyPositionDiffers = true;
                    break;
                }
            }
        }
        Check(anyPositionDiffers, "SampleTerrainPoints: a different seed produces a different point sequence");

        // Every point must land strictly within the terrain's own documented world-space extent
        // (worldOffset.x/z +/- width/length / 2), matching geom_terrain.comp's own local-grid-before-
        // worldOffset convention (PcgLandscapeData's own header comment).
        const float minX = terrain.worldOffset.x - terrain.width * 0.5f;
        const float maxX = terrain.worldOffset.x + terrain.width * 0.5f;
        const float minZ = terrain.worldOffset.z - terrain.length * 0.5f;
        const float maxZ = terrain.worldOffset.z + terrain.length * 0.5f;
        bool allWithinExtent = true;
        for (const pcg::PcgPoint& p : batchA.points) {
            if (p.position.x < minX || p.position.x > maxX || p.position.z < minZ || p.position.z > maxZ) {
                allWithinExtent = false;
                break;
            }
        }
        Check(allWithinExtent, "SampleTerrainPoints: every scattered point lies within the terrain's documented world-space extent");

        // Every point's rotation quaternion must be (approximately) unit-length -- QuatFromNormal's
        // FromAxisAngle construction always produces a unit quaternion for a unit axis, so this is a
        // basic sanity/no-NaN check on the whole pipeline, not a redundant re-test of QuatFromNormal
        // itself.
        bool allRotationsUnit = true;
        for (const pcg::PcgPoint& p : batchA.points) {
            const float lenSq = p.rotation.x * p.rotation.x + p.rotation.y * p.rotation.y +
                p.rotation.z * p.rotation.z + p.rotation.w * p.rotation.w;
            if (!NearlyEqual(lenSq, 1.0f, 1.0e-2f)) {
                allRotationsUnit = false;
                break;
            }
        }
        Check(allRotationsUnit, "SampleTerrainPoints: every point's rotation quaternion is unit-length");

        // Roughly matches the requested density: cell count is derived the same
        // "round(extent / (1/sqrt(density)))" way the sampler itself computes it, so this is really
        // checking that SampleTerrainPoints didn't silently drop/duplicate cells, not re-deriving an
        // independent density formula.
        const float cellSize = 1.0f / std::sqrt(kDensity);
        const uint32_t expectedCellCountX = static_cast<uint32_t>(std::round(terrain.width / cellSize));
        const uint32_t expectedCellCountZ = static_cast<uint32_t>(std::round(terrain.length / cellSize));
        Check(batchA.points.size() == static_cast<size_t>(expectedCellCountX) * static_cast<size_t>(expectedCellCountZ),
            "SampleTerrainPoints: point count matches the expected jittered-grid cell count for the requested density");
    }

} // namespace

int main() {
    TestGpuParity();
    TestHeightDeterminismAndWorldOffset();
    TestHeightBounds();
    TestSlopeAndNormalSanity();
    TestScatterDegenerateInputsReturnEmpty();
    TestScatterDeterminismAndInvariants();

    if (g_failCount == 0) {
        std::cout << "PcgTerrainSamplerTests: all checks passed.\n";
        return 0;
    }
    std::cerr << "PcgTerrainSamplerTests: " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
