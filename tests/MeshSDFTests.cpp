// Standalone, framework-free unit test for geometry::BuildMeshSDF / DecodeMeshSDFVoxel /
// SampleMeshSDF (src/geometry/MeshSDFGenerator.h). Purely CPU-side: zero Vulkan dependency.
// Exits 0 if every check passes, non-zero otherwise (see CMakeLists.txt's CTest registration).
//
// What is validated, on an axis-aligned unit cube (12 triangles, watertight -- the one closed
// mesh whose exact SDF is known analytically near every face) and on a tessellated sphere:
//   1. ZERO ON THE SURFACE (the core requirement): the signed distance sampled exactly ON the
//      mesh surface is 0. Verified at two rigor levels:
//        a. The RAW (uncompressed float) grid, trilinearly sampled at points on the cube's faces:
//           a plane's SDF is a linear function of position, and trilinear interpolation
//           reproduces linear functions exactly, so |d| must vanish to float precision -- the
//           mathematically exact "distance == 0 on the surface" statement.
//        b. The COMPRESSED (8-bit normalized, BC4-style 3-bit-indexed blocks) reconstruction at
//           the same points: |d| must stay within the provable worst-case decode error bound,
//           blockQuantError = (decodedMax - decodedMin)/(2*7) + 1/(2*255) in normalized units
//           (half an index step within the block's dequantized endpoint range, plus half an
//           endpoint quantization step), converted to world units.
//   2. Sign correctness: negative strictly inside (cube center), positive strictly outside.
//   3. Exact analytic distances at probe points near a cube face (within voxel/quantization
//      tolerance), proving the magnitude, not just the sign, is right.
//   4. Compression ratio: blocks store 26 bytes per 64 voxels (static_assert'd in the header;
//      re-checked here against the actual allocation), i.e. ~2.46:1 over raw 8-bit.
//   5. Round-trip consistency: every voxel's compressed reconstruction stays within the same
//      per-block worst-case bound of its raw value (checked exhaustively, all 32768 voxels).

#include "geometry/MeshSDFGenerator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

    int g_FailCount = 0;

    void Check(bool condition, const std::string& description) {
        if (!condition) {
            std::cerr << "[FAIL] " << description << std::endl;
            ++g_FailCount;
        }
    }

    // Axis-aligned cube [-half, +half]^3 as a 12-triangle watertight mesh, CCW outward winding.
    void MakeCube(float half, std::vector<maths::vec3>& outPositions, std::vector<uint32_t>& outTriangles) {
        outPositions = {
            { -half, -half, -half }, { +half, -half, -half },
            { +half, +half, -half }, { -half, +half, -half },
            { -half, -half, +half }, { +half, -half, +half },
            { +half, +half, +half }, { -half, +half, +half },
        };
        outTriangles = {
            0, 2, 1,  0, 3, 2,   // -Z face (normal (0,0,-1))
            4, 5, 6,  4, 6, 7,   // +Z face
            0, 1, 5,  0, 5, 4,   // -Y face
            3, 7, 6,  3, 6, 2,   // +Y face
            0, 4, 7,  0, 7, 3,   // -X face
            1, 2, 6,  1, 6, 5,   // +X face
        };
    }

    // Latitude/longitude sphere -- a second, curved, watertight mesh so the sign/zero checks are
    // not cube-specific.
    void MakeSphere(float radius, uint32_t rings, uint32_t segments,
        std::vector<maths::vec3>& outPositions, std::vector<uint32_t>& outTriangles) {
        outPositions.clear();
        outTriangles.clear();
        for (uint32_t r = 0; r <= rings; ++r) {
            const float phi = maths::PI * static_cast<float>(r) / static_cast<float>(rings);
            for (uint32_t s = 0; s <= segments; ++s) {
                const float theta = 2.0f * maths::PI * static_cast<float>(s) / static_cast<float>(segments);
                outPositions.push_back(maths::vec3{
                    radius * std::sin(phi) * std::cos(theta),
                    radius * std::cos(phi),
                    radius * std::sin(phi) * std::sin(theta) });
            }
        }
        // Winding: with P(phi, theta) = R(sin phi cos theta, cos phi, sin phi sin theta), the
        // outward normal is dP/dtheta x dP/dphi, so each quad's triangles must walk theta first
        // ((a, a+1, b) not (a, b, a+1)) for CCW-outward orientation -- the pseudo-normal sign
        // convention (negative inside) depends on outward-facing triangles.
        const uint32_t stride = segments + 1u;
        for (uint32_t r = 0; r < rings; ++r) {
            for (uint32_t s = 0; s < segments; ++s) {
                const uint32_t a = r * stride + s;
                const uint32_t b = a + stride;
                outTriangles.insert(outTriangles.end(), { a, a + 1u, b });
                outTriangles.insert(outTriangles.end(), { b, a + 1u, b + 1u });
            }
        }
    }

    // Worst-case world-unit decode error of the block containing voxel (x,y,z): half an index
    // step across the block's dequantized endpoint span, plus half an endpoint quantization step
    // -- the bound stated in the file header comment, evaluated per actual block.
    float BlockDecodeErrorBound(const geometry::MeshSDF& sdf, uint32_t x, uint32_t y, uint32_t z) {
        const uint32_t blocksPerAxis = sdf.resolution / geometry::kSDFBlockDim;
        const uint32_t bx = x / geometry::kSDFBlockDim;
        const uint32_t by = y / geometry::kSDFBlockDim;
        const uint32_t bz = z / geometry::kSDFBlockDim;
        const geometry::MeshSDFBlock& block =
            sdf.blocks[(static_cast<size_t>(bz) * blocksPerAxis + by) * blocksPerAxis + bx];
        const float span = static_cast<float>(block.maxValue - block.minValue) / 255.0f;
        const float normalizedBound = span / 14.0f + 1.0f / 510.0f;
        return normalizedBound * 2.0f * sdf.maxEncodedDistance;
    }

    void TestCubeSurfaceIsZero() {
        std::vector<maths::vec3> positions;
        std::vector<uint32_t> triangles;
        const float half = 1.0f;
        MakeCube(half, positions, triangles);

        std::vector<float> raw;
        geometry::MeshSDF sdf = geometry::BuildMeshSDF(positions, triangles, geometry::kMeshSDFResolution, &raw);
        Check(sdf.resolution == geometry::kMeshSDFResolution, "TestCubeSurfaceIsZero: build must succeed at 32^3");
        if (sdf.resolution == 0) return;

        // Raw-grid trilinear sampler (mirrors SampleMeshSDF's voxel-center convention, but over
        // the float ground truth instead of the compressed blocks).
        auto sampleRaw = [&](const maths::vec3& p) -> float {
            const float fx = std::clamp((p.x - sdf.volumeMin.x) / sdf.voxelSize - 0.5f, 0.0f, static_cast<float>(sdf.resolution - 1u));
            const float fy = std::clamp((p.y - sdf.volumeMin.y) / sdf.voxelSize - 0.5f, 0.0f, static_cast<float>(sdf.resolution - 1u));
            const float fz = std::clamp((p.z - sdf.volumeMin.z) / sdf.voxelSize - 0.5f, 0.0f, static_cast<float>(sdf.resolution - 1u));
            const uint32_t x0 = static_cast<uint32_t>(fx), y0 = static_cast<uint32_t>(fy), z0 = static_cast<uint32_t>(fz);
            const uint32_t x1 = std::min(x0 + 1u, sdf.resolution - 1u);
            const uint32_t y1 = std::min(y0 + 1u, sdf.resolution - 1u);
            const uint32_t z1 = std::min(z0 + 1u, sdf.resolution - 1u);
            const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0), tz = fz - static_cast<float>(z0);
            auto at = [&](uint32_t x, uint32_t y, uint32_t z) {
                return raw[(static_cast<size_t>(z) * sdf.resolution + y) * sdf.resolution + x];
                };
            const float c00 = at(x0, y0, z0) + (at(x1, y0, z0) - at(x0, y0, z0)) * tx;
            const float c10 = at(x0, y1, z0) + (at(x1, y1, z0) - at(x0, y1, z0)) * tx;
            const float c01 = at(x0, y0, z1) + (at(x1, y0, z1) - at(x0, y0, z1)) * tx;
            const float c11 = at(x0, y1, z1) + (at(x1, y1, z1) - at(x0, y1, z1)) * tx;
            const float c0 = c00 + (c10 - c00) * ty;
            const float c1 = c01 + (c11 - c01) * ty;
            return c0 + (c1 - c0) * tz;
            };

        // Surface sample points: centers and interior points of all 6 cube faces. Kept away from
        // edges/corners so the nearest feature is unambiguously the face plane (where the SDF is
        // exactly linear and trilinear interpolation is exact).
        const float inset = half * 0.5f;
        std::vector<maths::vec3> surfacePoints;
        for (int axis = 0; axis < 3; ++axis) {
            for (float side : { -half, +half }) {
                for (float u : { -inset, 0.0f, +inset }) {
                    for (float v : { -inset, 0.0f, +inset }) {
                        maths::vec3 p{ 0.0f, 0.0f, 0.0f };
                        if (axis == 0) { p = { side, u, v }; }
                        else if (axis == 1) { p = { u, side, v }; }
                        else { p = { u, v, side }; }
                        surfacePoints.push_back(p);
                    }
                }
            }
        }

        // 1a. Raw grid: exactly zero on the surface, up to float rounding. The tolerance scales
        // with maxEncodedDistance only through float ULPs of intermediate math -- 1e-4 * voxelSize
        // is orders of magnitude above rounding noise yet 4 orders below one voxel.
        const float rawTolerance = 1.0e-4f * sdf.voxelSize;
        for (const maths::vec3& p : surfacePoints) {
            const float d = sampleRaw(p);
            Check(std::fabs(d) <= rawTolerance,
                "TestCubeSurfaceIsZero: RAW trilinear distance must be 0 on the surface (got " +
                std::to_string(d) + " at face point, tolerance " + std::to_string(rawTolerance) + ")");
        }

        // 1b. Compressed reconstruction: zero within the provable per-block decode bound (the 8
        // voxels a trilinear sample touches can live in up to 8 blocks -- take the worst bound).
        for (const maths::vec3& p : surfacePoints) {
            const float fx = std::clamp((p.x - sdf.volumeMin.x) / sdf.voxelSize - 0.5f, 0.0f, static_cast<float>(sdf.resolution - 1u));
            const float fy = std::clamp((p.y - sdf.volumeMin.y) / sdf.voxelSize - 0.5f, 0.0f, static_cast<float>(sdf.resolution - 1u));
            const float fz = std::clamp((p.z - sdf.volumeMin.z) / sdf.voxelSize - 0.5f, 0.0f, static_cast<float>(sdf.resolution - 1u));
            const uint32_t x0 = static_cast<uint32_t>(fx), y0 = static_cast<uint32_t>(fy), z0 = static_cast<uint32_t>(fz);
            float bound = 0.0f;
            for (uint32_t dz = 0; dz <= 1u; ++dz)
                for (uint32_t dy = 0; dy <= 1u; ++dy)
                    for (uint32_t dx = 0; dx <= 1u; ++dx)
                        bound = std::max(bound, BlockDecodeErrorBound(sdf,
                            std::min(x0 + dx, sdf.resolution - 1u),
                            std::min(y0 + dy, sdf.resolution - 1u),
                            std::min(z0 + dz, sdf.resolution - 1u)));

            const float d = geometry::SampleMeshSDF(sdf, p);
            Check(std::fabs(d) <= bound + rawTolerance,
                "TestCubeSurfaceIsZero: COMPRESSED distance must be 0 on the surface within the decode bound (got " +
                std::to_string(d) + ", bound " + std::to_string(bound) + ")");
        }

        // 2. Sign correctness.
        Check(geometry::SampleMeshSDF(sdf, maths::vec3{ 0.0f, 0.0f, 0.0f }) < 0.0f,
            "TestCubeSurfaceIsZero: the cube center must be strictly inside (negative distance)");
        Check(geometry::SampleMeshSDF(sdf, maths::vec3{ half + 2.0f * sdf.voxelSize, 0.0f, 0.0f }) > 0.0f,
            "TestCubeSurfaceIsZero: a point 2 voxels outside the +X face must be strictly outside (positive distance)");

        // 3. Analytic magnitude near the +X face: d(x, 0, 0) = x - half for x slightly beyond the
        // face (nearest feature is the face plane). Tolerance: compressed decode bound + a voxel
        // fraction for trilinear curvature (none here -- plane -- so the bound dominates).
        const float probeOffset = 1.5f * sdf.voxelSize; // Inside the +/-4-voxel encoded band.
        const maths::vec3 probe{ half + probeOffset, 0.0f, 0.0f };
        const float probeD = geometry::SampleMeshSDF(sdf, probe);
        const float probeBound = BlockDecodeErrorBound(sdf,
            static_cast<uint32_t>(std::clamp((probe.x - sdf.volumeMin.x) / sdf.voxelSize - 0.5f, 0.0f, static_cast<float>(sdf.resolution - 1u))),
            sdf.resolution / 2u, sdf.resolution / 2u);
        Check(std::fabs(probeD - probeOffset) <= probeBound + rawTolerance,
            "TestCubeSurfaceIsZero: analytic distance near the +X face must match (expected " +
            std::to_string(probeOffset) + ", got " + std::to_string(probeD) + ")");

        // 4. Compression ratio: (res/4)^3 blocks of 26 bytes vs res^3 raw bytes (8-bit) -> 2.46:1.
        const size_t compressedBytes = sdf.blocks.size() * sizeof(geometry::MeshSDFBlock);
        const size_t rawEightBitBytes = static_cast<size_t>(sdf.resolution) * sdf.resolution * sdf.resolution;
        Check(compressedBytes * 2u < rawEightBitBytes,
            "TestCubeSurfaceIsZero: block compression must at least halve the 8-bit grid size");

        // 5. Exhaustive voxel-level round trip: |decoded - clamped(raw)| <= per-block bound.
        for (uint32_t z = 0; z < sdf.resolution; ++z) {
            for (uint32_t y = 0; y < sdf.resolution; ++y) {
                for (uint32_t x = 0; x < sdf.resolution; ++x) {
                    const float rawClamped = std::clamp(
                        raw[(static_cast<size_t>(z) * sdf.resolution + y) * sdf.resolution + x],
                        -sdf.maxEncodedDistance, sdf.maxEncodedDistance);
                    const float decoded = geometry::DecodeMeshSDFVoxel(sdf, x, y, z);
                    if (std::fabs(decoded - rawClamped) > BlockDecodeErrorBound(sdf, x, y, z) + rawTolerance) {
                        Check(false, "TestCubeSurfaceIsZero: voxel (" + std::to_string(x) + "," +
                            std::to_string(y) + "," + std::to_string(z) + ") exceeds the decode error bound");
                    }
                }
            }
        }
    }

    void TestSphereSignAndSurface() {
        std::vector<maths::vec3> positions;
        std::vector<uint32_t> triangles;
        const float radius = 1.0f;
        MakeSphere(radius, 16u, 24u, positions, triangles);

        geometry::MeshSDF sdf = geometry::BuildMeshSDF(positions, triangles);
        Check(sdf.resolution != 0u, "TestSphereSignAndSurface: build must succeed");
        if (sdf.resolution == 0) return;

        Check(geometry::SampleMeshSDF(sdf, maths::vec3{ 0.0f, 0.0f, 0.0f }) < 0.0f,
            "TestSphereSignAndSurface: the sphere center must be strictly inside");
        Check(geometry::SampleMeshSDF(sdf, maths::vec3{ radius + 2.0f * sdf.voxelSize, 0.0f, 0.0f }) > 0.0f,
            "TestSphereSignAndSurface: a point outside the sphere must be strictly outside");

        // On the tessellated surface (triangle vertices ARE exactly on the mesh): |d| must stay
        // below one voxel -- looser than the cube's plane-exact case because the surface is
        // curved between voxel centers (trilinear underestimates curvature) and quantized.
        const float surfaceTolerance = sdf.voxelSize;
        for (size_t v = 0; v < positions.size(); v += 37) { // Strided subset: coverage without O(N) log spam.
            const float d = geometry::SampleMeshSDF(sdf, positions[v]);
            Check(std::fabs(d) <= surfaceTolerance,
                "TestSphereSignAndSurface: |distance| at a mesh vertex must be below one voxel (got " +
                std::to_string(d) + ")");
        }
    }

    void TestDegenerateInputs() {
        // Empty mesh -> resolution 0, no crash.
        geometry::MeshSDF empty = geometry::BuildMeshSDF({}, {});
        Check(empty.resolution == 0u && empty.blocks.empty(),
            "TestDegenerateInputs: an empty mesh must produce an empty (resolution 0) SDF");

        // Non-multiple-of-block resolution -> rejected.
        std::vector<maths::vec3> positions;
        std::vector<uint32_t> triangles;
        MakeCube(1.0f, positions, triangles);
        geometry::MeshSDF bad = geometry::BuildMeshSDF(positions, triangles, 30u);
        Check(bad.resolution == 0u,
            "TestDegenerateInputs: a resolution that is not a multiple of the block dim must be rejected");
    }

} // namespace

int main() {
    TestCubeSurfaceIsZero();
    TestSphereSignAndSurface();
    TestDegenerateInputs();

    if (g_FailCount == 0) {
        std::cout << "[MeshSDFTests] All checks PASSED.\n";
        return 0;
    }
    std::cerr << "[MeshSDFTests] " << g_FailCount << " check(s) FAILED.\n";
    return 1;
}
