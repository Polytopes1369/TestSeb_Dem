#include "geometry/MeshSDFGenerator.h"
#include "core/Logger.h"
#include "geometry/GeometryHashUtil.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <unordered_map>

namespace geometry {

    namespace {

        // Closest-feature classification for the pseudo-normal sign lookup: which simplex of the
        // triangle the closest point landed on decides WHICH pseudo-normal is the mathematically
        // correct one (face normal / edge pseudo-normal / vertex pseudo-normal).
        enum class TriangleFeature : uint8_t {
            Face,
            EdgeAB, EdgeBC, EdgeCA,
            VertexA, VertexB, VertexC,
        };

        // Ericson, "Real-Time Collision Detection", 5.1.5: exact closest point on triangle ABC to
        // point P via barycentric region tests, extended to also report the closest FEATURE
        // (needed by the angle-weighted pseudo-normal sign rule; the book's version only returns
        // the point).
        maths::vec3 ClosestPointOnTriangle(
            const maths::vec3& p, const maths::vec3& a, const maths::vec3& b, const maths::vec3& c,
            TriangleFeature& outFeature) {

            const maths::vec3 ab = b - a;
            const maths::vec3 ac = c - a;
            const maths::vec3 ap = p - a;

            const float d1 = ab.Dot(ap);
            const float d2 = ac.Dot(ap);
            if (d1 <= 0.0f && d2 <= 0.0f) { outFeature = TriangleFeature::VertexA; return a; }

            const maths::vec3 bp = p - b;
            const float d3 = ab.Dot(bp);
            const float d4 = ac.Dot(bp);
            if (d3 >= 0.0f && d4 <= d3) { outFeature = TriangleFeature::VertexB; return b; }

            const float vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                const float v = d1 / (d1 - d3);
                outFeature = TriangleFeature::EdgeAB;
                return a + ab * v;
            }

            const maths::vec3 cp = p - c;
            const float d5 = ab.Dot(cp);
            const float d6 = ac.Dot(cp);
            if (d6 >= 0.0f && d5 <= d6) { outFeature = TriangleFeature::VertexC; return c; }

            const float vb = d5 * d2 - d1 * d6;
            if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                const float w = d2 / (d2 - d6);
                outFeature = TriangleFeature::EdgeCA;
                return a + ac * w;
            }

            const float va = d3 * d6 - d5 * d4;
            if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                outFeature = TriangleFeature::EdgeBC;
                return b + (c - b) * w;
            }

            const float denom = 1.0f / (va + vb + vc);
            const float v = vb * denom;
            const float w = vc * denom;
            outFeature = TriangleFeature::Face;
            return a + ab * v + ac * w;
        }

        // Interior angle of the triangle corner at `apex` -- the weight the angle-weighted vertex
        // pseudo-normal assigns to this triangle's face normal at that vertex.
        float CornerAngle(const maths::vec3& apex, const maths::vec3& v1, const maths::vec3& v2) {
            const maths::vec3 e1 = (v1 - apex).Normalize();
            const maths::vec3 e2 = (v2 - apex).Normalize();
            return std::acos(std::clamp(e1.Dot(e2), -1.0f, 1.0f));
        }

        // Per-triangle acceleration record for the centroid-sphere lower-bound prune: any point
        // farther from the centroid than (bestDistance + radius) cannot possibly improve the
        // current best distance, so the exact closest-point test is skipped entirely.
        struct TrianglePruneRecord {
            maths::vec3 centroid;
            float radius = 0.0f;
        };

        // Everything the per-voxel query loop needs, precomputed once per mesh.
        struct SignedDistanceContext {
            const std::vector<maths::vec3>* positions = nullptr;
            const std::vector<uint32_t>* triangles = nullptr;
            std::vector<maths::vec3> faceNormals;                   // Unit, one per triangle (zero for degenerate).
            std::vector<maths::vec3> vertexPseudoNormals;           // Angle-weighted sums, NOT normalized (only the sign of a dot is read).
            std::unordered_map<uint64_t, maths::vec3> edgePseudoNormals; // Sum of both adjacent face normals.
            std::vector<TrianglePruneRecord> pruneRecords;
        };

        SignedDistanceContext BuildSignContext(
            const std::vector<maths::vec3>& positions, const std::vector<uint32_t>& triangles) {

            SignedDistanceContext ctx;
            ctx.positions = &positions;
            ctx.triangles = &triangles;

            const size_t triangleCount = triangles.size() / 3u;
            ctx.faceNormals.assign(triangleCount, maths::vec3{ 0.0f, 0.0f, 0.0f });
            ctx.vertexPseudoNormals.assign(positions.size(), maths::vec3{ 0.0f, 0.0f, 0.0f });
            ctx.pruneRecords.resize(triangleCount);
            ctx.edgePseudoNormals.reserve(triangleCount * 3u);

            for (size_t t = 0; t < triangleCount; ++t) {
                const uint32_t i0 = triangles[t * 3 + 0];
                const uint32_t i1 = triangles[t * 3 + 1];
                const uint32_t i2 = triangles[t * 3 + 2];
                const maths::vec3& a = positions[i0];
                const maths::vec3& b = positions[i1];
                const maths::vec3& c = positions[i2];

                const maths::vec3 rawNormal = (b - a).Cross(c - a);
                // Degenerate (zero-area) triangles keep a zero face normal: they contribute
                // nothing to any pseudo-normal, and their closest-point distance still counts.
                const maths::vec3 n = (rawNormal.Length() > 1.0e-12f) ? rawNormal.Normalize()
                    : maths::vec3{ 0.0f, 0.0f, 0.0f };
                ctx.faceNormals[t] = n;

                ctx.edgePseudoNormals[PackOrderedPair(i0, i1)] = ctx.edgePseudoNormals[PackOrderedPair(i0, i1)] + n;
                ctx.edgePseudoNormals[PackOrderedPair(i1, i2)] = ctx.edgePseudoNormals[PackOrderedPair(i1, i2)] + n;
                ctx.edgePseudoNormals[PackOrderedPair(i2, i0)] = ctx.edgePseudoNormals[PackOrderedPair(i2, i0)] + n;

                ctx.vertexPseudoNormals[i0] = ctx.vertexPseudoNormals[i0] + n * CornerAngle(a, b, c);
                ctx.vertexPseudoNormals[i1] = ctx.vertexPseudoNormals[i1] + n * CornerAngle(b, c, a);
                ctx.vertexPseudoNormals[i2] = ctx.vertexPseudoNormals[i2] + n * CornerAngle(c, a, b);

                TrianglePruneRecord& prune = ctx.pruneRecords[t];
                prune.centroid = (a + b + c) * (1.0f / 3.0f);
                prune.radius = std::max({ (a - prune.centroid).Length(),
                                          (b - prune.centroid).Length(),
                                          (c - prune.centroid).Length() });
            }
            return ctx;
        }

        // The pseudo-normal associated with the closest feature of triangle `t` -- see the file
        // header comment for why each feature type demands a different normal.
        maths::vec3 FeaturePseudoNormal(const SignedDistanceContext& ctx, size_t t, TriangleFeature feature) {
            const std::vector<uint32_t>& tris = *ctx.triangles;
            const uint32_t i0 = tris[t * 3 + 0];
            const uint32_t i1 = tris[t * 3 + 1];
            const uint32_t i2 = tris[t * 3 + 2];
            switch (feature) {
            case TriangleFeature::Face:    return ctx.faceNormals[t];
            case TriangleFeature::EdgeAB:  return ctx.edgePseudoNormals.at(PackOrderedPair(i0, i1));
            case TriangleFeature::EdgeBC:  return ctx.edgePseudoNormals.at(PackOrderedPair(i1, i2));
            case TriangleFeature::EdgeCA:  return ctx.edgePseudoNormals.at(PackOrderedPair(i2, i0));
            case TriangleFeature::VertexA: return ctx.vertexPseudoNormals[i0];
            case TriangleFeature::VertexB: return ctx.vertexPseudoNormals[i1];
            default:                       return ctx.vertexPseudoNormals[i2];
            }
        }

        // Exact signed distance from `p` to the mesh: minimum unsigned distance over every
        // triangle (with the centroid-sphere prune), sign from the closest feature's
        // pseudo-normal. Equidistant ties (within epsilon) are resolved toward the candidate
        // whose pseudo-normal alignment |dot| is strongest -- the numerically most trustworthy
        // sign witness, which is what makes the method stable across shared edges/vertices.
        float SignedDistanceToMesh(const SignedDistanceContext& ctx, const maths::vec3& p) {
            const std::vector<maths::vec3>& positions = *ctx.positions;
            const std::vector<uint32_t>& tris = *ctx.triangles;
            const size_t triangleCount = tris.size() / 3u;

            float bestDistSq = std::numeric_limits<float>::max();
            float bestSignDot = 0.0f;    // dot(p - closestPoint, pseudoNormal) of the current winner.
            float bestAbsSignDot = -1.0f;

            for (size_t t = 0; t < triangleCount; ++t) {
                // Lower-bound prune: distance to this triangle is at least
                // |p - centroid| - radius; if that already exceeds the best, skip the exact test.
                const TrianglePruneRecord& prune = ctx.pruneRecords[t];
                const float centroidDist = (p - prune.centroid).Length();
                const float lowerBound = centroidDist - prune.radius;
                if (lowerBound > 0.0f && lowerBound * lowerBound > bestDistSq) {
                    continue;
                }

                TriangleFeature feature{};
                const maths::vec3 closest = ClosestPointOnTriangle(
                    p, positions[tris[t * 3 + 0]], positions[tris[t * 3 + 1]], positions[tris[t * 3 + 2]], feature);
                const maths::vec3 delta = p - closest;
                const float distSq = delta.Dot(delta);

                // Relative epsilon for the "same distance" tie window: two triangles sharing the
                // closest edge/vertex produce identical distances up to rounding.
                const float tieEpsilon = 1.0e-9f + bestDistSq * 1.0e-5f;
                if (distSq < bestDistSq - tieEpsilon) {
                    bestDistSq = distSq;
                    bestSignDot = delta.Dot(FeaturePseudoNormal(ctx, t, feature));
                    bestAbsSignDot = std::fabs(bestSignDot);
                }
                else if (distSq <= bestDistSq + tieEpsilon) {
                    const float signDot = delta.Dot(FeaturePseudoNormal(ctx, t, feature));
                    if (std::fabs(signDot) > bestAbsSignDot) {
                        bestDistSq = std::min(bestDistSq, distSq);
                        bestSignDot = signDot;
                        bestAbsSignDot = std::fabs(signDot);
                    }
                }
            }

            const float dist = std::sqrt(bestDistSq);
            return (bestSignDot < 0.0f) ? -dist : dist;
        }

        // Maps a signed distance to the normalized [0, 1] band encoding (0.5 == surface).
        float NormalizeDistance(float signedDistance, float maxEncodedDistance) {
            const float clamped = std::clamp(signedDistance, -maxEncodedDistance, maxEncodedDistance);
            return clamped / (2.0f * maxEncodedDistance) + 0.5f;
        }

        uint8_t QuantizeNormalized(float normalized) {
            return static_cast<uint8_t>(std::lround(std::clamp(normalized, 0.0f, 1.0f) * 255.0f));
        }

    } // namespace

    MeshSDF BuildMeshSDF(
        const std::vector<maths::vec3>& positions,
        const std::vector<uint32_t>& triangles,
        uint32_t resolution,
        std::vector<float>* outRawDistances) {

        if (positions.empty() || triangles.size() < 3u ||
            resolution == 0u || (resolution % kSDFBlockDim) != 0u ||
            resolution <= 2u * kSDFVolumeMarginVoxels) {
            return MeshSDF{}; // Return empty MeshSDF.
        }

        LOG_INFO(std::format("[MeshSDFGenerator] Generating SDF (res: {}) for mesh (vertices: {}, triangles: {})...", resolution, positions.size(), triangles.size() / 3u));
        MeshSDF sdf;

        // --- Volume fit: cubic voxels, mesh AABB centered, kSDFVolumeMarginVoxels margin -------
        maths::vec3 boundsMin, boundsMax;
        maths::ResetAABB(boundsMin, boundsMax);
        for (const maths::vec3& p : positions) {
            maths::ExpandAABB(boundsMin, boundsMax, p);
        }
        const maths::vec3 extent = boundsMax - boundsMin;
        const float maxExtent = std::max({ extent.x, extent.y, extent.z, 1.0e-4f });

        // Voxel size chosen so the mesh's largest axis spans the interior (margin-free) voxels.
        const uint32_t interiorVoxels = resolution - 2u * kSDFVolumeMarginVoxels;
        sdf.resolution = resolution;
        sdf.voxelSize = maxExtent / static_cast<float>(interiorVoxels);
        sdf.maxEncodedDistance = static_cast<float>(kSDFNarrowBandVoxels) * sdf.voxelSize;

        const float volumeExtent = sdf.voxelSize * static_cast<float>(resolution);
        const maths::vec3 center = (boundsMin + boundsMax) * 0.5f;
        sdf.volumeMin = center - maths::vec3{ volumeExtent * 0.5f, volumeExtent * 0.5f, volumeExtent * 0.5f };

        // --- Sample the raw signed distances at every voxel CENTER -----------------------------
        const SignedDistanceContext ctx = BuildSignContext(positions, triangles);
        const size_t voxelCount = static_cast<size_t>(resolution) * resolution * resolution;
        std::vector<float> raw(voxelCount);

        for (uint32_t z = 0; z < resolution; ++z) {
            for (uint32_t y = 0; y < resolution; ++y) {
                for (uint32_t x = 0; x < resolution; ++x) {
                    const maths::vec3 voxelCenter = sdf.volumeMin + maths::vec3{
                        (static_cast<float>(x) + 0.5f) * sdf.voxelSize,
                        (static_cast<float>(y) + 0.5f) * sdf.voxelSize,
                        (static_cast<float>(z) + 0.5f) * sdf.voxelSize };
                    raw[(static_cast<size_t>(z) * resolution + y) * resolution + x] =
                        SignedDistanceToMesh(ctx, voxelCenter);
                }
            }
        }

        if (outRawDistances != nullptr) {
            *outRawDistances = raw;
        }

        // --- BC4-style block compression of the normalized band --------------------------------
        const uint32_t blocksPerAxis = resolution / kSDFBlockDim;
        sdf.blocks.resize(static_cast<size_t>(blocksPerAxis) * blocksPerAxis * blocksPerAxis);

        for (uint32_t bz = 0; bz < blocksPerAxis; ++bz) {
            for (uint32_t by = 0; by < blocksPerAxis; ++by) {
                for (uint32_t bx = 0; bx < blocksPerAxis; ++bx) {
                    // Gather the block's 64 normalized values, voxel-linear (x fastest).
                    float blockValues[kSDFBlockVoxels];
                    float blockMin = 1.0f;
                    float blockMax = 0.0f;
                    for (uint32_t lz = 0; lz < kSDFBlockDim; ++lz) {
                        for (uint32_t ly = 0; ly < kSDFBlockDim; ++ly) {
                            for (uint32_t lx = 0; lx < kSDFBlockDim; ++lx) {
                                const uint32_t gx = bx * kSDFBlockDim + lx;
                                const uint32_t gy = by * kSDFBlockDim + ly;
                                const uint32_t gz = bz * kSDFBlockDim + lz;
                                const float v = NormalizeDistance(
                                    raw[(static_cast<size_t>(gz) * resolution + gy) * resolution + gx],
                                    sdf.maxEncodedDistance);
                                blockValues[(lz * kSDFBlockDim + ly) * kSDFBlockDim + lx] = v;
                                blockMin = std::min(blockMin, v);
                                blockMax = std::max(blockMax, v);
                            }
                        }
                    }

                    MeshSDFBlock block{};
                    block.minValue = QuantizeNormalized(blockMin);
                    block.maxValue = QuantizeNormalized(blockMax);

                    // Re-derive the endpoints the decoder will actually see, so index selection
                    // minimizes error against the DEQUANTIZED endpoints, not the exact ones.
                    const float decodedMin = static_cast<float>(block.minValue) / 255.0f;
                    const float decodedMax = static_cast<float>(block.maxValue) / 255.0f;
                    const float range = decodedMax - decodedMin;
                    const float invRange = (range > 0.0f) ? (1.0f / range) : 0.0f;

                    for (uint32_t v = 0; v < kSDFBlockVoxels; ++v) {
                        const float t = std::clamp((blockValues[v] - decodedMin) * invRange, 0.0f, 1.0f);
                        const uint32_t index = static_cast<uint32_t>(std::lround(t * 7.0f)); // 3-bit selector.
                        // Little-endian 3-bit packing: selector v occupies bits [v*3, v*3+3).
                        const uint32_t bitOffset = v * 3u;
                        block.packedIndices[bitOffset >> 3u] |= static_cast<uint8_t>((index << (bitOffset & 7u)) & 0xFFu);
                        if ((bitOffset & 7u) > 5u) { // Selector straddles a byte boundary.
                            block.packedIndices[(bitOffset >> 3u) + 1u] |=
                                static_cast<uint8_t>(index >> (8u - (bitOffset & 7u)));
                        }
                    }

                    sdf.blocks[(static_cast<size_t>(bz) * blocksPerAxis + by) * blocksPerAxis + bx] = block;
                }
            }
        }

        LOG_INFO(std::format("[MeshSDFGenerator] SDF compression complete: voxelSize={:.4f}, maxDistance={:.4f}, {} blocks generated.", sdf.voxelSize, sdf.maxEncodedDistance, sdf.blocks.size()));
        return sdf;
    }

    float DecodeMeshSDFVoxel(const MeshSDF& sdf, uint32_t x, uint32_t y, uint32_t z) {
        const uint32_t blocksPerAxis = sdf.resolution / kSDFBlockDim;
        const uint32_t bx = x / kSDFBlockDim;
        const uint32_t by = y / kSDFBlockDim;
        const uint32_t bz = z / kSDFBlockDim;
        const MeshSDFBlock& block = sdf.blocks[(static_cast<size_t>(bz) * blocksPerAxis + by) * blocksPerAxis + bx];

        const uint32_t lx = x % kSDFBlockDim;
        const uint32_t ly = y % kSDFBlockDim;
        const uint32_t lz = z % kSDFBlockDim;
        const uint32_t v = (lz * kSDFBlockDim + ly) * kSDFBlockDim + lx;

        // Unpack the 3-bit selector (mirror of the little-endian packing in BuildMeshSDF).
        const uint32_t bitOffset = v * 3u;
        uint32_t bits = block.packedIndices[bitOffset >> 3u] >> (bitOffset & 7u);
        if ((bitOffset & 7u) > 5u) {
            bits |= static_cast<uint32_t>(block.packedIndices[(bitOffset >> 3u) + 1u]) << (8u - (bitOffset & 7u));
        }
        const uint32_t index = bits & 0x7u;

        const float decodedMin = static_cast<float>(block.minValue) / 255.0f;
        const float decodedMax = static_cast<float>(block.maxValue) / 255.0f;
        const float normalized = decodedMin + (decodedMax - decodedMin) * (static_cast<float>(index) / 7.0f);

        // Back from the [0,1] band encoding to a world-unit signed distance.
        return (normalized - 0.5f) * 2.0f * sdf.maxEncodedDistance;
    }

    float SampleMeshSDF(const MeshSDF& sdf, const maths::vec3& localPos) {
        // Continuous voxel-center coordinates: voxel (i,j,k)'s center sits at index (i,j,k) here.
        const float fx = (localPos.x - sdf.volumeMin.x) / sdf.voxelSize - 0.5f;
        const float fy = (localPos.y - sdf.volumeMin.y) / sdf.voxelSize - 0.5f;
        const float fz = (localPos.z - sdf.volumeMin.z) / sdf.voxelSize - 0.5f;

        const float maxIndex = static_cast<float>(sdf.resolution - 1u);
        const float cx = std::clamp(fx, 0.0f, maxIndex);
        const float cy = std::clamp(fy, 0.0f, maxIndex);
        const float cz = std::clamp(fz, 0.0f, maxIndex);

        const uint32_t x0 = static_cast<uint32_t>(cx);
        const uint32_t y0 = static_cast<uint32_t>(cy);
        const uint32_t z0 = static_cast<uint32_t>(cz);
        const uint32_t x1 = std::min(x0 + 1u, sdf.resolution - 1u);
        const uint32_t y1 = std::min(y0 + 1u, sdf.resolution - 1u);
        const uint32_t z1 = std::min(z0 + 1u, sdf.resolution - 1u);

        const float tx = cx - static_cast<float>(x0);
        const float ty = cy - static_cast<float>(y0);
        const float tz = cz - static_cast<float>(z0);

        // Standard trilinear filtering over the 8 surrounding voxel centers.
        const float c000 = DecodeMeshSDFVoxel(sdf, x0, y0, z0);
        const float c100 = DecodeMeshSDFVoxel(sdf, x1, y0, z0);
        const float c010 = DecodeMeshSDFVoxel(sdf, x0, y1, z0);
        const float c110 = DecodeMeshSDFVoxel(sdf, x1, y1, z0);
        const float c001 = DecodeMeshSDFVoxel(sdf, x0, y0, z1);
        const float c101 = DecodeMeshSDFVoxel(sdf, x1, y0, z1);
        const float c011 = DecodeMeshSDFVoxel(sdf, x0, y1, z1);
        const float c111 = DecodeMeshSDFVoxel(sdf, x1, y1, z1);

        const float c00 = c000 + (c100 - c000) * tx;
        const float c10 = c010 + (c110 - c010) * tx;
        const float c01 = c001 + (c101 - c001) * tx;
        const float c11 = c011 + (c111 - c011) * tx;
        const float c0 = c00 + (c10 - c00) * ty;
        const float c1 = c01 + (c11 - c01) * ty;
        return c0 + (c1 - c0) * tz;
    }

}
