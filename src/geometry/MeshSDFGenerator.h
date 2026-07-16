#pragma once
// CPU Mesh SDF (Signed Distance Field) generator: samples a triangle mesh's volume as a cubic 3D
// grid (default 32x32x32) where every voxel stores the signed distance to the nearest surface
// point -- negative inside, positive outside, zero exactly on the surface. This is the per-object
// half of a Lumen-style two-level distance field GI setup: each procedural mesh gets its own
// compact Mesh SDF at build time, and a camera-centered Global SDF clipmap set composites them at
// runtime (min operator) for cone/sphere tracing without ever touching the heavy geometry.
//
// --- Distance & sign ---
// Unsigned distance is the exact point-to-triangle minimum over every triangle (Ericson's
// closest-point-on-triangle, with a centroid-sphere lower-bound prune so far triangles are
// skipped cheaply). The SIGN comes from the angle-weighted pseudo-normal method (Baerentzen &
// Aanaes, "Signed distance computation using the angle weighted pseudonormal", 2005): the sign of
// dot(p - closestPoint, pseudoNormal) where the pseudo-normal is the face normal when the closest
// feature is a face interior, the sum of both adjacent face normals when it is an edge, and the
// angle-weighted sum of all incident face normals when it is a vertex. Unlike ray-parity tests,
// this is exact for watertight meshes and degrades gracefully (locally consistent two-sided sign)
// on open meshes -- which matters here because aggressively QEM-simplified Fallback Meshes are not
// guaranteed watertight.
//
// --- Storage: 8-bit normalized, block-compressed (BC4-style) ---
// Distances are clamped to a narrow band [-maxEncodedDistance, +maxEncodedDistance] (the standard
// distance-field trick: beyond the band a consumer only needs "far", not "how far") and normalized
// to [0, 1] (0.5 == on-surface). The normalized grid is then compressed in 4x4x4 voxel blocks,
// each block storing two 8-bit normalized endpoints + 64 3-bit interpolation indices (8 evenly
// spaced reconstruction levels), i.e. exactly the BC4 single-channel scheme transposed to 3D:
// 26 bytes per 64 voxels vs 64 raw -- a 2.46:1 ratio on top of the float->8-bit 4:1.

#include <cstdint>
#include <vector>
#include "core/maths/Maths.h"

namespace geometry {

    // Default voxels per axis. Any multiple of kSDFBlockDim >= 2*kSDFVolumeMarginVoxels+kSDFBlockDim works.
    constexpr uint32_t kMeshSDFResolution = 32u;

    // Compression block edge (4x4x4 voxels per block, matching BC4's 4x4 texel footprint per axis).
    constexpr uint32_t kSDFBlockDim = 4u;
    constexpr uint32_t kSDFBlockVoxels = kSDFBlockDim * kSDFBlockDim * kSDFBlockDim;

    // Empty voxel shells kept between the mesh AABB and the volume border, so voxels just outside
    // the surface still hold meaningful (not clamped-by-truncation) distances.
    constexpr uint32_t kSDFVolumeMarginVoxels = 4u;

    // Narrow-band half-width, in voxels: distances are clamped to +/- this many voxel sizes.
    constexpr uint32_t kSDFNarrowBandVoxels = 4u;

#pragma pack(push, 1)
    // One 4x4x4 block, BC4-style: minValue/maxValue are 8-bit quantizations of the block's
    // normalized-distance endpoints; packedIndices holds 64 3-bit selectors (voxel-linear order
    // x fastest, then y, then z; little-endian bit packing), each reconstructing
    // lerp(min, max, index / 7).
    struct MeshSDFBlock {
        uint8_t minValue;
        uint8_t maxValue;
        uint8_t packedIndices[kSDFBlockVoxels * 3u / 8u]; // 64 * 3 bits = 24 bytes.
    };
#pragma pack(pop)
    static_assert(sizeof(MeshSDFBlock) == 26, "MeshSDFBlock size drifted from the expected 26 bytes");

    struct MeshSDF {
        uint32_t resolution = 0;        // Voxels per axis (cubic grid). 0 == empty/invalid.
        maths::vec3 volumeMin;          // Local-space corner of the volume (NOT the first voxel center).
        float voxelSize = 0.0f;         // Cubic voxel edge length; volume extent = resolution * voxelSize.
        float maxEncodedDistance = 0.0f;// Half-width of the encoded band, world units.
        std::vector<MeshSDFBlock> blocks; // (resolution/kSDFBlockDim)^3 blocks, block-linear order
                                          // (block x fastest, then block y, then block z).
    };

    // Builds the SDF for `positions`/`triangles` (triangle list, 3 indices per triangle) at the
    // given cubic resolution (must be a multiple of kSDFBlockDim). The volume is centered on the
    // mesh AABB with kSDFVolumeMarginVoxels of margin on every side. If `outRawDistances` is
    // non-null it receives the UNCOMPRESSED, UNCLAMPED signed distances (voxel-linear, x fastest)
    // -- the ground truth the unit test compares the compressed reconstruction against. Returns
    // an empty MeshSDF (resolution 0) for an empty mesh or a non-multiple-of-block resolution.
    MeshSDF BuildMeshSDF(
        const std::vector<maths::vec3>& positions,
        const std::vector<uint32_t>& triangles,
        uint32_t resolution = kMeshSDFResolution,
        std::vector<float>* outRawDistances = nullptr);

    // Decodes one voxel's signed distance (world units, clamped to the encoded band) straight
    // from the compressed blocks -- the CPU mirror of what the GPU decompression does.
    float DecodeMeshSDFVoxel(const MeshSDF& sdf, uint32_t x, uint32_t y, uint32_t z);

    // Trilinearly filtered signed distance (world units) at a local-space position, decoded from
    // the compressed blocks. Positions outside the volume are clamped to the border voxel centers
    // (consumers add their own distance-to-volume term for far-field queries).
    float SampleMeshSDF(const MeshSDF& sdf, const maths::vec3& localPos);

}
