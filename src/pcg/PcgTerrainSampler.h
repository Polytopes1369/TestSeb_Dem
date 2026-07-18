#pragma once

// PCG framework roadmap, Phase 2.2 ("Landscape / Terrain Sampler"): UE5.8 PCG's Landscape Sampler
// node equivalent -- scatters pcg::PcgPoint instances across the extent described by a
// pcg::PcgLandscapeData (Phase 1, src/pcg/PcgSpatialData.h) at a target point density, querying
// this codebase's existing GPU-driven procedural terrain heightfield (height + analytic normal) at
// each candidate position so downstream PCG nodes (a later phase's Slope Filter, Self-Pruning
// filter, etc.) can filter/place correctly -- e.g. exclude points on a cliff face.
//
// Two logically separate pieces live in this header/PcgTerrainSampler.cpp pair (kept in ONE
// file-pair rather than split into a PcgTerrainHeightCPU.h, since both halves are small and the
// sampler is the only consumer of the height port -- see this decision documented once more, in
// detail, above SampleTerrainHeightLocalCPU below):
//
//   1. A byte-for-byte-faithful CPU port of src/shaders/include/terrain_noise.glsl's
//      SampleTerrainHeight() (consumed GPU-side by src/shaders/src/PrimitiveGen/geom_terrain.comp
//      at terrain mesh-generation time) -- PcgSpatialData.h's own header comment states plainly
//      that no such CPU port existed before this phase; porting it was explicitly out of scope for
//      Phase 1 and is this phase's own unavoidable prerequisite.
//   2. The actual jittered-grid point-scatter sampler built on top of that port, producing
//      std::vector<pcg::PcgPoint> plus a parallel per-point slope array.
//
// Verification: this port was cross-checked against the REAL GPU shader (not just a glslc syntax
// check) -- a throwaway compute shader including the actual terrain_noise.glsl/displacement_noise.glsl
// headers unmodified was compiled with glslc and dispatched via a minimal headless Vulkan compute
// harness (NVIDIA GeForce RTX 5080 Laptop GPU) against 8 sample points, including several
// negative-coordinate cases. This cross-check caught a REAL bug in this port's first draft: GLSL's
// `uint(negativeFloatCellCoord)` cast (ValueNoise3D's cell-corner index computation) is
// implementation-defined/UB per both the GLSL and SPIR-V specs, and this port's first guess (wrap
// via a two's-complement bit-reinterpret, the "natural" portable-C++ choice) was WRONG -- it
// disagreed with the real GPU output by up to ~0.67 absolute at a negative sample point. The GPU
// actually SATURATES (clamps negative inputs to 0), matching NVIDIA's documented PTX
// `cvt.rzi.u32.f32` semantics -- see PcgTerrainSampler.cpp's CellCoordToUint for the corrected
// implementation and the full derivation. This means the port's exact negative-coordinate behavior
// is empirically matched to THIS GPU vendor's documented conversion semantics, not proven identical
// on every GPU vendor/driver (the GLSL/SPIR-V ambiguity is real) -- but SPIR-V float->unsigned
// conversion instructions are saturating on essentially every desktop GPU ISA in practice (NVIDIA,
// AMD, and Intel all document clamping, not wraparound, for this conversion), so this is expected to
// hold broadly, not just on the one GPU this was verified against. After the fix, all 8 GPU-vs-CPU
// height values matched to within ~5e-8 absolute (float32 rounding noise) -- see
// PcgTerrainSamplerTests.cpp's "GPU parity" test, which encodes the captured GPU reference values as
// a permanent regression check.

#include <cstdint>
#include <vector>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "pcg/PcgSpatialData.h"

namespace pcg {

    // ------------------------------------------------------------------------------------------
    // 1. Terrain-noise CPU port (mirrors terrain_noise.glsl / displacement_noise.glsl exactly).
    // ------------------------------------------------------------------------------------------

    // Local-space height at local (x, z), in world units -- byte-for-byte port of
    // terrain_noise.glsl's SampleTerrainHeight(vec2). "Local" here means the SAME pre-worldOffset
    // convention geom_terrain.comp itself uses (see that shader's own xPos/zPos, computed BEFORE
    // `v.position += vec3(worldOffsetX, worldOffsetY, worldOffsetZ)`): this function does not know
    // about, and must never be handed, a world-space coordinate directly. SampleHeightCPU below is
    // the world-space-facing convenience wrapper that performs the local/world conversion, mirroring
    // PcgLandscapeData's own documented contract (PcgSpatialData.h: "a future entity could subtract
    // its own worldOffsetX/Z before calling SampleTerrainHeight to recover the height at the correct
    // world point").
    float SampleTerrainHeightLocalCPU(float localX, float localZ);

    // World-space convenience wrapper: subtracts `terrain.worldOffset.x/z` to recover the local
    // sample coordinate, ports the local height, then re-adds `terrain.worldOffset.y` (the terrain
    // patch's own vertical anchor) so the return value is a genuine world-space Y. This is the
    // entry point the terrain sampler (and any future PCG node needing a terrain height query)
    // should call -- never SampleTerrainHeightLocalCPU directly, unless a caller already has a
    // pre-computed local coordinate.
    float SampleHeightCPU(const PcgLandscapeData& terrain, float worldX, float worldZ);

    // World units. geom_terrain.comp derives its own central-difference epsilon from mesh
    // tessellation (`width / widthSegments`, i.e. exactly one grid cell) -- PcgLandscapeData
    // deliberately carries NO segment-count field (Phase 1 scoped it as a pure reference/lookup
    // struct, not a mesh-authoring one), so this sampler cannot reproduce that literal formula. This
    // default instead matches terrain_noise.glsl's own documented target scale: that file's header
    // comment states its noise is tuned so the highest octave's wavelength comfortably exceeds
    // config::FLOOR_VERTEX_SPACING (1.0 world unit), so half of that spacing is a reasonable
    // resolution-independent default step for a finite-difference normal. Callers that DO know the
    // mesh tessellation they want to match (e.g. a future bake-time tool cross-checking an actual
    // generated terrain entity) should pass their own epsilon instead.
    constexpr float kDefaultTerrainNormalEpsilon = 0.5f;

    // World-space analytic central-difference normal at (worldX, worldZ) -- same TECHNIQUE as
    // geom_terrain.comp's own normal computation (central-difference tangent vectors along local
    // X/Z, cross-product them; valid because a heightfield's base normal is always +Y, unlike
    // displacement_noise.glsl's ComputeDisplacedNormal, which needs a general ONB since a hero
    // asset's base normal can point anywhere) -- see kDefaultTerrainNormalEpsilon's own comment for
    // why the literal epsilon VALUE necessarily differs from the GPU shader's mesh-resolution-derived
    // one.
    maths::vec3 ComputeTerrainNormalCPU(const PcgLandscapeData& terrain, float worldX, float worldZ,
        float epsilon = kDefaultTerrainNormalEpsilon);

    // Local slope, in radians, between a terrain normal and world-up (0,1,0) -- 0 on a perfectly
    // flat patch, PI/2 on a vertical cliff face. A future Phase 3 Slope Filter node reads exactly
    // this value (see PcgTerrainPointBatch::slopeRadians below), so it is exposed as its own small,
    // reusable function rather than being computed inline and discarded.
    float ComputeSlopeRadians(const maths::vec3& terrainNormal);

    // Builds a rotation quaternion that aligns local +Y (this codebase's convention for "up" --
    // matches PcgPoint's own identity-rotation default, and mat4::FromQuat/RotateVector's existing
    // basis) to `normal`. Small, generic "align to normal" utility -- Phase 2.1's Surface Sampler
    // (a parallel, independently-developed phase) may have already produced an equivalent helper by
    // the time this merges; per this phase's own task brief, re-deriving this small utility
    // independently (rather than blocking on that branch) is expected, not a duplication mistake.
    maths::quat QuatFromNormal(const maths::vec3& normal);

    // ------------------------------------------------------------------------------------------
    // 2. Jittered-grid terrain point scatter (the actual "Landscape Sampler" node).
    // ------------------------------------------------------------------------------------------

    // Parallel-array bundle: `points[i]`'s local slope (radians, see ComputeSlopeRadians above) is
    // `slopeRadians[i]`. Kept as a SEPARATE array rather than folded into PcgPoint itself -- PcgPoint
    // (Phase 1) has no dedicated slope field, and adding one there would touch a type Phase 2.1/2.3/
    // 2.4's parallel work also depends on; a future Phase 3 Slope Filter node is expected to consume
    // this parallel array directly (or fold it into a PcgAttributeSet per point, that node's own
    // choice to make, not this phase's).
    struct PcgTerrainPointBatch {
        std::vector<PcgPoint> points;
        std::vector<float> slopeRadians;
    };

    // Scatters points across `terrain`'s world-space extent ([worldOffset.x/z +/- width/2, length/2],
    // matching geom_terrain.comp's own local-grid-before-worldOffset convention) via a jittered grid:
    // the extent is divided into roughly `pointsPerSquareMeter`-density square cells, and exactly one
    // candidate point is placed per cell at a per-cell-seeded random jittered offset within that
    // cell. This is deliberately simple (grid-with-per-cell-jitter, not true blue-noise/Poisson-disc
    // sampling) -- per this phase's own task brief, Poisson-disc-quality point pruning is Phase 3's
    // Self-Pruning filter's job to layer on top of this sampler's raw output later, not this phase's.
    //
    // Determinism: for fixed (terrain, pointsPerSquareMeter, seed) inputs, every cell's jitter and
    // every output point's own `seed` field are derived from a per-cell PcgSeededRandom stream keyed
    // by PcgHashCombine(seed, cellIndex) (PcgSeededRandom.h) -- NOT from one shared stream advanced
    // cell-by-cell -- so the result is independent of iteration order and reproducible byte-for-byte
    // across runs/platforms given the same inputs (PcgSeededRandom's own header comment explains why
    // its stateless hash construction guarantees this).
    //
    // Returns an empty batch (zero points, not an error) if `terrain.width`/`terrain.length` are
    // non-positive or `pointsPerSquareMeter` is non-positive -- mirrors geom_terrain.comp's own
    // "reject degenerate Params, do nothing" validation convention (see that shader's own
    // `if (width <= 0.0 || length_ <= 0.0 || ...) return;`).
    PcgTerrainPointBatch SampleTerrainPoints(const PcgLandscapeData& terrain, float pointsPerSquareMeter,
        uint32_t seed, float normalEpsilon = kDefaultTerrainNormalEpsilon);

}
