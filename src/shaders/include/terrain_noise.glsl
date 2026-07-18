#ifndef TERRAIN_NOISE_GLSL
#define TERRAIN_NOISE_GLSL

// Phase 7b (UE5.8 parity roadmap, terrain heightfield): heightfield noise for the procedural
// terrain entity. Reuses displacement_noise.glsl's own HashUint3D/ValueNoise3D hash-mixer and
// interpolation verbatim (same avalanche mixer, same quality bar) -- only the fbm COMBINATION
// differs, since a heightfield needs a mountainous silhouette a plain smooth fbm alone can't give
// (a smooth fbm reads as "rolling hills", not "mountains").
#include "displacement_noise.glsl"
// Rivers/waterfalls feature: shared river-path math (control points, closest-point-on-polyline,
// channel blend mask) -- see that file's own header comment for why SampleTerrainHeight below is
// one of its three required call sites.
#include "river_spline.glsl"

// Kept low enough that the HIGHEST octave's wavelength still comfortably exceeds the terrain
// entity's own vertex spacing (VulkanContext::GenerateGeometry()'s terrain call site uses
// config::FLOOR_VERTEX_SPACING, 1.0 world unit/vertex, not the fine VERTEX_SPACING every curved
// hero primitive uses) -- 2 octaves doubling from 0.06 tops out at a ~8.3-unit wavelength (>8
// vertices/cycle), never at 0.05-scale detail this coarse a mesh can't actually represent. The
// original tuning (5 octaves from 0.20, topping out at 3.2 -> a ~0.3-unit wavelength, well BELOW
// vertex spacing) produced per-vertex-aliased "sandpaper" noise: every adjacent vertex pair reads
// as a sharp discontinuity to ClusterDAG's simplifier, which can never fold such clusters together
// (every merge hits its locked-vertex-saturation escape hatch) -- this is what caused a multi-
// minute DAG-build hang on the terrain's 90000-vertex mesh during Phase 7b's own verification.
const uint  kTerrainOctaves = 2u;
const float kTerrainBaseFrequency = 0.06;
// 0.0 (pure smooth fbm, no ridged folding): RidgedNoiseOctave's fold (1.0 - abs(noise*2-1)) is only
// C0 continuous -- it creates a genuine derivative discontinuity (a crease) at every zero-crossing,
// regardless of how low the base frequency is. ClusterDAG's simplifier assumes locally-smooth
// geometry to fold clusters together (the same reason a sphere/torus, though curved, simplifies
// fine -- they're C1-smooth everywhere); creases defeat that at THIS mesh's coarse
// FLOOR_VERTEX_SPACING (1.0 world unit/vertex, too coarse to actually resolve a crease as a crisp
// edge rather than jagged noise) no matter the frequency -- confirmed by testing: reducing octaves/
// frequency alone (see kTerrainBaseFrequency's own history in this file) was NOT enough to fix the
// resulting DAG-build blowup (763 un-mergeable roots for one 90000-vertex mesh, vs. tens for every
// other primitive) once ridging was still active. Kept as a blend weight (not deleted) so a future,
// finer-spacing terrain revision can safely reintroduce ridged creases if it wants a sharper look.
const float kTerrainRidgeWeight = 0.0;
// World units. Unlike the original Phase 7b scene (where primitives were placed to rest directly
// on the ground), this codebase's showcase gallery floats every zone primitive a fixed 0.8 world
// units above the floor plane (see VulkanContext::UpdateEntityRotations()'s floor case, kFloorTopY)
// without deriving per-entity ground clearance from the terrain height. Kept well under that 0.8
// margin so the undulating terrain reads as a gentle rolling backdrop and never pokes through a
// floating primitive; the old branch's own amplitude (2.2) assumed literal ground-resting placement
// this scene doesn't do -- see terrain_shading.glsl's own comment on kTerrainBaseWorldY.
const float kTerrainAmplitude = 0.4;

// Ridged noise octave: folds the [0,1) value-noise sample toward 1 at its zero-crossings, giving
// sharp creases instead of smooth bumps -- the standard "ridged multifractal" building block.
float RidgedNoiseOctave(vec3 p) {
    return 1.0 - abs(ValueNoise3D(p) * 2.0 - 1.0);
}

// Phase 7c (UE5.8 parity roadmap, water/erosion): domain warping -- perturbs the (x,z) sampling
// point by a second, much-lower-frequency noise sample before the main fbm evaluates it, giving
// the terrain a visibly eroded/warped look instead of a perfectly axis-aligned noise grid. Reuses
// ValueNoise3D's own 3D domain: TerrainHeightFbm always samples the y=0 plane of that volume, so
// this samples two OTHER decorrelated planes (y=7.0, y=13.0) for the X/Z offsets -- no second hash
// mixer needed. Safe at this terrain's own coarse vertex spacing (unlike the ridged component
// above): a LOW-frequency warp only distorts the macro shape's sampling coordinates, it adds no
// local high-frequency curvature of its own, so it does not reproduce the DAG-build blowup
// kTerrainRidgeWeight's own comment documents.
const float kWarpFrequency = 0.06;
const float kWarpAmplitude = 1.8;

vec2 DomainWarpOffset(vec2 xz) {
    float warpX = (ValueNoise3D(vec3(xz.x, 7.0,  xz.y) * kWarpFrequency) * 2.0 - 1.0) * kWarpAmplitude;
    float warpZ = (ValueNoise3D(vec3(xz.x, 13.0, xz.y) * kWarpFrequency) * 2.0 - 1.0) * kWarpAmplitude;
    return vec2(warpX, warpZ);
}

// Blends a smooth fbm (rolling base shape) with a ridged fbm (sharp mountain crests) so the result
// reads as genuinely mountainous rather than a uniformly undulating plane. Both sums share the same
// per-octave frequency/amplitude schedule (5 octaves, amplitude halves / frequency doubles), so
// they stay coherent with each other at every scale.
float TerrainHeightFbm(vec2 xz) {
    vec2 warped = xz + DomainWarpOffset(xz);
    float frequency = kTerrainBaseFrequency;
    float amplitude = 0.5;
    float smoothSum = 0.0;
    float ridgedSum = 0.0;
    float maxAmplitude = 0.0;
    for (uint i = 0u; i < kTerrainOctaves; ++i) {
        vec3 p = vec3(warped.x, 0.0, warped.y) * frequency;
        smoothSum += (ValueNoise3D(p) * 2.0 - 1.0) * amplitude;
        ridgedSum += RidgedNoiseOctave(p) * amplitude;
        maxAmplitude += amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    float smoothNorm = smoothSum / maxAmplitude;             // [-1, 1]
    float ridgedNorm = (ridgedSum / maxAmplitude) * 2.0 - 1.0; // [-1, 1]
    return mix(smoothNorm, ridgedNorm, kTerrainRidgeWeight);
}

// Public entry point: local-space height at local (x,z), in world units. Callers add their own
// world-space vertical anchor (this codebase's floor sits at kFloorTopY, not 0.0 -- see
// VulkanContext::GenerateGeometry()'s terrain call site) on top of this.
//
// Rivers/waterfalls feature: blends the ambient fbm toward river_spline.glsl's own authored,
// monotonic-downhill profile (RiverBedHeight) near the river path -- RiverChannelMask fades this
// from full effect at the channel centerline back to 0 (pure ambient fbm, unchanged) over
// kRiverBandOuterHalfWidth world units, comfortably wider than this terrain's own 4-world-unit
// generation vertex spacing (VulkanContext::GenerateGeometry's kTerrainVertexSpacing), so the
// blend itself never introduces a per-vertex-aliased discontinuity the way the old ridged-noise
// regression did (see kTerrainRidgeWeight's own comment) -- this is a deliberate, wide, low-
// frequency terrain-scale feature, not high-frequency noise. Blends toward the river profile in
// BOTH directions (raising ground near the hillside spring, lowering it into the valley/waterfall)
// rather than a one-sided "carve down only" clamp, since the spring's own authored height (2.2)
// sits well above the ambient terrain there -- see river_spline.glsl's own kRiverControlHeight
// comment.
float SampleTerrainHeight(vec2 xz) {
    float ambient = TerrainHeightFbm(xz) * kTerrainAmplitude;
    float mask = RiverChannelMask(xz);
    if (mask <= 0.0) {
        return ambient;
    }
    float bed = RiverBedHeight(xz);
    return mix(ambient, bed, mask);
}

#endif // TERRAIN_NOISE_GLSL
