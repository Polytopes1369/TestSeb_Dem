#ifndef ATMOS_VOLUMETRIC_FOG_MAPPING_GLSL
#define ATMOS_VOLUMETRIC_FOG_MAPPING_GLSL

// Atmos weather system, Subtask 3 (atmos_integration_plan.md): the view-space-depth <-> froxel-Z
// mapping shared, byte-for-byte identical, by AtmosVolumetricFog.comp (the WRITER -- which world
// position each (x, y, z) froxel represents) and PostProcessComposite.comp (the READER -- which
// froxel-space W coordinate a given pixel's own view-space depth falls at). Exponential distribution
// (matches Unreal Engine 5.8's own Volumetric Fog depth slicing): most of the kFroxelDepthSliceCount
// slices sit close to the camera, where fog detail actually matters, instead of wasting resolution
// on the (visually much less fog-detail-sensitive) far distance a LINEAR distribution would give
// equal weight to.

const float ATMOS_FOG_NEAR_Z = 0.5;  // World units -- closer than this, every froxel maps to slice 0.
const float ATMOS_FOG_FAR_Z = 100.0; // World units -- beyond this, fog is fully accumulated (last slice).

// Froxel-space W in [0, 1] (continuous, NOT a slice index -- sampled via hardware trilinear
// filtering, so a fractional W blends between the two nearest slices) -> view-space depth (distance
// along the camera's forward axis, NOT Euclidean ray length -- matches AtmosVolumetricFog.comp's own
// InjectMediaProps world-position reconstruction).
float FroxelWToViewZ(float w) {
    return ATMOS_FOG_NEAR_Z * pow(ATMOS_FOG_FAR_Z / ATMOS_FOG_NEAR_Z, clamp(w, 0.0, 1.0));
}

// Inverse of FroxelWToViewZ -- used by PostProcessComposite.comp to find which froxel-space W a
// pixel's own reconstructed view-space depth falls at.
float ViewZToFroxelW(float viewZ) {
    float z = clamp(viewZ, ATMOS_FOG_NEAR_Z, ATMOS_FOG_FAR_Z);
    return log(z / ATMOS_FOG_NEAR_Z) / log(ATMOS_FOG_FAR_Z / ATMOS_FOG_NEAR_Z);
}

#endif // ATMOS_VOLUMETRIC_FOG_MAPPING_GLSL
