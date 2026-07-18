#ifndef LOCAL_FOG_VOLUMES_GLSL
#define LOCAL_FOG_VOLUMES_GLSL

// Local Fog Volumes (UE5.8 rendering-parity gap G8): a list of localized, oriented-box or sphere
// fog regions -- each with its own density / scattering tint / vertical falloff and an optional
// "receives sun shadow" flag -- injected ADDITIVELY into renderer::AtmosVolumetricFogPass's own
// camera-aligned froxel grid, ON TOP OF the existing global Exponential Height Fog + Froxel
// Volumetric Fog term (see AtmosVolumetricFog.comp's own InjectMediaProps/InjectLight). This is the
// per-voxel density-injection extension point the task calls for -- no second raymarch pass, fully
// consistent with the existing froxel grid, so the integrated result flows unchanged into
// PostProcessComposite.comp's own ApplyVolumetricFog().
//
// The includer must #define LOCAL_FOG_VOLUMES_SET and LOCAL_FOG_VOLUMES_BINDING before including
// this header (same "leave inclusion order / set-binding macro definition to the consumer"
// convention as shadow_sun_sampling.glsl / megalights_ris.glsl already establish).
//
// std430 layout -- byte-for-byte mirror of renderer::AtmosVolumetricFogPass.cpp's LocalFogVolumeGPU
// (6 x vec4 == 96 bytes, natural std430 stride, no padding). The orientation is stored as three
// world-space orthonormal axes (right / up / forward) rather than a quaternion or a matrix so the
// world->local transform below is three dot products with NO inverse-rotation / quaternion math.
struct LocalFogVolume {
    vec4 centerAndShape;   // xyz = world-space center, w = shape (0.0 = oriented box, 1.0 = sphere).
    vec4 halfExtents;      // xyz = box half-extents along the axes below (world units), w = sphere radius.
    vec4 axisRightDensity; // xyz = world-space right axis (unit), w = base extinction density (world^-1).
    vec4 axisUpFalloff;    // xyz = world-space up axis (unit), w = vertical density falloff rate.
    vec4 axisFwdEdge;      // xyz = world-space forward axis (unit), w = edge softness in [0, 1).
    vec4 colorShadow;      // rgb = scattering albedo / tint, w = receivesSunShadow flag (0.0 / 1.0).
};

layout(std430, set = LOCAL_FOG_VOLUMES_SET, binding = LOCAL_FOG_VOLUMES_BINDING) readonly buffer LocalFogVolumesSSBO {
    LocalFogVolume volumes[];
} g_LocalFogVolumes;

// Additive extinction density this volume contributes at `worldPos` -- 0.0 strictly outside the
// shape, otherwise (baseDensity * softEdgeWeight * verticalFalloff). Shared by BOTH the density-
// injection kernel (mode 0) and the lighting kernel (mode 1) of AtmosVolumetricFog.comp so the two
// see byte-for-byte identical shape/density evaluation (mode 1 multiplies the returned density by
// the volume's own scattering albedo + phase-weighted light to form its in-scattered radiance).
float EvaluateLocalFogVolumeDensity(LocalFogVolume v, vec3 worldPos) {
    vec3 offset = worldPos - v.centerAndShape.xyz;
    float edge = clamp(v.axisFwdEdge.w, 0.0, 0.999);

    float shapeWeight;
    float heightAboveBottom; // Distance above the volume's lowest point, along its up axis.

    if (v.centerAndShape.w < 0.5) {
        // --- Oriented box: project `offset` onto the volume's own orthonormal axes. ---
        vec3 local = vec3(dot(offset, v.axisRightDensity.xyz),
                          dot(offset, v.axisUpFalloff.xyz),
                          dot(offset, v.axisFwdEdge.xyz));
        vec3 halfExt = max(v.halfExtents.xyz, vec3(1.0e-4));
        vec3 n = abs(local) / halfExt; // [0, 1] on each axis while inside the box.
        if (n.x > 1.0 || n.y > 1.0 || n.z > 1.0) {
            return 0.0;
        }
        // Soft edge: fade toward each face across the outermost `edge` fraction of the box, so
        // volumes blend into the surrounding air instead of terminating on a hard slab boundary.
        vec3 fade = vec3(1.0) - smoothstep(vec3(1.0 - edge), vec3(1.0), n);
        shapeWeight = fade.x * fade.y * fade.z;
        heightAboveBottom = local.y + halfExt.y; // 0 at the bottom face, 2*halfExt.y at the top.
    } else {
        // --- Sphere: radial soft edge. ---
        float radius = max(v.halfExtents.w, 1.0e-4);
        float r = length(offset) / radius;
        if (r > 1.0) {
            return 0.0;
        }
        shapeWeight = 1.0 - smoothstep(1.0 - edge, 1.0, r);
        heightAboveBottom = offset.y + radius; // Measured from the sphere's lowest point (world up).
    }

    // Vertical falloff: like the global exponential height fog, density is densest at the volume's
    // base and thins upward, so a low `heightFalloff` reads as an evenly-filled body and a high one
    // as ground-hugging mist that pools in the lower part of the volume.
    float verticalFalloff = exp(-max(heightAboveBottom, 0.0) * max(v.axisUpFalloff.w, 0.0));
    return max(v.axisRightDensity.w, 0.0) * shapeWeight * verticalFalloff;
}

#endif // LOCAL_FOG_VOLUMES_GLSL
