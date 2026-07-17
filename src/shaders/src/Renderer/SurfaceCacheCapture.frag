#version 460
#extension GL_GOOGLE_include_directive : enable

// Surface Cache capture fragment shader (see renderer::SurfaceCachePass): writes albedo/normal/
// emissive/direct-lighting/radiance/world-position for one texel of the global surface-cache
// atlas, at whatever position within the currently-bound Card's exclusive rect this invocation's
// pixel maps to (the render area/viewport/scissor are all set to exactly that rect by
// RecordCapture(), so gl_FragCoord never needs to be consulted here).
//
// --- Material ---
// Samples the same Substrate material table every other shading pass in this codebase reads
// (material_params.glsl / substrate_bsdf.glsl -- see ClusterResolve.comp's own Step 3 comment):
// this texel's owning entity's real materialID is looked up via EntityDataBuffer (binding 6) and
// used to index g_MaterialParams (binding 7, the SAME SSBO renderer::ClusterResolvePass::Init()
// already fills), plus a small triplanar value-noise modulation on top of the real albedo so a
// captured card is not perfectly flat-shaded.
//
// --- Direct lighting ---
// outDirectLighting accumulates the sun (shadowed via renderer::VirtualShadowMapPass's 3-level
// clipmap, see shadow_sun_sampling.glsl) plus every active point light (ALSO shadowed, Phase 3
// onward -- via that same pass's per-light 6-face cube, see shadow_point_sampling.glsl; this was
// unshadowed pre-Phase-3, see renderer::LightingTypes.h's PointLight comment for the historical
// reason), in radiance units (NOT yet multiplied by albedo -- that multiply happens in whatever
// future pass reads this atlas, exactly like a deferred-lighting G-buffer keeps albedo and
// lighting separate so lighting alone can be re-used/blurred/probed without re-deriving material
// color).
//
// --- Radiance + world position ---
// outRadiance is exactly that "future pass": the albedo-multiplied direct lighting plus emissive,
// i.e. this texel's full outgoing radiance as of THIS capture -- what a GI trace (SWRT/HWRT, see
// SurfaceCacheSWRTPass / SurfaceCacheRayTracingPass) samples as "the luminance stored here," and
// what SurfaceCacheGIInject.comp later read-modify-writes a secondary bounce into on top of.
// outWorldPos is the plain world-space (== local-space, see this shader's own header comment on
// entity transforms) hit position, full precision -- the 3D origin a GI injection pass fires its
// hemisphere rays from, since the capture pass's own depth buffer is a same-lifetime scratch image
// with no sampled-read usage (see renderer::SurfaceCachePass's class comment).

#include "include/procedural_material.glsl"
#include "include/math_utils.glsl"
#include "include/octahedral.glsl"
#include "include/struct_custo.glsl"
#include "include/material_params.glsl"
#include "include/substrate_bsdf.glsl"

#define SHADOW_ATLAS_SET 0
#define SHADOW_ATLAS_BINDING 1
#define SHADOW_PAGE_TABLE_SET 0
#define SHADOW_PAGE_TABLE_BINDING 2
#define SHADOW_FEEDBACK_SET 0
#define SHADOW_FEEDBACK_BINDING 3
#define SHADOW_SUN_LEVELS_SET 0
#define SHADOW_SUN_LEVELS_BINDING 4
#define SHADOW_POINT_FACES_SET 0
#define SHADOW_POINT_FACES_BINDING 5
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"
#include "include/shadow_point_sampling.glsl"

struct PointLightGPU {
    vec4 positionAndRadius; // xyz = world position, w = radius (attenuation reaches ~0 here).
    vec4 colorAndIntensity; // rgb = color, a = intensity.
};

// Mirrors renderer::SurfaceCacheLightingUBO byte-for-byte (SurfaceCachePass.cpp) -- flat vec4/mat4
// fields and an explicit padding field throughout, matching this codebase's existing convention
// for CPU/GPU struct pairs (see GlobalSDFCompositePC's own comment) so std140 layout is unambiguous.
// No longer carries a lightViewProj (Phase 3) -- shadow_sun_sampling.glsl / shadow_point_sampling
// .glsl read renderer::VirtualShadowMapPass's own dedicated UBOs (bindings 4/5 above) directly.
layout(set = 0, binding = 0, std140) uniform SurfaceCacheLightingUBO {
    vec4 sunDirectionAndIntensity; // xyz = direction (points FROM the light TOWARD the scene), w = intensity.
    vec4 sunColor;                 // rgb = color, a unused.
    uint pointLightCount;
    uvec3 _pad0;
    PointLightGPU pointLights[8]; // Must match renderer::kMaxPointLights.
} uLighting;

// Substrate integration: this texel's owning entity's real materialID (EntityData.materialID),
// looked up by pc.entityID -- see this shader's own "Material" header comment.
layout(std430, set = 0, binding = 6) readonly buffer EntityDataBuffer {
    EntityData entityData[];
};

// renderer::GenerateShowcaseMaterialTable()'s result, the SAME SSBO ClusterResolve.comp's own
// g_MaterialParams reads (renderer::ClusterResolvePass::GetMaterialParamsBuffer()) -- see
// material_params.glsl's own comment.
layout(std430, set = 0, binding = 7) readonly buffer MaterialParamsSSBO {
    MaterialParams params[];
} g_MaterialParams;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;

layout(push_constant) uniform SurfaceCaptureConstants {
    mat4 viewProj;
    uint entityID;
} pc;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outEmissive;
layout(location = 3) out vec4 outDirectLighting;
layout(location = 4) out vec4 outRadiance;
layout(location = 5) out vec4 outWorldPos;

// Direct lighting for one captured texel: the sun (shadowed via SampleSunShadowVSM,
// shadow_sun_sampling.glsl) plus every active point light (Phase 3 onward: ALSO shadowed, via
// SamplePointShadowVSM, shadow_point_sampling.glsl -- both Lambertian, point lights additionally
// windowed by a smooth distance-squared falloff).
vec3 ComputeDirectLighting(vec3 worldPos, vec3 n) {
    vec3 lighting = vec3(0.0);

    // uLighting.sunDirectionAndIntensity.xyz points FROM the light TOWARD the scene (see
    // renderer::DirectionalLight's own comment) -- negate for the surface-to-light direction
    // Lambertian shading needs.
    vec3 sunDir = normalize(-uLighting.sunDirectionAndIntensity.xyz);
    float sunNdotL = max(dot(n, sunDir), 0.0);
    if (sunNdotL > 0.0) {
        float visibility = SampleSunShadowVSM(worldPos);
        lighting += uLighting.sunColor.rgb * uLighting.sunDirectionAndIntensity.w * sunNdotL * visibility;
    }

    for (uint i = 0u; i < uLighting.pointLightCount; ++i) {
        vec3 lightPos = uLighting.pointLights[i].positionAndRadius.xyz;
        float radius = max(uLighting.pointLights[i].positionAndRadius.w, 1.0e-3);

        vec3 toLight = lightPos - worldPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        vec3 lightDir = toLight / max(dist, 1.0e-5);

        float ndotl = max(dot(n, lightDir), 0.0);
        if (ndotl <= 0.0) {
            continue;
        }

        // Smooth windowed inverse-square falloff (Karis/Frostbite-style): exactly zero at
        // dist >= radius, matching PointLight::radius's own "attenuation reaches zero" contract,
        // while still behaving like a physical inverse-square falloff well inside that radius.
        float windowed = clamp(1.0 - (distSq * distSq) / (radius * radius * radius * radius), 0.0, 1.0);
        float atten = (windowed * windowed) / max(distSq, 1.0e-4);

        float visibility = SamplePointShadowVSM(i, lightPos, worldPos);

        vec3 lightColor = uLighting.pointLights[i].colorAndIntensity.rgb;
        float intensity = uLighting.pointLights[i].colorAndIntensity.a;
        lighting += lightColor * intensity * ndotl * atten * visibility;
    }

    return lighting;
}

void main() {
    vec3 n = normalize(inWorldNormal);

    // Real material lookup (see this shader's own "Material" header comment): this entity's
    // authoritative materialID, not entityID itself (do not assume the two coincide).
    EntityData ed = entityData[pc.entityID];
    uint materialSlot = min(ed.materialID, MATERIAL_TABLE_SIZE - 1u);
    MaterialParams mat = g_MaterialParams.params[materialSlot];

    // Small triplanar value-noise modulation from world position on top of the real albedo, so a
    // captured card is not perfectly flat-shaded (a legitimate "captured surface isn't flat"
    // detail, kept as-is).
    vec3 blend = abs(n);
    blend /= (blend.x + blend.y + blend.z + 1e-5);
    float noise =
        Hash(inWorldPos.yz) * blend.x +
        Hash(inWorldPos.xz) * blend.y +
        Hash(inWorldPos.xy) * blend.z;
    vec3 albedo = mat.base.diffuseAlbedo * mix(0.85, 1.15, noise);

    outAlbedo = vec4(clamp(albedo, 0.0, 1.0), 1.0);
    outNormal = vec4(OctEncode(n), 0.0, 1.0);

    // Real per-material emissive value (base.emissive + top.emissive*topWeight), see
    // substrate_bsdf.glsl's EvaluateSubstrateEmissive.
    vec3 emissive = EvaluateSubstrateEmissive(mat);
    outEmissive = vec4(emissive, 1.0);

    vec3 directLighting = ComputeDirectLighting(inWorldPos, n);
    outDirectLighting = vec4(directLighting, 1.0);

    // Fold albedo into the direct-lighting term here (see this shader's own "Radiance + world
    // position" header comment) -- the combined, GI-ready outgoing radiance for this texel.
    // Physically-based Lambertian normalization (2026-07-17 recalibration, see ClusterResolve.comp's
    // own identical comment): directLighting is now real illuminance in LUX, so the GI capture this
    // feeds (renderer::WorldProbeGridPass/ScreenProbeGIPass) stays on the SAME physical radiance
    // scale the direct-lit opaque/transparent passes use, instead of the indirect bounce reading
    // ~1000x brighter or dimmer than the direct light that produced it.
    const float kPI = 3.14159265359;
    outRadiance = vec4(emissive + (albedo / kPI) * directLighting, 1.0);
    outWorldPos = vec4(inWorldPos, 1.0);
}
