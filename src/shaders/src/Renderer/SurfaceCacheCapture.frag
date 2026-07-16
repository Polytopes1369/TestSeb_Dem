#version 460
#extension GL_GOOGLE_include_directive : enable

// Surface Cache capture fragment shader (see renderer::SurfaceCachePass): writes albedo/normal/
// emissive/direct-lighting/radiance/world-position for one texel of the global surface-cache
// atlas, at whatever position within the currently-bound Card's exclusive rect this invocation's
// pixel maps to (the render area/viewport/scissor are all set to exactly that rect by
// RecordCapture(), so gl_FragCoord never needs to be consulted here). This codebase has no
// texture/material-binding system (see ClusterResolve.comp's own comment) -- this reuses the
// exact same procedural-material approach every other shading pass already uses
// (procedural_material.glsl's HashID/HsvToRgb, keyed by entityID) plus a small triplanar
// value-noise modulation so a captured card is not perfectly flat-shaded.
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

// Octahedral encoding of a unit vector into [0,1]^2 -- the same compact normal encoding this
// codebase already uses for cluster vertex normals (geometry::ClusterVertexNormal /
// GeometryEncoding.h), applied here to the captured world-space surface normal.
vec2 OctEncode(vec3 n) {
    vec2 p = n.xy * (1.0 / (abs(n.x) + abs(n.y) + abs(n.z)));
    if (n.z < 0.0) {
        vec2 signP = vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
        p = (1.0 - abs(p.yx)) * signP;
    }
    return p * 0.5 + 0.5;
}

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

    // Stable per-entity hue (matches ClusterResolve.comp / draw.frag's own procedural look), plus
    // a small triplanar value-noise modulation from world position so a card is not perfectly
    // flat-colored.
    float hue = float(HashID(pc.entityID) & 0xFFFFu) / 65536.0;
    vec3 baseColor = HsvToRgb(vec3(hue, 0.55, 0.85));

    vec3 blend = abs(n);
    blend /= (blend.x + blend.y + blend.z + 1e-5);
    float noise =
        Hash(inWorldPos.yz) * blend.x +
        Hash(inWorldPos.xz) * blend.y +
        Hash(inWorldPos.xy) * blend.z;
    vec3 albedo = baseColor * mix(0.85, 1.15, noise);

    outAlbedo = vec4(clamp(albedo, 0.0, 1.0), 1.0);
    outNormal = vec4(OctEncode(n), 0.0, 1.0);

    // Subtle procedural emissive tint, distinct per entity, so the channel is exercised
    // end-to-end rather than always writing zero -- no material system flags entities as
    // emissive/non-emissive yet (geometry::EntityMaterialProperties only carries WPO/mask data),
    // so every card gets the same small glow rather than an arbitrary on/off split.
    vec3 emissive = baseColor * 0.04;
    outEmissive = vec4(emissive, 1.0);

    vec3 directLighting = ComputeDirectLighting(inWorldPos, n);
    outDirectLighting = vec4(directLighting, 1.0);

    // Fold albedo into the direct-lighting term here (see this shader's own "Radiance + world
    // position" header comment) -- the combined, GI-ready outgoing radiance for this texel.
    outRadiance = vec4(emissive + albedo * directLighting, 1.0);
    outWorldPos = vec4(inWorldPos, 1.0);
}
