#version 460
#extension GL_GOOGLE_include_directive : enable

// Surface Cache capture fragment shader (see renderer::SurfaceCachePass): writes albedo/normal/
// emissive/direct-lighting for one texel of the global surface-cache atlas, at whatever position
// within the currently-bound Card's exclusive rect this invocation's pixel maps to (the render
// area/viewport/scissor are all set to exactly that rect by RecordCapture(), so gl_FragCoord never
// needs to be consulted here). This codebase has no texture/material-binding system (see
// ClusterResolve.comp's own comment) -- this reuses the exact same procedural-material approach
// every other shading pass already uses (procedural_material.glsl's HashID/HsvToRgb, keyed by
// entityID) plus a small triplanar value-noise modulation so a captured card is not perfectly
// flat-shaded.
//
// --- Direct lighting ---
// outDirectLighting accumulates the sun (shadowed, via renderer::ShadowMapPass's depth map, PCF-
// filtered) plus every active point light (unshadowed, distance-attenuated -- see renderer::
// LightingTypes.h's PointLight comment for why) contribution, in radiance units (NOT yet
// multiplied by albedo -- that multiply happens in whatever future pass reads this atlas, exactly
// like a deferred-lighting G-buffer keeps albedo and lighting separate so lighting alone can be
// re-used/blurred/probed without re-deriving material color).

#include "include/procedural_material.glsl"
#include "include/math_utils.glsl"

struct PointLightGPU {
    vec4 positionAndRadius; // xyz = world position, w = radius (attenuation reaches ~0 here).
    vec4 colorAndIntensity; // rgb = color, a = intensity.
};

// Mirrors renderer::SurfaceCacheLightingUBO byte-for-byte (SurfaceCachePass.cpp) -- flat vec4/mat4
// fields and an explicit padding field throughout, matching this codebase's existing convention
// for CPU/GPU struct pairs (see GlobalSDFCompositePC's own comment) so std140 layout is unambiguous.
layout(set = 0, binding = 0, std140) uniform SurfaceCacheLightingUBO {
    mat4 lightViewProj;
    vec4 sunDirectionAndIntensity; // xyz = direction (points FROM the light TOWARD the scene), w = intensity.
    vec4 sunColor;                 // rgb = color, a unused.
    uint pointLightCount;
    uvec3 _pad0;
    PointLightGPU pointLights[8]; // Must match renderer::kMaxPointLights.
} uLighting;

layout(set = 0, binding = 1) uniform sampler2D uShadowMap;

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

// Manual PCF (3x3 box filter) shadow test against uShadowMap -- a plain (non-comparison) sampler,
// matching this codebase's existing convention (see HZBPass.cpp's own "Plain sampler2D, not
// shadow/compare sampler" note), so the depth comparison happens explicitly here instead of via
// VK_COMPARE_OP on the sampler. Returns 1.0 (fully lit) for a world position that projects outside
// the light's orthographic frustum entirely (see renderer::ShadowMapPass's class comment on its
// single whole-scene-fit map: a card position can legitimately fall outside it only through
// floating-point edge cases at the very boundary of the scene's bounding sphere).
float SampleSunShadow(vec3 worldPos) {
    vec4 lightClip = uLighting.lightViewProj * vec4(worldPos, 1.0);
    vec3 lightNDC = lightClip.xyz / lightClip.w;
    vec2 shadowUV = lightNDC.xy * 0.5 + 0.5;

    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
        lightNDC.z < 0.0 || lightNDC.z > 1.0) {
        return 1.0;
    }

    // Fixed depth bias: constant (not slope-scaled) is sufficient here because every occluder is
    // the exact same Fallback Mesh geometry the shadow map itself rasterized, at the exact same
    // static (local == world) positions -- there is no skinning/animation jitter to compensate for,
    // only ordinary depth-precision self-shadowing acne, which a small constant bias already fixes.
    const float kDepthBias = 0.0015;
    const float currentDepth = lightNDC.z;

    vec2 texelSize = 1.0 / vec2(textureSize(uShadowMap, 0));
    float shadow = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float closestDepth = texture(uShadowMap, shadowUV + vec2(x, y) * texelSize).r;
            shadow += (currentDepth - kDepthBias <= closestDepth) ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

// Direct lighting for one captured texel: the sun (shadowed, Lambertian) plus every active point
// light (unshadowed, Lambertian with a smooth distance-squared windowed falloff -- see
// renderer::LightingTypes.h's PointLight comment for why point lights are not shadowed here).
vec3 ComputeDirectLighting(vec3 worldPos, vec3 n) {
    vec3 lighting = vec3(0.0);

    // uLighting.sunDirectionAndIntensity.xyz points FROM the light TOWARD the scene (see
    // renderer::DirectionalLight's own comment) -- negate for the surface-to-light direction
    // Lambertian shading needs.
    vec3 sunDir = normalize(-uLighting.sunDirectionAndIntensity.xyz);
    float sunNdotL = max(dot(n, sunDir), 0.0);
    if (sunNdotL > 0.0) {
        float visibility = SampleSunShadow(worldPos);
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

        vec3 lightColor = uLighting.pointLights[i].colorAndIntensity.rgb;
        float intensity = uLighting.pointLights[i].colorAndIntensity.a;
        lighting += lightColor * intensity * ndotl * atten;
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
    outEmissive = vec4(baseColor * 0.04, 1.0);

    outDirectLighting = vec4(ComputeDirectLighting(inWorldPos, n), 1.0);
}
