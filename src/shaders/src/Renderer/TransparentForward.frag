#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require

// renderer::TransparentForwardPass's fragment stage. Phase 5 (UE5.8 parity roadmap, translucency):
// upgraded from flat sun-only Lambertian to direct lighting (sun + point lights, both shadowed --
// ComputeDirectLighting below, ported from SurfaceCacheCapture.frag's own identically-named
// function) plus indirect diffuse sampled from the World Probe Grid (applied to every transparent
// material -- one 3D texture sample, cheap enough not to gate). A traced front-layer specular
// reflection (Lumen-style: single GGX-VNDF sample, HWRT/SWRT, ported from the old standalone
// TranslucentForwardPass this phase supersedes) is gated PER-MATERIAL by MaterialParams.
// hasReflections -- matching real UE5.8's "Output Reflections" material toggle (reserved for
// glass/water-like materials, not applied to every alpha-blended surface uniformly; see
// renderer::MaterialParameters::hasReflections' own comment for the exact category split).
//
// --- Compositing ---
// diffuseLight (direct+indirect, tinted by albedo) is always weighted by the material's own alpha,
// same as before Phase 5. When hasReflections is set, the reflection term is NOT alpha-weighted
// (real glass's reflected light isn't attenuated by the pane's own transmission) and the output
// alpha is boosted by the Fresnel term at grazing angles (alpha = max(materialAlpha,
// fresnelReflectance)) -- the standard real-time glass technique: a pane viewed edge-on should
// look almost fully reflective/opaque even at low base alpha. Blended via this pipeline's existing
// fixed-function "over" blend (srcAlpha, oneMinusSrcAlpha) against the color attachment, unchanged.
//
// No temporal accumulation exists here (unlike ReflectionPass' ping-pong history) -- a forward-
// shaded surface has no persistent per-pixel buffer to accumulate into, so the reflection is
// visibly noisier than the opaque path's own reflections; an accepted, documented simplification
// (matches the superseded TranslucentForwardPass's own same call).

#include "include/material_params.glsl"
#include "include/math_utils.glsl"
#include "include/ggx_brdf.glsl"

#define SHADOW_ATLAS_SET 0
#define SHADOW_ATLAS_BINDING 7
#define SHADOW_PAGE_TABLE_SET 0
#define SHADOW_PAGE_TABLE_BINDING 8
#define SHADOW_FEEDBACK_SET 0
#define SHADOW_FEEDBACK_BINDING 9
#define SHADOW_SUN_LEVELS_SET 0
#define SHADOW_SUN_LEVELS_BINDING 10
#define SHADOW_POINT_FACES_SET 0
#define SHADOW_POINT_FACES_BINDING 11
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"
#include "include/shadow_point_sampling.glsl"

// std140-exact mirror of a point light's GPU layout (2 naturally-aligned vec4 blocks, no padding
// needed) -- same field shape as SurfaceCacheCapture.frag's own PointLightGPU.
struct PointLightGPU {
    vec4 positionAndRadius; // xyz = world position, w = radius (attenuation reaches ~0 here).
    vec4 colorAndIntensity; // rgb = color, a = intensity.
};

// Mirrors renderer::TransparentForwardPass.cpp's own TransparentViewParams field-for-field (std140).
// Phase 5: grew from a bare sunDirection to a camera position (feeds the reflection trace's view-
// direction reconstruction) plus the full renderer::SceneLights shape, matching
// SurfaceCacheLightingUBO's own layout (SurfaceCacheCapture.frag) for the lighting portion so
// ComputeDirectLighting below is a straight port, not a re-derivation.
layout(std140, set = 0, binding = 5) uniform TransparentViewParamsUBO {
    mat4 view; // Unused here (vertex-stage only) -- see TransparentForward.vert.
    mat4 proj;
    vec3 cameraPositionWorld;
    float _pad0;
    vec4 sunDirectionAndIntensity; // xyz = direction (points FROM the light TOWARD the scene), w = intensity.
    vec4 sunColor;                 // rgb = color, a unused.
    uint pointLightCount;
    uvec3 _pad1;
    PointLightGPU pointLights[8]; // Must match renderer::kMaxPointLights.
} g_ViewParams;

layout(std430, set = 0, binding = 6) readonly buffer MaterialParamsSSBO {
    MaterialParams params[];
} g_MaterialParams;

#define WORLD_PROBE_GRID_SET 0
#define WORLD_PROBE_GRID_BINDING 13
#define WORLD_PROBE_GRID_PARAMS_BINDING 12
#include "include/world_probe_sampling.glsl"

// HWRT resources: own binding 14, NOT part of SurfaceCacheTraceContext's set 1/2 -- mirrors
// renderer::ReflectionTrace.comp's own set-0 TLAS binding exactly (SurfaceCacheTraceContext only
// covers the SWRT mesh-SDF-trace set (1) and the Surface Cache sampling set (2), never the
// TLAS/Fallback-Mesh buffers a HWRT consumer needs).
layout(set = 0, binding = 14) uniform accelerationStructureEXT g_TLAS;

#define FALLBACK_GEOMETRY_SET 0
#define FALLBACK_GEOMETRY_BASE_BINDING 15
#include "include/fallback_geometry.glsl"

#include "include/mesh_sdf_trace.glsl"        // set 1
#include "include/surface_cache_sampling.glsl" // set 2

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uint inMaterialID;

layout(push_constant) uniform TransparentPushConstants {
    uint entityCount; // Dynamically uniform -- see mesh_sdf_trace.glsl's own requirement.
    uint traceMode;   // 0 = SWRT (TraceMeshSDFScene), 1 = HWRT (TraceHWRT, inline ray query).
    uint frameIndex;
} pc;

layout(location = 0) out vec4 outColor;

// Direct lighting for this fragment: the sun (shadowed via SampleSunShadowVSM) plus every active
// point light (also shadowed, via SamplePointShadowVSM) -- ported verbatim from
// SurfaceCacheCapture.frag's own identically-named function, reading g_ViewParams instead of that
// shader's SurfaceCacheLightingUBO (same layout, see this file's own header comment).
vec3 ComputeDirectLighting(vec3 worldPos, vec3 n) {
    vec3 lighting = vec3(0.0);

    // g_ViewParams.sunDirectionAndIntensity.xyz points FROM the light TOWARD the scene -- negate
    // for the surface-to-light direction Lambertian shading needs.
    vec3 sunDir = normalize(-g_ViewParams.sunDirectionAndIntensity.xyz);
    float sunNdotL = max(dot(n, sunDir), 0.0);
    if (sunNdotL > 0.0) {
        float visibility = SampleSunShadowVSM(worldPos);
        lighting += g_ViewParams.sunColor.rgb * g_ViewParams.sunDirectionAndIntensity.w * sunNdotL * visibility;
    }

    for (uint i = 0u; i < g_ViewParams.pointLightCount; ++i) {
        vec3 lightPos = g_ViewParams.pointLights[i].positionAndRadius.xyz;
        float radius = max(g_ViewParams.pointLights[i].positionAndRadius.w, 1.0e-3);

        vec3 toLight = lightPos - worldPos;
        float distSq = dot(toLight, toLight);
        float dist = sqrt(distSq);
        vec3 lightDir = toLight / max(dist, 1.0e-5);

        float ndotl = max(dot(n, lightDir), 0.0);
        if (ndotl <= 0.0) {
            continue;
        }

        // Smooth windowed inverse-square falloff (Karis/Frostbite-style): exactly zero at
        // dist >= radius, while still behaving like a physical inverse-square falloff well inside.
        float windowed = clamp(1.0 - (distSq * distSq) / (radius * radius * radius * radius), 0.0, 1.0);
        float atten = (windowed * windowed) / max(distSq, 1.0e-4);

        float visibility = SamplePointShadowVSM(i, lightPos, worldPos);

        vec3 lightColor = g_ViewParams.pointLights[i].colorAndIntensity.rgb;
        float intensity = g_ViewParams.pointLights[i].colorAndIntensity.a;
        lighting += lightColor * intensity * ndotl * atten * visibility;
    }

    return lighting;
}

vec3 FetchPosition(uint globalVertexIndex) {
    FallbackVertexGpu v = g_Vertices[globalVertexIndex];
    return vec3(v.posX, v.posY, v.posZ);
}

// Nth copy of this codebase's established inline-rayQuery HWRT trace (ReflectionTrace.comp,
// ScreenProbeTrace.comp, SurfaceCacheGIInject.comp, WorldProbeInject.comp each carry their own
// identical copy -- see ReflectionTrace.comp's own header comment on why a further copy here
// follows the existing convention rather than refactoring already-working files).
bool TraceHWRT(vec3 rayOrigin, vec3 rayDir, float tMax, out uint outEntityIndex, out vec3 outLocalPos, out vec3 outLocalNormal) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, g_TLAS, gl_RayFlagsOpaqueEXT, 0xFF, rayOrigin, 1.0e-3, rayDir, tMax);
    while (rayQueryProceedEXT(rq)) {
        // Every BLAS triangle is unconditionally opaque -- nothing to confirm/reject manually.
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return false;
    }

    outEntityIndex = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true));
    int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    float hitT = rayQueryGetIntersectionTEXT(rq, true);
    vec3 objOrigin = rayQueryGetIntersectionObjectRayOriginEXT(rq, true);
    vec3 objDir = rayQueryGetIntersectionObjectRayDirectionEXT(rq, true);
    outLocalPos = objOrigin + objDir * hitT;

    EntityDrawRangeGpu range = g_DrawRanges[outEntityIndex];
    uint triangleFirstIndex = range.firstIndex + uint(primitiveID) * 3u;
    uint i0 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 0u];
    uint i1 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 1u];
    uint i2 = uint(range.vertexOffset) + g_Indices[triangleFirstIndex + 2u];
    vec3 p0 = FetchPosition(i0);
    vec3 p1 = FetchPosition(i1);
    vec3 p2 = FetchPosition(i2);
    outLocalNormal = normalize(cross(p1 - p0, p2 - p0));
    return true;
}

void main() {
    uint materialSlot = min(inMaterialID, MATERIAL_TABLE_SIZE - 1u);
    MaterialParams mat = g_MaterialParams.params[materialSlot];

    vec3 n = normalize(inNormal);

    // --- Direct + indirect diffuse (always computed -- cheap, applies to every transparent
    // material regardless of hasReflections, see this file's own header comment). ---
    vec3 directLighting = ComputeDirectLighting(inWorldPos, n);
    vec3 indirectLighting = SampleWorldProbeGrid(inWorldPos);
    // Metallic energy conservation -- see the pre-Phase-5 shader's identical comment (kept for
    // consistency with the opaque shaders' formula; metallic is always 0.0 for every transparent
    // category GenerateRandomMaterialTable produces).
    vec3 diffuseAlbedo = mat.baseColor * (1.0 - mat.metallic);
    vec3 diffuseLight = diffuseAlbedo * (directLighting + indirectLighting);

    vec3 outRGB = diffuseLight * mat.alpha + mat.emissive;
    float outAlpha = mat.alpha;

    // --- Optional front-layer specular reflection (Lumen-style, single GGX-VNDF sample) -- gated
    // per-material, see this file's own header comment. ---
    if (mat.hasReflections > 0.5) {
        vec3 viewDir = normalize(g_ViewParams.cameraPositionWorld - inWorldPos);
        float NdotV = max(dot(n, viewDir), 0.0);
        // Dielectric: F0 = 0.04 (standard non-metal baseline) -- the "Transparent: clear/glass-like"
        // category (the only one with hasReflections set) never sets metallic (see
        // MaterialParameterTable.h's own category).
        vec3 F0 = vec3(0.04);
        vec3 fresnel = F_Schlick(F0, NdotV);

        vec3 tangent, bitangent;
        BuildTangentBasis(n, tangent, bitangent);
        vec3 viewDirTangentSpace = vec3(dot(viewDir, tangent), dot(viewDir, bitangent), dot(viewDir, n));

        // Per-pixel-per-frame decorrelated jitter (gl_FragCoord takes the place of
        // ReflectionTrace.comp's gl_GlobalInvocationID -- same hashFloat construction).
        uint64_t pixelSeed = uint64_t(uint(gl_FragCoord.x)) | (uint64_t(uint(gl_FragCoord.y)) << 32);
        vec2 xi = vec2(
            hashFloat(pixelSeed, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0x517CC1B7UL),
            hashFloat(pixelSeed, uint64_t(pc.frameIndex) * 0x9E3779B97F4A7C15UL + 0xBF58476D1CE4E5B9UL));

        vec3 halfVectorTangentSpace = SampleGGXVNDF(xi, viewDirTangentSpace, mat.roughness);
        vec3 halfVectorWorld = normalize(
            tangent * halfVectorTangentSpace.x + bitangent * halfVectorTangentSpace.y + n * halfVectorTangentSpace.z);
        vec3 rayDir = reflect(-viewDir, halfVectorWorld);

        vec3 reflectionRadiance = vec3(0.0);
        // Grazing-angle GGX-VNDF edge case (sampled half-vector below the macro-surface plane) is
        // left as a miss (reflectionRadiance stays 0.0), same handling as ReflectionTrace.comp's own.
        if (dot(rayDir, n) > 0.0) {
            vec3 biasedOrigin = inWorldPos + n * 1.0e-2;

            uint hitEntityIndex;
            vec3 hitLocalPos, hitLocalNormal;
            bool hit;
            if (pc.traceMode == 0u) {
                hit = TraceMeshSDFScene(biasedOrigin, rayDir, kFarDistance, pc.entityCount, hitEntityIndex, hitLocalPos, hitLocalNormal);
            } else {
                hit = TraceHWRT(biasedOrigin, rayDir, kFarDistance, hitEntityIndex, hitLocalPos, hitLocalNormal);
            }

            if (hit) {
                EntityInfo hitEntity = g_Entities[hitEntityIndex];
                reflectionRadiance = SampleCardRadiance(hitLocalPos, hitLocalNormal, hitEntity.firstCardIndex, hitEntity.cardCount);
            }
        }

        // Reflected light is NOT alpha-weighted (real glass's reflection isn't attenuated by the
        // pane's own transmission) and boosts the output alpha at grazing angles -- see this
        // file's own header comment.
        outRGB += fresnel * reflectionRadiance;
        float fresnelScalar = clamp(max(fresnel.r, max(fresnel.g, fresnel.b)), 0.0, 1.0);
        outAlpha = clamp(max(mat.alpha, fresnelScalar), 0.0, 1.0);
    }

    outColor = vec4(outRGB, outAlpha);
}
