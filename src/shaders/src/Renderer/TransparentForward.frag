#version 460
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_query : require

// renderer::TransparentForwardPass's fragment stage: real Lambertian + sun-shadow shading (the
// exact same lighting model ClusterResolve.comp/ClusterResolveBinned.comp use for opaque surfaces,
// see either shader's own Step 3 comment) plus the material's own alpha, blended via this
// pipeline's fixed-function "over" blend (srcAlpha, oneMinusSrcAlpha) against whatever is already
// in the color attachment -- the opaque scene, already fully composited (GI/reflections included),
// see renderer::ClusterRenderPipeline::RecordFrame's own ordering comment for why this pass runs
// where it does. No opacity-mask/cutout sampling here (deliberately -- see TransparentForwardPass.h's
// class comment: masking and alpha-blending are both opacity mechanisms, and none of this demo's
// procedural primitives combine them today).
//
// MegaLights Phase A follow-up: this is translucent geometry, which has no GBuffer entry for
// renderer::MegaLightsPass's own compute-pass composite to reach (a Visibility Buffer stores
// exactly one winning OPAQUE surface per pixel) -- so the same RIS-weighted stochastic point-light
// selection + 1 ray-traced shadow-visibility ray is inlined directly here instead, via the shared
// megalights_ris.glsl (pure weighting/reservoir math) plus this shader's own TraceShadowRay copy
// (ray-tracing code is deliberately NOT shared across shaders in this codebase -- see megalights_ris
// .glsl's own header comment). No temporal/spatial reservoir reuse here either (Phase B), and no
// dedicated spatial denoiser (a forward-shaded surface has no persistent per-pixel history/GBuffer
// to filter against -- same accepted-noisier tradeoff Phase 5's glass reflections already documented
// for their own single-sample specular term).

#include "include/material_params.glsl"
#include "include/math_utils.glsl"
#include "include/megalights_types.glsl"

#define SHADOW_ATLAS_SET 0
#define SHADOW_ATLAS_BINDING 7
#define SHADOW_PAGE_TABLE_SET 0
#define SHADOW_PAGE_TABLE_BINDING 8
#define SHADOW_FEEDBACK_SET 0
#define SHADOW_FEEDBACK_BINDING 9
#define SHADOW_SUN_LEVELS_SET 0
#define SHADOW_SUN_LEVELS_BINDING 10
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"

#define MEGALIGHTS_LIGHTS_SET 0
#define MEGALIGHTS_LIGHTS_BINDING 12
#include "include/megalights_ris.glsl"

layout(set = 0, binding = 11) uniform accelerationStructureEXT g_TLAS;

layout(std140, set = 0, binding = 5) uniform TransparentViewParamsUBO {
    mat4 view; // Unused here (vertex-stage only) -- see TransparentForward.vert.
    mat4 proj;
    vec3 sunDirection;
    uint frameIndex; // Was a pure pad float -- see renderer::TransparentForwardPass.cpp's identical comment.
} g_ViewParams;

layout(std430, set = 0, binding = 6) readonly buffer MaterialParamsSSBO {
    MaterialParams params[];
} g_MaterialParams;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uint inMaterialID;

layout(location = 0) out vec4 outColor;

// Identical to MegaLightsShade.comp's own TraceShadowRay (a boolean any-hit-style occlusion query,
// not a full hit-surface reconstruction) -- duplicated per this codebase's established per-shader
// trace-code convention, see megalights_ris.glsl's own header comment.
bool TraceShadowRay(vec3 origin, vec3 dir, float tMax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, g_TLAS, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 1.0e-3, dir, tMax);
    while (rayQueryProceedEXT(rq)) {
        // Every BLAS triangle is unconditionally opaque -- nothing to confirm/reject manually.
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

void main() {
    uint materialSlot = min(inMaterialID, MATERIAL_TABLE_SIZE - 1u);
    MaterialParams mat = g_MaterialParams.params[materialSlot];

    vec3 normal = normalize(inNormal);
    // g_ViewParams.sunDirection points FROM the light TOWARD the scene -- see
    // ClusterResolve.comp's identical comment for the shared convention.
    vec3 lightDir = normalize(-g_ViewParams.sunDirection);
    float diff = max(dot(normal, lightDir), 0.0);
    float sunVisibility = (diff > 0.0) ? SampleSunShadowVSM(inWorldPos) : 1.0;

    // Metallic energy conservation -- see ClusterResolve.comp's identical fix (a pure metal has no
    // diffuse response). Transparent/translucent metals are not a category GenerateRandomMaterialTable
    // produces (metallic is forced 0.0 for the translucent/transparent categories), but the term is
    // kept for consistency with the opaque shaders' formula rather than assuming that invariant here.
    vec3 diffuseAlbedo = mat.baseColor * (1.0 - mat.metallic);
    vec3 finalColor = diffuseAlbedo * 0.15 + diffuseAlbedo * diff * sunVisibility + mat.emissive;

    // MegaLights Phase A follow-up: RIS-selected point light + 1 shadow-visibility ray, exactly
    // MegaLightsShade.comp's own algorithm -- see this file's own header comment.
    uint pixelSeed = uint(gl_FragCoord.y) * 65536u + uint(gl_FragCoord.x);
    uint selectedIndex;
    float invPdf;
    if (SelectLightRIS(inWorldPos, normal, pixelSeed, g_ViewParams.frameIndex, selectedIndex, invPdf)) {
        MegaLight light = g_Lights.lights[selectedIndex];
        vec3 toLight = light.position - inWorldPos;
        float dist = length(toLight);
        vec3 megaLightDir = toLight / max(dist, 1.0e-4);
        float megaNdotL = saturate(dot(normal, megaLightDir));

        vec3 biasedOrigin = inWorldPos + normal * 1.0e-2;
        bool occluded = (megaNdotL > 0.0) ? TraceShadowRay(biasedOrigin, megaLightDir, max(dist - 2.0e-2, 1.0e-3)) : true;
        float visibility = occluded ? 0.0 : 1.0;

        float distSq = max(dist * dist, 1.0e-4);
        float normalizedDist = saturate(dist / max(light.radius, 1.0e-4));
        float nd2 = normalizedDist * normalizedDist;
        float windowSq = 1.0 - nd2 * nd2;
        float window = saturate(windowSq * windowSq);

        finalColor += diffuseAlbedo * light.color * light.intensity * megaNdotL / distSq * window * visibility * invPdf;
    }

    outColor = vec4(finalColor, mat.alpha);
}
