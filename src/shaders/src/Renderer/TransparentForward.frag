#version 460
#extension GL_GOOGLE_include_directive : enable

// renderer::TransparentForwardPass's fragment stage: real Lambertian + sun-shadow shading (the
// exact same lighting model ClusterResolve.comp/ClusterResolveBinned.comp use for opaque surfaces,
// see either shader's own Step 3 comment) plus the material's own alpha, blended via this
// pipeline's fixed-function "over" blend (srcAlpha, oneMinusSrcAlpha) against whatever is already
// in the color attachment -- the opaque scene, already fully composited (GI/reflections included),
// see renderer::ClusterRenderPipeline::RecordFrame's own ordering comment for why this pass runs
// where it does. No opacity-mask/cutout sampling here (deliberately -- see TransparentForwardPass.h's
// class comment: masking and alpha-blending are both opacity mechanisms, and none of this demo's
// procedural primitives combine them today).

#include "include/material_params.glsl"

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

layout(std140, set = 0, binding = 5) uniform TransparentViewParamsUBO {
    mat4 view; // Unused here (vertex-stage only) -- see TransparentForward.vert.
    mat4 proj;
    vec3 sunDirection;
    float _pad0;
} g_ViewParams;

layout(std430, set = 0, binding = 6) readonly buffer MaterialParamsSSBO {
    MaterialParams params[];
} g_MaterialParams;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uint inMaterialID;

layout(location = 0) out vec4 outColor;

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

    outColor = vec4(finalColor, mat.alpha);
}
