#version 460
#extension GL_GOOGLE_include_directive : enable

// PCG Instance Draw Path (Phase 0.2, UE5.8-parity PCG roadmap) -- see PcgInstanceDraw.vert's own
// header comment for the full pipeline contract. This fragment shader deliberately implements only
// simplified direct-lighting-only shading (one fixed sun direction/color + a flat ambient term, no
// shadows, no Surface Cache GI, no MegaLights) -- Phase 0.2's own "prove the plumbing" scope (see
// renderer::PcgInstanceDrawPass's own class comment): the goal is proving real Nanite cluster
// geometry rasterizes through this new instance pool with correct per-instance transforms and
// per-cluster materials, not full parity with the shared opaque Nanite resolve pipeline's Substrate
// BSDF (ClusterResolve.comp). A future phase wiring PCG instances into that shared path (e.g. by
// emitting them as additional ClusterCullMetadata candidates alongside the fixed scene, once a real
// per-instance-transform mechanism exists there too) would get full lighting for free instead of
// needing this shader at all.

#include "include/material_params.glsl"

layout(std430, set = 0, binding = 4) readonly buffer MaterialParamsSSBO {
    MaterialParams materials[MATERIAL_TABLE_SIZE];
} g_MaterialParams;

// Fixed lighting parameters, uploaded once at renderer::PcgInstanceDrawPass::Init() time (not
// re-uploaded per frame -- unlike the shared scene's real SceneLights, this pipeline has no live
// per-frame sun-direction consumer yet, see that class' own Init() comment).
layout(std140, set = 0, binding = 5) uniform PcgLightingParamsUBO {
    vec3 sunDirection; // Unit vector, points FROM the surface TOWARD the sun (already normalized CPU-side).
    float sunIntensity;
    vec3 sunColor;
    float _pad0;
    vec3 ambientColor;
    float _pad1;
} g_Lighting;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) flat in uint inMaterialID;

layout(location = 0) out vec4 outColor;

void main() {
    uint matID = min(inMaterialID, MATERIAL_TABLE_SIZE - 1u);
    MaterialParams mat = g_MaterialParams.materials[matID];

    vec3 N = normalize(inWorldNormal);
    float NdotL = max(dot(N, g_Lighting.sunDirection), 0.0);

    // Plain Lambertian diffuse + flat ambient + emissive -- every Phase 0.2 smoke-test material
    // (the streaming-pool Rock/Bush/Tree/Debris archetypes, renderer::kStreamingArchetypeMaterialIDBase)
    // is a non-metallic dielectric with a real diffuseAlbedo, so this reads correctly for them; a
    // fully metallic material (diffuseAlbedo == 0, see renderer::MakeBaseSlab's own comment) would
    // render solid black here, which is an accepted Phase 0.2 scope limitation (no specular term),
    // not a bug -- flagged for Phase 4 if metallic PCG content is ever needed.
    vec3 diffuse = mat.base.diffuseAlbedo * g_Lighting.sunColor * g_Lighting.sunIntensity * NdotL;
    vec3 ambient = mat.base.diffuseAlbedo * g_Lighting.ambientColor;
    vec3 color = diffuse + ambient + mat.base.emissive;

    outColor = vec4(color, 1.0);
}
