#version 460
#extension GL_GOOGLE_include_directive : enable

// Vegetation scatter (UE5.8 rendering-parity gap G2), instanced draw fragment stage. A deliberately
// LIGHTWEIGHT dedicated forward shading path (NOT the Nanite deferred visbuffer/ClusterResolve
// material path): the deferred path is keyed on cluster+triangle IDs written into the 64-bit
// visbuffer by the Nanite rasterizers, so feeding thousands of arbitrary instanced (non-cluster)
// triangles through it would mean inventing cluster IDs and re-rasterizing into that visbuffer --
// infeasible and wasteful at foliage instance counts. Instead this uses the exact forward lighting
// model renderer::TransparentForwardPass / TessellationPass / WaterForwardPass / ParticleSystemPass
// already use for their own non-cluster geometry -- VSM-shadowed sun + World Probe Grid indirect
// diffuse -- so the scatter is lit consistently with the rest of the scene (sun shadows + Lumen-
// style GI) at a fraction of the deferred path's per-instance cost.

#include "include/VegetationCommon.glsl"

layout(std140, set = 1, binding = 0) uniform RenderParamsUBO {
    mat4 viewProj;
    vec3 cameraPos; float _pad0;
    vec3 sunDirection; float sunIntensity; // sunDirection points FROM the light TOWARD the scene (same convention as SceneLights::sun).
    vec3 sunColor; float _pad1;
} g_Params;

// Virtual Shadow Map (sun) -- same 4-resource contract + include block as ParticleRender.frag /
// TransparentForward.frag, at this pass' own set 2.
#define SHADOW_PAGE_TABLE_SET 2
#define SHADOW_PAGE_TABLE_BINDING 0
#define SHADOW_FEEDBACK_SET 2
#define SHADOW_FEEDBACK_BINDING 1
#define SHADOW_ATLAS_SET 2
#define SHADOW_ATLAS_BINDING 2
#define SHADOW_SUN_LEVELS_SET 2
#define SHADOW_SUN_LEVELS_BINDING 3
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"

// World Probe Grid (indirect diffuse) -- same macro contract as ParticleRender.frag.
#define WORLD_PROBE_GRID_SET 2
#define WORLD_PROBE_GRID_BINDING 4
#define WORLD_PROBE_GRID_PARAMS_BINDING 5
#include "include/world_probe_sampling.glsl"

layout(push_constant) uniform PushConstants {
    uint archetypeSegmentBase;
    uint archetype;
    uint _pad0;
    uint _pad1;
} pc;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inTint;

layout(location = 0) out vec4 outColor;

void main() {
    // Per-archetype base albedo (procedural, no texture) modulated by the per-instance tint.
    vec3 baseColor;
    if (pc.archetype == kVegArchetypeGrass) {
        // Blade tip lighter than base for a bit of vertical gradient (inUV.y = 0 at ground, 1 at tip).
        baseColor = mix(vec3(0.14, 0.30, 0.09), vec3(0.28, 0.50, 0.16), inUV.y);
    } else if (pc.archetype == kVegArchetypeBush) {
        baseColor = vec3(0.15, 0.31, 0.12);
    } else {
        baseColor = vec3(0.38, 0.35, 0.32);
    }
    vec3 albedo = baseColor * inTint;

    vec3 N = normalize(inWorldNormal);
    // sunDirection points from the light toward the scene, so the surface->light direction is -sunDirection.
    vec3 L = -normalize(g_Params.sunDirection);
    float NdotL = max(dot(N, L), 0.0);

    float sunVisibility = SampleSunShadowVSM(inWorldPos);
    vec3 sunRadiance = g_Params.sunColor * g_Params.sunIntensity * NdotL * sunVisibility;

    vec3 indirectDiffuse = SampleWorldProbeGrid(inWorldPos, g_Params.cameraPos);

    // A small ambient floor keeps foliage sitting in full shadow with no nearby probe coverage from
    // crushing to pure black (same convention ParticleRender.frag uses).
    vec3 lighting = sunRadiance + indirectDiffuse + vec3(0.03);

    outColor = vec4(albedo * lighting, 1.0);
}
