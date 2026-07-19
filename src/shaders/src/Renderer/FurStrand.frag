#version 460
#extension GL_GOOGLE_include_directive : enable

// Fur-strand system (UE5.8 rendering-parity gap G10a), instanced draw FRAGMENT stage. A dedicated
// LIGHTWEIGHT forward shading path (NOT the Nanite deferred visbuffer/ClusterResolve material path),
// exactly like renderer::VegetationScatterPass's own VegetationInstanced.frag: the deferred path is
// keyed on cluster+triangle IDs the Nanite rasterizers write, which arbitrary instanced strand
// geometry cannot produce, so fur is shaded forward with the same VSM-shadowed sun + World Probe Grid
// indirect-diffuse lighting set every other forward pass (TransparentForward / Tessellation /
// WaterForward / Particle / Vegetation) already uses.
//
// The one thing this pass does NOT share with those passes is its BSDF: fur is shaded with the
// DEDICATED hair model (include/hair_bsdf.glsl), a real-time Marschner subset (Karis 2016 lineage --
// see that header). Its anisotropic highlight slides along the per-fragment strand tangent, the
// unmistakable hair/fur look, distinct from the isotropic GGX response the Substrate path produces.
//
// --- Self-shadowing / density darkening (documented real-time simplification) ---
// Full hair self-shadowing (deep-opacity maps / dual-scattering) is explicitly out of scope for
// G10a. Two cheap stand-ins are used instead: (1) the existing sun VSM sample (real cast shadows
// from the rest of the scene onto the pelt, and the creature onto itself at the shadow-map's
// resolution), and (2) a root-to-tip ambient-occlusion falloff (rootDarken -> 1.0 across the strand)
// approximating how light is progressively occluded by neighbouring fibres the deeper into the pelt
// you go -- brightest at the exposed tips, darkest at the buried roots.

#include "include/hair_bsdf.glsl"

layout(std140, set = 1, binding = 0) uniform FurRenderParamsUBO {
    mat4 viewProj;
    vec3 cameraPos;     float furLength;
    vec3 sunDirection;  float sunIntensity;  // sunDirection points FROM the light TOWARD the scene (same convention as SceneLights::sun).
    vec3 sunColor;      float furWidth;
    vec3 rootColor;     float rootDarken;    // Hair albedo near the root; rootDarken (<1) is the pelt-depth AO floor.
    vec3 tipColor;      float curlAmount;    // Hair albedo at the tip (fur usually lightens toward the tip).
    float shiftR;       float shiftTRT;   float exponentR;    float exponentTRT;
    float specIntensity; float trtIntensity; float globalTime; uint creatureMeshID;
} g_Params;

// Virtual Shadow Map (sun) -- same 4-resource contract + include block as VegetationInstanced.frag /
// ParticleRender.frag / TransparentForward.frag, at this pass' own set 2.
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

// World Probe Grid (indirect diffuse) -- same macro contract as VegetationInstanced.frag.
#define WORLD_PROBE_GRID_SET 2
#define WORLD_PROBE_GRID_BINDING 4
#define WORLD_PROBE_GRID_PARAMS_BINDING 5
#include "include/world_probe_sampling.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inTangent;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in float inStrandT;
layout(location = 4) in float inTint;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 T = normalize(inTangent);
    vec3 N = normalize(inNormal);
    vec3 V = normalize(g_Params.cameraPos - inWorldPos);
    // sunDirection points from the light toward the scene, so the surface->light direction is -sunDirection.
    vec3 L = -normalize(g_Params.sunDirection);

    // Fur albedo runs root -> tip (fur commonly lightens toward the tip), modulated by the per-strand
    // brightness jitter so a pelt reads as many fibres, not one flat sheet.
    vec3 hairColor = mix(g_Params.rootColor, g_Params.tipColor, inStrandT) * inTint;

    HairParams hp;
    hp.shiftR = g_Params.shiftR;
    hp.shiftTRT = g_Params.shiftTRT;
    hp.exponentR = g_Params.exponentR;
    hp.exponentTRT = g_Params.exponentTRT;
    hp.specIntensity = g_Params.specIntensity;
    hp.trtIntensity = g_Params.trtIntensity;

    HairBSDFResult hair = EvaluateHairBSDF(T, N, V, L, hairColor, hp);

    // Direct sun (VSM-shadowed) + indirect diffuse (World Probe Grid).
    float sunVisibility = SampleSunShadowVSM(inWorldPos);
    vec3 sunLight = g_Params.sunColor * g_Params.sunIntensity * sunVisibility;
    vec3 indirectDiffuse = SampleWorldProbeGrid(inWorldPos, g_Params.cameraPos);

    // Root-to-tip density AO (see this file's own header comment): the cheap self-shadow stand-in.
    float rootAO = mix(clamp(g_Params.rootDarken, 0.0, 1.0), 1.0, inStrandT);

    // Diffuse receives sun + indirect (+ a small ambient floor so fully-shadowed fur never crushes to
    // pure black, matching every other forward pass' convention); specular receives only direct sun.
    vec3 diffuseLit = hair.diffuse * (sunLight + indirectDiffuse + vec3(0.03));
    vec3 specularLit = hair.specular * sunLight;

    vec3 color = (diffuseLit + specularLit) * rootAO;
    outColor = vec4(color, 1.0);
}
