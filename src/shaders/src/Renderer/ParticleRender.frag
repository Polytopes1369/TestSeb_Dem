#version 460
#extension GL_GOOGLE_include_directive : enable

// Particle system Subtask 4+5 (see ParticleRender.vert's own header comment for the full billboard
// contract this fragment shader receives). Effects on top of the plain per-particle color:
//
// --- Procedural sprite shape ---
// This project has zero on-disk texture assets (CLAUDE.md's "no data in the .exe" constraint --
// every material/noise/sky/cloud in this codebase is generated analytically, see e.g.
// procedural_material.glsl / atmos_clouds_density.glsl), so the plan doc's literal "sample a
// texture-atlas" instruction is adapted into a plain analytic soft-circle mask in UV space instead
// of a loaded sprite sheet -- consistent with every other visual system in this renderer.
//
// --- Soft particles ---
// Reconstructs the opaque scene's world position under this fragment from renderer::
// ClusterResolvePass's own GBuffer depth copy (bound once at Init(), see renderer::
// ParticleSystemPass::Init's own comment) and fades the sprite out as it nears an intersection with
// that surface, instead of hard-clipping at the fixed-function depth test's binary pass/fail edge.
// This codebase has no existing linear-depth-reconstruction helper (every other depth consumer works
// in raw NDC space directly, see e.g. SDFRayMarch.comp's own SampleClipmap) -- this shader
// reconstructs a genuine WORLD position via the inverse view-projection matrix (mirroring
// SSRFallback.comp's own ReconstructWorldPos) and fades on the resulting camera-space DISTANCE
// difference, which stays physically meaningful (a fixed world-unit fade band) at every depth,
// unlike a raw (and highly nonlinear, reversed-Z) NDC delta would.
//
// --- Subtask 5: sun (VSM-shadowed) + indirect diffuse (World Probe Grid) ---
// A billboard has no real surface normal, so this is deliberately NOT a BRDF evaluation (no
// NdotL/Henyey-Greenstein phase term -- the plan doc's own "simplified Lambertian" allowance) --
// just the shadowed sun radiance plus indirect diffuse irradiance, both isotropic, added together
// and modulating the particle's own albedo. Same VSM include block as TransparentForward.frag /
// SurfaceCacheCapture.frag (shadow_page_table/feedback/atlas_sampling/sun_sampling.glsl); same
// world_probe_sampling.glsl macro contract every other consumer (TransparentForward.frag,
// HeroTessellation.frag) already uses.
//
// --- Subtask 5: heat-shimmer refraction ---
// When g_Params.heatShimmerStrength > 0 (an emitter-level toggle -- see renderer::
// ParticleSystemPass::RecordDraw's own comment for why this is per-draw-call, not per-particle:
// GpuParticle's already-merged 64-byte layout has no spare "isRefractive" flag, and a demoscene
// emitter is realistically one thermal "kind" or another, never a per-particle mix), writes a
// small time-varying wobble into this pass' second color output -- renderer::
// TransparentForwardPass's OWN shared g_RefractionOffset image (this codebase's first SECOND writer
// of that image, see ParticleSystemPass::RecordDraw's own comment on why the load/store discipline
// there matters), later read by PostProcessComposite.comp exactly like glass's own contribution.

layout(std140, set = 2, binding = 0) uniform ParticleRenderParamsUBO {
    mat4 viewProj;
    mat4 invViewProj;
    vec3 cameraPosition; float _pad0;
    vec3 cameraRight; float _pad1;
    vec3 cameraUp; float _pad2;
    vec2 viewportSize; float softFadeDistance; float globalTime;
    vec3 sunDirection; float sunIntensity; // sunDirection points FROM the light TOWARD the scene.
    vec3 sunColor; float _pad3;
    float heatShimmerStrength; float _pad4, _pad5, _pad6;
} g_Params;

// renderer::ClusterResolvePass::GetOutputDepthView() -- the SAMPLED GBuffer depth copy (R32_SFLOAT,
// raw NDC z, reversed-Z: near = 1.0, far = 0.0), NOT the real depth-stencil attachment this pipeline
// depth-tests against (that one is bound as this draw's actual VkRenderingAttachmentInfo depth
// attachment, read-only, see RecordDraw's own comment -- a depth ATTACHMENT cannot simultaneously be
// sampled by the same draw's own fragment shader, hence needing this separate sampled copy).
layout(set = 2, binding = 1) uniform sampler2D g_SceneDepth;

#define SHADOW_PAGE_TABLE_SET 3
#define SHADOW_PAGE_TABLE_BINDING 0
#define SHADOW_FEEDBACK_SET 3
#define SHADOW_FEEDBACK_BINDING 1
#define SHADOW_ATLAS_SET 3
#define SHADOW_ATLAS_BINDING 2
#define SHADOW_SUN_LEVELS_SET 3
#define SHADOW_SUN_LEVELS_BINDING 3
#include "include/shadow_page_table.glsl"
#include "include/shadow_feedback.glsl"
#include "include/shadow_atlas_sampling.glsl"
#include "include/shadow_sun_sampling.glsl"

#define WORLD_PROBE_GRID_SET 3
#define WORLD_PROBE_GRID_BINDING 4
#define WORLD_PROBE_GRID_PARAMS_BINDING 5
#include "include/world_probe_sampling.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inNormalizedAge;

layout(location = 0) out vec4 outColor;
// renderer::TransparentForwardPass's own shared heat-distortion target -- see this file's own
// header comment. Written every fragment (even when heatShimmerStrength == 0, in which case it is
// simply (0,0) -- a plain overwrite, blendEnable=FALSE for this attachment, so an explicit zero is
// required rather than "just don't write," which would leave whatever a PREVIOUS forward pass wrote
// at this exact pixel untouched -- wrong, since a non-shimmering particle covering that pixel should
// suppress any earlier distortion there, exactly like an opaque particle would).
layout(location = 1) out vec2 outRefractionOffset;

void main() {
    vec2 centered = inUV * 2.0 - 1.0;
    float shapeMask = smoothstep(1.0, 0.0, length(centered));
    if (shapeMask <= 0.0) {
        discard; // Outside the sprite's soft circle -- skip the (otherwise wasted) work below.
    }

    vec2 screenUV = gl_FragCoord.xy / g_Params.viewportSize;
    float sceneNdcDepth = texture(g_SceneDepth, screenUV).r;

    // sceneNdcDepth <= 0.0 means sky/no-geometry-hit at this pixel (see renderer::ClusterResolvePass's
    // own g_OutputDepth clear-to-0 convention) -- nothing to soft-fade against, draw at full strength.
    float softFade = 1.0;
    if (sceneNdcDepth > 0.0) {
        vec4 clip = vec4(screenUV * 2.0 - 1.0, sceneNdcDepth, 1.0);
        vec4 sceneWorld4 = g_Params.invViewProj * clip;
        vec3 sceneWorldPos = sceneWorld4.xyz / sceneWorld4.w;

        float sceneDist = length(sceneWorldPos - g_Params.cameraPosition);
        float particleDist = length(inWorldPos - g_Params.cameraPosition);
        softFade = clamp((sceneDist - particleDist) / max(g_Params.softFadeDistance, 1.0e-4), 0.0, 1.0);
    }

    // Fade out over the last 20% of remaining life rather than vanishing the instant a particle is
    // recycled server-side (see ParticleRender.vert's own outNormalizedAge comment).
    float lifeFade = smoothstep(0.0, 0.2, inNormalizedAge);

    float alpha = inColor.a * shapeMask * softFade * lifeFade;

    // --- Sun (VSM-shadowed) + indirect diffuse (World Probe Grid) -- see this file's own header
    // comment for why there is no NdotL/phase-function term. A small constant floor keeps a
    // particle sitting in full shadow with no probe coverage nearby from going pure black. ---
    float sunVisibility = SampleSunShadowVSM(inWorldPos);
    vec3 sunRadiance = g_Params.sunColor * g_Params.sunIntensity * sunVisibility;
    vec3 indirectDiffuse = SampleWorldProbeGrid(inWorldPos);
    vec3 lighting = sunRadiance + indirectDiffuse + vec3(0.02);

    outColor = vec4(inColor.rgb * lighting, alpha);

    if (g_Params.heatShimmerStrength > 0.0) {
        vec2 shimmer = vec2(
            sin(g_Params.globalTime * 3.0 + inWorldPos.x * 5.0 + inWorldPos.z * 3.0),
            cos(g_Params.globalTime * 2.7 + inWorldPos.z * 5.0 - inWorldPos.x * 3.0));
        outRefractionOffset = shimmer * g_Params.heatShimmerStrength * shapeMask * alpha;
    } else {
        outRefractionOffset = vec2(0.0);
    }
}
