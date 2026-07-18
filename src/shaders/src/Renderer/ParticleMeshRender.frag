#version 460
#extension GL_GOOGLE_include_directive : enable

// Niagara-parity roadmap, subtask B1 (Mesh Particle render mode) -- see ParticleMeshRender.vert's
// own header comment for the full instancing contract this fragment shader receives.
//
// Unlike ParticleRender.frag's billboard shading (no real surface normal, no soft-particle/scene-
// depth fade, no heat-shimmer output either -- deliberately out of scope for this roadmap step, a
// real 3D mesh instance does not need a billboard's own depth-reconstruction workaround since the
// real depth TEST already handles it correctly), this shader has a genuine per-fragment normal and
// therefore does a real Lambertian NdotL term against the sun, plus the same isotropic World Probe
// Grid indirect diffuse every other lit forward pass in this codebase already samples. Rendered as
// an opaque color WRITE (blendEnable = false, alpha forced to 1.0 below) but with depth WRITE
// disabled (renderer::ParticleSystemPass::Init()'s own STEP 7 comment) -- this whole pass' shared
// rendering scope binds its depth attachment read-only (same constraint the billboard pipeline
// already lives with), so two overlapping mesh particles rely on back-to-front draw order
// (SortedPairsBuffer, same as billboards) rather than the depth buffer for correct relative
// occlusion, while still correctly depth-testing (and being hidden) behind real opaque scene
// geometry.

#include "include/ParticleCommon.glsl"

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
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inNormalizedAge;

layout(location = 0) out vec4 outColor;
// renderer::TransparentForwardPass's own shared heat-distortion target -- this pipeline shares
// renderer::ParticleSystemPass's dynamic-rendering scope (2 color attachments) with
// ParticleRender.vert/.frag, so it must ALSO write this attachment every fragment even though mesh
// particles do not support heat-shimmer distortion (out of scope for this roadmap step) -- an
// explicit (0,0), same "suppress whatever a PREVIOUS forward pass wrote at this exact pixel"
// rationale as ParticleRender.frag's own identical write.
layout(location = 1) out vec2 outRefractionOffset;

void main() {
    vec3 n = normalize(inNormal);

    // --- Sun (VSM-shadowed, real NdotL this time) + indirect diffuse (World Probe Grid) ---
    float sunVisibility = SampleSunShadowVSM(inWorldPos);
    float ndotl = max(dot(n, -g_Params.sunDirection), 0.0);
    vec3 sunRadiance = g_Params.sunColor * g_Params.sunIntensity * ndotl * sunVisibility;
    vec3 indirectDiffuse = SampleWorldProbeGrid(inWorldPos);
    vec3 lighting = sunRadiance + indirectDiffuse + vec3(0.02); // Small constant floor -- see ParticleRender.frag's own identical rationale.

    // Fully opaque -- alpha is deliberately NOT taken from inColor.a (colorCurve's alpha channel is
    // an authored "how see-through should a BILLBOARD sprite look" value; a solid mesh instance has
    // no such notion, and leaving alpha at 1.0 here avoids a curve authored for the billboard look
    // silently punching a hole in this image's alpha channel for some OTHER later consumer). A
    // particle that has faded its sizeCurve down to (near) zero simply becomes too small to see,
    // which already reproduces the intended "fades away" effect without needing real blending.
    outColor = vec4(inColor.rgb * lighting, 1.0);
    outRefractionOffset = vec2(0.0);
}
