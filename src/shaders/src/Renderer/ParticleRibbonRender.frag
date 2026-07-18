#version 460
#extension GL_GOOGLE_include_directive : enable

// Niagara-parity roadmap, subtask B2 (Ribbon/Trail render mode) -- see ParticleRibbonRender.vert's
// own header comment for the full quad-strip construction this fragment shader receives.
//
// Shading mirrors ParticleRender.frag's own billboard treatment closely (soft-particle fade against
// the scene depth, shadowed-sun + indirect-diffuse World Probe Grid lighting, no BRDF/normal term --
// a ribbon strip has no more of a real surface normal than a billboard does) with 2 ribbon-specific
// additions: an analytic soft edge across the strip's own width (outWidthCoord, this project's usual
// "no texture assets" analytic-shape-mask convention) and a head-to-tail fade (outTailFade) so the
// trail visibly dissipates toward its oldest segment instead of ending in a hard cut.

#include "include/ParticleCommon.glsl"

layout(std140, set = 2, binding = 0) uniform ParticleRenderParamsUBO {
    mat4 viewProj;
    mat4 invViewProj;
    vec3 cameraPosition; float _pad0;
    vec3 cameraRight; float _pad1;
    vec3 cameraUp; float _pad2;
    vec2 viewportSize; float softFadeDistance; float globalTime;
    vec3 sunDirection; float sunIntensity;
    vec3 sunColor; float _pad3;
    float heatShimmerStrength; float _pad4, _pad5, _pad6;
} g_Params;

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
layout(location = 1) in vec4 inColor;
layout(location = 2) in float inNormalizedAge;
layout(location = 3) in float inTailFade;
layout(location = 4) in float inWidthCoord;

layout(location = 0) out vec4 outColor;
// Same shared heat-distortion attachment every pipeline in this rendering scope must write --
// ribbons do not support heat-shimmer (out of scope for this roadmap step), always (0,0), same
// rationale as ParticleMeshRender.frag's own identical write.
layout(location = 1) out vec2 outRefractionOffset;

void main() {
    // Soft edge across the strip's own width -- analytic, no texture asset (this project's own
    // established convention, see ParticleRender.frag's own header comment).
    float edgeMask = smoothstep(0.0, 0.2, inWidthCoord) * smoothstep(1.0, 0.8, inWidthCoord);
    if (edgeMask <= 0.0 || inTailFade <= 0.0) {
        discard;
    }

    vec2 screenUV = gl_FragCoord.xy / g_Params.viewportSize;
    float sceneNdcDepth = texture(g_SceneDepth, screenUV).r;

    float softFade = 1.0;
    if (sceneNdcDepth > 0.0) {
        vec4 clip = vec4(screenUV * 2.0 - 1.0, sceneNdcDepth, 1.0);
        vec4 sceneWorld4 = g_Params.invViewProj * clip;
        vec3 sceneWorldPos = sceneWorld4.xyz / sceneWorld4.w;

        float sceneDist = length(sceneWorldPos - g_Params.cameraPosition);
        float ribbonDist = length(inWorldPos - g_Params.cameraPosition);
        softFade = clamp((sceneDist - ribbonDist) / max(g_Params.softFadeDistance, 1.0e-4), 0.0, 1.0);
    }

    float lifeFade = smoothstep(0.0, 0.2, inNormalizedAge);

    float alpha = inColor.a * edgeMask * softFade * lifeFade * inTailFade;

    float sunVisibility = SampleSunShadowVSM(inWorldPos);
    vec3 sunRadiance = g_Params.sunColor * g_Params.sunIntensity * sunVisibility;
    vec3 indirectDiffuse = SampleWorldProbeGrid(inWorldPos);
    vec3 lighting = sunRadiance + indirectDiffuse + vec3(0.02);

    outColor = vec4(inColor.rgb * lighting, alpha);
    outRefractionOffset = vec2(0.0);
}
