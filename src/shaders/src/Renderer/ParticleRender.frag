#version 460
#extension GL_GOOGLE_include_directive : enable

// Particle system Subtask 4 (see ParticleRender.vert's own header comment for the full billboard
// contract this fragment shader receives). Two effects on top of the plain per-particle color:
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

layout(std140, set = 2, binding = 0) uniform ParticleRenderParamsUBO {
    mat4 viewProj;
    mat4 invViewProj;
    vec3 cameraPosition; float _pad0;
    vec3 cameraRight; float _pad1;
    vec3 cameraUp; float _pad2;
    vec2 viewportSize; float softFadeDistance; float globalTime;
} g_Params;

// renderer::ClusterResolvePass::GetOutputDepthView() -- the SAMPLED GBuffer depth copy (R32_SFLOAT,
// raw NDC z, reversed-Z: near = 1.0, far = 0.0), NOT the real depth-stencil attachment this pipeline
// depth-tests against (that one is bound as this draw's actual VkRenderingAttachmentInfo depth
// attachment, read-only, see RecordDraw's own comment -- a depth ATTACHMENT cannot simultaneously be
// sampled by the same draw's own fragment shader, hence needing this separate sampled copy).
layout(set = 2, binding = 1) uniform sampler2D g_SceneDepth;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inNormalizedAge;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 centered = inUV * 2.0 - 1.0;
    float shapeMask = smoothstep(1.0, 0.0, length(centered));
    if (shapeMask <= 0.0) {
        discard; // Outside the sprite's soft circle -- skip the (otherwise wasted) depth reconstruction below.
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
    outColor = vec4(inColor.rgb, alpha);
}
