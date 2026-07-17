#ifndef MEGALIGHTS_RIS_GLSL
#define MEGALIGHTS_RIS_GLSL

// Phase A of the MegaLights native-port roadmap (see the approved plan,
// C:\Users\Seb\.claude\plans\memoized-splashing-seal.md, for the full derivation): pure weighting/
// reservoir math, shared by every shader that needs to pick one of g_Lights.lightCount point lights
// per shading point via RIS (Resampled Importance Sampling) -- MegaLightsShade.comp (opaque path)
// and TransparentForward.frag (forward/translucent path, inlined since translucent surfaces have no
// GBuffer entry for a separate compute pass to reach). Ray-tracing code itself is NOT shared here
// (this codebase's own established convention: shared weighting math, per-shader-duplicated
// TraceHWRT-style trace functions -- see e.g. ReflectionTrace.comp's own header comment on why a
// 4th/5th copy of that trace function follows precedent rather than refactoring).
//
// The caller must #include "math_utils.glsl" (for saturate()/Halton23()) before this include, AND
// define MEGALIGHTS_LIGHTS_SET/MEGALIGHTS_LIGHTS_BINDING -- same caller-defined-binding-macro
// convention as shadow_page_table.glsl/reflection_view_params.glsl. megalights_types.glsl is
// included directly below (not left to the caller) since the MegaLight struct it declares is used
// by this file's own SSBO block, not just by the caller's shading code.

#ifndef MEGALIGHTS_LIGHTS_SET
#error "Define MEGALIGHTS_LIGHTS_SET/MEGALIGHTS_LIGHTS_BINDING before including megalights_ris.glsl"
#endif

#include "megalights_types.glsl"

// 16-byte header (lightCount + 3 reserved) keeps the trailing array's own base offset 16-aligned --
// same header-then-array shape SurfaceCacheLightingUBO already uses for pointLightCount+padding.
layout(std430, set = MEGALIGHTS_LIGHTS_SET, binding = MEGALIGHTS_LIGHTS_BINDING) readonly buffer MegaLightsSSBO {
    uint lightCount;
    uint _pad0, _pad1, _pad2;
    MegaLight lights[];
} g_Lights;

const uint kMegaLightsCandidateCount = 16u; // RIS M -- see the approved plan's own justification.

// Unshadowed contribution-magnitude proxy used both to build reservoir weights AND (implicitly, via
// SelectLightRIS's outInvPdf) to cancel out of the final unbiased estimator -- see that function's
// own comment. Clamped to avoid a near-zero-distance sample dominating the reservoir.
float MegaLightTargetWeight(MegaLight light, vec3 worldPos, vec3 normal) {
    vec3 toLight = light.position - worldPos;
    float distSq = max(dot(toLight, toLight), 1.0e-4);
    float dist = sqrt(distSq);
    vec3 lightDir = toLight / dist;
    float NdotL = saturate(dot(normal, lightDir));

    // Smooth window reaching exactly 0 at light.radius (matches renderer::MegaLight::radius's own
    // doc comment) -- (1 - (d/r)^4)^2, the same UE-style smooth falloff shape
    // shadow_sun_sampling.glsl's cascade blending already uses elsewhere in this codebase.
    float normalizedDist = saturate(dist / max(light.radius, 1.0e-4));
    float nd2 = normalizedDist * normalizedDist;
    float windowSq = 1.0 - nd2 * nd2;
    float window = saturate(windowSq * windowSq);

    float w = light.intensity * NdotL / distSq * window;
    return min(w, 50.0);
}

// Streaming weighted reservoir (textbook WRS/Chao's algorithm, single reservoir, no cross-pixel/
// cross-frame state -- temporal/spatial reuse is explicitly Phase B) over kMegaLightsCandidateCount
// candidates, indexed via a per-pixel/per-frame-offset Halton23 sequence (math_utils.glsl) rather
// than a naive hash-modulo, to avoid visible index-selection banding. Returns false if every
// candidate's weight was zero (e.g. no light in range) -- `outLightIndex`/`outInvPdf` are only valid
// when this returns true. `outInvPdf` is the RIS unbiased estimator's (totalWeight / M) /
// targetWeight(selected) factor: multiplying the FULL shading contribution (not just the target
// proxy) of the selected light by this factor gives an unbiased estimate of the true multi-light sum
// -- see MegaLightsShade.comp's own call site for the exact final formula.
bool SelectLightRIS(vec3 worldPos, vec3 normal, uint pixelSeed, uint frameIndex, out uint outLightIndex, out float outInvPdf) {
    if (g_Lights.lightCount == 0u) {
        outLightIndex = 0u;
        outInvPdf = 0.0;
        return false;
    }

    float totalWeight = 0.0;
    uint selected = 0xFFFFFFFFu;
    float selectedTargetWeight = 0.0;

    uint baseIndex = pixelSeed * kMegaLightsCandidateCount + frameIndex * 9781u;
    for (uint i = 0u; i < kMegaLightsCandidateCount; ++i) {
        vec2 h = Halton23(baseIndex + i);
        uint candidateIndex = min(uint(h.x * float(g_Lights.lightCount)), g_Lights.lightCount - 1u);
        MegaLight candidate = g_Lights.lights[candidateIndex];

        float w = MegaLightTargetWeight(candidate, worldPos, normal);
        totalWeight += w;
        float acceptProb = (totalWeight > 0.0) ? (w / totalWeight) : 0.0;
        if (h.y < acceptProb) {
            selected = candidateIndex;
            selectedTargetWeight = w;
        }
    }

    if (selected == 0xFFFFFFFFu || selectedTargetWeight <= 0.0) {
        outLightIndex = 0u;
        outInvPdf = 0.0;
        return false;
    }

    outLightIndex = selected;
    outInvPdf = (totalWeight / float(kMegaLightsCandidateCount)) / selectedTargetWeight;
    return true;
}

#endif
