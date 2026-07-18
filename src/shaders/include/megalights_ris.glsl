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

// Phase 4, Feature 2 (temporal ReSTIR) / spatial reuse follow-up: standard ReSTIR-style temporal
// age-clamp -- how many "effective samples" a single long-lived history reservoir (or, via spatial
// reuse, a neighbor's already-temporally-combined reservoir) is allowed to represent before it
// starts dominating a freshly-relit surface's own fresh candidate. Shared here (not left duplicated
// per-shader) because BOTH MegaLightsShade.comp's own CombineTemporalReservoir (cross-FRAME reuse)
// and MegaLightsSpatialReuse.comp (cross-PIXEL reuse, same-frame) need the exact same cap for their
// own streaming-RIS combine's M-weighting term -- see each call site's own comment.
const float kMegaLightsTemporalHistoryMaxM = 20.0 * float(kMegaLightsCandidateCount);

// UE5.8 rendering-parity gap G3: per-light-type angular shaping, SHARED by the RIS target-weight
// proxy below AND by every consumer's own final shading (MegaLightsFinalShade.comp's opaque path,
// TransparentForward.frag/Tessellation.frag/ParticleRender.frag's inline forward paths,
// AtmosVolumetricFog.comp's fog in-scatter) so a spot cone / photometric profile / rect one-sided
// cut is applied identically wherever a MegaLight is evaluated. `L` is the NORMALIZED surface->light
// direction (so the light->surface / emission direction is -L). Returns a scalar in [0,1]:
//   POINT        -- always 1 (isotropic, no angular shaping).
//   SPOT         -- smoothstep between the outer and inner cone cosines (UE5.8's own spot cone falloff).
//   PHOTOMETRIC  -- one of the plain analytic MEGALIGHT_IES_* profiles (no baked data, CLAUDE.md rule).
//   RECT         -- the one-sided-emission cut only (1 in front of the emitter, 0 behind unless
//                   two-sided); a rect's real area response is handled by point-on-rect sampling +
//                   LTC in the shading code, not by this scalar, so this returns just the facing gate.
float MegaLightSpotWindow(MegaLight light, vec3 L) {
    // cos of the angle between the cone axis (light.direction, light->scene) and the light->surface
    // direction (-L). smoothstep(outer, inner, .) is 1 inside the inner cone, 0 outside the outer.
    float cosDir = dot(light.direction, -L);
    return smoothstep(light.spotCosOuter, light.spotCosInner, cosDir);
}

float MegaLightIESWindow(MegaLight light, vec3 L) {
    vec3 d = -L; // light -> surface (emission) direction.
    float cosTheta = clamp(dot(light.direction, d), -1.0, 1.0);
    uint profile = MegaLightIESProfile(light);
    if (profile == MEGALIGHT_IES_NARROW) {
        // Tight pencil beam: a high-exponent forward cosine lobe, zero on the back hemisphere.
        return pow(max(cosTheta, 0.0), 8.0);
    } else if (profile == MEGALIGHT_IES_WIDE) {
        // Broad flood: a gentle ramp that stays near-uniform across most of the forward hemisphere
        // and only fades near/behind the equator.
        return smoothstep(-0.25, 0.55, cosTheta);
    }
    // MEGALIGHT_IES_BARN -- asymmetric rectangular "barn-door" cut. Project the emission direction
    // onto the two in-plane axes (tangentU and the derived tangentV) and gate each independently with
    // a different width, plus a lateral offset on the U axis, to fake the characteristic lopsided
    // rectangular pool a real barn-door fixture throws.
    float fwd = cosTheta;
    if (fwd <= 1.0e-3) return 0.0;
    vec3 tV = cross(light.direction, light.tangentU);
    float u = dot(d, light.tangentU) / fwd; // tan(angle) in the U plane.
    float v = dot(d, tV) / fwd;             // tan(angle) in the V plane.
    float gateU = 1.0 - smoothstep(0.85, 1.45, abs(u - 0.20)); // wide, offset to one side.
    float gateV = 1.0 - smoothstep(0.28, 0.55, abs(v));         // narrow.
    return saturate(fwd) * gateU * gateV;
}

float MegaLightAngularShaping(MegaLight light, vec3 L) {
    if (light.lightType == MEGALIGHT_TYPE_SPOT) {
        return MegaLightSpotWindow(light, L);
    } else if (light.lightType == MEGALIGHT_TYPE_PHOTOMETRIC) {
        return MegaLightIESWindow(light, L);
    } else if (light.lightType == MEGALIGHT_TYPE_RECT) {
        // One-sided emission gate: a surface behind the rect (relative to its normal) receives
        // nothing unless the light is flagged two-sided.
        if (MegaLightRectTwoSided(light)) return 1.0;
        return (dot(light.direction, -L) > 0.0) ? 1.0 : 0.0;
    }
    return 1.0; // MEGALIGHT_TYPE_POINT.
}

// Unshadowed contribution-magnitude proxy used both to build reservoir weights AND (implicitly, via
// SelectLightRIS's outInvPdf) to cancel out of the final unbiased estimator -- see that function's
// own comment. Clamped to avoid a near-zero-distance sample dominating the reservoir. G3: for a
// non-POINT light the proxy is additionally scaled by MegaLightAngularShaping so the RIS candidate
// draw is biased AWAY from a light whose cone/profile/facing barely reaches this shading point (a
// spot pointing away contributes ~0 and should rarely win the reservoir) -- purely an importance
// bias, the unbiased estimator stays correct because the SAME proxy divides back out in outInvPdf.
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

    float w = light.intensity * NdotL / distSq * window * MegaLightAngularShaping(light, lightDir);
    return min(w, 50.0);
}

// Streaming weighted reservoir (textbook WRS/Chao's algorithm, single reservoir, no cross-pixel/
// cross-frame state -- temporal reuse across FRAMES is Feature 2 of Phase 4, layered on top of this
// function's own single-frame result by MegaLightsShade.comp's own CombineTemporalReservoir, not
// inside this function) over kMegaLightsCandidateCount candidates, indexed via a per-pixel/
// per-frame-offset Halton23 sequence (math_utils.glsl) rather than a naive hash-modulo, to avoid
// visible index-selection banding.
//
// Feature 1 of Phase 4 (light BVH for RIS spatial bias): `pool`/`poolCount` are
// megalights_bvh.glsl's GatherSpatialLightCandidates own output -- when poolCount > 0, each
// candidate index is drawn from THAT pool (biasing every one of the M draws toward lights actually
// near `worldPos`) instead of uniformly across the full g_Lights population; when poolCount == 0
// (an isolated accent-zone pixel with no light within the BVH's query radius, or a caller that
// never populated a pool at all), this falls back to the ORIGINAL full-population draw -- no
// regression for that case. Only the candidate index SOURCE changes; the streaming-reservoir/
// weighting math below is completely untouched from Phase A.
//
// Returns false if every candidate's weight was zero (e.g. no light in range) --
// `outLightIndex`/`outInvPdf` are only valid when this returns true. `outInvPdf` is the RIS
// unbiased estimator's (totalWeight / M) / targetWeight(selected) factor: multiplying the FULL
// shading contribution (not just the target proxy) of the selected light by this factor gives an
// unbiased estimate of the true multi-light sum -- see MegaLightsShade.comp's own call site for the
// exact final formula. This same (totalWeight / M) / targetWeight(selected) factor is also exactly
// the standard ReSTIR reservoir's own unbiased contribution weight W = wsum / (M * targetPdf(y)) --
// see CombineTemporalReservoir's own comment for why this lets Phase A's existing RIS output be fed
// directly into Feature 2's temporal combine with no extra bookkeeping.
bool SelectLightRIS(vec3 worldPos, vec3 normal, uint pixelSeed, uint frameIndex,
    uint pool[kMegaLightsSpatialPoolCapacity], uint poolCount, out uint outLightIndex, out float outInvPdf) {
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
        uint candidateIndex;
        if (poolCount > 0u) {
            candidateIndex = pool[min(uint(h.x * float(poolCount)), poolCount - 1u)];
        } else {
            candidateIndex = min(uint(h.x * float(g_Lights.lightCount)), g_Lights.lightCount - 1u);
        }
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
