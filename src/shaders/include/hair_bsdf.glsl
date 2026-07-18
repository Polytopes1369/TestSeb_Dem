#ifndef HAIR_BSDF_GLSL
#define HAIR_BSDF_GLSL

// UE5.8 rendering-parity gap G10a -- dedicated Hair/Fur shading model. This is DELIBERATELY a
// separate evaluation path from include/substrate_bsdf.glsl: UE5.8's Substrate documentation (and
// substrate_bsdf.glsl's own header comment) explicitly list "Hair/Eye/Water shading models" as
// SEPARATE, DEDICATED shading models OUTSIDE Substrate's own Slab BSDF -- exactly the way this
// codebase already gives Water (WaterForward.frag) and Terrain (terrain_shading.glsl) their own
// dedicated shading logic. Consumed only by renderer::FurStrandPass's FurStrand.frag (the
// GPU-instanced procedural fur-strand forward pass), never by the generic Substrate material path.
//
// --- Model: the real-time Marschner subset (Karis 2016 lineage) ---
// The reference real-time hair model is Brian Karis' "Physically Based Hair Shading in Unreal"
// (SIGGRAPH 2016), a real-time approximation of the Marschner multi-lobe (R / TT / TRT) scattering
// model. Karis himself frames it as the successor to the Kajiya-Kay / Scheuermann (ATI) real-time
// shifted-tangent anisotropic-highlight model, which is what produces hair's unmistakable visual
// signature: an anisotropic specular highlight that SLIDES ALONG THE STRAND TANGENT as the view
// moves (utterly distinct from an isotropic GGX highlight that stays pinned to the half-vector).
//
// Per gap G10a's explicit allowance to ship "a well-justified simplified subset ... document the
// simplification", this file implements the Kajiya-Kay / Scheuermann shifted-tangent DUAL-lobe
// subset of that lineage rather than the full longitudinal-Gaussian Marschner:
//   * R lobe   -- the PRIMARY specular highlight. White (light-colored), tangent shifted slightly
//                 TOWARD the strand normal (alphaR > 0). This is the sharp near-root highlight.
//   * TRT lobe -- the SECONDARY highlight. TINTED by the hair's own absorption color (light that
//                 has transmitted-reflected-transmitted through the fibre picks up its color),
//                 tangent shifted the OTHER way (alphaTRT < 0) and broader (lower exponent). This
//                 is the softer, colored highlight offset from the primary one.
//   * TT (forward transmission / multiple scattering) is folded into a Kajiya-Kay wrapped-diffuse
//     term rather than modelled as its own transmission lobe -- the documented simplification. A
//     true TT lobe needs a back-lit deep-opacity/transmittance estimate this real-time forward
//     pass does not maintain (see FurStrand.frag's root-AO density approximation for the same
//     "no deep-opacity map, cheap analytic stand-in" reasoning).
//
// Reuses include/math_utils.glsl's PI / saturate (same "no duplicate math" discipline every other
// shared BSDF header here follows -- see ggx_brdf.glsl's own header comment). No GGX term is used:
// hair's highlight is longitudinal-anisotropic, not a microfacet-normal-distribution response, so
// borrowing D_GGX here would be physically wrong -- this is precisely why hair needs its own header.

#include "include/math_utils.glsl"

// Per-material hair/fur appearance controls (all supplied by renderer::FurStrandPass's render-params
// UBO -- see FurRenderParamsUBO). Grouped into one struct so EvaluateHairBSDF's signature stays
// stable if more lobes are added later.
struct HairParams {
    float shiftR;        // Longitudinal tangent shift for the primary R lobe (radians-ish, toward N). Typically +0.03..+0.10.
    float shiftTRT;      // Longitudinal tangent shift for the secondary TRT lobe (away from N). Typically -0.05..-0.15.
    float exponentR;     // Sharpness of the primary highlight (higher = tighter). Typically 60..160.
    float exponentTRT;   // Sharpness of the secondary highlight (broader, so lower). Typically 15..40.
    float specIntensity; // Overall specular scale (both lobes).
    float trtIntensity;  // Extra weight on the colored secondary lobe relative to the white primary.
};

// Result split so the caller can light the two responses differently: hair DIFFUSE receives both the
// direct sun term AND the indirect World-Probe-Grid irradiance, while hair SPECULAR receives only the
// (shadowed) direct sun term -- mirrors exactly how ClusterResolve.comp / TransparentForward.frag
// keep their own diffuse and specular light budgets separate.
struct HairBSDFResult {
    vec3 diffuse;   // Kajiya-Kay wrapped diffuse (already tinted by hairColor).
    vec3 specular;  // R (white) + TRT (hairColor-tinted) shifted anisotropic highlights.
};

// Shift a hair tangent toward/away from the shading normal N, then renormalize -- the Scheuermann
// tangent-shift trick that separates the R and TRT highlights along the strand (Marschner's alpha_R
// / alpha_TRT longitudinal shift, approximated in tangent space). A positive `shift` moves the
// resulting highlight toward the strand root, a negative one toward the tip.
vec3 HairShiftTangent(vec3 T, vec3 N, float shift) {
    return normalize(T + shift * N);
}

// One Kajiya-Kay anisotropic specular lobe about a (shifted) tangent `Ts`. The highlight peaks where
// the half-vector H is PERPENDICULAR to the strand (sin of the T-H angle -> 1), and slides along the
// strand as V (hence H) rotates -- the anisotropic "sliding highlight" hallmark. `dirAtten` softly
// kills the lobe on the back side (dot(Ts,H) < 0) so the highlight does not mirror onto the far side
// of the fibre.
float HairStrandSpecularLobe(vec3 Ts, vec3 V, vec3 L, float exponent) {
    vec3 H = normalize(L + V);
    float dotTH = dot(Ts, H);
    float sinTH = sqrt(max(1.0 - dotTH * dotTH, 0.0));
    // Soften the transition across grazing angles so the lobe fades in/out smoothly rather than
    // clipping to a hard edge at the silhouette.
    float dirAtten = smoothstep(-1.0, 0.0, dotTH);
    return dirAtten * pow(sinTH, exponent);
}

// Full hair BSDF evaluation for a single light direction L (unit radiance). Inputs:
//   T         -- world-space strand TANGENT (direction along the strand, root->tip). MUST be the
//                per-fragment interpolated tangent so the highlight genuinely slides along the hair.
//   N         -- a world-space shading normal for the diffuse term. For fur grown off a surface this
//                is the outward growth direction (FurStrand.vert passes the skinned outward normal),
//                which shades the sunny side of the pelt bright and the shadow side dark.
//   V, L      -- world-space view and light directions (both pointing AWAY from the shading point).
//   hairColor -- the fibre's base albedo/absorption color (root->tip lerp already applied by caller).
//   p         -- HairParams appearance controls.
HairBSDFResult EvaluateHairBSDF(vec3 T, vec3 N, vec3 V, vec3 L, vec3 hairColor, HairParams p) {
    HairBSDFResult r;

    // --- Diffuse: Kajiya-Kay wrapped Lambert ---
    // Pure hair diffuse uses sin(angle(T,L)) (a strand catches the most light when lit perpendicular
    // to its length). We blend that with a softly-wrapped N.L so a fur pelt still reads its overall
    // rounded form (lit side vs shadow side), not just a flat isotropic sheet. The 0.5 wrap keeps
    // strands facing slightly away from the light from crushing instantly to black (light scatters
    // THROUGH the semi-translucent fibre -- the cheap stand-in for the missing TT lobe).
    float sinTL = sqrt(max(1.0 - dot(T, L) * dot(T, L), 0.0));
    float wrappedNL = saturate(dot(N, L) * 0.5 + 0.5);
    float diffuseTerm = sinTL * wrappedNL;
    r.diffuse = hairColor * diffuseTerm;

    // --- Specular: two shifted anisotropic lobes ---
    vec3 Tr = HairShiftTangent(T, N, p.shiftR);
    vec3 Ttrt = HairShiftTangent(T, N, p.shiftTRT);
    float lobeR = HairStrandSpecularLobe(Tr, V, L, p.exponentR);
    float lobeTRT = HairStrandSpecularLobe(Ttrt, V, L, p.exponentTRT);

    // Primary highlight is white; secondary is tinted by the hair color (transmitted-reflected-
    // transmitted light carries the fibre's own color). Both scale by specIntensity.
    vec3 specR = vec3(lobeR);
    vec3 specTRT = p.trtIntensity * hairColor * lobeTRT;
    r.specular = p.specIntensity * (specR + specTRT);

    return r;
}

#endif // HAIR_BSDF_GLSL
