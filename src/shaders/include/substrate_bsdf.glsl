#ifndef SUBSTRATE_BSDF_GLSL
#define SUBSTRATE_BSDF_GLSL

// UE5.8 Substrate material evaluation -- shared by every lighting path that shades a
// renderer::MaterialParameters/MaterialParams entry (ClusterResolve.comp/ClusterResolveBinned.comp's
// direct sun term, TransparentForward.frag, ReflectionGather.comp's Lumen reflection composite,
// MegaLightsShade.comp's local-light term). See the Substrate integration plan's own Context section
// for what this reproduces (the Slab BSDF's full parameter set + the vertical layering operator) and
// what is deliberately out of scope (horizontal mixing, Hair/Eye/Water shading models, full
// screen-space SSS diffusion, glint).
//
// Reuses include/ggx_brdf.glsl's D_GGX/G_SmithGGXCorrelated/F_Schlick/BuildTangentBasis verbatim --
// no duplicate BRDF math (that file is this codebase's one and only isotropic-GGX implementation,
// see its own header comment).

#include "include/math_utils.glsl"
#include "include/ggx_brdf.glsl"
#include "include/material_params.glsl"

// Generalized Schlick Fresnel with Substrate's F90 edge-tint parameter (F_Schlick in ggx_brdf.glsl
// hardcodes F90 = vec3(1.0); a Slab's f90Luminance can tint the grazing-angle response instead).
vec3 F_SchlickF90(vec3 F0, float f90Luminance, float cosTheta) {
    float f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    vec3 F90 = vec3(f90Luminance);
    return F0 + (F90 - F0) * f;
}

// Anisotropic Trowbridge-Reitz/GGX normal distribution function (Burley, "Physically-Based Shading
// at Disney", 2012, sec 5.3) -- D_GGX in ggx_brdf.glsl is the isotropic special case (roughnessU ==
// roughnessV). T/B must be an orthonormal tangent/bitangent basis about N (see
// BuildProceduralTangentBasis below).
float D_GGXAnisotropic(vec3 H, vec3 T, vec3 B, vec3 N, float roughnessU, float roughnessV) {
    float NdotH = dot(N, H);
    if (NdotH <= 0.0) return 0.0;
    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    float ax = max(roughnessU, 1.0e-3);
    float ay = max(roughnessV, 1.0e-3);
    float term = (TdotH * TdotH) / (ax * ax) + (BdotH * BdotH) / (ay * ay) + NdotH * NdotH;
    return 1.0 / max(PI * ax * ay * term * term, 1.0e-7);
}

// Charlie sheen distribution (Estevez & Kulla, "Production Friendly Microfacet Sheen BRDF", 2017) --
// Substrate's Fuzz term. Unlike the metallic specular lobe above, cloth/velvet sheen peaks at
// grazing angles rather than at the mirror direction, which D_GGX cannot reproduce.
float D_Charlie(float roughness, float NdotH) {
    float a = max(roughness, 1.0e-3);
    float invAlpha = 1.0 / a;
    float cos2h = NdotH * NdotH;
    float sin2h = max(1.0 - cos2h, 7.8125e-3);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / TAU;
}

// Neubelt & Pettineo's cheap ambient-ish sheen visibility term ("Crafting a Next-Gen Material
// Pipeline for The Order: 1886", 2013) -- paired with D_Charlie above the same way this codebase's
// other real-time BRDFs (ggx_brdf.glsl) pair a distribution with a matching visibility term.
float V_Neubelt(float NdotV, float NdotL) {
    return saturate(1.0 / max(4.0 * (NdotL + NdotV - NdotL * NdotV), 1.0e-4));
}

// This codebase decodes position/normal/UV only (cluster_vertex_decode.glsl) -- no authored
// per-vertex tangents exist for anisotropic specular to align to. BuildTangentBasis (ggx_brdf.glsl,
// Duff et al. 2017) derives a deterministic, per-pixel-stable orthonormal basis from the normal
// alone; consistent (no visible seams -- the same normal always yields the same tangent) even though
// it cannot follow an authored "grain direction" the way a real anisotropic material would.
void BuildProceduralTangentBasis(vec3 N, out vec3 T, out vec3 B) {
    BuildTangentBasis(N, T, B);
}

// Substrate Slab diffuse term: Lambertian, blended toward an analytic wrap-diffuse term by
// sssAmount/sssRadius (the Subsurface approximation -- see SubstrateSlab's own header comment for
// why this is analytic rather than a screen-space diffusion pass). No metallic energy-conservation
// multiply here -- unlike the old metallic-workflow shading, a Substrate Slab's diffuseAlbedo is
// authored directly (renderer::GenerateShowcaseMaterialTable bakes a metal's diffuseAlbedo to zero at
// generation time -- see that function's own comment), matching how a real Substrate material is
// authored: a metal's color lives entirely in F0, never in diffuseAlbedo.
// Real photometric light-unit recalibration (2026-07-17, see renderer::DirectionalLight's own
// comment): a properly normalized Lambertian BRDF is `albedo / PI`, not bare `albedo` -- every
// caller of EvaluateSubstrateMaterial now multiplies its result directly by the light's real
// illuminance (lux/candela), so this term (and this term alone -- the specular/fuzz lobes below are
// already correctly-normalized microfacet BRDFs with no such factor) needs the /PI baked in here to
// stay physically consistent.
vec3 EvaluateSlabDiffuse(SubstrateSlab slab, vec3 N, vec3 L) {
    float NdotL = dot(N, L);
    float lambert = max(NdotL, 0.0);
    if (slab.sssAmount > 0.0) {
        float wrap = clamp(slab.sssRadius, 0.0, 1.0);
        float wrapped = max((NdotL + wrap) / (1.0 + wrap), 0.0);
        lambert = mix(lambert, wrapped, clamp(slab.sssAmount, 0.0, 1.0));
    }
    return (slab.diffuseAlbedo / PI) * lambert;
}

// Substrate Slab specular term: anisotropic GGX (Smith height-correlated visibility, single
// averaged-roughness -- a standard real-time simplification of the full anisotropic visibility term,
// matching this codebase's own "Lumen Performance mode" simplifications elsewhere), Fresnel with F90
// edge tint, and an optional second lobe (Substrate's "Haze") blended into the same distribution by
// secondRoughnessWeight. Returns the D*Vis*F BRDF value -- caller applies NdotL (see
// EvaluateSlabBSDF), matching ggx_brdf.glsl's own G_SmithGGXCorrelated convention of returning
// visibility (already divided by 4*NdotV*NdotL) rather than the raw geometry term.
vec3 EvaluateSlabSpecular(SubstrateSlab slab, vec3 N, vec3 V, vec3 L, vec3 T, vec3 B) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float NdotV = max(dot(N, V), 1.0e-4);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Disney's anisotropic roughness remap (Burley 2012, sec 5.3): aspect in (0,1], 0 anisotropy ->
    // roughnessU == roughnessV == slab.roughness (falls back to the isotropic case exactly).
    float aspect = sqrt(max(1.0 - 0.9 * abs(slab.anisotropy), 0.0));
    float roughnessU = max(slab.roughness / max(aspect, 1.0e-3), 1.0e-3);
    float roughnessV = max(slab.roughness * aspect, 1.0e-3);

    float D = D_GGXAnisotropic(H, T, B, N, roughnessU, roughnessV);
    if (slab.secondRoughnessWeight > 0.0) {
        float D2 = D_GGX(NdotH, max(slab.secondRoughness, 1.0e-3));
        D = mix(D, D2, clamp(slab.secondRoughnessWeight, 0.0, 1.0));
    }
    float Vis = G_SmithGGXCorrelated(NdotV, NdotL, max(slab.roughness, 1.0e-3));
    vec3 F = F_SchlickF90(slab.f0, slab.f90Luminance, VdotH);
    return D * Vis * F;
}

// Substrate Fuzz term: additive grazing sheen (cloth/velvet), independent of the specular lobe above
// -- Substrate applies Fuzz as a per-slab additive modifier, not a separate material category.
vec3 EvaluateSlabFuzz(SubstrateSlab slab, vec3 N, vec3 V, vec3 L) {
    if (slab.fuzzAmount <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float NdotH = saturate(dot(N, H));
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float D = D_Charlie(slab.fuzzRoughness, NdotH);
    float Vis = V_Neubelt(NdotV, NdotL);
    return slab.fuzzColor * slab.fuzzAmount * D * Vis;
}

// Full single-Slab BSDF response to one light direction L (diffuse + specular + fuzz, each already
// NdotL-weighted) -- caller multiplies by the light's own color/intensity/attenuation.
vec3 EvaluateSlabBSDF(SubstrateSlab slab, vec3 N, vec3 V, vec3 L, vec3 T, vec3 B) {
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = EvaluateSlabDiffuse(slab, N, L);
    vec3 specular = EvaluateSlabSpecular(slab, N, V, L, T, B) * NdotL;
    vec3 fuzz = EvaluateSlabFuzz(slab, N, V, L) * NdotL;
    return diffuse + specular + fuzz;
}

// Substrate's vertical layering operator: composites MaterialParams' base+top Slabs. topWeight == 0
// is the fast path (mat.top is never evaluated -- identical result and cost to a single flat Slab).
// Otherwise the top slab's own BSDF response is weighted by its coverage, and the base slab's
// response is attenuated by how much energy the top layer reflects/absorbs before any light reaches
// through to the base -- approximated by the top layer's own Fresnel reflectance at this view angle
// (a coat that reflects more at this angle transmits correspondingly less to the base), which is
// exactly the physical intent of Substrate's vertical stack, not a naive linear cross-fade.
vec3 EvaluateSubstrateMaterial(MaterialParams mat, vec3 N, vec3 V, vec3 L) {
    vec3 T, B;
    BuildProceduralTangentBasis(N, T, B);
    vec3 baseResponse = EvaluateSlabBSDF(mat.base, N, V, L, T, B);
    if (mat.topWeight <= 0.0) {
        return baseResponse;
    }
    vec3 topResponse = EvaluateSlabBSDF(mat.top, N, V, L, T, B);
    float NdotV = max(dot(N, V), 1.0e-4);
    vec3 topFresnel = F_SchlickF90(mat.top.f0, mat.top.f90Luminance, NdotV);
    float topFresnelLuma = dot(topFresnel, vec3(0.2126, 0.7152, 0.0722));
    float baseTransmittance = 1.0 - mat.topWeight * clamp(topFresnelLuma, 0.0, 1.0);
    return topResponse * mat.topWeight + baseResponse * baseTransmittance;
}

// Additive emissive term (not a function of any light direction, added once per pixel, not per
// light) -- the top slab's own emissive is weighted by its coverage, matching how its BSDF response
// above is weighted.
vec3 EvaluateSubstrateEmissive(MaterialParams mat) {
    return mat.base.emissive + mat.top.emissive * mat.topWeight;
}

// Lumen reflection composite weight: the Fresnel-only "Lumen Performance mode" weighting
// ReflectionGather.comp already used pre-Substrate (see that file's own header comment), generalized
// to Substrate's vertical layering the same way EvaluateSubstrateMaterial's direct-lighting term is
// -- a traced reflection ray has no single L, so there is no BRDF lobe to evaluate, only the
// Fresnel term weighting how much of the traced radiance this surface reflects back.
vec3 EvaluateSubstrateReflectionWeight(MaterialParams mat, vec3 N, vec3 V) {
    float NdotV = max(dot(N, V), 1.0e-4);
    vec3 baseF = F_SchlickF90(mat.base.f0, mat.base.f90Luminance, NdotV);
    if (mat.topWeight <= 0.0) {
        return baseF;
    }
    vec3 topF = F_SchlickF90(mat.top.f0, mat.top.f90Luminance, NdotV);
    float topFresnelLuma = dot(topF, vec3(0.2126, 0.7152, 0.0722));
    float baseTransmittance = 1.0 - mat.topWeight * clamp(topFresnelLuma, 0.0, 1.0);
    return topF * mat.topWeight + baseF * baseTransmittance;
}

#endif // SUBSTRATE_BSDF_GLSL
