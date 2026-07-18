#ifndef SUBSTRATE_BSDF_GLSL
#define SUBSTRATE_BSDF_GLSL

// UE5.8 Substrate material evaluation -- shared by every lighting path that shades a
// renderer::MaterialParameters/MaterialParams entry (ClusterResolve.comp/ClusterResolveBinned.comp's
// direct sun term, TransparentForward.frag, ReflectionGather.comp's Lumen reflection composite,
// MegaLightsShade.comp's local-light term). See the Substrate integration plan's own Context section
// for what this reproduces (the Slab BSDF's full parameter set + the vertical layering operator +,
// as of gap G6, the horizontal mixing operator -- see that NOTE below) and what remains deliberately
// out of scope (Hair/Eye/Water shading models).
// NOTE (UE5.8 rendering-parity gap G4): full screen-space SSS diffusion was ORIGINALLY out of scope
// here (this file only ever provided the cheap analytic wrap-diffuse approximation, EvaluateSlabDiffuse's
// sssAmount/sssRadius blend) -- it is now implemented as a real separable screen-space diffusion pass
// (renderer::SubsurfaceScatteringPass, driven by SubstrateSlab::sssProfileScale, material_params.glsl),
// which runs AFTER this BSDF has been fully evaluated and composited, so it needs no term here: SSS is
// a post-lighting screen-space material response, not a per-light BSDF lobe.
// NOTE (UE5.8 rendering-parity gap G5): the Glint / sparkle term (the sparkly discrete-microfacet
// "flake" specular response Substrate supports -- metal-flake paint, glittery snow, sugar/salt-like
// surfaces) was ALSO originally out of scope here -- it is now implemented as EvaluateSubstrateGlint /
// EvaluateSlabGlint below, a purely additive high-frequency specular term evaluated per-light ON TOP OF
// the smooth GGX lobe (it never replaces EvaluateSlabSpecular). Unlike the two SSS terms above it is a
// genuine per-light BSDF lobe, but it needs two extra inputs the smooth lobes do not (the shading
// point's world position, to anchor the procedural flake field, and this pixel's world-space footprint,
// to anti-alias it) -- so rather than widen EvaluateSubstrateMaterial's signature and every call site,
// it is exposed as a SEPARATE additive function the direct-lighting shaders sum in alongside the smooth
// response (see ClusterResolve.comp/ClusterResolveBinned.comp's own sun-term call sites). Driven by
// SubstrateSlab::glintDensity/glintIntensity (material_params.glsl); glintIntensity == 0.0 (default)
// disables it entirely at zero cost, exactly like every other optional Slab term here.
// NOTE (UE5.8 rendering-parity gap G6): HORIZONTAL material mixing -- the last of the three
// deliberately-sequential Substrate BSDF gaps (G4 SSS -> G5 Glint -> G6 mixing) -- is now
// implemented as EvaluateSubstrateMixMask / MixSlabs / ApplySubstrateHorizontalMix below, closing
// out what this file's own header (see the line above the #includes) previously listed as its last
// out-of-scope Substrate feature. It is DISTINCT from the vertical-layering operator
// (EvaluateSubstrateMaterial): vertical layering stacks a coat Slab ON TOP OF a base (a
// Fresnel-weighted coat-over-base energy split), whereas horizontal mixing blends two side-by-side
// base Slabs (A == mat.base, B == mat.mixB) across the SAME surface by a per-pixel procedural mask
// (rust over metal, moss over rock, dirt over paint). It is a material PRE-PROCESSING operator, not
// a per-light lobe: ApplySubstrateHorizontalMix resolves A and B into ONE effective base Slab
// BEFORE any lighting term runs (called once per pixel in ClusterResolve.comp/ClusterResolveBinned.
// comp, right after the material lookup), so the vertical-layer/SSS/glint/direct-lighting terms all
// transparently shade the correctly-blended surface with no per-term awareness of mixing.
// mat.mixAmount == 0.0 (default) disables it entirely at zero cost, exactly like every other
// optional Slab term here.
//
// Reuses include/ggx_brdf.glsl's D_GGX/G_SmithGGXCorrelated/F_Schlick/BuildTangentBasis verbatim --
// no duplicate BRDF math (that file is this codebase's one and only isotropic-GGX implementation,
// see its own header comment) -- and include/displacement_noise.glsl's Fbm3D verbatim for the
// horizontal-mix mask (this codebase's one established world-space procedural-noise convention, see
// that file's / ProceduralMaskSampler.h's own headers on never duplicating the noise math).

#include "include/math_utils.glsl"
#include "include/ggx_brdf.glsl"
#include "include/material_params.glsl"
#include "include/iridescence_bsdf.glsl"
#include "include/displacement_noise.glsl"

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
    vec3 response;
    if (mat.topWeight <= 0.0) {
        response = baseResponse;
    } else {
        vec3 topResponse = EvaluateSlabBSDF(mat.top, N, V, L, T, B);
        float NdotV = max(dot(N, V), 1.0e-4);
        vec3 topFresnel = F_SchlickF90(mat.top.f0, mat.top.f90Luminance, NdotV);
        float topFresnelLuma = dot(topFresnel, vec3(0.2126, 0.7152, 0.0722));
        float baseTransmittance = 1.0 - mat.topWeight * clamp(topFresnelLuma, 0.0, 1.0);
        response = topResponse * mat.topWeight + baseResponse * baseTransmittance;
    }
    // Wave 2 (UE5.8 Substrate iridescence layer): thin-film shimmer tint applied to the fully
    // composited BSDF response (post vertical-layering, not per-slab) -- EvaluateIridescence's own
    // early exit makes this a zero-cost no-op for every material that doesn't opt in
    // (iridescenceAmount == 0.0, the default -- see MaterialParameters' own comment).
    return EvaluateIridescence(response, mat.iridescenceAmount, mat.iridescenceThickness, N, V, L);
}

// Additive emissive term (not a function of any light direction, added once per pixel, not per
// light) -- the top slab's own emissive is weighted by its coverage, matching how its BSDF response
// above is weighted.
vec3 EvaluateSubstrateEmissive(MaterialParams mat) {
    return mat.base.emissive + mat.top.emissive * mat.topWeight;
}

// -----------------------------------------------------------------------------------------------
// Substrate HORIZONTAL material mixing (UE5.8 rendering-parity gap G6) -- see this file's own G6
// header NOTE for how it differs from the vertical-layering operator above.
// -----------------------------------------------------------------------------------------------

// The per-pixel A/B mix weight, in [0, mixAmount]. A world-space procedural mask so the blend varies
// spatially across the surface (a constant weight would be a visually flat average -- this project's
// whole ethos is procedural variation), thresholded/sharpened by the material's mixBias/mixContrast.
float EvaluateSubstrateMixMask(MaterialParams mat, vec3 worldPos, float sharpnessScale) {
    // World-space fbm value noise (displacement_noise.glsl's Fbm3D, reused verbatim -- the
    // codebase's one established world-space procedural-noise convention), remapped from Fbm3D's
    // [-1,1] range to [0,1]. WORLD-anchored (not screen- or UV-space): the mask stays glued to the
    // surface point, so it is temporally stable under TAA (same reasoning as the glint field's world
    // anchoring, see EvaluateSlabGlint) and continuous across cluster/LOD boundaries (no per-cluster
    // UV seam, unlike a decoded-UV sample). mixScale floored so a degenerate 0 still samples a
    // finite noise domain rather than collapsing every point onto one cell.
    float n = Fbm3D(worldPos * max(mat.mixScale, 1.0e-4)) * 0.5 + 0.5;
    // Threshold the noise into the mix weight: mixBias sets the 50% point, mixContrast the transition
    // width (higher -> narrower band -> crisper A/B patch edges). sharpnessScale is the Debug-only
    // live tuning multiplier (1.0 in Release, see g_ViewParams.mixMaskSharpnessScale). Both are
    // floored so a degenerate 0 contrast yields a finite (very soft) band, never a divide-by-zero
    // hard seam.
    float sharp = max(mat.mixContrast * sharpnessScale, 1.0e-3);
    float halfBand = 0.5 / sharp;
    float t = smoothstep(mat.mixBias - halfBand, mat.mixBias + halfBand, n);
    return t * clamp(mat.mixAmount, 0.0, 1.0);
}

// Parameter-space lerp of two Slabs into one. Interpolates the physically meaningful Slab quantities
// (albedo, roughness, F0/F90, emissive, anisotropy, the second-specular/SSS/fuzz/glint parameters)
// -- NOT a blend of the two Slabs' final BRDF responses, which is not energy-correct. Lerping the
// inputs then evaluating ONE BRDF is the standard, energy-sane real-time horizontal-mix
// simplification (it is exactly what most engines, UE5.8 included for its default "coverage" blend,
// actually do). No normal term: SubstrateSlab carries no per-material normal (this codebase has no
// normal maps -- the shading normal comes from the decoded cluster geometry and is shared by both
// Slabs), so there is nothing to blend there.
SubstrateSlab MixSlabs(SubstrateSlab a, SubstrateSlab b, float t) {
    SubstrateSlab r;
    r.diffuseAlbedo         = mix(a.diffuseAlbedo, b.diffuseAlbedo, t);
    r.roughness             = mix(a.roughness, b.roughness, t);
    r.f0                    = mix(a.f0, b.f0, t);
    r.f90Luminance          = mix(a.f90Luminance, b.f90Luminance, t);
    r.emissive              = mix(a.emissive, b.emissive, t);
    r.anisotropy            = mix(a.anisotropy, b.anisotropy, t);
    r.secondRoughness       = mix(a.secondRoughness, b.secondRoughness, t);
    r.secondRoughnessWeight = mix(a.secondRoughnessWeight, b.secondRoughnessWeight, t);
    r.sssAmount             = mix(a.sssAmount, b.sssAmount, t);
    r.sssRadius             = mix(a.sssRadius, b.sssRadius, t);
    r.fuzzColor             = mix(a.fuzzColor, b.fuzzColor, t);
    r.fuzzAmount            = mix(a.fuzzAmount, b.fuzzAmount, t);
    r.fuzzRoughness         = mix(a.fuzzRoughness, b.fuzzRoughness, t);
    r.sssProfileScale       = mix(a.sssProfileScale, b.sssProfileScale, t);
    r.glintDensity          = mix(a.glintDensity, b.glintDensity, t);
    r.glintIntensity        = mix(a.glintIntensity, b.glintIntensity, t);
    return r;
}

// Resolves horizontal mixing IN PLACE: replaces mat.base with the mask-blended lerp of base (A) and
// mixB (B), leaving mat.top / mat.topWeight untouched (the vertical coat still layers over the mixed
// result in EvaluateSubstrateMaterial). Returns the raw mask weight via `outMixMask` for the Debug
// mix-mask visualization (ClusterResolve.comp's DEBUG_VIEW_SUBSTRATE_MIXING). Fast path: mixAmount
// <= 0.0 (every material but an authored horizontal-mix one) returns mat unchanged at zero cost --
// mixB is never even read -- exactly the pre-G6 single-base behavior.
MaterialParams ApplySubstrateHorizontalMix(MaterialParams mat, vec3 worldPos, float sharpnessScale, out float outMixMask) {
    outMixMask = 0.0;
    if (mat.mixAmount <= 0.0) {
        return mat;
    }
    float t = EvaluateSubstrateMixMask(mat, worldPos, sharpnessScale);
    outMixMask = t;
    mat.base = MixSlabs(mat.base, mat.mixB, t);
    return mat;
}

// Substrate Glint / sparkle term for ONE Slab (UE5.8 rendering-parity gap G5) -- the discrete-
// microfacet "flake" specular response (metal-flake paint, glittery snow, sugar/salt-like surfaces).
// ADDITIVE to and distinct from the smooth GGX lobe (EvaluateSlabSpecular): where that lobe is the
// STATISTICAL AVERAGE of a surface's microfacets (a single smooth highlight), glint reintroduces the
// individual, spatially-discrete flakes that average washes out -- so on top of the smooth highlight
// the surface twinkles with pinpoint sparkles that pop in and out as the view/light angle changes.
//
// Technique (a footprint-filtered procedural flake population, in the spirit of the real-time
// "faceted-normal-jitter + threshold" glint approaches -- Real-Time Rendering 4th ed. sec 9.9.4, the
// Zirr/Kettunen "stochastic glints" family): a grid of tiny mirror facets is anchored to the surface
// in WORLD space; each cell carries one flake whose microfacet normal is a hashed random perturbation
// of the shading normal, and a cell "sparkles" only when that flake normal is near-perfectly aligned
// with the half-vector H (a high-exponent threshold -> a crisp pinpoint, not a broad blur). It is
// fully procedural -- no texture/data asset (CLAUDE.md hard rule) -- reusing math_utils.glsl's Hash4.
//
// TAA/TSR stability: the flake grid is anchored in WORLD space (surfUV is derived from worldPos, not
// from screen coordinates), so each flake stays glued to its surface point -- TAATSR.comp reprojects
// history by that same world position, so a world-anchored flake reprojects to itself frame to frame
// (it twinkles as the angle changes, which is correct, but it never swims/crawls across the surface,
// which WOULD alias). `footprint` (this pixel's world-space size) drives the anti-alias: as the
// surface recedes and many flakes fall inside one pixel, their mean is already captured by the smooth
// lobe, so the high-frequency sparkle variance is faded out (rather than aliasing into random noise).
// Fading AMPLITUDE while keeping the world grid FIXED avoids the LOD-re-quantization shimmer a
// distance-varying cell size would cause.
vec3 EvaluateSlabGlint(SubstrateSlab slab, vec3 N, vec3 V, vec3 L, vec3 T, vec3 B,
                       vec3 worldPos, float footprint, float densityScale, float intensityScale) {
    float intensity = slab.glintIntensity * intensityScale;
    if (intensity <= 0.0) return vec3(0.0);
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    float VdotH = max(dot(V, H), 0.0);

    // Flake cell size in world units: denser (higher effective glintDensity) -> smaller cells ->
    // more, finer flakes. Clamped so a degenerate 0 still yields a finite, non-zero grid.
    float density = clamp(slab.glintDensity * densityScale, 0.0, 1.0);
    float cellSize = mix(0.20, 0.015, density);

    // Parametrize the surface plane into a stable 2D grid via the procedural tangent basis. T/B are a
    // deterministic function of N (BuildTangentBasis), so this UV is continuous wherever N is, and
    // it is world-anchored (built from worldPos) for the temporal stability described above.
    vec2 surfUV = vec2(dot(worldPos, T), dot(worldPos, B)) / cellSize;
    vec2 cellBase = floor(surfUV);
    vec2 cellFrac = surfUV - cellBase;

    // kFlakeSpread: tangent-plane jitter magnitude of the flake normals about N (a wider cone lets
    // flakes catch the light over a broader range of view/light angles -> a livelier sparkle spread
    // across the whole sunlit hemisphere, not just a pinhole around the smooth highlight where H==N).
    // kSparkExponent: high so only near-mirror-aligned flakes light up (a crisp pinpoint sparkle --
    // the broad response is already the smooth lobe's job, this term must stay high-frequency).
    // Values validated numerically against a curved showcase surface: on a lit sphere they light a
    // sparse ~0.3% of points at a good angle (discrete pinpoints, NOT a broad lobe) at a strong
    // intensity, and the lit set shifts substantially as the view angle changes (the glint "twinkle").
    // A tighter cone / higher exponent (e.g. 0.35 / 700) is too narrow -- the perturbed flake normals
    // can then never reach the half-vector, which typically sits tens of degrees off N, so nothing
    // sparkles at all on curved geometry.
    const float kFlakeSpread = 0.7;
    const float kSparkExponent = 220.0;

    // 2x2 bilinear over the neighboring cells so the sparkle field has no hard cell-boundary popping
    // as worldPos crosses a cell edge (spatially and, with the world anchoring above, temporally
    // smooth) -- each cell contributes exactly one discrete flake.
    float sparkle = 0.0;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            vec2 cellCoord = cellBase + vec2(float(dx), float(dy));
            vec4 h = Hash4(cellCoord);
            // Random tangent-plane jitter -> a perturbed per-flake microfacet normal.
            vec2 jitter = (h.xy * 2.0 - 1.0) * kFlakeSpread;
            vec3 flakeN = normalize(N + T * jitter.x + B * jitter.y);
            float align = max(dot(flakeN, H), 0.0);
            float flake = pow(align, kSparkExponent);
            float wx = (dx == 0) ? (1.0 - cellFrac.x) : cellFrac.x;
            float wy = (dy == 0) ? (1.0 - cellFrac.y) : cellFrac.y;
            sparkle += flake * wx * wy;
        }
    }

    // Footprint anti-alias (see this function's header comment): fade sparkle amplitude toward 0 once
    // a pixel's footprint spans many flakes, keeping the world grid fixed.
    float aa = clamp(cellSize / max(footprint, 1.0e-4), 0.0, 1.0);
    sparkle *= aa;

    // Fresnel-tinted (each flake is a little mirror -> reflects the slab's own specular color) and
    // NdotL-weighted, exactly like the smooth specular lobe this ADDS to.
    vec3 F = F_SchlickF90(slab.f0, slab.f90Luminance, VdotH);
    return F * sparkle * intensity * NdotL;
}

// Full-material Glint term: the additive sparkle contribution of MaterialParams' base (+ optional
// top) Slabs to one light direction L. Mirrors EvaluateSubstrateMaterial's own vertical-layering
// structure (a top coat attenuates the base's glint by how much energy it reflects at this view
// angle, exactly like the base BSDF response is attenuated there) so a glint-flagged base still reads
// correctly under a Clear Coat. Returns vec3(0.0) at zero cost when no Slab uses glint. Kept separate
// from EvaluateSubstrateMaterial (rather than folded into it) so the smooth-lobe call sites that do
// not want to pay for -- or plumb the worldPos/footprint of -- glint are entirely unaffected.
vec3 EvaluateSubstrateGlint(MaterialParams mat, vec3 N, vec3 V, vec3 L,
                            vec3 worldPos, float footprint, float densityScale, float intensityScale) {
    // Fast path: neither slab uses glint (the common case) -> zero cost, matching every other
    // Substrate term's "0.0 default disables it entirely" convention.
    bool baseHasGlint = mat.base.glintIntensity > 0.0;
    bool topHasGlint = (mat.topWeight > 0.0) && (mat.top.glintIntensity > 0.0);
    if (!baseHasGlint && !topHasGlint) {
        return vec3(0.0);
    }
    vec3 T, B;
    BuildProceduralTangentBasis(N, T, B);
    vec3 baseGlint = EvaluateSlabGlint(mat.base, N, V, L, T, B, worldPos, footprint, densityScale, intensityScale);
    if (mat.topWeight <= 0.0) {
        return baseGlint;
    }
    vec3 topGlint = EvaluateSlabGlint(mat.top, N, V, L, T, B, worldPos, footprint, densityScale, intensityScale);
    // Same energy-conserving vertical-layer weighting EvaluateSubstrateMaterial uses (see its own
    // comment): the top layer's Fresnel reflectance at this view angle sets how much reaches the base.
    float NdotV = max(dot(N, V), 1.0e-4);
    vec3 topFresnel = F_SchlickF90(mat.top.f0, mat.top.f90Luminance, NdotV);
    float topFresnelLuma = dot(topFresnel, vec3(0.2126, 0.7152, 0.0722));
    float baseTransmittance = 1.0 - mat.topWeight * clamp(topFresnelLuma, 0.0, 1.0);
    return topGlint * mat.topWeight + baseGlint * baseTransmittance;
}

// Atmos weather system, surface response extension (UE5.8 Substrate / Ubisoft Atmos parity: dynamic
// wet/snow surfaces): mutates `mat.base` in place to fake a thin surface layer of rainwater and/or
// snow before the BSDF is evaluated -- a cheap material-parameter modulation, NOT a second Substrate
// Slab/vertical-layering operator (a genuinely physical wet/snow layer would be exactly that, but a
// full extra Slab per pixel is unjustified cost for a demoscene-scale global weather effect). Only
// `mat.base` is touched -- `mat.top` (Clear Coat/Fuzz, see EvaluateSubstrateMaterial's own vertical-
// layering comment) already models its own distinct surface response and is left alone, the same way
// ComputeTerrainAlbedo (terrain_shading.glsl) only ever touches mat.base.diffuseAlbedo too.
//
// `wetness`/`snowCoverage` are both global [0,1] scalars (renderer::AtmosClimatePass::
// GetSurfaceWetness()/GetSnowCoverage(), integrated once per frame CPU-side from the live climate
// state -- see that class' own RecordUpdate() comment for the exact exponential-decay formula and
// RAIN_STRENGTH/RELATIVE_HUMIDITY/TEMPERATURE_CELSIUS derivation) threaded down through
// ResolveViewParamsUBO -- see ClusterResolve.comp's own binding comment for why this reuses that
// UBO's previously-unused std140 padding floats rather than a new binding.
//
// Wetness (rain/humidity): darkens diffuse albedo (a wet surface absorbs more of the diffusely
// scattered light -- water fills in a rough surface's micro-shadowing) and lowers roughness (a thin
// water film smooths the effective micro-surface, sharpening the specular highlight), applied
// UNIFORMLY across every exposed surface -- unlike snow below, rain wets front/side/top faces alike
// (no normal-based masking), matching this feature's own "global, not per-surface" scope. The F0/F90
// bump is a cheap stand-in for a genuine thin-film interference term (explicitly out of scope, see
// this file's own header comment on what Substrate features this reproduces) -- just enough of a
// grazing-angle sheen boost to read as "wet" without a real thin-film BRDF.
//
// Snow coverage (cold + precipitation): blends albedo/roughness/F0 toward a flat snow response,
// MASKED by how upward-facing the surface normal is (dot(N, up), smoothstepped) -- snow settles on
// roofs/ground/upward slopes, not vertical cliff faces or the undersides of overhangs, so skipping
// this mask (a naive uniform blend like wetness's own) would look physically wrong in a way a
// demoscene audience immediately notices, unlike wetness's uniform application which reads fine
// because rain genuinely does wet every exposed face.
MaterialParams ApplySurfaceWeather(MaterialParams mat, vec3 N, float wetness, float snowCoverage) {
    if (wetness > 0.0) {
        // Glossier (lower roughness) the wetter the surface is -- floor at 0.02 so a fully wet,
        // already-smooth material (e.g. metal) never divides by a near-zero roughness downstream in
        // EvaluateSlabSpecular's D_GGXAnisotropic/G_SmithGGXCorrelated terms.
        float wetRoughnessScale = mix(1.0, 0.35, wetness);
        mat.base.roughness = max(mat.base.roughness * wetRoughnessScale, 0.02);
        // Darken toward ~65% of the dry albedo at full wetness (matches this feature's own "lerp
        // toward ~60-70% of dry value" spec).
        mat.base.diffuseAlbedo *= mix(vec3(1.0), vec3(0.65), wetness);
        // Thin-film specular boost: raise F0 toward water's own ~0.03 dielectric baseline (a no-op
        // for anything already at or above that, e.g. metals) and brighten the grazing-angle F90 edge
        // tint -- see this function's own header comment on why this is a cheap stand-in, not a real
        // thin-film BRDF term.
        mat.base.f0 = mix(mat.base.f0, max(mat.base.f0, vec3(0.03)), wetness);
        mat.base.f90Luminance = mix(mat.base.f90Luminance, max(mat.base.f90Luminance, 1.0), wetness * 0.5);
    }
    if (snowCoverage > 0.0) {
        // smoothstep(0.3, 0.8, upFacing): near-0 on a vertical cliff face (dot ~= 0), ramping to
        // near-full snow response only once a surface is genuinely mostly-upward-facing (dot >= 0.8,
        // i.e. within ~37 degrees of straight up) -- a gentle, not binary, transition band so a
        // rolling hillside doesn't show a hard snow-line seam at some arbitrary slope angle.
        float upFacing = clamp(dot(N, vec3(0.0, 1.0, 0.0)), -1.0, 1.0);
        float snowMask = smoothstep(0.3, 0.8, upFacing) * snowCoverage;
        if (snowMask > 0.0) {
            const vec3 kSnowAlbedo = vec3(0.92, 0.94, 0.98);
            const float kSnowRoughness = 0.85; // Fresh snow is a soft, near-Lambertian diffuse scatterer, not glossy.
            const vec3 kSnowF0 = vec3(0.03);   // Flat, uncolored dielectric F0 -- snow has no authored metal tint.
            mat.base.diffuseAlbedo = mix(mat.base.diffuseAlbedo, kSnowAlbedo, snowMask);
            mat.base.roughness = mix(mat.base.roughness, kSnowRoughness, snowMask);
            mat.base.f0 = mix(mat.base.f0, kSnowF0, snowMask);
        }
    }
    return mat;
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
