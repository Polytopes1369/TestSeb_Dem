#ifndef MATERIAL_PARAMS_GLSL
#define MATERIAL_PARAMS_GLSL

// Must match renderer::kMaxMaterials (MaterialParameterTable.h) exactly -- bounds the SSBO a
// consumer binds g_MaterialParams to, so a materialID at or beyond this count can be clamped
// in-shader instead of indexing past the buffer's real allocated size.
#define MATERIAL_TABLE_SIZE 32u

// UE5.8 Substrate "Slab" BSDF -- the single atomic building block every Substrate material is made
// of. GLSL-side, std430-compatible mirror of renderer::SubstrateSlab
// (src/renderer/MaterialParameterTable.h) -- 96 bytes, six 16-byte blocks. See that struct's own
// header comment for the full field-by-field rationale (F0/F90 stored directly rather than derived
// from a metallic scalar, anisotropy/second-specular/fuzz/subsurface all zero-cost-disabled at their
// default of 0.0).
struct SubstrateSlab {
    vec3 diffuseAlbedo;
    float roughness;
    vec3 f0;
    float f90Luminance;
    vec3 emissive;
    float anisotropy;
    float secondRoughness;
    float secondRoughnessWeight;
    float sssAmount;
    float sssRadius;
    vec3 fuzzColor;
    float fuzzAmount;
    float fuzzRoughness;
    // Screen-space Subsurface Scattering (UE5.8 rendering-parity gap G4, "Subsurface Profile"
    // shading model) -- world-space diffusion radius consumed by renderer::SubsurfaceScatteringPass'
    // separable screen-space diffusion blur (the Jimenez/Burley "Separable SSS" technique). 0.0
    // (default) = disabled: the pixel is never touched by that pass, exactly the pre-G4 behavior at
    // zero extra cost. DISTINCT from sssAmount/sssRadius above: those drive the cheap analytic
    // wrap-diffuse baked into EvaluateSlabDiffuse (substrate_bsdf.glsl), whereas this drives a real
    // post-lighting screen-space diffusion pass -- exactly Substrate's own distinction between its
    // "Subsurface" (wrap) and "Subsurface Profile" (screen-space diffusion) models. A material
    // authors ONE or the other, never both (both together would double-count subsurface transport).
    // Occupies what was _pad0. See renderer::SubstrateSlab::sssProfileScale (MaterialParameterTable.h).
    float sssProfileScale;
    // Glint / sparkle (UE5.8 rendering-parity gap G5, discrete-microfacet "flake" specular) -- drive
    // substrate_bsdf.glsl's EvaluateSlabGlint/EvaluateSubstrateGlint. glintDensity [0,1] controls how
    // fine/numerous the flakes are (denser -> smaller world-space flake cells); glintIntensity is the
    // sparkle brightness and its "enable" flag in one -- glintIntensity == 0.0 (default) disables the
    // whole term at zero cost, exactly like sssProfileScale/fuzzAmount/secondRoughnessWeight above.
    // Occupy what were _pad1/_pad2 (no struct-size change). See renderer::SubstrateSlab::glintDensity.
    float glintDensity;
    float glintIntensity;
};

// GLSL-side, std430-compatible mirror of renderer::MaterialParameters
// (src/renderer/MaterialParameterTable.h) -- 336 bytes: three SubstrateSlab blocks (base + the
// horizontal-mix partner `mixB` + optional vertical-layer `top`) plus three trailing 16-byte blocks
// (opacity/reflection controls, horizontal-mix controls, Wave 2 iridescence), matching Substrate's
// own two composition operators plus the iridescence layer -- the C++ mirror's own static_assert
// verifies the match field-for-field. Populated once at renderer::ClusterResolvePass::Init() from
// renderer::GenerateShowcaseMaterialTable() (a fully deterministic, hand-authored table -- no
// on-disk .cache section: materialID is per-entity, not per-triangle, so this table is small and
// fully built once at startup). Consumed by ClusterResolve.comp/ClusterResolveBinned.comp/
// TransparentForward.frag/ReflectionGather.comp/MegaLightsShade.comp/the Surface Cache capture
// shader via a plain fixed-size SSBO array, matching this codebase's own "bindless == fixed-size
// compile-time array" convention (see mask_sampling.glsl's g_MaskTextures for the same pattern
// applied to images instead of a parameter buffer). See substrate_bsdf.glsl's
// EvaluateSubstrateMaterial for how base+top are composited VERTICALLY, and
// ApplySubstrateHorizontalMix for how base+mixB are composited HORIZONTALLY.
struct MaterialParams {
    SubstrateSlab base;
    // Horizontal mixing (UE5.8 rendering-parity gap G6): the "B side" Slab that `base` (the "A
    // side") is spatially blended with across the SAME surface by a per-pixel procedural mask --
    // rust patches over bare metal, moss over rock, etc. DISTINCT from `top` below: `top` is a coat
    // stacked VERTICALLY on top of the base (Substrate's layering operator), whereas `mixB` sits
    // side-by-side with `base` and one wins per pixel by the mask weight (Substrate's mixing
    // operator). Never evaluated at all unless `mixAmount` (below) > 0.0 -- exactly the pre-G6
    // single-base behavior at identical cost. See substrate_bsdf.glsl's ApplySubstrateHorizontalMix.
    SubstrateSlab mixB;
    SubstrateSlab top;
    // Substrate's vertical-layer Coverage/weight, [0, 1]. 0.0 = `top` is never evaluated -- exactly
    // the pre-Substrate single-Slab behavior, at identical cost.
    float topWeight;
    // 1.0 = opaque. < 1.0 = translucent/transparent -- see renderer::MaterialParameters::alpha's
    // own comment for why such materials never reach this struct's usual opaque consumers and are
    // shaded by TransparentForward.frag instead.
    float alpha;
    // Phase PP3: >0.0 = writes a procedural refraction offset into TransparentForward.frag's own
    // second output attachment -- see renderer::MaterialParameters::heatDistortion's own comment.
    float heatDistortion;
    // UE5.8 Lumen "Output Reflections" equivalent -- see renderer::MaterialParameters::
    // hasReflections' own comment (MaterialParameterTable.h). 0.0 = off, 1.0 = on.
    float hasReflections;
    // Horizontal mixing (UE5.8 rendering-parity gap G6) control block -- see the matching fields in
    // renderer::MaterialParameters (MaterialParameterTable.h) for the full authored-range rationale.
    // mixAmount [0,1]: master coverage AND enable flag in one -- 0.0 (default) disables the whole
    // operator at zero cost (mixB never touched), exactly like topWeight/glintIntensity's own
    // "0.0 == off" convention. mixScale: world-space noise frequency (cycles/world-unit) of the
    // procedural blend mask. mixContrast: transition sharpness (higher -> crisper A/B boundary,
    // more like distinct patches; lower -> a soft gradient). mixBias [0,1]: mask threshold (shifts
    // how much of the surface reads as B vs A). See substrate_bsdf.glsl's EvaluateSubstrateMixMask.
    float mixAmount;
    float mixScale;
    float mixContrast;
    float mixBias;
    // Wave 2 (UE5.8 Substrate iridescence layer) -- see renderer::MaterialParameters::
    // iridescenceAmount/iridescenceThickness's own comment (MaterialParameterTable.h). Consumed by
    // substrate_bsdf.glsl's EvaluateSubstrateMaterial via include/iridescence_bsdf.glsl.
    float iridescenceAmount;
    float iridescenceThickness;
    float _padIridescence0;
    float _padIridescence1;
};

// Phase 7b (UE5.8 parity roadmap, terrain heightfield): mirror of renderer::kTerrainMaterialID
// (MaterialParameterTable.h) -- keep in sync.
#define MATERIAL_ID_TERRAIN 16u

#endif // MATERIAL_PARAMS_GLSL
