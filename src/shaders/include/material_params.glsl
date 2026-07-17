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
    float _pad0;
    float _pad1;
    float _pad2;
};

// GLSL-side, std430-compatible mirror of renderer::MaterialParameters
// (src/renderer/MaterialParameterTable.h) -- 208 bytes: two SubstrateSlab blocks (base + optional
// top) plus a trailing 16-byte block, matching Substrate's own "vertical layering" model -- the
// C++ mirror's own static_assert verifies the match field-for-field. Populated once at
// renderer::ClusterResolvePass::Init() from renderer::GenerateShowcaseMaterialTable() (a fully
// deterministic, hand-authored table -- no on-disk .cache section: materialID is per-entity, not
// per-triangle, so this table is small and fully built once at startup). Consumed by
// ClusterResolve.comp/ClusterResolveBinned.comp/TransparentForward.frag/ReflectionGather.comp/
// MegaLightsShade.comp/the Surface Cache capture shader via a plain fixed-size SSBO array, matching
// this codebase's own "bindless == fixed-size compile-time array" convention (see
// mask_sampling.glsl's g_MaskTextures for the same pattern applied to images instead of a parameter
// buffer). See substrate_bsdf.glsl's EvaluateSubstrateMaterial for how base+top are composited.
struct MaterialParams {
    SubstrateSlab base;
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
};

// Phase 7b (UE5.8 parity roadmap, terrain heightfield): mirror of renderer::kTerrainMaterialID
// (MaterialParameterTable.h) -- keep in sync.
#define MATERIAL_ID_TERRAIN 16u

#endif // MATERIAL_PARAMS_GLSL
