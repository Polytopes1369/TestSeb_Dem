#ifndef MATERIAL_PARAMS_GLSL
#define MATERIAL_PARAMS_GLSL

// Must match renderer::kMaxMaterials (MaterialParameterTable.h) exactly -- bounds the SSBO a
// consumer binds g_MaterialParams to, so a materialID at or beyond this count can be clamped
// in-shader instead of indexing past the buffer's real allocated size.
#define MATERIAL_TABLE_SIZE 32u

// GLSL-side, std430-compatible mirror of renderer::MaterialParameters
// (src/renderer/MaterialParameterTable.h) -- 32 bytes, two {vec3, float} pairs, the C++ mirror's
// own static_assert verifies the match field-for-field. Populated once at
// renderer::ClusterResolvePass::Init() from renderer::kMaterialParameterTable (a CPU-authored
// constexpr table -- no on-disk .cache section, no procedural generation involved: materialID is
// per-entity, not per-triangle, so this table is small and fully static at compile time). Consumed
// by ClusterResolve.comp via a plain fixed-size SSBO array, matching this codebase's own
// "bindless == fixed-size compile-time array" convention (see mask_sampling.glsl's g_MaskTextures
// for the same pattern applied to images instead of a parameter buffer).
struct MaterialParams {
    vec3 baseColor;
    float roughness;
    vec3 emissive;
    float metallic;
};

#endif // MATERIAL_PARAMS_GLSL
