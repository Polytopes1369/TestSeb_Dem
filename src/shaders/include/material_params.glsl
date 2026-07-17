#ifndef MATERIAL_PARAMS_GLSL
#define MATERIAL_PARAMS_GLSL

// Must match renderer::kMaxMaterials (MaterialParameterTable.h) exactly -- bounds the SSBO a
// consumer binds g_MaterialParams to, so a materialID at or beyond this count can be clamped
// in-shader instead of indexing past the buffer's real allocated size.
#define MATERIAL_TABLE_SIZE 32u

// GLSL-side, std430-compatible mirror of renderer::MaterialParameters
// (src/renderer/MaterialParameterTable.h) -- 48 bytes, three {vec3, float}-shaped 16-byte blocks,
// the C++ mirror's own static_assert verifies the match field-for-field. Populated once at
// renderer::ClusterResolvePass::Init() from renderer::GenerateRandomMaterialTable() (a seeded,
// runtime-generated table -- no on-disk .cache section: materialID is per-entity, not
// per-triangle, so this table is small and fully built once at startup). Consumed by
// ClusterResolve.comp/ClusterResolveBinned.comp/TransparentForward.frag via a plain fixed-size
// SSBO array, matching this codebase's own "bindless == fixed-size compile-time array" convention
// (see mask_sampling.glsl's g_MaskTextures for the same pattern applied to images instead of a
// parameter buffer).
struct MaterialParams {
    vec3 baseColor;
    float roughness;
    vec3 emissive;
    float metallic;
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
    float _pad2;
};

// Phase 7b (UE5.8 parity roadmap, terrain heightfield): mirror of renderer::kTerrainMaterialID
// (MaterialParameterTable.h) -- keep in sync.
#define MATERIAL_ID_TERRAIN 16u

#endif // MATERIAL_PARAMS_GLSL
