#ifndef MASK_SAMPLING_GLSL
#define MASK_SAMPLING_GLSL
#extension GL_EXT_nonuniform_qualifier : require

// Bindless procedural cutout mask array (renderer::ProceduralMaskGenerator generates every slot
// once at startup -- see ProceduralMaskGenerate.comp) -- shared by ClusterRaster.frag (hard
// discard), ClusterSoftwareRaster.comp (skip the atomic write), and ClusterResolve.comp (soft
// edge-feathering blend). The includer must #define MASK_ARRAY_SET / MASK_ARRAY_BINDING before
// including this header, matching whichever binding slot that shader's own descriptor set
// reserves for it (matches renderer::ProceduralMaskGenerator::kMaxMaskTextures).
#define K_MAX_MASK_TEXTURES 64u

layout(set = MASK_ARRAY_SET, binding = MASK_ARRAY_BINDING) uniform sampler2D g_MaskTextures[K_MAX_MASK_TEXTURES];

// Returns 1.0 (fully opaque, no cutout) for the geometry::kInvalidMaskTextureIndex sentinel
// (0xFFFFFFFF -- see ClusterCullMetadata::maskTextureIndex) without ever touching the array --
// the common case for any non-foliage cluster. Otherwise samples the indexed mask slot's red
// channel (alpha), using nonuniformEXT since maskTextureIndex varies per-cluster/per-invocation
// and is therefore not dynamically uniform across the shader invocation group (unlike
// ProceduralMaskGenerate.comp's own workgroup-uniform slot indexing at generation time).
float SampleMaskAlpha(uint maskTextureIndex, vec2 uv) {
    if (maskTextureIndex == 0xFFFFFFFFu) {
        return 1.0;
    }
    return texture(g_MaskTextures[nonuniformEXT(maskTextureIndex)], uv).r;
}

#endif // MASK_SAMPLING_GLSL
