#ifndef MASK_TEXTURE_LIMITS_GLSL
#define MASK_TEXTURE_LIMITS_GLSL

// Bindless procedural cutout mask array slot count. Matches
// renderer::ProceduralMaskGenerator::kMaxMaskTextures -- must be kept in sync with that C++-side
// constant by hand, since GLSL has no way to #include a C++ constant. Shared by the producer
// (ProceduralMaskGenerate.comp, which writes every slot) and every consumer (mask_sampling.glsl,
// which reads a slot per masked cluster) so the two array declarations can never silently drift
// out of sync in size.
#define K_MAX_MASK_TEXTURES 64u

#endif // MASK_TEXTURE_LIMITS_GLSL
