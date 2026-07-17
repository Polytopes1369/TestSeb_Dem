#ifndef VIRTUAL_TEXTURE_LIMITS_GLSL
#define VIRTUAL_TEXTURE_LIMITS_GLSL

// Bindless physical pool array slot count -- a bounded array (mirrors mask_texture_limits.glsl's
// own K_MAX_MASK_TEXTURES convention exactly), NOT a true unsized/runtime-length GLSL array: this
// engine only enables VkPhysicalDeviceVulkan12Features::shaderSampledImageArrayNonUniformIndexing
// (see renderer::VulkanContext's own feature chain), not ::runtimeDescriptorArray -- a genuinely
// unsized `sampler2DArray g_PhysicalPools[]` declaration would require that second feature. A fixed
// upper bound sized generously for every physical pool channel a virtual texture could plausibly
// need (Albedo, Normal, ORM, ...) sidesteps the extra device-feature dependency entirely, exactly
// like the mask array already does for its own bindless texture slots. Must be kept in sync by hand
// with whatever `physicalPoolFormats` a caller passes to renderer::VirtualTextureManager::Init --
// GLSL has no way to #include a C++ constant, same caveat mask_texture_limits.glsl's own comment
// already documents.
#define K_MAX_VT_PHYSICAL_POOLS 8u

#endif // VIRTUAL_TEXTURE_LIMITS_GLSL
