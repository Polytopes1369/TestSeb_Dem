#ifndef SHADOW_ATLAS_SAMPLING_GLSL
#define SHADOW_ATLAS_SAMPLING_GLSL

// Phase 3 (UE5.8 parity roadmap): the shared physical page atlas (renderer::
// VirtualShadowMapPool::GetPhysicalPoolArrayView(), a sampler2DArray -- one array layer per
// physical page, see that class's own header comment for why a texture array beats a packed flat
// atlas for uniform-size pages) plus the manual PCF depth test every VSM consumer (sun clipmap,
// point light cube faces) shares. Requires shadow_page_table.glsl's SHADOW_PAGE_TEXELS -- include
// that header first.
//
// The includer must #define SHADOW_ATLAS_SET / SHADOW_ATLAS_BINDING before including this header.

layout(set = SHADOW_ATLAS_SET, binding = SHADOW_ATLAS_BINDING) uniform sampler2DArray g_ShadowPhysicalAtlas;

// Manual PCF (3x3 box filter), plain (non-comparison) sampler -- same convention as this
// codebase's pre-Phase-3 single shadow map (see renderer::ShadowMapPass's own "Plain sampler2D,
// not shadow/compare sampler" note). `pageLocalUV` is this sample's position within the page,
// already in [0,1]^2 (the caller has already split world-space NDC into page index + local UV --
// see shadow_sun_sampling.glsl / shadow_point_sampling.glsl). Fixed depth bias: sufficient here
// for the same reason as the pre-Phase-3 map -- every occluder is the same static Fallback Mesh
// geometry the page itself was rasterized from, no skinning/animation jitter to compensate for.
float SampleShadowPagePCF(uint physicalLayer, vec2 pageLocalUV, float currentDepthNDC) {
    const float kDepthBias = 0.0015;
    vec2 texelSize = vec2(1.0 / float(SHADOW_PAGE_TEXELS));
    float shadow = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float closestDepth = texture(g_ShadowPhysicalAtlas,
                vec3(pageLocalUV + vec2(x, y) * texelSize, float(physicalLayer))).r;
            shadow += (currentDepthNDC - kDepthBias <= closestDepth) ? 1.0 : 0.0;
        }
    }
    return shadow / 9.0;
}

#endif // SHADOW_ATLAS_SAMPLING_GLSL
