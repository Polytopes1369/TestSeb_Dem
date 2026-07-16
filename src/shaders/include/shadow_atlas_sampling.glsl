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
// see shadow_sun_sampling.glsl / shadow_point_sampling.glsl). Fixed (not slope-scaled) depth bias:
// sufficient for animation jitter (every occluder is the same static Fallback Mesh geometry the
// page itself was rasterized from, no skinning to compensate for) -- but a fixed bias is inherently
// a worse fit at grazing surface-to-light angles than head-on ones (the same NDC-space bias maps
// to a smaller effective world-space offset as the angle gets shallower), so curved surfaces (a
// continuous range of angles, unlike a flat face's single constant angle) show self-shadowing
// acne more readily. Bumped from the original 0.0015 when the sun's default direction (previously
// a steep ~64 degree elevation) became a much shallower ~45.5 degree one (Toronto, 16:30 EDT,
// see ClusterRenderPipeline::Init()'s own comment) -- that lower elevation made exactly this
// grazing-angle acne visible on curved primitives (spheres/capsules) that weren't showing it
// before. A full slope-scaled bias would need every caller (sun AND point lights, several shading
// shaders) to also thread a normal/NdotL through -- this single-constant bump is the smaller,
// lower-risk fix for the actual angle range this scene's sun now uses.
float SampleShadowPagePCF(uint physicalLayer, vec2 pageLocalUV, float currentDepthNDC) {
    const float kDepthBias = 0.004;
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
