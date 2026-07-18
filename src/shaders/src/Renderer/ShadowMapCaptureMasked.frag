#version 460
#extension GL_GOOGLE_include_directive : enable

// VSM advanced roadmap, Feature 3 (transparency mask support): opacity-mask cutout for VSM shadow
// capture -- mirrors ClusterRaster.frag's own opacity-mask discard exactly (same mask_sampling.glsl
// helper, same 0.5 cutoff), adapted to a depth-only render target: renderer::VirtualShadowMapPass's
// two capture pipelines both have 0 color attachments (see that pass' own Init()), so this shader
// has no VisBuffer/color output at all -- it exists purely to `discard` a masked-out shadow-caster
// pixel BEFORE the depth write happens, exactly like ClusterRaster.frag's own documented
// consequence (a cut-out pixel never occupies depth, so real geometry behind it is correctly NOT
// occluded by it).
//
// Paired with ShadowMapCaptureAnimated.vert -- see that file's own header comment for the full
// per-entity capture design this belongs to. renderer::VirtualShadowMapPass::RenderPage() only
// binds the pipeline built from this shader for an entity whose precomputed maskTextureIndex is
// not geometry::kInvalidMaskTextureIndex; every other entity uses the vertex-only unmasked
// pipeline, which has no `discard` anywhere and stays early-Z-eligible.

#define MASK_ARRAY_SET 0
#define MASK_ARRAY_BINDING 4
#include "include/mask_sampling.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 1) flat in uint inMaskTextureIndex;

// Below this alpha, a masked fragment is treated as fully cut out -- same convention as
// ClusterRaster.frag's own kMaskAlphaCutoff.
const float kMaskAlphaCutoff = 0.5;

void main() {
    if (SampleMaskAlpha(inMaskTextureIndex, inUV) < kMaskAlphaCutoff) {
        discard;
    }
}
