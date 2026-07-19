#version 460
#extension GL_GOOGLE_include_directive : enable

// Feature F7 (shadow-casting particles): alpha-tested depth-only capture -- mirrors
// ShadowMapCaptureMasked.frag's own "discard before the depth write, no color attachment" contract
// exactly (see that file's own header comment: renderer::VirtualShadowMapPass's capture pipelines
// all have 0 color attachments), testing a particle's analytic soft-circle footprint (this project
// has zero on-disk texture assets, see ParticleRender.frag's own header comment on why every
// particle shape in this codebase is generated analytically instead of sampled from a sprite atlas)
// -- simplified here to a single hard cutoff radius rather than ParticleRender.frag's own
// smoothstep-blended edge, since no blending happens in a depth-only pass (a fragment either passes
// the alpha test and writes real depth, or is discarded outright).

layout(location = 0) in vec2 inUV;

const float kAlphaCutoffRadius = 0.5; // UV-space distance from the quad's own center beyond which a particle's footprint is fully cut out.

void main() {
    vec2 centered = inUV - vec2(0.5);
    if (length(centered) > kAlphaCutoffRadius) {
        discard;
    }
}
