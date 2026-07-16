#version 460
#extension GL_GOOGLE_include_directive : enable

// Hardware rasterization fragment shader for "large" clusters (see
// renderer::ClusterHardwareRasterPass / ClusterRaster.vert). Writes the Visibility Buffer only --
// no lighting, no albedo, no standard color output -- exactly like draw.frag's own VisBuffer
// write, reusing the same 2x R32_UINT attachment convention (ClusterID + local TriangleID) so both
// the flat legacy draw path and this clustered hardware path can share one VisBuffer target.
//
// gl_PrimitiveID supplies the local TriangleID directly: the Vulkan spec resets primitive
// numbering to 0 at the start of each indirect sub-draw (i.e. each cluster's own
// VkDrawIndexedIndirectCommand within the vkCmdDrawIndexedIndirectCount batch), so it is exactly
// this cluster's own triangle ordinal in [0, indexCount / 3) with no manual computation needed in
// either shader stage.
//
// Opacity-mask cutout: this is this shader's first-ever descriptor (the bindless mask array,
// mask_sampling.glsl) -- a masked-out fragment `discard`s before writing either VisBuffer output,
// so a cut-out pixel never occupies the Visibility Buffer or depth at all (correct occlusion for
// whatever real geometry may lie behind it), unlike a hypothetical post-hoc resolve-time-only
// cutout which would leave the depth/VisBuffer wrongly claimed by discarded geometry.

#define MASK_ARRAY_SET 0
#define MASK_ARRAY_BINDING 3
#include "include/mask_sampling.glsl"

layout(location = 0) flat in uint inClusterID;
layout(location = 1) in vec2 inUV;
layout(location = 2) flat in uint inMaskTextureIndex;

layout(location = 0) out uint outClusterID;
layout(location = 1) out uint outTriangleID;

// Below this alpha, a masked fragment is treated as fully cut out.
const float kMaskAlphaCutoff = 0.5;

void main() {
    if (SampleMaskAlpha(inMaskTextureIndex, inUV) < kMaskAlphaCutoff) {
        discard;
    }

    outClusterID = inClusterID;
    outTriangleID = uint(gl_PrimitiveID);
}
