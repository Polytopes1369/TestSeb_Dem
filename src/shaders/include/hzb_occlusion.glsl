#ifndef HZB_OCCLUSION_GLSL
#define HZB_OCCLUSION_GLSL

// GPU-side Hierarchical-Z occlusion test against the pyramid built by renderer::HZBPass
// (HZBBuildInit.comp / HZBReduce.comp): format R32G32_SFLOAT, R = per-texel min depth, G = per-
// texel max depth, each mip a 2x2 min/max reduction of the mip below it.
//
// The includer must #define HZB_TEXTURE_SET / HZB_TEXTURE_BINDING to the descriptor set/binding
// renderer::HZBPass::GetFullView() (sampled with a dedicated nearest/nearest-mipmap sampler -- see
// renderer::ClusterOcclusionCullingPass, which owns that sampler) is bound to before including
// this header.

layout(set = HZB_TEXTURE_SET, binding = HZB_TEXTURE_BINDING) uniform sampler2D g_HZBTexture;

// Additional per-frame parameters needed to project a cluster's world-space AABB into the HZB's
// screen space and pick the correct mip level, layered on top of CullingViewParams
// (cluster_culling_common.glsl), which already carries the frustum planes + camera position
// consumed by cluster_culling_tests.glsl.
struct HZBOcclusionViewParams {
    mat4 viewProj;      // proj * view, matching ExtractFrustumPlanes' expected combined matrix.
    vec2 hzbMip0Size;   // Texel dimensions of the HZB pyramid's mip 0 (renderer::HZBPass::GetMipExtent(0)).
    float hzbMipCount;  // renderer::HZBPass::GetMipLevelCount(), as a float for the log2 mip-select math below.
    // 1 / tan(fovYRadians * 0.5) -- i.e. abs(proj[1][1]) of the camera's own projection matrix
    // (maths::mat4::PerspectiveVulkan's `g` term) -- used by ProjectedScreenRadius below for the
    // hardware-vs-software rasterization size classification (see ClusterHZBOcclusionCull.comp).
    // Not derivable from viewProj alone (that's already the *combined* view*proj matrix), so the
    // caller (renderer::ClusterOcclusionCullingPass::RecordEarlyPass) supplies it directly from
    // the camera's projection matrix.
    float projScaleY;
};

float SampleHZBMaxDepth(vec2 uv, float mipLevel) {
    return textureLod(g_HZBTexture, clamp(uv, 0.0, 1.0), mipLevel).g;
}

// Conservative HZB occlusion test. Projects the cluster's world-space AABB's 8 corners into NDC,
// derives its screen-space bounding rectangle and nearest (minimum) NDC depth, picks the coarsest
// HZB mip whose texel footprint still fully covers that rectangle, and compares the cluster's
// nearest depth against the farthest already-drawn depth recorded there.
//
// Why the *max* channel is the correct conservative bound: a mip texel's G channel is the max
// (farthest) depth among every full-resolution sample that was reduced into it, so by
// construction every already-rasterized pixel within that texel's footprint is at a depth <= that
// stored max -- i.e. at least as near as it. If the cluster's own *nearest* possible point
// (nearestDepth, the minimum over its 8 projected corners) is still farther away than that stored
// max, then every already-drawn pixel across the whole footprint is nearer than every point the
// cluster could possibly occupy there, so the cluster cannot be visible anywhere in that
// footprint and is safe to cull. This is the exact inverse of the min-channel test that would be
// used to answer "is anything definitely NOT occluded" -- occlusion culling only ever needs this
// max-channel, nearest-corner comparison.
//
// Mip selection rounds UP (ceil) so a single mip texel's reduced footprint is guaranteed to be at
// least as large as the cluster's screen rectangle; the 4 corner taps below then guard against the
// rectangle straddling a texel boundary at that mip (a single center sample could otherwise miss
// part of the footprint), each contributing its own max-depth value combined with max() -- still a
// valid upper bound on "farthest already-drawn depth across the whole footprint" as required by
// the derivation above.
bool IsClusterOccluded(vec3 boundsMin, vec3 boundsMax, mat4 viewProj, vec2 hzbMip0Size, float hzbMipCount) {
    vec2 ndcMin = vec2(1.0, 1.0);
    vec2 ndcMax = vec2(-1.0, -1.0);
    float nearestDepth = 1.0;

    for (int i = 0; i < 8; ++i) {
        vec3 corner = vec3(
            (i & 1) != 0 ? boundsMax.x : boundsMin.x,
            (i & 2) != 0 ? boundsMax.y : boundsMin.y,
            (i & 4) != 0 ? boundsMax.z : boundsMin.z);

        vec4 clip = viewProj * vec4(corner, 1.0);
        if (clip.w <= 1.0e-5) {
            // Corner behind (or on) the camera plane: the box straddles the near plane, so its
            // true screen-space footprint cannot be bounded by a simple NDC min/max over the 8
            // corners. Conservatively treat the whole cluster as visible (never claim occlusion)
            // rather than produce a meaningless rectangle -- the frustum test upstream already
            // handles rejecting boxes that are genuinely entirely behind the camera.
            return false;
        }

        vec3 ndc = clip.xyz / clip.w;
        ndcMin = min(ndcMin, ndc.xy);
        ndcMax = max(ndcMax, ndc.xy);
        nearestDepth = min(nearestDepth, ndc.z);
    }

    // Vulkan clip space is already y-down, matching texel-space directly (unlike OpenGL's
    // y-up NDC) -- NDC [-1, 1] xy maps straight to [0, 1] screen-space UV with no extra flip.
    vec2 uvMin = clamp(ndcMin * 0.5 + 0.5, 0.0, 1.0);
    vec2 uvMax = clamp(ndcMax * 0.5 + 0.5, 0.0, 1.0);

    vec2 footprintTexels = (uvMax - uvMin) * hzbMip0Size;
    float mipLevel = clamp(ceil(log2(max(footprintTexels.x, footprintTexels.y))), 0.0, hzbMipCount - 1.0);

    float storedMaxDepth = max(
        max(SampleHZBMaxDepth(uvMin, mipLevel), SampleHZBMaxDepth(vec2(uvMax.x, uvMin.y), mipLevel)),
        max(SampleHZBMaxDepth(vec2(uvMin.x, uvMax.y), mipLevel), SampleHZBMaxDepth(uvMax, mipLevel)));

    return nearestDepth > storedMaxDepth;
}

// Approximate screen-space radius, in pixels, of a bounding sphere -- used by
// ClusterHZBOcclusionCull.comp to classify each surviving cluster as hardware- or software-
// rasterizable (see renderer::ClusterHardwareRasterPass / renderer::ClusterSoftwareRasterPass).
// Standard perspective-projection size estimate: projScaleY * (viewportHeightPixels * 0.5) is the
// number of screen pixels one world-space unit subtends at one unit of view-space depth, so
// dividing by the actual distance to the sphere gives pixels-per-world-unit at that depth, times
// sphereRadius for the projected screen radius.
//
// Uses the Euclidean distance from the camera to the sphere center rather than strict view-space
// Z depth (which would require the raw view matrix -- not available here, only the combined
// viewProj): this is an approximation that overestimates projected size for spheres far off the
// view axis (near the frustum edges) and is exact along the view axis, entirely adequate for a
// coarse hardware/software rasterization classification threshold -- not a precision-critical
// occlusion or LOD-selection calculation.
float ProjectedScreenRadius(vec3 sphereCenter, float sphereRadius, vec3 cameraPositionWorld, float projScaleY, float viewportHeightPixels) {
    float dist = max(length(sphereCenter - cameraPositionWorld), 1.0e-4);
    return sphereRadius * projScaleY * (viewportHeightPixels * 0.5) / dist;
}

#endif // HZB_OCCLUSION_GLSL
