#ifndef CLUSTER_SOFTWARE_RASTER_CORE_GLSL
#define CLUSTER_SOFTWARE_RASTER_CORE_GLSL

// Shared half-space rasterize + atomic VisBuffer write core for ClusterSoftwareRaster.comp (the
// masked-capable variant) and ClusterSoftwareRasterOpaque.comp (the opaque-only, zero-mask-
// overhead variant) -- factored out here to avoid duplicating ~100 lines of perspective-divide /
// screen-space / half-space-test / atomic-write logic between the two shaders, which differ only
// in whether they decode and sample the opacity mask per pixel.
//
// The includer must `#define HAS_MASK_SUPPORT 0` or `1` BEFORE including this file: with 0 (the
// opaque variant), every mask-related line below is preprocessed out entirely -- not merely
// skipped at runtime -- so that variant's compiled SPIR-V has no UV decode, no mask array
// reference, and no per-pixel mask branch at all, mirroring ClusterRasterOpaque.frag's own "no
// discard, no mask sampling" contract for the hardware path. The includer must also already have
// declared (before this include): ClusterCullMetadata (cluster_culling_common.glsl),
// DecodeClusterPosition/DecodeClusterUV (cluster_vertex_decode.glsl), EdgeFunction
// (half_space_raster.glsl), ApplyWPODeformation (wpo_deformation.glsl), SampleMaskAlpha
// (mask_sampling.glsl, only when HAS_MASK_SUPPORT == 1), and the g_ViewParams/g_WPOGlobals/
// g_VisBufferAtomic bindings this function reads/writes.

// Packs (clusterSlotIndex, localTriangleOrdinal) into the atomic word's low 32 bits. 7 bits are
// enough for CLUSTER_MAX_TRIANGLES (128), so the shift never loses triangle-ordinal bits and never
// collides with the cluster-slot bits above it.
uint PackVisibilityID(uint clusterSlotIndex, uint triangleOrdinal) {
    return (clusterSlotIndex << 7) | (triangleOrdinal & 0x7Fu);
}

// Rasterizes one cluster triangle (vertex-decode through the atomic per-pixel VisBuffer write) --
// see ClusterSoftwareRaster.comp's own class comment for the full algorithm rationale (near-plane
// rejection, screen-space bbox, half-space inside test, depth-dominant 64-bit atomic packing).
void RasterizeClusterTriangle(ClusterCullMetadata cluster, uint clusterSlotIndex, uint triangleOrdinal,
    uint pageByteBase, uint i0, uint i1, uint i2) {

    vec3 p0World = DecodeClusterPosition(pageByteBase, i0, cluster.boundsMin, cluster.boundsMax);
    vec3 p1World = DecodeClusterPosition(pageByteBase, i1, cluster.boundsMin, cluster.boundsMax);
    vec3 p2World = DecodeClusterPosition(pageByteBase, i2, cluster.boundsMin, cluster.boundsMax);

    // Apply entity self-rotation
    EntityData ed = entityData[cluster.entityID];
    EntityTransform xform = entityTransforms[ed.meshID];
    mat3 rotation = mat3(xform.rotation);
    p0World = xform.center + rotation * (p0World - xform.center);
    p1World = xform.center + rotation * (p1World - xform.center);
    p2World = xform.center + rotation * (p2World - xform.center);

#if HAS_MASK_SUPPORT
    // Only decoded when this cluster is actually masked -- sparing the opaque case (impossible in
    // this variant's own dispatch, but a masked-variant cluster can still be fully opaque) a
    // wasted decode.
    vec2 uv0, uv1, uv2;
    if (cluster.maskTextureIndex != 0xFFFFFFFFu) {
        uv0 = DecodeClusterUV(pageByteBase, i0);
        uv1 = DecodeClusterUV(pageByteBase, i1);
        uv2 = DecodeClusterUV(pageByteBase, i2);
    }
#endif

    // Applied identically to all 3 vertices, exactly like ClusterRaster.vert, so the half-space
    // test below operates on the same deformed triangle the hardware path would have rasterized.
    p0World = ApplyWPODeformation(p0World, cluster.clusterID, cluster.maxWPOAmplitude, g_WPOGlobals.globalTime);
    p1World = ApplyWPODeformation(p1World, cluster.clusterID, cluster.maxWPOAmplitude, g_WPOGlobals.globalTime);
    p2World = ApplyWPODeformation(p2World, cluster.clusterID, cluster.maxWPOAmplitude, g_WPOGlobals.globalTime);

    vec4 clip0 = g_ViewParams.viewProj * vec4(p0World, 1.0);
    vec4 clip1 = g_ViewParams.viewProj * vec4(p1World, 1.0);
    vec4 clip2 = g_ViewParams.viewProj * vec4(p2World, 1.0);

    // Reject any triangle with a vertex behind (or on) the camera plane -- see
    // ClusterSoftwareRaster.comp's class comment for why proper near-plane clipping is skipped at
    // this scale.
    if (clip0.w <= 1.0e-5 || clip1.w <= 1.0e-5 || clip2.w <= 1.0e-5) {
        return;
    }

    vec3 ndc0 = clip0.xyz / clip0.w;
    vec3 ndc1 = clip1.xyz / clip1.w;
    vec3 ndc2 = clip2.xyz / clip2.w;

    // NDC [-1, 1] xy -> pixel screen-space coordinates. Vulkan clip space is already y-down, so no
    // extra flip is needed.
    vec2 screen0 = (ndc0.xy * 0.5 + 0.5) * g_ViewParams.viewportSize;
    vec2 screen1 = (ndc1.xy * 0.5 + 0.5) * g_ViewParams.viewportSize;
    vec2 screen2 = (ndc2.xy * 0.5 + 0.5) * g_ViewParams.viewportSize;

    vec2 bboxMin = clamp(min(screen0, min(screen1, screen2)), vec2(0.0), g_ViewParams.viewportSize - 1.0);
    vec2 bboxMax = clamp(max(screen0, max(screen1, screen2)), vec2(0.0), g_ViewParams.viewportSize - 1.0);
    ivec2 pixelMin = ivec2(floor(bboxMin));
    ivec2 pixelMax = ivec2(ceil(bboxMax));

    // area2 == EdgeFunction(screen0, screen1, screen2) is the triangle's own signed doubled area:
    // zero for a degenerate (zero-area) triangle, its sign tells the winding used to normalize the
    // per-pixel inside test below.
    float area2 = EdgeFunction(screen0, screen1, screen2);
    if (abs(area2) < 1.0e-8) {
        return;
    }
    float invArea2 = 1.0 / area2;

    for (int y = pixelMin.y; y <= pixelMax.y; ++y) {
        for (int x = pixelMin.x; x <= pixelMax.x; ++x) {
            vec2 pixelCenter = vec2(float(x) + 0.5, float(y) + 0.5);

            float w0 = EdgeFunction(screen1, screen2, pixelCenter);
            float w1 = EdgeFunction(screen2, screen0, pixelCenter);
            float w2 = EdgeFunction(screen0, screen1, pixelCenter);

            // Top-left tie-break (half_space_raster.glsl's IsTopLeftEdge -- see its own comment for
            // why this is required, not optional, to avoid double-covering a shared triangle edge):
            // a strictly-inside pixel (w != 0) always passes on the correct side regardless of
            // area2's sign; only a pixel EXACTLY on an edge (w == 0) additionally needs the
            // tie-break, evaluated with the edge in the same vertex order EdgeFunction itself used
            // for that edge (reversed for the area2 < 0 / clockwise-in-screen-space case, since the
            // "inside" half-plane itself is mirrored there).
            bool c0, c1, c2;
            if (area2 > 0.0) {
                c0 = (w0 > 0.0) || (w0 == 0.0 && IsTopLeftEdge(screen1, screen2));
                c1 = (w1 > 0.0) || (w1 == 0.0 && IsTopLeftEdge(screen2, screen0));
                c2 = (w2 > 0.0) || (w2 == 0.0 && IsTopLeftEdge(screen0, screen1));
            } else {
                c0 = (w0 < 0.0) || (w0 == 0.0 && IsTopLeftEdge(screen2, screen1));
                c1 = (w1 < 0.0) || (w1 == 0.0 && IsTopLeftEdge(screen0, screen2));
                c2 = (w2 < 0.0) || (w2 == 0.0 && IsTopLeftEdge(screen1, screen0));
            }
            bool inside = c0 && c1 && c2;
            if (!inside) {
                continue;
            }

            vec3 bary = vec3(w0, w1, w2) * invArea2;

#if HAS_MASK_SUPPORT
            // Opacity-mask cutout: skip this pixel's atomic write entirely when masked out (this
            // shader's equivalent of ClusterRaster.frag's `discard`) -- screen-space-linear
            // barycentric UV, matching this rasterizer's own existing (non-perspective-corrected)
            // depth interpolation precision immediately below.
            if (cluster.maskTextureIndex != 0xFFFFFFFFu) {
                vec2 uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
                if (SampleMaskAlpha(cluster.maskTextureIndex, uv) < 0.5) {
                    continue;
                }
            }
#endif

            float ndcDepth = bary.x * ndc0.z + bary.y * ndc1.z + bary.z * ndc2.z;

            // Reversed-Z (see maths::mat4::PerspectiveVulkan's own comment): larger ndcDepth is now
            // nearer, so it packs DIRECTLY (no more "1.0 - ndcDepth" inversion) into the atomic
            // word's high bits -- imageAtomicMax below still means "keep whichever write is
            // nearer," just without the extra subtraction the old [0,1]-non-reversed convention
            // needed. min(...) guards against the float literal 4294967295.0 (2^32-1) not being
            // exactly representable in float32 -- it rounds up to 2^32, and clamp(...,0,1)==1.0
            // would otherwise convert to a uint value one past UINT32_MAX (undefined behavior per
            // the GLSL spec) for any ndcDepth close enough to 1.0 (i.e. very close to the camera).
            uint depthBits = min(uint(clamp(ndcDepth, 0.0, 1.0) * 4294967295.0), 0xFFFFFFFEu);
            uint visibilityID = PackVisibilityID(clusterSlotIndex, triangleOrdinal);
            uint64_t packed = (uint64_t(depthBits) << 32) | uint64_t(visibilityID);

            imageAtomicMax(g_VisBufferAtomic, ivec2(x, y), packed);
        }
    }
}

#endif // CLUSTER_SOFTWARE_RASTER_CORE_GLSL
