#ifndef VISBUFFER_ARBITRATION_GLSL
#define VISBUFFER_ARBITRATION_GLSL

// Step 1 of the Visibility Buffer resolve: reconciles renderer::ClusterHardwareRasterPass's
// (ClusterID slot + local TriangleID R32_UINT attachments, plus an ordinary hardware depth
// buffer) output against renderer::ClusterSoftwareRasterPass's (one packed 64-bit atomic
// (depth, visibilityID) word) output per pixel. Since these are two independent rasterizers with
// no shared depth buffer between them (there is no cross-path atomic seeding -- seeding a plain
// hardware depth buffer into an atomic buffer intended for lock-free contention is unnecessary
// complexity a plain per-pixel comparison avoids entirely), this is what actually reconciles
// them: whichever has the nearer decoded depth at a given pixel wins, exactly matching how real
// dual-rasterization engines (Nanite) defer this exact decision to their own resolve/material
// pass.
//
// Shared verbatim between ClusterResolve.comp (the full-screen, debug-view-capable resolve path)
// and ClusterShadingBinClassify.comp (the Phase 1b binned path's own classification pass) so the
// depth-arbitration epsilon and the software path's bit-packing convention can never silently
// drift apart between the two -- both consumers must bind the exact same 4 resources at these
// fixed binding numbers (set 0, bindings 2-5).
layout(r32ui, set = 0, binding = 2) uniform readonly uimage2D g_HWClusterIDImage;
layout(r32ui, set = 0, binding = 3) uniform readonly uimage2D g_HWTriangleIDImage;
layout(set = 0, binding = 4) uniform sampler2D g_HWDepthTexture;
layout(r64ui, set = 0, binding = 5) uniform readonly u64image2D g_SWVisBufferAtomic;

// Result of arbitrating hw-vs-sw visibility at one pixel -- hwDepth/swDepth are both returned
// (not just the winning one) since some callers' debug views need both regardless of who won
// (e.g. DEBUG_VIEW_OVERDRAWS, DEBUG_VIEW_DEPTH in ClusterResolve.comp).
struct VisibilityArbitration {
    bool hasAnyData;
    bool hwWins;
    uint clusterSlotIndex;
    uint triangleOrdinal;
    float hwDepth;
    float swDepth;
};

VisibilityArbitration ArbitrateVisibility(ivec2 pixel) {
    VisibilityArbitration result;

    uint hwClusterSlot = imageLoad(g_HWClusterIDImage, pixel).r;
    uint hwTriangleOrdinal = imageLoad(g_HWTriangleIDImage, pixel).r;
    result.hwDepth = texelFetch(g_HWDepthTexture, pixel, 0).r; // NDC [0,1], reversed-Z -- LARGER is nearer.
    bool hwHasData = hwClusterSlot != 0xFFFFFFFFu; // Matches main.cpp's VisBuffer clear value (kInvalidClusterID).

    uint64_t swPacked = imageLoad(g_SWVisBufferAtomic, pixel).x;
    bool swHasData = swPacked != 0uL; // Matches ClearVisBufferAtomic.comp's clear value.
    uint swDepthBits = uint(swPacked >> 32);
    uint swVisibilityID = uint(swPacked & 0xFFFFFFFFuL);
    // Direct decode, reversed-Z (see cluster_software_raster_core.glsl's own encoding comment --
    // no more "1.0 -" inversion, larger swDepthBits already means nearer).
    result.swDepth = float(swDepthBits) / 4294967295.0;

    // Epsilon-tolerant, single source of truth for "which path wins," reused by every consumer
    // (arbitration, GBuffer depth capture, and the OVERDRAWS/DEPTH debug views) so they can never
    // disagree with each other. The epsilon absorbs the residual hw-vs-sw depth disagreement
    // documented in ClusterSoftwareRasterPass's own class comment (independent rasterizers, no
    // shared depth buffer, plus per-cluster position requantization at cluster boundaries)
    // instead of a zero-tolerance `>=` compare picking a winner based on sub-ULP floating-point
    // noise that shifts every frame as the camera moves -- a real, if partial, mitigation for the
    // z-fighting this codebase's own bug reports describe at hw/sw cluster-routing boundaries.
    const float kDepthArbitrationEpsilon = 1.0e-6;
    result.hwWins = hwHasData && (!swHasData || result.hwDepth >= result.swDepth - kDepthArbitrationEpsilon);
    result.hasAnyData = hwHasData || swHasData;

    if (result.hwWins) {
        result.clusterSlotIndex = hwClusterSlot;
        result.triangleOrdinal = hwTriangleOrdinal;
    } else {
        // swHasData is guaranteed true here (hasAnyData covers the all-false case the caller
        // handles itself).
        result.clusterSlotIndex = swVisibilityID >> 7;
        result.triangleOrdinal = swVisibilityID & 0x7Fu;
    }

    return result;
}

#endif // VISBUFFER_ARBITRATION_GLSL
