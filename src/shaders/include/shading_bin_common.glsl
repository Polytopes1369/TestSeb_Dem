#ifndef SHADING_BIN_COMMON_GLSL
#define SHADING_BIN_COMMON_GLSL

// Shared data layout for the Phase 1b shading-bin pipeline (renderer::ClusterShadingBinPass +
// ClusterResolvePass::RecordResolveBinned): a 3-stage GPU counting sort -- histogram (Classify) ->
// exclusive prefix-sum -> scatter -- that buckets VisBuffer pixels by materialID into ONE shared
// buffer sized exactly viewportWidth*viewportHeight (not one fixed-capacity list per bin, which
// would waste hundreds of MB at typical resolutions), so ClusterResolveBinned.comp can dispatch
// once per bin (indirect, `binIndex` a push constant) with every thread in that dispatch reading
// the SAME material -- genuine warp coherence.

#include "material_params.glsl" // MATERIAL_TABLE_SIZE -- one shading bin per material slot, no separate bin-count constant needed.

// Sentinel PixelScratchEntry::materialSlot meaning "background pixel, no geometry, not part of any
// bin" -- matches this codebase's existing 0xFFFFFFFF sentinel convention (geometry::
// kInvalidClusterID, geometry::kInvalidMaskTextureIndex).
#define SHADING_BIN_BACKGROUND_SENTINEL 0xFFFFFFFFu

// Local reimplementation of cluster_software_raster_core.glsl's PackVisibilityID/unpack formula
// (NOT included directly -- that file requires HAS_MASK_SUPPORT and several unrelated bindings to
// already be declared, which none of this pipeline's shaders have). Must stay byte-for-byte in
// sync with that file's PackVisibilityID (clusterSlotIndex << 7 | triangleOrdinal & 0x7Fu) and with
// ClusterResolve.comp's own unpack (>> 7 / & 0x7Fu) -- 7 bits is exactly CLUSTER_MAX_TRIANGLES (128,
// see geometry::ClusterFormat.h::kMaxClusterTriangles).
uint PackVisibilityIDForBin(uint clusterSlotIndex, uint triangleOrdinal) {
    return (clusterSlotIndex << 7) | (triangleOrdinal & 0x7Fu);
}

void UnpackVisibilityIDForBin(uint packed, out uint clusterSlotIndex, out uint triangleOrdinal) {
    clusterSlotIndex = packed >> 7;
    triangleOrdinal = packed & 0x7Fu;
}

// Written once per pixel by ClusterShadingBinClassify.comp, indexed implicitly by the pixel's own
// linear index (pixel.y * viewportWidth + pixel.x) -- read back by ClusterShadingBinScatter.comp.
// winningDepth carries stage A's own hw-vs-sw ArbitrateVisibility() result forward so stage D never
// needs to re-read the VisBuffer/depth images at all -- visibility is fully resolved by stage A.
struct PixelScratchEntry {
    uint materialSlot; // SHADING_BIN_BACKGROUND_SENTINEL for a no-geometry pixel.
    uint packedVisID;  // PackVisibilityIDForBin(clusterSlotIndex, triangleOrdinal).
    float winningDepth; // ArbitrateVisibility()'s hwWins ? hwDepth : swDepth -- unused for background pixels.
};

// Written by ClusterShadingBinScatter.comp into g_SortedPixelList[binOffset + withinBinIndex] --
// read back by ClusterResolveBinned.comp, one indirect dispatch per bin.
struct SortedPixelEntry {
    uint pixelLinearIndex; // pixel.y * viewportWidth + pixel.x -- this entry's own screen position.
    uint packedVisID;      // Same encoding as PixelScratchEntry::packedVisID.
    float winningDepth;    // Forwarded verbatim from PixelScratchEntry::winningDepth.
};

// Byte-for-byte mirror of VkDispatchIndirectCommand (3x uint32) -- see
// BuildDispatchIndirectArgs.comp's identical mirror for the established precedent this follows.
struct BinDispatchArgs {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

// Must match ClusterResolveBinned.comp's own local_size_x exactly -- ClusterShadingBinPrefixSum.comp
// uses this same constant to size each bin's groupCountX (ceil(binCount / this)).
#define SHADING_BIN_RESOLVE_WORKGROUP_SIZE 64u

#endif // SHADING_BIN_COMMON_GLSL
