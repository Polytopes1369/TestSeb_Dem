#pragma once
// Binary on-disk format for the Virtual Texture tile cache (SVT streaming source, Step 4). A
// SEPARATE, self-contained file format from ClusterFormat.h's geometry .cache -- not an extension
// of it -- matching this codebase's established "a fresh, self-contained copy for a different
// domain" convention (see e.g. renderer::VirtualShadowMapPool.h's own header comment on why it
// re-derived its own page/LRU bookkeeping instead of repurposing geometry::GpuPageTable): a virtual
// texture tile has no cluster ID, no DAG parent/child, no fallback mesh -- only a page key
// (renderer::VirtualTextureManager::PackPageKey(x,y,mip)) and a raw texel blob per physical pool
// channel, so reusing CacheFileHeader/ClusterIndexEntry's cluster-shaped fields would only add
// unused/misleading ones.
//
// What this file caches: NOT hand-authored source art (this engine ships zero data in its .exe --
// see CLAUDE.md's project-wide "100% procedural GPU driven" constraint) but the RESULT of an
// expensive procedural bake (renderer::VirtualTextureRenderPass rendering a terrain/decal page),
// persisted to disk once so a later run can stream it back in near-instantly instead of re-running
// the GPU-driven bake for every page on every single launch -- the same "cache the expensive
// procedural result" role ClusterFormat.h's own geometry .cache file already plays for Nanite-style
// cluster simplification.
//
// Layout, two 4096-byte-aligned sections (mirrors ClusterFormat.h's own page-aligned-section
// convention, so the same AsyncFileStreamer unbuffered-I/O granularity applies identically):
//   [0]                    VirtualTextureCacheFileHeader        (exactly kVTPageSizeBytes bytes)
//   [tileIndexTableOffset] VirtualTextureTileIndexEntry[tileCount] (zero-padded to a page boundary)
//   [tileDataBaseOffset]   raw texel blob per tile, poolCount channels concatenated per tile
//                          (each tile's own blob individually page-aligned, exactly like
//                          ClusterFormat.h's ClusterData blocks -- a tile is paged in on demand by
//                          renderer::VirtualTextureStreamingCoordinator, never read as one bulk blob)
//
// Every structure below is `#pragma pack(push, 1)`-ed for the identical byte-for-byte-portable
// reason ClusterFormat.h's own structs are (see that file's header comment).

#include <cstdint>

namespace io {

    // Matches geometry::AsyncFileStreamer::kStreamerBlockSizeBytes exactly (both this format and
    // the geometry one are read through the same low-level streamer, which fixes this granularity
    // for every caller -- see AsyncFileStreamer.h's own header comment).
    constexpr uint32_t kVTPageSizeBytes = 4096u;

#pragma pack(push, 1)

    // -----------------------------------------------------------------------------------------
    // VirtualTextureCacheFileHeader -- the very first bytes of a .vtcache file. Its own on-disk
    // footprint is exactly kVTPageSizeBytes (zero-padded, mirroring CacheFileHeader's own
    // kCacheFileHeaderPaddedSizeBytes convention).
    // -----------------------------------------------------------------------------------------
    struct VirtualTextureCacheFileHeader {
        static constexpr uint32_t kMagic = 0x58545647u;   // "GVTX" little-endian ("Generated Virtual TeXture cache").
        static constexpr uint32_t kVersion = 1u;

        uint32_t magic;
        uint32_t version;

        uint32_t tileCount;         // Total cached tiles in this file.
        uint32_t poolCount;         // Physical pool channels stored per tile (e.g. 1 = Albedo only).
        uint32_t tileSizeTexels;    // Content-only tile size (renderer::VirtualTextureManager::GetTileSize()).
        uint32_t borderSizeTexels;  // Border padding (renderer::VirtualTextureManager::GetBorderSize()).

        // Tile index table: VirtualTextureTileIndexEntry[tileCount].
        uint64_t tileIndexTableOffset;
        uint64_t tileIndexTableSizeBytes; // == tileCount * sizeof(VirtualTextureTileIndexEntry), unpadded.

        // Tile texel data section: see file header comment. Individual tiles are addressed via
        // their own VirtualTextureTileIndexEntry::virtualAddress, not sequentially from this field --
        // this only marks where the section as a whole begins.
        uint64_t tileDataBaseOffset;

        uint64_t totalFileSizeBytes; // Full file size, for a reader's sanity check against fstat.
    };

    constexpr uint32_t kVTCacheFileHeaderPaddedSizeBytes = kVTPageSizeBytes;
    static_assert(sizeof(VirtualTextureCacheFileHeader) <= kVTCacheFileHeaderPaddedSizeBytes,
        "VirtualTextureCacheFileHeader must fit within one page so it can be zero-padded up to kVTPageSizeBytes");
    static_assert(sizeof(VirtualTextureCacheFileHeader) == 56,
        "VirtualTextureCacheFileHeader size drifted from the expected 56 bytes");

    // -----------------------------------------------------------------------------------------
    // VirtualTextureTileIndexEntry -- one per cached tile, in the tile index table.
    // -----------------------------------------------------------------------------------------
    struct VirtualTextureTileIndexEntry {
        // renderer::VirtualTextureManager::PackPageKey(x, y, mip) -- the SAME dense encoding a GPU
        // feedback report (virtual_texture_feedback.glsl) and geometry::StreamingRequestQueue both
        // already use, so a popped feedback request's key resolves directly to a tile via this
        // table with no separate (x,y,mip)-to-key translation needed anywhere in the streaming path.
        uint32_t pageKey;

        uint64_t virtualAddress; // Byte offset from the start of the file where this tile's texel
                                  // blob begins. Always a multiple of kVTPageSizeBytes.
        uint32_t blockSizeBytes; // Size in bytes reserved for this tile's texel blob (poolCount
                                  // channels concatenated, each (tileSizeTexels + 2*borderSizeTexels)^2
                                  // texels of the physical pool's own bytes-per-texel). Always a
                                  // multiple of kVTPageSizeBytes, stored explicitly (not assumed)
                                  // for the same future-proofing reason ClusterIndexEntry::
                                  // blockSizeBytes is -- see that field's own comment.
    };

#pragma pack(pop)

}
