#pragma once
// Disk I/O layer for the Virtual Texture tile cache format (see VirtualTextureCacheFormat.h).
// Mirrors geometry::CacheFileManager's shape (PurgeExistingCacheFiles/WriteCacheFile/ReadHeader/
// Read*Table) but trimmed to exactly this format's two sections -- no cluster/DAG/fallback-mesh/
// surface-cache-card concepts apply here (see VirtualTextureCacheFormat.h's own header comment on
// why this is a fresh, self-contained format rather than an extension of ClusterFormat.h's).
//
// Like geometry::CacheFileManager, this class only ever hands callers fully-built header/table
// data and a filesystem path -- never raw OS handles -- and the actual per-tile async streaming
// read (once a tile's VirtualTextureTileIndexEntry::virtualAddress is known) goes directly through
// geometry::AsyncFileStreamer, exactly like renderer::GeometryStreamingCoordinator talks to
// AsyncFileStreamer directly rather than through geometry::CacheFileManager::ReadClusterDataAsync's
// higher-level std::future wrapper (see that method's own comment on why the coordinator bypasses
// it) -- renderer::VirtualTextureStreamingCoordinator does the identical thing here.

#include <cstdint>
#include <filesystem>
#include <vector>
#include "io/VirtualTextureCacheFormat.h"

namespace io {

    // One tile's full data as the writer receives it: the pool channels concatenated in physical-
    // pool order (index 0 = renderer::VirtualTextureManager::GetPhysicalPoolFormat(0), etc.), each
    // exactly (tileSizeTexels + 2*borderSizeTexels)^2 * bytesPerTexel bytes.
    struct VirtualTextureTileData {
        uint32_t pageKey; // VirtualTextureManager::PackPageKey(x, y, mip).
        std::vector<uint8_t> texelBytes; // Concatenated pool channels, see struct comment above.
    };

    class VirtualTextureCacheFileManager {
    public:
        VirtualTextureCacheFileManager() = default;
        ~VirtualTextureCacheFileManager() = default;

        VirtualTextureCacheFileManager(const VirtualTextureCacheFileManager&) = delete;
        VirtualTextureCacheFileManager& operator=(const VirtualTextureCacheFileManager&) = delete;

        // Deletes every "*.vtcache" file found directly inside `directory` -- same startup-hygiene
        // role as geometry::CacheFileManager::PurgeExistingCacheFiles, kept as a SEPARATE call (not
        // folded into that one) so a caller can purge the geometry cache and the VT tile cache on
        // independent schedules (e.g. only rebuilding VT tiles when the procedural material graph
        // changes, without forcing a full geometry re-simplification).
        void PurgeExistingCacheFiles(const std::filesystem::path& directory = ".") const;

        // Writes one complete .vtcache file: header + tile index table (each entry's
        // virtualAddress/blockSizeBytes overwritten in place by this call, exactly mirroring
        // geometry::CacheFileManager::WriteCacheFile's own "the writer owns physical layout, not
        // the caller" contract) + the concatenated, individually page-aligned tile texel blobs.
        // Returns false (and logs the OS error) on any failure.
        bool WriteCacheFile(
            const std::filesystem::path& filePath,
            std::vector<VirtualTextureTileIndexEntry>& indexEntries,
            std::vector<VirtualTextureTileData>& tileData,
            uint32_t poolCount, uint32_t tileSizeTexels, uint32_t borderSizeTexels) const;

        // Reads just the VirtualTextureCacheFileHeader (the file's first kVTPageSizeBytes bytes)
        // via a plain buffered read and validates its magic/version. Returns false on any I/O
        // failure or a magic/version mismatch.
        bool ReadHeader(const std::filesystem::path& filePath, VirtualTextureCacheFileHeader& outHeader) const;

        // Reads the full tile index table described by an already-read `header` via a plain
        // buffered read. Returns false on any I/O failure or a size mismatch against
        // `header.tileCount`.
        bool ReadTileIndexTable(
            const std::filesystem::path& filePath, const VirtualTextureCacheFileHeader& header,
            std::vector<VirtualTextureTileIndexEntry>& outEntries) const;
    };

}
