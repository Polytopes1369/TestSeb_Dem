#pragma once
// Disk I/O layer for the virtual geometry .cache format (see ClusterFormat.h). Isolated from
// the rest of the engine: callers only ever hand this class a fully-built CacheHeader/Page set
// and a filesystem path, never raw OS handles.

#include <cstdint>
#include <filesystem>
#include <future>
#include <vector>
#include "geometry/ClusterFormat.h"

namespace geometry {

    // Manages .cache files on disk: purges stale files at startup, writes a full entity cache
    // using a 4KB-aligned RAM staging buffer, and streams individual 4KB pages back in
    // asynchronously via unbuffered (Windows FILE_FLAG_NO_BUFFERING) I/O — matching how a real
    // virtual-geometry streaming system would page cluster data in without going through the OS
    // page cache twice.
    class CacheFileManager {
    public:
        CacheFileManager() = default;
        ~CacheFileManager() = default;

        CacheFileManager(const CacheFileManager&) = delete;
        CacheFileManager& operator=(const CacheFileManager&) = delete;

        // Deletes every "*.cache" file found directly inside `directory`. Call once at startup
        // so re-running the engine never mixes pages from a previous, possibly stale, build.
        void PurgeExistingCacheFiles(const std::filesystem::path& directory = ".");

        // Writes one complete entity cache file: the CacheHeader followed by pages.size() Page
        // blocks, back-to-back, each written from a single reused 4KB-aligned (VirtualAlloc) RAM
        // buffer. Returns false (and logs the OS error) on any failure.
        bool WriteCacheFile(const std::filesystem::path& filePath,
            const CacheHeader& header,
            const std::vector<Page>& pages) const;

        // Issues an unbuffered, overlapped (asynchronous) read of a single 4KB page at
        // `pageIndex` (0-based, counted after the CacheHeader) from `filePath` into `outPage`.
        //
        // `outPage` must be 4096-byte aligned in memory — guaranteed automatically since Page is
        // declared alignas(4096) — and must stay alive until the returned future is resolved:
        // FILE_FLAG_NO_BUFFERING requires the destination buffer address, file offset, and read
        // length to all be multiples of the volume's sector size, which a page-aligned Page
        // always satisfies. The future resolves to true once the OS completes the I/O.
        std::future<bool> ReadPageAsync(const std::filesystem::path& filePath,
            uint32_t pageIndex,
            Page& outPage) const;
    };

}
