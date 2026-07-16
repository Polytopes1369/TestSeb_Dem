#pragma once
// Disk I/O layer for the virtual geometry .cache format (see ClusterFormat.h). Isolated from the
// rest of the engine: callers only ever hand this class fully-built header/table/geometry data
// and a filesystem path, never raw OS handles.

#include <cstdint>
#include <filesystem>
#include <future>
#include <vector>
#include "geometry/ClusterFormat.h"

namespace geometry {

    // Manages the single consolidated .cache file on disk: purges a stale file at startup,
    // writes the whole scene's clusters (header + cluster index table + DAG table + geometry
    // blocks, each major section explicitly zero-padded up to the next kPageSizeBytes boundary)
    // in one call, and streams individual cluster geometry blocks back in asynchronously via
    // unbuffered (Windows FILE_FLAG_NO_BUFFERING) I/O -- matching how a real virtual-geometry
    // streaming system would page cluster data in without going through the OS page cache twice.
    class CacheFileManager {
    public:
        CacheFileManager() = default;
        ~CacheFileManager() = default;

        CacheFileManager(const CacheFileManager&) = delete;
        CacheFileManager& operator=(const CacheFileManager&) = delete;

        // Deletes every "*.cache" file found directly inside `directory`. Call once at startup so
        // re-running the engine never mixes clusters from a previous, possibly stale, build.
        void PurgeExistingCacheFiles(const std::filesystem::path& directory = ".");

        // Writes one complete, consolidated .cache file for the whole scene.
        //
        // `indexEntries`, `dagEntries` and `clusterData` must all have the same size (one entry
        // per cluster) and be index-aligned: entry i of each vector describes the same cluster.
        // `indexEntries` is taken by reference and its `virtualAddress`/`blockSizeBytes` fields
        // are overwritten in place by this call -- the writer, not the caller, owns the physical
        // on-disk layout decision, so callers may pass them zero-initialized and read the real,
        // final values back out of their own vector afterwards (e.g. to call
        // ReadClusterDataAsync immediately without first re-reading the table from disk).
        //
        // Writes four sections back-to-back, each explicitly zero-padded (via a zeroed, reused
        // scratch buffer) up to the next kPageSizeBytes boundary: CacheFileHeader, the cluster
        // index table, the DAG table, then one page-aligned ClusterData block per cluster.
        // Returns false (and logs the OS error) on any failure.
        bool WriteCacheFile(
            const std::filesystem::path& filePath,
            std::vector<ClusterIndexEntry>& indexEntries,
            const std::vector<DAGNodeEntry>& dagEntries,
            const std::vector<ClusterData>& clusterData,
            uint32_t entityCount) const;

        // Reads just the CacheFileHeader (the file's first kPageSizeBytes bytes) via a plain
        // buffered read and validates its magic/version. Returns false on any I/O failure or a
        // magic/version mismatch.
        bool ReadHeader(const std::filesystem::path& filePath, CacheFileHeader& outHeader) const;

        // Reads the full cluster index table / DAG table described by an already-read `header`
        // via a plain buffered read. Returns false on any I/O failure or a size mismatch against
        // `header.clusterCount`.
        bool ReadClusterIndexTable(
            const std::filesystem::path& filePath, const CacheFileHeader& header,
            std::vector<ClusterIndexEntry>& outEntries) const;
        bool ReadDAGTable(
            const std::filesystem::path& filePath, const CacheFileHeader& header,
            std::vector<DAGNodeEntry>& outEntries) const;

        // Issues an unbuffered, overlapped (asynchronous) read of one cluster's ClusterData block
        // at `virtualAddress` (an offset taken from that cluster's ClusterIndexEntry, always a
        // multiple of kPageSizeBytes) from `filePath` into `outData`. The full kPageSizeBytes
        // page is read into an internally-owned, 4096-byte-aligned scratch buffer (required by
        // FILE_FLAG_NO_BUFFERING) that is kept alive until the read completes, then only the
        // leading sizeof(ClusterData) bytes are copied into `outData`. The future resolves to
        // true once the OS completes the I/O and the copy has been performed.
        std::future<bool> ReadClusterDataAsync(
            const std::filesystem::path& filePath, uint64_t virtualAddress, ClusterData& outData) const;
    };

}
