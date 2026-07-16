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
    // One entity's Fallback Mesh, bundled with its (to-be-filled-in-place) index entry -- mirrors
    // how WriteCacheFile already takes `indexEntries` by reference and fills in
    // virtualAddress/blockSizeBytes itself; here it fills in vertexDataOffset/indexDataOffset.
    struct FallbackMeshData {
        FallbackMeshIndexEntry indexEntry; // vertexDataOffset/indexDataOffset are overwritten by WriteCacheFile.
        std::vector<FallbackVertex> vertices;
        std::vector<uint32_t> indices; // 3 per triangle.
    };

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
        // `fallbackMeshes` is likewise taken by reference: each entry's
        // indexEntry.vertexDataOffset/indexDataOffset are overwritten in place. May be empty (an
        // entity with no fallback mesh simply contributes no entry).
        //
        // Writes seven sections back-to-back, each explicitly zero-padded (via a zeroed, reused
        // scratch buffer) up to the next kPageSizeBytes boundary: CacheFileHeader, the cluster
        // index table, the DAG table, one page-aligned ClusterData block per cluster, the
        // fallback-mesh index table, the concatenated fallback-mesh vertex/index data blob, then
        // the surface-cache card table. `cardEntries` must already be atlas-packed
        // (geometry::PackCardsIntoAtlas) -- this writer persists the mapping verbatim, it never
        // recomputes placement. Returns false (and logs the OS error) on any failure.
        bool WriteCacheFile(
            const std::filesystem::path& filePath,
            std::vector<ClusterIndexEntry>& indexEntries,
            const std::vector<DAGNodeEntry>& dagEntries,
            const std::vector<ClusterData>& clusterData,
            std::vector<FallbackMeshData>& fallbackMeshes,
            const std::vector<SurfaceCacheCardEntry>& cardEntries,
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

        // Reads the full fallback-mesh index table described by an already-read `header` via a
        // plain buffered read. Returns false on any I/O failure or a size mismatch against
        // `header.fallbackMeshCount`.
        bool ReadFallbackMeshTable(
            const std::filesystem::path& filePath, const CacheFileHeader& header,
            std::vector<FallbackMeshIndexEntry>& outEntries) const;

        // Reads the full surface-cache card table described by an already-read `header` via a
        // plain buffered read (like the fallback-mesh table, this is read once at startup by the
        // surface cache capture pass, never paged on demand). Returns false on any I/O failure
        // or a size mismatch against `header.cardCount`.
        bool ReadSurfaceCacheCardTable(
            const std::filesystem::path& filePath, const CacheFileHeader& header,
            std::vector<SurfaceCacheCardEntry>& outEntries) const;

        // Reads one entity's fallback-mesh geometry (FallbackVertex[vertexCount] at
        // entry.vertexDataOffset, uint32_t[triangleCount * 3] at entry.indexDataOffset) via plain
        // buffered reads. Unlike ReadClusterDataAsync, this is synchronous and unbuffered-I/O-free:
        // the fallback-mesh table is read once at startup for BVH construction, never paged on
        // demand, so the async/unbuffered machinery built for streamed ClusterData is unwarranted
        // overhead here. Returns false on any I/O failure.
        bool ReadFallbackMeshGeometry(
            const std::filesystem::path& filePath, const FallbackMeshIndexEntry& entry,
            std::vector<FallbackVertex>& outVertices, std::vector<uint32_t>& outIndices) const;

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
