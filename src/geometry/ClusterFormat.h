#pragma once
// Binary on-disk (and, eventually, GPU-streamed) format for the virtual geometry cache
// (Nanite-style clustered geometry). The whole scene's clusters -- every entity, every DAG level
// -- are consolidated into ONE .cache file, laid out as four 4096-byte-aligned sections so a
// future GPU-driven streamer can page any section in with unbuffered, sector-aligned I/O:
//
//   [0]                      CacheFileHeader                  (exactly kPageSizeBytes bytes)
//   [clusterIndexTableOffset] ClusterIndexEntry[clusterCount]  (zero-padded up to a page boundary)
//   [dagTableOffset]          DAGNodeEntry[clusterCount]       (zero-padded up to a page boundary)
//   [geometryDataBaseOffset]  ClusterData[clusterCount]        (each entry individually page-aligned)
//
// Every structure below is `#pragma pack(push, 1)`-ed: no compiler-inserted inter-member padding
// is ever relied upon, so the layout is identical byte-for-byte across compilers/platforms (a
// requirement for a format a separate GPU-side loader must decode without pulling in this C++
// struct definition). Where a section must additionally land on a page boundary (the header
// itself, and the start of each geometry block), that alignment is guaranteed by the *writer*
// (CacheFileManager::WriteCacheFile) explicitly zero-filling the gap -- not by struct-level
// alignas/padding tricks, so the byte-for-byte zero-fill is visible in the write path itself.
//
// clusterIndexTableOffset/dagTableOffset/geometryDataBaseOffset are always multiples of
// kPageSizeBytes; clusterIndexTableSizeBytes/dagTableSizeBytes are the *unpadded* table sizes
// (clusterCount * sizeof(entry)) -- a reader uses those to know how many real entries to parse,
// while the writer separately pads the file up to the next page boundary afterwards.

#include <cstdint>
#include "core/maths/Maths.h"

namespace geometry {

    // Sentinel used by DAGNodeEntry::parentClusterID/childClusterID to mean "no such cluster" --
    // this is a DAG root (no parent) or a leaf (no children).
    constexpr uint32_t kInvalidClusterID = 0xFFFFFFFFu;

    // Maximum vertices/triangles a single cluster (meshlet) can hold. Chosen to match common GPU
    // mesh-shader meshlet limits (VK_EXT_mesh_shader) and to keep ClusterData a fixed size.
    constexpr uint32_t kMaxClusterVertices = 64u;
    constexpr uint32_t kMaxClusterTriangles = 128u;
    constexpr uint32_t kMaxClusterIndices = kMaxClusterTriangles * 3u;

    // The atomic disk I/O / section-alignment granularity: every major section of the file (the
    // header, and the start of each cluster's geometry block) begins on a multiple of this many
    // bytes, matching a real storage device's 4K sector size for FILE_FLAG_NO_BUFFERING streaming.
    constexpr uint32_t kPageSizeBytes = 4096u;

#pragma pack(push, 1)

    // -----------------------------------------------------------------------------------------
    // CacheFileHeader — the very first bytes of a .cache file. Its own on-disk footprint is
    // exactly kPageSizeBytes (the struct itself is much smaller; CacheFileManager::WriteCacheFile
    // explicitly zero-fills the remainder of the page, see kCacheFileHeaderPaddedSizeBytes below).
    // -----------------------------------------------------------------------------------------
    struct CacheFileHeader {
        static constexpr uint32_t kMagic = 0x4F45474Cu;   // "LGEO" little-endian ("Local GEOmetry cache")
        static constexpr uint32_t kVersion = 2u;           // Bumped from the single-entity-per-file v1 format.

        uint32_t magic;
        uint32_t version;

        uint32_t clusterCount; // Total clusters across every entity and every DAG level in this file.
        uint32_t entityCount;  // Number of distinct entities (meshID values) contributing clusters.

        // Cluster index table: ClusterIndexEntry[clusterCount], one entry per cluster, giving its
        // virtual address (byte offset) and block size in the geometry data section, plus enough
        // metadata (bounds, sphere, cone) for CPU/GPU-side coarse culling without touching the
        // (potentially not-yet-streamed-in) geometry block itself.
        uint64_t clusterIndexTableOffset;
        uint64_t clusterIndexTableSizeBytes; // == clusterCount * sizeof(ClusterIndexEntry), unpadded.

        // DAG table: DAGNodeEntry[clusterCount], index-aligned 1:1 with the cluster index table
        // (DAGNodeEntry[i] describes the same cluster as ClusterIndexEntry[i]) -- parent/child
        // links and the e_local/e_parent error pair used for a runtime LOD cut.
        uint64_t dagTableOffset;
        uint64_t dagTableSizeBytes; // == clusterCount * sizeof(DAGNodeEntry), unpadded.

        // Geometry data section: ClusterData[clusterCount], NOT necessarily index-aligned with
        // the tables above (a cluster's actual byte offset is ClusterIndexEntry::virtualAddress,
        // since a future variable-resolution cluster could reserve more than one page here) --
        // this field only marks where the section as a whole begins.
        uint64_t geometryDataBaseOffset;

        uint64_t totalFileSizeBytes; // Full file size, for a reader's sanity check against fstat.
    };

    // How many bytes CacheFileHeader itself occupies on disk once page-aligned (the writer zero-
    // fills [sizeof(CacheFileHeader), kCacheFileHeaderPaddedSizeBytes) -- see file header comment).
    constexpr uint32_t kCacheFileHeaderPaddedSizeBytes = kPageSizeBytes;
    static_assert(sizeof(CacheFileHeader) <= kCacheFileHeaderPaddedSizeBytes,
        "CacheFileHeader must fit within one page so it can be zero-padded up to kPageSizeBytes");
    static_assert(sizeof(CacheFileHeader) == 64, "CacheFileHeader size drifted from the expected 64 bytes");

    // -----------------------------------------------------------------------------------------
    // ClusterIndexEntry — one per cluster, in the cluster index table.
    // -----------------------------------------------------------------------------------------
    struct ClusterIndexEntry {
        uint32_t clusterID; // Unique across the whole file; matches the same-index DAGNodeEntry::clusterID.
        uint32_t entityID;  // The owning entity's meshID.

        uint64_t virtualAddress; // Byte offset from the start of the file where this cluster's
                                  // ClusterData block begins. Always a multiple of kPageSizeBytes.
        uint32_t blockSizeBytes; // Size in bytes reserved for this cluster's geometry block.
                                  // Always a multiple of kPageSizeBytes; currently always exactly
                                  // kPageSizeBytes (one cluster == one page), but stored explicitly
                                  // rather than assumed so a future variable-resolution cluster
                                  // spanning multiple pages needs no format change.

        uint32_t vertexCount; // Valid entries in the referenced ClusterData (<= kMaxClusterVertices).
        uint32_t indexCount;  // Valid entries in ClusterData::indices (<= kMaxClusterIndices).

        // Axis-aligned bounding box, in the owning entity's local space. Doubles as the decode
        // range for this cluster's ClusterData quantized vertex positions (see GeometryEncoding.h).
        float boundsMin[3];
        float boundsMax[3];

        // Bounding sphere: redundant with the AABB but cheaper to test for coarse frustum /
        // hierarchical-LOD-cut culling (one distance check instead of six plane tests).
        float sphereCenter[3];
        float sphereRadius;

        // Normal cone (meshopt_Bounds-style), signed-8-bit quantized: every triangle normal in
        // this cluster lies within acos(coneCutoff / 127) of coneAxis. Lets the GPU cull an
        // entire back-facing cluster before rasterizing a single triangle of it.
        int8_t coneAxisX;
        int8_t coneAxisY;
        int8_t coneAxisZ;
        int8_t coneCutoff;
    };
    static_assert(sizeof(ClusterIndexEntry) == 72, "ClusterIndexEntry size drifted from the expected 72 bytes");

    // -----------------------------------------------------------------------------------------
    // DAGNodeEntry — one per cluster, in the DAG table, index-aligned with the cluster index
    // table (DAGNodeEntry[i] and ClusterIndexEntry[i] describe the same cluster).
    // -----------------------------------------------------------------------------------------
    struct DAGNodeEntry {
        uint32_t clusterID;         // Matches the same-index ClusterIndexEntry::clusterID.
        uint32_t parentClusterID;   // kInvalidClusterID if this cluster is a DAG root.
        uint32_t childClusterID[2]; // kInvalidClusterID for unused slots (a leaf has both unused).
        uint32_t level;             // 0 = leaf (exact geometry); +1 per grouping/simplification pass above that.

        // e_local / e_parent (see ClusterDAG.h): the geometric error introduced by this cluster,
        // and the error its own parent introduces. A runtime LOD cut compares a view-dependent
        // threshold against these to pick the coarsest acceptable cluster along each DAG path.
        // parentError is +infinity for a root; clusterError is exactly 0 for a leaf.
        float clusterError;
        float parentError;
    };
    static_assert(sizeof(DAGNodeEntry) == 28, "DAGNodeEntry size drifted from the expected 28 bytes");

    // -----------------------------------------------------------------------------------------
    // ClusterData — quantized vertex attributes + local triangle list for one cluster. This is
    // the payload written at each cluster's ClusterIndexEntry::virtualAddress, zero-padded up to
    // ClusterIndexEntry::blockSizeBytes by the writer.
    // -----------------------------------------------------------------------------------------

    // Vertex position quantized to 16 bits/channel, normalized against the owning
    // ClusterIndexEntry's [boundsMin, boundsMax] range (see GeometryEncoding::QuantizePosition).
    struct ClusterVertexPosition {
        uint16_t x;
        uint16_t y;
        uint16_t z;
    };
    static_assert(sizeof(ClusterVertexPosition) == 6, "ClusterVertexPosition must be 6 bytes");

    // Unit normal, octahedral-encoded into two 12-bit channels packed across 3 bytes (24 bits
    // total). See GeometryEncoding::EncodeOctNormal24 / DecodeOctNormal24.
    struct ClusterVertexNormal {
        uint8_t data[3];
    };
    static_assert(sizeof(ClusterVertexNormal) == 3, "ClusterVertexNormal must be 3 bytes");

    // Texture coordinates stored as IEEE-754 binary16 (half float) per channel.
    struct ClusterVertexUV {
        uint16_t u;
        uint16_t v;
    };
    static_assert(sizeof(ClusterVertexUV) == 4, "ClusterVertexUV must be 4 bytes");

    struct ClusterData {
        ClusterVertexPosition positions[kMaxClusterVertices];
        ClusterVertexNormal normals[kMaxClusterVertices];
        ClusterVertexUV uvs[kMaxClusterVertices];
        // Local (cluster-relative) triangle-list indices, in [0, kMaxClusterVertices), indexing
        // into the arrays above.
        uint8_t indices[kMaxClusterIndices];
    };
    static_assert(sizeof(ClusterData) == 1216, "ClusterData size drifted from the expected 1216 bytes");
    static_assert(kPageSizeBytes >= sizeof(ClusterData),
        "ClusterData must fit within one page so a cluster's geometry block can be a single page");

#pragma pack(pop)

}
