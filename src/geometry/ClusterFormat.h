#pragma once
// Binary on-disk (and, eventually, GPU-streamed) format for the virtual geometry cache
// (Nanite-style clustered geometry). Every structure in this file is a trivial, standard-layout
// POD so it can be memcpy'd directly to/from disk via CacheFileManager (including unbuffered,
// sector-aligned I/O) with no serialization/deserialization pass beyond what the GPU shaders
// eventually decode themselves (quantized positions, oct-encoded normals, half-float UVs...).

#include <array>
#include <cstddef>
#include <cstdint>
#include "core/maths/Maths.h"

namespace geometry {

    // Sentinel used by ClusterHeader::parentID to mean "this cluster is a DAG root: it has no
    // coarser parent to collapse into".
    constexpr uint32_t kInvalidClusterID = 0xFFFFFFFFu;

    // Maximum vertices/triangles a single cluster (meshlet) can hold. Chosen to match common
    // GPU mesh-shader meshlet limits (VK_EXT_mesh_shader) and to keep ClusterData a fixed,
    // page-friendly size (see the static_asserts below).
    constexpr uint32_t kMaxClusterVertices = 128u;
    constexpr uint32_t kMaxClusterTriangles = 128u;
    constexpr uint32_t kMaxClusterIndices = kMaxClusterTriangles * 3u;

    // -----------------------------------------------------------------------------------------
    // ClusterHeader — one per cluster. Exactly 64 bytes (one CPU cache line) so an array of
    // headers can be scanned/culled on the CPU, or copied verbatim into a GPU SSBO, with zero
    // padding waste.
    // -----------------------------------------------------------------------------------------
    struct alignas(64) ClusterHeader {
        // Axis-aligned bounding box, in the owning entity's local space. Doubles as the decode
        // range for this cluster's ClusterData quantized vertex positions (see GeometryEncoding.h).
        maths::vec3 boundsMin;
        maths::vec3 boundsMax;

        // Bounding sphere: redundant with the AABB but cheaper to test for coarse frustum /
        // hierarchical-LOD-cut culling (one distance check instead of six plane tests).
        maths::vec3 sphereCenter;
        float sphereRadius;

        // Normal cone (meshopt_Bounds-style), signed-8-bit quantized: every triangle normal in
        // this cluster lies within acos(coneCutoff / 127) of coneAxis. Lets the GPU cull an
        // entire back-facing cluster before rasterizing a single triangle of it.
        int8_t coneAxisX;
        int8_t coneAxisY;
        int8_t coneAxisZ;
        int8_t coneCutoff;

        // DAG-based hierarchical LOD error metrics (Nanite-style): clusterError is the world-
        // space geometric error introduced by simplifying down to this cluster; parentError is
        // the error that would be introduced if the renderer collapsed this cluster into its DAG
        // parent instead. A view-dependent cut of the DAG selects the coarsest cluster whose
        // projected screen-space error is still below threshold.
        float clusterError;
        float parentError;

        uint16_t vertexCount; // Valid entries in the associated ClusterData (<= kMaxClusterVertices)
        uint16_t indexCount;  // Valid entries in ClusterData::indices (<= kMaxClusterIndices), = triangleCount * 3

        uint32_t parentID;  // Index of the parent cluster in the DAG, or kInvalidClusterID if this is a root
        uint32_t clusterID; // This cluster's own index within its owning entity's DAG
    };
    static_assert(sizeof(ClusterHeader) == 64, "ClusterHeader must be exactly 64 bytes (one cache line)");
    static_assert(alignof(ClusterHeader) == 64, "ClusterHeader must be 64-byte aligned");

    // -----------------------------------------------------------------------------------------
    // ClusterData — quantized vertex attributes + local triangle list for one cluster.
    // -----------------------------------------------------------------------------------------

    // Vertex position quantized to 16 bits/channel, normalized against the owning
    // ClusterHeader's [boundsMin, boundsMax] range (see GeometryEncoding::QuantizePosition).
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
        std::array<ClusterVertexPosition, kMaxClusterVertices> positions;
        std::array<ClusterVertexNormal, kMaxClusterVertices> normals;
        std::array<ClusterVertexUV, kMaxClusterVertices> uvs;
        // Local (cluster-relative) triangle-list indices, in [0, kMaxClusterVertices), indexing
        // into the arrays above.
        std::array<uint8_t, kMaxClusterIndices> indices;
    };
    static_assert(sizeof(ClusterData) == 2048, "ClusterData size drifted from the expected 2048 bytes");

    // With ClusterHeader = 64B and ClusterData = 2048B, one cluster occupies 2112B; two such
    // clusters (4224B) would overflow a 4KB page, so exactly one cluster fits per page at this
    // configuration. The array-based layout below still generalizes to N clusters/page if
    // kMaxClusterVertices or kPageSizeBytes are retuned later.
    constexpr uint32_t kPageSizeBytes = 4096u;
    constexpr uint32_t kMaxClustersPerPage = 1u;

    namespace detail {
        // Mirrors Page's members (minus the tail padding) so its sizeof can be used to compute
        // that padding below. MSVC rejects `offsetof(Page, padding)` from inside Page's own
        // definition (the type is still incomplete at that point) even though GCC/Clang accept
        // that self-referential pattern, so the byte count has to come from a standalone type
        // with an identical, alignas-free member sequence instead.
        struct PageBody {
            uint32_t clusterCount;
            uint32_t reserved;
            std::array<ClusterHeader, kMaxClustersPerPage> headers;
            std::array<ClusterData, kMaxClustersPerPage> data;
        };
    }

    // -----------------------------------------------------------------------------------------
    // Page — the atomic disk I/O unit: exactly 4096 bytes so it maps 1:1 onto a 4K storage
    // sector/OS page, letting CacheFileManager stream it with FILE_FLAG_NO_BUFFERING.
    // -----------------------------------------------------------------------------------------
    struct alignas(4096) Page {
        static constexpr uint32_t kPageSizeBytes = geometry::kPageSizeBytes;
        static constexpr uint32_t kMaxClustersPerPage = geometry::kMaxClustersPerPage;

        uint32_t clusterCount; // Valid entries in headers/data (<= kMaxClustersPerPage)
        uint32_t reserved;     // Reserved for future per-page flags (compression, version...)
        std::array<ClusterHeader, kMaxClustersPerPage> headers;
        std::array<ClusterData, kMaxClustersPerPage> data;

        // Explicit tail padding, sized from detail::PageBody's size (identical member layout)
        // so the struct's total size always lands on exactly kPageSizeBytes regardless of future
        // edits above (also absorbs the compiler-inserted padding needed to 64-byte-align the
        // `headers` array).
        uint8_t padding[kPageSizeBytes - sizeof(detail::PageBody)];
    };
    static_assert(sizeof(Page) == Page::kPageSizeBytes, "Page must be exactly 4096 bytes");
    static_assert(alignof(Page) == 4096, "Page must be 4096-byte aligned");

    constexpr uint32_t kCacheHeaderSizeBytes = 4096u;
    // Sized so the allocation table below occupies most of the header, then self-pads (via the
    // trailing `padding` member) to exactly kCacheHeaderSizeBytes.
    constexpr uint32_t kMaxPageEntries = 500u;

    namespace detail {
        // Mirrors CacheHeader's members (minus the tail padding); see PageBody above for why this
        // indirection is needed instead of a self-referential offsetof(CacheHeader, padding).
        struct CacheHeaderBody {
            uint32_t magic;
            uint32_t version;
            uint32_t pageCount;
            uint32_t pageSizeBytes;

            uint64_t entityID;
            uint32_t meshID;
            uint32_t materialID;
            uint32_t rootClusterIndex;
            uint32_t totalVertexCount;
            uint32_t totalTriangleCount;

            std::array<uint64_t, kMaxPageEntries> pageOffsets;
        };
    }

    // -----------------------------------------------------------------------------------------
    // CacheHeader — present once at the very start of a .cache file. Exactly 4096 bytes (one
    // Page's worth) so pages always start at a page-aligned file offset: (1 + pageIndex) * 4096.
    // -----------------------------------------------------------------------------------------
    struct alignas(4096) CacheHeader {
        static constexpr uint32_t kMagic = 0x48434143u; // "CACH" little-endian
        static constexpr uint32_t kVersion = 1u;
        static constexpr uint32_t kHeaderSizeBytes = geometry::kCacheHeaderSizeBytes;
        static constexpr uint32_t kMaxPageEntries = geometry::kMaxPageEntries;

        uint32_t magic;
        uint32_t version;
        uint32_t pageCount;     // Number of valid entries in pageOffsets (<= kMaxPageEntries)
        uint32_t pageSizeBytes; // = Page::kPageSizeBytes; lets a reader sanity-check format compatibility

        uint64_t entityID; // core::EntityID this cache file describes
        uint32_t meshID;
        uint32_t materialID;
        uint32_t rootClusterIndex;   // Index into pageOffsets of the DAG root cluster's page
        uint32_t totalVertexCount;   // Sum of vertexCount across every cluster, for stats/debug
        uint32_t totalTriangleCount; // Sum of (indexCount / 3) across every cluster

        // Global page allocation table: pageOffsets[i] is the absolute byte offset from the
        // start of the file where page i begins. Always a multiple of Page::kPageSizeBytes,
        // since pages are written back-to-back immediately after this header
        // (see CacheFileManager::WriteCacheFile).
        std::array<uint64_t, kMaxPageEntries> pageOffsets;

        uint8_t padding[kHeaderSizeBytes - sizeof(detail::CacheHeaderBody)];
    };
    static_assert(sizeof(CacheHeader) == CacheHeader::kHeaderSizeBytes, "CacheHeader must be exactly 4096 bytes");
    static_assert(alignof(CacheHeader) == 4096, "CacheHeader must be 4096-byte aligned");

}
