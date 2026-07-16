#pragma once
// Binary on-disk (and, eventually, GPU-streamed) format for the virtual geometry cache
// (Nanite-style clustered geometry). The whole scene's clusters -- every entity, every DAG level
// -- are consolidated into ONE .cache file, laid out as seven 4096-byte-aligned sections so a
// future GPU-driven streamer can page any section in with unbuffered, sector-aligned I/O:
//
//   [0]                       CacheFileHeader                     (exactly kPageSizeBytes bytes)
//   [clusterIndexTableOffset] ClusterIndexEntry[clusterCount]      (zero-padded up to a page boundary)
//   [dagTableOffset]          DAGNodeEntry[clusterCount]           (zero-padded up to a page boundary)
//   [geometryDataBaseOffset]  ClusterData[clusterCount]            (each entry individually page-aligned)
//   [fallbackMeshTableOffset] FallbackMeshIndexEntry[fallbackMeshCount] (zero-padded up to a page boundary)
//   [fallbackMeshDataBaseOffset] FallbackVertex[]/uint32_t[] blobs (section start page-aligned; NOT
//                                                                   individually page-aligned per
//                                                                   entity -- unlike ClusterData,
//                                                                   this is read once at startup for
//                                                                   BVH construction, never paged on
//                                                                   demand, so per-entry padding
//                                                                   would only waste space)
//   [cardTableOffset]         SurfaceCacheCardEntry[cardCount]      (zero-padded up to a page boundary)
//
// Every structure below is `#pragma pack(push, 1)`-ed: no compiler-inserted inter-member padding
// is ever relied upon, so the layout is identical byte-for-byte across compilers/platforms (a
// requirement for a format a separate GPU-side loader must decode without pulling in this C++
// struct definition). Where a section must additionally land on a page boundary (the header
// itself, and the start of each geometry block), that alignment is guaranteed by the *writer*
// (CacheFileManager::WriteCacheFile) explicitly zero-filling the gap -- not by struct-level
// alignas/padding tricks, so the byte-for-byte zero-fill is visible in the write path itself.
//
// clusterIndexTableOffset/dagTableOffset/geometryDataBaseOffset/fallbackMeshTableOffset/
// fallbackMeshDataBaseOffset are always multiples of kPageSizeBytes; clusterIndexTableSizeBytes/
// dagTableSizeBytes/fallbackMeshTableSizeBytes are the *unpadded* table sizes
// (count * sizeof(entry)) -- a reader uses those to know how many real entries to parse, while the
// writer separately pads the file up to the next page boundary afterwards.

#include <cstdint>
#include "core/maths/Maths.h"
#include "core/EngineConfig.h"

namespace geometry {

    // Sentinel used by DAGNodeEntry::parentClusterID/childClusterID to mean "no such cluster" --
    // this is a DAG root (no parent) or a leaf (no children).
    constexpr uint32_t kInvalidClusterID = 0xFFFFFFFFu;

    // Maximum vertices/triangles a single cluster (meshlet) can hold. Chosen to match common GPU
    // mesh-shader meshlet limits (VK_EXT_mesh_shader) and to keep ClusterData a fixed size.
    constexpr uint32_t kMaxClusterVertices = config::nanite::MAX_CLUSTER_VERTICES;
    constexpr uint32_t kMaxClusterTriangles = config::nanite::MAX_CLUSTER_TRIANGLES;
    constexpr uint32_t kMaxClusterIndices = kMaxClusterTriangles * 3u;

    // The atomic disk I/O / section-alignment granularity: every major section of the file (the
    // header, and the start of each cluster's geometry block) begins on a multiple of this many
    // bytes, matching a real storage device's 4K sector size for FILE_FLAG_NO_BUFFERING streaming.
    constexpr uint32_t kPageSizeBytes = config::nanite::PAGE_SIZE_BYTES;

#pragma pack(push, 1)

    // -----------------------------------------------------------------------------------------
    // CacheFileHeader — the very first bytes of a .cache file. Its own on-disk footprint is
    // exactly kPageSizeBytes (the struct itself is much smaller; CacheFileManager::WriteCacheFile
    // explicitly zero-fills the remainder of the page, see kCacheFileHeaderPaddedSizeBytes below).
    // -----------------------------------------------------------------------------------------
    struct CacheFileHeader {
        static constexpr uint32_t kMagic = 0x4F45474Cu;   // "LGEO" little-endian ("Local GEOmetry cache")
        static constexpr uint32_t kVersion = 6u;           // Bumped: added ClusterIndexEntry::materialID.

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

        // Fallback-mesh index table: FallbackMeshIndexEntry[fallbackMeshCount], one entry per
        // entity that produced a Fallback Mesh (geometry::BuildFallbackMesh) -- the coarse BVH
        // proxy geometry for hardware ray tracing acceleration structures.
        uint64_t fallbackMeshTableOffset;
        uint64_t fallbackMeshTableSizeBytes; // == fallbackMeshCount * sizeof(FallbackMeshIndexEntry), unpadded.

        // Fallback-mesh data section: concatenated FallbackVertex[]/uint32_t[] blobs, one pair per
        // entry in the fallback-mesh index table, addressed by that entry's own
        // vertexDataOffset/indexDataOffset (absolute file offsets) -- NOT individually page-aligned
        // within this section (see file header comment for why).
        uint64_t fallbackMeshDataBaseOffset;

        uint32_t fallbackMeshCount;

        // Surface-cache card table: SurfaceCacheCardEntry[cardCount], up to kMaxCardsPerEntity
        // orthographic box-face projections per entity (geometry::GenerateEntityCards), each with
        // a unique, non-overlapping rect pre-packed into the global surface cache atlas
        // (geometry::PackCardsIntoAtlas) -- the CPU-authored mapping the GPU surface cache capture
        // and GI sampling passes both consume unmodified.
        uint64_t cardTableOffset;
        uint64_t cardTableSizeBytes; // == cardCount * sizeof(SurfaceCacheCardEntry), unpadded.
        uint32_t cardCount;
    };

    // How many bytes CacheFileHeader itself occupies on disk once page-aligned (the writer zero-
    // fills [sizeof(CacheFileHeader), kCacheFileHeaderPaddedSizeBytes) -- see file header comment).
    constexpr uint32_t kCacheFileHeaderPaddedSizeBytes = kPageSizeBytes;
    static_assert(sizeof(CacheFileHeader) <= kCacheFileHeaderPaddedSizeBytes,
        "CacheFileHeader must fit within one page so it can be zero-padded up to kPageSizeBytes");
    static_assert(sizeof(CacheFileHeader) == 112, "CacheFileHeader size drifted from the expected 112 bytes");

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

        // World Position Offset / opacity-mask metadata. maxWPOAmplitude is stamped uniformly
        // across every DAG level of a given entity (see geometry::GetEntityMaterialProperties /
        // VirtualGeometryCacheTest.cpp's per-entity loop) -- the worst-case world-space
        // displacement the vertex shader's procedural sway function can ever apply to a vertex of
        // this cluster; the culling/LOD-error shaders inflate their bounding volumes by exactly
        // this much (see cluster_culling_tests.glsl's InflateForWPO) so a swaying cluster is never
        // mis-culled or mis-LOD-cut. maskTextureIndex indexes the bindless procedural cutout mask
        // array (mask_sampling.glsl) and is PER-CLUSTER, not per-entity: geometry::ClusterDAGNode::
        // isMasked (set by PartitionMeshIntoClusters's opacity classification/split, propagated
        // purity-checked through the whole DAG) decides, per cluster, whether this is the entity's
        // real mask slot or kInvalidMaskTextureIndex ("fully opaque, no cutout, safe for the
        // zero-overhead opaque rasterizer path") -- see VirtualGeometryCacheTest.cpp's BuildIndexEntry
        // call site.
        float maxWPOAmplitude;
        uint32_t maskTextureIndex;

        // The owning entity's core::EntityData::materialID, copied verbatim (same "stamped
        // uniformly across every DAG level of a given entity" contract as maxWPOAmplitude/
        // maskTextureIndex above) -- see VirtualGeometryCacheTest.cpp's BuildIndexEntry call site.
        // Indexes renderer::MaterialParameterTable's runtime PBR parameter lookup (baseColor/
        // roughness/metallic/emissive), consumed by ClusterResolve.comp. Unlike maskTextureIndex
        // (per-cluster, purity-enforced through ClusterDAG because opacity can vary by triangle),
        // materialID is per-ENTITY: every cluster of a given entity carries the same value, exactly
        // like entityID itself, so no DAG purity enforcement is needed for this field.
        uint32_t materialID;
    };
    static_assert(sizeof(ClusterIndexEntry) == 84, "ClusterIndexEntry size drifted from the expected 84 bytes");

    // Sentinel for ClusterIndexEntry::maskTextureIndex (and its GPU-side mirrors) meaning "this
    // cluster is fully opaque -- do not sample the cutout mask array at all."
    constexpr uint32_t kInvalidMaskTextureIndex = 0xFFFFFFFFu;

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

    // -----------------------------------------------------------------------------------------
    // Fallback Mesh section — one coarse proxy mesh per entity (geometry::BuildFallbackMesh), used
    // as input geometry for hardware ray tracing acceleration structures. Unlike ClusterData, this
    // is free-form (not fixed-capacity): full 32-bit indices, unquantized float attributes,
    // variable vertex/triangle count. Read once at startup for BVH construction, never paged.
    // -----------------------------------------------------------------------------------------

    // One per entity that produced a Fallback Mesh, in the fallback-mesh index table.
    struct FallbackMeshIndexEntry {
        uint32_t entityID;      // The owning entity's meshID (matches ClusterIndexEntry::entityID).
        uint32_t vertexCount;
        uint32_t triangleCount; // indexCount / 3.

        // Absolute file offsets (not page-aligned -- see file header comment) into the fallback-
        // mesh data section: FallbackVertex[vertexCount] at vertexDataOffset, then
        // uint32_t[triangleCount * 3] at indexDataOffset.
        uint64_t vertexDataOffset;
        uint64_t indexDataOffset;

        // Axis-aligned bounding box, in the owning entity's local space -- same convention as
        // ClusterIndexEntry::boundsMin/boundsMax.
        float boundsMin[3];
        float boundsMax[3];
    };
    static_assert(sizeof(FallbackMeshIndexEntry) == 52, "FallbackMeshIndexEntry size drifted from the expected 52 bytes");

    // One fallback-mesh vertex: full-precision (unquantized) position/normal/UV, unlike
    // ClusterData's quantized ClusterVertexPosition/Normal/UV -- this mesh is not part of the
    // paged streaming system, so there is no bandwidth motivation to quantize it, and a BVH build
    // wants full-precision positions directly.
    struct FallbackVertex {
        float position[3];
        float normal[3];
        float uv[2];
    };
    static_assert(sizeof(FallbackVertex) == 32, "FallbackVertex size drifted from the expected 32 bytes");

    // -----------------------------------------------------------------------------------------
    // Surface-cache card section — one entry per orthographic box-face projection ("Card") of an
    // entity, at most kMaxCardsPerEntity per entity (geometry::GenerateEntityCards skips a face
    // whose projected footprint is degenerate, e.g. the +/-X faces of a flat ground plane). The
    // lighting capture pass rasterizes the entity's Fallback Mesh through each card's
    // orthographic projection into the card's exclusive atlas rect; GI shaders later sample that
    // rect by reprojecting a world position through the same mapping -- so this entry IS the
    // shared contract between capture and sampling, persisted in the .cache file.
    // -----------------------------------------------------------------------------------------

    // Upper bound on cards per entity: the 6 faces of its axis-aligned bounding box.
    constexpr uint32_t kMaxCardsPerEntity = 6u;

    // SurfaceCacheCardEntry::faceDirection values: the outward axis the orthographic camera looks
    // AGAINST (a +X card captures the geometry as seen from +X looking toward -X, etc.).
    enum CardFaceDirection : uint32_t {
        kCardFacePosX = 0u,
        kCardFaceNegX = 1u,
        kCardFacePosY = 2u,
        kCardFaceNegY = 3u,
        kCardFacePosZ = 4u,
        kCardFaceNegZ = 5u,
    };

    struct SurfaceCacheCardEntry {
        uint32_t entityID;      // The owning entity's meshID (matches ClusterIndexEntry::entityID).
        uint32_t faceDirection; // One of CardFaceDirection.

        // The entity-local AABB this card's orthographic projection covers -- the full entity
        // bounds, identical for every card of one entity (each face projects the WHOLE volume, so
        // depth-tested capture keeps only the surfaces facing that card).
        float localBoundsMin[3];
        float localBoundsMax[3];

        // Exclusive (non-overlapping, guaranteed by geometry::PackCardsIntoAtlas) placement in
        // the global surface cache atlas, in integer texels...
        uint32_t atlasOffset[2];
        uint32_t atlasSize[2];

        // ...and the same rect pre-divided by the atlas dimensions: the normalized UV rect a GI
        // shader samples directly (uvMin maps to the card-local projection's (0,0) corner).
        float uvMin[2];
        float uvMax[2];
    };
    static_assert(sizeof(SurfaceCacheCardEntry) == 64, "SurfaceCacheCardEntry size drifted from the expected 64 bytes");

#pragma pack(pop)

}
