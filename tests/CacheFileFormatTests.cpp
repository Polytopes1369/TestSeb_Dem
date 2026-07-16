// Standalone, framework-free unit test for the .cache binary format itself
// (src/geometry/ClusterFormat.h + CacheFileManager.h/.cpp), independent of Vulkan/GLFW/the GPU
// readback path VirtualGeometryCacheTest.cpp drives at runtime. Builds a real multi-level cluster
// DAG on synthetic geometry, encodes and writes it as a consolidated .cache file exactly the way
// VirtualGeometryCacheTest.cpp does, then reads the raw file bytes back to directly verify the
// binary-format contract requested for this pass:
//   1. Every major section (header, cluster index table, DAG table, each cluster's geometry
//      block) starts at a file offset that is an exact multiple of kPageSizeBytes (4096).
//   2. The padding gap between a section's real data and the next page boundary is all zero
//      bytes -- checked by reading the raw file, not inferred from how the struct was built.
//   3. Every struct in ClusterFormat.h has the exact size its #pragma pack(push, 1) layout
//      implies (already static_assert'd in the header; this test additionally cross-checks the
//      file's actual byte layout against those sizes).
//   4. The header/tables/a sample cluster round-trip byte-exact through a real write + read.
//   5. The on-disk DAG table, decoded back into a ClusterDAG, still satisfies every
//      ValidateClusterDAG invariant (acyclic, strictly increasing error toward the root).
//
// Exits 0 if every check passes, non-zero otherwise, so it can be registered with CTest without
// pulling in any external test framework or requiring a window/device.

#include "io/CacheFileManager.h"
#include "geometry/CardGenerator.h"
#include "geometry/ClusterDAG.h"
#include "geometry/ClusterFormat.h"
#include "geometry/FallbackMeshBuilder.h"
#include "geometry/GeometryEncoding.h"
#include "core/maths/Maths.h"
#include "renderer/RenderTypes.h"
#include "SyntheticMesh.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

    using testutil::GenerateUVSphere;

    int g_failCount = 0;

    bool Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
        return condition;
    }

    // Mirrors VirtualGeometryCacheTest.cpp's ComputeVertexNormals/EncodeClusterData/
    // BuildIndexEntry (kept private to that file): recomputes face-derived normals since
    // SimplifiableMesh only carries positions.
    std::vector<maths::vec3> ComputeVertexNormals(const geometry::SimplifiableMesh& mesh) {
        std::vector<maths::vec3> normals(mesh.positions.size(), maths::vec3{ 0.0f, 0.0f, 0.0f });
        for (size_t t = 0; t + 2 < mesh.triangles.size(); t += 3) {
            uint32_t i0 = mesh.triangles[t + 0], i1 = mesh.triangles[t + 1], i2 = mesh.triangles[t + 2];
            maths::vec3 faceNormal = (mesh.positions[i1] - mesh.positions[i0]).Cross(mesh.positions[i2] - mesh.positions[i0]);
            normals[i0] = normals[i0] + faceNormal;
            normals[i1] = normals[i1] + faceNormal;
            normals[i2] = normals[i2] + faceNormal;
        }
        for (maths::vec3& n : normals) {
            n = n.Normalize();
        }
        return normals;
    }

    bool EncodeCluster(const geometry::ClusterDAGNode& node, geometry::ClusterData& outData) {
        uint32_t vertexCount = static_cast<uint32_t>(node.mesh.positions.size());
        uint32_t indexCount = static_cast<uint32_t>(node.mesh.triangles.size());
        if (vertexCount > geometry::kMaxClusterVertices || indexCount > geometry::kMaxClusterIndices) {
            return false;
        }
        std::vector<maths::vec3> normals = ComputeVertexNormals(node.mesh);
        outData = geometry::ClusterData{};
        for (uint32_t v = 0; v < vertexCount; ++v) {
            outData.positions[v] = geometry::QuantizePosition(node.mesh.positions[v], node.boundsMin, node.boundsMax);
            outData.normals[v] = geometry::EncodeOctNormal24(normals[v]);
            outData.uvs[v] = geometry::EncodeUV(maths::vec2{ 0.0f, 0.0f });
        }
        for (uint32_t i = 0; i < indexCount; ++i) {
            outData.indices[i] = static_cast<uint8_t>(node.mesh.triangles[i]);
        }
        return true;
    }

    geometry::ClusterIndexEntry BuildIndexEntry(uint32_t globalClusterID, uint32_t entityID, const geometry::ClusterDAGNode& node) {
        geometry::ClusterIndexEntry entry{};
        entry.clusterID = globalClusterID;
        entry.entityID = entityID;
        entry.vertexCount = static_cast<uint32_t>(node.mesh.positions.size());
        entry.indexCount = static_cast<uint32_t>(node.mesh.triangles.size());
        entry.boundsMin[0] = node.boundsMin.x; entry.boundsMin[1] = node.boundsMin.y; entry.boundsMin[2] = node.boundsMin.z;
        entry.boundsMax[0] = node.boundsMax.x; entry.boundsMax[1] = node.boundsMax.y; entry.boundsMax[2] = node.boundsMax.z;
        entry.sphereCenter[0] = node.sphereCenter.x; entry.sphereCenter[1] = node.sphereCenter.y; entry.sphereCenter[2] = node.sphereCenter.z;
        entry.sphereRadius = node.sphereRadius;
        return entry;
    }

    // Mirrors VirtualGeometryCacheTest.cpp's own BuildFallbackMeshData (kept private to that file):
    // converts a geometry::FallbackMesh into an on-disk-ready geometry::FallbackMeshData.
    // vertexDataOffset/indexDataOffset are left zeroed -- WriteCacheFile fills them in place.
    geometry::FallbackMeshData BuildFallbackMeshData(uint32_t entityID, const geometry::FallbackMesh& mesh) {
        geometry::FallbackMeshData data;
        data.indexEntry.entityID = entityID;
        data.indexEntry.vertexCount = static_cast<uint32_t>(mesh.positions.size());
        data.indexEntry.triangleCount = static_cast<uint32_t>(mesh.triangles.size() / 3u);
        data.indexEntry.vertexDataOffset = 0;
        data.indexEntry.indexDataOffset = 0;

        maths::vec3 boundsMin{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        maths::vec3 boundsMax{ std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

        data.vertices.reserve(mesh.positions.size());
        for (size_t v = 0; v < mesh.positions.size(); ++v) {
            const maths::vec3& p = mesh.positions[v];
            boundsMin.x = std::min(boundsMin.x, p.x); boundsMin.y = std::min(boundsMin.y, p.y); boundsMin.z = std::min(boundsMin.z, p.z);
            boundsMax.x = std::max(boundsMax.x, p.x); boundsMax.y = std::max(boundsMax.y, p.y); boundsMax.z = std::max(boundsMax.z, p.z);

            geometry::FallbackVertex fv{};
            fv.position[0] = p.x; fv.position[1] = p.y; fv.position[2] = p.z;
            fv.normal[0] = mesh.normals[v].x; fv.normal[1] = mesh.normals[v].y; fv.normal[2] = mesh.normals[v].z;
            fv.uv[0] = mesh.uvs[v].x; fv.uv[1] = mesh.uvs[v].y;
            data.vertices.push_back(fv);
        }
        if (mesh.positions.empty()) {
            boundsMin = maths::vec3{ 0.0f, 0.0f, 0.0f };
            boundsMax = maths::vec3{ 0.0f, 0.0f, 0.0f };
        }
        data.indexEntry.boundsMin[0] = boundsMin.x; data.indexEntry.boundsMin[1] = boundsMin.y; data.indexEntry.boundsMin[2] = boundsMin.z;
        data.indexEntry.boundsMax[0] = boundsMax.x; data.indexEntry.boundsMax[1] = boundsMax.y; data.indexEntry.boundsMax[2] = boundsMax.z;

        data.indices = mesh.triangles;
        return data;
    }

    // Reads `sizeBytes` raw bytes at `offset` from `filePath` via a plain buffered read.
    bool ReadRawBytes(const std::filesystem::path& filePath, uint64_t offset, size_t sizeBytes, std::vector<uint8_t>& outBytes) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file) {
            return false;
        }
        file.seekg(static_cast<std::streamoff>(offset));
        outBytes.assign(sizeBytes, 0xCC); // Poison value: any byte left un-overwritten by a short read is caught below.
        file.read(reinterpret_cast<char*>(outBytes.data()), static_cast<std::streamsize>(sizeBytes));
        return file.gcount() == static_cast<std::streamsize>(sizeBytes);
    }

    bool AllZero(const std::vector<uint8_t>& bytes) {
        return std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == 0; });
    }

} // namespace

int main() {
    // --- Build a real multi-level DAG on synthetic geometry, two "entities" sharing one buffer -
    //
    // Resolution is deliberately modest: when two ~kMaxClusterTriangles-sized clusters are
    // merged into a group, every vertex on their *combined outer boundary* is locked (see
    // ClusterGrouping.h) and can never be removed by SimplifyMeshQEM, no matter how tight a
    // vertex target it is given. For a dense enough partition, a merged group's boundary alone
    // can legitimately exceed kMaxClusterVertices -- EncodeCluster below is expected to reject
    // such a node (this is exercised on purpose, with a tiny pathological mesh, in the dedicated
    // rejection-path scenario further down) rather than silently truncate or corrupt it. A
    // coarser partition keeps this round-trip demonstration on the well-behaved happy path.
    std::vector<renderer::Vertex> vertices;
    std::vector<uint32_t> indices;
    GenerateUVSphere(0u, 10u, 10u, 2.0f, maths::vec3{ 0.0f, 0.0f, 0.0f }, vertices, indices);
    GenerateUVSphere(1u, 6u, 6u, 1.0f, maths::vec3{ 8.0f, 0.0f, 0.0f }, vertices, indices);

    std::vector<geometry::ClusterIndexEntry> indexEntries;
    std::vector<geometry::DAGNodeEntry> dagEntries;
    std::vector<geometry::ClusterData> clusterData;
    std::vector<geometry::FallbackMeshData> fallbackMeshes;
    std::vector<geometry::SurfaceCacheCardEntry> cardEntries;
    uint32_t entitiesWithGeometry = 0;

    for (uint32_t meshID : { 0u, 1u }) {
        geometry::ClusterDAG dag = geometry::BuildClusterDAG(meshID, vertices, indices, geometry::kInvalidMaskTextureIndex);
        if (!Check(!dag.nodes.empty(), "expected a non-empty DAG for meshID " + std::to_string(meshID))) {
            continue;
        }
        ++entitiesWithGeometry;

        std::vector<std::string> dagErrors;
        Check(geometry::ValidateClusterDAG(dag, dagErrors), "pre-write DAG must validate cleanly for meshID " + std::to_string(meshID));

        geometry::FallbackMesh fallbackMesh = geometry::BuildFallbackMesh(dag);
        Check(!fallbackMesh.triangles.empty(), "expected a non-empty fallback mesh for meshID " + std::to_string(meshID));
        if (!fallbackMesh.triangles.empty()) {
            fallbackMeshes.push_back(BuildFallbackMeshData(meshID, fallbackMesh));

            // Surface-cache cards from the same AABB the fallback-mesh entry carries -- a solid
            // sphere's AABB has 6 non-degenerate faces, so exactly kMaxCardsPerEntity cards.
            const geometry::FallbackMeshIndexEntry& fbEntry = fallbackMeshes.back().indexEntry;
            std::vector<geometry::SurfaceCacheCardEntry> entityCards = geometry::GenerateEntityCards(meshID,
                maths::vec3{ fbEntry.boundsMin[0], fbEntry.boundsMin[1], fbEntry.boundsMin[2] },
                maths::vec3{ fbEntry.boundsMax[0], fbEntry.boundsMax[1], fbEntry.boundsMax[2] });
            Check(entityCards.size() == geometry::kMaxCardsPerEntity,
                "a solid sphere's AABB must produce all 6 cards for meshID " + std::to_string(meshID));
            cardEntries.insert(cardEntries.end(), entityCards.begin(), entityCards.end());
        }

        uint32_t baseGlobalID = static_cast<uint32_t>(indexEntries.size());
        std::vector<uint32_t> localToGlobal(dag.nodes.size());
        for (size_t i = 0; i < dag.nodes.size(); ++i) {
            localToGlobal[i] = baseGlobalID + static_cast<uint32_t>(i);
        }

        for (size_t i = 0; i < dag.nodes.size(); ++i) {
            const geometry::ClusterDAGNode& node = dag.nodes[i];
            uint32_t globalID = localToGlobal[i];

            geometry::ClusterData data{};
            Check(EncodeCluster(node, data), "cluster " + std::to_string(globalID) + " must encode within fixed capacity");
            clusterData.push_back(data);
            indexEntries.push_back(BuildIndexEntry(globalID, meshID, node));

            geometry::DAGNodeEntry dagEntry{};
            dagEntry.clusterID = globalID;
            dagEntry.parentClusterID = (node.parentIndex == geometry::kInvalidDAGNodeIndex) ? geometry::kInvalidClusterID : localToGlobal[node.parentIndex];
            dagEntry.childClusterID[0] = (node.childIndices.size() > 0) ? localToGlobal[node.childIndices[0]] : geometry::kInvalidClusterID;
            dagEntry.childClusterID[1] = (node.childIndices.size() > 1) ? localToGlobal[node.childIndices[1]] : geometry::kInvalidClusterID;
            dagEntry.level = node.level;
            dagEntry.clusterError = node.clusterError;
            dagEntry.parentError = node.parentError;
            dagEntries.push_back(dagEntry);
        }
    }

    Check(!indexEntries.empty(), "expected at least one cluster across both entities");

    // --- Deliberately exercise the capacity-rejection safety rail --------------------------
    // A fully-locked, pathologically vertex-heavy mesh: no edge can ever legally collapse (every
    // endpoint is locked), so SimplifyMeshQEM must leave it untouched, and EncodeCluster must
    // then refuse to encode it into the fixed-capacity ClusterData block rather than truncate or
    // overrun the fixed arrays. This is the same rejection path a real, densely-partitioned mesh
    // can legitimately hit (see the resolution comment above) -- tested here on purpose instead
    // of relying on stumbling into it.
    {
        geometry::SimplifiableMesh pathological;
        constexpr uint32_t kTriangleCount = 25; // 75 unique, fully-locked vertices > kMaxClusterVertices (64).
        for (uint32_t t = 0; t < kTriangleCount; ++t) {
            float x = static_cast<float>(t);
            pathological.positions.push_back(maths::vec3{ x, 0.0f, 0.0f });
            pathological.positions.push_back(maths::vec3{ x + 0.5f, 1.0f, 0.0f });
            pathological.positions.push_back(maths::vec3{ x + 1.0f, 0.0f, 0.0f });
            pathological.triangles.push_back(t * 3 + 0);
            pathological.triangles.push_back(t * 3 + 1);
            pathological.triangles.push_back(t * 3 + 2);
        }
        pathological.locked.assign(pathological.positions.size(), true);

        uint32_t resultTriangleCount = geometry::SimplifyMeshQEM(pathological, 1u, geometry::kMaxClusterVertices);
        Check(resultTriangleCount == kTriangleCount, "a fully-locked mesh must be left untouched by SimplifyMeshQEM (no legal collapse exists)");
        Check(pathological.positions.size() > geometry::kMaxClusterVertices,
            "sanity: this pathological mesh must still exceed kMaxClusterVertices after simplification");

        geometry::ClusterDAGNode fakeNode;
        fakeNode.mesh = pathological;
        geometry::ClusterData discardedData;
        Check(!EncodeCluster(fakeNode, discardedData),
            "EncodeCluster must reject a cluster exceeding kMaxClusterVertices rather than truncate/corrupt it");
    }

    // --- Write the consolidated .cache file --------------------------------------------------
    geometry::CacheFileManager cacheManager;
    std::filesystem::path filePath = "cache_format_test.cache";
    std::error_code removeEc;
    std::filesystem::remove(filePath, removeEc);

    // Pack every card into the atlas BEFORE writing -- WriteCacheFile persists the mapping
    // verbatim, so an unpacked table would round-trip "successfully" while being useless.
    Check(geometry::PackCardsIntoAtlas(cardEntries), "PackCardsIntoAtlas must fit both entities' cards");

    bool wrote = cacheManager.WriteCacheFile(filePath, indexEntries, dagEntries, clusterData, fallbackMeshes, cardEntries, entitiesWithGeometry);
    Check(wrote, "WriteCacheFile must succeed");
    if (!wrote) {
        std::cerr << "[CacheFileFormatTests] " << g_failCount << " check(s) FAILED.\n";
        return 1;
    }

    // --- Structural checks: every major section starts on a 4096-byte boundary ---------------
    geometry::CacheFileHeader header{};
    Check(cacheManager.ReadHeader(filePath, header), "ReadHeader must succeed");
    Check(header.clusterIndexTableOffset % geometry::kPageSizeBytes == 0, "cluster index table offset must be page-aligned");
    Check(header.dagTableOffset % geometry::kPageSizeBytes == 0, "DAG table offset must be page-aligned");
    Check(header.geometryDataBaseOffset % geometry::kPageSizeBytes == 0, "geometry data base offset must be page-aligned");
    Check(header.clusterIndexTableOffset == geometry::kCacheFileHeaderPaddedSizeBytes,
        "cluster index table must immediately follow the page-sized header");
    Check(header.fallbackMeshTableOffset % geometry::kPageSizeBytes == 0, "fallback-mesh table offset must be page-aligned");
    Check(header.fallbackMeshDataBaseOffset % geometry::kPageSizeBytes == 0, "fallback-mesh data base offset must be page-aligned");
    Check(header.fallbackMeshCount == fallbackMeshes.size(), "fallback-mesh count must match how many entities produced a fallback mesh");
    Check(header.cardTableOffset % geometry::kPageSizeBytes == 0, "surface-cache card table offset must be page-aligned");
    Check(header.cardCount == cardEntries.size(), "surface-cache card count must match the packed card list");
    Check(header.cardTableSizeBytes == cardEntries.size() * sizeof(geometry::SurfaceCacheCardEntry),
        "surface-cache card table size must be the unpadded entry count * entry size");

    std::error_code sizeEc;
    uint64_t actualFileSize = std::filesystem::file_size(filePath, sizeEc);
    Check(!sizeEc && actualFileSize == header.totalFileSizeBytes,
        "actual file size must match CacheFileHeader::totalFileSizeBytes");
    Check(actualFileSize % geometry::kPageSizeBytes == 0, "total file size must be a whole number of pages");

    // --- Zero-padding checks: read the raw gap bytes directly, don't infer them ---------------
    {
        std::vector<uint8_t> headerPadding;
        uint64_t gapStart = sizeof(geometry::CacheFileHeader);
        uint64_t gapSize = geometry::kCacheFileHeaderPaddedSizeBytes - gapStart;
        Check(ReadRawBytes(filePath, gapStart, static_cast<size_t>(gapSize), headerPadding), "must read the header's padding gap");
        Check(AllZero(headerPadding), "header padding gap must be all zero bytes");
    }
    {
        uint64_t realTableBytes = header.clusterIndexTableSizeBytes;
        uint64_t paddedTableBytes = header.dagTableOffset - header.clusterIndexTableOffset;
        uint64_t gapStart = header.clusterIndexTableOffset + realTableBytes;
        uint64_t gapSize = paddedTableBytes - realTableBytes;
        if (Check(gapSize <= 4096, "cluster index table padding gap must be less than one page (sanity)")) {
            std::vector<uint8_t> tablePadding;
            Check(ReadRawBytes(filePath, gapStart, static_cast<size_t>(gapSize), tablePadding), "must read the cluster index table's padding gap");
            Check(AllZero(tablePadding), "cluster index table padding gap must be all zero bytes");
        }
    }
    for (uint32_t i = 0; i < indexEntries.size(); ++i) {
        Check(indexEntries[i].virtualAddress % geometry::kPageSizeBytes == 0,
            "cluster " + std::to_string(i) + "'s virtualAddress must be page-aligned");
        uint64_t geometryGapStart = indexEntries[i].virtualAddress + sizeof(geometry::ClusterData);
        uint64_t geometryGapSize = indexEntries[i].blockSizeBytes - sizeof(geometry::ClusterData);
        std::vector<uint8_t> blockPadding;
        if (Check(ReadRawBytes(filePath, geometryGapStart, static_cast<size_t>(geometryGapSize), blockPadding),
            "must read cluster " + std::to_string(i) + "'s geometry block padding gap")) {
            Check(AllZero(blockPadding), "cluster " + std::to_string(i) + "'s geometry block padding gap must be all zero bytes");
        }
    }

    // --- Round-trip checks: header/tables/a sample cluster decode byte-exact -------------------
    std::vector<geometry::ClusterIndexEntry> readIndexEntries;
    bool indexOk = cacheManager.ReadClusterIndexTable(filePath, header, readIndexEntries)
        && readIndexEntries.size() == indexEntries.size()
        && std::memcmp(readIndexEntries.data(), indexEntries.data(), indexEntries.size() * sizeof(geometry::ClusterIndexEntry)) == 0;
    Check(indexOk, "cluster index table must round-trip byte-exact");

    std::vector<geometry::DAGNodeEntry> readDagEntries;
    bool dagOk = cacheManager.ReadDAGTable(filePath, header, readDagEntries)
        && readDagEntries.size() == dagEntries.size()
        && std::memcmp(readDagEntries.data(), dagEntries.data(), dagEntries.size() * sizeof(geometry::DAGNodeEntry)) == 0;
    Check(dagOk, "DAG table must round-trip byte-exact");

    if (dagOk) {
        geometry::ClusterDAG reconstructed;
        reconstructed.nodes.resize(readDagEntries.size());
        for (size_t i = 0; i < readDagEntries.size(); ++i) {
            const geometry::DAGNodeEntry& e = readDagEntries[i];
            geometry::ClusterDAGNode& n = reconstructed.nodes[i];
            n.level = e.level;
            n.clusterError = e.clusterError;
            n.parentError = e.parentError;
            n.parentIndex = (e.parentClusterID == geometry::kInvalidClusterID) ? geometry::kInvalidDAGNodeIndex : e.parentClusterID;
            for (uint32_t c : e.childClusterID) {
                if (c != geometry::kInvalidClusterID) n.childIndices.push_back(c);
            }
            if (n.parentIndex == geometry::kInvalidDAGNodeIndex) {
                reconstructed.rootIndices.push_back(static_cast<uint32_t>(i));
            }
        }
        std::vector<std::string> reErrors;
        Check(geometry::ValidateClusterDAG(reconstructed, reErrors), "on-disk DAG table must still validate after round-trip");
    }

    uint32_t sampleIdx = static_cast<uint32_t>(indexEntries.size() / 2);
    if (Check(readIndexEntries.size() > sampleIdx, "sample index must be in range")) {
        geometry::ClusterData readBack{};
        std::future<bool> future = cacheManager.ReadClusterDataAsync(filePath, readIndexEntries[sampleIdx].virtualAddress, readBack);
        bool clusterOk = future.get() && std::memcmp(&readBack, &clusterData[sampleIdx], sizeof(geometry::ClusterData)) == 0;
        Check(clusterOk, "sample cluster geometry block must round-trip byte-exact");
    }

    // --- Fallback-mesh section: table + every entity's geometry round-trip byte-exact ---------
    std::vector<geometry::FallbackMeshIndexEntry> expectedFallbackTable;
    expectedFallbackTable.reserve(fallbackMeshes.size());
    for (const geometry::FallbackMeshData& f : fallbackMeshes) {
        expectedFallbackTable.push_back(f.indexEntry);
    }
    std::vector<geometry::FallbackMeshIndexEntry> readFallbackTable;
    bool fallbackTableOk = cacheManager.ReadFallbackMeshTable(filePath, header, readFallbackTable)
        && readFallbackTable.size() == expectedFallbackTable.size()
        && (expectedFallbackTable.empty() || std::memcmp(readFallbackTable.data(), expectedFallbackTable.data(),
            expectedFallbackTable.size() * sizeof(geometry::FallbackMeshIndexEntry)) == 0);
    Check(fallbackTableOk, "fallback-mesh index table must round-trip byte-exact");

    for (size_t i = 0; i < fallbackMeshes.size(); ++i) {
        const geometry::FallbackMeshData& expected = fallbackMeshes[i];
        std::vector<geometry::FallbackVertex> readVertices;
        std::vector<uint32_t> readIndices;
        bool geometryOk = cacheManager.ReadFallbackMeshGeometry(filePath, expected.indexEntry, readVertices, readIndices)
            && readVertices.size() == expected.vertices.size()
            && readIndices.size() == expected.indices.size()
            && (expected.vertices.empty() || std::memcmp(readVertices.data(), expected.vertices.data(), expected.vertices.size() * sizeof(geometry::FallbackVertex)) == 0)
            && (expected.indices.empty() || std::memcmp(readIndices.data(), expected.indices.data(), expected.indices.size() * sizeof(uint32_t)) == 0);
        Check(geometryOk, "fallback mesh " + std::to_string(i) + " geometry must round-trip byte-exact");
    }

    // --- Surface-cache card section: byte-exact round-trip + the two invariants every consumer
    // relies on, verified on the ON-DISK bytes: (1) pairwise non-overlapping atlas rects,
    // (2) uvMin/uvMax exactly equal to the texel rect normalized by the atlas size ---------------
    std::vector<geometry::SurfaceCacheCardEntry> readCardTable;
    bool cardTableOk = cacheManager.ReadSurfaceCacheCardTable(filePath, header, readCardTable)
        && readCardTable.size() == cardEntries.size()
        && (cardEntries.empty() || std::memcmp(readCardTable.data(), cardEntries.data(),
            cardEntries.size() * sizeof(geometry::SurfaceCacheCardEntry)) == 0);
    Check(cardTableOk, "surface-cache card table must round-trip byte-exact");

    if (cardTableOk) {
        const float invAtlas = 1.0f / static_cast<float>(geometry::kSurfaceCacheAtlasSize);
        for (size_t a = 0; a < readCardTable.size(); ++a) {
            const geometry::SurfaceCacheCardEntry& ca = readCardTable[a];
            Check(ca.atlasOffset[0] + ca.atlasSize[0] <= geometry::kSurfaceCacheAtlasSize
                && ca.atlasOffset[1] + ca.atlasSize[1] <= geometry::kSurfaceCacheAtlasSize,
                "card " + std::to_string(a) + " must lie fully inside the atlas");
            Check(ca.uvMin[0] == static_cast<float>(ca.atlasOffset[0]) * invAtlas
                && ca.uvMin[1] == static_cast<float>(ca.atlasOffset[1]) * invAtlas
                && ca.uvMax[0] == static_cast<float>(ca.atlasOffset[0] + ca.atlasSize[0]) * invAtlas
                && ca.uvMax[1] == static_cast<float>(ca.atlasOffset[1] + ca.atlasSize[1]) * invAtlas,
                "card " + std::to_string(a) + "'s UV rect must be its texel rect normalized by the atlas size");
            for (size_t b = a + 1; b < readCardTable.size(); ++b) {
                const geometry::SurfaceCacheCardEntry& cb = readCardTable[b];
                bool overlap =
                    ca.atlasOffset[0] < cb.atlasOffset[0] + cb.atlasSize[0] &&
                    cb.atlasOffset[0] < ca.atlasOffset[0] + ca.atlasSize[0] &&
                    ca.atlasOffset[1] < cb.atlasOffset[1] + cb.atlasSize[1] &&
                    cb.atlasOffset[1] < ca.atlasOffset[1] + ca.atlasSize[1];
                Check(!overlap, "cards " + std::to_string(a) + " and " + std::to_string(b) + " must not overlap in the atlas");
            }
        }
    }

    std::cout << "[CacheFileFormatTests] " << indexEntries.size() << " cluster(s), "
        << actualFileSize << " byte file, header+2 tables+" << indexEntries.size() << " geometry block(s), all page-aligned.\n";

    if (g_failCount == 0) {
        std::cout << "[CacheFileFormatTests] All checks PASSED.\n";
        return 0;
    }
    std::cerr << "[CacheFileFormatTests] " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
