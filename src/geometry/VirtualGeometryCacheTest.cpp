#include "geometry/VirtualGeometryCacheTest.h"

#include "geometry/CacheFileManager.h"
#include "geometry/ClusterFormat.h"
#include "geometry/GeometryEncoding.h"
#include "core/Logger.h"
#include "renderer/RenderTypes.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <limits>
#include <unordered_map>
#include <vector>

namespace geometry {

    namespace {

        // -------------------------------------------------------------------------------------
        // Full (non-sampled) readback of the live procedural Vertex/Index SSBOs to host memory.
        // Mirrors VulkanContext::DebugReadbackGeometrySample's one-shot staging-buffer pattern,
        // but copies the *entire* used range instead of a small debug window, and returns the
        // data instead of just logging it.
        // -------------------------------------------------------------------------------------
        bool ReadbackFullGeometry(
            VkDevice device, VmaAllocator allocator, VkQueue graphicsQueue, VkCommandPool commandPool,
            VkBuffer vertexBuffer, VkBuffer indexBuffer,
            uint32_t totalVertexCount, uint32_t totalIndexCount,
            std::vector<renderer::Vertex>& outVertices, std::vector<uint32_t>& outIndices) {

            const VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(totalVertexCount) * sizeof(renderer::Vertex);
            const VkDeviceSize indexBytes = static_cast<VkDeviceSize>(totalIndexCount) * sizeof(uint32_t);

            VkBuffer stagingVertex = VK_NULL_HANDLE;
            VmaAllocation stagingVertexAlloc = VK_NULL_HANDLE;
            VkBufferCreateInfo vInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            vInfo.size = vertexBytes;
            vInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VmaAllocationCreateInfo stagingAllocInfo{ .usage = VMA_MEMORY_USAGE_GPU_TO_CPU };
            if (vmaCreateBuffer(allocator, &vInfo, &stagingAllocInfo, &stagingVertex, &stagingVertexAlloc, nullptr) != VK_SUCCESS) {
                Logger::Log(LogLevel::Error, "[GeometryCacheTest] Failed to allocate vertex staging buffer for full readback!");
                return false;
            }

            VkBuffer stagingIndex = VK_NULL_HANDLE;
            VmaAllocation stagingIndexAlloc = VK_NULL_HANDLE;
            VkBufferCreateInfo iInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            iInfo.size = indexBytes;
            iInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vmaCreateBuffer(allocator, &iInfo, &stagingAllocInfo, &stagingIndex, &stagingIndexAlloc, nullptr) != VK_SUCCESS) {
                Logger::Log(LogLevel::Error, "[GeometryCacheTest] Failed to allocate index staging buffer for full readback!");
                vmaDestroyBuffer(allocator, stagingVertex, stagingVertexAlloc);
                return false;
            }

            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd) != VK_SUCCESS) {
                Logger::Log(LogLevel::Error, "[GeometryCacheTest] Failed to allocate the full-readback command buffer!");
                vmaDestroyBuffer(allocator, stagingVertex, stagingVertexAlloc);
                vmaDestroyBuffer(allocator, stagingIndex, stagingIndexAlloc);
                return false;
            }

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkBufferCopy vertexCopyRegion{ 0, 0, vertexBytes };
            vkCmdCopyBuffer(cmd, vertexBuffer, stagingVertex, 1, &vertexCopyRegion);
            VkBufferCopy indexCopyRegion{ 0, 0, indexBytes };
            vkCmdCopyBuffer(cmd, indexBuffer, stagingIndex, 1, &indexCopyRegion);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphicsQueue);
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);

            outVertices.resize(totalVertexCount);
            outIndices.resize(totalIndexCount);

            bool ok = true;

            void* mappedVerts = nullptr;
            if (vmaMapMemory(allocator, stagingVertexAlloc, &mappedVerts) == VK_SUCCESS) {
                std::memcpy(outVertices.data(), mappedVerts, static_cast<size_t>(vertexBytes));
                vmaUnmapMemory(allocator, stagingVertexAlloc);
            }
            else {
                Logger::Log(LogLevel::Error, "[GeometryCacheTest] Failed to map vertex staging buffer!");
                ok = false;
            }

            void* mappedIndices = nullptr;
            if (ok && vmaMapMemory(allocator, stagingIndexAlloc, &mappedIndices) == VK_SUCCESS) {
                std::memcpy(outIndices.data(), mappedIndices, static_cast<size_t>(indexBytes));
                vmaUnmapMemory(allocator, stagingIndexAlloc);
            }
            else if (ok) {
                Logger::Log(LogLevel::Error, "[GeometryCacheTest] Failed to map index staging buffer!");
                ok = false;
            }

            vmaDestroyBuffer(allocator, stagingVertex, stagingVertexAlloc);
            vmaDestroyBuffer(allocator, stagingIndex, stagingIndexAlloc);
            return ok;
        }

        // -------------------------------------------------------------------------------------
        // Greedy per-entity meshlet builder: walks the entity's triangles (identified via the
        // per-vertex Vertex::meshID field the compute shaders already stamp on every vertex) and
        // buckets them into clusters of at most kMaxClusterVertices unique vertices / at most
        // kMaxClusterTriangles triangles, exactly as ClusterData's fixed-size arrays require.
        //
        // This does not attempt spatial-locality-aware meshlet construction (à la meshoptimizer)
        // — it only needs to produce *valid, decodable* clusters to prove the on-disk format and
        // CacheFileManager round-trip correctly. A real DAG/LOD simplification pass is a
        // separate, future piece of work; every cluster produced here is accordingly marked as
        // both a leaf and a DAG root (clusterError = 0, parentError = +infinity).
        // -------------------------------------------------------------------------------------
        struct EntityClusterBuildResult {
            std::vector<Page> pages;
            uint32_t totalVertexCount = 0;
            uint32_t totalTriangleCount = 0;
        };

        EntityClusterBuildResult BuildClustersForEntity(
            uint32_t targetMeshID,
            const std::vector<renderer::Vertex>& allVertices,
            const std::vector<uint32_t>& allIndices) {

            EntityClusterBuildResult result;

            std::unordered_map<uint32_t, uint8_t> globalToLocal;
            std::vector<uint32_t> localToGlobal;
            std::vector<uint8_t> localIndices;

            auto flushCluster = [&]() {
                if (localToGlobal.empty()) {
                    return;
                }

                maths::vec3 boundsMin{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
                maths::vec3 boundsMax{ std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
                for (uint32_t globalIdx : localToGlobal) {
                    const maths::vec3& p = allVertices[globalIdx].position;
                    boundsMin.x = std::min(boundsMin.x, p.x); boundsMin.y = std::min(boundsMin.y, p.y); boundsMin.z = std::min(boundsMin.z, p.z);
                    boundsMax.x = std::max(boundsMax.x, p.x); boundsMax.y = std::max(boundsMax.y, p.y); boundsMax.z = std::max(boundsMax.z, p.z);
                }

                maths::vec3 sphereCenter = (boundsMin + boundsMax) * 0.5f;
                float sphereRadius = (boundsMax - boundsMin).Length() * 0.5f;

                // Average normal direction defines the cluster's normal-cone axis; the cutoff is
                // the worst-case (minimum) dot product between the axis and any member normal.
                maths::vec3 axisAccum{ 0.0f, 0.0f, 0.0f };
                for (uint32_t globalIdx : localToGlobal) {
                    axisAccum = axisAccum + allVertices[globalIdx].normal;
                }
                maths::vec3 coneAxis = axisAccum.Normalize();
                float coneCutoff = 1.0f;
                for (uint32_t globalIdx : localToGlobal) {
                    maths::vec3 n = allVertices[globalIdx].normal.Normalize();
                    coneCutoff = std::min(coneCutoff, coneAxis.Dot(n));
                }
                coneCutoff = std::clamp(coneCutoff, -1.0f, 1.0f);

                ClusterHeader header{};
                header.boundsMin = boundsMin;
                header.boundsMax = boundsMax;
                header.sphereCenter = sphereCenter;
                header.sphereRadius = sphereRadius;
                header.coneAxisX = static_cast<int8_t>(std::lround(coneAxis.x * 127.0f));
                header.coneAxisY = static_cast<int8_t>(std::lround(coneAxis.y * 127.0f));
                header.coneAxisZ = static_cast<int8_t>(std::lround(coneAxis.z * 127.0f));
                header.coneCutoff = static_cast<int8_t>(std::lround(coneCutoff * 127.0f));
                // Single-LOD export: every cluster here is simultaneously a leaf and a DAG root
                // (no simplification pass exists yet to build coarser parents).
                header.clusterError = 0.0f;
                header.parentError = std::numeric_limits<float>::infinity();
                header.vertexCount = static_cast<uint16_t>(localToGlobal.size());
                header.indexCount = static_cast<uint16_t>(localIndices.size());
                header.parentID = kInvalidClusterID;
                header.clusterID = static_cast<uint32_t>(result.pages.size());

                ClusterData data{};
                for (size_t local = 0; local < localToGlobal.size(); ++local) {
                    const renderer::Vertex& v = allVertices[localToGlobal[local]];
                    data.positions[local] = QuantizePosition(v.position, boundsMin, boundsMax);
                    data.normals[local] = EncodeOctNormal24(v.normal);
                    data.uvs[local] = EncodeUV(v.uv);
                }
                std::copy(localIndices.begin(), localIndices.end(), data.indices.begin());

                Page page{};
                page.clusterCount = 1;
                page.reserved = 0;
                page.headers[0] = header;
                page.data[0] = data;
                result.pages.push_back(page);

                result.totalVertexCount += header.vertexCount;
                result.totalTriangleCount += header.indexCount / 3u;

                globalToLocal.clear();
                localToGlobal.clear();
                localIndices.clear();
                };

            for (size_t tri = 0; tri + 2 < allIndices.size(); tri += 3) {
                uint32_t i0 = allIndices[tri + 0];
                uint32_t i1 = allIndices[tri + 1];
                uint32_t i2 = allIndices[tri + 2];

                if (allVertices[i0].meshID != targetMeshID) {
                    continue; // This triangle belongs to a different entity.
                }

                uint32_t newVertsNeeded = 0;
                for (uint32_t g : { i0, i1, i2 }) {
                    if (globalToLocal.find(g) == globalToLocal.end()) {
                        ++newVertsNeeded;
                    }
                }
                bool wouldOverflowVerts = (localToGlobal.size() + newVertsNeeded) > kMaxClusterVertices;
                bool wouldOverflowTris = (localIndices.size() / 3u + 1u) > kMaxClusterTriangles;
                if (wouldOverflowVerts || wouldOverflowTris) {
                    flushCluster();
                }

                for (uint32_t g : { i0, i1, i2 }) {
                    auto it = globalToLocal.find(g);
                    uint8_t local;
                    if (it == globalToLocal.end()) {
                        local = static_cast<uint8_t>(localToGlobal.size());
                        globalToLocal.emplace(g, local);
                        localToGlobal.push_back(g);
                    }
                    else {
                        local = it->second;
                    }
                    localIndices.push_back(local);
                }
            }
            flushCluster(); // Flush the final, possibly-partial cluster.

            return result;
        }

    } // namespace

    bool RunVirtualGeometryCacheTest(
        VkDevice device, VmaAllocator allocator, VkQueue graphicsQueue, VkCommandPool commandPool,
        VkBuffer vertexBuffer, VkBuffer indexBuffer,
        uint32_t totalVertexCount, uint32_t totalIndexCount,
        const core::EntityData* entityData, uint32_t entityCount) {

        Logger::Log(LogLevel::Info, "[GeometryCacheTest] Starting virtual geometry cache round-trip test...");

        std::vector<renderer::Vertex> allVertices;
        std::vector<uint32_t> allIndices;
        if (!ReadbackFullGeometry(device, allocator, graphicsQueue, commandPool, vertexBuffer, indexBuffer,
            totalVertexCount, totalIndexCount, allVertices, allIndices)) {
            Logger::Log(LogLevel::Error, "[GeometryCacheTest] Aborting: full geometry readback failed.");
            return false;
        }

        CacheFileManager cacheManager;
        cacheManager.PurgeExistingCacheFiles(".");

        bool allPassed = true;

        for (uint32_t entityIdx = 0; entityIdx < entityCount; ++entityIdx) {
            uint32_t meshID = entityData[entityIdx].meshID;

            EntityClusterBuildResult built = BuildClustersForEntity(meshID, allVertices, allIndices);
            if (built.pages.empty()) {
                Logger::Log(LogLevel::Warning, std::format(
                    "[GeometryCacheTest] Entity meshID={} produced zero clusters (no matching triangles); skipping.",
                    meshID));
                continue;
            }

            CacheHeader header{};
            header.magic = CacheHeader::kMagic;
            header.version = CacheHeader::kVersion;
            header.pageCount = static_cast<uint32_t>(built.pages.size());
            header.pageSizeBytes = Page::kPageSizeBytes;
            // BuildEntityData() always allocates entity IDs via core::IDManager::Init(0), which
            // zeroes the context's upper 16 bits — so for this engine's single-factory setup,
            // the full 64-bit EntityID and the 32-bit meshID are numerically identical.
            header.entityID = static_cast<uint64_t>(meshID);
            header.meshID = meshID;
            header.materialID = entityData[entityIdx].materialID;
            header.rootClusterIndex = 0; // No DAG hierarchy yet: cluster 0 stands in as the single "root".
            header.totalVertexCount = built.totalVertexCount;
            header.totalTriangleCount = built.totalTriangleCount;
            for (uint32_t p = 0; p < header.pageCount; ++p) {
                header.pageOffsets[p] = static_cast<uint64_t>(p + 1u) * Page::kPageSizeBytes;
            }

            std::filesystem::path filePath = std::filesystem::path(".") / std::format("entity_{}.cache", meshID);

            if (!cacheManager.WriteCacheFile(filePath, header, built.pages)) {
                Logger::Log(LogLevel::Error, std::format("[GeometryCacheTest] WriteCacheFile failed for entity meshID={}.", meshID));
                allPassed = false;
                continue;
            }

            std::error_code sizeEc;
            uint64_t fileSizeBytes = std::filesystem::file_size(filePath, sizeEc);
            uint64_t expectedSizeBytes = static_cast<uint64_t>(1u + built.pages.size()) * Page::kPageSizeBytes;
            bool sizeOk = !sizeEc && fileSizeBytes == expectedSizeBytes && (fileSizeBytes % Page::kPageSizeBytes) == 0;
            if (!sizeOk) {
                Logger::Log(LogLevel::Error, std::format(
                    "[GeometryCacheTest] '{}' size mismatch: got {} bytes, expected {} (must be a multiple of {}).",
                    filePath.string(), fileSizeBytes, expectedSizeBytes, Page::kPageSizeBytes));
            }

            // Read an arbitrary (not just the first) page back to make sure the allocation-table
            // math and unbuffered/overlapped I/O both handle a non-zero page offset correctly.
            uint32_t samplePageIndex = static_cast<uint32_t>(built.pages.size() / 2);
            Page readBackPage{};
            std::future<bool> readFuture = cacheManager.ReadPageAsync(filePath, samplePageIndex, readBackPage);
            bool readOk = readFuture.get();

            bool integrityOk = readOk && std::memcmp(&readBackPage, &built.pages[samplePageIndex], sizeof(Page)) == 0;
            if (!integrityOk) {
                Logger::Log(LogLevel::Error, std::format(
                    "[GeometryCacheTest] '{}' page[{}] failed to round-trip byte-for-byte through disk.",
                    filePath.string(), samplePageIndex));
            }

            Logger::Log((sizeOk && integrityOk) ? LogLevel::Info : LogLevel::Error, std::format(
                "[GeometryCacheTest] Entity meshID={} '{}': {} page(s), {} vertices, {} triangles. "
                "Size check: {}. Page[{}] round-trip integrity: {}.",
                meshID, filePath.string(), built.pages.size(), built.totalVertexCount, built.totalTriangleCount,
                sizeOk ? "OK" : "FAIL", samplePageIndex, integrityOk ? "OK" : "FAIL"));

            allPassed = allPassed && sizeOk && integrityOk;
        }

        Logger::Log(allPassed ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Virtual geometry cache round-trip test {}.", allPassed ? "PASSED" : "FAILED"));

        return allPassed;
    }

}
