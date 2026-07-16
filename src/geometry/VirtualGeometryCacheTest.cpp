#include "geometry/VirtualGeometryCacheTest.h"

#include "io/CacheFileManager.h"
#include "geometry/ClusterDAG.h"
#include "geometry/ClusterFormat.h"
#include "geometry/EntityMaterialTable.h"
#include "geometry/CardGenerator.h"
#include "geometry/FallbackMeshBuilder.h"
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
#include <string>
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
                LOG_ERROR("[GeometryCacheTest] Failed to allocate vertex staging buffer for full readback!");
                return false;
            }

            VkBuffer stagingIndex = VK_NULL_HANDLE;
            VmaAllocation stagingIndexAlloc = VK_NULL_HANDLE;
            VkBufferCreateInfo iInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            iInfo.size = indexBytes;
            iInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vmaCreateBuffer(allocator, &iInfo, &stagingAllocInfo, &stagingIndex, &stagingIndexAlloc, nullptr) != VK_SUCCESS) {
                LOG_ERROR("[GeometryCacheTest] Failed to allocate index staging buffer for full readback!");
                vmaDestroyBuffer(allocator, stagingVertex, stagingVertexAlloc);
                return false;
            }

            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd) != VK_SUCCESS) {
                LOG_ERROR("[GeometryCacheTest] Failed to allocate the full-readback command buffer!");
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
                LOG_ERROR("[GeometryCacheTest] Failed to map vertex staging buffer!");
                ok = false;
            }

            void* mappedIndices = nullptr;
            if (ok && vmaMapMemory(allocator, stagingIndexAlloc, &mappedIndices) == VK_SUCCESS) {
                std::memcpy(outIndices.data(), mappedIndices, static_cast<size_t>(indexBytes));
                vmaUnmapMemory(allocator, stagingIndexAlloc);
            }
            else if (ok) {
                LOG_ERROR("[GeometryCacheTest] Failed to map index staging buffer!");
                ok = false;
            }

            vmaDestroyBuffer(allocator, stagingVertex, stagingVertexAlloc);
            vmaDestroyBuffer(allocator, stagingIndex, stagingIndexAlloc);
            return ok;
        }

        // -------------------------------------------------------------------------------------
        // Computes area-weighted per-vertex normals directly from a DAG node's own triangle
        // geometry. Needed because SimplifiableMesh (ClusterGrouping/MeshSimplifier/ClusterDAG's
        // shared working type) only tracks vertex positions/UVs, not normals -- carrying real
        // per-vertex normals through cluster grouping and QEM simplification would need an
        // attribute-aware quadric (à la Hoppe), a separate, larger piece of work than this
        // .cache-format pass. Recomputing a face-derived normal here is correct and complete for
        // every DAG level (leaf or simplified). UV, unlike normals, IS carried through grouping/
        // simplification as a genuine mesh attribute (SimplifiableMesh::uvs, midpoint-blended
        // across each QEM collapse) -- see EncodeClusterData below, which reads node.mesh.uvs
        // directly instead of needing a geometric fallback.
        // -------------------------------------------------------------------------------------
        std::vector<maths::vec3> ComputeVertexNormals(const SimplifiableMesh& mesh) {
            std::vector<maths::vec3> normals(mesh.positions.size(), maths::vec3{ 0.0f, 0.0f, 0.0f });
            for (size_t t = 0; t + 2 < mesh.triangles.size(); t += 3) {
                uint32_t i0 = mesh.triangles[t + 0];
                uint32_t i1 = mesh.triangles[t + 1];
                uint32_t i2 = mesh.triangles[t + 2];
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

        // Encodes one DAG node's geometry into a fixed-size, quantized ClusterData block. Returns
        // false (after logging why) if the node's vertex/index count would overflow the
        // fixed-size on-disk arrays: SimplifyMeshQEM only targets a triangle-count budget today
        // (see MeshSimplifier.h), so a pathological group can in principle end up with more
        // surviving vertices than kMaxClusterVertices allows. Refusing to encode it -- rather than
        // silently truncating or overrunning the fixed arrays -- is the explicit-failure contract
        // this fixed-capacity format requires.
        bool EncodeClusterData(const ClusterDAGNode& node, ClusterData& outData) {
            uint32_t vertexCount = static_cast<uint32_t>(node.mesh.positions.size());
            uint32_t indexCount = static_cast<uint32_t>(node.mesh.triangles.size());
            if (vertexCount > kMaxClusterVertices || indexCount > kMaxClusterIndices) {
                LOG_ERROR(std::format(
                    "[GeometryCacheTest] DAG node (level {}) exceeds the fixed cluster capacity: "
                    "{} vertices (max {}), {} indices (max {}).",
                    node.level, vertexCount, kMaxClusterVertices, indexCount, kMaxClusterIndices));
                return false;
            }

            std::vector<maths::vec3> normals = ComputeVertexNormals(node.mesh);

            outData = ClusterData{};
            for (uint32_t v = 0; v < vertexCount; ++v) {
                outData.positions[v] = QuantizePosition(node.mesh.positions[v], node.boundsMin, node.boundsMax);
                outData.normals[v] = EncodeOctNormal24(normals[v]);
                outData.uvs[v] = EncodeUV(node.mesh.uvs[v]);
            }
            for (uint32_t i = 0; i < indexCount; ++i) {
                outData.indices[i] = static_cast<uint8_t>(node.mesh.triangles[i]);
            }
            return true;
        }

        // Builds the cluster index table entry for one DAG node: bounds/sphere/cone metadata for
        // coarse CPU/GPU-side culling, kept resident even while the node's own geometry block may
        // not be streamed in. virtualAddress/blockSizeBytes are left zeroed -- CacheFileManager::
        // WriteCacheFile fills them in once it has computed the file's physical layout.
        // maxWPOAmplitude/maskTextureIndex come from the owning entity's GetEntityMaterialProperties
        // lookup (see the call site below) and are stamped identically into every DAG level of a
        // given entity, exactly like entityID itself. materialID is the raw core::EntityData::
        // materialID the caller looked maxWPOAmplitude/maskTextureIndex up with -- stamped verbatim
        // (no further resolution needed here), same "identical across every DAG level" contract.
        ClusterIndexEntry BuildIndexEntry(uint32_t globalClusterID, uint32_t entityID, const ClusterDAGNode& node,
            float maxWPOAmplitude, uint32_t maskTextureIndex, uint32_t materialID) {
            // The cone must bound each triangle's flat FACE normal -- what IsClusterBackFacing
            // (cluster_culling_tests.glsl) actually tests against -- not smoothed per-vertex
            // normals. Vertex normals average together neighboring face normals, which narrows
            // the apparent spread and pushes coneCutoff higher (more aggressive) than the true
            // bound, wrongly rejecting still-front-facing clusters on high-curvature/twisted
            // geometry (torus knot) or near hard edges/tips (cone's cap rim and apex). Degenerate
            // (near-zero-area) triangles -- e.g. geom_cone.comp/geom_cylinder.comp's innermost
            // cap-ring index padding -- are skipped since they have no well-defined orientation
            // and would otherwise skew the bound toward an artificially wide (but harmless) cone.
            std::vector<maths::vec3> faceNormals;
            faceNormals.reserve(node.mesh.triangles.size() / 3);
            for (size_t t = 0; t + 2 < node.mesh.triangles.size(); t += 3) {
                uint32_t i0 = node.mesh.triangles[t + 0];
                uint32_t i1 = node.mesh.triangles[t + 1];
                uint32_t i2 = node.mesh.triangles[t + 2];
                maths::vec3 rawNormal = (node.mesh.positions[i1] - node.mesh.positions[i0]).Cross(node.mesh.positions[i2] - node.mesh.positions[i0]);
                if (rawNormal.Length() < 1.0e-12f) {
                    continue;
                }
                faceNormals.push_back(rawNormal.Normalize());
            }

            maths::vec3 axisAccum{ 0.0f, 0.0f, 0.0f };
            for (const maths::vec3& n : faceNormals) {
                axisAccum = axisAccum + n;
            }
            maths::vec3 coneAxis = faceNormals.empty() ? maths::vec3{ 0.0f, 1.0f, 0.0f } : axisAccum.Normalize();
            float coneCutoff = 1.0f;
            for (const maths::vec3& n : faceNormals) {
                coneCutoff = std::min(coneCutoff, coneAxis.Dot(n));
            }
            coneCutoff = std::clamp(coneCutoff, -1.0f, 1.0f);

            ClusterIndexEntry entry{};
            entry.clusterID = globalClusterID;
            entry.entityID = entityID;
            entry.virtualAddress = 0;
            entry.blockSizeBytes = 0;
            entry.vertexCount = static_cast<uint32_t>(node.mesh.positions.size());
            entry.indexCount = static_cast<uint32_t>(node.mesh.triangles.size());
            entry.boundsMin[0] = node.boundsMin.x; entry.boundsMin[1] = node.boundsMin.y; entry.boundsMin[2] = node.boundsMin.z;
            entry.boundsMax[0] = node.boundsMax.x; entry.boundsMax[1] = node.boundsMax.y; entry.boundsMax[2] = node.boundsMax.z;
            entry.sphereCenter[0] = node.sphereCenter.x; entry.sphereCenter[1] = node.sphereCenter.y; entry.sphereCenter[2] = node.sphereCenter.z;
            entry.sphereRadius = node.sphereRadius;
            entry.coneAxisX = static_cast<int8_t>(std::lround(coneAxis.x * 127.0f));
            entry.coneAxisY = static_cast<int8_t>(std::lround(coneAxis.y * 127.0f));
            entry.coneAxisZ = static_cast<int8_t>(std::lround(coneAxis.z * 127.0f));
            entry.coneCutoff = static_cast<int8_t>(std::lround(coneCutoff * 127.0f));
            entry.maxWPOAmplitude = maxWPOAmplitude;
            entry.maskTextureIndex = maskTextureIndex;
            entry.materialID = materialID;
            return entry;
        }

        // Converts a geometry::FallbackMesh into an on-disk-ready CacheFileManager::FallbackMeshData:
        // full-precision FallbackVertex array, flattened triangle indices, and a freshly computed
        // AABB. vertexDataOffset/indexDataOffset are left zeroed -- CacheFileManager::WriteCacheFile
        // fills them in once it has computed the file's physical layout (mirrors BuildIndexEntry's
        // own virtualAddress/blockSizeBytes convention above).
        FallbackMeshData BuildFallbackMeshData(uint32_t entityID, const FallbackMesh& mesh) {
            FallbackMeshData data;
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

                FallbackVertex fv{};
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

        // Reconstructs an (geometry-less) in-memory ClusterDAG purely from an on-disk DAG table,
        // so ValidateClusterDAG can be re-run *after* a write+read round trip -- proving the
        // on-disk bytes, not just the pre-write in-memory structure, encode a structurally and
        // error-monotonically valid DAG. Relies on this file's own invariant that clusterID is
        // always assigned as a simple 0-based running index matching array position exactly (see
        // the clusterID assignment loop in RunVirtualGeometryCacheTest below), so
        // parentClusterID/childClusterID values translate directly into node array indices.
        ClusterDAG ReconstructDAGFromTable(const std::vector<DAGNodeEntry>& entries) {
            ClusterDAG dag;
            dag.nodes.resize(entries.size());
            for (size_t i = 0; i < entries.size(); ++i) {
                const DAGNodeEntry& entry = entries[i];
                ClusterDAGNode& node = dag.nodes[i];
                node.level = entry.level;
                node.clusterError = entry.clusterError;
                node.parentError = entry.parentError;
                node.parentIndex = (entry.parentClusterID == kInvalidClusterID) ? kInvalidDAGNodeIndex : entry.parentClusterID;
                for (uint32_t childID : entry.childClusterID) {
                    if (childID != kInvalidClusterID) {
                        node.childIndices.push_back(childID);
                    }
                }
                if (node.parentIndex == kInvalidDAGNodeIndex) {
                    dag.rootIndices.push_back(static_cast<uint32_t>(i));
                }
            }
            return dag;
        }

    } // namespace

    bool RunVirtualGeometryCacheTest(
        VkDevice device, VmaAllocator allocator, VkQueue graphicsQueue, VkCommandPool commandPool,
        VkBuffer vertexBuffer, VkBuffer indexBuffer,
        uint32_t totalVertexCount, uint32_t totalIndexCount,
        const core::EntityData* entityData, uint32_t entityCount) {

        LOG_INFO("[GeometryCacheTest] Starting virtual geometry cache round-trip test...");

        std::vector<renderer::Vertex> allVertices;
        std::vector<uint32_t> allIndices;
        if (!ReadbackFullGeometry(device, allocator, graphicsQueue, commandPool, vertexBuffer, indexBuffer,
            totalVertexCount, totalIndexCount, allVertices, allIndices)) {
            LOG_ERROR("[GeometryCacheTest] Aborting: full geometry readback failed.");
            return false;
        }

        // --- DIAGNOSTIC: per-meshID vertex/triangle histogram -------------------------------
        // Temporary instrumentation to determine whether missing geometry (e.g. cone/torus
        // knot reported as "produced zero clusters") is caused by vertices never landing in
        // the readback with the expected meshID (a generation-side bug) or by vertices being
        // present but not matched by ClusterPartitioner's index-triangle filter (a
        // partitioning-side bug). Mirrors PartitionMeshIntoClusters's exact filter
        // (allVertices[i0].meshID) so the triangle counts reproduce what it sees.
        {
            std::unordered_map<uint32_t, uint32_t> vertsByMeshID;
            for (const renderer::Vertex& v : allVertices) {
                ++vertsByMeshID[v.meshID];
            }
            std::unordered_map<uint32_t, uint32_t> trisByMeshID;
            for (size_t t = 0; t + 2 < allIndices.size(); t += 3) {
                uint32_t i0 = allIndices[t + 0];
                ++trisByMeshID[allVertices[i0].meshID];
            }
            LOG_INFO("[GeometryCacheTest] Per-meshID diagnostic histogram:");
            for (uint32_t id = 0; id < entityCount; ++id) {
                uint32_t vc = vertsByMeshID.count(id) ? vertsByMeshID[id] : 0u;
                uint32_t tc = trisByMeshID.count(id) ? trisByMeshID[id] : 0u;
                LOG_INFO(std::format(
                    "[GeometryCacheTest]   meshID={}: vertices={} triangles={}", id, vc, tc));
            }
            for (const auto& [id, vc] : vertsByMeshID) {
                if (id >= entityCount) {
                    LOG_WARNING(std::format(
                        "[GeometryCacheTest]   OUT-OF-RANGE meshID={} found on {} vertices (expected range [0,{}))! "
                        "This indicates memory corruption (likely a buffer capacity overflow).", id, vc, entityCount));
                }
            }
        }

        CacheFileManager cacheManager;
        cacheManager.PurgeExistingCacheFiles(".");

        // --- Build every entity's DAG and flatten them into one global, index-aligned set -------
        std::vector<ClusterIndexEntry> indexEntries;
        std::vector<DAGNodeEntry> dagEntries;
        std::vector<ClusterData> clusterData;
        std::vector<FallbackMeshData> fallbackMeshes;
        std::vector<SurfaceCacheCardEntry> cardEntries;

        bool allEntitiesOk = true;
        uint32_t entitiesWithGeometry = 0;

        for (uint32_t entityIdx = 0; entityIdx < entityCount; ++entityIdx) {
            uint32_t meshID = entityData[entityIdx].meshID;
            EntityMaterialProperties materialProps = GetEntityMaterialProperties(entityData[entityIdx].materialID);

            ClusterDAG dag = BuildClusterDAG(meshID, allVertices, allIndices, materialProps.maskTextureIndex);
            if (dag.nodes.empty()) {
                LOG_WARNING(std::format(
                    "[GeometryCacheTest] Entity meshID={} produced zero clusters (no matching triangles); skipping.", meshID));
                continue;
            }
            ++entitiesWithGeometry;

            // Validate the in-memory DAG *before* persisting it, so a builder bug is caught here
            // rather than silently baked into the .cache file.
            std::vector<std::string> dagErrors;
            if (!ValidateClusterDAG(dag, dagErrors)) {
                for (const std::string& err : dagErrors) {
                    LOG_ERROR(std::format("[GeometryCacheTest] entity meshID={} DAG: {}", meshID, err));
                }
                allEntitiesOk = false;
                continue;
            }

            // Fallback Mesh: one coarse BVH proxy for this entity, built by continuing DAG-root
            // simplification with every vertex unlocked (see FallbackMeshBuilder.h). Deliberately
            // independent of the opaque/masked cluster split above -- it is a ray tracing
            // acceleration-structure input, not a rasterization path.
            FallbackMesh fallbackMesh = BuildFallbackMesh(dag);
            if (!fallbackMesh.triangles.empty()) {
                fallbackMeshes.push_back(BuildFallbackMeshData(meshID, fallbackMesh));

                // Surface-cache cards: up to 6 orthographic box-face projections of this entity,
                // sized from the SAME AABB the fallback mesh's index entry carries -- the capture
                // pass rasterizes exactly that mesh through these cards, so using its bounds (not
                // the leaf geometry's) guarantees the projection volume encloses everything the
                // capture will ever draw. Atlas placement is deferred to the global pack below.
                const FallbackMeshIndexEntry& fbEntry = fallbackMeshes.back().indexEntry;
                std::vector<SurfaceCacheCardEntry> entityCards = GenerateEntityCards(meshID,
                    maths::vec3{ fbEntry.boundsMin[0], fbEntry.boundsMin[1], fbEntry.boundsMin[2] },
                    maths::vec3{ fbEntry.boundsMax[0], fbEntry.boundsMax[1], fbEntry.boundsMax[2] });
                cardEntries.insert(cardEntries.end(), entityCards.begin(), entityCards.end());
                LOG_INFO(std::format(
                    "[GeometryCacheTest] entity meshID={}: {} surface-cache card(s) generated.",
                    meshID, entityCards.size()));

                size_t leafTriangleCount = 0;
                for (const ClusterDAGNode& n : dag.nodes) {
                    if (n.level == 0u) {
                        leafTriangleCount += n.mesh.triangles.size() / 3u;
                    }
                }
                size_t fallbackTriangleCount = fallbackMesh.triangles.size() / 3u;
                LOG_INFO(std::format(
                    "[GeometryCacheTest] entity meshID={}: fallback mesh {} triangle(s) ({:.2f}% of {} leaf triangle(s)).",
                    meshID, fallbackTriangleCount,
                    100.0f * static_cast<float>(fallbackTriangleCount) / static_cast<float>(std::max<size_t>(1, leafTriangleCount)),
                    leafTriangleCount));
            }

            uint32_t baseGlobalID = static_cast<uint32_t>(indexEntries.size());
            std::vector<uint32_t> localToGlobal(dag.nodes.size());
            for (size_t i = 0; i < dag.nodes.size(); ++i) {
                localToGlobal[i] = baseGlobalID + static_cast<uint32_t>(i);
            }

            for (size_t i = 0; i < dag.nodes.size(); ++i) {
                const ClusterDAGNode& node = dag.nodes[i];
                uint32_t globalID = localToGlobal[i];

                ClusterData data{};
                if (!EncodeClusterData(node, data)) {
                    allEntitiesOk = false; // Logged inside EncodeClusterData; keep bookkeeping aligned below.
                }
                clusterData.push_back(data);
                // Per-cluster, not per-entity: a cluster only carries the entity's real mask slot
                // if BuildClusterDAG's opacity split actually classified it as masked (node.isMasked)
                // -- an opaque cluster gets kInvalidMaskTextureIndex even if its owning entity has a
                // cutout material, which is exactly what lets the (Part 2) opaque rasterizer path
                // route it with zero mask-sampling overhead.
                indexEntries.push_back(BuildIndexEntry(globalID, meshID, node, materialProps.maxWPOAmplitude,
                    node.isMasked ? materialProps.maskTextureIndex : kInvalidMaskTextureIndex,
                    entityData[entityIdx].materialID));

                DAGNodeEntry dagEntry{};
                dagEntry.clusterID = globalID;
                dagEntry.parentClusterID = (node.parentIndex == kInvalidDAGNodeIndex) ? kInvalidClusterID : localToGlobal[node.parentIndex];
                dagEntry.childClusterID[0] = (node.childIndices.size() > 0) ? localToGlobal[node.childIndices[0]] : kInvalidClusterID;
                dagEntry.childClusterID[1] = (node.childIndices.size() > 1) ? localToGlobal[node.childIndices[1]] : kInvalidClusterID;
                dagEntry.level = node.level;
                dagEntry.clusterError = node.clusterError;
                dagEntry.parentError = node.parentError;
                dagEntries.push_back(dagEntry);
            }
        }

        // --- DIAGNOSTIC: per-entity coneCutoff range (sanity check for the backface-cone fix) ----
        // coneCutoff close to +1.0 means a very narrow/aggressive cone (culls unless the camera is
        // almost perfectly aligned with the axis); close to -1.0 means a near-omnidirectional cone
        // (rarely culls). High-curvature/twisted meshes (torus knot) and hard-edged tips (cone)
        // should show low (wide) cutoffs -- a value hovering near 1.0 for those would indicate the
        // per-cluster normal cone is still too narrow and would over-cull.
        {
            std::unordered_map<uint32_t, float> minCutoffByEntity;
            std::unordered_map<uint32_t, float> maxCutoffByEntity;
            std::unordered_map<uint32_t, uint32_t> countByEntity;
            for (const ClusterIndexEntry& e : indexEntries) {
                float cutoff = static_cast<float>(e.coneCutoff) / 127.0f;
                auto itMin = minCutoffByEntity.find(e.entityID);
                if (itMin == minCutoffByEntity.end() || cutoff < itMin->second) minCutoffByEntity[e.entityID] = cutoff;
                auto itMax = maxCutoffByEntity.find(e.entityID);
                if (itMax == maxCutoffByEntity.end() || cutoff > itMax->second) maxCutoffByEntity[e.entityID] = cutoff;
                ++countByEntity[e.entityID];
            }
            LOG_INFO("[GeometryCacheTest] Per-entity coneCutoff range (all DAG levels, not just leaves):");
            for (const auto& [id, count] : countByEntity) {
                LOG_INFO(std::format(
                    "[GeometryCacheTest]   meshID={}: clusters={} coneCutoff min={:.3f} max={:.3f}",
                    id, count, minCutoffByEntity[id], maxCutoffByEntity[id]));
            }
        }

        if (!allEntitiesOk) {
            LOG_ERROR(
                "[GeometryCacheTest] Aborting before write: one or more entities failed DAG validation or cluster encoding (see errors above).");
            return false;
        }
        if (indexEntries.empty()) {
            LOG_ERROR("[GeometryCacheTest] Aborting: no entity produced any cluster.");
            return false;
        }

        // --- Pack every entity's cards into the global surface cache atlas ----------------------
        // A pack failure means the scene's combined card area exceeds the atlas -- a build-time
        // tuning error (kCardTexelsPerWorldUnit vs kSurfaceCacheAtlasSize), fatal by design since
        // the surface cache capture pass cannot run without a complete, non-overlapping mapping.
        if (!PackCardsIntoAtlas(cardEntries)) {
            LOG_ERROR(std::format(
                "[GeometryCacheTest] Aborting: {} surface-cache card(s) do not fit the {}x{} atlas.",
                cardEntries.size(), kSurfaceCacheAtlasSize, kSurfaceCacheAtlasSize));
            return false;
        }
        LOG_INFO(std::format(
            "[GeometryCacheTest] Packed {} surface-cache card(s) into the {}x{} atlas.",
            cardEntries.size(), kSurfaceCacheAtlasSize, kSurfaceCacheAtlasSize));

        // --- Write the whole scene's clusters into ONE consolidated .cache file -----------------
        std::filesystem::path filePath = std::filesystem::path(".") / "scene.cache";
        if (!cacheManager.WriteCacheFile(filePath, indexEntries, dagEntries, clusterData, fallbackMeshes, cardEntries, entitiesWithGeometry)) {
            LOG_ERROR(std::format("[GeometryCacheTest] WriteCacheFile failed for '{}'.", filePath.string()));
            return false;
        }

        // --- Read everything back and verify a byte-exact round trip ----------------------------
        bool allPassed = true;

        CacheFileHeader readHeader{};
        if (!cacheManager.ReadHeader(filePath, readHeader)) {
            LOG_ERROR("[GeometryCacheTest] Failed to read back the CacheFileHeader.");
            return false;
        }
        bool headerOk = readHeader.clusterCount == indexEntries.size() && readHeader.entityCount == entitiesWithGeometry;
        allPassed = allPassed && headerOk;
        LOG(headerOk ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Header round-trip: {} (clusterCount={}, entityCount={}).",
            headerOk ? "OK" : "FAIL", readHeader.clusterCount, readHeader.entityCount));

        std::vector<ClusterIndexEntry> readIndexEntries;
        bool indexTableOk = cacheManager.ReadClusterIndexTable(filePath, readHeader, readIndexEntries)
            && readIndexEntries.size() == indexEntries.size()
            && std::memcmp(readIndexEntries.data(), indexEntries.data(), indexEntries.size() * sizeof(ClusterIndexEntry)) == 0;
        allPassed = allPassed && indexTableOk;
        LOG(indexTableOk ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Cluster index table round-trip: {}.", indexTableOk ? "OK" : "FAIL"));

        std::vector<DAGNodeEntry> readDagEntries;
        bool dagTableOk = cacheManager.ReadDAGTable(filePath, readHeader, readDagEntries)
            && readDagEntries.size() == dagEntries.size()
            && std::memcmp(readDagEntries.data(), dagEntries.data(), dagEntries.size() * sizeof(DAGNodeEntry)) == 0;
        allPassed = allPassed && dagTableOk;
        LOG(dagTableOk ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] DAG table round-trip: {}.", dagTableOk ? "OK" : "FAIL"));

        // Re-validate the DAG *as reconstructed from the on-disk table*, proving the persisted
        // bytes (not just the pre-write in-memory structure) still satisfy every structural and
        // error-monotonicity invariant.
        bool onDiskDagValid = false;
        if (dagTableOk) {
            ClusterDAG reconstructed = ReconstructDAGFromTable(readDagEntries);
            std::vector<std::string> reconstructedErrors;
            onDiskDagValid = ValidateClusterDAG(reconstructed, reconstructedErrors);
            for (const std::string& err : reconstructedErrors) {
                LOG_ERROR(std::format("[GeometryCacheTest] on-disk DAG: {}", err));
            }
        }
        allPassed = allPassed && onDiskDagValid;
        LOG(onDiskDagValid ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] On-disk DAG re-validation: {}.", onDiskDagValid ? "OK" : "FAIL"));

        // Sample one arbitrary (not just the first) cluster's geometry block and verify it
        // decodes byte-exact, exercising the unbuffered/overlapped read path and the page
        // alignment/zero-padding CacheFileManager::WriteCacheFile is responsible for.
        uint32_t sampleClusterIndex = static_cast<uint32_t>(indexEntries.size() / 2);
        bool clusterReadOk = false;
        if (readIndexEntries.size() > sampleClusterIndex) {
            ClusterData readClusterData{};
            std::future<bool> readFuture = cacheManager.ReadClusterDataAsync(
                filePath, readIndexEntries[sampleClusterIndex].virtualAddress, readClusterData);
            clusterReadOk = readFuture.get()
                && std::memcmp(&readClusterData, &clusterData[sampleClusterIndex], sizeof(ClusterData)) == 0;
        }
        allPassed = allPassed && clusterReadOk;
        LOG(clusterReadOk ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Sample cluster [{}] geometry block round-trip: {}.", sampleClusterIndex, clusterReadOk ? "OK" : "FAIL"));

        // Fallback-mesh section round trip: table + one entity's full vertex/index geometry,
        // byte-exact against the in-memory data WriteCacheFile was handed (whose indexEntry
        // offsets it filled in place, so fallbackMeshes[i].indexEntry is already the on-disk truth).
        std::vector<FallbackMeshIndexEntry> readFallbackTable;
        std::vector<FallbackMeshIndexEntry> expectedFallbackTable;
        expectedFallbackTable.reserve(fallbackMeshes.size());
        for (const FallbackMeshData& f : fallbackMeshes) {
            expectedFallbackTable.push_back(f.indexEntry);
        }
        bool fallbackTableOk = cacheManager.ReadFallbackMeshTable(filePath, readHeader, readFallbackTable)
            && readFallbackTable.size() == expectedFallbackTable.size()
            && (expectedFallbackTable.empty() || std::memcmp(readFallbackTable.data(), expectedFallbackTable.data(),
                expectedFallbackTable.size() * sizeof(FallbackMeshIndexEntry)) == 0);
        allPassed = allPassed && fallbackTableOk;
        LOG(fallbackTableOk ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Fallback-mesh index table round-trip: {} ({} entit(y/ies)).",
            fallbackTableOk ? "OK" : "FAIL", expectedFallbackTable.size()));

        bool fallbackGeometryOk = true;
        if (!fallbackMeshes.empty()) {
            uint32_t sampleFallbackIndex = static_cast<uint32_t>(fallbackMeshes.size() / 2);
            const FallbackMeshData& expected = fallbackMeshes[sampleFallbackIndex];
            std::vector<FallbackVertex> readVertices;
            std::vector<uint32_t> readIndices;
            fallbackGeometryOk = cacheManager.ReadFallbackMeshGeometry(filePath, expected.indexEntry, readVertices, readIndices)
                && readVertices.size() == expected.vertices.size()
                && readIndices.size() == expected.indices.size()
                && (expected.vertices.empty() || std::memcmp(readVertices.data(), expected.vertices.data(), expected.vertices.size() * sizeof(FallbackVertex)) == 0)
                && (expected.indices.empty() || std::memcmp(readIndices.data(), expected.indices.data(), expected.indices.size() * sizeof(uint32_t)) == 0);
        }
        allPassed = allPassed && fallbackGeometryOk;
        LOG(fallbackGeometryOk ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Sample fallback mesh geometry round-trip: {}.", fallbackGeometryOk ? "OK" : "FAIL"));

        // Surface-cache card table round trip: byte-exact against the packed in-memory table,
        // plus a re-verification that the ON-DISK rects are pairwise non-overlapping (gutter
        // included) -- the invariant every GI sampler depends on, proven on the persisted bytes
        // exactly like the DAG's own post-round-trip re-validation above.
        std::vector<SurfaceCacheCardEntry> readCardTable;
        bool cardTableOk = cacheManager.ReadSurfaceCacheCardTable(filePath, readHeader, readCardTable)
            && readCardTable.size() == cardEntries.size()
            && (cardEntries.empty() || std::memcmp(readCardTable.data(), cardEntries.data(),
                cardEntries.size() * sizeof(SurfaceCacheCardEntry)) == 0);
        if (cardTableOk) {
            for (size_t a = 0; a < readCardTable.size() && cardTableOk; ++a) {
                for (size_t b = a + 1; b < readCardTable.size() && cardTableOk; ++b) {
                    const SurfaceCacheCardEntry& ca = readCardTable[a];
                    const SurfaceCacheCardEntry& cb = readCardTable[b];
                    // Standard AABB separation test in atlas texel space: overlap iff neither
                    // rect is fully left of / above the other.
                    bool overlap =
                        ca.atlasOffset[0] < cb.atlasOffset[0] + cb.atlasSize[0] &&
                        cb.atlasOffset[0] < ca.atlasOffset[0] + ca.atlasSize[0] &&
                        ca.atlasOffset[1] < cb.atlasOffset[1] + cb.atlasSize[1] &&
                        cb.atlasOffset[1] < ca.atlasOffset[1] + ca.atlasSize[1];
                    if (overlap) {
                        LOG_ERROR(std::format(
                            "[GeometryCacheTest] On-disk cards {} and {} overlap in the atlas!", a, b));
                        cardTableOk = false;
                    }
                }
            }
        }
        allPassed = allPassed && cardTableOk;
        LOG(cardTableOk ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Surface-cache card table round-trip + non-overlap: {} ({} card(s)).",
            cardTableOk ? "OK" : "FAIL", cardEntries.size()));

        LOG(allPassed ? LogLevel::Info : LogLevel::Error, std::format(
            "[GeometryCacheTest] Virtual geometry cache round-trip test {} -- '{}': {} cluster(s) across {} entit(y/ies).",
            allPassed ? "PASSED" : "FAILED", filePath.string(), indexEntries.size(), entitiesWithGeometry));

        return allPassed;
    }

}
