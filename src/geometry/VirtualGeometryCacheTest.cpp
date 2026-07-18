#include "geometry/VirtualGeometryCacheTest.h"

#include "io/CacheFileManager.h"
#include "geometry/ClusterDAG.h"
#include "geometry/ClusterFormat.h"
#include "geometry/EntityMaterialTable.h"
#include "geometry/CardGenerator.h"
#include "geometry/FallbackMeshBuilder.h"
#include "geometry/GeometryEncoding.h"
#include "core/Logger.h"
#include "core/EngineConfig.h"
#include "core/LoadingManager.h"
#include "renderer/RenderTypes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <limits>
#include <map>
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
            VkBuffer vertexBuffer, VkBuffer indexBuffer, VkBuffer vertexSkinBuffer,
            uint32_t totalVertexCount, uint32_t totalIndexCount,
            std::vector<renderer::Vertex>& outVertices, std::vector<uint32_t>& outIndices,
            std::vector<ClusterVertexSkin>& outSkins) {

            const VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(totalVertexCount) * sizeof(renderer::Vertex);
            const VkDeviceSize indexBytes = static_cast<VkDeviceSize>(totalIndexCount) * sizeof(uint32_t);
            // Skeletal-animation feature: mirrors vertexBytes' own sizing (same totalVertexCount,
            // index-aligned 1:1 with outVertices) -- see VulkanContext::GetVertexSkinBuffer()'s own
            // comment for why this buffer exists at full scene scale even though only the creature
            // entity's own vertex range is ever meaningfully written by geom_creature.comp.
            const VkDeviceSize skinBytes = static_cast<VkDeviceSize>(totalVertexCount) * sizeof(ClusterVertexSkin);

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

            VkBuffer stagingSkin = VK_NULL_HANDLE;
            VmaAllocation stagingSkinAlloc = VK_NULL_HANDLE;
            VkBufferCreateInfo sInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            sInfo.size = skinBytes;
            sInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vmaCreateBuffer(allocator, &sInfo, &stagingAllocInfo, &stagingSkin, &stagingSkinAlloc, nullptr) != VK_SUCCESS) {
                LOG_ERROR("[GeometryCacheTest] Failed to allocate vertex-skin staging buffer for full readback!");
                vmaDestroyBuffer(allocator, stagingVertex, stagingVertexAlloc);
                vmaDestroyBuffer(allocator, stagingIndex, stagingIndexAlloc);
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
                vmaDestroyBuffer(allocator, stagingSkin, stagingSkinAlloc);
                return false;
            }

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkBufferCopy vertexCopyRegion{ 0, 0, vertexBytes };
            vkCmdCopyBuffer(cmd, vertexBuffer, stagingVertex, 1, &vertexCopyRegion);
            VkBufferCopy indexCopyRegion{ 0, 0, indexBytes };
            vkCmdCopyBuffer(cmd, indexBuffer, stagingIndex, 1, &indexCopyRegion);
            VkBufferCopy skinCopyRegion{ 0, 0, skinBytes };
            vkCmdCopyBuffer(cmd, vertexSkinBuffer, stagingSkin, 1, &skinCopyRegion);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphicsQueue);
            vkFreeCommandBuffers(device, commandPool, 1, &cmd);

            outVertices.resize(totalVertexCount);
            outIndices.resize(totalIndexCount);
            outSkins.resize(totalVertexCount);

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

            void* mappedSkins = nullptr;
            if (ok && vmaMapMemory(allocator, stagingSkinAlloc, &mappedSkins) == VK_SUCCESS) {
                std::memcpy(outSkins.data(), mappedSkins, static_cast<size_t>(skinBytes));
                vmaUnmapMemory(allocator, stagingSkinAlloc);
            }
            else if (ok) {
                LOG_ERROR("[GeometryCacheTest] Failed to map vertex-skin staging buffer!");
                ok = false;
            }

            vmaDestroyBuffer(allocator, stagingVertex, stagingVertexAlloc);
            vmaDestroyBuffer(allocator, stagingIndex, stagingIndexAlloc);
            vmaDestroyBuffer(allocator, stagingSkin, stagingSkinAlloc);
            return ok;
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

            std::vector<maths::vec3> normals = ComputeFaceAccumulatedNormals(node.mesh);

            // Skeletal-animation feature: only ever non-empty for a forceSingleLevelDAG (level-0
            // only) entity's leaf nodes -- see SimplifiableMesh::boneIndices' own comment. Size is
            // checked (not just non-empty) since it must match vertexCount exactly to index safely
            // below; ClusterDAG.cpp's leaf loop always populates both in lockstep with positions/
            // uvs when allVertexSkins was supplied, so a mismatch here would indicate a logic bug
            // upstream, not a legitimate "partially populated" state.
            bool hasSkin = node.mesh.boneIndices.size() == vertexCount && node.mesh.boneWeights.size() == vertexCount;

            outData = ClusterData{};
            for (uint32_t v = 0; v < vertexCount; ++v) {
                outData.positions[v] = QuantizePosition(node.mesh.positions[v], node.boundsMin, node.boundsMax);
                outData.normals[v] = EncodeOctNormal24(normals[v]);
                outData.uvs[v] = EncodeUV(node.mesh.uvs[v]);
                if (hasSkin) {
                    const std::array<uint8_t, 4>& idx = node.mesh.boneIndices[v];
                    const std::array<uint8_t, 4>& w = node.mesh.boneWeights[v];
                    outData.skin[v] = ClusterVertexSkin{
                        { idx[0], idx[1], idx[2], idx[3] },
                        { w[0], w[1], w[2], w[3] }
                    };
                }
                // else: outData.skin[v] stays zero-initialized (ClusterData{} above), the documented
                // "no skinning influence" state for every non-skeletally-animated cluster.
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

            maths::vec3 boundsMin, boundsMax;
            maths::ResetAABB(boundsMin, boundsMax);

            data.vertices.reserve(mesh.positions.size());
            for (size_t v = 0; v < mesh.positions.size(); ++v) {
                const maths::vec3& p = mesh.positions[v];
                maths::ExpandAABB(boundsMin, boundsMax, p);

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
        //
        // The on-disk format has no separate group table (see ClusterFormat.h's DAGNodeEntry
        // comment: every member bakes its parent group's groupError/groupSphereCenter directly),
        // so one ClusterDAGGroup per distinct source group is re-derived here by canonicalizing on
        // its (sorted) child-index set: every one of a group's (1 or kMaxGroupOutputClusters)
        // output entries carries byte-identical childClusterID content, since they were all split
        // from the same merged+simplified mesh, so two output entries sharing that exact set can
        // only have come from the same group.
        ClusterDAG ReconstructDAGFromTable(const std::vector<DAGNodeEntry>& entries) {
            ClusterDAG dag;
            dag.nodes.resize(entries.size());

            std::map<std::vector<uint32_t>, uint32_t> childSetToGroupIndex;

            for (size_t i = 0; i < entries.size(); ++i) {
                const DAGNodeEntry& entry = entries[i];
                ClusterDAGNode& node = dag.nodes[i];
                node.level = entry.level;
                node.clusterError = entry.clusterError;
                node.parentError = entry.parentError;

                bool isRoot = true;
                for (uint32_t parentID : entry.parentClusterID) {
                    if (parentID != kInvalidClusterID) { isRoot = false; break; }
                }
                if (isRoot) {
                    dag.rootIndices.push_back(static_cast<uint32_t>(i));
                }

                std::vector<uint32_t> children;
                for (uint32_t childID : entry.childClusterID) {
                    if (childID != kInvalidClusterID) {
                        children.push_back(childID);
                    }
                }
                if (children.empty()) {
                    continue; // Leaf: no source group.
                }
                std::sort(children.begin(), children.end());

                auto it = childSetToGroupIndex.find(children);
                uint32_t groupIndex;
                if (it != childSetToGroupIndex.end()) {
                    groupIndex = it->second;
                }
                else {
                    // Every member of this group already bakes the shared groupError/
                    // groupSphereCenter as its own parentError/parentSphereCenter (see
                    // ClusterDAGGroup's own comment) -- read it back from the first member instead
                    // of this output entry's own fields, which describe a DIFFERENT (this node's
                    // own future) parent group, not this one.
                    const DAGNodeEntry& firstMember = entries[children[0]];
                    groupIndex = static_cast<uint32_t>(dag.groups.size());
                    ClusterDAGGroup group;
                    group.memberClusterIndices = children;
                    group.groupError = firstMember.parentError;
                    group.groupSphereCenter = maths::vec3{
                        firstMember.parentSphereCenter[0], firstMember.parentSphereCenter[1], firstMember.parentSphereCenter[2] };
                    dag.groups.push_back(std::move(group));
                    childSetToGroupIndex.emplace(children, groupIndex);
                }

                node.sourceGroupIndex = groupIndex;
                dag.groups[groupIndex].outputClusterIndices.push_back(static_cast<uint32_t>(i));
                for (uint32_t childID : children) {
                    dag.nodes[childID].parentGroupIndex = groupIndex;
                }
            }

            return dag;
        }

    } // namespace

    bool IsCacheUpToDate(
        uint32_t totalVertexCount,
        uint32_t totalIndexCount,
        uint32_t entityCount) {
        std::ifstream cfgFile("scene.cache.cfg");
        if (!cfgFile.is_open()) {
            return false;
        }
        if (!std::filesystem::exists("scene.cache")) {
            return false;
        }
        float spacing = 0.0f;
        uint32_t vCount = 0;
        uint32_t iCount = 0;
        uint32_t eCount = 0;
        std::string profileName;
        uint32_t generationVersion = 0;
        // generationVersion is read last and defaults to 0 (never matches a real
        // kGeometryGenerationVersion, which starts at 1) if the field is missing entirely -- so a
        // scene.cache.cfg written by an older build of this engine (before this field existed) is
        // correctly treated as stale rather than crashing on a short read.
        if (!(cfgFile >> spacing >> vCount >> iCount >> eCount >> profileName >> generationVersion)) {
            return false;
        }
        if (std::abs(spacing - config::VERTEX_SPACING) > 1e-5f) {
            return false;
        }
        if (vCount != totalVertexCount) {
            return false;
        }
        if (iCount != totalIndexCount) {
            return false;
        }
        if (eCount != entityCount) {
            return false;
        }
        if (profileName != config::g_ActiveProfileName) {
            return false;
        }
        if (generationVersion != kGeometryGenerationVersion) {
            return false;
        }
        return true;
    }

    void SaveCacheConfig(
        uint32_t totalVertexCount,
        uint32_t totalIndexCount,
        uint32_t entityCount) {
        std::ofstream cfgFile("scene.cache.cfg");
        if (cfgFile.is_open()) {
            cfgFile << config::VERTEX_SPACING << " "
                    << totalVertexCount << " "
                    << totalIndexCount << " "
                    << entityCount << " "
                    << config::g_ActiveProfileName << " "
                    << kGeometryGenerationVersion << "\n";
        }
    }

    bool RunVirtualGeometryCacheTest(
        VkDevice device, VmaAllocator allocator, VkQueue graphicsQueue, VkCommandPool commandPool,
        VkBuffer vertexBuffer, VkBuffer indexBuffer, VkBuffer vertexSkinBuffer,
        uint32_t totalVertexCount, uint32_t totalIndexCount,
        const core::EntityData* entityData, uint32_t entityCount) {

        LOG_INFO("[GeometryCacheTest] Starting virtual geometry cache round-trip test...");

        std::vector<renderer::Vertex> allVertices;
        std::vector<uint32_t> allIndices;
        std::vector<ClusterVertexSkin> allVertexSkins;
        if (!ReadbackFullGeometry(device, allocator, graphicsQueue, commandPool, vertexBuffer, indexBuffer, vertexSkinBuffer,
            totalVertexCount, totalIndexCount, allVertices, allIndices, allVertexSkins)) {
            LOG_ERROR("[GeometryCacheTest] Aborting: full geometry readback failed.");
            return false;
        }

        // --- DIAGNOSTIC: per-meshID vertex/triangle histogram -------------------------------
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

        // We process entities in parallel through core::LoadingManager's bounded worker pool rather
        // than firing one raw std::async(std::launch::async, ...) thread per entity unconditionally.
        // The unbounded version (this codebase's own prior approach) never actually oversubscribed
        // CPU cores -- hardware_concurrency() comfortably exceeds a typical entity count -- but it did
        // launch every entity's DAG build (each one a burst of thousands of small std::vector
        // allocations: per-cluster positions/uvs/triangles, repeated across every merge level) fully
        // concurrently, and every one of those allocations serializes on the C++ runtime's single
        // process-wide heap lock (far worse in a Debug CRT build, where malloc/free/new/delete also do
        // heap-integrity validation under that same lock). The result was minutes-long apparent hangs
        // building scene.cache (observed on a 13-entity scene with a 90k-vertex floor primitive) with
        // most worker threads sitting blocked on the heap lock rather than computing -- the clustered
        // ("Nanite") render pipeline can't run without scene.cache (see this function's own header
        // comment), so this looked exactly like Nanite rendering itself being broken. Capping
        // concurrency to a small, fixed worker count keeps enough parallelism to overlap entities'
        // work while keeping simultaneous heap-lock contention low enough that each build actually
        // makes progress.
        constexpr uint32_t kMaxConcurrentDagBuilds = 4u;
        struct EntityResult {
            uint32_t meshID = 0;
            ClusterDAG dag;
            bool success = true;
            std::vector<std::string> dagErrors;
            FallbackMesh fallbackMesh;
            bool hasFallback = false;
            FallbackMeshData fallbackMeshData;
            std::vector<SurfaceCacheCardEntry> entityCards;
            std::vector<ClusterIndexEntry> localIndexEntries;
            std::vector<DAGNodeEntry> localDagEntries;
            std::vector<ClusterData> localClusterData;
        };

        // Pre-sized so each submitted job writes only into its own disjoint results[entityIdx] slot --
        // safe without any extra locking, exactly like a parallel-for (see core::LoadingManager's
        // class comment on its fan-out/blocking usage mode).
        std::vector<EntityResult> results(entityCount);
        {
            core::LoadingManager dagBuildPool(std::min(kMaxConcurrentDagBuilds, std::max(entityCount, 1u)));

            for (uint32_t entityIdx = 0; entityIdx < entityCount; ++entityIdx) {
                dagBuildPool.Submit([entityIdx, &entityData, &allVertices, &allIndices, &allVertexSkins, &results]() {
                    EntityResult& res = results[entityIdx];
                    res.meshID = entityData[entityIdx].meshID;
                    EntityMaterialProperties materialProps = GetEntityMaterialProperties(entityData[entityIdx].materialID);

                    // Skeletal-animation feature: a core::EntityFlags::IsSkeletallyAnimated entity's
                    // DAG must skip grouping/simplification entirely (forceSingleLevelDAG) and gets
                    // its leaf vertices' bone influence threaded in from allVertexSkins -- see
                    // BuildClusterDAG's own header comment for why (categorical bone data has no
                    // valid QEM/grouping carry-through).
                    bool isSkeletallyAnimated = core::GetFlag(entityData[entityIdx].flags, core::EntityFlags::IsSkeletallyAnimated);
                    res.dag = BuildClusterDAG(res.meshID, allVertices, allIndices, materialProps.maskTextureIndex,
                        isSkeletallyAnimated ? &allVertexSkins : nullptr, isSkeletallyAnimated);
                    if (res.dag.nodes.empty()) {
                        return; // Empty DAG, success remains true
                    }

                    if (!ValidateClusterDAG(res.dag, res.dagErrors)) {
                        res.success = false;
                        return;
                    }

                    res.fallbackMesh = BuildFallbackMesh(res.dag);
                    if (!res.fallbackMesh.triangles.empty()) {
                        res.hasFallback = true;
                        res.fallbackMeshData = BuildFallbackMeshData(res.meshID, res.fallbackMesh);

                        // Surface-cache cards
                        const FallbackMeshIndexEntry& fbEntry = res.fallbackMeshData.indexEntry;
                        res.entityCards = GenerateEntityCards(res.meshID,
                            maths::vec3{ fbEntry.boundsMin[0], fbEntry.boundsMin[1], fbEntry.boundsMin[2] },
                            maths::vec3{ fbEntry.boundsMax[0], fbEntry.boundsMax[1], fbEntry.boundsMax[2] });
                    }

                    res.localIndexEntries.reserve(res.dag.nodes.size());
                    res.localDagEntries.reserve(res.dag.nodes.size());
                    res.localClusterData.reserve(res.dag.nodes.size());

                    for (size_t i = 0; i < res.dag.nodes.size(); ++i) {
                        const ClusterDAGNode& node = res.dag.nodes[i];

                        ClusterData data{};
                        if (!EncodeClusterData(node, data)) {
                            res.success = false;
                        }
                        res.localClusterData.push_back(data);

                        res.localIndexEntries.push_back(BuildIndexEntry(0, res.meshID, node, materialProps.maxWPOAmplitude,
                            node.isMasked ? materialProps.maskTextureIndex : kInvalidMaskTextureIndex,
                            entityData[entityIdx].materialID));

                        DAGNodeEntry dagEntry{};
                        dagEntry.clusterID = static_cast<uint32_t>(i);

                        for (uint32_t& parentID : dagEntry.parentClusterID) {
                            parentID = kInvalidClusterID;
                        }
                        dagEntry.parentSphereCenter[0] = 0.0f;
                        dagEntry.parentSphereCenter[1] = 0.0f;
                        dagEntry.parentSphereCenter[2] = 0.0f;
                        if (node.parentGroupIndex != kInvalidDAGGroupIndex) {
                            const ClusterDAGGroup& parentGroup = res.dag.groups[node.parentGroupIndex];
                            for (size_t p = 0; p < parentGroup.outputClusterIndices.size() && p < kMaxGroupOutputClusters; ++p) {
                                dagEntry.parentClusterID[p] = parentGroup.outputClusterIndices[p];
                            }
                            dagEntry.parentSphereCenter[0] = parentGroup.groupSphereCenter.x;
                            dagEntry.parentSphereCenter[1] = parentGroup.groupSphereCenter.y;
                            dagEntry.parentSphereCenter[2] = parentGroup.groupSphereCenter.z;
                        }

                        for (uint32_t& childID : dagEntry.childClusterID) {
                            childID = kInvalidClusterID;
                        }
                        if (node.sourceGroupIndex != kInvalidDAGGroupIndex) {
                            const auto& members = res.dag.groups[node.sourceGroupIndex].memberClusterIndices;
                            for (size_t c = 0; c < members.size() && c < 4; ++c) {
                                dagEntry.childClusterID[c] = members[c];
                            }
                        }

                        dagEntry.level = node.level;
                        dagEntry.clusterError = node.clusterError;
                        dagEntry.parentError = node.parentError;
                        res.localDagEntries.push_back(dagEntry);
                    }
                });
            }

            // Blocks until every submitted job's background work has finished (core::LoadingManager's
            // fan-out/blocking mode); dagBuildPool's destructor (end of this scope) then shuts the pool
            // down, matching the "Init()-time bake all of a class's callers assume is complete" pattern
            // its own class comment documents.
            dagBuildPool.WaitIdle();
        }

        // Stitch everything sequentially
        for (const auto& res : results) {
            if (res.dag.nodes.empty()) {
                LOG_WARNING(std::format(
                    "[GeometryCacheTest] Entity meshID={} produced zero clusters (no matching triangles); skipping.", res.meshID));
                continue;
            }
            ++entitiesWithGeometry;

            if (!res.success || !res.dagErrors.empty()) {
                for (const std::string& err : res.dagErrors) {
                    LOG_ERROR(std::format("[GeometryCacheTest] entity meshID={} DAG: {}", res.meshID, err));
                }
                allEntitiesOk = false;
                continue;
            }

            if (res.hasFallback) {
                fallbackMeshes.push_back(res.fallbackMeshData);
                cardEntries.insert(cardEntries.end(), res.entityCards.begin(), res.entityCards.end());

                LOG_INFO(std::format(
                    "[GeometryCacheTest] entity meshID={}: {} surface-cache card(s) generated.",
                    res.meshID, res.entityCards.size()));

                size_t leafTriangleCount = 0;
                for (const ClusterDAGNode& n : res.dag.nodes) {
                    if (n.level == 0u) {
                        leafTriangleCount += n.mesh.triangles.size() / 3u;
                    }
                }
                size_t fallbackTriangleCount = res.fallbackMesh.triangles.size() / 3u;
                LOG_INFO(std::format(
                    "[GeometryCacheTest] entity meshID={}: fallback mesh {} triangle(s) ({:.2f}% of {} leaf triangle(s)).",
                    res.meshID, fallbackTriangleCount,
                    100.0f * static_cast<float>(fallbackTriangleCount) / static_cast<float>(std::max<size_t>(1, leafTriangleCount)),
                    leafTriangleCount));
            }

            uint32_t baseGlobalID = static_cast<uint32_t>(indexEntries.size());

            // Append cluster data
            clusterData.insert(clusterData.end(), res.localClusterData.begin(), res.localClusterData.end());

            // Build/patch global index entries and DAG entries
            for (size_t i = 0; i < res.dag.nodes.size(); ++i) {
                uint32_t globalID = baseGlobalID + static_cast<uint32_t>(i);
                
                ClusterIndexEntry idxEntry = res.localIndexEntries[i];
                idxEntry.clusterID = globalID;
                indexEntries.push_back(idxEntry);

                const DAGNodeEntry& localDag = res.localDagEntries[i];
                DAGNodeEntry globalDag{};
                globalDag.clusterID = globalID;
                for (size_t p = 0; p < kMaxGroupOutputClusters; ++p) {
                    globalDag.parentClusterID[p] = (localDag.parentClusterID[p] == kInvalidClusterID)
                        ? kInvalidClusterID : baseGlobalID + localDag.parentClusterID[p];
                }
                for (size_t c = 0; c < 4; ++c) {
                    globalDag.childClusterID[c] = (localDag.childClusterID[c] == kInvalidClusterID)
                        ? kInvalidClusterID : baseGlobalID + localDag.childClusterID[c];
                }
                globalDag.level = localDag.level;
                globalDag.clusterError = localDag.clusterError;
                globalDag.parentError = localDag.parentError;
                globalDag.parentSphereCenter[0] = localDag.parentSphereCenter[0];
                globalDag.parentSphereCenter[1] = localDag.parentSphereCenter[1];
                globalDag.parentSphereCenter[2] = localDag.parentSphereCenter[2];
                dagEntries.push_back(globalDag);
            }
        }

        // --- DIAGNOSTIC: per-entity coneCutoff range (sanity check for the backface-cone fix) ----
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
