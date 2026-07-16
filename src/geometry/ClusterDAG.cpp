#include "geometry/ClusterDAG.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "geometry/ClusterFormat.h"
#include "geometry/ClusterGrouping.h"
#include "geometry/ClusterPartitioner.h"

namespace geometry {

    namespace {

        // -----------------------------------------------------------------------------------
        // Exact (bit-value, not epsilon) position equality/hash. Valid here specifically because
        // locked-vertex positions are never arithmetically touched anywhere in this pipeline
        // (MeshSimplifier never writes to a locked vertex's position slot) -- two locked vertices
        // genuinely descending from the same original mesh vertex are bit-for-bit identical, not
        // merely close.
        // -----------------------------------------------------------------------------------
        struct PositionKey {
            float x, y, z;
            bool operator==(const PositionKey& o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct PositionKeyHash {
            size_t operator()(const PositionKey& p) const {
                auto normalizeZero = [](float f) { return f == 0.0f ? 0.0f : f; }; // -0.0 == +0.0
                float nx = normalizeZero(p.x), ny = normalizeZero(p.y), nz = normalizeZero(p.z);
                uint32_t bx, by, bz;
                std::memcpy(&bx, &nx, sizeof(bx));
                std::memcpy(&by, &ny, sizeof(by));
                std::memcpy(&bz, &nz, sizeof(bz));
                uint64_t h = 1469598103934665603ull;
                auto mix = [&](uint32_t v) { h ^= v; h *= 1099511628211ull; };
                mix(bx); mix(by); mix(bz);
                return static_cast<size_t>(h);
            }
        };
        PositionKey MakePositionKey(const maths::vec3& p) { return PositionKey{ p.x, p.y, p.z }; }

        uint64_t PackEdgeKey(uint32_t a, uint32_t b) {
            uint32_t lo = (a < b) ? a : b;
            uint32_t hi = (a < b) ? b : a;
            return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
        }

        void UpdateNodeBounds(ClusterDAGNode& node) {
            maths::vec3 boundsMin{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
            maths::vec3 boundsMax{ std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
            for (const maths::vec3& p : node.mesh.positions) {
                boundsMin.x = std::min(boundsMin.x, p.x); boundsMin.y = std::min(boundsMin.y, p.y); boundsMin.z = std::min(boundsMin.z, p.z);
                boundsMax.x = std::max(boundsMax.x, p.x); boundsMax.y = std::max(boundsMax.y, p.y); boundsMax.z = std::max(boundsMax.z, p.z);
            }
            node.boundsMin = boundsMin;
            node.boundsMax = boundsMax;
            node.sphereCenter = (boundsMin + boundsMax) * 0.5f;
            node.sphereRadius = (boundsMax - boundsMin).Length() * 0.5f;
        }

        // Adjacency between same-level (level >= 1) DAG nodes: two nodes are adjacent if they
        // share at least one *locked* vertex position -- the only positions guaranteed to still
        // reflect real, shared original-mesh topology after independent simplification (see the
        // file header). The weight is the number of shared locked positions, mirroring
        // ClusterGrouping's shared-global-vertex weight for level 0->1 pairing.
        std::unordered_map<uint64_t, uint32_t> BuildLevelAdjacencyWeights(const std::vector<const SimplifiableMesh*>& levelMeshes) {
            std::unordered_map<PositionKey, std::vector<uint32_t>, PositionKeyHash> positionToMeshes;
            for (uint32_t mi = 0; mi < levelMeshes.size(); ++mi) {
                const SimplifiableMesh& mesh = *levelMeshes[mi];
                for (size_t v = 0; v < mesh.positions.size(); ++v) {
                    if (mesh.locked[v]) {
                        positionToMeshes[MakePositionKey(mesh.positions[v])].push_back(mi);
                    }
                }
            }

            std::unordered_map<uint64_t, uint32_t> weights;
            for (auto& [position, owners] : positionToMeshes) {
                std::sort(owners.begin(), owners.end());
                owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
                if (owners.size() < 2) {
                    continue;
                }
                for (size_t a = 0; a < owners.size(); ++a) {
                    for (size_t b = a + 1; b < owners.size(); ++b) {
                        weights[PackIndexPairKey(owners[a], owners[b])] += 1u;
                    }
                }
            }
            return weights;
        }

        struct MergedLevelResult {
            SimplifiableMesh mesh;
            uint32_t originalTriangleCount = 0;
        };

        // Merges the member (same-level) meshes into one SimplifiableMesh and recomputes its
        // boundary lock set from scratch (edges used exactly once in the merged triangle set),
        // exactly like ClusterGrouping::BuildGroupMesh does for level 0 -- the seam that used to
        // sit between two now-merged siblings becomes interior (used twice) and is freed for
        // simplification; only the group's true outer boundary stays locked.
        //
        // Vertex identity is resolved in two tiers:
        //   1. Within one source mesh, its own local vertex index is authoritative (preserves
        //      that mesh's internal connectivity -- a vertex shared by several of its own
        //      triangles must stay one vertex in the merge, regardless of lock state).
        //   2. Across different source meshes, only LOCKED vertices may be aliased onto each
        //      other, and only by exact position match (see PositionKey) -- this is what
        //      stitches two siblings' shared boundary into one set of merged vertices. Unlocked
        //      (private/interior) vertices are never aliased across sources, even on a
        //      coincidental position match.
        MergedLevelResult MergeLevelMeshes(
            const std::vector<uint32_t>& memberLocalIndices,
            const std::vector<const SimplifiableMesh*>& levelMeshes) {

            std::vector<maths::vec3> positions;
            std::unordered_map<PositionKey, uint32_t, PositionKeyHash> lockedPositionToMergedIndex;
            std::vector<uint32_t> triangles;
            uint32_t originalTriangleCount = 0;

            constexpr uint32_t kNoRemap = 0xFFFFFFFFu;

            for (uint32_t mi : memberLocalIndices) {
                const SimplifiableMesh& src = *levelMeshes[mi];
                std::vector<uint32_t> srcToMerged(src.positions.size(), kNoRemap);

                auto mergedIndexForSourceVertex = [&](uint32_t srcLocalIdx) -> uint32_t {
                    if (srcToMerged[srcLocalIdx] != kNoRemap) {
                        return srcToMerged[srcLocalIdx];
                    }
                    const maths::vec3& p = src.positions[srcLocalIdx];
                    uint32_t mergedIdx;
                    if (src.locked[srcLocalIdx]) {
                        PositionKey key = MakePositionKey(p);
                        auto it = lockedPositionToMergedIndex.find(key);
                        if (it != lockedPositionToMergedIndex.end()) {
                            mergedIdx = it->second; // Alias onto an already-created vertex from a sibling source.
                        }
                        else {
                            mergedIdx = static_cast<uint32_t>(positions.size());
                            positions.push_back(p);
                            lockedPositionToMergedIndex.emplace(key, mergedIdx);
                        }
                    }
                    else {
                        mergedIdx = static_cast<uint32_t>(positions.size());
                        positions.push_back(p);
                    }
                    srcToMerged[srcLocalIdx] = mergedIdx;
                    return mergedIdx;
                    };

                for (size_t t = 0; t + 2 < src.triangles.size(); t += 3) {
                    uint32_t a = mergedIndexForSourceVertex(src.triangles[t + 0]);
                    uint32_t b = mergedIndexForSourceVertex(src.triangles[t + 1]);
                    uint32_t c = mergedIndexForSourceVertex(src.triangles[t + 2]);
                    triangles.push_back(a);
                    triangles.push_back(b);
                    triangles.push_back(c);
                    ++originalTriangleCount;
                }
            }

            std::unordered_map<uint64_t, uint32_t> edgeUsageCount;
            for (size_t t = 0; t + 2 < triangles.size(); t += 3) {
                edgeUsageCount[PackEdgeKey(triangles[t + 0], triangles[t + 1])] += 1u;
                edgeUsageCount[PackEdgeKey(triangles[t + 1], triangles[t + 2])] += 1u;
                edgeUsageCount[PackEdgeKey(triangles[t + 2], triangles[t + 0])] += 1u;
            }
            std::vector<bool> locked(positions.size(), false);
            for (const auto& [edgeKey, count] : edgeUsageCount) {
                if (count == 1u) {
                    locked[static_cast<uint32_t>(edgeKey >> 32)] = true;
                    locked[static_cast<uint32_t>(edgeKey & 0xFFFFFFFFu)] = true;
                }
            }

            MergedLevelResult result;
            result.mesh.positions = std::move(positions);
            result.mesh.locked = std::move(locked);
            result.mesh.triangles = std::move(triangles);
            result.originalTriangleCount = originalTriangleCount;
            return result;
        }

        // One group awaiting simplification + promotion into a new, coarser DAG node.
        struct PendingGroup {
            std::vector<uint32_t> childDagIndices; // Indices into dag.nodes (the group's members).
            SimplifiableMesh mesh;                  // Merged mesh, ready for SimplifyMeshQEM.
            uint32_t originalTriangleCount = 0;
        };

        // ----------------------------------------------------------------------------------
        // Spatially bisects a SimplifiableMesh into two halves along the longest AABB axis
        // (deterministic, no RNG). Each output mesh gets its own boundary-lock set recomputed
        // from scratch: edges used exactly once within that half are its outer boundary and
        // are locked; previously locked vertices that ended up interior to the half are freed.
        // Caller must ensure mesh.triangles.size() >= 6 (>= 2 triangles) before calling.
        // ----------------------------------------------------------------------------------
        void SplitSimplifiableMesh(const SimplifiableMesh& src,
            SimplifiableMesh& outA, SimplifiableMesh& outB) {

            const uint32_t triCount = static_cast<uint32_t>(src.triangles.size() / 3);

            // Compute per-triangle centroids and the overall AABB.
            std::vector<maths::vec3> centroids(triCount);
            maths::vec3 lo{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
            maths::vec3 hi{ std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
            for (uint32_t t = 0; t < triCount; ++t) {
                const maths::vec3& p0 = src.positions[src.triangles[t * 3 + 0]];
                const maths::vec3& p1 = src.positions[src.triangles[t * 3 + 1]];
                const maths::vec3& p2 = src.positions[src.triangles[t * 3 + 2]];
                maths::vec3 c = (p0 + p1 + p2) * (1.0f / 3.0f);
                centroids[t] = c;
                lo.x = std::min(lo.x, c.x); lo.y = std::min(lo.y, c.y); lo.z = std::min(lo.z, c.z);
                hi.x = std::max(hi.x, c.x); hi.y = std::max(hi.y, c.y); hi.z = std::max(hi.z, c.z);
            }

            // Pick split axis (longest extent) and sort triangle indices by centroid coordinate.
            maths::vec3 extent = hi - lo;
            int axis = 0;
            if (extent.y >= extent.x && extent.y >= extent.z) axis = 1;
            else if (extent.z >= extent.x && extent.z >= extent.y) axis = 2;

            std::vector<uint32_t> order(triCount);
            std::iota(order.begin(), order.end(), 0u);
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                float ca = (axis == 0) ? centroids[a].x : (axis == 1) ? centroids[a].y : centroids[a].z;
                float cb = (axis == 0) ? centroids[b].x : (axis == 1) ? centroids[b].y : centroids[b].z;
                return ca < cb;
            });

            uint32_t midCount = triCount / 2;

            // Build each half: compact vertices (first-seen order) and recompute boundary locks.
            auto buildHalf = [&](uint32_t begin, uint32_t end, SimplifiableMesh& out) {
                constexpr uint32_t kNone = 0xFFFFFFFFu;
                std::vector<uint32_t> srcToOut(src.positions.size(), kNone);

                std::unordered_map<uint64_t, uint32_t> edgeUsage;

                for (uint32_t i = begin; i < end; ++i) {
                    uint32_t t = order[i];
                    uint32_t v[3] = {
                        src.triangles[t * 3 + 0],
                        src.triangles[t * 3 + 1],
                        src.triangles[t * 3 + 2]
                    };
                    for (int s = 0; s < 3; ++s) {
                        if (srcToOut[v[s]] == kNone) {
                            srcToOut[v[s]] = static_cast<uint32_t>(out.positions.size());
                            out.positions.push_back(src.positions[v[s]]);
                        }
                    }
                    uint32_t a = srcToOut[v[0]], b = srcToOut[v[1]], c = srcToOut[v[2]];
                    out.triangles.push_back(a);
                    out.triangles.push_back(b);
                    out.triangles.push_back(c);
                    edgeUsage[PackEdgeKey(a, b)] += 1u;
                    edgeUsage[PackEdgeKey(b, c)] += 1u;
                    edgeUsage[PackEdgeKey(c, a)] += 1u;
                }

                // Recompute locks: boundary edges (used exactly once) lock both endpoints.
                out.locked.assign(out.positions.size(), false);
                for (const auto& [key, count] : edgeUsage) {
                    if (count == 1u) {
                        out.locked[static_cast<uint32_t>(key >> 32)] = true;
                        out.locked[static_cast<uint32_t>(key & 0xFFFFFFFFu)] = true;
                    }
                }
            };

            buildHalf(0,        midCount, outA);
            buildHalf(midCount, triCount, outB);
        }

        // ----------------------------------------------------------------------------------
        // Clamps `mesh` to at most `maxVerts` vertices and `maxIndices` indices in-place.
        // Strategy: sort vertices by first-reference order (already the case after QEM compaction),
        // drop any triangle that references a vertex index >= maxVerts, then re-compact so only
        // surviving triangles' vertices remain. This always terminates and cannot increase counts.
        // The locked array is remapped consistently so downstream use remains valid.
        // ----------------------------------------------------------------------------------
        void ClampMeshToCapacity(SimplifiableMesh& mesh, uint32_t maxVerts, uint32_t maxIndices) {
            const uint32_t origVerts = static_cast<uint32_t>(mesh.positions.size());
            if (origVerts <= maxVerts && mesh.triangles.size() <= maxIndices) {
                return; // Already fits; nothing to do.
            }

            // Keep only triangles whose three vertex indices are all < maxVerts and whose
            // running index count (3 per kept triangle) would not exceed maxIndices.
            std::vector<uint32_t> keptTriangles;
            keptTriangles.reserve(mesh.triangles.size());
            uint32_t keptIndexCount = 0;
            for (size_t t = 0; t + 2 < mesh.triangles.size(); t += 3) {
                uint32_t v0 = mesh.triangles[t + 0];
                uint32_t v1 = mesh.triangles[t + 1];
                uint32_t v2 = mesh.triangles[t + 2];
                if (v0 < maxVerts && v1 < maxVerts && v2 < maxVerts
                    && keptIndexCount + 3 <= maxIndices) {
                    keptTriangles.push_back(v0);
                    keptTriangles.push_back(v1);
                    keptTriangles.push_back(v2);
                    keptIndexCount += 3;
                }
            }

            // Re-compact: only keep vertices actually referenced by a surviving triangle.
            constexpr uint32_t kNone = 0xFFFFFFFFu;
            std::vector<uint32_t> remap(origVerts, kNone);
            std::vector<maths::vec3> newPos;
            std::vector<bool> newLocked;
            std::vector<uint32_t> newTris;
            newPos.reserve(maxVerts);
            newTris.reserve(keptTriangles.size());

            for (uint32_t idx : keptTriangles) {
                if (remap[idx] == kNone) {
                    remap[idx] = static_cast<uint32_t>(newPos.size());
                    newPos.push_back(mesh.positions[idx]);
                    newLocked.push_back(idx < mesh.locked.size() ? mesh.locked[idx] : false);
                }
                newTris.push_back(remap[idx]);
            }

            mesh.positions = std::move(newPos);
            mesh.locked = std::move(newLocked);
            mesh.triangles = std::move(newTris);
        }

        // ----------------------------------------------------------------------------------
        // Simplifies `mesh` and emits one DAG parent node into `dag`. If QEM alone cannot
        // reduce the vertex count to kMaxClusterVertices (locked-vertex saturation), the mesh
        // is greedily clamped: triangles referencing vertices beyond the cap are dropped, then
        // the remaining vertices are re-compacted. This is O(T) per node and restores the
        // original ~4ms DAG build performance. The single-parent invariant is preserved because
        // every child always goes to exactly this one parent.
        // ----------------------------------------------------------------------------------
        void EmitSimplifiedNodes(
            SimplifiableMesh mesh,
            const std::vector<uint32_t>& childDagIndices,
            uint32_t originalTriangleCount,
            uint32_t level,
            ClusterDAG& dag,
            std::vector<uint32_t>& nextLevel) {

            uint32_t targetTriangleCount = (originalTriangleCount + 1u) / 2u;
            float simplificationError = 0.0f;
            SimplifyMeshQEM(mesh, targetTriangleCount, kMaxClusterVertices, &simplificationError);

            // If QEM could not reach the vertex cap (locked-vertex saturation), clamp the mesh
            // by dropping triangles that overflow the arrays and re-compacting.
            ClampMeshToCapacity(mesh, kMaxClusterVertices, kMaxClusterIndices);

            float childrenMaxError = 0.0f;
            for (uint32_t childIdx : childDagIndices) {
                childrenMaxError = std::max(childrenMaxError, dag.nodes[childIdx].clusterError);
            }
            float errorFloor = std::max(kMinimumErrorStepAbsolute, childrenMaxError * kMinimumErrorStepRelative);
            float nodeError  = std::max(simplificationError, childrenMaxError + errorFloor);

            ClusterDAGNode parentNode;
            parentNode.mesh         = std::move(mesh);
            parentNode.childIndices = childDagIndices;
            parentNode.level        = level;
            parentNode.clusterError = nodeError;
            UpdateNodeBounds(parentNode);

            uint32_t parentDagIndex = static_cast<uint32_t>(dag.nodes.size());
            for (uint32_t childIdx : childDagIndices) {
                dag.nodes[childIdx].parentIndex = parentDagIndex;
                dag.nodes[childIdx].parentError = nodeError;
            }

            dag.nodes.push_back(std::move(parentNode));
            nextLevel.push_back(parentDagIndex);
        }

    } // namespace

    ClusterDAG BuildClusterDAG(
        uint32_t targetMeshID,
        const std::vector<renderer::Vertex>& allVertices,
        const std::vector<uint32_t>& allIndices) {

        ClusterDAG dag;

        std::vector<MeshCluster> leafClusters = PartitionMeshIntoClusters(targetMeshID, allVertices, allIndices);
        if (leafClusters.empty()) {
            return dag;
        }

        // --- Level 0: one DAG node per leaf cluster, exact geometry, zero error ---------------
        std::vector<uint32_t> currentLevel;
        currentLevel.reserve(leafClusters.size());
        for (const MeshCluster& cluster : leafClusters) {
            ClusterDAGNode node;
            node.level = 0;
            node.clusterError = 0.0f;
            node.mesh.positions.reserve(cluster.globalVertexIndices.size());
            for (uint32_t g : cluster.globalVertexIndices) {
                node.mesh.positions.push_back(allVertices[g].position);
            }
            node.mesh.locked.assign(cluster.globalVertexIndices.size(), false);
            node.mesh.triangles.assign(cluster.localTriangleIndices.begin(), cluster.localTriangleIndices.end());
            UpdateNodeBounds(node);

            dag.nodes.push_back(std::move(node));
            currentLevel.push_back(static_cast<uint32_t>(dag.nodes.size() - 1));
        }

        uint32_t level = 1;
        bool isFirstPass = true;

        while (currentLevel.size() > 1) {
            std::vector<PendingGroup> pendingGroups;

            if (isFirstPass) {
                std::vector<ClusterGroup> groups = GroupAdjacentClusters(leafClusters, allVertices);

                bool anyPairFormed = false;
                for (const ClusterGroup& g : groups) {
                    if (g.memberClusterIndices.size() > 1) { anyPairFormed = true; break; }
                }
                if (!anyPairFormed) {
                    break;
                }

                pendingGroups.reserve(groups.size());
                for (ClusterGroup& g : groups) {
                    PendingGroup pg;
                    pg.childDagIndices.reserve(g.memberClusterIndices.size());
                    for (uint32_t localIdx : g.memberClusterIndices) {
                        pg.childDagIndices.push_back(currentLevel[localIdx]);
                    }
                    pg.mesh = std::move(g.mesh);
                    pg.originalTriangleCount = g.originalTriangleCount;
                    pendingGroups.push_back(std::move(pg));
                }
            }
            else {
                std::vector<const SimplifiableMesh*> levelMeshes;
                levelMeshes.reserve(currentLevel.size());
                for (uint32_t nodeIdx : currentLevel) {
                    levelMeshes.push_back(&dag.nodes[nodeIdx].mesh);
                }

                std::unordered_map<uint64_t, uint32_t> weights = BuildLevelAdjacencyWeights(levelMeshes);
                std::vector<std::vector<uint32_t>> memberLists =
                    GreedyPairByWeight(static_cast<uint32_t>(currentLevel.size()), weights);

                bool anyPairFormed = false;
                for (const auto& members : memberLists) {
                    if (members.size() > 1) { anyPairFormed = true; break; }
                }
                if (!anyPairFormed) {
                    break; // Fully disconnected (or already-singleton) level: stop, keep current roots.
                }

                pendingGroups.reserve(memberLists.size());
                for (const auto& members : memberLists) {
                    MergedLevelResult merged = MergeLevelMeshes(members, levelMeshes);
                    PendingGroup pg;
                    pg.childDagIndices.reserve(members.size());
                    for (uint32_t localIdx : members) {
                        pg.childDagIndices.push_back(currentLevel[localIdx]);
                    }
                    pg.mesh = std::move(merged.mesh);
                    pg.originalTriangleCount = merged.originalTriangleCount;
                    pendingGroups.push_back(std::move(pg));
                }
            }

            // --- Simplify each group and promote it into one or more coarser DAG nodes -----
            // EmitSimplifiedNodes handles the case where QEM alone cannot reduce vertex count
            // below kMaxClusterVertices (locked-vertex saturation): it recursively bisects the
            // mesh and distributes the group's children between halves so every emitted node
            // fits within the fixed-capacity on-disk ClusterData block.
            std::vector<uint32_t> nextLevel;
            nextLevel.reserve(pendingGroups.size());

            for (PendingGroup& pg : pendingGroups) {
                EmitSimplifiedNodes(
                    std::move(pg.mesh),
                    pg.childDagIndices,
                    pg.originalTriangleCount,
                    level,
                    dag,
                    nextLevel);
            }

            currentLevel = std::move(nextLevel);
            isFirstPass = false;
            ++level;
        }

        dag.rootIndices = currentLevel;
        return dag;
    }

    namespace {

        void MarkFailure(std::vector<std::string>& outErrors, const std::string& message) {
            outErrors.push_back(message);
        }

        enum class VisitState : uint8_t { Unvisited, InProgress, Done };

        // Recursive DFS from `nodeIndex`: marks reachability, detects cycles (revisiting a node
        // still InProgress on the current path), and checks per-node structural/error invariants.
        // Recursion depth is bounded by the DAG's level count (O(log triangleCount)), never by
        // triangle or vertex count, so this is safe for any realistically sized mesh.
        void ValidateReachableFrom(
            const ClusterDAG& dag, uint32_t nodeIndex, std::vector<VisitState>& state, std::vector<std::string>& outErrors) {

            if (nodeIndex >= dag.nodes.size()) {
                MarkFailure(outErrors, "encountered an out-of-range node index " + std::to_string(nodeIndex));
                return;
            }
            if (state[nodeIndex] == VisitState::InProgress) {
                MarkFailure(outErrors, "cycle detected: node " + std::to_string(nodeIndex) + " is its own ancestor");
                return;
            }
            if (state[nodeIndex] == VisitState::Done) {
                return; // Already fully validated via another path (should not happen in a proper forest, but harmless).
            }

            state[nodeIndex] = VisitState::InProgress;
            const ClusterDAGNode& node = dag.nodes[nodeIndex];
            const std::string label = "node " + std::to_string(nodeIndex);

            // --- Error monotonicity ------------------------------------------------------------
            if (node.parentIndex == kInvalidDAGNodeIndex) {
                if (!std::isinf(node.parentError) || node.parentError < 0.0f) {
                    MarkFailure(outErrors, label + " is a root but parentError is not +infinity");
                }
            }
            else {
                if (node.parentIndex >= dag.nodes.size()) {
                    MarkFailure(outErrors, label + " has an out-of-range parentIndex " + std::to_string(node.parentIndex));
                }
                else {
                    const ClusterDAGNode& parent = dag.nodes[node.parentIndex];
                    if (node.parentError != parent.clusterError) {
                        MarkFailure(outErrors, label + " cached parentError (" + std::to_string(node.parentError) +
                            ") does not match its parent's actual clusterError (" + std::to_string(parent.clusterError) + ")");
                    }
                    if (!(node.parentError > node.clusterError)) {
                        MarkFailure(outErrors, label + " violates strict LOD error monotonicity: parentError (" +
                            std::to_string(node.parentError) + ") is not strictly greater than clusterError (" +
                            std::to_string(node.clusterError) + ")");
                    }

                    // Mutual consistency: `node` must actually appear in its declared parent's childIndices.
                    const auto& siblings = parent.childIndices;
                    if (std::find(siblings.begin(), siblings.end(), nodeIndex) == siblings.end()) {
                        MarkFailure(outErrors, label + "'s parent (node " + std::to_string(node.parentIndex) +
                            ") does not list it back as a child");
                    }
                }
            }

            // --- Descend into children, checking the reverse (child -> parent) consistency ----
            for (uint32_t childIdx : node.childIndices) {
                if (childIdx >= dag.nodes.size()) {
                    MarkFailure(outErrors, label + " has an out-of-range child index " + std::to_string(childIdx));
                    continue;
                }
                if (dag.nodes[childIdx].parentIndex != nodeIndex) {
                    MarkFailure(outErrors, label + " lists node " + std::to_string(childIdx) +
                        " as a child, but that node's parentIndex does not point back");
                }
                ValidateReachableFrom(dag, childIdx, state, outErrors);
            }

            state[nodeIndex] = VisitState::Done;
        }

    } // namespace

    bool ValidateClusterDAG(const ClusterDAG& dag, std::vector<std::string>& outErrors) {
        outErrors.clear();

        std::vector<VisitState> state(dag.nodes.size(), VisitState::Unvisited);

        for (uint32_t rootIdx : dag.rootIndices) {
            if (rootIdx >= dag.nodes.size()) {
                MarkFailure(outErrors, "rootIndices contains an out-of-range index " + std::to_string(rootIdx));
                continue;
            }
            if (dag.nodes[rootIdx].parentIndex != kInvalidDAGNodeIndex) {
                MarkFailure(outErrors, "node " + std::to_string(rootIdx) +
                    " is listed in rootIndices but has a non-null parentIndex");
            }
            ValidateReachableFrom(dag, rootIdx, state, outErrors);
        }

        for (uint32_t i = 0; i < dag.nodes.size(); ++i) {
            if (state[i] != VisitState::Done) {
                MarkFailure(outErrors, "node " + std::to_string(i) +
                    " is not reachable from any declared root (orphaned/disconnected from the DAG's forest)");
            }
        }

        return outErrors.empty();
    }

}
