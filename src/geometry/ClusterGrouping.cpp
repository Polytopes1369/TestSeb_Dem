#include "geometry/ClusterGrouping.h"
#include "core/Logger.h"

#include <algorithm>
#include <array>
#include <format>
#include <unordered_map>

namespace geometry {

    uint64_t PackIndexPairKey(uint32_t a, uint32_t b) {
        uint32_t lo = (a < b) ? a : b;
        uint32_t hi = (a < b) ? b : a;
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
    }

    std::vector<std::vector<uint32_t>> GreedyPairByWeight(
        uint32_t count, const std::unordered_map<uint64_t, uint32_t>& weights) {

        // Per item, its neighbors sorted by descending weight (then by ascending index for
        // determinism).
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> neighborsByItem(count); // (weight, neighborIndex)
        for (const auto& [packedKey, weight] : weights) {
            uint32_t a = static_cast<uint32_t>(packedKey >> 32);
            uint32_t b = static_cast<uint32_t>(packedKey & 0xFFFFFFFFu);
            neighborsByItem[a].emplace_back(weight, b);
            neighborsByItem[b].emplace_back(weight, a);
        }
        for (auto& neighbors : neighborsByItem) {
            std::sort(neighbors.begin(), neighbors.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first) return lhs.first > rhs.first; // Highest weight first.
                return lhs.second < rhs.second;                          // Deterministic tie-break.
                });
        }

        std::vector<bool> paired(count, false);
        std::vector<std::vector<uint32_t>> groups;
        groups.reserve((count + 1) / 2);

        for (uint32_t ci = 0; ci < count; ++ci) {
            if (paired[ci]) {
                continue;
            }
            uint32_t bestNeighbor = UINT32_MAX;
            for (const auto& [weight, neighbor] : neighborsByItem[ci]) {
                if (!paired[neighbor]) {
                    bestNeighbor = neighbor;
                    break; // Already sorted by descending weight -- first unpaired hit is best.
                }
            }

            paired[ci] = true;
            if (bestNeighbor != UINT32_MAX) {
                paired[bestNeighbor] = true;
                groups.push_back({ ci, bestNeighbor });
            }
            else {
                groups.push_back({ ci }); // No unpaired neighbor left: singleton group.
            }
        }

        return groups;
    }

    namespace {

        uint64_t PackEdgeKey(uint32_t a, uint32_t b) {
            uint32_t lo = (a < b) ? a : b;
            uint32_t hi = (a < b) ? b : a;
            return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
        }

        // Builds, for every unordered cluster pair sharing at least one global vertex, the count
        // of shared global vertices -- used both as the adjacency test (count > 0) and as the
        // greedy pairing's preference weight (more shared vertices => more natural merge).
        //
        // Cross-classification pairs (one cluster isMasked, the other not) are never recorded, even
        // if spatially adjacent: merging an opaque and a masked cluster would produce a coarser DAG
        // node that is no longer provably pure, defeating the entire point of the opaque/masked
        // split one level up. A cluster left with no same-class neighbor simply becomes a singleton
        // group below, exactly like an already-disconnected mesh island.
        std::unordered_map<uint64_t, uint32_t> BuildClusterAdjacencyWeights(const std::vector<MeshCluster>& clusters) {
            std::unordered_map<uint32_t, std::vector<uint32_t>> vertexToClusters;
            for (uint32_t ci = 0; ci < clusters.size(); ++ci) {
                for (uint32_t g : clusters[ci].globalVertexIndices) {
                    vertexToClusters[g].push_back(ci);
                }
            }

            std::unordered_map<uint64_t, uint32_t> weights;
            for (const auto& [globalVertex, owningClusters] : vertexToClusters) {
                if (owningClusters.size() < 2) {
                    continue;
                }
                for (size_t a = 0; a < owningClusters.size(); ++a) {
                    for (size_t b = a + 1; b < owningClusters.size(); ++b) {
                        if (clusters[owningClusters[a]].isMasked != clusters[owningClusters[b]].isMasked) {
                            continue;
                        }
                        uint64_t key = PackIndexPairKey(owningClusters[a], owningClusters[b]);
                        weights[key] += 1u;
                    }
                }
            }
            return weights;
        }

        // Merges the member clusters' triangles into one SimplifiableMesh, locking every vertex
        // that lies on an edge used by exactly one triangle within the merged set (the group's
        // outer boundary).
        ClusterGroup BuildGroupMesh(
            const std::vector<uint32_t>& memberClusterIndices,
            const std::vector<MeshCluster>& clusters,
            const std::vector<renderer::Vertex>& allVertices) {

            ClusterGroup group;
            group.memberClusterIndices = memberClusterIndices;

            std::vector<std::array<uint32_t, 3>> globalTriangles;
            for (uint32_t memberIdx : memberClusterIndices) {
                const MeshCluster& cluster = clusters[memberIdx];
                size_t triangleCount = cluster.localTriangleIndices.size() / 3u;
                globalTriangles.reserve(globalTriangles.size() + triangleCount);
                for (size_t t = 0; t < triangleCount; ++t) {
                    uint8_t l0 = cluster.localTriangleIndices[t * 3 + 0];
                    uint8_t l1 = cluster.localTriangleIndices[t * 3 + 1];
                    uint8_t l2 = cluster.localTriangleIndices[t * 3 + 2];
                    globalTriangles.push_back({
                        cluster.globalVertexIndices[l0],
                        cluster.globalVertexIndices[l1],
                        cluster.globalVertexIndices[l2]
                        });
                }
            }
            group.originalTriangleCount = static_cast<uint32_t>(globalTriangles.size());

            std::unordered_map<uint64_t, uint32_t> edgeUsageCount;
            for (const auto& tri : globalTriangles) {
                edgeUsageCount[PackEdgeKey(tri[0], tri[1])] += 1u;
                edgeUsageCount[PackEdgeKey(tri[1], tri[2])] += 1u;
                edgeUsageCount[PackEdgeKey(tri[2], tri[0])] += 1u;
            }
            std::unordered_map<uint32_t, bool> lockedGlobal;
            for (const auto& [edgeKey, count] : edgeUsageCount) {
                if (count == 1u) {
                    lockedGlobal[static_cast<uint32_t>(edgeKey >> 32)] = true;
                    lockedGlobal[static_cast<uint32_t>(edgeKey & 0xFFFFFFFFu)] = true;
                }
            }

            std::unordered_map<uint32_t, uint32_t> globalToLocal;
            globalToLocal.reserve(globalTriangles.size());
            group.mesh.positions.reserve(globalTriangles.size());
            group.mesh.locked.reserve(globalTriangles.size());
            group.mesh.uvs.reserve(globalTriangles.size());
            group.mesh.triangles.reserve(globalTriangles.size() * 3);

            auto localIndexFor = [&](uint32_t globalVertex) -> uint32_t {
                auto it = globalToLocal.find(globalVertex);
                if (it != globalToLocal.end()) {
                    return it->second;
                }
                uint32_t local = static_cast<uint32_t>(group.mesh.positions.size());
                globalToLocal.emplace(globalVertex, local);
                group.mesh.positions.push_back(allVertices[globalVertex].position);
                group.mesh.locked.push_back(lockedGlobal.count(globalVertex) != 0);
                group.mesh.uvs.push_back(allVertices[globalVertex].uv);
                return local;
                };

            for (const auto& tri : globalTriangles) {
                group.mesh.triangles.push_back(localIndexFor(tri[0]));
                group.mesh.triangles.push_back(localIndexFor(tri[1]));
                group.mesh.triangles.push_back(localIndexFor(tri[2]));
            }

            return group;
        }

    } // namespace

    std::vector<ClusterGroup> GroupAdjacentClusters(
        const std::vector<MeshCluster>& clusters,
        const std::vector<renderer::Vertex>& allVertices) {

        if (clusters.empty()) {
            return {};
        }

        std::unordered_map<uint64_t, uint32_t> adjacencyWeights = BuildClusterAdjacencyWeights(clusters);
        std::vector<std::vector<uint32_t>> memberLists = GreedyPairByWeight(static_cast<uint32_t>(clusters.size()), adjacencyWeights);

        std::vector<ClusterGroup> groups;
        groups.reserve(memberLists.size());
        for (const auto& members : memberLists) {
            groups.push_back(BuildGroupMesh(members, clusters, allVertices));
        }
        LOG_INFO(std::format("[ClusterGrouping] Grouped {} clusters into {} groups.", clusters.size(), groups.size()));
        return groups;
    }

}
