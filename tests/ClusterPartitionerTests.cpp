// Standalone, framework-free unit test for geometry::PartitionMeshIntoClusters
// (src/geometry/ClusterPartitioner.h/.cpp).
//
// Purely CPU-side: builds synthetic vertex/index buffers shaped like the ones the PrimitiveGen
// compute shaders produce (per-vertex meshID stamping, indexed triangle lists), so it needs
// neither Vulkan nor a running window/device. Exits 0 if every check passes, non-zero otherwise,
// so it can be registered with CTest (see the top-level CMakeLists.txt) without pulling in any
// external test framework.
//
// What is validated for every scenario:
//   1. No cluster ever exceeds geometry::kMaxClusterTriangles triangles or
//      geometry::kMaxClusterVertices unique vertices (the hard partitioning limits).
//   2. Every original triangle appears in exactly one cluster (no triangle is dropped or
//      duplicated across clusters), traced via MeshCluster::originalTriangleIndices.
//   3. Each cluster triangle's locally-renumbered vertex indices reconstruct the exact same
//      global vertex indices as the original triangle (no vertex-remapping corruption).

#include "geometry/ClusterFormat.h"
#include "geometry/ClusterPartitioner.h"
#include "core/maths/Maths.h"
#include "renderer/RenderTypes.h"
#include "SyntheticMesh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

    using testutil::GenerateUVSphere;

    struct GroundTruthTriangle {
        uint32_t v0, v1, v2;
    };

    std::vector<GroundTruthTriangle> ExtractGroundTruthTriangles(
        uint32_t meshID, const std::vector<renderer::Vertex>& vertices, const std::vector<uint32_t>& indices) {

        std::vector<GroundTruthTriangle> result;
        for (size_t t = 0; t + 2 < indices.size(); t += 3) {
            uint32_t i0 = indices[t + 0];
            uint32_t i1 = indices[t + 1];
            uint32_t i2 = indices[t + 2];
            if (vertices[i0].meshID != meshID) {
                continue;
            }
            result.push_back(GroundTruthTriangle{ i0, i1, i2 });
        }
        return result;
    }

    // -------------------------------------------------------------------------------------
    // Runs every check for one (meshID, mesh) scenario. Accumulates failures into failCount
    // instead of aborting at the first one, so a single run reports every violation found.
    // -------------------------------------------------------------------------------------
    bool ValidatePartition(const std::string& label, uint32_t meshID,
        const std::vector<renderer::Vertex>& vertices, const std::vector<uint32_t>& indices, int& failCount) {

        auto check = [&](bool condition, const std::string& message) {
            if (!condition) {
                std::cerr << "[FAIL] " << label << ": " << message << "\n";
                ++failCount;
            }
            return condition;
            };

        std::vector<GroundTruthTriangle> groundTruth = ExtractGroundTruthTriangles(meshID, vertices, indices);
        std::vector<geometry::MeshCluster> clusters = geometry::PartitionMeshIntoClusters(meshID, vertices, indices, geometry::kInvalidMaskTextureIndex);

        if (groundTruth.empty()) {
            return check(clusters.empty(), "expected zero clusters for a meshID with no matching triangles");
        }

        bool ok = check(!clusters.empty(), "expected at least one cluster for a non-empty mesh");

        std::vector<bool> covered(groundTruth.size(), false);
        uint64_t totalClusterTriangles = 0;

        for (size_t ci = 0; ci < clusters.size(); ++ci) {
            const geometry::MeshCluster& cluster = clusters[ci];
            const std::string clusterLabel = "cluster " + std::to_string(ci);
            size_t triangleCount = cluster.localTriangleIndices.size() / 3u;

            ok &= check(cluster.localTriangleIndices.size() % 3u == 0u, clusterLabel + " has a non-multiple-of-3 index count");
            ok &= check(triangleCount <= geometry::kMaxClusterTriangles,
                clusterLabel + " exceeds kMaxClusterTriangles: " + std::to_string(triangleCount) +
                " > " + std::to_string(geometry::kMaxClusterTriangles));
            ok &= check(cluster.globalVertexIndices.size() <= geometry::kMaxClusterVertices,
                clusterLabel + " exceeds kMaxClusterVertices: " + std::to_string(cluster.globalVertexIndices.size()) +
                " > " + std::to_string(geometry::kMaxClusterVertices));
            ok &= check(cluster.originalTriangleIndices.size() == triangleCount,
                clusterLabel + " originalTriangleIndices size does not match its triangle count");

            for (size_t li = 0; li < triangleCount; ++li) {
                uint32_t originalIdx = cluster.originalTriangleIndices[li];
                if (!check(originalIdx < groundTruth.size(), clusterLabel + " references an out-of-range original triangle index")) {
                    continue;
                }
                if (!check(!covered[originalIdx], "original triangle " + std::to_string(originalIdx) + " appears in more than one cluster")) {
                    continue;
                }
                covered[originalIdx] = true;

                uint8_t lv0 = cluster.localTriangleIndices[li * 3 + 0];
                uint8_t lv1 = cluster.localTriangleIndices[li * 3 + 1];
                uint8_t lv2 = cluster.localTriangleIndices[li * 3 + 2];
                bool localsInRange = lv0 < cluster.globalVertexIndices.size()
                    && lv1 < cluster.globalVertexIndices.size()
                    && lv2 < cluster.globalVertexIndices.size();
                if (!check(localsInRange, clusterLabel + " triangle " + std::to_string(li) + " has a local vertex index out of range")) {
                    continue;
                }

                const GroundTruthTriangle& gt = groundTruth[originalIdx];
                bool reconstructs = cluster.globalVertexIndices[lv0] == gt.v0
                    && cluster.globalVertexIndices[lv1] == gt.v1
                    && cluster.globalVertexIndices[lv2] == gt.v2;
                check(reconstructs, clusterLabel + " triangle " + std::to_string(li) + " does not reconstruct the original triangle's vertex indices");
            }

            totalClusterTriangles += triangleCount;
        }

        ok &= check(totalClusterTriangles == groundTruth.size(),
            "total clustered triangle count (" + std::to_string(totalClusterTriangles) +
            ") does not match the original triangle count (" + std::to_string(groundTruth.size()) + ")");

        bool allCovered = std::all_of(covered.begin(), covered.end(), [](bool b) { return b; });
        ok &= check(allCovered, "not every original triangle was covered by a cluster");

        return ok;
    }

} // namespace

int main() {
    int failCount = 0;

    // Scenario 1: two indexed UV-spheres sharing one vertex/index buffer, mirroring how the
    // engine packs multiple procedural entities into shared SSBOs. Exercises many recursive
    // bisection levels (dense sphere) and the meshID filter (sparse sphere / missing meshID).
    {
        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        GenerateUVSphere(0u, 60u, 60u, 2.0f, maths::vec3{ 0.0f, 0.0f, 0.0f }, vertices, indices);   // 7200 triangles
        GenerateUVSphere(1u, 10u, 10u, 1.0f, maths::vec3{ 10.0f, 0.0f, 0.0f }, vertices, indices);   //  200 triangles

        ValidatePartition("DenseSphere(meshID=0)", 0u, vertices, indices, failCount);
        ValidatePartition("SparseSphere(meshID=1)", 1u, vertices, indices, failCount);
        ValidatePartition("MissingMeshID(meshID=999)", 999u, vertices, indices, failCount);
    }

    // Scenario 2: degenerate mesh — every triangle has 3 brand-new, never-shared vertices, all
    // collapsed onto the exact same position. All triangle centroids are therefore identical,
    // which drives Lloyd's k-means to assign everything to a single side (distances tie at 0),
    // exercising the deterministic spatial-sort fallback bisection in SplitGroupInTwo.
    {
        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        constexpr uint32_t meshID = 5u;
        constexpr uint32_t triangleCount = 300u;
        for (uint32_t t = 0; t < triangleCount; ++t) {
            for (int k = 0; k < 3; ++k) {
                renderer::Vertex vert{};
                vert.position = maths::vec3{ 0.0f, 0.0f, 0.0f };
                vert.normal = maths::vec3{ 0.0f, 1.0f, 0.0f };
                vert.meshID = meshID;
                indices.push_back(static_cast<uint32_t>(vertices.size()));
                vertices.push_back(vert);
            }
        }
        ValidatePartition("DegenerateCollapsedMesh(meshID=5)", meshID, vertices, indices, failCount);
    }

    // Scenario 3: a single triangle — the recursion's trivial base case.
    {
        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        for (int k = 0; k < 3; ++k) {
            renderer::Vertex vert{};
            vert.position = maths::vec3{ static_cast<float>(k), 0.0f, 0.0f };
            vert.normal = maths::vec3{ 0.0f, 1.0f, 0.0f };
            vert.meshID = 7u;
            indices.push_back(static_cast<uint32_t>(vertices.size()));
            vertices.push_back(vert);
        }
        ValidatePartition("SingleTriangle(meshID=7)", 7u, vertices, indices, failCount);
    }

    if (failCount == 0) {
        std::cout << "[ClusterPartitionerTests] All checks PASSED.\n";
        return 0;
    }

    std::cerr << "[ClusterPartitionerTests] " << failCount << " check(s) FAILED.\n";
    return 1;
}
