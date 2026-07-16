// Standalone, framework-free unit test for the cluster-group QEM simplification pipeline:
//   geometry::PartitionMeshIntoClusters (ClusterPartitioner.h)
//     -> geometry::GroupAdjacentClusters   (ClusterGrouping.h)
//     -> geometry::SimplifyMeshQEM         (MeshSimplifier.h)
//     -> geometry::ExportClusterGroupsToOBJ(ObjExport.h)
//
// Purely CPU-side, no Vulkan/GLFW required. Exits 0 if every check passes, non-zero otherwise, so
// it can be registered with CTest without pulling in any external test framework.
//
// What is validated:
//   1. Grouping preserves every triangle: sum of group.originalTriangleCount across all groups
//      equals the sum of triangle counts across all input clusters (no triangle lost/duplicated
//      while merging cluster pairs into groups).
//   2. Simplification never increases a group's triangle count.
//   3. THE crack-prevention contract: every vertex marked locked (i.e. on the group's outer
//      boundary, shared with a neighboring cluster that was not merged into this group) keeps
//      its exact pre-simplification position and is still referenced by at least one triangle
//      after simplification -- this is what guarantees neighboring groups can still stitch
//      against this group's boundary with zero gap.
//   4. A debug .obj file containing every simplified group is written so the crack-free result
//      (or lack thereof) can additionally be inspected visually in a 3D viewer.

#include "geometry/ClusterFormat.h"
#include "geometry/ClusterGrouping.h"
#include "geometry/ClusterPartitioner.h"
#include "geometry/MeshSimplifier.h"
#include "io/ObjExport.h"
#include "core/maths/Maths.h"
#include "renderer/RenderTypes.h"
#include "SyntheticMesh.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

    using testutil::GenerateUVSphere;

    // Counts how many times `target` appears, bit-exactly, among `positions`.
    uint32_t CountExactPositionOccurrences(const std::vector<maths::vec3>& positions, const maths::vec3& target) {
        uint32_t count = 0;
        for (const maths::vec3& p : positions) {
            if (p.x == target.x && p.y == target.y && p.z == target.z) {
                ++count;
            }
        }
        return count;
    }

} // namespace

int main() {
    int failCount = 0;
    auto check = [&](bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++failCount;
        }
        return condition;
        };

    // A single, reasonably dense UV-sphere: closed/watertight, big enough to be partitioned into
    // several kMaxClusterTriangles-capped clusters and paired into multiple groups, so both the
    // grouping and the simplification logic get real, non-trivial adjacency to work with.
    std::vector<renderer::Vertex> vertices;
    std::vector<uint32_t> indices;
    GenerateUVSphere(0u, 40u, 40u, 3.0f, maths::vec3{ 0.0f, 0.0f, 0.0f }, vertices, indices); // 3200 triangles

    std::vector<geometry::MeshCluster> clusters = geometry::PartitionMeshIntoClusters(0u, vertices, indices, geometry::kInvalidMaskTextureIndex);
    check(!clusters.empty(), "expected at least one cluster from the synthetic sphere");

    uint32_t totalClusterTriangles = 0;
    for (const geometry::MeshCluster& c : clusters) {
        totalClusterTriangles += static_cast<uint32_t>(c.localTriangleIndices.size() / 3u);
    }

    std::vector<geometry::ClusterGroup> groups = geometry::GroupAdjacentClusters(clusters, vertices);
    check(!groups.empty(), "expected at least one cluster group");

    uint32_t singletonGroups = 0;
    for (const geometry::ClusterGroup& g : groups) {
        if (g.memberClusterIndices.size() == 1) {
            ++singletonGroups;
        }
    }

    // --- Invariant 1: grouping loses/duplicates no triangle -------------------------------
    uint32_t totalGroupTriangles = 0;
    for (const geometry::ClusterGroup& g : groups) {
        totalGroupTriangles += g.originalTriangleCount;
    }
    check(totalGroupTriangles == totalClusterTriangles,
        "grouped triangle count (" + std::to_string(totalGroupTriangles) +
        ") does not match total clustered triangle count (" + std::to_string(totalClusterTriangles) + ")");

    // --- Simplify every group to half its triangle count, validating invariants 2 and 3 ----
    uint32_t totalOriginal = 0;
    uint32_t totalSimplified = 0;
    uint32_t groupsReachedTarget = 0;

    std::vector<geometry::ClusterGroup> groupsCopyForExport; // Holds the post-simplification meshes for the OBJ export.
    groupsCopyForExport.reserve(groups.size());

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        geometry::ClusterGroup& group = groups[gi];
        const std::string label = "group " + std::to_string(gi);

        uint32_t originalTriangleCount = static_cast<uint32_t>(group.mesh.triangles.size() / 3u);
        check(originalTriangleCount == group.originalTriangleCount, label + " mesh triangle count does not match originalTriangleCount");

        // Snapshot every locked vertex's position before simplification touches anything.
        std::vector<maths::vec3> lockedPositionsBefore;
        for (size_t v = 0; v < group.mesh.positions.size(); ++v) {
            if (group.mesh.locked[v]) {
                lockedPositionsBefore.push_back(group.mesh.positions[v]);
            }
        }

        // Halve the triangle count -- with kMaxClusterTriangles clusters (128 triangles) paired
        // up, this is the requested 256 -> 128 reduction; for smaller/odd-sized groups it scales
        // proportionally (ceil of half).
        uint32_t targetTriangleCount = (originalTriangleCount + 1u) / 2u;
        uint32_t simplifiedTriangleCount = geometry::SimplifyMeshQEM(group.mesh, targetTriangleCount);

        // --- Invariant 2: simplification never increases triangle count -------------------
        check(simplifiedTriangleCount <= originalTriangleCount,
            label + " triangle count increased after simplification: " +
            std::to_string(originalTriangleCount) + " -> " + std::to_string(simplifiedTriangleCount));
        check(group.mesh.triangles.size() == static_cast<size_t>(simplifiedTriangleCount) * 3u,
            label + " mesh.triangles size is inconsistent with the returned triangle count");

        if (simplifiedTriangleCount <= targetTriangleCount) {
            ++groupsReachedTarget;
        }
        else if (!lockedPositionsBefore.empty() && lockedPositionsBefore.size() >= group.mesh.positions.size()) {
            // Fully (or almost fully) locked groups are legitimately unable to reach the target;
            // this is expected and not a failure -- only logged for visibility below.
        }

        // --- Invariant 3: every locked vertex kept its exact position and is still present -
        for (const maths::vec3& lockedPos : lockedPositionsBefore) {
            bool stillPresentAndLocked = false;
            for (size_t v = 0; v < group.mesh.positions.size(); ++v) {
                if (!group.mesh.locked[v]) {
                    continue;
                }
                const maths::vec3& p = group.mesh.positions[v];
                if (p.x == lockedPos.x && p.y == lockedPos.y && p.z == lockedPos.z) {
                    stillPresentAndLocked = true;
                    break;
                }
            }
            check(stillPresentAndLocked, label + " lost or moved a locked boundary vertex during simplification (crack risk)");
        }

        // Every remaining vertex (locked or not) must still be referenced by at least one
        // triangle -- SimplifyMeshQEM's compaction step is expected to drop orphans.
        std::vector<bool> referenced(group.mesh.positions.size(), false);
        for (uint32_t idx : group.mesh.triangles) {
            check(idx < group.mesh.positions.size(), label + " has an out-of-range triangle index after compaction");
            if (idx < group.mesh.positions.size()) {
                referenced[idx] = true;
            }
        }
        for (size_t v = 0; v < referenced.size(); ++v) {
            check(referenced[v], label + " retained an orphan (unreferenced) vertex after compaction");
        }

        totalOriginal += originalTriangleCount;
        totalSimplified += simplifiedTriangleCount;
    }

    std::cout << "[ClusterSimplificationTests] " << clusters.size() << " cluster(s), "
        << groups.size() << " group(s) (" << singletonGroups << " singleton), "
        << totalOriginal << " -> " << totalSimplified << " triangles ("
        << groupsReachedTarget << "/" << groups.size() << " groups reached their target).\n";

    // --- Debug OBJ export for manual visual inspection -------------------------------------
    std::filesystem::path objPath = "cluster_group_simplification_debug.obj";
    bool exported = geometry::ExportClusterGroupsToOBJ(objPath, groups);
    check(exported, "failed to write the debug OBJ export to '" + objPath.string() + "'");
    if (exported) {
        std::error_code ec;
        std::filesystem::path absolutePath = std::filesystem::absolute(objPath, ec);
        std::cout << "[ClusterSimplificationTests] Debug mesh exported to: "
            << (ec ? objPath.string() : absolutePath.string())
            << " -- open it in Blender/MeshLab and confirm no gaps appear between group boundaries.\n";
    }

    if (failCount == 0) {
        std::cout << "[ClusterSimplificationTests] All checks PASSED.\n";
        return 0;
    }

    std::cerr << "[ClusterSimplificationTests] " << failCount << " check(s) FAILED.\n";
    return 1;
}
