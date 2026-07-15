// Standalone, framework-free unit test for the cluster DAG builder/validator
// (src/geometry/ClusterDAG.h/.cpp). Purely CPU-side, no Vulkan/GLFW required. Exits 0 if every
// check passes, non-zero otherwise, so it can be registered with CTest without pulling in any
// external test framework.
//
// Two kinds of scenarios are covered:
//   1. A real multi-level DAG built by BuildClusterDAG from a synthetic sphere, checked for every
//      structural/error invariant a valid DAG must satisfy, PLUS a full ValidateClusterDAG pass
//      expected to report zero errors.
//   2. Small, hand-crafted DAGs that are deliberately tampered with (a self-referential cycle, an
//      inverted e_parent < e_local, an unreachable orphan node) to prove ValidateClusterDAG
//      actually detects each specific class of corruption rather than trivially accepting
//      anything -- a validator that never fails is not a validator.

#include "geometry/ClusterDAG.h"
#include "geometry/ClusterFormat.h"
#include "geometry/ClusterPartitioner.h"
#include "core/maths/Maths.h"
#include "renderer/RenderTypes.h"
#include "SyntheticMesh.h"

#include <algorithm>
#include <cmath>
#include <iostream>
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

    // -------------------------------------------------------------------------------------
    // A tiny, hand-crafted, already-valid 3-node DAG (2 leaves + 1 parent) used as a clean
    // baseline for the tamper-and-detect scenarios below. Geometry content is irrelevant here
    // (ValidateClusterDAG only inspects childIndices/parentIndex/clusterError/parentError), so
    // each mesh is a single arbitrary triangle.
    // -------------------------------------------------------------------------------------
    geometry::ClusterDAG MakeTinyValidDAG() {
        geometry::ClusterDAG dag;

        geometry::ClusterDAGNode leaf0;
        leaf0.level = 0;
        leaf0.clusterError = 0.0f;
        leaf0.mesh.positions = { maths::vec3{0,0,0}, maths::vec3{1,0,0}, maths::vec3{0,1,0} };
        leaf0.mesh.locked = { false, false, false };
        leaf0.mesh.triangles = { 0u, 1u, 2u };
        dag.nodes.push_back(leaf0); // index 0

        geometry::ClusterDAGNode leaf1;
        leaf1.level = 0;
        leaf1.clusterError = 0.0f;
        leaf1.mesh.positions = { maths::vec3{1,0,0}, maths::vec3{1,1,0}, maths::vec3{0,1,0} };
        leaf1.mesh.locked = { false, false, false };
        leaf1.mesh.triangles = { 0u, 1u, 2u };
        dag.nodes.push_back(leaf1); // index 1

        geometry::ClusterDAGNode parent;
        parent.level = 1;
        parent.clusterError = 0.5f;
        parent.childIndices = { 0u, 1u };
        parent.mesh.positions = { maths::vec3{0,0,0}, maths::vec3{1,1,0}, maths::vec3{0,1,0} };
        parent.mesh.locked = { false, false, false };
        parent.mesh.triangles = { 0u, 1u, 2u };
        dag.nodes.push_back(parent); // index 2

        dag.nodes[0].parentIndex = 2u;
        dag.nodes[0].parentError = 0.5f;
        dag.nodes[1].parentIndex = 2u;
        dag.nodes[1].parentError = 0.5f;
        dag.rootIndices = { 2u };

        return dag;
    }

    void RunTamperScenarios() {
        // Baseline: the hand-crafted DAG itself must validate cleanly, otherwise the scenarios
        // below would be testing a broken baseline instead of isolating each specific tamper.
        {
            geometry::ClusterDAG dag = MakeTinyValidDAG();
            std::vector<std::string> errors;
            bool valid = geometry::ValidateClusterDAG(dag, errors);
            Check(valid, "MakeTinyValidDAG's baseline is expected to validate cleanly");
            Check(errors.empty(), "MakeTinyValidDAG's baseline produced unexpected validation errors");
        }

        // Scenario: self-referential cycle (a node lists itself as its own child).
        {
            geometry::ClusterDAG dag = MakeTinyValidDAG();
            dag.nodes[2].childIndices.push_back(2u); // Parent claims itself as a third child.

            std::vector<std::string> errors;
            bool valid = geometry::ValidateClusterDAG(dag, errors);
            Check(!valid, "a self-referential cycle must fail validation");

            bool mentionsCycle = false;
            for (const std::string& e : errors) {
                if (e.find("cycle") != std::string::npos) { mentionsCycle = true; break; }
            }
            Check(mentionsCycle, "a self-referential cycle's error list should mention 'cycle'");
        }

        // Scenario: inverted error monotonicity (a leaf's clusterError exceeds its parent's).
        {
            geometry::ClusterDAG dag = MakeTinyValidDAG();
            dag.nodes[0].clusterError = dag.nodes[2].clusterError + 1.0f; // e_local > e_parent: illegal.

            std::vector<std::string> errors;
            bool valid = geometry::ValidateClusterDAG(dag, errors);
            Check(!valid, "an inverted e_parent < e_local must fail validation");

            bool mentionsMonotonicity = false;
            for (const std::string& e : errors) {
                if (e.find("monoton") != std::string::npos) { mentionsMonotonicity = true; break; }
            }
            Check(mentionsMonotonicity, "an inverted error relationship's error list should mention monotonicity");
        }

        // Scenario: an orphan node present in dag.nodes but unreachable from any declared root.
        {
            geometry::ClusterDAG dag = MakeTinyValidDAG();
            geometry::ClusterDAGNode orphan;
            orphan.level = 0;
            orphan.clusterError = 0.0f;
            dag.nodes.push_back(orphan); // Index 3: never referenced by any child/parent/root link.

            std::vector<std::string> errors;
            bool valid = geometry::ValidateClusterDAG(dag, errors);
            Check(!valid, "an unreachable orphan node must fail validation");

            bool mentionsOrphan = false;
            for (const std::string& e : errors) {
                if (e.find("not reachable") != std::string::npos) { mentionsOrphan = true; break; }
            }
            Check(mentionsOrphan, "an orphan node's error list should mention unreachability");
        }
    }

    void RunRealSphereScenario() {
        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        GenerateUVSphere(0u, 40u, 40u, 3.0f, maths::vec3{ 0.0f, 0.0f, 0.0f }, vertices, indices); // 3200 triangles

        std::vector<geometry::MeshCluster> expectedLeaves = geometry::PartitionMeshIntoClusters(0u, vertices, indices);
        Check(!expectedLeaves.empty(), "expected at least one leaf cluster from the synthetic sphere");

        geometry::ClusterDAG dag = geometry::BuildClusterDAG(0u, vertices, indices);
        Check(!dag.nodes.empty(), "expected a non-empty DAG");
        Check(!dag.rootIndices.empty(), "expected at least one root");

        std::vector<std::string> errors;
        bool valid = geometry::ValidateClusterDAG(dag, errors);
        for (const std::string& e : errors) {
            std::cerr << "[ClusterDAGTests] validator reported: " << e << "\n";
        }
        Check(valid, "ValidateClusterDAG must report the real sphere's DAG as fully valid");

        uint32_t leafCount = 0;
        uint32_t internalCount = 0;
        uint32_t maxLevel = 0;

        for (size_t i = 0; i < dag.nodes.size(); ++i) {
            const geometry::ClusterDAGNode& node = dag.nodes[i];
            const std::string label = "dag node " + std::to_string(i);
            maxLevel = std::max(maxLevel, node.level);

            if (node.level == 0) {
                ++leafCount;
                Check(node.childIndices.empty(), label + ": a level-0 leaf must have no children");
                Check(node.clusterError == 0.0f, label + ": a level-0 leaf must have zero clusterError");
            }
            else {
                ++internalCount;
                Check(!node.childIndices.empty() && node.childIndices.size() <= 2,
                    label + ": an internal node must have 1 or 2 children");

                uint32_t maxChildLevel = 0;
                float maxChildError = 0.0f;
                for (uint32_t childIdx : node.childIndices) {
                    Check(childIdx < dag.nodes.size(), label + ": child index out of range");
                    if (childIdx < dag.nodes.size()) {
                        maxChildLevel = std::max(maxChildLevel, dag.nodes[childIdx].level);
                        maxChildError = std::max(maxChildError, dag.nodes[childIdx].clusterError);
                    }
                }
                Check(node.level == maxChildLevel + 1u, label + ": level must be exactly 1 + max child level");
                Check(node.clusterError > maxChildError,
                    label + ": clusterError must strictly exceed every child's clusterError");
            }

            if (node.parentIndex == geometry::kInvalidDAGNodeIndex) {
                Check(std::isinf(node.parentError), label + ": a root's parentError must be +infinity");
            }
            else {
                Check(node.parentIndex < dag.nodes.size(), label + ": parentIndex out of range");
                Check(node.parentError > node.clusterError, label + ": parentError must strictly exceed clusterError");
            }
        }

        Check(leafCount == expectedLeaves.size(),
            "DAG leaf count (" + std::to_string(leafCount) + ") does not match PartitionMeshIntoClusters's cluster count (" +
            std::to_string(expectedLeaves.size()) + ")");

        std::cout << "[ClusterDAGTests] " << leafCount << " leaf node(s), " << internalCount
            << " internal node(s), " << dag.rootIndices.size() << " root(s), max level " << maxLevel << ".\n";

        for (uint32_t rootIdx : dag.rootIndices) {
            const geometry::ClusterDAGNode& root = dag.nodes[rootIdx];
            std::cout << "[ClusterDAGTests] root " << rootIdx << ": level " << root.level
                << ", clusterError " << root.clusterError << ", "
                << (root.mesh.triangles.size() / 3) << " triangle(s).\n";
        }
    }

} // namespace

int main() {
    RunRealSphereScenario();
    RunTamperScenarios();

    if (g_failCount == 0) {
        std::cout << "[ClusterDAGTests] All checks PASSED.\n";
        return 0;
    }

    std::cerr << "[ClusterDAGTests] " << g_failCount << " check(s) FAILED.\n";
    return 1;
}
