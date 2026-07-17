// Standalone, framework-free unit test for the opaque/masked cluster split (src/geometry/
// ClusterPartitioner.h/.cpp's opacity classification + split, ClusterGrouping.cpp/ClusterDAG.cpp's
// cross-classification adjacency filtering, and ValidateClusterDAG's isMasked-purity invariant).
//
// Purely CPU-side, no Vulkan/GLFW required. Exits 0 if every check passes, non-zero otherwise, so
// it can be registered with CTest without pulling in any external test framework.
//
// What is validated:
//   1. PartitionMeshIntoClusters, called with geometry::kInvalidMaskTextureIndex, never analyzes
//      opacity at all (every cluster stays isMasked == false regardless of UVs) -- the zero-cost
//      fast path for entities with no cutout material.
//   2. A cluster whose masked-triangle fraction exceeds kMaskedClusterSplitThreshold splits into
//      exactly one pure-opaque (isMasked == false) and one pure-masked (isMasked == true) cluster,
//      and the union of both replacements' originalTriangleIndices still covers [0, N) with no gaps
//      or duplicates.
//   3. A cluster at or below the threshold stays a single cluster, tagged isMasked == true if it has
//      any masked triangle at all, or isMasked == false if it has none.
//   4. A full BuildClusterDAG over a real, spatially-connected mesh (a UV sphere) with a real mask
//      slot: ValidateClusterDAG reports zero errors (the isMasked-purity invariant holds throughout
//      construction), and the opaque/masked classes never merged into a shared root.
//   5. A hand-tampered DAG with a masked leaf under an opaque parent is correctly rejected by
//      ValidateClusterDAG's isMasked-purity check (tamper-and-detect, mirrors ClusterDAGTests.cpp's
//      cycle/error-monotonicity tests).

#include "geometry/ClusterDAG.h"
#include "geometry/ClusterFormat.h"
#include "geometry/ClusterPartitioner.h"
#include "geometry/ProceduralMaskSampler.h"
#include "core/maths/Maths.h"
#include "renderer/RenderTypes.h"
#include "SyntheticMesh.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_set>
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

    // Appends one fresh (non-shared) triangle with all 3 vertices stamped to the exact same UV --
    // since ClusterPartitioner's opacity sampler barycentrically interpolates a triangle's 3 corner
    // UVs, giving all 3 the identical value makes every sample point land on that exact UV, so the
    // triangle's classification is deterministically whatever that one point classifies as.
    void AddUniformUVTriangle(uint32_t meshID, float positionOffset, const maths::vec2& uv,
        std::vector<renderer::Vertex>& outVertices, std::vector<uint32_t>& outIndices) {

        uint32_t base = static_cast<uint32_t>(outVertices.size());
        maths::vec3 p0{ positionOffset, 0.0f, 0.0f };
        maths::vec3 p1{ positionOffset + 0.1f, 0.0f, 0.0f };
        maths::vec3 p2{ positionOffset, 0.1f, 0.0f };

        for (const maths::vec3& p : { p0, p1, p2 }) {
            renderer::Vertex v{};
            v.position = p;
            v.materialID = 0.0f;
            v.normal = maths::vec3{ 0.0f, 0.0f, 1.0f };
            v.meshID = meshID;
            v.uv = uv;
            v.uv2 = uv;
            outVertices.push_back(v);
        }
        outIndices.push_back(base + 0u);
        outIndices.push_back(base + 1u);
        outIndices.push_back(base + 2u);
    }

    // Scans a coarse UV grid against the real procedural mask formula (the same one
    // ClusterPartitioner.cpp classifies triangles with) and returns up to `count` UV points whose
    // classification matches `wantMasked` -- lets the test pick real, verifiable opaque/masked UV
    // coordinates instead of guessing at the noise pattern's shape.
    std::vector<maths::vec2> FindUVPoints(uint32_t maskTextureIndex, bool wantMasked, int count) {
        std::vector<maths::vec2> found;
        constexpr int kGridSize = 200;
        for (int gy = 0; gy < kGridSize && static_cast<int>(found.size()) < count; ++gy) {
            for (int gx = 0; gx < kGridSize && static_cast<int>(found.size()) < count; ++gx) {
                maths::vec2 uv{ (static_cast<float>(gx) + 0.5f) / static_cast<float>(kGridSize),
                                 (static_cast<float>(gy) + 0.5f) / static_cast<float>(kGridSize) };
                bool masked = geometry::SampleMaskAlphaCPU(maskTextureIndex, uv) < geometry::kMaskAlphaCutoff;
                if (masked == wantMasked) {
                    found.push_back(uv);
                }
            }
        }
        return found;
    }

    std::unordered_set<uint32_t> CollectOriginalTriangleIndices(const std::vector<geometry::MeshCluster>& clusters) {
        std::unordered_set<uint32_t> covered;
        for (const geometry::MeshCluster& cluster : clusters) {
            for (uint32_t idx : cluster.originalTriangleIndices) {
                covered.insert(idx);
            }
        }
        return covered;
    }

    void CheckFullCoverage(const std::vector<geometry::MeshCluster>& clusters, uint32_t expectedTriangleCount, const std::string& label) {
        std::unordered_set<uint32_t> covered = CollectOriginalTriangleIndices(clusters);
        size_t totalEntries = 0;
        for (const geometry::MeshCluster& c : clusters) {
            totalEntries += c.originalTriangleIndices.size();
        }
        Check(covered.size() == expectedTriangleCount, label + ": expected " + std::to_string(expectedTriangleCount) +
            " distinct original triangle indices covered, got " + std::to_string(covered.size()));
        Check(totalEntries == covered.size(), label + ": originalTriangleIndices entries must be unique across clusters (no duplicates)");
        for (uint32_t i = 0; i < expectedTriangleCount; ++i) {
            if (covered.find(i) == covered.end()) {
                Check(false, label + ": original triangle " + std::to_string(i) + " missing from every output cluster");
                break;
            }
        }
    }

    // --- Scenario 1: kInvalidMaskTextureIndex skips analysis entirely ------------------------------
    void RunNoMaskEntityScenario() {
        constexpr uint32_t kMaskSlot = 0u;
        std::vector<maths::vec2> maskedUVs = FindUVPoints(kMaskSlot, true, 5);
        Check(maskedUVs.size() == 5, "setup: expected to find 5 masked UV points for slot 0");

        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        for (size_t i = 0; i < maskedUVs.size(); ++i) {
            AddUniformUVTriangle(0u, static_cast<float>(i) * 2.0f, maskedUVs[i], vertices, indices);
        }

        std::vector<geometry::MeshCluster> clusters =
            geometry::PartitionMeshIntoClusters(0u, vertices, indices, geometry::kInvalidMaskTextureIndex);

        Check(clusters.size() == 1u, "no-mask-entity: expected exactly one cluster (no split without a mask slot)");
        if (!clusters.empty()) {
            Check(clusters[0].isMasked == false, "no-mask-entity: cluster must stay isMasked==false even with UVs that would classify as masked");
        }
    }

    // --- Scenario 2: masked fraction exceeds the threshold -> split into two pure clusters --------
    void RunAboveThresholdSplitScenario() {
        constexpr uint32_t kMaskSlot = 0u;
        std::vector<maths::vec2> opaqueUVs = FindUVPoints(kMaskSlot, false, 12);
        std::vector<maths::vec2> maskedUVs = FindUVPoints(kMaskSlot, true, 6);
        Check(opaqueUVs.size() == 12, "above-threshold: expected to find 12 opaque UV points");
        Check(maskedUVs.size() == 6, "above-threshold: expected to find 6 masked UV points");

        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        float offset = 0.0f;
        for (const maths::vec2& uv : opaqueUVs) { AddUniformUVTriangle(0u, offset, uv, vertices, indices); offset += 2.0f; }
        for (const maths::vec2& uv : maskedUVs) { AddUniformUVTriangle(0u, offset, uv, vertices, indices); offset += 2.0f; }
        uint32_t totalTriangles = static_cast<uint32_t>(opaqueUVs.size() + maskedUVs.size()); // 18 total, 6/18 = 33% > 10%

        std::vector<geometry::MeshCluster> clusters =
            geometry::PartitionMeshIntoClusters(0u, vertices, indices, kMaskSlot);

        Check(clusters.size() == 2u, "above-threshold: expected exactly two clusters after the opacity split");
        if (clusters.size() == 2u) {
            const geometry::MeshCluster* opaqueCluster = clusters[0].isMasked ? &clusters[1] : &clusters[0];
            const geometry::MeshCluster* maskedCluster = clusters[0].isMasked ? &clusters[0] : &clusters[1];

            Check(opaqueCluster->isMasked == false, "above-threshold: one output cluster must be pure opaque");
            Check(maskedCluster->isMasked == true, "above-threshold: the other output cluster must be pure masked");
            Check(opaqueCluster->originalTriangleIndices.size() == opaqueUVs.size(),
                "above-threshold: opaque cluster triangle count must equal the number of opaque-UV triangles injected");
            Check(maskedCluster->originalTriangleIndices.size() == maskedUVs.size(),
                "above-threshold: masked cluster triangle count must equal the number of masked-UV triangles injected");
        }

        CheckFullCoverage(clusters, totalTriangles, "above-threshold");
    }

    // --- Scenario 3a: masked fraction at/below the threshold -> stays merged, tagged masked -------
    void RunAtOrBelowThresholdScenario() {
        constexpr uint32_t kMaskSlot = 0u;
        std::vector<maths::vec2> opaqueUVs = FindUVPoints(kMaskSlot, false, 19);
        std::vector<maths::vec2> maskedUVs = FindUVPoints(kMaskSlot, true, 1);
        Check(opaqueUVs.size() == 19, "at-or-below-threshold: expected to find 19 opaque UV points");
        Check(maskedUVs.size() == 1, "at-or-below-threshold: expected to find 1 masked UV point");

        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        float offset = 0.0f;
        for (const maths::vec2& uv : opaqueUVs) { AddUniformUVTriangle(0u, offset, uv, vertices, indices); offset += 2.0f; }
        for (const maths::vec2& uv : maskedUVs) { AddUniformUVTriangle(0u, offset, uv, vertices, indices); offset += 2.0f; }
        uint32_t totalTriangles = static_cast<uint32_t>(opaqueUVs.size() + maskedUVs.size()); // 20 total, 1/20 = 5% <= 10%

        std::vector<geometry::MeshCluster> clusters =
            geometry::PartitionMeshIntoClusters(0u, vertices, indices, kMaskSlot);

        Check(clusters.size() == 1u, "at-or-below-threshold: expected exactly one cluster (below the split threshold)");
        if (!clusters.empty()) {
            Check(clusters[0].isMasked == true, "at-or-below-threshold: cluster with any masked triangle at all must be tagged isMasked==true");
        }
        CheckFullCoverage(clusters, totalTriangles, "at-or-below-threshold");
    }

    // --- Scenario 3b: zero masked triangles -> stays merged, tagged opaque -------------------------
    void RunZeroMaskedScenario() {
        constexpr uint32_t kMaskSlot = 0u;
        std::vector<maths::vec2> opaqueUVs = FindUVPoints(kMaskSlot, false, 10);
        Check(opaqueUVs.size() == 10, "zero-masked: expected to find 10 opaque UV points");

        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        float offset = 0.0f;
        for (const maths::vec2& uv : opaqueUVs) { AddUniformUVTriangle(0u, offset, uv, vertices, indices); offset += 2.0f; }

        std::vector<geometry::MeshCluster> clusters =
            geometry::PartitionMeshIntoClusters(0u, vertices, indices, kMaskSlot);

        Check(clusters.size() == 1u, "zero-masked: expected exactly one cluster");
        if (!clusters.empty()) {
            Check(clusters[0].isMasked == false, "zero-masked: cluster with no masked triangle must stay isMasked==false");
        }
        CheckFullCoverage(clusters, static_cast<uint32_t>(opaqueUVs.size()), "zero-masked");
    }

    // --- Scenario 4: full DAG over a real, spatially-connected mesh --------------------------------
    void RunFullDagPurityScenario() {
        constexpr uint32_t kMaskSlot = 0u;

        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;
        GenerateUVSphere(0u, 24u, 24u, 3.0f, maths::vec3{ 0.0f, 0.0f, 0.0f }, vertices, indices);
        Check(!vertices.empty(), "full-dag: expected a non-empty synthetic sphere");

        geometry::ClusterDAG dag = geometry::BuildClusterDAG(0u, vertices, indices, kMaskSlot);
        Check(!dag.nodes.empty(), "full-dag: expected a non-empty DAG");
        Check(!dag.rootIndices.empty(), "full-dag: expected at least one root");

        bool sawOpaqueLeaf = false;
        bool sawMaskedLeaf = false;
        for (const geometry::ClusterDAGNode& node : dag.nodes) {
            if (node.level != 0u) {
                continue;
            }
            (node.isMasked ? sawMaskedLeaf : sawOpaqueLeaf) = true;
        }
        Check(sawOpaqueLeaf, "full-dag: expected at least one opaque leaf cluster on the synthetic sphere (mask slot 0 covers the full [0,1] UV range)");
        Check(sawMaskedLeaf, "full-dag: expected at least one masked leaf cluster on the synthetic sphere (mask slot 0 covers the full [0,1] UV range)");

        // Opaque and masked leaves can never share a parent (BuildClusterAdjacencyWeights/
        // BuildLevelAdjacencyWeights filter cross-classification pairs at every level), so if both
        // classes exist among the leaves, at least two distinct roots must exist: one lineage per
        // classification (each lineage may itself still merge down to a single root).
        if (sawOpaqueLeaf && sawMaskedLeaf) {
            Check(dag.rootIndices.size() >= 2u,
                "full-dag: expected at least two roots when both opaque and masked leaves are present (they must never share a root)");
        }

        std::vector<std::string> errors;
        bool valid = geometry::ValidateClusterDAG(dag, errors);
        for (const std::string& e : errors) {
            std::cerr << "[ClusterMaskSplitTests] validator reported: " << e << "\n";
        }
        Check(valid, "full-dag: ValidateClusterDAG must report zero errors, including the isMasked-purity invariant");
    }

    // --- Scenario 5: tamper-and-detect for the isMasked-purity invariant ---------------------------
    void RunTamperedPurityScenario() {
        geometry::ClusterDAG dag;

        geometry::ClusterDAGNode leaf0;
        leaf0.level = 0;
        leaf0.clusterError = 0.0f;
        leaf0.isMasked = false; // Opaque leaf...
        leaf0.mesh.positions = { maths::vec3{0,0,0}, maths::vec3{1,0,0}, maths::vec3{0,1,0} };
        leaf0.mesh.locked = { false, false, false };
        leaf0.mesh.triangles = { 0u, 1u, 2u };
        dag.nodes.push_back(leaf0); // index 0

        geometry::ClusterDAGNode leaf1;
        leaf1.level = 0;
        leaf1.clusterError = 0.0f;
        leaf1.isMasked = true; // ...deliberately paired under an opaque parent below: this pairing
                                // could never happen via BuildClusterDAG itself (the adjacency filter
                                // would reject it), which is exactly what this hand-crafted DAG tests.
        leaf1.mesh.positions = { maths::vec3{1,0,0}, maths::vec3{1,1,0}, maths::vec3{0,1,0} };
        leaf1.mesh.locked = { false, false, false };
        leaf1.mesh.triangles = { 0u, 1u, 2u };
        dag.nodes.push_back(leaf1); // index 1

        geometry::ClusterDAGNode parent;
        parent.level = 1;
        parent.clusterError = 0.5f;
        parent.isMasked = false; // Matches leaf0, but not leaf1 -- the tamper.
        parent.sourceGroupIndex = 0u;
        parent.mesh.positions = { maths::vec3{0,0,0}, maths::vec3{1,1,0}, maths::vec3{0,1,0} };
        parent.mesh.locked = { false, false, false };
        parent.mesh.triangles = { 0u, 1u, 2u };
        dag.nodes.push_back(parent); // index 2

        geometry::ClusterDAGGroup group;
        group.memberClusterIndices = { 0u, 1u };
        group.outputClusterIndices = { 2u };
        group.groupError = 0.5f;
        dag.groups.push_back(group); // index 0

        dag.nodes[0].parentGroupIndex = 0u;
        dag.nodes[0].parentError = 0.5f;
        dag.nodes[1].parentGroupIndex = 0u;
        dag.nodes[1].parentError = 0.5f;
        dag.rootIndices = { 2u };

        std::vector<std::string> errors;
        bool valid = geometry::ValidateClusterDAG(dag, errors);
        Check(!valid, "tampered-purity: a masked leaf under an opaque parent must be rejected");

        bool mentionsIsMasked = false;
        for (const std::string& e : errors) {
            if (e.find("isMasked") != std::string::npos) { mentionsIsMasked = true; break; }
        }
        Check(mentionsIsMasked, "tampered-purity: the error list should mention isMasked");
    }

} // namespace

int main() {
    RunNoMaskEntityScenario();
    RunAboveThresholdSplitScenario();
    RunAtOrBelowThresholdScenario();
    RunZeroMaskedScenario();
    RunFullDagPurityScenario();
    RunTamperedPurityScenario();

    if (g_failCount == 0) {
        std::cout << "[ClusterMaskSplitTests] All checks passed.\n";
        return 0;
    }
    std::cerr << "[ClusterMaskSplitTests] " << g_failCount << " check(s) failed.\n";
    return 1;
}
