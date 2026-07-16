#include "geometry/FallbackMeshBuilder.h"
#include "core/Logger.h"
#include "geometry/GeometryHashUtil.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <unordered_map>

#include "geometry/MeshSimplifier.h"

namespace geometry {

    namespace {

        // Merges every dag.rootIndices mesh into one SimplifiableMesh, welding ALL coincident
        // vertices by exact position -- unlike ClusterDAG.cpp's own MergeLevelMeshes, which only
        // aliases LOCKED vertices across sources (because an interior-level merge must still keep
        // its own outer boundary distinguishable for later locking). Here every root is being
        // consumed into the single, final Fallback Mesh with no further merge/lock step to come, so
        // welding every shared position -- not just previously-locked ones -- yields the most
        // compact possible starting mesh for simplification. The result's `locked` is left entirely
        // false: with no sibling left outside this mesh, nothing needs to stay watertight against
        // anything.
        SimplifiableMesh MergeRootsWeldingAllVertices(const ClusterDAG& dag) {
            SimplifiableMesh merged;
            std::unordered_map<PositionKey, uint32_t, PositionKeyHash> positionToMergedIndex;

            for (uint32_t rootIdx : dag.rootIndices) {
                const SimplifiableMesh& src = dag.nodes[rootIdx].mesh;
                std::vector<uint32_t> srcToMerged(src.positions.size());

                for (size_t v = 0; v < src.positions.size(); ++v) {
                    PositionKey key = MakePositionKey(src.positions[v]);
                    auto it = positionToMergedIndex.find(key);
                    if (it != positionToMergedIndex.end()) {
                        srcToMerged[v] = it->second;
                    }
                    else {
                        uint32_t newIndex = static_cast<uint32_t>(merged.positions.size());
                        merged.positions.push_back(src.positions[v]);
                        merged.uvs.push_back(src.uvs[v]);
                        positionToMergedIndex.emplace(key, newIndex);
                        srcToMerged[v] = newIndex;
                    }
                }

                for (size_t t = 0; t + 2 < src.triangles.size(); t += 3) {
                    merged.triangles.push_back(srcToMerged[src.triangles[t + 0]]);
                    merged.triangles.push_back(srcToMerged[src.triangles[t + 1]]);
                    merged.triangles.push_back(srcToMerged[src.triangles[t + 2]]);
                }
            }

            merged.locked.assign(merged.positions.size(), false);
            return merged;
        }

        uint32_t CountLeafTriangles(const ClusterDAG& dag) {
            uint32_t count = 0u;
            for (const ClusterDAGNode& node : dag.nodes) {
                if (node.level == 0u) {
                    count += static_cast<uint32_t>(node.mesh.triangles.size() / 3u);
                }
            }
            return count;
        }

    } // namespace

    FallbackMesh BuildFallbackMesh(const ClusterDAG& dag) {
        FallbackMesh result;
        if (dag.rootIndices.empty()) {
            return result;
        }

        LOG_INFO("[FallbackMeshBuilder] Building fallback mesh...");
        SimplifiableMesh merged = MergeRootsWeldingAllVertices(dag);

        // UE5-Nanite-"Auto"-style: keep applying the same QEM error metric used everywhere else in
        // the DAG, halving the target each pass, until either a pass makes no further progress
        // (topology-limited -- e.g. every remaining vertex is essential to a non-degenerate
        // triangle) or the triangle count drops to a 1%-of-original-leaf-triangle-count safety
        // floor. No fixed target percentage is imposed; the floor only prevents a pathologically
        // simple mesh from collapsing to a near-empty proxy.
        uint32_t originalLeafTriangleCount = CountLeafTriangles(dag);
        uint32_t floorTriangleCount = std::max(1u, originalLeafTriangleCount / 100u);

        // Each pass runs on a scratch copy rather than `merged` directly: for a very small mesh, a
        // single collapse near the end can legitimately eliminate BOTH triangles sharing the
        // collapsed edge at once (e.g. exactly two triangles left, forming a hinge -- collapsing
        // their shared edge degenerates both simultaneously), overshooting straight to zero
        // triangles instead of landing near the requested target. A zero-triangle Fallback Mesh is
        // useless as a BVH proxy, so that attempt is discarded and `merged` keeps the last
        // known-good (non-empty) result instead of committing it.
        uint32_t currentTriangleCount = static_cast<uint32_t>(merged.triangles.size() / 3u);
        while (currentTriangleCount > floorTriangleCount) {
            uint32_t nextTarget = std::max(floorTriangleCount, currentTriangleCount / 2u);

            SimplifiableMesh attempt = merged;
            SimplifyMeshQEM(attempt, nextTarget, 0xFFFFFFFFu /* unbounded vertex cap */, nullptr);

            uint32_t afterTriangleCount = static_cast<uint32_t>(attempt.triangles.size() / 3u);
            if (afterTriangleCount == 0u || afterTriangleCount == currentTriangleCount) {
                break; // Either this pass would empty the mesh, or no further collapse was possible
                       // (topology-limited, not a bug -- see MeshSimplifier.h). Keep `merged` as-is.
            }
            merged = std::move(attempt);
            currentTriangleCount = afterTriangleCount;
        }

        result.positions = merged.positions;
        result.normals = ComputeFaceAccumulatedNormals(merged);
        result.uvs = merged.uvs;
        result.triangles = merged.triangles;

        LOG_INFO(std::format("[FallbackMeshBuilder] Built fallback mesh with {} vertices, {} triangles (reduced from {} original leaf triangles).", result.positions.size(), result.triangles.size() / 3u, originalLeafTriangleCount));
        return result;
    }

}
