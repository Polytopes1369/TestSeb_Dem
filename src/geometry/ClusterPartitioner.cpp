#include "geometry/ClusterPartitioner.h"

#include "geometry/ClusterFormat.h"
#include "geometry/ProceduralMaskSampler.h"
#include "core/Logger.h"

#include <algorithm>
#include <format>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace geometry {

    namespace {

        // Maximum Lloyd relaxation passes per bisection level. k-means on triangle centroids
        // converges very quickly (a handful of iterations) for the compact, locally-coherent
        // primitive meshes this engine generates; capping it bounds worst-case build time
        // without materially affecting cluster quality.
        constexpr int kMaxLloydIterations = 10;

        // One matched triangle, resolved once up front so the recursive splitter never has to
        // re-touch the (potentially large) global vertex/index arrays.
        struct TriangleRef {
            uint32_t originalIndex;
            uint32_t v0, v1, v2;
            maths::vec3 centroid;
        };

        uint32_t CountUniqueVertices(const std::vector<TriangleRef>& tris, const std::vector<uint32_t>& group) {
            std::unordered_set<uint32_t> verts;
            verts.reserve(group.size() * 3);
            for (uint32_t idx : group) {
                verts.insert(tris[idx].v0);
                verts.insert(tris[idx].v1);
                verts.insert(tris[idx].v2);
            }
            return static_cast<uint32_t>(verts.size());
        }

        float DistanceSquared(const maths::vec3& a, const maths::vec3& b) {
            maths::vec3 d = a - b;
            return d.Dot(d);
        }

        // Deterministic fallback split: sorts the group by centroid coordinate along its own
        // largest-extent axis and bisects it by count. Used whenever Lloyd's algorithm collapses
        // to a degenerate (empty-sided) assignment, e.g. when every triangle in the group shares
        // an identical centroid. Always produces two non-empty halves for group.size() >= 2,
        // which is what guarantees the recursive partitioner always makes progress and terminates.
        void SpatialSortBisect(const std::vector<TriangleRef>& tris, const std::vector<uint32_t>& group,
            std::vector<uint32_t>& outA, std::vector<uint32_t>& outB) {

            maths::vec3 lo{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
            maths::vec3 hi{ std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
            for (uint32_t idx : group) {
                const maths::vec3& c = tris[idx].centroid;
                lo.x = std::min(lo.x, c.x); lo.y = std::min(lo.y, c.y); lo.z = std::min(lo.z, c.z);
                hi.x = std::max(hi.x, c.x); hi.y = std::max(hi.y, c.y); hi.z = std::max(hi.z, c.z);
            }
            maths::vec3 extent = hi - lo;

            int axis = 0; // 0=x, 1=y, 2=z
            if (extent.y >= extent.x && extent.y >= extent.z) axis = 1;
            else if (extent.z >= extent.x && extent.z >= extent.y) axis = 2;

            std::vector<uint32_t> sorted = group;
            std::sort(sorted.begin(), sorted.end(), [&](uint32_t a, uint32_t b) {
                const maths::vec3& ca = tris[a].centroid;
                const maths::vec3& cb = tris[b].centroid;
                float va = (axis == 0) ? ca.x : (axis == 1) ? ca.y : ca.z;
                float vb = (axis == 0) ? cb.x : (axis == 1) ? cb.y : cb.z;
                return va < vb;
                });

            size_t mid = sorted.size() / 2;
            outA.assign(sorted.begin(), sorted.begin() + static_cast<long>(mid));
            outB.assign(sorted.begin() + static_cast<long>(mid), sorted.end());
        }

        // Splits `group` (size() >= 2) into two spatially coherent, non-empty halves using
        // bounded 2-means: seeds are chosen via a farthest-pair heuristic (deterministic, no
        // RNG needed for reproducible builds/tests), then relaxed with Lloyd's algorithm.
        void SplitGroupInTwo(const std::vector<TriangleRef>& tris, const std::vector<uint32_t>& group,
            std::vector<uint32_t>& outA, std::vector<uint32_t>& outB) {

            // Farthest-pair seeding: start from an arbitrary member, find the point farthest from
            // it (seedB), then the point farthest from seedB (seedA). This two-pass heuristic
            // reliably finds a well-separated pair of seeds without an O(n^2) all-pairs search.
            uint32_t seedB = group[0];
            {
                float bestDist = -1.0f;
                for (uint32_t idx : group) {
                    float d = DistanceSquared(tris[idx].centroid, tris[group[0]].centroid);
                    if (d > bestDist) { bestDist = d; seedB = idx; }
                }
            }
            uint32_t seedA = seedB;
            {
                float bestDist = -1.0f;
                for (uint32_t idx : group) {
                    float d = DistanceSquared(tris[idx].centroid, tris[seedB].centroid);
                    if (d > bestDist) { bestDist = d; seedA = idx; }
                }
            }

            maths::vec3 centroidA = tris[seedA].centroid;
            maths::vec3 centroidB = tris[seedB].centroid;

            std::vector<uint8_t> assignment(group.size(), 0);

            for (int iter = 0; iter < kMaxLloydIterations; ++iter) {
                bool changed = false;
                for (size_t i = 0; i < group.size(); ++i) {
                    const maths::vec3& c = tris[group[i]].centroid;
                    uint8_t newAssign = (DistanceSquared(c, centroidA) <= DistanceSquared(c, centroidB)) ? 0u : 1u;
                    if (newAssign != assignment[i]) {
                        changed = true;
                        assignment[i] = newAssign;
                    }
                }

                maths::vec3 sumA{ 0.0f, 0.0f, 0.0f };
                maths::vec3 sumB{ 0.0f, 0.0f, 0.0f };
                uint32_t countA = 0, countB = 0;
                for (size_t i = 0; i < group.size(); ++i) {
                    if (assignment[i] == 0u) { sumA = sumA + tris[group[i]].centroid; ++countA; }
                    else { sumB = sumB + tris[group[i]].centroid; ++countB; }
                }
                if (countA == 0u || countB == 0u) {
                    break; // Degenerate this pass; the empty-side fallback below will handle it.
                }
                centroidA = sumA * (1.0f / static_cast<float>(countA));
                centroidB = sumB * (1.0f / static_cast<float>(countB));

                if (!changed) {
                    break; // Converged.
                }
            }

            outA.clear();
            outB.clear();
            for (size_t i = 0; i < group.size(); ++i) {
                (assignment[i] == 0u ? outA : outB).push_back(group[i]);
            }

            if (outA.empty() || outB.empty()) {
                SpatialSortBisect(tris, group, outA, outB);
            }
        }

        // Recursively bisects `group` until every leaf satisfies both hard caps, appending each
        // finished leaf to outGroups. Guaranteed to terminate: SplitGroupInTwo always returns two
        // strictly smaller, non-empty halves for a group of size >= 2, and any group of size 1
        // (a single triangle: <= 3 vertices, 1 triangle) trivially satisfies both caps.
        void PartitionGroupRecursive(const std::vector<TriangleRef>& tris, std::vector<uint32_t> group,
            std::vector<std::vector<uint32_t>>& outGroups) {

            if (group.empty()) {
                return;
            }

            bool trianglesOk = group.size() <= kMaxClusterTriangles;
            bool verticesOk = CountUniqueVertices(tris, group) <= kMaxClusterVertices;
            if (trianglesOk && verticesOk) {
                outGroups.push_back(std::move(group));
                return;
            }

            std::vector<uint32_t> a, b;
            SplitGroupInTwo(tris, group, a, b);
            PartitionGroupRecursive(tris, std::move(a), outGroups);
            PartitionGroupRecursive(tris, std::move(b), outGroups);
        }

        // Converts one finished triangle group into a MeshCluster: renumbers its vertices locally
        // (first-seen order) and computes its AABB / enclosing bounding sphere.
        MeshCluster BuildClusterFromGroup(const std::vector<TriangleRef>& tris,
            const std::vector<renderer::Vertex>& allVertices,
            const std::vector<uint32_t>& group) {

            MeshCluster cluster;
            std::unordered_map<uint32_t, uint8_t> globalToLocal;
            globalToLocal.reserve(group.size() * 3);
            cluster.globalVertexIndices.reserve(std::min<size_t>(group.size() * 3, kMaxClusterVertices));
            cluster.localTriangleIndices.reserve(group.size() * 3);
            cluster.originalTriangleIndices.reserve(group.size());

            for (uint32_t triIdx : group) {
                const TriangleRef& tri = tris[triIdx];
                for (uint32_t g : { tri.v0, tri.v1, tri.v2 }) {
                    auto it = globalToLocal.find(g);
                    uint8_t local;
                    if (it == globalToLocal.end()) {
                        local = static_cast<uint8_t>(cluster.globalVertexIndices.size());
                        globalToLocal.emplace(g, local);
                        cluster.globalVertexIndices.push_back(g);
                    }
                    else {
                        local = it->second;
                    }
                    cluster.localTriangleIndices.push_back(local);
                }
                cluster.originalTriangleIndices.push_back(tri.originalIndex);
            }

            maths::vec3 boundsMin, boundsMax;
            maths::ResetAABB(boundsMin, boundsMax);
            for (uint32_t g : cluster.globalVertexIndices) {
                maths::ExpandAABB(boundsMin, boundsMax, allVertices[g].position);
            }
            cluster.boundsMin = boundsMin;
            cluster.boundsMax = boundsMax;
            cluster.sphereCenter = maths::AABBCenter(boundsMin, boundsMax);
            cluster.sphereRadius = maths::AABBRadius(boundsMin, boundsMax);

            return cluster;
        }

        // Barycentric sample grid resolution for opacity classification: a triangular grid with
        // kOpacitySampleSubdivisions steps per edge yields (N+1)(N+2)/2 = 15 sample points,
        // including all 3 corners exactly. A fixed, small sample count (rather than one driven by
        // mask texture resolution or the triangle's UV footprint size) keeps classification cost
        // bounded regardless of how large a triangle's UV footprint is.
        constexpr int kOpacitySampleSubdivisions = 4;

        // True if any of the fixed barycentric samples across this triangle's UV footprint would be
        // discarded by the alpha-test cutoff -- i.e. the triangle is not safely, uniformly opaque.
        bool IsTriangleMasked(const maths::vec2& uv0, const maths::vec2& uv1, const maths::vec2& uv2, uint32_t maskTextureIndex) {
            for (int a = 0; a <= kOpacitySampleSubdivisions; ++a) {
                for (int b = 0; a + b <= kOpacitySampleSubdivisions; ++b) {
                    float u = static_cast<float>(a) / static_cast<float>(kOpacitySampleSubdivisions);
                    float v = static_cast<float>(b) / static_cast<float>(kOpacitySampleSubdivisions);
                    float w = 1.0f - u - v;
                    maths::vec2 uv = uv0 * w + uv1 * u + uv2 * v;
                    if (SampleMaskAlphaCPU(maskTextureIndex, uv) < kMaskAlphaCutoff) {
                        return true;
                    }
                }
            }
            return false;
        }

        // Recompacts the subset of `src`'s triangles selected by `keepTriangle` (parallel to its
        // triangle list, src.localTriangleIndices.size()/3 entries) into a new, independently
        // locally-indexed MeshCluster -- the same "select triangles, then recompact vertices"
        // pattern ClusterDAG.cpp's ClampMeshToCapacity/SplitSimplifiableMesh already use for the
        // same kind of triangle-subset extraction, specialized here to MeshCluster's own layout
        // (uint8_t local indices, parallel originalTriangleIndices for coverage traceability).
        MeshCluster ExtractTriangleSubset(const MeshCluster& src, const std::vector<bool>& keepTriangle,
            bool isMasked, const std::vector<renderer::Vertex>& allVertices) {

            MeshCluster out;
            out.isMasked = isMasked;

            uint32_t triCount = static_cast<uint32_t>(src.localTriangleIndices.size() / 3u);
            std::unordered_map<uint8_t, uint8_t> localRemap;
            out.globalVertexIndices.reserve(src.globalVertexIndices.size());
            out.localTriangleIndices.reserve(src.localTriangleIndices.size());
            out.originalTriangleIndices.reserve(src.originalTriangleIndices.size());

            for (uint32_t t = 0; t < triCount; ++t) {
                if (!keepTriangle[t]) {
                    continue;
                }
                for (uint32_t k = 0; k < 3u; ++k) {
                    uint8_t srcLocal = src.localTriangleIndices[t * 3u + k];
                    auto it = localRemap.find(srcLocal);
                    uint8_t newLocal;
                    if (it == localRemap.end()) {
                        newLocal = static_cast<uint8_t>(out.globalVertexIndices.size());
                        localRemap.emplace(srcLocal, newLocal);
                        out.globalVertexIndices.push_back(src.globalVertexIndices[srcLocal]);
                    }
                    else {
                        newLocal = it->second;
                    }
                    out.localTriangleIndices.push_back(newLocal);
                }
                out.originalTriangleIndices.push_back(src.originalTriangleIndices[t]);
            }

            maths::vec3 boundsMin, boundsMax;
            maths::ResetAABB(boundsMin, boundsMax);
            for (uint32_t g : out.globalVertexIndices) {
                maths::ExpandAABB(boundsMin, boundsMax, allVertices[g].position);
            }
            out.boundsMin = boundsMin;
            out.boundsMax = boundsMax;
            out.sphereCenter = maths::AABBCenter(boundsMin, boundsMax);
            out.sphereRadius = maths::AABBRadius(boundsMin, boundsMax);

            return out;
        }

        // Classifies every triangle of every cluster in `clusters` against the entity's opacity
        // mask, splitting any cluster whose masked-triangle fraction exceeds
        // kMaskedClusterSplitThreshold into a pure-opaque + pure-masked replacement pair (see
        // ClusterPartitioner.h's PartitionMeshIntoClusters doc comment for the full contract).
        // No-op (every cluster keeps isMasked == false) when maskTextureIndex is the sentinel.
        void ClassifyAndSplitForOpacity(std::vector<MeshCluster>& clusters, uint32_t maskTextureIndex,
            const std::vector<renderer::Vertex>& allVertices) {

            if (maskTextureIndex == kInvalidMaskTextureIndex) {
                return;
            }

            std::vector<MeshCluster> result;
            result.reserve(clusters.size());

            for (MeshCluster& cluster : clusters) {
                uint32_t triCount = static_cast<uint32_t>(cluster.localTriangleIndices.size() / 3u);
                std::vector<bool> triangleMasked(triCount, false);
                uint32_t maskedCount = 0u;
                for (uint32_t t = 0; t < triCount; ++t) {
                    uint32_t g0 = cluster.globalVertexIndices[cluster.localTriangleIndices[t * 3u + 0u]];
                    uint32_t g1 = cluster.globalVertexIndices[cluster.localTriangleIndices[t * 3u + 1u]];
                    uint32_t g2 = cluster.globalVertexIndices[cluster.localTriangleIndices[t * 3u + 2u]];
                    bool masked = IsTriangleMasked(allVertices[g0].uv, allVertices[g1].uv, allVertices[g2].uv, maskTextureIndex);
                    triangleMasked[t] = masked;
                    maskedCount += masked ? 1u : 0u;
                }

                if (maskedCount == 0u) {
                    cluster.isMasked = false;
                    result.push_back(std::move(cluster));
                    continue;
                }

                float maskedFraction = static_cast<float>(maskedCount) / static_cast<float>(triCount);
                if (maskedFraction <= kMaskedClusterSplitThreshold) {
                    // Below the split threshold: fold the whole cluster into the masked path rather
                    // than fragment a near-pure cluster over a small minority of triangles.
                    cluster.isMasked = true;
                    result.push_back(std::move(cluster));
                    continue;
                }

                std::vector<bool> keepOpaque(triCount);
                std::vector<bool> keepMasked(triCount);
                for (uint32_t t = 0; t < triCount; ++t) {
                    keepOpaque[t] = !triangleMasked[t];
                    keepMasked[t] = triangleMasked[t];
                }

                MeshCluster opaquePart = ExtractTriangleSubset(cluster, keepOpaque, false, allVertices);
                MeshCluster maskedPart = ExtractTriangleSubset(cluster, keepMasked, true, allVertices);
                if (!opaquePart.localTriangleIndices.empty()) {
                    result.push_back(std::move(opaquePart));
                }
                if (!maskedPart.localTriangleIndices.empty()) {
                    result.push_back(std::move(maskedPart));
                }
            }

            clusters = std::move(result);
        }

    } // namespace

    std::vector<MeshCluster> PartitionMeshIntoClusters(
        uint32_t targetMeshID,
        const std::vector<renderer::Vertex>& allVertices,
        const std::vector<uint32_t>& allIndices,
        uint32_t maskTextureIndex) {

        std::vector<TriangleRef> tris;
        for (size_t t = 0; t + 2 < allIndices.size(); t += 3) {
            uint32_t i0 = allIndices[t + 0];
            uint32_t i1 = allIndices[t + 1];
            uint32_t i2 = allIndices[t + 2];

            if (allVertices[i0].meshID != targetMeshID) {
                continue; // This triangle belongs to a different entity.
            }

#ifndef NDEBUG
            // Sanity check for the 2026-07-16 "clusters mis-generated during triangulation ->
            // cluster conversion" investigation: this filter only inspects i0. If a procedural
            // Generate* function ever emits a triangle whose 3 vertices don't share one meshID
            // (e.g. an off-by-one in a per-entity vertex/index base offset), it would silently be
            // attributed whole to i0's entity here, with i1/i2 actually belonging to a neighboring
            // entity -- exactly the kind of "cluster looks wrong" symptom the hypothesis describes.
            if (allVertices[i1].meshID != targetMeshID || allVertices[i2].meshID != targetMeshID) {
                LOG_WARNING(std::format(
                    "[ClusterPartitioner] Triangle at index buffer offset {} has mixed meshIDs "
                    "(i0={}, i1={}, i2={}, targetMeshID={}) -- this triangle straddles two entities "
                    "and was attributed whole to targetMeshID via i0 only. Likely a vertex/index "
                    "base-offset bug in whichever Generate* function produced entity {}.",
                    t, allVertices[i0].meshID, allVertices[i1].meshID, allVertices[i2].meshID,
                    targetMeshID, targetMeshID));
            }
#endif

            maths::vec3 centroid = (allVertices[i0].position + allVertices[i1].position + allVertices[i2].position) * (1.0f / 3.0f);
            tris.push_back(TriangleRef{ static_cast<uint32_t>(tris.size()), i0, i1, i2, centroid });
        }

        if (tris.empty()) {
            return {};
        }

        std::vector<uint32_t> allGroup(tris.size());
        std::iota(allGroup.begin(), allGroup.end(), 0u);

        std::vector<std::vector<uint32_t>> groups;
        PartitionGroupRecursive(tris, std::move(allGroup), groups);

        std::vector<MeshCluster> result;
        result.reserve(groups.size());
        for (const auto& group : groups) {
            result.push_back(BuildClusterFromGroup(tris, allVertices, group));
        }

        ClassifyAndSplitForOpacity(result, maskTextureIndex, allVertices);

#ifndef NDEBUG
        // Runtime enforcement of this function's own documented guarantee (see the header comment
        // on PartitionMeshIntoClusters): the union of every cluster's originalTriangleIndices must
        // be exactly [0, tris.size()) with no gaps and no duplicates. Previously only checked by
        // ClusterPartitionerTests.cpp against synthetic meshes -- this re-runs the same check
        // against the REAL procedurally-generated geometry every time the cache is built, which is
        // the only way to catch a partitioning bug that only manifests on real mesh topology.
        {
            std::vector<uint32_t> coverageCount(tris.size(), 0u);
            for (const MeshCluster& cluster : result) {
                for (uint32_t originalTriIndex : cluster.originalTriangleIndices) {
                    if (originalTriIndex >= coverageCount.size()) {
                        LOG_ERROR(std::format(
                            "[ClusterPartitioner] mesh {}: originalTriangleIndices contains {}, out of range for {} matched triangles.",
                            targetMeshID, originalTriIndex, tris.size()));
                        continue;
                    }
                    ++coverageCount[originalTriIndex];
                }
            }
            uint32_t missingCount = 0;
            uint32_t duplicateCount = 0;
            uint32_t firstBadIndex = 0xFFFFFFFFu;
            for (uint32_t i = 0; i < coverageCount.size(); ++i) {
                if (coverageCount[i] == 0u) {
                    ++missingCount;
                    if (firstBadIndex == 0xFFFFFFFFu) firstBadIndex = i;
                } else if (coverageCount[i] > 1u) {
                    ++duplicateCount;
                    if (firstBadIndex == 0xFFFFFFFFu) firstBadIndex = i;
                }
            }
            if (missingCount > 0 || duplicateCount > 0) {
                LOG_ERROR(std::format(
                    "[ClusterPartitioner] mesh {}: triangle-coverage bijection VIOLATED -- {} triangle(s) missing "
                    "from every cluster, {} triangle(s) duplicated across clusters (out of {} total), first bad "
                    "original triangle index={}. Clusters for this entity are provably mis-generated.",
                    targetMeshID, missingCount, duplicateCount, tris.size(), firstBadIndex));
            }
        }
#endif

        LOG_INFO(std::format("[ClusterPartitioner] Partitioned mesh {} into {} clusters (mask texture index: {}).", targetMeshID, result.size(), maskTextureIndex));
        return result;
    }

}
