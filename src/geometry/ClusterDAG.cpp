#include "geometry/ClusterDAG.h"
#include "core/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <numeric>
#include <unordered_map>

#include "geometry/ClusterFormat.h"
#include "geometry/ClusterGrouping.h"
#include "geometry/ClusterPartitioner.h"
#include "geometry/GeometryHashUtil.h"

namespace geometry {

    namespace {

        // Shared by UpdateNodeBounds (a node's own bounds) and EmitSimplifiedGroup (a group's
        // shared bounding sphere, computed from its merged+simplified mesh before any re-split --
        // see ClusterDAGGroup's own comment for why that value must be computed once, here, rather
        // than re-derived per output fragment).
        void ComputeMeshBounds(const SimplifiableMesh& mesh,
            maths::vec3& outBoundsMin, maths::vec3& outBoundsMax,
            maths::vec3& outSphereCenter, float& outSphereRadius) {
            maths::ResetAABB(outBoundsMin, outBoundsMax);
            for (const maths::vec3& p : mesh.positions) {
                maths::ExpandAABB(outBoundsMin, outBoundsMax, p);
            }
            outSphereCenter = maths::AABBCenter(outBoundsMin, outBoundsMax);
            outSphereRadius = maths::AABBRadius(outBoundsMin, outBoundsMax);
        }

        void UpdateNodeBounds(ClusterDAGNode& node) {
            ComputeMeshBounds(node.mesh, node.boundsMin, node.boundsMax, node.sphereCenter, node.sphereRadius);
        }

        // Adjacency between same-level (level >= 1) DAG nodes: two nodes are adjacent if they
        // share at least one *locked* vertex position -- the only positions guaranteed to still
        // reflect real, shared original-mesh topology after independent simplification (see the
        // file header). The weight is the number of shared locked positions, mirroring
        // ClusterGrouping's shared-global-vertex weight for level 0->1 pairing.
        //
        // `nodeIsMasked` is parallel to `levelMeshes` (the owning ClusterDAGNode::isMasked for each
        // entry); a cross-classification pair (one opaque, one masked) is never recorded, exactly
        // like ClusterGrouping::BuildClusterAdjacencyWeights at level 0->1 -- see that function's
        // doc comment for why.
        //
        // `nodeSourceGroupIndex` is likewise parallel to `levelMeshes`: two nodes produced by the
        // SAME immediately-preceding re-split (EmitSimplifiedGroup, same non-invalid
        // sourceGroupIndex) never get a weight entry either, even though they typically share by
        // far the largest number of locked positions of any pair at this level -- the entire split
        // plane SplitSimplifiableMesh cut them along. Without this exclusion, GreedyPairByWeight's
        // "always prefer the highest-weight neighbor" rule would pick these two siblings right back
        // over any other candidate, re-merging them into essentially the same oversized mesh their
        // shared group just failed to fit in one cluster -- forcing the exact same re-split again,
        // producing near-identical siblings that are once again each other's top candidate, forever
        // (observed running past level 11000 with no convergence in testing). This is the same
        // class of non-termination as the reverted 2026-07-16 pass-through-node fix attempt
        // (ClusterDAG.h's own comment on ClusterDAGGroup), just reached through re-pairing instead
        // of a trivial wrapper -- excluding the edge here forces each sibling to look for a
        // genuinely different neighbor (their own external boundary, distinct from the shared cut),
        // making real progress instead of oscillating between merge and re-split.
        std::unordered_map<uint64_t, uint32_t> BuildLevelAdjacencyWeights(
            const std::vector<const SimplifiableMesh*>& levelMeshes, const std::vector<bool>& nodeIsMasked,
            const std::vector<uint32_t>& nodeSourceGroupIndex) {

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
                        if (nodeIsMasked[owners[a]] != nodeIsMasked[owners[b]]) {
                            continue;
                        }
                        if (nodeSourceGroupIndex[owners[a]] != kInvalidDAGGroupIndex
                            && nodeSourceGroupIndex[owners[a]] == nodeSourceGroupIndex[owners[b]]) {
                            continue; // Re-split siblings -- see this function's own comment.
                        }
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
            std::vector<maths::vec2> uvs;
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
                            uvs.push_back(src.uvs[srcLocalIdx]);
                            lockedPositionToMergedIndex.emplace(key, mergedIdx);
                        }
                    }
                    else {
                        mergedIdx = static_cast<uint32_t>(positions.size());
                        positions.push_back(p);
                        uvs.push_back(src.uvs[srcLocalIdx]);
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
                edgeUsageCount[PackOrderedPair(triangles[t + 0], triangles[t + 1])] += 1u;
                edgeUsageCount[PackOrderedPair(triangles[t + 1], triangles[t + 2])] += 1u;
                edgeUsageCount[PackOrderedPair(triangles[t + 2], triangles[t + 0])] += 1u;
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
            result.mesh.uvs = std::move(uvs);
            result.mesh.triangles = std::move(triangles);
            result.originalTriangleCount = originalTriangleCount;
            return result;
        }

        // One group awaiting simplification + promotion into one or more new, coarser DAG nodes.
        struct PendingGroup {
            std::vector<uint32_t> memberDagIndices; // Indices into dag.nodes (the group's members).
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
                            out.uvs.push_back(src.uvs[v[s]]);
                        }
                    }
                    uint32_t a = srcToOut[v[0]], b = srcToOut[v[1]], c = srcToOut[v[2]];
                    out.triangles.push_back(a);
                    out.triangles.push_back(b);
                    out.triangles.push_back(c);
                    edgeUsage[PackOrderedPair(a, b)] += 1u;
                    edgeUsage[PackOrderedPair(b, c)] += 1u;
                    edgeUsage[PackOrderedPair(c, a)] += 1u;
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
        // Simplifies `mesh` (merged from up to 4 same-level members, see ClusterGrouping::
        // GroupItemsIntoQuads) and emits one ClusterDAGGroup into `dag`, producing either a single
        // output node (the common case: the simplified mesh already fits one cluster's vertex/
        // triangle cap) or, when it doesn't, exactly kMaxGroupOutputClusters output nodes via one
        // spatial re-split (SplitSimplifiableMesh) -- both siblings sharing this group's error and
        // bounding sphere (see ClusterDAGGroup's own comment). This indirection through a group,
        // rather than simplifying straight into one parent node the way a 2-member merge always
        // could, is what lets a group of up to 4 members legitimately need more than one coarser
        // representative without any single member ever needing more than one parent reference.
        //
        // If QEM alone cannot reduce the vertex count to kMaxClusterVertices even after the
        // re-split (locked-vertex saturation -- common on hard-edged, multi-face-seam primitives
        // like a box/cylinder/tube/pyramid/plane, whose many permanently-locked boundary vertices
        // QEM can never collapse, vs. a smoothly-curved shape's much sparser locked set), this
        // used to (2026-07-16, before groups existed at all) silently CLAMP the mesh: drop every
        // triangle referencing a vertex beyond the cap and re-compact. That is a real, confirmed
        // bug ("some clusters just don't render" investigation) -- it punches an actual hole in
        // the merged surface at that DAG level, and every level built on top of it inherits the
        // hole.
        //
        // Fixed by making every one of this group's MEMBERS a ROOT right here instead of forcing
        // them into a lossy merged/re-split output -- NOT by re-wrapping them in a trivial pass-
        // through parent and feeding them back into `nextLevel` (a first attempt at this fix that
        // shipped briefly and was caught before merging: since a pass-through node is byte-for-byte
        // the same mesh with the same locked vertices, the exact same merge fails again next level,
        // and the level after that, forever -- observed running past level 300 with no convergence
        // in testing). Diverting straight to `outEarlyRoots` instead guarantees
        // `currentLevel.size()` strictly shrinks every time this fires, so BuildClusterDAG's
        // `while (currentLevel.size() > 1)` loop is still guaranteed to terminate. The re-split
        // path this function adds does not reopen that risk: SplitSimplifiableMesh strictly
        // shrinks the triangle count of whichever half still doesn't fit relative to the
        // already-simplified group mesh it split from, so this function still only ever calls it
        // once per group (never recursively) -- an oversized half after that one split falls
        // straight through to the exact same, already-proven-terminating earlyRoots path, never a
        // second split attempt. Costs strictly less LOD compression for just this one problematic
        // group (each member becomes its own root one or more levels earlier than it would if it
        // had been mergeable), never any lost geometry.
        // ----------------------------------------------------------------------------------
        void EmitSimplifiedGroup(
            SimplifiableMesh mesh,
            const std::vector<uint32_t>& memberDagIndices,
            uint32_t originalTriangleCount,
            uint32_t level,
            ClusterDAG& dag,
            std::vector<uint32_t>& nextLevel,
            std::vector<uint32_t>& outEarlyRoots) {

            uint32_t targetTriangleCount = (originalTriangleCount + 1u) / 2u;
            float simplificationError = 0.0f;
            SimplifyMeshQEM(mesh, targetTriangleCount, kMaxClusterVertices, &simplificationError);

            float membersMaxError = 0.0f;
            for (uint32_t memberIdx : memberDagIndices) {
                membersMaxError = std::max(membersMaxError, dag.nodes[memberIdx].clusterError);
            }
            float errorFloor = std::max(kMinimumErrorStepAbsolute, membersMaxError * kMinimumErrorStepRelative);
            float groupError = std::max(simplificationError, membersMaxError + errorFloor);

            // The group's shared bounding sphere: computed from the merged+simplified mesh BEFORE
            // any re-split below, so both potential output siblings project the identical "parent"
            // error/position -- see ClusterDAGGroup's own comment for why that must not depend on
            // which specific sibling a runtime LOD cut ends up actually drawing.
            maths::vec3 groupBoundsMin, groupBoundsMax, groupSphereCenter;
            float groupSphereRadius = 0.0f;
            ComputeMeshBounds(mesh, groupBoundsMin, groupBoundsMax, groupSphereCenter, groupSphereRadius);

            std::vector<SimplifiableMesh> outputMeshes;
            bool fits = mesh.positions.size() <= kMaxClusterVertices && mesh.triangles.size() <= kMaxClusterIndices;
            if (fits) {
                outputMeshes.push_back(std::move(mesh));
            }
            else if (mesh.triangles.size() >= 6) { // SplitSimplifiableMesh's own precondition (>= 2 triangles).
                static_assert(kMaxGroupOutputClusters == 2u,
                    "EmitSimplifiedGroup performs exactly one SplitSimplifiableMesh call, producing exactly "
                    "2 halves -- update this split logic if kMaxGroupOutputClusters ever changes.");
                SimplifiableMesh half0, half1;
                SplitSimplifiableMesh(mesh, half0, half1);
                outputMeshes.push_back(std::move(half0));
                outputMeshes.push_back(std::move(half1));
            }
            else {
                // Fewer than 2 triangles yet still over the vertex cap: pathological (would need
                // hundreds of locked vertices packed onto a single triangle) but structurally
                // possible, and SplitSimplifiableMesh cannot be called on it -- treat exactly like
                // an unsplit oversized mesh below (this group produces zero outputs).
                outputMeshes.push_back(std::move(mesh));
            }

            for (const SimplifiableMesh& piece : outputMeshes) {
                if (piece.positions.size() > kMaxClusterVertices || piece.triangles.size() > kMaxClusterIndices) {
                    LOG_WARNING(std::format(
                        "[ClusterDAG] Level {}: a {}-member group still has an oversized piece ({} vertices/{} "
                        "triangles; cap: {} vertices/{} triangles) after simplification{} -- locked-vertex "
                        "saturation. Every member becomes its own root here instead of merging, to avoid "
                        "dropping geometry.",
                        level, memberDagIndices.size(), piece.positions.size(), piece.triangles.size() / 3,
                        kMaxClusterVertices, kMaxClusterIndices / 3,
                        outputMeshes.size() > 1 ? " and re-splitting" : ""));
                    for (uint32_t memberIdx : memberDagIndices) {
                        outEarlyRoots.push_back(memberIdx); // dag.nodes[memberIdx].parentGroupIndex is
                                                             // already kInvalidDAGGroupIndex (never
                                                             // touched) -- a genuine root, not a
                                                             // dangling reference.
                    }
                    return;
                }
            }

            uint32_t groupIndex = static_cast<uint32_t>(dag.groups.size()); // Slot this group will occupy.

            std::vector<uint32_t> outputIndices;
            outputIndices.reserve(outputMeshes.size());
            for (SimplifiableMesh& outputMesh : outputMeshes) {
                ClusterDAGNode outputNode;
                outputNode.mesh            = std::move(outputMesh);
                outputNode.sourceGroupIndex = groupIndex;
                outputNode.level            = level;
                outputNode.clusterError     = groupError;
                // Every member is guaranteed the same isMasked classification (the adjacency-
                // weight filters above never let an opaque and a masked node merge into one
                // group), so any one member's value is authoritative for every output node.
                outputNode.isMasked = dag.nodes[memberDagIndices[0]].isMasked;
                UpdateNodeBounds(outputNode);

                uint32_t outputDagIndex = static_cast<uint32_t>(dag.nodes.size());
                dag.nodes.push_back(std::move(outputNode));
                outputIndices.push_back(outputDagIndex);
                nextLevel.push_back(outputDagIndex);
            }

            ClusterDAGGroup group;
            group.memberClusterIndices = memberDagIndices;
            group.outputClusterIndices = std::move(outputIndices);
            group.groupError           = groupError;
            group.groupSphereCenter    = groupSphereCenter;
            group.groupSphereRadius    = groupSphereRadius;
            dag.groups.push_back(std::move(group));

            for (uint32_t memberIdx : memberDagIndices) {
                dag.nodes[memberIdx].parentGroupIndex = groupIndex;
                dag.nodes[memberIdx].parentError = groupError;
            }
        }

    } // namespace

    ClusterDAG BuildClusterDAG(
        uint32_t targetMeshID,
        const std::vector<renderer::Vertex>& allVertices,
        const std::vector<uint32_t>& allIndices,
        uint32_t maskTextureIndex) {

        LOG_INFO(std::format("[ClusterDAG] Building DAG for mesh ID {} (mask texture index: {})...", targetMeshID, maskTextureIndex));
        ClusterDAG dag;

        std::vector<MeshCluster> leafClusters = PartitionMeshIntoClusters(targetMeshID, allVertices, allIndices, maskTextureIndex);
        if (leafClusters.empty()) {
            LOG_WARNING(std::format("[ClusterDAG] No triangles matched targetMeshID {} during DAG creation.", targetMeshID));
            return dag;
        }

        // --- Level 0: one DAG node per leaf cluster, exact geometry, zero error ---------------
        std::vector<uint32_t> currentLevel;
        currentLevel.reserve(leafClusters.size());
        for (const MeshCluster& cluster : leafClusters) {
            ClusterDAGNode node;
            node.level = 0;
            node.clusterError = 0.0f;
            node.isMasked = cluster.isMasked;
            node.mesh.positions.reserve(cluster.globalVertexIndices.size());
            node.mesh.uvs.reserve(cluster.globalVertexIndices.size());
            for (uint32_t g : cluster.globalVertexIndices) {
                node.mesh.positions.push_back(allVertices[g].position);
                node.mesh.uvs.push_back(allVertices[g].uv);
            }
            node.mesh.locked.assign(cluster.globalVertexIndices.size(), false);
            node.mesh.triangles.assign(cluster.localTriangleIndices.begin(), cluster.localTriangleIndices.end());
            UpdateNodeBounds(node);

            dag.nodes.push_back(std::move(node));
            currentLevel.push_back(static_cast<uint32_t>(dag.nodes.size() - 1));
        }

        uint32_t level = 1;
        bool isFirstPass = true;

        // Nodes EmitSimplifiedGroup diverts straight to root status because their group's merge
        // never fit within capacity even after simplification and one re-split attempt (see that
        // function's own comment). Accumulated across every level and appended to dag.rootIndices
        // at the end, alongside whatever's left in currentLevel once the main loop below
        // terminates normally.
        std::vector<uint32_t> earlyRoots;

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
                    // A singleton "group" (no adjacent partner -- e.g. a disconnected mesh
                    // island) has nothing to merge with; wrapping it in a new node anyway would
                    // simplify it against ITSELF (no other member contributes any new locked-
                    // boundary constraint), producing an identical or near-identical mesh every
                    // time this same singleton keeps failing to find a partner at the next level
                    // too -- an unbounded chain of pass-through nodes, one per level, the same
                    // non-termination class EmitSimplifiedGroup's own comment describes for its
                    // reverted 2026-07-16 fix attempt. Promoting it straight to a root instead
                    // costs nothing (it was never going to compress further) and guarantees this
                    // level's node count can only shrink, never idle in place.
                    if (g.memberClusterIndices.size() < 2) {
                        earlyRoots.push_back(currentLevel[g.memberClusterIndices[0]]);
                        continue;
                    }
                    PendingGroup pg;
                    pg.memberDagIndices.reserve(g.memberClusterIndices.size());
                    for (uint32_t localIdx : g.memberClusterIndices) {
                        pg.memberDagIndices.push_back(currentLevel[localIdx]);
                    }
                    pg.mesh = std::move(g.mesh);
                    pg.originalTriangleCount = g.originalTriangleCount;
                    pendingGroups.push_back(std::move(pg));
                }
            }
            else {
                std::vector<const SimplifiableMesh*> levelMeshes;
                std::vector<bool> levelIsMasked;
                std::vector<uint32_t> levelSourceGroupIndex;
                levelMeshes.reserve(currentLevel.size());
                levelIsMasked.reserve(currentLevel.size());
                levelSourceGroupIndex.reserve(currentLevel.size());
                for (uint32_t nodeIdx : currentLevel) {
                    levelMeshes.push_back(&dag.nodes[nodeIdx].mesh);
                    levelIsMasked.push_back(dag.nodes[nodeIdx].isMasked);
                    levelSourceGroupIndex.push_back(dag.nodes[nodeIdx].sourceGroupIndex);
                }

                std::unordered_map<uint64_t, uint32_t> weights =
                    BuildLevelAdjacencyWeights(levelMeshes, levelIsMasked, levelSourceGroupIndex);
                std::vector<std::vector<uint32_t>> memberLists =
                    GroupItemsIntoQuads(static_cast<uint32_t>(currentLevel.size()), weights);

                bool anyPairFormed = false;
                for (const auto& members : memberLists) {
                    if (members.size() > 1) { anyPairFormed = true; break; }
                }
                if (!anyPairFormed) {
                    break; // Fully disconnected (or already-singleton) level: stop, keep current roots.
                }

                pendingGroups.reserve(memberLists.size());
                for (const auto& members : memberLists) {
                    // A singleton (no adjacent partner left this level -- see the isFirstPass
                    // branch's own comment for why this must promote straight to a root instead
                    // of wrapping a new pass-through node around it).
                    if (members.size() < 2) {
                        earlyRoots.push_back(currentLevel[members[0]]);
                        continue;
                    }
                    MergedLevelResult merged = MergeLevelMeshes(members, levelMeshes);
                    PendingGroup pg;
                    pg.memberDagIndices.reserve(members.size());
                    for (uint32_t localIdx : members) {
                        pg.memberDagIndices.push_back(currentLevel[localIdx]);
                    }
                    pg.mesh = std::move(merged.mesh);
                    pg.originalTriangleCount = merged.originalTriangleCount;
                    pendingGroups.push_back(std::move(pg));
                }
            }

            // --- Simplify each group and promote it into one (or, when QEM alone can't reduce
            // vertex count below kMaxClusterVertices even after a re-split -- locked-vertex
            // saturation -- zero) coarser DAG node(s), diverting the group's members straight into
            // `earlyRoots` instead of ever dropping geometry to force a fit -- see
            // EmitSimplifiedGroup's own comment.
            std::vector<uint32_t> nextLevel;
            nextLevel.reserve(pendingGroups.size());

            for (PendingGroup& pg : pendingGroups) {
                EmitSimplifiedGroup(
                    std::move(pg.mesh),
                    pg.memberDagIndices,
                    pg.originalTriangleCount,
                    level,
                    dag,
                    nextLevel,
                    earlyRoots);
            }

            LOG_INFO(std::format("[ClusterDAG] Level {} completed: {} nodes (total nodes: {}).", level, nextLevel.size(), dag.nodes.size()));
            currentLevel = std::move(nextLevel);
            isFirstPass = false;
            ++level;
        }

        dag.rootIndices = std::move(currentLevel);
        dag.rootIndices.insert(dag.rootIndices.end(), earlyRoots.begin(), earlyRoots.end());

        std::vector<std::string> validationErrors;
        bool isValid = ValidateClusterDAG(dag, validationErrors);
        if (isValid) {
            LOG_INFO(std::format("[ClusterDAG] DAG generation complete. Total levels: {}, Total nodes: {}, Root count: {}. Validation: PASSED.", level, dag.nodes.size(), dag.rootIndices.size()));
        } else {
            LOG_ERROR(std::format("[ClusterDAG] DAG validation FAILED with {} errors!", validationErrors.size()));
            for (const auto& err : validationErrors) {
                LOG_ERROR(std::format("[ClusterDAG]   Validation Error: {}", err));
            }
        }

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
                return; // Already fully validated via another path -- the EXPECTED case for a node
                        // reachable through more than one of its group's output siblings (this is a
                        // true DAG, not a strict tree; see ClusterDAGGroup's own comment), not just
                        // a defensive "should not happen" fallback.
            }

            state[nodeIndex] = VisitState::InProgress;
            const ClusterDAGNode& node = dag.nodes[nodeIndex];
            const std::string label = "node " + std::to_string(nodeIndex);

#ifndef NDEBUG
            // --- Geometric sanity (2026-07-16 "clusters mis-generated during triangulation ->
            // cluster conversion" investigation) --------------------------------------------------
            // Everything above/below this block checks the DAG's GRAPH structure (indices, error
            // monotonicity); nothing in ValidateClusterDAG previously looked at a node's actual
            // mesh data at all. Debug-only (unlike the structural checks above, which are load-
            // bearing correctness gates main.cpp treats as fatal in every config): these are new,
            // purely diagnostic checks added to test a specific hypothesis, not an existing invariant
            // this engine has ever relied on, so they must not change Release behavior.
            {
                const SimplifiableMesh& mesh = node.mesh;
                const uint32_t triangleCount = static_cast<uint32_t>(mesh.triangles.size() / 3);

                // 0. Every position must be finite. A NaN/Inf vertex (e.g. from a pathological QEM
                // edge collapse -- a singular/near-singular quadric solve, or a UV/position blend
                // that divides by a degenerate weight) would silently PASS every other check below:
                // any comparison against NaN is false in IEEE-754, so the degenerate-area check
                // (#2) and the bounds-enclosure check (#3) both simply skip a NaN vertex instead of
                // flagging it. At render time, quantizing a NaN t via std::lround(t * 65535.0f) is
                // undefined behavior that typically yields a garbage-but-representable uint16_t,
                // decoding to a position essentially arbitrary within (or unrelated to) the
                // cluster's own AABB -- a small number of triangles shooting off into a
                // disconnected "shard", exactly the still-open "persistent cluster holes /
                // shattered geometry" symptom (see project memory) this check exists to confirm or
                // rule out.
                uint32_t nonFiniteCount = 0;
                for (const maths::vec3& p : mesh.positions) {
                    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                        ++nonFiniteCount;
                    }
                }
                if (nonFiniteCount > 0) {
                    MarkFailure(outErrors, label + " (level " + std::to_string(node.level) + ") has " +
                        std::to_string(nonFiniteCount) + " non-finite (NaN/Inf) vertex position(s) out of " +
                        std::to_string(mesh.positions.size()));
                }

                // 1. Every triangle vertex index must stay within this node's own positions array.
                uint32_t outOfRangeIndexCount = 0;
                for (uint32_t idx : mesh.triangles) {
                    if (idx >= mesh.positions.size()) {
                        ++outOfRangeIndexCount;
                    }
                }
                if (outOfRangeIndexCount > 0) {
                    MarkFailure(outErrors, label + " has " + std::to_string(outOfRangeIndexCount) +
                        " triangle vertex index/indices out of range for its own " +
                        std::to_string(mesh.positions.size()) + "-vertex positions array");
                }

                // 2. Degenerate (near-zero-area) triangles -- only meaningful if every index above
                // was actually in range (an out-of-range index would make this check itself unsafe).
                if (outOfRangeIndexCount == 0) {
                    constexpr float kMinTriangleAreaSq = 1e-14f; // Matches MeshSimplifier.cpp's own threshold.
                    uint32_t degenerateCount = 0;
                    for (uint32_t t = 0; t < triangleCount; ++t) {
                        const maths::vec3& p0 = mesh.positions[mesh.triangles[t * 3 + 0]];
                        const maths::vec3& p1 = mesh.positions[mesh.triangles[t * 3 + 1]];
                        const maths::vec3& p2 = mesh.positions[mesh.triangles[t * 3 + 2]];
                        maths::vec3 cross = (p1 - p0).Cross(p2 - p0);
                        if (cross.Dot(cross) * 0.25f < kMinTriangleAreaSq) {
                            ++degenerateCount;
                        }
                    }
                    if (degenerateCount > 0) {
                        MarkFailure(outErrors, label + " (level " + std::to_string(node.level) + ") contains " +
                            std::to_string(degenerateCount) + " degenerate (near-zero-area) triangle(s) out of " +
                            std::to_string(triangleCount));
                    }

                    // 3. boundsMin/boundsMax must actually enclose every vertex -- this is the exact
                    // invariant ClusterCullMetadata::boundsMin/boundsMax's dual purpose (culling AABB
                    // AND vertex dequantization range, see project memory on that bug class) depends
                    // on; a violation here would silently corrupt vertex decode at render time.
                    constexpr float kBoundsEpsilon = 1e-4f;
                    uint32_t outOfBoundsCount = 0;
                    for (const maths::vec3& p : mesh.positions) {
                        if (p.x < node.boundsMin.x - kBoundsEpsilon || p.y < node.boundsMin.y - kBoundsEpsilon || p.z < node.boundsMin.z - kBoundsEpsilon ||
                            p.x > node.boundsMax.x + kBoundsEpsilon || p.y > node.boundsMax.y + kBoundsEpsilon || p.z > node.boundsMax.z + kBoundsEpsilon) {
                            ++outOfBoundsCount;
                        }
                    }
                    if (outOfBoundsCount > 0) {
                        MarkFailure(outErrors, label + " has " + std::to_string(outOfBoundsCount) +
                            " vertex/vertices outside its own cached boundsMin/boundsMax");
                    }
                }

                // 4. On-disk format contract: childClusterID is a fixed 4-element array (ClusterFormat.h's
                // DAGNodeEntry) -- a node with more than 4 children could never be represented on disk.
                size_t childCount = (node.sourceGroupIndex != kInvalidDAGGroupIndex && node.sourceGroupIndex < dag.groups.size())
                    ? dag.groups[node.sourceGroupIndex].memberClusterIndices.size()
                    : 0;
                if (childCount > 4) {
                    MarkFailure(outErrors, label + " has " + std::to_string(childCount) +
                        " children, more than the on-disk format's 4-child limit");
                }

                // 5. A leaf (level 0, no children) must carry exactly zero clusterError, per this
                // struct's own documented invariant -- it IS the exact original geometry.
                if (childCount == 0 && node.clusterError != 0.0f) {
                    MarkFailure(outErrors, label + " is a leaf but clusterError is " +
                        std::to_string(node.clusterError) + " instead of the documented 0.0f");
                }
            }
#endif

            // --- Error monotonicity (parent GROUP side) ----------------------------------------
            if (node.parentGroupIndex == kInvalidDAGGroupIndex) {
                if (!std::isinf(node.parentError) || node.parentError < 0.0f) {
                    MarkFailure(outErrors, label + " is a root but parentError is not +infinity");
                }
            }
            else if (node.parentGroupIndex >= dag.groups.size()) {
                MarkFailure(outErrors, label + " has an out-of-range parentGroupIndex " + std::to_string(node.parentGroupIndex));
            }
            else {
                const ClusterDAGGroup& parentGroup = dag.groups[node.parentGroupIndex];
                if (node.parentError != parentGroup.groupError) {
                    MarkFailure(outErrors, label + " cached parentError (" + std::to_string(node.parentError) +
                        ") does not match its parent group's actual groupError (" + std::to_string(parentGroup.groupError) + ")");
                }
                if (!(node.parentError > node.clusterError)) {
                    MarkFailure(outErrors, label + " violates strict LOD error monotonicity: parentError (" +
                        std::to_string(node.parentError) + ") is not strictly greater than clusterError (" +
                        std::to_string(node.clusterError) + ")");
                }

                // Mutual consistency: `node` must actually appear in its declared parent group's
                // memberClusterIndices.
                const auto& members = parentGroup.memberClusterIndices;
                if (std::find(members.begin(), members.end(), nodeIndex) == members.end()) {
                    MarkFailure(outErrors, label + "'s parent group (" + std::to_string(node.parentGroupIndex) +
                        ") does not list it back as a member");
                }

                // isMasked purity: a node's classification must match every one of its parent
                // group's other members -- proves the opaque/masked split's purity guarantee
                // (never merging differently-classified nodes, see BuildClusterAdjacencyWeights/
                // BuildLevelAdjacencyWeights) actually held for the DAG as constructed, not just
                // for one merge in isolation.
                for (uint32_t siblingIdx : members) {
                    if (siblingIdx < dag.nodes.size() && dag.nodes[siblingIdx].isMasked != node.isMasked) {
                        MarkFailure(outErrors, label + "'s isMasked (" + std::to_string(node.isMasked) +
                            ") does not match sibling group member " + std::to_string(siblingIdx) +
                            "'s isMasked (" + std::to_string(dag.nodes[siblingIdx].isMasked) + ")");
                        break;
                    }
                }
            }

            // --- If this node was itself produced by a group, verify that group lists it back as
            // one of its outputs, then descend into that group's members (this node's children),
            // checking the reverse (child -> parent-group) consistency ---------------------------
            if (node.sourceGroupIndex != kInvalidDAGGroupIndex) {
                if (node.sourceGroupIndex >= dag.groups.size()) {
                    MarkFailure(outErrors, label + " has an out-of-range sourceGroupIndex " + std::to_string(node.sourceGroupIndex));
                }
                else {
                    const ClusterDAGGroup& sourceGroup = dag.groups[node.sourceGroupIndex];
                    const auto& outputs = sourceGroup.outputClusterIndices;
                    if (std::find(outputs.begin(), outputs.end(), nodeIndex) == outputs.end()) {
                        MarkFailure(outErrors, label + "'s source group (" + std::to_string(node.sourceGroupIndex) +
                            ") does not list it back as an output");
                    }

                    for (uint32_t childIdx : sourceGroup.memberClusterIndices) {
                        if (childIdx >= dag.nodes.size()) {
                            MarkFailure(outErrors, label + " has an out-of-range child index " + std::to_string(childIdx));
                            continue;
                        }
                        if (dag.nodes[childIdx].parentGroupIndex != node.sourceGroupIndex) {
                            MarkFailure(outErrors, label + " lists node " + std::to_string(childIdx) +
                                " as a child, but that node's parentGroupIndex does not point back to this node's source group");
                        }
                        ValidateReachableFrom(dag, childIdx, state, outErrors);
                    }
                }
            }

            state[nodeIndex] = VisitState::Done;
        }

        // Group-scoped invariants ValidateReachableFrom's per-node walk doesn't already cover
        // directly: every group must actually have produced at least one output (a group with zero
        // outputs would mean EmitSimplifiedGroup's earlyRoots bailout path leaked a group entry it
        // should never have created -- that path returns before ever calling dag.groups.push_back)
        // and must stay within the documented 1-4 member / 1-kMaxGroupOutputClusters output range.
        void ValidateGroupShapes(const ClusterDAG& dag, std::vector<std::string>& outErrors) {
            for (uint32_t gi = 0; gi < dag.groups.size(); ++gi) {
                const ClusterDAGGroup& group = dag.groups[gi];
                const std::string label = "group " + std::to_string(gi);

                if (group.memberClusterIndices.empty() || group.memberClusterIndices.size() > 4) {
                    MarkFailure(outErrors, label + " has " + std::to_string(group.memberClusterIndices.size()) +
                        " members, outside the documented 1-4 range");
                }
                if (group.outputClusterIndices.empty() || group.outputClusterIndices.size() > kMaxGroupOutputClusters) {
                    MarkFailure(outErrors, label + " has " + std::to_string(group.outputClusterIndices.size()) +
                        " outputs, outside the documented 1-" + std::to_string(kMaxGroupOutputClusters) + " range");
                }
            }
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
            if (dag.nodes[rootIdx].parentGroupIndex != kInvalidDAGGroupIndex) {
                MarkFailure(outErrors, "node " + std::to_string(rootIdx) +
                    " is listed in rootIndices but has a non-null parentGroupIndex");
            }
            ValidateReachableFrom(dag, rootIdx, state, outErrors);
        }

        for (uint32_t i = 0; i < dag.nodes.size(); ++i) {
            if (state[i] != VisitState::Done) {
                MarkFailure(outErrors, "node " + std::to_string(i) +
                    " is not reachable from any declared root (orphaned/disconnected from the DAG's forest)");
            }
        }

        ValidateGroupShapes(dag, outErrors);

        return outErrors.empty();
    }

}
