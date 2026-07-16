#include "geometry/MeshSimplifier.h"
#include "geometry/GeometryHashUtil.h"
#include "core/Logger.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <queue>
#include <unordered_set>

namespace geometry {

    namespace {

        // -----------------------------------------------------------------------------------
        // Quadric error matrix: a symmetric 4x4 matrix Q such that v^T Q v approximates the sum
        // of squared distances from v to the set of triangle planes that contributed to Q
        // (Garland & Heckbert, "Surface Simplification Using Quadric Error Metrics", 1997).
        // Stored as its 10 independent upper-triangular coefficients in double precision, since
        // repeated accumulation across many triangles is numerically sensitive in float.
        //
        //     | a b c d |
        //     | b e f g |
        //     | c f h i |
        //     | d g i j |
        // -----------------------------------------------------------------------------------
        struct Quadric {
            double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
            double e = 0.0, f = 0.0, g = 0.0;
            double h = 0.0, i = 0.0;
            double j = 0.0;

            Quadric& operator+=(const Quadric& o) {
                a += o.a; b += o.b; c += o.c; d += o.d;
                e += o.e; f += o.f; g += o.g;
                h += o.h; i += o.i;
                j += o.j;
                return *this;
            }

            Quadric operator+(const Quadric& o) const {
                Quadric r = *this;
                r += o;
                return r;
            }

            // v^T Q v for v = (x,y,z,1).
            double Evaluate(const maths::vec3& v) const {
                double x = v.x, y = v.y, z = v.z;
                return a * x * x + 2.0 * b * x * y + 2.0 * c * x * z + 2.0 * d * x
                    + e * y * y + 2.0 * f * y * z + 2.0 * g * y
                    + h * z * z + 2.0 * i * z
                    + j;
            }
        };

        // Plane quadric contribution of one triangle. The plane is stored unit-normalized, so
        // this is the classic unweighted Garland-Heckbert quadric (each plane contributes
        // equally regardless of triangle area).
        Quadric PlaneQuadric(const maths::vec3& p0, const maths::vec3& p1, const maths::vec3& p2) {
            maths::vec3 e1 = p1 - p0;
            maths::vec3 e2 = p2 - p0;
            maths::vec3 n = e1.Cross(e2);
            float lengthSq = n.Dot(n);
            if (lengthSq <= 1e-20f) {
                return Quadric{}; // Degenerate (zero-area) triangle contributes nothing.
            }
            maths::vec3 unitNormal = n.Normalize();
            double nx = unitNormal.x, ny = unitNormal.y, nz = unitNormal.z;
            double dd = -(nx * p0.x + ny * p0.y + nz * p0.z);

            Quadric q;
            q.a = nx * nx; q.b = nx * ny; q.c = nx * nz; q.d = nx * dd;
            q.e = ny * ny; q.f = ny * nz; q.g = ny * dd;
            q.h = nz * nz; q.i = nz * dd;
            q.j = dd * dd;
            return q;
        }

        // Solves the 3x3 linear system from Q's upper-left block for the quadric-optimal
        // position: A*v = -[d,g,i], A = [[a,b,c],[b,e,f],[c,f,h]]. Returns false (leaving outPos
        // untouched) if A is (near-)singular, in which case the caller falls back to the
        // cheapest of a small set of candidate positions instead.
        bool SolveOptimalPosition(const Quadric& q, maths::vec3& outPos) {
            double A00 = q.a, A01 = q.b, A02 = q.c;
            double A10 = q.b, A11 = q.e, A12 = q.f;
            double A20 = q.c, A21 = q.f, A22 = q.h;

            double det = A00 * (A11 * A22 - A12 * A21)
                - A01 * (A10 * A22 - A12 * A20)
                + A02 * (A10 * A21 - A11 * A20);

            constexpr double kMinDeterminant = 1e-12;
            if (std::fabs(det) < kMinDeterminant) {
                return false;
            }

            double bx = -q.d, by = -q.g, bz = -q.i;
            double invDet = 1.0 / det;

            // Cramer's rule.
            double x = (bx * (A11 * A22 - A12 * A21) - A01 * (by * A22 - A12 * bz) + A02 * (by * A21 - A11 * bz)) * invDet;
            double y = (A00 * (by * A22 - A12 * bz) - bx * (A10 * A22 - A12 * A20) + A02 * (A10 * bz - by * A20)) * invDet;
            double z = (A00 * (A11 * bz - by * A21) - A01 * (A10 * bz - by * A20) + bx * (A10 * A21 - A11 * A20)) * invDet;

            outPos = maths::vec3{ static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) };
            return true;
        }

        // One candidate edge collapse sitting in the priority queue. `keep`/`remove` are fixed
        // at push time: the locked endpoint, if any, is always `keep`. version{Keep,Remove}
        // snapshot MeshState::version[...] at push time, so a stale entry (either vertex touched
        // by a later collapse) is detected cheaply on pop instead of eagerly invalidated.
        struct CollapseCandidate {
            double cost;
            uint32_t keep;
            uint32_t remove;
            uint32_t versionKeep;
            uint32_t versionRemove;
            maths::vec3 target;
        };
        struct CollapseCandidateGreater {
            bool operator()(const CollapseCandidate& a, const CollapseCandidate& b) const {
                return a.cost > b.cost; // Inverted so std::priority_queue behaves as a min-heap.
            }
        };

        // Mutable working state for one simplification pass.
        struct MeshState {
            std::vector<maths::vec3> positions;
            std::vector<bool> locked;
            std::vector<maths::vec2> uvs;
            std::vector<bool> aliveVertex;
            std::vector<uint32_t> version;

            std::vector<uint32_t> triVerts; // 3 slots per triangle
            std::vector<bool> aliveTriangle;

            std::vector<Quadric> quadric;
            // Incident triangle indices per vertex. Every entry is guaranteed (by construction,
            // see the collapse-application loop below) to genuinely reference that vertex for as
            // long as the vertex itself is alive; entries for triangles later killed are simply
            // skipped via aliveTriangle at each use site rather than eagerly erased.
            std::vector<std::vector<uint32_t>> vertexTriangles;

            uint32_t triangleCount = 0;
            uint32_t vertexCount = 0; // Live count of alive vertices; decremented by one per applied collapse.
        };

        constexpr uint32_t kDeadSentinel = std::numeric_limits<uint32_t>::max();

        // Triangles whose post-collapse normal deviates too far from their pre-collapse normal
        // (cosine of the angle between them drops below this) are treated as a fold-over and
        // reject the whole candidate collapse, guarding against self-intersecting geometry.
        constexpr float kFoldOverCosineThreshold = 1e-3f;
        // Minimum acceptable squared (twice-)area for a post-collapse triangle; anything smaller
        // is rejected as a degenerate sliver.
        constexpr float kMinTriangleAreaSq = 1e-14f;

        maths::vec3 TriangleNormalUnnormalized(const maths::vec3& p0, const maths::vec3& p1, const maths::vec3& p2) {
            return (p1 - p0).Cross(p2 - p0);
        }

        void AddTriangleQuadrics(MeshState& state, uint32_t triIndex) {
            uint32_t v0 = state.triVerts[triIndex * 3 + 0];
            uint32_t v1 = state.triVerts[triIndex * 3 + 1];
            uint32_t v2 = state.triVerts[triIndex * 3 + 2];
            Quadric q = PlaneQuadric(state.positions[v0], state.positions[v1], state.positions[v2]);
            state.quadric[v0] += q;
            state.quadric[v1] += q;
            state.quadric[v2] += q;
        }

        bool TriangleContainsVertex(const MeshState& state, uint32_t triIndex, uint32_t vertex) {
            return state.triVerts[triIndex * 3 + 0] == vertex
                || state.triVerts[triIndex * 3 + 1] == vertex
                || state.triVerts[triIndex * 3 + 2] == vertex;
        }

        // Computes the cost/target position for collapsing the edge (v0,v1), honoring locks.
        // Returns false if the edge must never be offered (both endpoints locked).
        bool EvaluateCandidate(const MeshState& state, uint32_t v0, uint32_t v1, CollapseCandidate& out) {
            bool lockedA = state.locked[v0];
            bool lockedB = state.locked[v1];
            if (lockedA && lockedB) {
                return false;
            }

            uint32_t keep = lockedA ? v0 : (lockedB ? v1 : v0);
            uint32_t remove = (keep == v0) ? v1 : v0;

            Quadric combined = state.quadric[keep] + state.quadric[remove];

            maths::vec3 target;
            double cost;
            if (state.locked[keep]) {
                // Locked survivor: position is fixed, no optimization freedom whatsoever.
                target = state.positions[keep];
                cost = combined.Evaluate(target);
            }
            else {
                maths::vec3 optimal;
                if (SolveOptimalPosition(combined, optimal)) {
                    target = optimal;
                    cost = combined.Evaluate(target);
                }
                else {
                    // Singular quadric (e.g. perfectly planar neighborhood): fall back to the
                    // cheapest of {keep, remove, midpoint} — the standard pragmatic QEM fallback.
                    maths::vec3 mid = (state.positions[keep] + state.positions[remove]) * 0.5f;
                    double costKeepPos = combined.Evaluate(state.positions[keep]);
                    double costRemovePos = combined.Evaluate(state.positions[remove]);
                    double costMid = combined.Evaluate(mid);

                    target = state.positions[keep];
                    cost = costKeepPos;
                    if (costRemovePos < cost) { cost = costRemovePos; target = state.positions[remove]; }
                    if (costMid < cost) { cost = costMid; target = mid; }
                }
            }

            out.cost = cost;
            out.keep = keep;
            out.remove = remove;
            out.versionKeep = state.version[keep];
            out.versionRemove = state.version[remove];
            out.target = target;
            return true;
        }

        // Checks whether moving `keep` to `target` (and remapping `remove` -> `keep`) would flip
        // or degenerate any surviving triangle incident to either vertex. Triangles that contain
        // BOTH keep and remove collapse to a line/point and are excluded from this check — they
        // are the ones that get killed outright, not remapped.
        bool CollapseWouldFoldOver(const MeshState& state, uint32_t keep, uint32_t remove, const maths::vec3& target) {
            auto checkList = [&](const std::vector<uint32_t>& triList) -> bool {
                for (uint32_t triIndex : triList) {
                    if (!state.aliveTriangle[triIndex]) {
                        continue;
                    }
                    if (TriangleContainsVertex(state, triIndex, keep) && TriangleContainsVertex(state, triIndex, remove)) {
                        continue; // Degenerates away; not a surviving triangle.
                    }

                    uint32_t v0 = state.triVerts[triIndex * 3 + 0];
                    uint32_t v1 = state.triVerts[triIndex * 3 + 1];
                    uint32_t v2 = state.triVerts[triIndex * 3 + 2];
                    maths::vec3 p0 = state.positions[v0];
                    maths::vec3 p1 = state.positions[v1];
                    maths::vec3 p2 = state.positions[v2];
                    maths::vec3 oldNormal = TriangleNormalUnnormalized(p0, p1, p2);

                    if (v0 == keep || v0 == remove) p0 = target;
                    if (v1 == keep || v1 == remove) p1 = target;
                    if (v2 == keep || v2 == remove) p2 = target;
                    maths::vec3 newNormal = TriangleNormalUnnormalized(p0, p1, p2);

                    float newAreaSq = newNormal.Dot(newNormal);
                    if (newAreaSq < kMinTriangleAreaSq) {
                        return true; // Would collapse to a sliver/degenerate triangle.
                    }

                    float oldAreaSq = oldNormal.Dot(oldNormal);
                    if (oldAreaSq <= 1e-24f) {
                        continue; // Was already degenerate; don't block on it.
                    }
                    float cosAngle = oldNormal.Dot(newNormal) / std::sqrt(oldAreaSq * newAreaSq);
                    if (cosAngle < kFoldOverCosineThreshold) {
                        return true; // Normal flipped (or came close to it): reject.
                    }
                }
                return false;
                };

            if (checkList(state.vertexTriangles[keep])) return true;
            if (checkList(state.vertexTriangles[remove])) return true;
            return false;
        }

    } // namespace

    uint32_t SimplifyMeshQEM(
        SimplifiableMesh& mesh, uint32_t targetTriangleCount, uint32_t targetVertexCount, float* outMaxError) {
        const uint32_t vertexCount = static_cast<uint32_t>(mesh.positions.size());
        const uint32_t initialTriangleCount = static_cast<uint32_t>(mesh.triangles.size() / 3);

        if (vertexCount == 0 || (initialTriangleCount <= targetTriangleCount && vertexCount <= targetVertexCount)) {
            if (outMaxError) {
                *outMaxError = 0.0f; // No collapse applied: geometry is unchanged, zero error introduced.
            }
            return initialTriangleCount;
        }

        double maxAppliedCollapseCostSq = 0.0;

        MeshState state;
        state.positions = mesh.positions;
        state.locked = mesh.locked;
        state.locked.resize(vertexCount, false); // Defensive: tolerate a shorter locked array.
        state.uvs = mesh.uvs;
        state.uvs.resize(vertexCount, maths::vec2{ 0.0f, 0.0f }); // Defensive: tolerate a shorter/absent uvs array.
        state.aliveVertex.assign(vertexCount, true);
        state.version.assign(vertexCount, 0u);
        state.quadric.assign(vertexCount, Quadric{});
        state.vertexTriangles.assign(vertexCount, {});

        state.triVerts = mesh.triangles;
        state.aliveTriangle.assign(initialTriangleCount, true);
        state.triangleCount = initialTriangleCount;
        state.vertexCount = vertexCount;

        for (uint32_t t = 0; t < initialTriangleCount; ++t) {
            uint32_t v0 = state.triVerts[t * 3 + 0];
            uint32_t v1 = state.triVerts[t * 3 + 1];
            uint32_t v2 = state.triVerts[t * 3 + 2];
            state.vertexTriangles[v0].push_back(t);
            state.vertexTriangles[v1].push_back(t);
            state.vertexTriangles[v2].push_back(t);
            AddTriangleQuadrics(state, t);
        }

        std::priority_queue<CollapseCandidate, std::vector<CollapseCandidate>, CollapseCandidateGreater> queue;

        std::unordered_set<uint64_t> seenEdges;
        auto pushEdgeIfNew = [&](uint32_t a, uint32_t b) {
            uint64_t packed = PackOrderedPair(a, b);
            if (!seenEdges.insert(packed).second) {
                return;
            }
            CollapseCandidate candidate;
            if (EvaluateCandidate(state, a, b, candidate)) {
                queue.push(candidate);
            }
            };

        for (uint32_t t = 0; t < initialTriangleCount; ++t) {
            uint32_t v0 = state.triVerts[t * 3 + 0];
            uint32_t v1 = state.triVerts[t * 3 + 1];
            uint32_t v2 = state.triVerts[t * 3 + 2];
            pushEdgeIfNew(v0, v1);
            pushEdgeIfNew(v1, v2);
            pushEdgeIfNew(v2, v0);
        }

        // Defensive iteration cap: each successful collapse strictly reduces triangleCount by at
        // least one, and every rejected/stale pop is O(1), so this bound is generous relative to
        // any realistic cluster-group size and only guards against an unforeseen logic error
        // looping forever; it is never expected to be hit in practice.
        const uint64_t kMaxIterations = static_cast<uint64_t>(queue.size()) * 64ull + 4096ull;
        uint64_t iterations = 0;

        while (!queue.empty()
            && (state.triangleCount > targetTriangleCount || state.vertexCount > targetVertexCount)
            && iterations < kMaxIterations) {
            ++iterations;

            CollapseCandidate candidate = queue.top();
            queue.pop();

            uint32_t keep = candidate.keep;
            uint32_t remove = candidate.remove;

            if (!state.aliveVertex[keep] || !state.aliveVertex[remove]) {
                continue; // One side no longer exists.
            }
            if (state.version[keep] != candidate.versionKeep || state.version[remove] != candidate.versionRemove) {
                continue; // Stale: something about either vertex changed since this was pushed.
            }
            if (CollapseWouldFoldOver(state, keep, remove, candidate.target)) {
                continue; // Reject this specific collapse; do not requeue it.
            }

            maxAppliedCollapseCostSq = std::max(maxAppliedCollapseCostSq, candidate.cost);

            // --- Apply the collapse -------------------------------------------------------
            if (!state.locked[keep]) {
                state.positions[keep] = candidate.target;
                // UV midpoint blend: QEM has no attribute-aware error term for UV, so this is a
                // simple, reviewable approximation (not a placeholder) -- see SimplifiableMesh::uvs.
                state.uvs[keep] = (state.uvs[keep] + state.uvs[remove]) * 0.5f;
            }

            // Remap every triangle touching `remove`: either it also touches `keep` (degenerate,
            // kill it) or it survives with `remove` replaced by `keep`.
            for (uint32_t triIndex : state.vertexTriangles[remove]) {
                if (!state.aliveTriangle[triIndex]) {
                    continue;
                }
                if (TriangleContainsVertex(state, triIndex, keep)) {
                    state.aliveTriangle[triIndex] = false;
                    state.triangleCount -= 1;
                    continue;
                }
                for (int slot = 0; slot < 3; ++slot) {
                    if (state.triVerts[triIndex * 3 + slot] == remove) {
                        state.triVerts[triIndex * 3 + slot] = keep;
                    }
                }
                state.vertexTriangles[keep].push_back(triIndex);
            }

            state.quadric[keep] += state.quadric[remove];
            state.aliveVertex[remove] = false;
            state.vertexCount -= 1;
            state.version[keep] += 1;
            state.version[remove] += 1;
            state.vertexTriangles[remove].clear();
            state.vertexTriangles[remove].shrink_to_fit();

            // Re-evaluate every edge still incident to `keep` against its new position/quadric.
            std::unordered_set<uint32_t> neighbors;
            for (uint32_t triIndex : state.vertexTriangles[keep]) {
                if (!state.aliveTriangle[triIndex]) {
                    continue;
                }
                for (int slot = 0; slot < 3; ++slot) {
                    uint32_t v = state.triVerts[triIndex * 3 + slot];
                    if (v != keep) {
                        neighbors.insert(v);
                    }
                }
            }
            for (uint32_t neighbor : neighbors) {
                CollapseCandidate refreshed;
                if (EvaluateCandidate(state, keep, neighbor, refreshed)) {
                    queue.push(refreshed);
                }
            }
        }

        // --- Compact: keep only vertices referenced by a surviving triangle -------------------
        std::vector<uint32_t> remap(vertexCount, kDeadSentinel);
        std::vector<maths::vec3> newPositions;
        std::vector<bool> newLocked;
        std::vector<maths::vec2> newUVs;
        std::vector<uint32_t> newTriangles;
        newPositions.reserve(vertexCount);
        newTriangles.reserve(static_cast<size_t>(state.triangleCount) * 3);

        for (uint32_t t = 0; t < initialTriangleCount; ++t) {
            if (!state.aliveTriangle[t]) {
                continue;
            }
            for (int slot = 0; slot < 3; ++slot) {
                uint32_t v = state.triVerts[t * 3 + slot];
                if (remap[v] == kDeadSentinel) {
                    remap[v] = static_cast<uint32_t>(newPositions.size());
                    newPositions.push_back(state.positions[v]);
                    newLocked.push_back(state.locked[v]);
                    newUVs.push_back(state.uvs[v]);
                }
                newTriangles.push_back(remap[v]);
            }
        }

        mesh.positions = std::move(newPositions);
        mesh.locked = std::move(newLocked);
        mesh.uvs = std::move(newUVs);
        mesh.triangles = std::move(newTriangles);

#ifndef NDEBUG
        // Sanity check for the 2026-07-16 "clusters mis-generated during triangulation -> cluster
        // conversion" investigation: CollapseWouldFoldOver (above) rejects any COLLAPSE that would
        // CREATE a degenerate/near-zero-area triangle, but a triangle that was ALREADY degenerate in
        // the input mesh (e.g. cone/cylinder cap-ring index padding -- see
        // VirtualGeometryCacheTest.cpp's own comment on this) contributes a zero quadric
        // (PlaneQuadric's kMinTriangleAreaSq-equivalent early-out above) and is simply never touched
        // by a collapse, so it survives unflagged into the output mesh. This walks the FINAL
        // triangle list once to report how many (still) are degenerate, so it's possible to tell
        // whether bad input geometry or the simplifier itself is the source.
        {
            uint32_t degenerateCount = 0;
            uint32_t firstDegenerateTriangle = 0xFFFFFFFFu;
            for (uint32_t t = 0; t < mesh.triangles.size() / 3; ++t) {
                const maths::vec3& p0 = mesh.positions[mesh.triangles[t * 3 + 0]];
                const maths::vec3& p1 = mesh.positions[mesh.triangles[t * 3 + 1]];
                const maths::vec3& p2 = mesh.positions[mesh.triangles[t * 3 + 2]];
                maths::vec3 cross = (p1 - p0).Cross(p2 - p0);
                float areaSq = cross.Dot(cross) * 0.25f;
                if (areaSq < kMinTriangleAreaSq) {
                    ++degenerateCount;
                    if (firstDegenerateTriangle == 0xFFFFFFFFu) firstDegenerateTriangle = t;
                }
            }
            if (degenerateCount > 0) {
                LOG_WARNING(std::format(
                    "[MeshSimplifier] output mesh has {} degenerate (near-zero-area) triangle(s) out of {} "
                    "(first at triangle index {}) -- these were never collapsed away because a zero-area "
                    "triangle contributes nothing to any vertex's quadric. Likely inherited from the input "
                    "mesh (e.g. primitive cap-ring padding) rather than introduced by simplification itself.",
                    degenerateCount, mesh.triangles.size() / 3, firstDegenerateTriangle));
            }
        }
#endif

        if (outMaxError) {
            // cost is a squared-distance quadric evaluation; sqrt gives an approximate RMS
            // geometric deviation in the same units as the mesh's positions.
            *outMaxError = static_cast<float>(std::sqrt(std::max(0.0, maxAppliedCollapseCostSq)));
        }

        return static_cast<uint32_t>(mesh.triangles.size() / 3);
    }

    std::vector<maths::vec3> ComputeFaceAccumulatedNormals(const SimplifiableMesh& mesh) {
        std::vector<maths::vec3> normals(mesh.positions.size(), maths::vec3{ 0.0f, 0.0f, 0.0f });
        for (size_t t = 0; t + 2 < mesh.triangles.size(); t += 3) {
            uint32_t i0 = mesh.triangles[t + 0];
            uint32_t i1 = mesh.triangles[t + 1];
            uint32_t i2 = mesh.triangles[t + 2];
            maths::vec3 faceNormal = (mesh.positions[i1] - mesh.positions[i0]).Cross(mesh.positions[i2] - mesh.positions[i0]);
            normals[i0] = normals[i0] + faceNormal;
            normals[i1] = normals[i1] + faceNormal;
            normals[i2] = normals[i2] + faceNormal;
        }
        for (maths::vec3& n : normals) {
            n = n.Normalize();
        }
        return normals;
    }

}
