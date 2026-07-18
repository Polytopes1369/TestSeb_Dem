// Standalone, framework-free unit test for the World Partition HLOD pipeline architecture
// (tools/WorldPartition/HlodPipeline.h): mesh gathering, merging, the native QEM simplification
// backend, the HLOD level-chain builder, and the uniform-grid atlas packer. Exits 0 if every
// check passes, non-zero otherwise -- registered with CTest (see the top-level CMakeLists.txt),
// matching this project's existing tests/*.cpp convention.
//
// PCG roadmap Phase 4.3 ("PCG -> HLOD Integration") coverage added below: GatherPcgScatterMeshes /
// BuildHlodForPcgScatter / BuildHlodForSyntheticPcgScatterDemo -- the PCG-scatter analogue of this
// file's existing TestGatherAndMerge / TestBuildHlodForCell coverage.

#include "WorldPartition/HlodPipeline.h"
#include "WorldPartition/ArchetypeMeshLibrary.h"
#include "WorldPartition/Uuid.h"
#include "geometry/MeshSimplifier.h"
#include "pcg/PcgPointData.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    // A simple 2-triangle quad, offset on X so two instances never share a vertex position.
    geometry::SimplifiableMesh MakeQuad(float offsetX) {
        geometry::SimplifiableMesh mesh;
        mesh.positions = {
            { offsetX + 0.0f, 0.0f, 0.0f },
            { offsetX + 1.0f, 0.0f, 0.0f },
            { offsetX + 1.0f, 1.0f, 0.0f },
            { offsetX + 0.0f, 1.0f, 0.0f },
        };
        mesh.uvs = { {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f} };
        mesh.locked = { false, false, false, false };
        mesh.triangles = { 0, 1, 2, 0, 2, 3 };
        return mesh;
    }

    void TestGatherAndMerge() {
        worldpartition::UuidGenerator gen(42u);
        worldpartition::Uuid uuidA = gen.Generate();
        worldpartition::Uuid uuidB = gen.Generate();
        worldpartition::Uuid uuidMissing = gen.Generate();

        worldpartition::SpatialHashCell cell;
        cell.coord = { 0, 0, 0 };
        cell.actorUuids = { uuidA, uuidB, uuidMissing };

        worldpartition::ActorMeshFetchFn fetch = [&](const worldpartition::Uuid& id, geometry::SimplifiableMesh& out) -> bool {
            if (id == uuidA) { out = MakeQuad(0.0f); return true; }
            if (id == uuidB) { out = MakeQuad(10.0f); return true; }
            return false; // uuidMissing: no mesh contributed (e.g. a non-visual actor) -- must be skipped, not an error.
            };

        std::vector<geometry::SimplifiableMesh> gathered = worldpartition::GatherCellMeshes(cell, fetch);
        Check(gathered.size() == 2, "GatherCellMeshes should skip the actor its fetch callback rejects");

        geometry::SimplifiableMesh merged = worldpartition::MergeCellMeshes(gathered);
        Check(merged.positions.size() == 8, "MergeCellMeshes should concatenate both quads' 4 vertices each");
        Check(merged.triangles.size() == 12, "MergeCellMeshes should concatenate both quads' 2 triangles each (6 indices each)");

        // Second source mesh's triangle indices must be offset by the first mesh's vertex count (4).
        bool secondMeshOffsetCorrect = merged.triangles.size() >= 12 &&
            merged.triangles[6] == 4 && merged.triangles[7] == 5 && merged.triangles[8] == 6;
        Check(secondMeshOffsetCorrect, "MergeCellMeshes: second source mesh's triangle indices were not correctly offset");
    }

    void TestBuildHlodForCell() {
        worldpartition::UuidGenerator gen(43u);
        worldpartition::Uuid uuidA = gen.Generate();
        worldpartition::Uuid uuidB = gen.Generate();

        worldpartition::SpatialHashCell cell;
        cell.coord = { 0, 0, 0 };
        cell.actorUuids = { uuidA, uuidB };

        worldpartition::ActorMeshFetchFn fetch = [&](const worldpartition::Uuid& id, geometry::SimplifiableMesh& out) -> bool {
            if (id == uuidA) { out = MakeQuad(0.0f); return true; }
            if (id == uuidB) { out = MakeQuad(10.0f); return true; }
            return false;
            };

        worldpartition::NativeQEMSimplificationBackend backend;
        worldpartition::HlodLevel level;
        level.levelIndex = 0;
        level.cellSize = 100.0f;
        level.triangleBudget = 2;

        worldpartition::HlodProxyMesh proxy = worldpartition::BuildHlodForCell(cell, fetch, level, backend);

        uint32_t triangleCount = static_cast<uint32_t>(proxy.mesh.triangles.size() / 3);
        Check(triangleCount <= 4, "BuildHlodForCell: simplification must never increase the original triangle count");

        if (!proxy.mesh.positions.empty()) {
            Check(proxy.bounds.boundsMin.x <= proxy.bounds.boundsMax.x &&
                proxy.bounds.boundsMin.y <= proxy.bounds.boundsMax.y &&
                proxy.bounds.boundsMin.z <= proxy.bounds.boundsMax.z,
                "BuildHlodForCell: resulting bounds must be a valid (min <= max) AABB");

            // Both source quads span x in [0,1] and [10,11]; the merged/simplified proxy's bounds
            // must still cover that full combined extent (simplification moves geometry, but
            // never discards a whole disjoint component when unconstrained by any lock).
            Check(proxy.bounds.boundsMin.x <= 0.5f && proxy.bounds.boundsMax.x >= 10.5f,
                "BuildHlodForCell: proxy bounds should still span both source quads' X extent");
        }
    }

    void TestBuildHlodLevelChain() {
        std::vector<worldpartition::HlodLevel> levels = worldpartition::BuildHlodLevelChain(100.0f, 500u, 3u);
        Check(levels.size() == 3, "BuildHlodLevelChain: expected exactly numLevels entries");

        if (levels.size() == 3) {
            Check(levels[0].levelIndex == 0 && levels[1].levelIndex == 1 && levels[2].levelIndex == 2, "BuildHlodLevelChain: levelIndex should be sequential");
            Check(levels[0].cellSize == 100.0f && levels[1].cellSize == 200.0f && levels[2].cellSize == 400.0f, "BuildHlodLevelChain: cellSize should double each level");
            Check(levels[0].triangleBudget == 500u && levels[1].triangleBudget == 500u && levels[2].triangleBudget == 500u, "BuildHlodLevelChain: triangleBudget should stay constant across levels");
        }
    }

    void TestShelfPackAtlasBaker() {
        worldpartition::ShelfPackAtlasBaker baker;

        std::vector<uint32_t> materials = { 1, 2, 3, 4 };
        std::vector<worldpartition::HlodAtlasTile> tiles;
        bool ok = baker.PackMaterialsIntoAtlas(materials, 64u, 128u, tiles);

        Check(ok, "ShelfPackAtlasBaker: 4 tiles of 64 should fit a 128x128 atlas (2x2 grid)");
        Check(tiles.size() == 4, "ShelfPackAtlasBaker: expected one tile per material");

        if (tiles.size() == 4) {
            Check(tiles[0].atlasOffsetX == 0 && tiles[0].atlasOffsetY == 0, "ShelfPackAtlasBaker: tile 0 should be at origin");
            Check(tiles[1].atlasOffsetX == 64 && tiles[1].atlasOffsetY == 0, "ShelfPackAtlasBaker: tile 1 should be to the right of tile 0");
            Check(tiles[2].atlasOffsetX == 0 && tiles[2].atlasOffsetY == 64, "ShelfPackAtlasBaker: tile 2 should start the second row");
            Check(tiles[3].atlasOffsetX == 64 && tiles[3].atlasOffsetY == 64, "ShelfPackAtlasBaker: tile 3 should be the second row's second column");
        }

        std::vector<uint32_t> tooMany = { 1, 2, 3, 4, 5 };
        std::vector<worldpartition::HlodAtlasTile> overflowTiles;
        bool overflowOk = baker.PackMaterialsIntoAtlas(tooMany, 64u, 128u, overflowTiles);
        Check(!overflowOk, "ShelfPackAtlasBaker: 5 tiles of 64 must NOT fit a 128x128 atlas (capacity is 4)");
    }

    // -------------------------------------------------------------------------------------------
    // PCG roadmap Phase 4.3 ("PCG -> HLOD Integration") coverage.
    // -------------------------------------------------------------------------------------------

    // Builds a deterministic PcgPoint at `position`/`scale` with identity rotation -- enough control
    // to verify GatherPcgScatterMeshes' transform composition (Translate * FromQuat * Scale) without
    // needing a non-identity rotation case (mat4::FromQuat itself is exercised elsewhere, e.g.
    // PcgDataModelSmokeTest.cpp; this file's concern is HlodPipeline's OWN use of GetLocalToWorld(),
    // not re-testing the matrix math itself).
    pcg::PcgPoint MakePoint(const maths::vec3& position, float uniformScale = 1.0f) {
        pcg::PcgPoint point;
        point.position = position;
        point.scale = maths::vec3{ uniformScale, uniformScale, uniformScale };
        return point;
    }

    void TestGatherPcgScatterMeshes_TransformComposition() {
        // Rock (archetype shape 0) is a radius-0.5 icosphere built in LOCAL space centered at its
        // own origin (see ArchetypeMeshLibrary.h) -- translating a single instance to (5,0,0) must
        // therefore produce a mesh whose AABB is centered exactly at (5,0,0) with a ~0.5 half-extent
        // on every axis.
        std::vector<pcg::PcgPoint> points = { MakePoint(maths::vec3{ 5.0f, 0.0f, 0.0f }) };
        std::vector<uint32_t> shapes = { 0u };

        std::vector<geometry::SimplifiableMesh> gathered = worldpartition::GatherPcgScatterMeshes(points, shapes);
        Check(gathered.size() == 1, "GatherPcgScatterMeshes: one point in, one instance mesh out");

        if (!gathered.empty()) {
            geometry::SimplifiableMesh reference = worldpartition::BuildArchetypeMesh(0u);
            Check(gathered[0].triangles.size() == reference.triangles.size(),
                "GatherPcgScatterMeshes: transforming an instance must not change its topology/triangle count");

            maths::vec3 boundsMin, boundsMax;
            maths::ResetAABB(boundsMin, boundsMax);
            for (const maths::vec3& p : gathered[0].positions) maths::ExpandAABB(boundsMin, boundsMax, p);

            maths::vec3 center = maths::AABBCenter(boundsMin, boundsMax);
            Check(std::abs(center.x - 5.0f) < 1.0e-3f && std::abs(center.y) < 1.0e-3f && std::abs(center.z) < 1.0e-3f,
                "GatherPcgScatterMeshes: instance AABB must be centered at the point's world position");

            float halfExtentX = (boundsMax.x - boundsMin.x) * 0.5f;
            Check(halfExtentX > 0.4f && halfExtentX < 0.6f,
                "GatherPcgScatterMeshes: instance AABB half-extent should match the Rock archetype's ~0.5 radius");
        }

        // Non-uniform-scale-free (uniform) scale of 2x must double the instance's radius.
        std::vector<pcg::PcgPoint> scaledPoints = { MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }, 2.0f) };
        std::vector<geometry::SimplifiableMesh> scaledGathered = worldpartition::GatherPcgScatterMeshes(scaledPoints, shapes);
        if (!scaledGathered.empty()) {
            maths::vec3 boundsMin, boundsMax;
            maths::ResetAABB(boundsMin, boundsMax);
            for (const maths::vec3& p : scaledGathered[0].positions) maths::ExpandAABB(boundsMin, boundsMax, p);
            float halfExtentX = (boundsMax.x - boundsMin.x) * 0.5f;
            Check(halfExtentX > 0.9f && halfExtentX < 1.1f,
                "GatherPcgScatterMeshes: PcgPoint::scale must be applied (2x scale -> ~1.0 radius)");
        }
    }

    void TestGatherPcgScatterMeshes_ParallelArrayMismatchIsSafe() {
        // A caller-side bug (mismatched array lengths) must never read either vector out of bounds;
        // only the shorter length's worth of pairs is processed.
        std::vector<pcg::PcgPoint> points = { MakePoint({0,0,0}), MakePoint({1,0,0}), MakePoint({2,0,0}) };
        std::vector<uint32_t> shapes = { 0u, 1u }; // Deliberately shorter than `points`.

        std::vector<geometry::SimplifiableMesh> gathered = worldpartition::GatherPcgScatterMeshes(points, shapes);
        Check(gathered.size() == 2, "GatherPcgScatterMeshes: mismatched parallel arrays must process only the shorter length, not crash");
    }

    void TestBuildHlodForPcgScatter_EmptyScatterIsGraceful() {
        worldpartition::NativeQEMSimplificationBackend backend;
        worldpartition::HlodLevel level;
        level.levelIndex = 0;
        level.cellSize = 20.0f;
        level.triangleBudget = 64u;

        worldpartition::HlodProxyMesh proxy =
            worldpartition::BuildHlodForPcgScatter({}, {}, level, backend);

        Check(proxy.mesh.positions.empty(), "BuildHlodForPcgScatter: an empty scatter must produce an empty mesh, not crash");
        Check(proxy.mesh.triangles.empty(), "BuildHlodForPcgScatter: an empty scatter must produce zero triangles");
        // ResetAABB leaves boundsMin > boundsMax when never expanded -- the documented "degenerate,
        // never-touched" AABB representation this codebase's own ExpandAABB/ResetAABB pair uses.
        Check(proxy.bounds.boundsMin.x > proxy.bounds.boundsMax.x,
            "BuildHlodForPcgScatter: an empty scatter's bounds must stay in the degenerate ResetAABB state");
    }

    void TestBuildHlodForPcgScatter_SingleInstancePerArchetypeShape() {
        worldpartition::NativeQEMSimplificationBackend backend;
        worldpartition::HlodLevel level;
        level.levelIndex = 0;
        level.cellSize = 20.0f;
        level.triangleBudget = 100000u; // Deliberately unreachable-large: forces a simplification no-op so exact archetype topology survives.

        for (uint32_t shape = 0; shape < worldpartition::kArchetypeShapeCount; ++shape) {
            std::vector<pcg::PcgPoint> points = { MakePoint(maths::vec3{ 0.0f, 0.0f, 0.0f }) };
            std::vector<uint32_t> shapes = { shape };

            worldpartition::HlodProxyMesh proxy = worldpartition::BuildHlodForPcgScatter(points, shapes, level, backend);
            geometry::SimplifiableMesh reference = worldpartition::BuildArchetypeMesh(shape);

            uint32_t proxyTriangleCount = static_cast<uint32_t>(proxy.mesh.triangles.size() / 3);
            uint32_t referenceTriangleCount = static_cast<uint32_t>(reference.triangles.size() / 3);
            Check(proxyTriangleCount > 0, "BuildHlodForPcgScatter: single-instance proxy must be non-degenerate (archetype shape " + std::to_string(shape) + ")");
            // Not a strict equality: SimplifyMeshQEM's early-exit path (target already met) still
            // calls WeldResidualSliverTriangles unconditionally (see MeshSimplifier.cpp's own
            // comment on that call), which removes any ALREADY-DEGENERATE zero-area triangle the
            // source archetype mesh itself contains -- e.g. the Bush (UV sphere) archetype's own
            // pole triangles, explicitly called out as "collapse to a zero-area triangle harmlessly"
            // in BuildBushMesh's header comment. So the proxy's count can be slightly BELOW the raw
            // archetype's nominal count, but must never exceed it.
            Check(proxyTriangleCount <= referenceTriangleCount,
                "BuildHlodForPcgScatter: with an unreachably large budget, single-instance triangle count must never exceed the source archetype's own count (shape " + std::to_string(shape) + ")");

            Check(proxy.bounds.boundsMin.x <= proxy.bounds.boundsMax.x &&
                proxy.bounds.boundsMin.y <= proxy.bounds.boundsMax.y &&
                proxy.bounds.boundsMin.z <= proxy.bounds.boundsMax.z,
                "BuildHlodForPcgScatter: resulting bounds must be a valid (min <= max) AABB (shape " + std::to_string(shape) + ")");

            // Every archetype shape's local geometry stays within a 1-unit radius of its own origin
            // (Rock/Bush: 0.5 radius, Tree: ~0.65 max extent, Debris: ~0.47 max extent) -- at the
            // origin with no scale, the proxy's bounds must therefore stay comfortably inside [-1, 1].
            Check(std::abs(proxy.bounds.boundsMin.x) <= 1.0f && std::abs(proxy.bounds.boundsMax.x) <= 1.0f &&
                std::abs(proxy.bounds.boundsMin.y) <= 1.0f && std::abs(proxy.bounds.boundsMax.y) <= 1.0f &&
                std::abs(proxy.bounds.boundsMin.z) <= 1.0f && std::abs(proxy.bounds.boundsMax.z) <= 1.0f,
                "BuildHlodForPcgScatter: single origin-centered instance bounds must stay within a 1-unit radius (shape " + std::to_string(shape) + ")");
        }
    }

    void TestBuildHlodForPcgScatter_BoundsEncompassAllScatteredPositions() {
        // A moderate, explicit (non-random) scatter across a grid, cycling all 4 archetype shapes --
        // with a deliberately unreachable-large triangle budget (no simplification collapse actually
        // runs, see SimplifyMeshQEM's own early-exit contract for target >= current triangle count),
        // every instance's full, un-reduced local geometry survives into the final proxy. Since every
        // archetype mesh's local origin (0,0,0) lies strictly inside its own vertex convex hull, and
        // GetLocalToWorld() maps that local origin exactly onto the point's world position, the
        // resulting merged AABB is GUARANTEED (not just probabilistically likely) to contain every
        // scattered point's exact world position.
        std::vector<pcg::PcgPoint> points;
        std::vector<uint32_t> shapes;
        constexpr uint32_t kGridSize = 6; // 6x6 = 36 points.
        for (uint32_t gx = 0; gx < kGridSize; ++gx) {
            for (uint32_t gz = 0; gz < kGridSize; ++gz) {
                maths::vec3 position{ static_cast<float>(gx) * 3.0f, 0.0f, static_cast<float>(gz) * 3.0f };
                points.push_back(MakePoint(position));
                shapes.push_back((gx + gz) % worldpartition::kArchetypeShapeCount);
            }
        }

        worldpartition::NativeQEMSimplificationBackend backend;
        worldpartition::HlodLevel level;
        level.levelIndex = 0;
        level.cellSize = 20.0f;
        level.triangleBudget = 1000000u; // Unreachable: no collapse should actually be applied.

        worldpartition::HlodProxyMesh proxy = worldpartition::BuildHlodForPcgScatter(points, shapes, level, backend);

        bool allEncompassed = true;
        constexpr float kArchetypeMaxExtent = 1.0f; // See TestBuildHlodForPcgScatter_SingleInstancePerArchetypeShape's own bound.
        for (const pcg::PcgPoint& point : points) {
            bool inside =
                point.position.x >= proxy.bounds.boundsMin.x - kArchetypeMaxExtent && point.position.x <= proxy.bounds.boundsMax.x + kArchetypeMaxExtent &&
                point.position.y >= proxy.bounds.boundsMin.y - kArchetypeMaxExtent && point.position.y <= proxy.bounds.boundsMax.y + kArchetypeMaxExtent &&
                point.position.z >= proxy.bounds.boundsMin.z - kArchetypeMaxExtent && point.position.z <= proxy.bounds.boundsMax.z + kArchetypeMaxExtent;
            allEncompassed = allEncompassed && inside;
        }
        Check(allEncompassed, "BuildHlodForPcgScatter: proxy bounds must encompass every scattered instance's world position");

        // The un-simplified merged mesh's own triangle count must be AT MOST the exact sum of every
        // instance's own archetype triangle count (no real QEM collapse applied at this budget) --
        // not necessarily equal, since SimplifyMeshQEM's early-exit path still unconditionally welds
        // away any already-degenerate zero-area sliver present in the source (e.g. the Bush
        // archetype's own pole triangles, see TestBuildHlodForPcgScatter_SingleInstancePerArchetypeShape's
        // own comment on this exact behavior) but must still be non-zero (36 instances contribute
        // real geometry) and reasonably close to the raw sum (sliver welding only removes a handful
        // of degenerate triangles per Bush instance, never a large fraction of the mesh).
        uint64_t expectedTriangleCount = 0;
        for (uint32_t shape : shapes) expectedTriangleCount += worldpartition::BuildArchetypeMesh(shape).triangles.size() / 3;
        uint64_t actualTriangleCount = proxy.mesh.triangles.size() / 3;
        Check(actualTriangleCount > 0 && actualTriangleCount <= expectedTriangleCount,
            "BuildHlodForPcgScatter: with an unreachable budget, merged triangle count must be non-zero and never exceed the exact sum of every instance's archetype triangle count");
        Check(actualTriangleCount >= (expectedTriangleCount * 9) / 10,
            "BuildHlodForPcgScatter: with an unreachable budget, sliver-welding must only remove a small fraction of the raw merged triangle count, not silently drop real geometry");
    }

    void TestBuildHlodForPcgScatter_LargeScatterTriangleCountIsBounded() {
        // Real-world-shaped case (Phase 4.3's own stated verification target): 100-200 PCG-scattered
        // instances feeding into a small, realistic per-cell HLOD triangle budget -- must collapse
        // down to a bounded, non-exploding proxy, not choke or silently ignore the budget.
        constexpr uint32_t kPointCount = 180;
        constexpr uint32_t kTriangleBudget = 300u;

        worldpartition::HlodProxyMesh proxy =
            worldpartition::BuildHlodForSyntheticPcgScatterDemo(kPointCount, /*scatterRadius=*/20.0f, /*seed=*/0x1234ABCDu, kTriangleBudget);

        uint32_t proxyTriangleCount = static_cast<uint32_t>(proxy.mesh.triangles.size() / 3);
        Check(proxyTriangleCount > 0, "BuildHlodForSyntheticPcgScatterDemo: a 180-instance scatter must produce a non-degenerate proxy");

        // Generous slack over the nominal budget (QEM may overshoot a hard target slightly when the
        // candidate queue empties before reaching it exactly, see SimplifyMeshQEM's own contract) --
        // still tight enough to prove the proxy did NOT simply pass the ~20,000+ raw input triangles
        // through unreduced.
        Check(proxyTriangleCount <= kTriangleBudget * 2,
            "BuildHlodForSyntheticPcgScatterDemo: simplification must bound the proxy's triangle count near the requested budget, not let it explode");

        Check(proxy.bounds.boundsMin.x <= proxy.bounds.boundsMax.x &&
            proxy.bounds.boundsMin.y <= proxy.bounds.boundsMax.y &&
            proxy.bounds.boundsMin.z <= proxy.bounds.boundsMax.z,
            "BuildHlodForSyntheticPcgScatterDemo: resulting bounds must be a valid (min <= max) AABB");

        // Every point is scattered within `scatterRadius` (20) of the origin on the flat XZ plane, so
        // the proxy's bounds must stay within a generous margin of that disc (never explode outward).
        constexpr float kMaxPlausibleExtent = 20.0f + 2.0f; // scatterRadius + max archetype half-extent.
        Check(std::abs(proxy.bounds.boundsMin.x) <= kMaxPlausibleExtent && std::abs(proxy.bounds.boundsMax.x) <= kMaxPlausibleExtent &&
            std::abs(proxy.bounds.boundsMin.z) <= kMaxPlausibleExtent && std::abs(proxy.bounds.boundsMax.z) <= kMaxPlausibleExtent,
            "BuildHlodForSyntheticPcgScatterDemo: proxy bounds must stay within a plausible margin of the scatter disc, not explode");
    }

}

int main() {
    TestGatherAndMerge();
    TestBuildHlodForCell();
    TestBuildHlodLevelChain();
    TestShelfPackAtlasBaker();

    TestGatherPcgScatterMeshes_TransformComposition();
    TestGatherPcgScatterMeshes_ParallelArrayMismatchIsSafe();
    TestBuildHlodForPcgScatter_EmptyScatterIsGraceful();
    TestBuildHlodForPcgScatter_SingleInstancePerArchetypeShape();
    TestBuildHlodForPcgScatter_BoundsEncompassAllScatteredPositions();
    TestBuildHlodForPcgScatter_LargeScatterTriangleCountIsBounded();

    if (g_failCount == 0) {
        std::cout << "[PASS] All HLOD pipeline checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
