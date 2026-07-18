#include "HlodPipeline.h"

#include <algorithm>
#include <utility>

#include "ArchetypeMeshLibrary.h"
#include "pcg/PcgSeededRandom.h"

#ifdef WORLDPARTITION_WITH_MESHOPTIMIZER
#include <meshoptimizer.h>
#endif

namespace worldpartition {

    namespace {

        // Transforms a LOCAL-space position by a local-to-world matrix, treating it as a point
        // (w=1, translation applied) rather than a direction -- the same "column-major, w implicitly
        // 1, drop the w row" convention renderer::PcgInstanceDrawPass.cpp's own local TransformPoint
        // helper already uses for exactly this purpose (per-instance PCG point transform). Kept as
        // its own local helper here (not promoted to a shared maths:: free function) since this
        // codebase's established convention -- see that same PcgInstanceDrawPass.cpp precedent -- is
        // for each call site owning a small anonymous-namespace copy rather than growing Maths.h's
        // public surface for a two-line matrix-vector multiply.
        maths::vec3 TransformPoint(const maths::mat4& localToWorld, const maths::vec3& p) {
            return maths::vec3(
                localToWorld.m[0] * p.x + localToWorld.m[4] * p.y + localToWorld.m[8] * p.z + localToWorld.m[12],
                localToWorld.m[1] * p.x + localToWorld.m[5] * p.y + localToWorld.m[9] * p.z + localToWorld.m[13],
                localToWorld.m[2] * p.x + localToWorld.m[6] * p.y + localToWorld.m[10] * p.z + localToWorld.m[14]);
        }

        // Shared tail of every HLOD entry point (BuildHlodForCell and BuildHlodForPcgScatter alike):
        // merge the already-gathered source meshes into one buffer, simplify down to the level's
        // triangle budget via the caller-selected backend, then re-derive world-space bounds from the
        // POST-simplification vertex positions (see HlodProxyMesh's own comment for why bounds must
        // be recomputed here rather than inherited from the pre-simplification source). Factored out
        // so neither entry point duplicates this merge/simplify/bounds sequence or the QEM call it
        // wraps -- the only thing that differs between an authored-actor cell and a PCG instance
        // scatter is HOW the source SimplifiableMesh list is gathered (GatherCellMeshes vs.
        // GatherPcgScatterMeshes), never what happens to it afterward.
        HlodProxyMesh MergeSimplifyAndBoundProxy(
            std::vector<geometry::SimplifiableMesh>&& gathered, uint32_t triangleBudget, ISimplificationBackend& backend) {

            HlodProxyMesh proxy;
            proxy.mesh = MergeCellMeshes(gathered);
            backend.Simplify(proxy.mesh, triangleBudget);

            maths::ResetAABB(proxy.bounds.boundsMin, proxy.bounds.boundsMax);
            for (const maths::vec3& position : proxy.mesh.positions) {
                maths::ExpandAABB(proxy.bounds.boundsMin, proxy.bounds.boundsMax, position);
            }

            return proxy;
        }

    }

    std::vector<HlodLevel> BuildHlodLevelChain(float baseCellSize, uint32_t baseTriangleBudget, uint32_t numLevels) {
        std::vector<HlodLevel> levels;
        levels.reserve(numLevels);

        float cellSize = baseCellSize;
        for (uint32_t i = 0; i < numLevels; ++i) {
            HlodLevel level;
            level.levelIndex = i;
            level.cellSize = cellSize;
            level.triangleBudget = baseTriangleBudget;
            levels.push_back(level);
            cellSize *= 2.0f;
        }

        return levels;
    }

    std::vector<geometry::SimplifiableMesh> GatherCellMeshes(const SpatialHashCell& cell, const ActorMeshFetchFn& fetchMesh) {
        std::vector<geometry::SimplifiableMesh> meshes;
        meshes.reserve(cell.actorUuids.size());

        for (const Uuid& actorUuid : cell.actorUuids) {
            geometry::SimplifiableMesh mesh;
            if (fetchMesh(actorUuid, mesh)) {
                meshes.push_back(std::move(mesh));
            }
        }

        return meshes;
    }

    geometry::SimplifiableMesh MergeCellMeshes(const std::vector<geometry::SimplifiableMesh>& sourceMeshes) {
        geometry::SimplifiableMesh merged;

        size_t totalVertices = 0;
        size_t totalTriangleIndices = 0;
        for (const geometry::SimplifiableMesh& source : sourceMeshes) {
            totalVertices += source.positions.size();
            totalTriangleIndices += source.triangles.size();
        }
        merged.positions.reserve(totalVertices);
        merged.uvs.reserve(totalVertices);
        merged.locked.reserve(totalVertices);
        merged.triangles.reserve(totalTriangleIndices);

        for (const geometry::SimplifiableMesh& source : sourceMeshes) {
            uint32_t vertexOffset = static_cast<uint32_t>(merged.positions.size());

            merged.positions.insert(merged.positions.end(), source.positions.begin(), source.positions.end());
            merged.uvs.insert(merged.uvs.end(), source.uvs.begin(), source.uvs.end());
            merged.locked.insert(merged.locked.end(), source.locked.begin(), source.locked.end());

            for (uint32_t index : source.triangles) {
                merged.triangles.push_back(index + vertexOffset);
            }
        }

        return merged;
    }

    uint32_t NativeQEMSimplificationBackend::Simplify(geometry::SimplifiableMesh& mesh, uint32_t targetTriangleCount) {
        return geometry::SimplifyMeshQEM(mesh, targetTriangleCount);
    }

#ifdef WORLDPARTITION_WITH_MESHOPTIMIZER
    MeshOptimizerSimplificationBackend::MeshOptimizerSimplificationBackend(float targetErrorRatio)
        : targetErrorRatio_(targetErrorRatio) {
    }

    uint32_t MeshOptimizerSimplificationBackend::Simplify(geometry::SimplifiableMesh& mesh, uint32_t targetTriangleCount) {
        if (mesh.triangles.empty() || mesh.positions.empty()) return 0;

        size_t targetIndexCount = static_cast<size_t>(targetTriangleCount) * 3;
        std::vector<uint32_t> simplifiedIndices(mesh.triangles.size());
        float resultError = 0.0f;

        // maths::vec3 is exactly 3 contiguous floats (no padding, no vtable), so it can be handed
        // to meshoptimizer's flat-float-array API directly via its stride parameter -- no
        // intermediate copy needed.
        size_t newIndexCount = meshopt_simplify(
            simplifiedIndices.data(),
            mesh.triangles.data(), mesh.triangles.size(),
            reinterpret_cast<const float*>(mesh.positions.data()), mesh.positions.size(), sizeof(maths::vec3),
            targetIndexCount, targetErrorRatio_, /*options*/ 0, &resultError);

        simplifiedIndices.resize(newIndexCount);
        mesh.triangles = std::move(simplifiedIndices);

        return static_cast<uint32_t>(newIndexCount / 3);
    }
#endif

    HlodProxyMesh BuildHlodForCell(
        const SpatialHashCell& cell, const ActorMeshFetchFn& fetchMesh,
        const HlodLevel& level, ISimplificationBackend& backend) {

        std::vector<geometry::SimplifiableMesh> gathered = GatherCellMeshes(cell, fetchMesh);
        return MergeSimplifyAndBoundProxy(std::move(gathered), level.triangleBudget, backend);
    }

    std::vector<geometry::SimplifiableMesh> GatherPcgScatterMeshes(
        const std::vector<pcg::PcgPoint>& scatteredPoints, const std::vector<uint32_t>& perPointArchetypeShape) {

        // Parallel arrays: process only the shorter length's worth of pairs so a caller-side size
        // mismatch (a bug elsewhere) can never read either vector out of bounds here.
        const size_t count = std::min(scatteredPoints.size(), perPointArchetypeShape.size());

        std::vector<geometry::SimplifiableMesh> meshes;
        meshes.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            // Every archetype mesh is authored in LOCAL space, centered at its own origin (see
            // ArchetypeMeshLibrary.h's own header comment) -- exactly the space PcgPoint's bounds/
            // density fields are already defined in, so composing GetLocalToWorld() (Translate *
            // FromQuat * Scale, see PcgPointData.h) and applying it to every vertex is the correct,
            // complete way to place this instance's contribution in world space before merging.
            geometry::SimplifiableMesh instanceMesh = BuildArchetypeMesh(perPointArchetypeShape[i]);

            const maths::mat4 localToWorld = scatteredPoints[i].GetLocalToWorld();
            for (maths::vec3& position : instanceMesh.positions) {
                position = TransformPoint(localToWorld, position);
            }

            meshes.push_back(std::move(instanceMesh));
        }

        return meshes;
    }

    HlodProxyMesh BuildHlodForPcgScatter(
        const std::vector<pcg::PcgPoint>& scatteredPoints, const std::vector<uint32_t>& perPointArchetypeShape,
        const HlodLevel& level, ISimplificationBackend& backend) {

        std::vector<geometry::SimplifiableMesh> gathered = GatherPcgScatterMeshes(scatteredPoints, perPointArchetypeShape);
        return MergeSimplifyAndBoundProxy(std::move(gathered), level.triangleBudget, backend);
    }

    HlodProxyMesh BuildHlodForSyntheticPcgScatterDemo(
        uint32_t pointCount, float scatterRadius, uint32_t seed, uint32_t triangleBudget) {

        std::vector<pcg::PcgPoint> scatteredPoints;
        std::vector<uint32_t> perPointArchetypeShape;
        scatteredPoints.reserve(pointCount);
        perPointArchetypeShape.reserve(pointCount);

        // PcgSeededRandom is a stateless "hash(seed, index) -> value" stream (see PcgSeededRandom.h's
        // own header comment): reusing the exact same seed here always reproduces the exact same
        // synthetic scatter, matching this codebase's hard "the show must play back identically every
        // run" determinism requirement -- no different than a real PCG graph's own sampler output.
        pcg::PcgSeededRandom rng(seed);

        for (uint32_t i = 0; i < pointCount; ++i) {
            pcg::PcgPoint point;

            // Uniform-disc placement (rejection-free polar sampling: sqrt(u) for radius keeps area
            // density uniform rather than clumping toward the center) on the flat XZ plane, y=0 --
            // a simple, reviewable stand-in for what a real PcgVolumeSampler/PcgSurfaceSampler
            // (Phase 2, already merged to main) would produce for a ground-hugging prop scatter.
            const float angle = rng.NextFloatRange(0.0f, 2.0f * maths::PI);
            const float radius = std::sqrt(rng.NextFloat01()) * scatterRadius;
            point.position = maths::vec3{ radius * std::cos(angle), 0.0f, radius * std::sin(angle) };

            // Random yaw (rotation about world-up) -- a rock/bush/tree/debris instance has no
            // preferred forward-facing direction, so a full [0, 2*PI) range is appropriate (unlike
            // e.g. a directional prop that would need a constrained range).
            point.rotation = maths::quat::FromAxisAngle(maths::vec3{ 0.0f, 1.0f, 0.0f }, rng.NextFloatRange(0.0f, 2.0f * maths::PI));

            // Small uniform-scale jitter (+/-25%) -- enough per-instance size variation to be
            // visually distinguishable from a perfectly uniform scatter, without producing degenerate
            // (near-zero or wildly oversized) archetype instances that would skew the merged proxy's
            // bounds unrealistically.
            const float uniformScale = rng.NextFloatRange(0.75f, 1.25f);
            point.scale = maths::vec3{ uniformScale, uniformScale, uniformScale };

            point.seed = rng.NextUint32();

            scatteredPoints.push_back(point);
            // Uniform random archetype shape assignment over the full fixed 4-shape pool
            // (Rock/Bush/Tree/Debris, see ArchetypeMeshLibrary::kArchetypeShapeCount).
            perPointArchetypeShape.push_back(static_cast<uint32_t>(rng.NextIntRange(0, static_cast<int32_t>(kArchetypeShapeCount) - 1)));
        }

        HlodLevel level;
        level.levelIndex = 0;
        level.cellSize = scatterRadius * 2.0f;
        level.triangleBudget = triangleBudget;

        NativeQEMSimplificationBackend backend;
        return BuildHlodForPcgScatter(scatteredPoints, perPointArchetypeShape, level, backend);
    }

    bool ShelfPackAtlasBaker::PackMaterialsIntoAtlas(
        const std::vector<uint32_t>& materialIds, uint32_t requestedTileSize,
        uint32_t atlasSize, std::vector<HlodAtlasTile>& outTiles) {

        if (requestedTileSize == 0 || atlasSize < requestedTileSize) return false;

        uint32_t tilesPerRow = atlasSize / requestedTileSize;
        uint32_t rowCount = atlasSize / requestedTileSize;
        uint32_t capacity = tilesPerRow * rowCount;

        if (materialIds.size() > capacity) return false; // Budget mistuned: caller must lower requestedTileSize, raise atlasSize, or split into multiple atlases.

        outTiles.clear();
        outTiles.reserve(materialIds.size());

        for (size_t i = 0; i < materialIds.size(); ++i) {
            uint32_t row = static_cast<uint32_t>(i) / tilesPerRow;
            uint32_t col = static_cast<uint32_t>(i) % tilesPerRow;

            HlodAtlasTile tile;
            tile.sourceMaterialId = materialIds[i];
            tile.atlasOffsetX = col * requestedTileSize;
            tile.atlasOffsetY = row * requestedTileSize;
            tile.width = requestedTileSize;
            tile.height = requestedTileSize;
            outTiles.push_back(tile);
        }

        return true;
    }

}
