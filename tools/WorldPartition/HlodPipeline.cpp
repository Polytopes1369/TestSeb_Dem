#include "HlodPipeline.h"

#ifdef WORLDPARTITION_WITH_MESHOPTIMIZER
#include <meshoptimizer.h>
#endif

namespace worldpartition {

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

        HlodProxyMesh proxy;
        proxy.mesh = MergeCellMeshes(gathered);
        backend.Simplify(proxy.mesh, level.triangleBudget);

        maths::ResetAABB(proxy.bounds.boundsMin, proxy.bounds.boundsMax);
        for (const maths::vec3& position : proxy.mesh.positions) {
            maths::ExpandAABB(proxy.bounds.boundsMin, proxy.bounds.boundsMax, position);
        }

        return proxy;
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
