#include "RuntimeCellManifest.h"

#include <fstream>

namespace worldpartition {

    std::vector<RuntimeCellManifestRecord> BuildRuntimeCellManifest(
        const SpatialHashGrid& grid,
        const ActorClassNameFetchFn& fetchActor,
        const ClassNameToArchetypeShapeFn& classNameToShape,
        const HlodProxyFetchFn& fetchHlodProxy,
        std::vector<RuntimeCellManifestHlodVertex>& outHlodVertexBlob,
        std::vector<uint32_t>& outHlodIndexBlob) {

        std::vector<RuntimeCellManifestRecord> records;
        records.reserve(grid.CellCount());

        for (const auto& [coord, cell] : grid.Cells()) {
            if (cell.actorUuids.empty()) continue;

            // One representative actor per cell -- see this header's own comment for why. Always
            // the first (and, by BakeDemoWorld.cpp's own authoring convention, only) UUID.
            std::string className;
            ActorTransform transform;
            if (!fetchActor(cell.actorUuids[0], className, transform)) continue; // Skip, don't abort -- same convention as RebuildSceneIndexFromActorFiles.

            RuntimeCellManifestRecord record;
            record.cellX = coord.x;
            record.cellZ = coord.z;
            record.archetypeShape = classNameToShape(className);
            record.localOffsetX = transform.position.x;
            record.localOffsetY = transform.position.y;
            record.localOffsetZ = transform.position.z;

            // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2): append this cell's HLOD proxy
            // mesh (if any) to the shared blob, recording its LOCAL (0-based, see this header's own
            // field comment) vertex/index range in the record.
            geometry::SimplifiableMesh proxyMesh;
            if (fetchHlodProxy(cell, proxyMesh) && !proxyMesh.triangles.empty()) {
                record.hlodVertexOffset = static_cast<uint32_t>(outHlodVertexBlob.size());
                record.hlodVertexCount = static_cast<uint32_t>(proxyMesh.positions.size());
                record.hlodIndexOffset = static_cast<uint32_t>(outHlodIndexBlob.size());
                record.hlodIndexCount = static_cast<uint32_t>(proxyMesh.triangles.size());

                outHlodVertexBlob.reserve(outHlodVertexBlob.size() + proxyMesh.positions.size());
                for (size_t i = 0; i < proxyMesh.positions.size(); ++i) {
                    const maths::vec3& p = proxyMesh.positions[i];
                    // proxyMesh.uvs is always sized to match proxyMesh.positions (SimplifiableMesh's
                    // own documented invariant, maintained through every ArchetypeMeshLibrary
                    // constructor and geometry::SimplifyMeshQEM's own collapse-application step).
                    maths::vec2 uv = (i < proxyMesh.uvs.size()) ? proxyMesh.uvs[i] : maths::vec2{ 0.0f, 0.0f };
                    outHlodVertexBlob.push_back(RuntimeCellManifestHlodVertex{ p.x, p.y, p.z, uv.x, uv.y });
                }

                // Indices are already LOCAL to this mesh (0-based into proxyMesh.positions), which
                // is exactly this record's own local index convention -- no rebasing needed here,
                // only at the point a reader copies this sub-range into a DIFFERENT destination.
                outHlodIndexBlob.insert(outHlodIndexBlob.end(), proxyMesh.triangles.begin(), proxyMesh.triangles.end());
            }

            records.push_back(record);
        }

        return records;
    }

    bool WriteRuntimeCellManifest(const std::filesystem::path& filePath, float cellSize,
                                   const std::vector<RuntimeCellManifestRecord>& records,
                                   const std::vector<RuntimeCellManifestHlodVertex>& hlodVertexBlob,
                                   const std::vector<uint32_t>& hlodIndexBlob) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);

        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) return false;

        RuntimeCellManifestHeader header;
        header.cellSize = cellSize;
        header.recordCount = static_cast<uint32_t>(records.size());
        header.hlodVertexBlobCount = static_cast<uint32_t>(hlodVertexBlob.size());
        header.hlodIndexBlobCount = static_cast<uint32_t>(hlodIndexBlob.size());
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        if (!records.empty()) {
            out.write(reinterpret_cast<const char*>(records.data()),
                static_cast<std::streamsize>(records.size() * sizeof(RuntimeCellManifestRecord)));
        }
        if (!hlodVertexBlob.empty()) {
            out.write(reinterpret_cast<const char*>(hlodVertexBlob.data()),
                static_cast<std::streamsize>(hlodVertexBlob.size() * sizeof(RuntimeCellManifestHlodVertex)));
        }
        if (!hlodIndexBlob.empty()) {
            out.write(reinterpret_cast<const char*>(hlodIndexBlob.data()),
                static_cast<std::streamsize>(hlodIndexBlob.size() * sizeof(uint32_t)));
        }

        return out.good();
    }

}
