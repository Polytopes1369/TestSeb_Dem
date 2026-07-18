#include "CellManifest.h"

#include <cstring>
#include <fstream>

namespace world {

    namespace {
        // Byte-for-byte mirror of worldpartition::RuntimeCellManifestHeader/Record/HlodVertex
        // (tools/WorldPartition/RuntimeCellManifest.h) -- deliberately re-declared rather than
        // shared, see this file's own header comment. Phase 5 (Streaming & Monde roadmap, Part 2,
        // Gap 2): v1 -> v2 bump -- keep this struct set edited together with the offline writer's
        // own copy whenever the format changes again (see that header's own version-bump comment).
        inline constexpr uint32_t kRuntimeCellManifestMagic = 0x4D434C57u;
        inline constexpr uint32_t kRuntimeCellManifestVersion = 2u;

        struct RuntimeCellManifestHeader {
            uint32_t magic = 0;
            uint32_t version = 0;
            float cellSize = 0.0f;
            uint32_t recordCount = 0;
            uint32_t hlodVertexBlobCount = 0;
            uint32_t hlodIndexBlobCount = 0;
        };

        struct RuntimeCellManifestRecord {
            int32_t cellX = 0;
            int32_t cellZ = 0;
            uint32_t archetypeShape = 0;
            float localOffsetX = 0.0f;
            float localOffsetY = 0.0f;
            float localOffsetZ = 0.0f;
            uint32_t hlodVertexOffset = 0;
            uint32_t hlodVertexCount = 0;
            uint32_t hlodIndexOffset = 0;
            uint32_t hlodIndexCount = 0;
        };

        struct RuntimeCellManifestHlodVertex {
            float x = 0.0f, y = 0.0f, z = 0.0f;
            float u = 0.0f, v = 0.0f;
        };
    }

    bool CellManifest::Load(const std::filesystem::path& filePath) {
        m_Loaded = false;
        m_Placements.clear();
        m_HlodVertices.clear();
        m_HlodIndices.clear();

        std::ifstream in(filePath, std::ios::binary);
        if (!in.is_open()) return false;

        RuntimeCellManifestHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in) return false;
        if (header.magic != kRuntimeCellManifestMagic || header.version != kRuntimeCellManifestVersion) return false;

        std::vector<RuntimeCellManifestRecord> records(header.recordCount);
        if (header.recordCount > 0) {
            std::streamsize expectedBytes = static_cast<std::streamsize>(header.recordCount * sizeof(RuntimeCellManifestRecord));
            in.read(reinterpret_cast<char*>(records.data()), expectedBytes);
            if (in.gcount() != expectedBytes) return false; // Truncated file.
        }

        std::vector<RuntimeCellManifestHlodVertex> hlodVertices(header.hlodVertexBlobCount);
        if (header.hlodVertexBlobCount > 0) {
            std::streamsize expectedBytes = static_cast<std::streamsize>(header.hlodVertexBlobCount * sizeof(RuntimeCellManifestHlodVertex));
            in.read(reinterpret_cast<char*>(hlodVertices.data()), expectedBytes);
            if (in.gcount() != expectedBytes) return false; // Truncated file.
        }

        std::vector<uint32_t> hlodIndices(header.hlodIndexBlobCount);
        if (header.hlodIndexBlobCount > 0) {
            std::streamsize expectedBytes = static_cast<std::streamsize>(header.hlodIndexBlobCount * sizeof(uint32_t));
            in.read(reinterpret_cast<char*>(hlodIndices.data()), expectedBytes);
            if (in.gcount() != expectedBytes) return false; // Truncated file.
        }

        m_Placements.reserve(records.size());
        for (const RuntimeCellManifestRecord& record : records) {
            CellCoord coord{ record.cellX, record.cellZ };
            m_Placements[coord] = CellPlacement{
                record.archetypeShape,
                maths::vec3{ record.localOffsetX, record.localOffsetY, record.localOffsetZ },
                record.hlodVertexOffset, record.hlodVertexCount,
                record.hlodIndexOffset, record.hlodIndexCount
            };
        }

        m_HlodVertices.reserve(hlodVertices.size());
        for (const RuntimeCellManifestHlodVertex& v : hlodVertices) {
            m_HlodVertices.push_back(CellHlodVertex{ v.x, v.y, v.z, v.u, v.v });
        }
        m_HlodIndices = std::move(hlodIndices);

        m_CellSize = header.cellSize;
        m_Loaded = true;
        return true;
    }

    std::optional<CellPlacement> CellManifest::GetPlacement(const CellCoord& coord) const {
        auto it = m_Placements.find(coord);
        if (it == m_Placements.end()) return std::nullopt;
        return it->second;
    }

}
