#include "CellManifest.h"

#include <cstring>
#include <fstream>

namespace world {

    namespace {
        // Byte-for-byte mirror of worldpartition::RuntimeCellManifestHeader/Record
        // (tools/WorldPartition/RuntimeCellManifest.h) -- deliberately re-declared rather than
        // shared, see this file's own header comment.
        inline constexpr uint32_t kRuntimeCellManifestMagic = 0x4D434C57u;
        inline constexpr uint32_t kRuntimeCellManifestVersion = 1u;

        struct RuntimeCellManifestHeader {
            uint32_t magic = 0;
            uint32_t version = 0;
            float cellSize = 0.0f;
            uint32_t recordCount = 0;
        };

        struct RuntimeCellManifestRecord {
            int32_t cellX = 0;
            int32_t cellZ = 0;
            uint32_t archetypeShape = 0;
            float localOffsetX = 0.0f;
            float localOffsetY = 0.0f;
            float localOffsetZ = 0.0f;
        };
    }

    bool CellManifest::Load(const std::filesystem::path& filePath) {
        m_Loaded = false;
        m_Placements.clear();

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

        m_Placements.reserve(records.size());
        for (const RuntimeCellManifestRecord& record : records) {
            CellCoord coord{ record.cellX, record.cellZ };
            m_Placements[coord] = CellPlacement{
                record.archetypeShape,
                maths::vec3{ record.localOffsetX, record.localOffsetY, record.localOffsetZ }
            };
        }

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
