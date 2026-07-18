#pragma once
// Runtime reader for the flat binary manifest tools/WorldPartition/BakeDemoWorld.cpp exports (see
// tools/WorldPartition/RuntimeCellManifest.h for the shared byte layout both sides agree on).
// Deliberately depends on nothing under tools/WorldPartition/ -- only the plain POD record layout,
// duplicated here by convention (see StreamingTypes.h's own header comment on why src/ and tools/
// never share C++ types even when they describe the same on-disk format).
//
// Small enough to read wholesale at streaming-system startup (this demo's world_data/cellmanifest.bin
// holds on the order of dozens of records) -- exactly the same "index is cheap, read it all up
// front" reasoning SceneIndex.h's own header comment gives for its own (much larger, per-actor)
// index.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <vector>

#include "StreamingTypes.h"
#include "core/maths/Maths.h"

namespace world {

    // One HLOD proxy vertex, byte-for-byte mirror of worldpartition::RuntimeCellManifestHlodVertex
    // (tools/WorldPartition/RuntimeCellManifest.h) -- position + UV only, see that header's own
    // comment for why normals are deliberately excluded (recomputed at load time instead, via
    // geometry::ComputeFaceAccumulatedNormals -- see this class' own Load() comment).
    struct CellHlodVertex {
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float u = 0.0f, v = 0.0f;
    };

    // One representative prop placement for one occupied cell -- see RuntimeCellManifest.h's own
    // comment for why this is one-per-cell rather than a full actor list.
    //
    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2) v2 additions: hlodVertexOffset/Count and
    // hlodIndexOffset/Count index into THIS CellManifest's own GetHlodVertices()/GetHlodIndices()
    // blob arrays (NOT the raw on-disk byte offsets) -- see CellManifest::Load()'s own comment.
    // hlodIndexCount == 0 means this cell has no baked HLOD proxy (never a hard failure -- the
    // coarse streaming slot degrades exactly like "no authored content" in that case).
    struct CellPlacement {
        uint32_t archetypeShape = 0;
        maths::vec3 worldPosition{};
        uint32_t hlodVertexOffset = 0;
        uint32_t hlodVertexCount = 0;
        uint32_t hlodIndexOffset = 0;
        uint32_t hlodIndexCount = 0;
    };

    class CellManifest {
    public:
        // Reads `filePath` (the file BakeDemoWorld.cpp's WriteRuntimeCellManifest wrote). Returns
        // false on any I/O failure, magic/version mismatch, or truncated record/blob table --
        // exactly the same failure contract as worldpartition::ReadSceneIndex, whose format this
        // mirrors. A version mismatch (including an old v1 file against this v2 reader, or vice
        // versa) is NOT a special-cased migration path -- per this format's own header comment, it
        // is treated exactly like "file missing": this method returns false, and the whole session's
        // streaming stays gracefully disabled (see main.cpp's own "streaming disabled" log line at
        // its own Load() call site) -- no partial/best-effort parse of a mismatched layout is ever
        // attempted.
        bool Load(const std::filesystem::path& filePath);

        bool IsLoaded() const { return m_Loaded; }
        float CellSize() const { return m_CellSize; }
        size_t RecordCount() const { return m_Placements.size(); }

        // Returns the cell's placement, or std::nullopt if this cell has no authored content (most
        // cells, outside the small baked demo grid, will not).
        std::optional<CellPlacement> GetPlacement(const CellCoord& coord) const;

        // Every cell's HLOD proxy vertices/indices, concatenated in file order -- a CellPlacement's
        // own hlodVertexOffset/hlodIndexOffset index into these SAME arrays. Indices within one
        // cell's own [hlodIndexOffset, +hlodIndexCount) range are LOCAL to that cell's own
        // [hlodVertexOffset, +hlodVertexCount) range (0-based, matching RuntimeCellManifestRecord's
        // own documented convention) -- a caller copying one cell's sub-range elsewhere must add
        // hlodVertexOffset itself. NOTE (Phase 5, Part 2, Gap 3 -- NOT YET IMPLEMENTED as of this
        // comment): the intended consumer is a future renderer::VulkanContext::GenerateGeometry()
        // HLOD proxy bake-in step (one dedicated streaming slot's vertex/index range per authored
        // cell, staged-upload via memcpy + vkCmdCopyBuffer, no compute shader) -- this accessor and
        // the rest of the v2 manifest format are already real and round-tripped (see
        // tests/RuntimeCellManifestV2Tests.cpp), but nothing in this runtime reads them yet; the
        // streaming pool still falls back to its pre-Phase-5 shared-archetype behavior in the
        // meantime.
        const std::vector<CellHlodVertex>& GetHlodVertices() const { return m_HlodVertices; }
        const std::vector<uint32_t>& GetHlodIndices() const { return m_HlodIndices; }

    private:
        bool m_Loaded = false;
        float m_CellSize = 0.0f;
        std::unordered_map<CellCoord, CellPlacement, CellCoordHash> m_Placements;
        std::vector<CellHlodVertex> m_HlodVertices;
        std::vector<uint32_t> m_HlodIndices;
    };

}
