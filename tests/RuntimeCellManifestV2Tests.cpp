// Standalone, framework-free round-trip test for the v1 -> v2 RuntimeCellManifest format bump
// (Phase 5, Streaming & Monde roadmap, Part 2, Gap 2): writes a manifest via the OFFLINE writer
// (tools/WorldPartition/RuntimeCellManifest.cpp) and reads it back via the RUNTIME reader
// (src/world/CellManifest.cpp) -- the two intentionally-duplicated byte layouts this format's own
// header comment documents -- verifying every v2 field (the new HLOD proxy vertex/index blob
// sections and each record's range into them) round-trips exactly, and that a version-mismatched
// file is gracefully rejected (world::CellManifest::Load() returns false), never partially parsed.
// Exits 0 if every check passes, non-zero otherwise -- matches this project's existing
// tests/*.cpp / CTest convention (see e.g. tests/HlodPipelineTests.cpp).

#include "WorldPartition/RuntimeCellManifest.h"
#include "world/CellManifest.h"

#include <cstdio>
#include <filesystem>
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

    void TestV2RoundTrip() {
        const std::filesystem::path testPath = "runtime_cell_manifest_v2_roundtrip_test.bin";

        // Cell A: a real HLOD proxy (1 triangle, 3 vertices). Cell B: no proxy (hlodVertexCount
        // stays 0, exactly as BuildRuntimeCellManifest leaves it when fetchHlodProxy rejects a
        // cell) -- both cases must round-trip correctly, since "no proxy" is the expected common
        // case for any cell outside the small baked demo grid.
        worldpartition::RuntimeCellManifestRecord recordA;
        recordA.cellX = 3;
        recordA.cellZ = -2;
        recordA.archetypeShape = 1;
        recordA.localOffsetX = 61.0f;
        recordA.localOffsetY = 0.0f;
        recordA.localOffsetZ = -38.5f;
        recordA.hlodVertexOffset = 0;
        recordA.hlodVertexCount = 3;
        recordA.hlodIndexOffset = 0;
        recordA.hlodIndexCount = 3;

        worldpartition::RuntimeCellManifestRecord recordB;
        recordB.cellX = 0;
        recordB.cellZ = 0;
        recordB.archetypeShape = 2;
        recordB.localOffsetX = 5.0f;
        recordB.localOffsetY = 0.0f;
        recordB.localOffsetZ = 5.0f;
        // hlodVertexOffset/Count/hlodIndexOffset/Count all default to 0 -- "no proxy".

        std::vector<worldpartition::RuntimeCellManifestRecord> records = { recordA, recordB };
        std::vector<worldpartition::RuntimeCellManifestHlodVertex> vertexBlob = {
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
            { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f, 0.0f, 1.0f },
        };
        std::vector<uint32_t> indexBlob = { 0, 1, 2 };

        constexpr float kCellSize = 20.0f;
        bool wrote = worldpartition::WriteRuntimeCellManifest(testPath, kCellSize, records, vertexBlob, indexBlob);
        Check(wrote, "WriteRuntimeCellManifest should succeed writing a v2 manifest");

        world::CellManifest manifest;
        bool loaded = manifest.Load(testPath);
        Check(loaded, "world::CellManifest::Load should successfully parse a v2 manifest written by the offline writer");

        if (loaded) {
            Check(manifest.IsLoaded(), "CellManifest::IsLoaded should be true after a successful Load");
            Check(manifest.CellSize() == kCellSize, "CellManifest::CellSize should round-trip exactly");
            Check(manifest.RecordCount() == 2, "CellManifest::RecordCount should match the number of records written");

            std::optional<world::CellPlacement> placementA = manifest.GetPlacement(world::CellCoord{ 3, -2 });
            Check(placementA.has_value(), "GetPlacement should find cell A");
            if (placementA.has_value()) {
                Check(placementA->archetypeShape == 1, "Cell A archetypeShape should round-trip");
                Check(placementA->worldPosition.x == 61.0f && placementA->worldPosition.z == -38.5f,
                      "Cell A worldPosition should round-trip");
                Check(placementA->hlodVertexCount == 3, "Cell A hlodVertexCount should round-trip");
                Check(placementA->hlodIndexCount == 3, "Cell A hlodIndexCount should round-trip");
                Check(placementA->hlodVertexOffset == 0, "Cell A hlodVertexOffset should round-trip");
                Check(placementA->hlodIndexOffset == 0, "Cell A hlodIndexOffset should round-trip");
            }

            std::optional<world::CellPlacement> placementB = manifest.GetPlacement(world::CellCoord{ 0, 0 });
            Check(placementB.has_value(), "GetPlacement should find cell B");
            if (placementB.has_value()) {
                Check(placementB->hlodVertexCount == 0, "Cell B (no authored HLOD proxy) should round-trip hlodVertexCount == 0");
                Check(placementB->hlodIndexCount == 0, "Cell B (no authored HLOD proxy) should round-trip hlodIndexCount == 0");
            }

            std::optional<world::CellPlacement> missing = manifest.GetPlacement(world::CellCoord{ 99, 99 });
            Check(!missing.has_value(), "GetPlacement should return nullopt for an unauthored cell");

            const std::vector<world::CellHlodVertex>& blobVerts = manifest.GetHlodVertices();
            const std::vector<uint32_t>& blobIndices = manifest.GetHlodIndices();
            Check(blobVerts.size() == 3, "CellManifest::GetHlodVertices should contain exactly the 3 written vertices");
            Check(blobIndices.size() == 3, "CellManifest::GetHlodIndices should contain exactly the 3 written indices");
            if (blobVerts.size() == 3) {
                Check(blobVerts[1].x == 1.0f && blobVerts[1].u == 1.0f,
                      "CellManifest::GetHlodVertices should round-trip position/UV exactly");
            }
            if (blobIndices.size() == 3) {
                // Indices are LOCAL to each record's own vertex range (0-based) -- see this format's
                // own header comment -- so cell A's [0,1,2] indices stay untouched here (offset 0).
                Check(blobIndices[0] == 0 && blobIndices[1] == 1 && blobIndices[2] == 2,
                      "CellManifest::GetHlodIndices should round-trip local (0-based) indices exactly");
            }
        }

        std::error_code ec;
        std::filesystem::remove(testPath, ec);
    }

    void TestOldVersionRejected() {
        // Fabricate a v1-shaped file: same magic, but version == 1 and the OLD (smaller, no-HLOD)
        // record layout -- world::CellManifest::Load() (a v2-only reader) must reject this outright
        // (return false), never attempt a partial/best-effort parse of a mismatched layout, per this
        // format's own header comment ("no migration code is needed or written").
        struct OldHeader {
            uint32_t magic = 0x4D434C57u;
            uint32_t version = 1u; // Deliberately stale.
            float cellSize = 20.0f;
            uint32_t recordCount = 0;
        };

        const std::filesystem::path testPath = "runtime_cell_manifest_v1_reject_test.bin";
        {
            OldHeader header;
            FILE* f = nullptr;
            fopen_s(&f, testPath.string().c_str(), "wb");
            Check(f != nullptr, "Test setup: should be able to open the fabricated v1 file for writing");
            if (f) {
                fwrite(&header, sizeof(header), 1, f);
                fclose(f);
            }
        }

        world::CellManifest manifest;
        bool loaded = manifest.Load(testPath);
        Check(!loaded, "world::CellManifest::Load must reject a v1-versioned file (version mismatch), never partially parse it");
        Check(!manifest.IsLoaded(), "CellManifest::IsLoaded must stay false after a rejected Load");

        std::error_code ec;
        std::filesystem::remove(testPath, ec);
    }

}

int main() {
    TestV2RoundTrip();
    TestOldVersionRejected();

    if (g_failCount == 0) {
        std::cout << "[PASS] All RuntimeCellManifest v1->v2 round-trip checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
