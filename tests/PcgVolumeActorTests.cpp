// Standalone, framework-free unit test for the PCG Volume authoring layer
// (tools/WorldPartition/PcgVolumeActor.h, PCG roadmap Phase 6.1): round-trips a PcgVolumeDesc
// through BuildPcgVolumeActorRecord -> WriteActorFile -> ReadActorFile -> TryParsePcgVolumeDesc,
// verifies TryParsePcgVolumeDesc rejects a non-PcgVolume (and an incomplete PcgVolume-classed)
// actor record without crashing, and verifies ComputeOverlappingCells against single-cell,
// multi-cell, and degenerate/zero-size bounds cases. Exits 0 if every check passes, non-zero
// otherwise -- registered with CTest (see the top-level CMakeLists.txt), matching this project's
// existing tests/*.cpp convention (see tests/OfpaSerializationTests.cpp, tests/SpatialHashGridTests.cpp).

#include "WorldPartition/PcgVolumeActor.h"
#include "WorldPartition/OfpaActor.h"
#include "WorldPartition/SpatialHashGrid.h"
#include "WorldPartition/Uuid.h"
#include "core/maths/Maths.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

    int g_failCount = 0;

    void Check(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "[FAIL] " << message << "\n";
            ++g_failCount;
        }
    }

    bool NearlyEqual(float a, float b, float epsilon = 1.0e-4f) {
        return std::fabs(a - b) <= epsilon;
    }

    bool NearlyEqual(const maths::vec3& a, const maths::vec3& b, float epsilon = 1.0e-4f) {
        return NearlyEqual(a.x, b.x, epsilon) && NearlyEqual(a.y, b.y, epsilon) && NearlyEqual(a.z, b.z, epsilon);
    }

    bool AABBNearlyEqual(const worldpartition::AABB& a, const worldpartition::AABB& b) {
        return NearlyEqual(a.boundsMin, b.boundsMin) && NearlyEqual(a.boundsMax, b.boundsMax);
    }

    // Round-trips one PcgVolumeDesc through Build -> WriteActorFile -> ReadActorFile -> Parse and
    // checks every field comes back exactly. Parameterized by seed so both an ordinary and an
    // extreme (top-bit-set) seed value get exercised -- PropertyValue has no unsigned-integer
    // alternative, so PcgVolumeActor.cpp reinterprets uint32_t seeds through int32_t, and a seed
    // >= 0x80000000 is exactly the case that would silently break a naive (saturating, or
    // truncating) conversion instead of a true bit-pattern reinterpretation.
    void CheckPcgVolumeRoundTrip(const std::filesystem::path& scratchDir, const std::string& caseName,
        const worldpartition::AABB& bounds, const std::string& graphAssetPath, uint32_t seed) {
        worldpartition::UuidGenerator gen(4242u + seed);
        worldpartition::Uuid uuid = gen.Generate();

        worldpartition::PcgVolumeDesc original;
        original.bounds = bounds;
        original.graphAssetPath = graphAssetPath;
        original.seed = seed;

        worldpartition::ActorRecord record = worldpartition::BuildPcgVolumeActorRecord(uuid, original);
        Check(record.className == worldpartition::kPcgVolumeClassName, caseName + ": BuildPcgVolumeActorRecord className mismatch");
        Check(record.uuid == uuid, caseName + ": BuildPcgVolumeActorRecord uuid mismatch");
        Check(AABBNearlyEqual(record.worldBounds, original.bounds),
            caseName + ": BuildPcgVolumeActorRecord worldBounds should exactly reproduce desc.bounds under an identity transform");

        std::filesystem::path actorPath = worldpartition::MakeActorFilePath(scratchDir, uuid);
        Check(worldpartition::WriteActorFile(actorPath, record), caseName + ": WriteActorFile failed for a PcgVolume actor record");

        worldpartition::ActorRecord loaded;
        Check(worldpartition::ReadActorFile(actorPath, loaded), caseName + ": ReadActorFile failed for a PcgVolume actor record");

        worldpartition::PcgVolumeDesc parsed;
        bool parseOk = worldpartition::TryParsePcgVolumeDesc(loaded, parsed);
        Check(parseOk, caseName + ": TryParsePcgVolumeDesc should succeed for a record built by BuildPcgVolumeActorRecord and round-tripped through disk");
        if (!parseOk) return;

        Check(AABBNearlyEqual(parsed.bounds, original.bounds), caseName + ": round-trip bounds mismatch");
        Check(parsed.graphAssetPath == original.graphAssetPath, caseName + ": round-trip graphAssetPath mismatch");
        Check(parsed.seed == original.seed, caseName + ": round-trip seed mismatch (got " + std::to_string(parsed.seed) + ", expected " + std::to_string(original.seed) + ")");
    }

    void TestPcgVolumeRoundTrip(const std::filesystem::path& scratchDir) {
        worldpartition::AABB bounds;
        bounds.boundsMin = { -40.0f, 0.0f, -10.0f };
        bounds.boundsMax = { 60.0f, 25.0f, 30.0f };
        CheckPcgVolumeRoundTrip(scratchDir, "OrdinarySeed", bounds, "graphs/ForestScatter.pcggraph.json", 0xC0FFEEu);

        // Top-bit-set seed: exercises the int32_t bit-reinterpretation edge case (see
        // CheckPcgVolumeRoundTrip's own comment). Also uses an empty graphAssetPath to confirm an
        // empty-but-present string property still round-trips (as opposed to being mistaken for a
        // missing property).
        worldpartition::AABB tinyBounds;
        tinyBounds.boundsMin = { 0.0f, 0.0f, 0.0f };
        tinyBounds.boundsMax = { 1.0f, 1.0f, 1.0f };
        CheckPcgVolumeRoundTrip(scratchDir, "MaxSeed", tinyBounds, "", 0xFFFFFFFFu);
    }

    void TestTryParseRejectsNonPcgVolumeRecord() {
        worldpartition::UuidGenerator gen(99u);
        worldpartition::ActorRecord record;
        record.uuid = gen.Generate();
        record.className = "ProceduralTree"; // Deliberately NOT kPcgVolumeClassName.
        record.actorLabel = "Oak_001";
        record.localBounds.boundsMin = { -1.0f, 0.0f, -1.0f };
        record.localBounds.boundsMax = { 1.0f, 4.0f, 1.0f };
        record.RecomputeWorldBounds();
        record.properties.push_back({ "SpeciesId", worldpartition::PropertyValue{ std::string{"quercus_robur"} } });

        worldpartition::PcgVolumeDesc parsed;
        bool ok = worldpartition::TryParsePcgVolumeDesc(record, parsed);
        Check(!ok, "TryParsePcgVolumeDesc must reject a record whose className is not kPcgVolumeClassName");
    }

    void TestTryParseRejectsIncompletePcgVolumeRecord() {
        // A record that claims to be a PcgVolume (right className) but is missing the properties
        // BuildPcgVolumeActorRecord always writes -- e.g. hand-authored, or from a truncated/older
        // tool version. Must fail closed (return false), never crash or fabricate defaults.
        worldpartition::ActorRecord record;
        record.uuid = worldpartition::UuidGenerator(7u).Generate();
        record.className = worldpartition::kPcgVolumeClassName;
        record.localBounds.boundsMin = { 0.0f, 0.0f, 0.0f };
        record.localBounds.boundsMax = { 10.0f, 10.0f, 10.0f };
        record.RecomputeWorldBounds();
        // Deliberately no properties at all -- neither the graph path nor the seed.

        worldpartition::PcgVolumeDesc parsed;
        bool ok = worldpartition::TryParsePcgVolumeDesc(record, parsed);
        Check(!ok, "TryParsePcgVolumeDesc must reject a PcgVolume-classed record missing its required properties");
    }

    void TestComputeOverlappingCellsSingleCell() {
        constexpr float kCellSize = 20.0f; // Matches BakeDemoWorld.cpp's kDemoWorldCellSize convention.
        worldpartition::AABB bounds;
        bounds.boundsMin = { 2.0f, 0.0f, 3.0f };
        bounds.boundsMax = { 15.0f, 5.0f, 18.0f }; // Fully inside cell (0,0).

        std::vector<worldpartition::CellCoord> cells = worldpartition::ComputeOverlappingCells(bounds, kCellSize);
        Check(cells.size() == 1, "a volume fully inside one cell should overlap exactly 1 cell, got " + std::to_string(cells.size()));
        if (!cells.empty()) {
            Check(cells[0] == worldpartition::CellCoord{ 0, 0, 0 }, "a volume inside cell (0,0) should report CellCoord{0,0,0}");
        }
    }

    void TestComputeOverlappingCellsMultipleCells() {
        constexpr float kCellSize = 20.0f;
        worldpartition::AABB bounds;
        bounds.boundsMin = { -5.0f, 0.0f, -5.0f };
        bounds.boundsMax = { 25.0f, 10.0f, 25.0f }; // Spans cell x/z in {-1, 0, 1} at cellSize 20 -> 3x3 = 9 cells.

        std::vector<worldpartition::CellCoord> cells = worldpartition::ComputeOverlappingCells(bounds, kCellSize);
        Check(cells.size() == 9, "a volume spanning a 3x3 cell range should overlap exactly 9 cells, got " + std::to_string(cells.size()));

        for (int32_t cx : { -1, 0, 1 }) {
            for (int32_t cz : { -1, 0, 1 }) {
                worldpartition::CellCoord expected{ cx, 0, cz };
                bool found = std::find(cells.begin(), cells.end(), expected) != cells.end();
                Check(found, "multi-cell overlap missing expected CellCoord{" + std::to_string(cx) + ",0," + std::to_string(cz) + "}");
            }
        }

        for (const worldpartition::CellCoord& c : cells) {
            Check(c.y == 0, "ComputeOverlappingCells must always report y == 0 (Grid2D convention -- height never affects cell membership)");
        }
    }

    void TestComputeOverlappingCellsDegenerateBounds() {
        constexpr float kCellSize = 20.0f;
        worldpartition::AABB pointBounds;
        pointBounds.boundsMin = { 45.0f, 3.0f, -25.0f };
        pointBounds.boundsMax = { 45.0f, 3.0f, -25.0f }; // Zero-size: a single point.

        std::vector<worldpartition::CellCoord> cells = worldpartition::ComputeOverlappingCells(pointBounds, kCellSize);
        Check(cells.size() == 1, "a zero-size (point) volume should still overlap exactly 1 cell, got " + std::to_string(cells.size()));
        if (!cells.empty()) {
            // floor(45/20) = 2, floor(-25/20) = -2.
            Check(cells[0] == worldpartition::CellCoord{ 2, 0, -2 }, "point volume at (45,*,-25) should land in CellCoord{2,0,-2}");
        }
    }

}

int main() {
    std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "PcgVolumeActorTests";
    std::error_code ec;
    std::filesystem::create_directories(scratchDir, ec);

    TestPcgVolumeRoundTrip(scratchDir);
    TestTryParseRejectsNonPcgVolumeRecord();
    TestTryParseRejectsIncompletePcgVolumeRecord();
    TestComputeOverlappingCellsSingleCell();
    TestComputeOverlappingCellsMultipleCells();
    TestComputeOverlappingCellsDegenerateBounds();

    if (g_failCount == 0) {
        std::cout << "[PASS] All PCG Volume actor checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
