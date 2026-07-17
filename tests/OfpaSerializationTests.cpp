// Standalone, framework-free unit test for the World Partition OFPA toolset
// (tools/WorldPartition/OfpaActor.h, SceneIndex.h): round-trips an ActorRecord through
// WriteActorFile/ReadActorFile, verifies MakeActorFilePath's 2-hex-char sharding, round-trips a
// SceneIndex, and verifies RebuildSceneIndexFromActorFiles reconstructs an index that matches the
// actor files on disk. Exits 0 if every check passes, non-zero otherwise -- registered with CTest
// (see the top-level CMakeLists.txt) without pulling in any external test framework, matching
// this project's existing tests/*.cpp convention.

#include "WorldPartition/OfpaActor.h"
#include "WorldPartition/SceneIndex.h"
#include "WorldPartition/Uuid.h"
#include "core/maths/Maths.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>
#include <variant>

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

    // maths::vec3 has no operator== (see core/maths/Maths.h), so PropertyValue's std::variant
    // cannot use its own built-in equality -- compare by alternative index, then by value with
    // NearlyEqual for the float-bearing alternatives.
    bool PropertyValuesEqual(const worldpartition::PropertyValue& a, const worldpartition::PropertyValue& b) {
        if (a.index() != b.index()) return false;
        return std::visit([&b](const auto& aValue) {
            using T = std::decay_t<decltype(aValue)>;
            const T& bValue = std::get<T>(b);
            if constexpr (std::is_same_v<T, float>) {
                return NearlyEqual(aValue, bValue);
            } else if constexpr (std::is_same_v<T, maths::vec3>) {
                return NearlyEqual(aValue, bValue);
            } else {
                return aValue == bValue;
            }
            }, a);
    }

    worldpartition::ActorRecord MakeSampleRecord(worldpartition::UuidGenerator& gen) {
        worldpartition::ActorRecord record;
        record.uuid = gen.Generate();
        record.parentUuid = worldpartition::kNilUuid;
        record.className = "ProceduralTree";
        record.actorLabel = "Oak_042";

        record.transform.position = { 100.0f, 0.0f, -50.0f };
        record.transform.rotation = maths::quat{}; // Identity: simplifies the worldBounds sanity check below.
        record.transform.scale = { 1.0f, 1.0f, 1.0f };

        record.localBounds.boundsMin = { -2.0f, 0.0f, -2.0f };
        record.localBounds.boundsMax = { 2.0f, 8.0f, 2.0f };
        record.RecomputeWorldBounds();

        record.streamingFlags = worldpartition::ActorStreamingFlags::SpatiallyLoaded | worldpartition::ActorStreamingFlags::HLODGenerated;

        record.tags = { "Foliage", "Procedural" };

        record.properties.push_back({ "IsEvergreen", worldpartition::PropertyValue{ true } });
        record.properties.push_back({ "SeedIndex", worldpartition::PropertyValue{ int32_t{42} } });
        record.properties.push_back({ "TrunkRadius", worldpartition::PropertyValue{ 0.35f } });
        record.properties.push_back({ "WindResponse", worldpartition::PropertyValue{ maths::vec3{0.1f, 1.0f, 0.1f} } });
        record.properties.push_back({ "SpeciesId", worldpartition::PropertyValue{ std::string{"quercus_robur"} } });

        return record;
    }

    void TestActorRoundTrip(const std::filesystem::path& scratchDir) {
        worldpartition::UuidGenerator gen(1234567u);
        worldpartition::ActorRecord original = MakeSampleRecord(gen);

        // Identity rotation, unit scale: worldBounds must be exactly localBounds translated by position.
        Check(NearlyEqual(original.worldBounds.boundsMin, original.localBounds.boundsMin + original.transform.position),
            "RecomputeWorldBounds: boundsMin mismatch under identity rotation/scale");
        Check(NearlyEqual(original.worldBounds.boundsMax, original.localBounds.boundsMax + original.transform.position),
            "RecomputeWorldBounds: boundsMax mismatch under identity rotation/scale");

        std::filesystem::path actorPath = worldpartition::MakeActorFilePath(scratchDir, original.uuid);
        std::string hex = original.uuid.ToHexString();
        Check(actorPath.filename() == (hex + ".actor"), "MakeActorFilePath: unexpected filename");
        Check(actorPath.parent_path().filename() == hex.substr(0, 2), "MakeActorFilePath: unexpected shard subfolder");

        Check(worldpartition::WriteActorFile(actorPath, original), "WriteActorFile failed");

        worldpartition::ActorRecord loaded;
        Check(worldpartition::ReadActorFile(actorPath, loaded), "ReadActorFile failed");

        Check(loaded.uuid == original.uuid, "round-trip: uuid mismatch");
        Check(loaded.parentUuid == original.parentUuid, "round-trip: parentUuid mismatch");
        Check(loaded.className == original.className, "round-trip: className mismatch");
        Check(loaded.actorLabel == original.actorLabel, "round-trip: actorLabel mismatch");
        Check(NearlyEqual(loaded.transform.position, original.transform.position), "round-trip: transform.position mismatch");
        Check(NearlyEqual(loaded.localBounds.boundsMin, original.localBounds.boundsMin), "round-trip: localBounds.boundsMin mismatch");
        Check(NearlyEqual(loaded.worldBounds.boundsMax, original.worldBounds.boundsMax), "round-trip: worldBounds.boundsMax mismatch");
        Check(loaded.streamingFlags == original.streamingFlags, "round-trip: streamingFlags mismatch");
        Check(loaded.tags == original.tags, "round-trip: tags mismatch");
        Check(loaded.properties.size() == original.properties.size(), "round-trip: property count mismatch");

        if (loaded.properties.size() == original.properties.size()) {
            for (size_t i = 0; i < loaded.properties.size(); ++i) {
                Check(loaded.properties[i].key == original.properties[i].key, "round-trip: property key mismatch at index " + std::to_string(i));
                Check(PropertyValuesEqual(loaded.properties[i].value, original.properties[i].value), "round-trip: property value mismatch at index " + std::to_string(i));
            }
        }
    }

    void TestSceneIndexRoundTrip(const std::filesystem::path& scratchDir) {
        worldpartition::UuidGenerator gen(987654u);

        std::vector<worldpartition::SceneIndexEntry> entries;
        for (int i = 0; i < 5; ++i) {
            worldpartition::SceneIndexEntry entry;
            entry.uuid = gen.Generate();
            entry.bounds.boundsMin = { static_cast<float>(i) * 10.0f, 0.0f, 0.0f };
            entry.bounds.boundsMax = { static_cast<float>(i) * 10.0f + 5.0f, 5.0f, 5.0f };
            entry.streamingFlags = (i % 2 == 0) ? worldpartition::ActorStreamingFlags::SpatiallyLoaded : worldpartition::ActorStreamingFlags::AlwaysLoaded;
            entries.push_back(entry);
        }

        std::filesystem::path indexPath = scratchDir / "SceneIndex.bin";
        Check(worldpartition::WriteSceneIndex(indexPath, entries), "WriteSceneIndex failed");

        std::vector<worldpartition::SceneIndexEntry> loaded;
        Check(worldpartition::ReadSceneIndex(indexPath, loaded), "ReadSceneIndex failed");
        Check(loaded.size() == entries.size(), "SceneIndex round-trip: entry count mismatch");

        if (loaded.size() == entries.size()) {
            for (size_t i = 0; i < loaded.size(); ++i) {
                Check(loaded[i].uuid == entries[i].uuid, "SceneIndex round-trip: uuid mismatch at index " + std::to_string(i));
                Check(NearlyEqual(loaded[i].bounds.boundsMin, entries[i].bounds.boundsMin), "SceneIndex round-trip: boundsMin mismatch at index " + std::to_string(i));
                Check(loaded[i].streamingFlags == entries[i].streamingFlags, "SceneIndex round-trip: streamingFlags mismatch at index " + std::to_string(i));
            }
        }
    }

    void TestRebuildSceneIndexFromActorFiles(const std::filesystem::path& scratchDir) {
        std::filesystem::path actorsRoot = scratchDir / "RebuildActors";
        std::error_code ec;
        std::filesystem::remove_all(actorsRoot, ec);

        worldpartition::UuidGenerator gen(555111u);
        std::vector<worldpartition::ActorRecord> written;
        for (int i = 0; i < 3; ++i) {
            worldpartition::ActorRecord record = MakeSampleRecord(gen);
            record.uuid = gen.Generate();
            record.transform.position = { static_cast<float>(i) * 20.0f, 0.0f, 0.0f };
            record.RecomputeWorldBounds();

            std::filesystem::path path = worldpartition::MakeActorFilePath(actorsRoot, record.uuid);
            Check(worldpartition::WriteActorFile(path, record), "RebuildSceneIndexFromActorFiles setup: WriteActorFile failed");
            written.push_back(record);
        }

        std::vector<worldpartition::SceneIndexEntry> rebuilt = worldpartition::RebuildSceneIndexFromActorFiles(actorsRoot);
        Check(rebuilt.size() == written.size(), "RebuildSceneIndexFromActorFiles: entry count mismatch");

        for (const worldpartition::ActorRecord& record : written) {
            bool found = false;
            for (const worldpartition::SceneIndexEntry& entry : rebuilt) {
                if (entry.uuid == record.uuid) {
                    found = true;
                    Check(NearlyEqual(entry.bounds.boundsMin, record.worldBounds.boundsMin), "RebuildSceneIndexFromActorFiles: bounds mismatch for a known actor");
                    Check(entry.streamingFlags == record.streamingFlags, "RebuildSceneIndexFromActorFiles: streamingFlags mismatch for a known actor");
                    break;
                }
            }
            Check(found, "RebuildSceneIndexFromActorFiles: a written actor is missing from the rebuilt index");
        }
    }

}

int main() {
    std::filesystem::path scratchDir = std::filesystem::temp_directory_path() / "OfpaSerializationTests";
    std::error_code ec;
    std::filesystem::create_directories(scratchDir, ec);

    TestActorRoundTrip(scratchDir);
    TestSceneIndexRoundTrip(scratchDir);
    TestRebuildSceneIndexFromActorFiles(scratchDir);

    if (g_failCount == 0) {
        std::cout << "[PASS] All OFPA serialization checks passed.\n";
        return 0;
    }
    std::cerr << g_failCount << " check(s) failed.\n";
    return 1;
}
