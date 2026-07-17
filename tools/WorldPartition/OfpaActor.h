#pragma once
// "One File Per Actor" (OFPA) serialization: every placed actor in the level lives in its own
// small binary file on disk, named after its Uuid, rather than all actors being packed into one
// monolithic level file. This is what makes large open-world levels tractable in source control
// (a designer moving one tree only ever touches that tree's file, never a multi-gigabyte level
// blob) and lets the streaming/HLOD tooling below (SceneIndex.h, SpatialHashGrid.h,
// HlodPipeline.h) load only the coarse index up front instead of every actor's full data.
//
// File layout (ActorFileHeader immediately followed by the serialized ActorRecord body):
//   [ActorFileHeader]
//   [Uuid uuid][Uuid parentUuid]
//   [uint32 classNameLen][classNameLen bytes (UTF-8, no terminator)]
//   [uint32 actorLabelLen][actorLabelLen bytes]
//   [ActorTransform: vec3 position, quat rotation, vec3 scale -- 10 floats]
//   [AABB localBounds: 6 floats][AABB worldBounds: 6 floats]
//   [uint32 streamingFlags]
//   [uint32 tagCount][per tag: uint32 len + bytes]
//   [uint32 propertyCount][per property: PropertyEntry, see WritePropertyEntry]
//
// Every length-prefixed field uses uint32_t; every float is a raw 4-byte IEEE-754 write (no
// endianness handling -- this project targets Windows x64 only, matching CacheFileManager.h's
// same assumption for the runtime .cache format).

#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "WorldPartitionTypes.h"
#include "Uuid.h"

namespace worldpartition {

    inline constexpr uint32_t kActorFileMagic = 0x4150464Fu; // 'OFPA' read little-endian as bytes 'O','F','P','A'.
    inline constexpr uint32_t kActorFileVersion = 1u;

    struct ActorFileHeader {
        uint32_t magic = kActorFileMagic;
        uint32_t version = kActorFileVersion;
    };

    // A single arbitrary, named property carried on an actor -- the closest thing this codebase
    // has to a reflected UPROPERTY bag, since there is no runtime reflection system to hang
    // per-component data off of. Deliberately a closed std::variant (not an open type-erased
    // blob): every (de)serialization site is an exhaustive std::visit, so adding a 6th supported
    // type is a compile error at every call site that needs updating, not a silent runtime gap.
    using PropertyValue = std::variant<bool, int32_t, float, maths::vec3, std::string>;

    struct PropertyEntry {
        std::string key;
        PropertyValue value;
    };

    struct ActorRecord {
        Uuid uuid;
        Uuid parentUuid = kNilUuid; // kNilUuid: no parent (root-level placement / no attachment or outliner-folder hierarchy).

        std::string className;  // e.g. "ProceduralTree", "TerrainPatch" -- offline/editor metadata, drives which importer/spawner tool interprets `properties`.
        std::string actorLabel; // Human-readable editor label, independent of className and never used for identity (uuid is).

        ActorTransform transform;

        AABB localBounds; // Object-space bounds, authored/imported once and stable across moves.
        AABB worldBounds; // transform applied to localBounds's 8 corners -- kept in sync by RecomputeWorldBounds(), and this exact field is what SceneIndexEntry::bounds is copied from (see SceneIndex.h).

        ActorStreamingFlags streamingFlags = ActorStreamingFlags::SpatiallyLoaded;

        std::vector<std::string> tags;
        std::vector<PropertyEntry> properties;

        // Re-derives worldBounds from localBounds + transform by transforming all 8 local-space
        // corners and re-expanding an AABB around them (the only correct approach once rotation
        // is involved -- transforming just boundsMin/boundsMax would produce a box that does not
        // actually contain the rotated geometry). Callers must invoke this after touching
        // localBounds or transform and before relying on worldBounds (e.g. before feeding this
        // record into RebuildSceneIndexFromActorFiles).
        void RecomputeWorldBounds();
    };

    // Writes `record` to `filePath`, overwriting any existing file. Returns false on any I/O failure.
    bool WriteActorFile(const std::filesystem::path& filePath, const ActorRecord& record);

    // Reads one actor file written by WriteActorFile. Returns false on any I/O failure or a
    // magic/version mismatch, leaving `outRecord` in an unspecified state.
    bool ReadActorFile(const std::filesystem::path& filePath, ActorRecord& outRecord);

    // Derives the canonical OFPA path for `uuid` under `actorsRootDir`:
    // <actorsRootDir>/<first 2 hex chars of uuid>/<full 32 hex chars>.actor
    // The 2-hex-char sharding subfolder (matching UE5.8's own OFPA convention) spreads a
    // level's actors across up to 256 subfolders so no single directory accumulates tens of
    // thousands of entries as the level grows -- both Windows Explorer and most source-control
    // clients degrade badly well before that point.
    std::filesystem::path MakeActorFilePath(const std::filesystem::path& actorsRootDir, const Uuid& uuid);

}
