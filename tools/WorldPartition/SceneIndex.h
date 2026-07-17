#pragma once
// Global scene index: one small binary file summarizing every actor in the level down to
// exactly the three fields streaming/culling decisions need -- Uuid, world-space AABB and
// streaming flags -- deliberately excluding everything else an ActorRecord carries (className,
// transform, properties, ...). This is what SpatialHashGrid.h and the runtime streaming source
// load at startup: a level with 200,000 actors reduces to a ~7 MB flat index instead of opening
// 200,000 individual .actor files (OfpaActor.h) just to find out where they are.
//
// The OFPA actor files remain the single source of truth; this index is a derived, rebuildable
// cache (see RebuildSceneIndexFromActorFiles) -- never hand-edited, never diverging safely from
// the actor files it summarizes.

#include <cstdint>
#include <filesystem>
#include <vector>

#include "WorldPartitionTypes.h"
#include "Uuid.h"

namespace worldpartition {

    inline constexpr uint32_t kSceneIndexMagic = 0x58444953u; // 'SIDX' little-endian.
    inline constexpr uint32_t kSceneIndexVersion = 1u;

    struct SceneIndexHeader {
        uint32_t magic = kSceneIndexMagic;
        uint32_t version = kSceneIndexVersion;
        uint32_t entryCount = 0;
    };

    // Exactly the 3 fields a streaming/culling/HLOD-grouping decision needs -- no path, no
    // className: the corresponding .actor file's path is always re-derivable from `uuid` alone
    // via OfpaActor.h::MakeActorFilePath, so persisting it here would be redundant, mutable state
    // that could drift out of sync with the real file location.
    struct SceneIndexEntry {
        Uuid uuid;
        AABB bounds; // World-space -- copied verbatim from the source ActorRecord::worldBounds.
        ActorStreamingFlags streamingFlags = ActorStreamingFlags::SpatiallyLoaded;
    };

    // Writes a flat [SceneIndexHeader][SceneIndexEntry * entryCount] file, overwriting any
    // existing file. Returns false on any I/O failure.
    bool WriteSceneIndex(const std::filesystem::path& filePath, const std::vector<SceneIndexEntry>& entries);

    // Reads a scene index written by WriteSceneIndex. Returns false on any I/O failure, a
    // magic/version mismatch, or a truncated entry table.
    bool ReadSceneIndex(const std::filesystem::path& filePath, std::vector<SceneIndexEntry>& outEntries);

    // Rebuilds a full scene index by scanning every "*.actor" file under `actorsRootDir`
    // (recursively, to walk the 2-hex-char shard subfolders OfpaActor.h::MakeActorFilePath
    // creates) and extracting just the 3 index-worthy fields from each. Actor files that fail to
    // parse (ReadActorFile returns false -- corruption, or a version this tool predates) are
    // skipped rather than aborting the whole rebuild, since one bad actor file must never block
    // every other actor's streaming. Entries are returned in filesystem iteration order (not
    // sorted), matching WriteSceneIndex/ReadSceneIndex's format, which is order-agnostic.
    std::vector<SceneIndexEntry> RebuildSceneIndexFromActorFiles(const std::filesystem::path& actorsRootDir);

}
