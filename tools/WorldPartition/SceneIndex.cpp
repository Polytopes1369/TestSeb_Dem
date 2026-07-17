#include "SceneIndex.h"
#include "OfpaActor.h"

#include <fstream>

namespace worldpartition {

    bool WriteSceneIndex(const std::filesystem::path& filePath, const std::vector<SceneIndexEntry>& entries) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);

        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) return false;

        SceneIndexHeader header;
        header.entryCount = static_cast<uint32_t>(entries.size());
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        if (!entries.empty()) {
            out.write(reinterpret_cast<const char*>(entries.data()),
                static_cast<std::streamsize>(entries.size() * sizeof(SceneIndexEntry)));
        }

        return out.good();
    }

    bool ReadSceneIndex(const std::filesystem::path& filePath, std::vector<SceneIndexEntry>& outEntries) {
        std::ifstream in(filePath, std::ios::binary);
        if (!in.is_open()) return false;

        SceneIndexHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!in) return false;
        if (header.magic != kSceneIndexMagic || header.version != kSceneIndexVersion) return false;

        outEntries.resize(header.entryCount);
        if (header.entryCount > 0) {
            std::streamsize expectedBytes = static_cast<std::streamsize>(header.entryCount * sizeof(SceneIndexEntry));
            in.read(reinterpret_cast<char*>(outEntries.data()), expectedBytes);
            if (in.gcount() != expectedBytes) return false; // Truncated file.
        }

        return true;
    }

    std::vector<SceneIndexEntry> RebuildSceneIndexFromActorFiles(const std::filesystem::path& actorsRootDir) {
        std::vector<SceneIndexEntry> entries;

        if (!std::filesystem::exists(actorsRootDir)) return entries;

        for (const std::filesystem::directory_entry& dirEntry :
            std::filesystem::recursive_directory_iterator(actorsRootDir)) {

            if (!dirEntry.is_regular_file() || dirEntry.path().extension() != ".actor") continue;

            ActorRecord record;
            if (!ReadActorFile(dirEntry.path(), record)) continue; // Corrupt/unreadable actor file: skip, never abort the whole rebuild.

            SceneIndexEntry entry;
            entry.uuid = record.uuid;
            entry.bounds = record.worldBounds;
            entry.streamingFlags = record.streamingFlags;
            entries.push_back(entry);
        }

        return entries;
    }

}
