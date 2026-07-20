// Per-mesh bake cache implementation -- see MeshBakeCache.h's own header comment for the full
// design (dirty model, hash coverage, Release exclusion). Whole file compiled out in Release:
// CMakeLists.txt GLOBs every src/*.cpp unconditionally, so the guard below is what actually
// keeps this out of the shipping executable (the codebase's established pattern for debug-only
// translation units).
#ifndef NDEBUG

#include "geometry/MeshBakeCache.h"

#include "core/Logger.h"
#include "geometry/VirtualGeometryCacheTest.h" // kGeometryGenerationVersion

#include <charconv>
#include <cstring>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>

namespace geometry {

    namespace {

        constexpr uint32_t kMeshBakeMagic = 0x314B424Du; // 'MBK1', little-endian.

        // Fixed-size file header. Every field is validated by TryLoad() before any payload byte
        // is trusted; the sizeof* fields catch persisted-POD layout drift that no version bump
        // remembered to accompany (same "explicit, manually-maintained key" philosophy as
        // kGeometryGenerationVersion, but automated for the struct-layout case).
        struct MeshBakeFileHeader {
            uint32_t magic = 0;
            uint32_t formatVersion = 0;
            uint32_t geometryGenerationVersion = 0;
            uint32_t meshID = 0;
            uint64_t contentHash = 0;

            uint32_t sizeofClusterIndexEntry = 0;
            uint32_t sizeofDagNodeEntry = 0;
            uint32_t sizeofClusterData = 0;
            uint32_t sizeofFallbackIndexEntry = 0;
            uint32_t sizeofFallbackVertex = 0;
            uint32_t sizeofCardEntry = 0;

            uint32_t clusterCount = 0;
            uint32_t hasFallback = 0;
            uint32_t fallbackVertexCount = 0;
            uint32_t fallbackIndexCount = 0;
            uint32_t cardCount = 0;
            uint32_t _pad0 = 0;
            uint64_t leafTriangleCount = 0;
        };
        static_assert(sizeof(MeshBakeFileHeader) == 80,
            "MeshBakeFileHeader layout drifted -- bump kMeshBakeCacheFormatVersion and update this assert");

        // Upper bound sanity caps: a corrupt header must not be able to drive a multi-gigabyte
        // allocation before the payload read fails. Generous vs. any real entity in this scene
        // (the largest, the terrain, is ~2k clusters).
        constexpr uint32_t kMaxSaneClusterCount = 1u << 20;
        constexpr uint32_t kMaxSaneFallbackVerts = 1u << 24;
        constexpr uint32_t kMaxSaneCardCount = 1u << 16;

        template <typename T>
        bool ReadPodArray(std::ifstream& file, std::vector<T>& out, uint32_t count) {
            out.resize(count);
            if (count == 0u) {
                return true;
            }
            file.read(reinterpret_cast<char*>(out.data()),
                static_cast<std::streamsize>(sizeof(T) * count));
            return file.good();
        }

        template <typename T>
        void WritePodArray(std::ofstream& file, const std::vector<T>& data) {
            if (!data.empty()) {
                file.write(reinterpret_cast<const char*>(data.data()),
                    static_cast<std::streamsize>(sizeof(T) * data.size()));
            }
        }

    } // namespace

    MeshBakeCache::MeshBakeCache(std::filesystem::path directory)
        : m_Directory(std::move(directory)) {
        std::error_code ec;
        std::filesystem::create_directories(m_Directory, ec);
        if (ec) {
            LOG_WARNING(std::format(
                "[MeshBakeCache] Could not create cache directory '{}': {} -- every lookup will "
                "miss and every save will fail (full rebake each run, functionally identical to "
                "Release behavior).",
                m_Directory.string(), ec.message()));
        }
    }

    std::filesystem::path MeshBakeCache::FilePathFor(uint32_t meshID) const {
        return m_Directory / std::format("mesh_{}.bin", meshID);
    }

    bool MeshBakeCache::TryLoad(uint32_t meshID, uint64_t contentHash, MeshBakeEntry& out) const {
        std::filesystem::path path = FilePathFor(meshID);
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return false; // "added" case: no file yet -- plain miss, no log (expected on first run).
        }

        MeshBakeFileHeader header{};
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!file.good()) {
            LOG_WARNING(std::format(
                "[MeshBakeCache] meshID={}: '{}' shorter than its own header -- treating as dirty.",
                meshID, path.string()));
            return false;
        }

        // Full header validation before trusting a single payload byte. A contentHash mismatch
        // is the NORMAL "regen" dirty case (inputs changed), so it logs at info level via the
        // caller's own hit/miss accounting rather than warning here; every other mismatch means
        // the file predates a format/generator/struct change.
        if (header.magic != kMeshBakeMagic ||
            header.formatVersion != kMeshBakeCacheFormatVersion ||
            header.geometryGenerationVersion != kGeometryGenerationVersion ||
            header.meshID != meshID ||
            header.sizeofClusterIndexEntry != sizeof(ClusterIndexEntry) ||
            header.sizeofDagNodeEntry != sizeof(DAGNodeEntry) ||
            header.sizeofClusterData != sizeof(ClusterData) ||
            header.sizeofFallbackIndexEntry != sizeof(FallbackMeshIndexEntry) ||
            header.sizeofFallbackVertex != sizeof(FallbackVertex) ||
            header.sizeofCardEntry != sizeof(SurfaceCacheCardEntry)) {
            return false;
        }
        if (header.contentHash != contentHash) {
            return false; // "regen" case: same mesh, different inputs.
        }
        if (header.clusterCount == 0u || header.clusterCount > kMaxSaneClusterCount ||
            header.fallbackVertexCount > kMaxSaneFallbackVerts ||
            header.fallbackIndexCount > kMaxSaneFallbackVerts * 3u ||
            header.cardCount > kMaxSaneCardCount ||
            header.hasFallback > 1u) {
            LOG_WARNING(std::format(
                "[MeshBakeCache] meshID={}: '{}' has implausible counts (clusters={}, fbVerts={}, "
                "fbIndices={}, cards={}) -- treating as corrupt/dirty.",
                meshID, path.string(), header.clusterCount, header.fallbackVertexCount,
                header.fallbackIndexCount, header.cardCount));
            return false;
        }

        bool ok = ReadPodArray(file, out.indexEntries, header.clusterCount)
            && ReadPodArray(file, out.dagEntries, header.clusterCount)
            && ReadPodArray(file, out.clusterData, header.clusterCount);
        out.hasFallback = header.hasFallback != 0u;
        if (ok && out.hasFallback) {
            file.read(reinterpret_cast<char*>(&out.fallbackMeshData.indexEntry),
                sizeof(FallbackMeshIndexEntry));
            ok = file.good()
                && ReadPodArray(file, out.fallbackMeshData.vertices, header.fallbackVertexCount)
                && ReadPodArray(file, out.fallbackMeshData.indices, header.fallbackIndexCount);
        } else {
            out.fallbackMeshData = FallbackMeshData{};
        }
        ok = ok && ReadPodArray(file, out.cards, header.cardCount);
        if (!ok) {
            LOG_WARNING(std::format(
                "[MeshBakeCache] meshID={}: '{}' payload truncated mid-read -- treating as "
                "corrupt/dirty.", meshID, path.string()));
            return false;
        }
        out.leafTriangleCount = header.leafTriangleCount;
        return true;
    }

    bool MeshBakeCache::Save(uint32_t meshID, uint64_t contentHash, const MeshBakeEntry& entry) const {
        if (entry.clusterData.empty() ||
            entry.clusterData.size() != entry.indexEntries.size() ||
            entry.clusterData.size() != entry.dagEntries.size()) {
            LOG_WARNING(std::format(
                "[MeshBakeCache] meshID={}: refusing to save inconsistent entry (clusters={}, "
                "indexEntries={}, dagEntries={}).",
                meshID, entry.clusterData.size(), entry.indexEntries.size(), entry.dagEntries.size()));
            return false;
        }

        MeshBakeFileHeader header{};
        header.magic = kMeshBakeMagic;
        header.formatVersion = kMeshBakeCacheFormatVersion;
        header.geometryGenerationVersion = kGeometryGenerationVersion;
        header.meshID = meshID;
        header.contentHash = contentHash;
        header.sizeofClusterIndexEntry = sizeof(ClusterIndexEntry);
        header.sizeofDagNodeEntry = sizeof(DAGNodeEntry);
        header.sizeofClusterData = sizeof(ClusterData);
        header.sizeofFallbackIndexEntry = sizeof(FallbackMeshIndexEntry);
        header.sizeofFallbackVertex = sizeof(FallbackVertex);
        header.sizeofCardEntry = sizeof(SurfaceCacheCardEntry);
        header.clusterCount = static_cast<uint32_t>(entry.clusterData.size());
        header.hasFallback = entry.hasFallback ? 1u : 0u;
        header.fallbackVertexCount = entry.hasFallback
            ? static_cast<uint32_t>(entry.fallbackMeshData.vertices.size()) : 0u;
        header.fallbackIndexCount = entry.hasFallback
            ? static_cast<uint32_t>(entry.fallbackMeshData.indices.size()) : 0u;
        header.cardCount = static_cast<uint32_t>(entry.cards.size());
        header.leafTriangleCount = entry.leafTriangleCount;

        // Write to a .tmp sibling then rename over the final name -- see this method's own
        // header-comment contract. std::filesystem::rename replaces an existing destination
        // atomically-enough on the same volume (POSIX rename / Win32 MoveFileEx-replace
        // semantics), so readers can never observe a half-written final file.
        std::filesystem::path finalPath = FilePathFor(meshID);
        std::filesystem::path tmpPath = finalPath;
        tmpPath += ".tmp";
        {
            std::ofstream file(tmpPath, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) {
                LOG_WARNING(std::format(
                    "[MeshBakeCache] meshID={}: cannot open '{}' for writing -- entry not cached.",
                    meshID, tmpPath.string()));
                return false;
            }
            file.write(reinterpret_cast<const char*>(&header), sizeof(header));
            WritePodArray(file, entry.indexEntries);
            WritePodArray(file, entry.dagEntries);
            WritePodArray(file, entry.clusterData);
            if (entry.hasFallback) {
                file.write(reinterpret_cast<const char*>(&entry.fallbackMeshData.indexEntry),
                    sizeof(FallbackMeshIndexEntry));
                WritePodArray(file, entry.fallbackMeshData.vertices);
                WritePodArray(file, entry.fallbackMeshData.indices);
            }
            WritePodArray(file, entry.cards);
            if (!file.good()) {
                LOG_WARNING(std::format(
                    "[MeshBakeCache] meshID={}: write to '{}' failed -- entry not cached.",
                    meshID, tmpPath.string()));
                return false;
            }
        } // ofstream closed (flushed) before the rename below.

        std::error_code ec;
        std::filesystem::rename(tmpPath, finalPath, ec);
        if (ec) {
            LOG_WARNING(std::format(
                "[MeshBakeCache] meshID={}: rename '{}' -> '{}' failed: {} -- entry not cached.",
                meshID, tmpPath.string(), finalPath.string(), ec.message()));
            return false;
        }
        return true;
    }

    uint32_t MeshBakeCache::PruneStale(const std::vector<uint32_t>& liveMeshIDs) const {
        std::unordered_set<uint32_t> live(liveMeshIDs.begin(), liveMeshIDs.end());

        uint32_t removed = 0;
        std::error_code ec;
        // An error-code iterator over an unreadable/missing directory compares equal to end(),
        // so the loop below simply runs zero times in that case -- nothing to prune.
        for (const std::filesystem::directory_entry& dirEntry :
             std::filesystem::directory_iterator(m_Directory, ec)) {
            if (!dirEntry.is_regular_file()) {
                continue;
            }
            std::string name = dirEntry.path().filename().string();

            // Leftover .tmp from a crashed Save(): always stale by definition.
            bool isTmp = name.size() > 4u && name.ends_with(".tmp");

            // mesh_<id>.bin whose id is no longer in the scene: the "deleted" dirty case.
            bool isStaleMesh = false;
            constexpr std::string_view kPrefix = "mesh_";
            constexpr std::string_view kSuffix = ".bin";
            if (!isTmp && name.starts_with(kPrefix) && name.ends_with(kSuffix) &&
                name.size() > kPrefix.size() + kSuffix.size()) {
                std::string_view idText(name.data() + kPrefix.size(),
                    name.size() - kPrefix.size() - kSuffix.size());
                uint32_t id = 0;
                auto [ptr, parseEc] = std::from_chars(idText.data(), idText.data() + idText.size(), id);
                if (parseEc == std::errc{} && ptr == idText.data() + idText.size()) {
                    isStaleMesh = !live.contains(id);
                }
            }

            if (isTmp || isStaleMesh) {
                std::error_code removeEc;
                if (std::filesystem::remove(dirEntry.path(), removeEc) && !removeEc) {
                    LOG_INFO(std::format("[MeshBakeCache] Pruned stale cache file '{}'.", name));
                    ++removed;
                }
            }
        }
        return removed;
    }

}

#endif // NDEBUG
