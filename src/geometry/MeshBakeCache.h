#pragma once
// Per-mesh bake cache with content-hash dirty tracking -- DEBUG BUILDS ONLY (whole file compiled
// out in Release, matching CLAUDE.md's build-separation rule and this codebase's established
// whole-file-guard convention for debug-only code, see renderer::SDFRayMarchPass).
//
// --- Why this exists ---
// scene.cache is monolithic: geometry::IsCacheUpToDate() is an all-or-nothing check over global
// counts/config, so changing ONE entity's generated geometry (e.g. a single tree species
// parameter) invalidates the whole file and forces RunVirtualGeometryCacheTest() to re-run
// BuildClusterDAG/BuildFallbackMesh/GenerateEntityCards for EVERY entity in the scene (~11
// minutes cold on the current 37-entity + streaming-pool scene, measured 2026-07-20). The
// expensive work is strictly per-entity and deterministic, so it caches perfectly at per-entity
// granularity: this class stores each entity's complete pre-stitch bake output in its own small
// file, keyed by a content hash of exactly the inputs that bake consumed. On the next cold
// scene.cache rebuild, only entities whose inputs actually changed re-bake; the rest deserialize
// in milliseconds. The cheap global steps (local->global clusterID remap, atlas packing,
// consolidated scene.cache write) intentionally stay outside this cache -- they re-run every
// time, which is also WHY cards are cached with their atlas placement fields untouched/zeroed:
// PackCardsIntoAtlas()'s result depends on every OTHER entity's cards, so a per-entity cache may
// only ever store pre-pack data.
//
// --- Dirty model ("regen / deleted / added", one file per meshID) ---
//   * regen:   mesh_<meshID>.bin exists but its stored contentHash differs from the current
//              inputs' hash -> treated as a miss; the fresh bake Save() overwrites the same file
//              (no orphan accumulation across parameter tweaks).
//   * deleted: PruneStale() removes every mesh_<id>.bin whose id is absent from the current
//              entity list (entity removed from the scene).
//   * added:   a meshID with no file is simply a miss -> baked -> Save()d.
// The content hash (FNV-1a 64) covers, per entity: its full raw triangle geometry as read back
// from the GPU (every referenced renderer::Vertex's bytes -- positions/normals/uvs/materialID
// all included -- plus connectivity as base-relative index triples, deliberately relative to the
// entity's own first vertex so one entity growing does NOT shift-invalidate every entity baked
// after it in the shared SSBO), its skin bytes when skeletally animated, the bake-relevant
// entity parameters (materialID, maskTextureIndex, maxWPOAmplitude, IsSkeletallyAnimated), and
// kGeometryGenerationVersion. Struct-layout drift of any persisted POD invalidates via explicit
// sizeof fields in the file header. Corrupt/truncated files fail TryLoad()'s validation and are
// treated as misses (then overwritten by the fresh Save()).
//
// --- Release builds ---
// Nothing here is compiled, no meshbakes/ directory is ever created or read: Release keeps
// computing everything at runtime, per the project's demoscene "no local data" mandate. The
// consolidated scene.cache itself is unchanged in both configurations.

#ifndef NDEBUG

#include <cstdint>
#include <filesystem>
#include <vector>

#include "geometry/ClusterFormat.h"
#include "io/CacheFileManager.h" // FallbackMeshData

namespace geometry {

    // Bump when this file's on-disk layout changes (fields added/reordered/resized). Mixed into
    // every file header (NOT into the content hash -- a format bump must invalidate even files
    // whose content hash would still match).
    inline constexpr uint32_t kMeshBakeCacheFormatVersion = 1u;

    // One entity's complete baked output, exactly the per-entity intermediate state
    // RunVirtualGeometryCacheTest() produces before its sequential global stitch: every
    // clusterID-bearing field is LOCAL (0-based within this entity; the stitch loop's
    // baseGlobalID remap is re-applied on every load, cached or not). Cards carry zeroed atlas
    // placement (see this file's header comment on PackCardsIntoAtlas).
    struct MeshBakeEntry {
        std::vector<ClusterIndexEntry> indexEntries;
        std::vector<DAGNodeEntry> dagEntries;
        std::vector<ClusterData> clusterData;
        bool hasFallback = false;
        FallbackMeshData fallbackMeshData;
        std::vector<SurfaceCacheCardEntry> cards;
        // Level-0 triangle total, carried purely so a cache-hit load can reproduce the exact
        // fallback-mesh-ratio log line the bake path prints (the full ClusterDAG the bake path
        // derives it from is deliberately NOT persisted -- only its flattened tables are).
        uint64_t leafTriangleCount = 0;
    };

    // Streaming FNV-1a 64-bit hasher for the per-entity content key. A plain struct (not a
    // std::hash specialization) so callers can feed heterogeneous byte ranges incrementally in
    // one deterministic order.
    struct MeshBakeHasher {
        uint64_t state = 14695981039346656037ull; // FNV-1a 64 offset basis.

        void Feed(const void* data, size_t sizeBytes) {
            const unsigned char* bytes = static_cast<const unsigned char*>(data);
            uint64_t h = state;
            for (size_t i = 0; i < sizeBytes; ++i) {
                h ^= static_cast<uint64_t>(bytes[i]);
                h *= 1099511628211ull; // FNV-1a 64 prime.
            }
            state = h;
        }
        void FeedU32(uint32_t v) { Feed(&v, sizeof(v)); }
        void FeedF32(float v) { Feed(&v, sizeof(v)); }
    };

    // Thread-safety: all methods are const and touch only the one mesh_<meshID>.bin file they
    // are given -- RunVirtualGeometryCacheTest()'s worker pool calls TryLoad()/Save() for
    // DISJOINT meshIDs concurrently, which is safe with no locking (distinct files, no shared
    // mutable state). PruneStale() must only run after the pool has been drained (single
    // threaded), which its one call site guarantees.
    class MeshBakeCache {
    public:
        // Creates `directory` (and parents) if missing. All cache files live directly inside it
        // as mesh_<meshID>.bin.
        explicit MeshBakeCache(std::filesystem::path directory);

        MeshBakeCache(const MeshBakeCache&) = delete;
        MeshBakeCache& operator=(const MeshBakeCache&) = delete;

        // Loads meshID's entry if its file exists, every header field validates (magic, format
        // version, kGeometryGenerationVersion, persisted-POD sizeofs, meshID) AND the stored
        // content hash equals `contentHash`. Any mismatch or short read returns false with `out`
        // untouched semantics not guaranteed (caller only uses it on true).
        bool TryLoad(uint32_t meshID, uint64_t contentHash, MeshBakeEntry& out) const;

        // Atomically-enough persists `entry` for meshID: written to a .tmp sibling first, then
        // rename-replaced over the final name, so a crash mid-write can never leave a
        // plausible-looking truncated mesh_<meshID>.bin (a stale-but-valid previous file, or no
        // file, are the only possible outcomes -- both handled by TryLoad()).
        bool Save(uint32_t meshID, uint64_t contentHash, const MeshBakeEntry& entry) const;

        // Deletes every mesh_<id>.bin whose id does not appear in `liveMeshIDs` (entity removed
        // from the scene since the file was written), plus any leftover .tmp files from a
        // crashed Save(). Returns how many files were removed.
        uint32_t PruneStale(const std::vector<uint32_t>& liveMeshIDs) const;

        const std::filesystem::path& Directory() const { return m_Directory; }

    private:
        std::filesystem::path FilePathFor(uint32_t meshID) const;

        std::filesystem::path m_Directory;
    };

}

#endif // NDEBUG
