#pragma once
// Offline export of a built SpatialHashGrid + its actors' authored placements into a small, flat,
// POD binary format the SHIPPING executable can read without linking any of tools/WorldPartition/
// (see WorldPartitionTypes.h's own header comment for why tools/ never links into the .exe). This
// is the bridge between the offline OFPA/SceneIndex/SpatialHashGrid authoring chain and the
// runtime world::CellManifest reader (src/world/CellManifest.h) -- the two sides agree only on
// this file's plain byte layout, never on any worldpartition:: C++ type.
//
// One record per occupied cell (not per actor): this demo's streaming pool is a small bounded set
// of GPU entity slots (see VulkanContext::kStreamingUnitCount), so rather than pretend to stream
// every individual actor in a busy cell, BakeDemoWorld.cpp deliberately authors exactly one actor
// per cell and this format mirrors that -- one representative placement per cell is what the
// runtime loader actually instantiates. A denser multi-actor-per-cell scene is future scope, not
// silently approximated here: see RuntimeCellManifest.cpp's own comment.
//
// Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2) -- v1 -> v2: each record now also references a
// real, offline-baked HLOD proxy mesh (HlodPipeline::BuildHlodForCell's output), appended as a new
// blob section AFTER the fixed [RuntimeCellManifestRecord * recordCount] table in this SAME file:
//   [RuntimeCellManifestHeader]
//   [RuntimeCellManifestRecord * header.recordCount]
//   [RuntimeCellManifestHlodVertex * header.hlodVertexBlobCount]
//   [uint32_t (index) * header.hlodIndexBlobCount]
// Mirrors this codebase's own established .cache version-bump convention (see
// geometry::CacheFileManager / geometry/ClusterFormat.h's own version-field comments): the magic +
// version pair is checked FIRST, before any record is trusted, and a mismatch (including an old v1
// file, which this v2 reader can no longer parse) is treated as "no manifest" -- world::CellManifest::
// Load() returns false, world::StreamingManager streaming is gracefully disabled for the whole
// session, exactly the same degraded-but-not-crashing path a missing file already takes. No
// migration code is needed or written: this is an additive, whole-file version bump, not an
// in-place record patch.
//
// The blob deliberately stores POSITIONS + UVs only, never normals -- geometry::
// ComputeFaceAccumulatedNormals (src/geometry/MeshSimplifier.h) recomputes them from the blob's own
// triangle winding at load time, exactly matching that function's own "correct and complete for any
// SimplifiableMesh, at any DAG level or merge stage" contract, so this format never has to store or
// round-trip a redundant, could-go-stale copy.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

#include "SpatialHashGrid.h"
#include "geometry/MeshSimplifier.h"

namespace worldpartition {

    inline constexpr uint32_t kRuntimeCellManifestMagic = 0x4D434C57u; // 'WLCM' little-endian.
    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2): v1 -> v2 bump -- see this file's own
    // header comment for the exact additive record/blob layout this version now includes.
    // Intentionally duplicated (not shared) with src/world/CellManifest.cpp's own copy of this same
    // constant, matching StreamingTypes.h's documented src//tools/ type-duplication convention --
    // keep both edited together whenever this value changes again.
    inline constexpr uint32_t kRuntimeCellManifestVersion = 2u;

    struct RuntimeCellManifestHeader {
        uint32_t magic = kRuntimeCellManifestMagic;
        uint32_t version = kRuntimeCellManifestVersion;
        float cellSize = 0.0f;   // Must match the world::StreamingManager cellSize the runtime constructs with.
        uint32_t recordCount = 0;
        // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2) v2 additions: sizes of the 2 blob
        // sections appended after the fixed record table -- see this file's own header comment for
        // the exact layout.
        uint32_t hlodVertexBlobCount = 0;
        uint32_t hlodIndexBlobCount = 0;
    };

    // One HLOD proxy vertex in the shared blob -- position + UV only, see this file's own header
    // comment for why normals are deliberately excluded.
    struct RuntimeCellManifestHlodVertex {
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float u = 0.0f, v = 0.0f;
    };

    // One representative placement for one occupied ground-plane cell. Deliberately plain int32/
    // float fields only -- no worldpartition::Uuid, no worldpartition::CellCoord -- so the runtime
    // reader (src/world/CellManifest.h) never needs any tools/WorldPartition/ type or header.
    struct RuntimeCellManifestRecord {
        int32_t cellX = 0;
        int32_t cellZ = 0;
        uint32_t archetypeShape = 0;  // Index into the runtime's small fixed archetype shape table (see WorldCellStreamingLoader.h).
        float localOffsetX = 0.0f;    // World-space placement, already resolved (cell center + authored jitter) -- the runtime applies this verbatim as the entity's translation.
        float localOffsetY = 0.0f;
        float localOffsetZ = 0.0f;
        // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2) v2 additions: this cell's baked HLOD
        // proxy mesh range within the file's shared blob sections (see this file's own header
        // comment). `hlodIndexOffset..+hlodIndexCount` indices are LOCAL to this record's own vertex
        // range (0-based, i.e. the runtime must add `hlodVertexOffset` to rebase each one into the
        // shared blob array, or into whatever destination buffer it copies this record's own
        // sub-range into) -- kept local rather than pre-offset into the global blob so a reader
        // copying just ONE record's sub-range (the runtime's actual use case: one dedicated
        // streaming slot per cell) never needs to know any other record's vertex count.
        // `hlodVertexCount == 0` means this cell has no baked HLOD proxy (e.g. BuildHlodForCell
        // produced an empty mesh) -- the runtime must treat that exactly like "no authored content"
        // for the coarse slot, never a hard failure.
        uint32_t hlodVertexOffset = 0;
        uint32_t hlodVertexCount = 0;
        uint32_t hlodIndexOffset = 0;
        uint32_t hlodIndexCount = 0;
    };

    // Derives one RuntimeCellManifestRecord per occupied cell in `grid`: takes the first actor
    // UUID in each SpatialHashCell (by construction there is ever exactly one, see BakeDemoWorld.cpp),
    // looks it up via `fetchActor`, and maps its className to an archetype shape index via
    // `classNameToShape`. Cells whose actor fails to resolve are skipped (never aborts the whole
    // export), matching RebuildSceneIndexFromActorFiles' own "one bad record must never block
    // every other cell" convention.
    using ActorClassNameFetchFn = std::function<bool(const Uuid&, std::string& outClassName, ActorTransform& outTransform)>;
    using ClassNameToArchetypeShapeFn = std::function<uint32_t(const std::string&)>;

    // Phase 5 (Streaming & Monde roadmap, Part 2, Gap 2): builds this cell's HLOD proxy mesh
    // (typically a thin wrapper around HlodPipeline::BuildHlodForCell -- see BakeDemoWorld.cpp's own
    // call site). Returns false (skip -- never abort the whole export, same convention as
    // ActorClassNameFetchFn above) if this cell has no HLOD proxy to contribute.
    using HlodProxyFetchFn = std::function<bool(const SpatialHashCell& cell, geometry::SimplifiableMesh& outProxyMesh)>;

    // `outHlodVertexBlob`/`outHlodIndexBlob` are appended to (not cleared first) so a caller building
    // more than one manifest export in the same process can share them -- BakeDemoWorld.cpp itself
    // only ever calls this once, starting from empty vectors.
    std::vector<RuntimeCellManifestRecord> BuildRuntimeCellManifest(
        const SpatialHashGrid& grid,
        const ActorClassNameFetchFn& fetchActor,
        const ClassNameToArchetypeShapeFn& classNameToShape,
        const HlodProxyFetchFn& fetchHlodProxy,
        std::vector<RuntimeCellManifestHlodVertex>& outHlodVertexBlob,
        std::vector<uint32_t>& outHlodIndexBlob);

    // Writes a flat [RuntimeCellManifestHeader][RuntimeCellManifestRecord * recordCount]
    // [RuntimeCellManifestHlodVertex * hlodVertexBlob.size()][uint32_t * hlodIndexBlob.size()] file,
    // overwriting any existing file. Returns false on any I/O failure.
    bool WriteRuntimeCellManifest(const std::filesystem::path& filePath, float cellSize,
                                   const std::vector<RuntimeCellManifestRecord>& records,
                                   const std::vector<RuntimeCellManifestHlodVertex>& hlodVertexBlob,
                                   const std::vector<uint32_t>& hlodIndexBlob);

}
