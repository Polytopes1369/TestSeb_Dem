#pragma once
// Architecture for the offline Hierarchical LOD (HLOD) generation pipeline: groups a World
// Partition cell's actors into one coarse proxy mesh + texture atlas per HLOD level, the same
// role UE5.8's HLOD Layer / HLOD Builder pipeline plays. Three concerns are kept strictly
// separate, mirroring how a real production pipeline is staged and cached independently:
//   1. Cell -> mesh gathering (GatherCellMeshes)            -- which actors, resolved into raw geometry.
//   2. Merge + simplify (MergeCellMeshes, ISimplification*) -- combine + reduce to a triangle budget.
//   3. Texture atlas baking (ITextureAtlasBaker)             -- unique material -> single flat texture.
//
// This module never touches the runtime virtual-geometry .cache format (geometry/ClusterFormat.h)
// directly: mesh gathering is dependency-inverted through a caller-supplied fetch callback (see
// ActorMeshFetchFn) so this offline tool has zero compile-time dependency on the streaming
// runtime, and can equally be pointed at a completely different mesh source (e.g. raw imported
// FBX/OBJ actor meshes) without modification.

#include <cstdint>
#include <functional>
#include <vector>

#include "SpatialHashGrid.h"
#include "Uuid.h"
#include "WorldPartitionTypes.h"
#include "geometry/MeshSimplifier.h"

namespace worldpartition {

    // One HLOD tier: a coarser cell size (grouping more of the original grid's cells together)
    // paired with a per-proxy-mesh triangle budget. Built by BuildHlodLevelChain.
    struct HlodLevel {
        uint32_t levelIndex = 0;     // 0 = finest HLOD tier (nearest camera distance before falling back to full detail); increasing = coarser.
        float cellSize = 0.0f;       // World-space cell size at this level.
        uint32_t triangleBudget = 0; // Target triangle count for ONE proxy mesh at this level, regardless of how many source actors/cells feed it.
    };

    // Builds a `numLevels`-tier chain: level 0 uses baseCellSize/baseTriangleBudget verbatim,
    // each subsequent level doubles cellSize (so level N groups 4^N of level-0's Grid2D cells
    // together, matching a standard image-mip-style progression) while keeping triangleBudget
    // constant -- a coarser proxy mesh represents proportionally MORE world area per triangle,
    // which is exactly the intended, increasing simplification ratio of an HLOD chain (UE5.8's
    // own HLOD Layers behave the same way: budget-per-proxy stays roughly flat, cell footprint
    // grows).
    std::vector<HlodLevel> BuildHlodLevelChain(float baseCellSize, uint32_t baseTriangleBudget, uint32_t numLevels);

    // ---------------------------------------------------------------------------------------
    // Stage 1: mesh gathering
    // ---------------------------------------------------------------------------------------

    // Resolves one actor's raw geometry into a locally-indexed SimplifiableMesh (the same mesh
    // representation geometry::SimplifyMeshQEM already operates on -- see MeshSimplifier.h).
    // Returns false if the actor has no mesh to contribute (e.g. a light or trigger volume),
    // which GatherCellMeshes treats as "skip this actor", not an error.
    using ActorMeshFetchFn = std::function<bool(const Uuid& actorUuid, geometry::SimplifiableMesh& outMesh)>;

    // Fetches every actor referenced by `cell` via `fetchMesh` and returns their meshes
    // side-by-side (NOT yet merged into one buffer -- see MergeCellMeshes for that). Actors the
    // fetch callback rejects are silently skipped.
    std::vector<geometry::SimplifiableMesh> GatherCellMeshes(const SpatialHashCell& cell, const ActorMeshFetchFn& fetchMesh);

    // ---------------------------------------------------------------------------------------
    // Stage 2: merge + simplify
    // ---------------------------------------------------------------------------------------

    // Concatenates every mesh in `sourceMeshes` into one SimplifiableMesh: vertex indices are
    // remapped by each source mesh's running vertex-offset (no vertex welding across source-mesh
    // boundaries -- welding coincident seams belongs to the simplifier's own topology handling,
    // not this merge step, matching how ClusterGrouping.h keeps merge and simplification as
    // separate passes upstream of this same SimplifyMeshQEM call). No vertex in the result is
    // pre-locked; a caller that needs boundary vertices preserved (e.g. to stay crack-free
    // against a neighboring, independently-built cell) must lock them explicitly before invoking
    // a simplification backend.
    geometry::SimplifiableMesh MergeCellMeshes(const std::vector<geometry::SimplifiableMesh>& sourceMeshes);

    // Pluggable simplification backend, so the pipeline's orchestration code (BuildHlodForCell
    // below) never hard-codes which simplifier runs. Two implementations are provided:
    //   - NativeQEMSimplificationBackend: wraps this codebase's own geometry::SimplifyMeshQEM
    //     (already proven correct/crack-free for the Nanite cluster DAG, see ClusterGrouping.h) --
    //     zero new external dependency, used by default and honors mesh.locked exactly.
    //   - MeshOptimizerSimplificationBackend: wraps zeux/meshoptimizer's meshopt_simplify, built
    //     only when WORLDPARTITION_WITH_MESHOPTIMIZER is defined (see this header's #ifdef block
    //     and CMakeLists.txt's WorldPartitionTools/WITH_MESHOPTIMIZER option) -- an offline HLOD
    //     bake is infrequent and non-realtime, so trading a vendored dependency for
    //     meshoptimizer's faster simplifier is a reasonable per-project opt-in that leaves every
    //     other part of this pipeline unchanged.
    class ISimplificationBackend {
    public:
        virtual ~ISimplificationBackend() = default;

        // Simplifies `mesh` in place down to (at most) targetTriangleCount triangles. Returns the
        // actual resulting triangle count (may be higher than the target if the mesh's locked
        // topology prevents reaching it, see geometry::SimplifyMeshQEM's own contract).
        virtual uint32_t Simplify(geometry::SimplifiableMesh& mesh, uint32_t targetTriangleCount) = 0;
    };

    class NativeQEMSimplificationBackend final : public ISimplificationBackend {
    public:
        uint32_t Simplify(geometry::SimplifiableMesh& mesh, uint32_t targetTriangleCount) override;
    };

#ifdef WORLDPARTITION_WITH_MESHOPTIMIZER
    // Only compiled when vendor/meshoptimizer is present and WORLDPARTITION_WITH_MESHOPTIMIZER is
    // defined by CMakeLists.txt -- keeps the default build dependency-free, matching this
    // project's "no heavy frameworks unless opted in" rule.
    //
    // Known limitation, documented rather than papered over: meshopt_simplify's own API only
    // supports AUTO-DETECTED open-border locking (the meshopt_SimplifyLockBorder option flag),
    // not arbitrary caller-specified per-vertex locks -- so this backend does NOT honor
    // mesh.locked at all (unlike NativeQEMSimplificationBackend, which enforces it exactly). A
    // caller with hard boundary-lock requirements (e.g. cell-seam-critical geometry) must use
    // NativeQEMSimplificationBackend instead.
    class MeshOptimizerSimplificationBackend final : public ISimplificationBackend {
    public:
        // targetErrorRatio bounds meshopt_simplify's allowed deviation, as a fraction of the
        // mesh's bounding sphere radius (meshopt_simplify's own convention) -- 1e-2f (1%) is a
        // reasonable default for a coarse HLOD proxy meant to be viewed from far away.
        explicit MeshOptimizerSimplificationBackend(float targetErrorRatio = 1.0e-2f);

        uint32_t Simplify(geometry::SimplifiableMesh& mesh, uint32_t targetTriangleCount) override;

    private:
        float targetErrorRatio_;
    };
#endif

    // A finished HLOD proxy: the merged + simplified mesh, and its world-space bounds re-derived
    // from the POST-simplification vertex positions (deliberately not SpatialHashCell::actorBounds
    // -- simplification can only shrink, never grow, the visible extent, and downstream culling
    // must use the tighter, accurate box).
    struct HlodProxyMesh {
        geometry::SimplifiableMesh mesh;
        AABB bounds;
    };

    // Gathers, merges and simplifies one HLOD cell down to `level.triangleBudget` triangles using
    // `backend`.
    HlodProxyMesh BuildHlodForCell(
        const SpatialHashCell& cell, const ActorMeshFetchFn& fetchMesh,
        const HlodLevel& level, ISimplificationBackend& backend);

    // ---------------------------------------------------------------------------------------
    // Stage 3: texture atlas baking
    // ---------------------------------------------------------------------------------------

    // One source material's flat placement inside the shared HLOD atlas -- deliberately the same
    // shape as geometry::SurfaceCacheCardEntry's atlas-placement fields (see CardGenerator.h),
    // since both solve the same "many rects -> one packed atlas" problem; kept as its own type
    // here rather than reusing SurfaceCacheCardEntry, to avoid coupling this offline tool to the
    // Lumen surface-cache runtime format.
    struct HlodAtlasTile {
        uint32_t sourceMaterialId = 0;
        uint32_t atlasOffsetX = 0;
        uint32_t atlasOffsetY = 0;
        uint32_t width = 0;
        uint32_t height = 0;
    };

    class ITextureAtlasBaker {
    public:
        virtual ~ITextureAtlasBaker() = default;

        // Packs one requestedTileSize-square tile per entry in `materialIds` into a single
        // `atlasSize`-square atlas. Returns false, leaving `outTiles` unspecified, if the tiles
        // do not fit at that size (a mistuned-budget condition the caller must resolve by
        // lowering requestedTileSize, raising atlasSize, or baking multiple atlases -- never
        // silently degraded here).
        //
        // Actually baking each tile's pixel content (rasterizing/downsampling the source
        // materials' textures into the atlas image) is intentionally NOT this interface's job:
        // that step needs a GPU raster pass (render each source material's UV chart into its
        // assigned tile), which belongs in the Vulkan renderer, not this offline CPU tool -- this
        // interface only owns the placement decision the renderer then executes against, exactly
        // the same split GenerateEntityCards/PackCardsIntoAtlas (CardGenerator.h) already uses
        // between CPU-side placement and the GPU-side SurfaceCachePass capture.
        virtual bool PackMaterialsIntoAtlas(
            const std::vector<uint32_t>& materialIds, uint32_t requestedTileSize,
            uint32_t atlasSize, std::vector<HlodAtlasTile>& outTiles) = 0;
    };

    // Uniform-size grid packer: every tile is exactly requestedTileSize square, so placement
    // reduces to filling atlasSize/requestedTileSize columns per row, row-major -- simpler than
    // CardGenerator.h's variable-height shelf packer (which exists because surface cache cards
    // vary in size per entity AABB; HLOD atlas tiles here are deliberately kept uniform-size to
    // avoid needing that complexity for what is already a coarse, far-viewed proxy).
    class ShelfPackAtlasBaker final : public ITextureAtlasBaker {
    public:
        bool PackMaterialsIntoAtlas(
            const std::vector<uint32_t>& materialIds, uint32_t requestedTileSize,
            uint32_t atlasSize, std::vector<HlodAtlasTile>& outTiles) override;
    };

}
