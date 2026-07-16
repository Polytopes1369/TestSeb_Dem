#pragma once
// CPU-side "Card" generation for the Lumen-style Surface Cache: analyses each virtual geometry
// entity (its Fallback Mesh AABB -- the same coarse proxy the ray tracing BVH and the surface
// cache capture pass consume) and produces at most geometry::kMaxCardsPerEntity orthographic
// box-face projections. Each card owns an exclusive, non-overlapping rect of the global surface
// cache atlas so lighting data (albedo/normal/emissive) captured through one card can never bleed
// into another entity's texels. The resulting SurfaceCacheCardEntry table is persisted into the
// .cache file (ClusterFormat.h, format v5) by CacheFileManager::WriteCacheFile.
//
// Two-phase API, because atlas placement is a GLOBAL decision across every entity while card
// shapes are a purely per-entity decision:
//   1. GenerateEntityCards(entityID, boundsMin, boundsMax) -- per entity, sizes up to 6 cards
//      from the AABB face extents (degenerate faces skipped, e.g. the side faces of a flat
//      plane). atlasOffset/uvMin/uvMax are left zeroed.
//   2. PackCardsIntoAtlas(allCards) -- once, over the concatenated card list of every entity:
//      shelf-packs every card into the atlas and stamps atlasOffset/uvMin/uvMax in place.

#include <cstdint>
#include <vector>
#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"

namespace geometry {

    // Global surface cache atlas dimensions, in texels (square). Shared by the capture pass's
    // atlas images (renderer::SurfaceCachePass) and by the UV normalization done at pack time.
    constexpr uint32_t kSurfaceCacheAtlasSize = 2048u;

    // World-space capture density: how many atlas texels one world unit of card footprint gets.
    // With the demo scene's ~1.5 m primitives this yields ~96 texel cards -- comfortably above
    // the minimum below, comfortably below one atlas shelf.
    constexpr float kCardTexelsPerWorldUnit = 64.0f;

    // Per-axis card resolution clamp. The minimum keeps even a sliver-thin footprint sampleable
    // (bilinear needs >= 2 texels; 8 leaves margin for the 1-texel gutter). The maximum stops a
    // single huge entity (terrain slab) from monopolizing the atlas.
    constexpr uint32_t kMinCardResolution = 8u;
    constexpr uint32_t kMaxCardResolution = 256u;

    // A face whose projected footprint is thinner than this (world units) on either projected
    // axis produces no card: its orthographic capture would be a degenerate line of texels.
    constexpr float kDegenerateCardExtent = 1.0e-4f;

    // Padding texels left empty around every packed card so bilinear sampling at a card's border
    // can never fetch a neighboring card's texels.
    constexpr uint32_t kCardGutterTexels = 1u;

    // Phase 1 -- builds up to kMaxCardsPerEntity cards for one entity from its local-space AABB
    // (use the entity's FallbackMeshIndexEntry bounds). Fills entityID/faceDirection/
    // localBoundsMin/localBoundsMax/atlasSize; atlasOffset/uvMin/uvMax stay zeroed until
    // PackCardsIntoAtlas runs. Returns an empty vector for a fully degenerate (point-like) AABB.
    std::vector<SurfaceCacheCardEntry> GenerateEntityCards(
        uint32_t entityID, const maths::vec3& boundsMin, const maths::vec3& boundsMax);

    // Phase 2 -- assigns every card in `cards` a unique, non-overlapping atlas rect (shelf
    // packing: cards sorted by height, placed left-to-right into full-width shelves) and stamps
    // atlasOffset + the normalized uvMin/uvMax in place. Returns false (leaving `cards`
    // partially stamped -- callers must treat that as fatal) if the combined cards do not fit
    // the atlas, which indicates kCardTexelsPerWorldUnit / kSurfaceCacheAtlasSize are mistuned
    // for the scene, not a runtime condition to recover from.
    bool PackCardsIntoAtlas(std::vector<SurfaceCacheCardEntry>& cards,
        uint32_t atlasSize = kSurfaceCacheAtlasSize);

}
