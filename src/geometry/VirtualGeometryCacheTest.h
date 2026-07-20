#pragma once
// End-to-end validation test for the virtual geometry .cache format (ClusterFormat.h) and its
// I/O layer (CacheFileManager). Reads back the *real* procedurally-generated Vertex/Index SSBOs
// currently living on the GPU, builds a full cluster DAG (ClusterDAG.h) per spawned entity,
// consolidates every entity's clusters into ONE .cache file (header + cluster index table + DAG
// table + page-aligned geometry blocks, see ClusterFormat.h), then re-reads the header, both
// tables, and a sample cluster back from disk and checks everything round-trips byte-exact --
// including re-validating the on-disk DAG table's structural/error-monotonicity invariants via
// ValidateClusterDAG after the round trip, not just before writing.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "core/EntityData.h"

namespace geometry {

    // Bump this by hand whenever VulkanContext::GenerateGeometry()'s primitive generation code
    // changes in a way that changes the produced geometry WITHOUT changing any of the runtime
    // config values IsCacheUpToDate() already checks (VERTEX_SPACING, vertex/index/entity counts,
    // GPU profile) -- e.g. editing a hardcoded primitive parameter (a GenerateBox() dimension, a
    // sphere's segment count, ...), reordering/adding/removing a GenerateX() call, or changing
    // GeometryEncoding.h's quantization. None of the config-value checks below can see a plain
    // source-code edit, since procedural generation has no imported source asset to content-hash
    // (unlike UE 5.8's DDC, which hashes imported source data -- see this project's own DDC-parity
    // investigation) -- an explicit, manually-maintained generator-version key is the standard
    // substitute UE 5.8 itself uses for its own non-asset-backed procedural generators. Forgetting
    // to bump this after such an edit just means a developer manually deletes scene.cache once
    // (same recovery as today, before this field existed) -- this is a convenience/correctness
    // improvement, not a safety-critical guarantee.
    // Bumped 1 -> 2: added renderer::ProceduralTreePass's TREES block to GenerateGeometry() (new
    // GenerateX()-equivalent generation step, plus the pre-existing entity-count check below also
    // independently invalidates any pre-tree-feature scene.cache since kEntityCount itself grew).
    // Bumped 2 -> 3: 10-tree-species scene -- the TREES block's per-tree shape parameters and
    // placement all changed (kTreeSpecies recipe table), which the entity-count check would ALSO
    // catch (kEntityCount 25 -> 37), but the explicit bump keeps this key honest about the
    // generated geometry actually differing.
    // Bumped 3 -> 4: fixed a cross-invocation race in TerrainHydrology.comp's kModeErodeDeposit
    // (see that shader mode's own comment) that made the eroded terrain/water height fields
    // non-deterministic run-to-run -- entity/vertex/index counts are UNCHANGED by this fix (same
    // mesh topology every time now, just numerically different from any pre-fix bake), so only
    // this explicit bump invalidates a stale pre-fix scene.cache; the count-based checks below
    // would otherwise wrongly call it up to date.
    // Bumped 4 -> 5: relocated the tree grove (kTreeSpecies's own header comment, VulkanContext.cpp)
    // from east of the gallery to west -- the old placement was never actually inside the default
    // camera's frustum. Only worldOffsetX/Z changed (same species shapes, same entity/vertex/index
    // counts), which the count-based checks below cannot see; the per-mesh bake cache's own
    // content hash WOULD independently catch this (baked vertex positions differ), but only once
    // IsCacheUpToDate() actually lets RunVirtualGeometryCacheTest() run at all.
    // Bumped 5 -> 6: geom_tree_bark.comp/geom_tree_leaves.comp now sample the real terrain-
    // hydrology mesh-height bake for a tree's vertical placement instead of trusting a flat
    // CPU-supplied constant (renderer::ProceduralTreePass::Init()'s own header comment) -- fixes
    // the west grove (relocated in the previous bump) turning out to sit in a low-lying spot the
    // hydrology sim's initial flood pass left underwater, submerging every tree completely
    // invisible despite being squarely inside the camera's frustum. Same entity/vertex/index
    // counts as before (only the baked Y values change), so only this explicit bump catches it.
    // Bumped 6 -> 7: TerrainHydrology.comp's kModeErodeDeposit now force-drains standing water
    // (both the ambient rain/evaporation equilibrium AND kModeInit's initial below-sea-level
    // flood) toward 0 anywhere inside the gallery's erosion-protected radius that ISN'T part of
    // the authored river/lake channel (RiverChannelMask gate) -- ErosionStrengthAt already kept
    // that radius from being carved, but never stopped it from flooding, which is what was
    // actually submerging the west grove (the sampled-height fix in the previous bump was
    // necessary but not sufficient: it made trees sample the real terrain, and that terrain was
    // sitting under ~0.4 units of water). Same entity/vertex/index counts, only baked water/mesh-
    // height values change, so only this explicit bump catches it.
    constexpr uint32_t kGeometryGenerationVersion = 7;

    // Must be called after VulkanContext::GenerateGeometry() has completed (i.e. any time after
    // VulkanContext::Init() returns) so the Vertex/Index SSBOs already hold the live scene's
    // procedural geometry.
    //
    // vertexBuffer/indexBuffer must have been created with VK_BUFFER_USAGE_TRANSFER_SRC_BIT (the
    // engine's shared procedural geometry SSBOs already are, for the pre-existing debug
    // readback). entityData/entityCount describe the spawned entities exactly as authored by
    // VulkanContext::BuildEntityData() (meshID, materialID, ...).
    //
    // vertexSkinBuffer (skeletal-animation feature) is VulkanContext::GetVertexSkinBuffer() -- also
    // VK_BUFFER_USAGE_TRANSFER_SRC_BIT, index-aligned 1:1 with vertexBuffer at the same
    // totalVertexCount scale. Read back and threaded into BuildClusterDAG for every
    // core::EntityFlags::IsSkeletallyAnimated entity; ignored (never read) for every other entity.
    //
    // Returns true iff every entity produced a structurally valid DAG, every cluster encoded
    // within the fixed-size on-disk format's capacity, the consolidated .cache file was written,
    // and reading the header/tables/a sample cluster back from disk reproduces byte-exact the
    // same data that was written (with the on-disk DAG table re-validated after the round trip).
    bool RunVirtualGeometryCacheTest(
        VkDevice device,
        VmaAllocator allocator,
        VkQueue graphicsQueue,
        VkCommandPool commandPool,
        VkBuffer vertexBuffer,
        VkBuffer indexBuffer,
        VkBuffer vertexSkinBuffer,
        uint32_t totalVertexCount,
        uint32_t totalIndexCount,
        const core::EntityData* entityData,
        uint32_t entityCount);

    // Checks if the cached geometry file (scene.cache) exists and is up to date with current
    // configuration settings, vertex/index counts, entity count, and kGeometryGenerationVersion
    // (see that constant's own comment).
    bool IsCacheUpToDate(
        uint32_t totalVertexCount,
        uint32_t totalIndexCount,
        uint32_t entityCount);

    // Persists configuration settings, vertex/index counts, entity count, and
    // kGeometryGenerationVersion to scene.cache.cfg alongside the compiled scene.cache.
    void SaveCacheConfig(
        uint32_t totalVertexCount,
        uint32_t totalIndexCount,
        uint32_t entityCount);

}
