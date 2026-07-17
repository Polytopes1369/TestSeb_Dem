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
    constexpr uint32_t kGeometryGenerationVersion = 1;

    // Must be called after VulkanContext::GenerateGeometry() has completed (i.e. any time after
    // VulkanContext::Init() returns) so the Vertex/Index SSBOs already hold the live scene's
    // procedural geometry.
    //
    // vertexBuffer/indexBuffer must have been created with VK_BUFFER_USAGE_TRANSFER_SRC_BIT (the
    // engine's shared procedural geometry SSBOs already are, for the pre-existing debug
    // readback). entityData/entityCount describe the spawned entities exactly as authored by
    // VulkanContext::BuildEntityData() (meshID, materialID, ...).
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
