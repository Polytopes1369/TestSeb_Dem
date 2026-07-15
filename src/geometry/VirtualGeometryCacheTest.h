#pragma once
// End-to-end validation test for the virtual geometry .cache format (ClusterFormat.h) and its
// I/O layer (CacheFileManager). Reads back the *real* procedurally-generated Vertex/Index SSBOs
// currently living on the GPU, partitions them per spawned entity into clusters, writes one
// .cache file per entity, then re-reads a page from disk and checks it round-trips exactly.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "core/EntityData.h"

namespace geometry {

    // Must be called after VulkanContext::GenerateGeometry() has completed (i.e. any time after
    // VulkanContext::Init() returns) so the Vertex/Index SSBOs already hold the live scene's
    // procedural geometry.
    //
    // vertexBuffer/indexBuffer must have been created with VK_BUFFER_USAGE_TRANSFER_SRC_BIT (the
    // engine's shared procedural geometry SSBOs already are, for the pre-existing debug
    // readback). entityData/entityCount describe the spawned entities exactly as authored by
    // VulkanContext::BuildEntityData() (meshID, materialID, ...).
    //
    // Returns true iff every entity's cache file was written, has a size that is a strict
    // multiple of 4096 bytes, and an arbitrary page read back from disk decodes to byte-exact
    // the same cluster data that was written.
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

}
