#pragma once
// Shared GPU resource owner for every Surface Cache trace consumer (renderer::SurfaceCacheSWRTPass,
// renderer::SurfaceCacheRayTracingPass, renderer::SurfaceCacheGIInjectPass): builds, once, the two
// descriptor sets the shared shader includes mesh_sdf_trace.glsl (set 1) and
// surface_cache_sampling.glsl (set 2) both expect, from data two ALREADY-Init'd passes own --
// renderer::GlobalSDFPass's per-entity Mesh SDF images (GetTracedEntityInfos()) and
// renderer::SurfaceCachePass's card table + combined Fallback Mesh buffers (GetCards() /
// GetEntityRanges() / GetVertexBuffer() / GetIndexBuffer()) -- so none of the 3 consumers above
// re-decode or re-upload the same data a second time, and so a traced entity's dense array index
// here is THE single index every consumer agrees on (used as renderer::SurfaceCacheRayTracingPass's
// TLAS instanceCustomIndex, and as mesh_sdf_trace.glsl's g_EntitySDF array slot).
//
// --- Why a fixed-size sampler array instead of true bindless descriptor indexing ---
// This codebase's existing "bindless" layer (VulkanContext::CreateBindlessDescriptorSetLayout) is
// a single STORAGE_BUFFER binding, not a descriptor-indexed sampler array, and adding
// VK_EXT_descriptor_indexing purely for this feature would be a third RT-adjacent extension on top
// of the two/three this feature already needs (see VulkanContext.cpp's device-extension comment).
// A compile-time-sized array of kMaxTracedEntities sampler3D bindings, indexed by a dynamically
// uniform loop counter (see mesh_sdf_trace.glsl's TraceMeshSDFScene), is valid under core Vulkan
// 1.0 with no extra extension and is more than sufficient at demoscene entity-count scale.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/GpuBuffer.h"
#include "renderer/SurfaceCachePass.h"

namespace renderer {

    class GlobalSDFPass;

    class SurfaceCacheTraceContext {
    public:
        SurfaceCacheTraceContext() = default;

        SurfaceCacheTraceContext(const SurfaceCacheTraceContext&) = delete;
        SurfaceCacheTraceContext& operator=(const SurfaceCacheTraceContext&) = delete;

        // Must match mesh_sdf_trace.glsl's kMaxTracedEntities exactly.
        static constexpr uint32_t kMaxTracedEntities = 64u;
        // Sentinel SDF value the unused (>= entityCount) g_EntitySDF array slots are filled with --
        // matches GlobalSDFPass::kFarValue's role (always farther than kSphereTraceEpsilon, so a
        // ray that reaches an unused slot's dummy volume can never register a false hit).
        static constexpr float kDummyFarDistance = 1.0e4f;

        // One traced entity's identity + Fallback Mesh draw range, index-aligned with the GPU-side
        // EntityInfo array (mesh_sdf_trace.glsl) this class builds -- i.e. TracedEntities()[i]
        // describes exactly the same entity as g_Entities[i] on the GPU. entityID is the original
        // (possibly sparse -- see GlobalSDFPass::GetTracedEntityInfos()'s own comment) meshID;
        // drawRange is straight from renderer::SurfaceCachePass::GetEntityRanges(), letting
        // renderer::SurfaceCacheRayTracingPass build one BLAS per entity directly against
        // SurfaceCachePass's own combined vertex/index buffers with zero extra geometry upload.
        struct TracedEntity {
            uint32_t entityID = 0;
            SurfaceCachePass::EntityDrawRange drawRange{};
        };

        // Reads globalSDF.GetTracedEntityInfos() + surfaceCache.GetCards()/GetEntityRanges(),
        // both of which must already be Init'd, builds the EntityInfo / EntityCardIndices host
        // tables, uploads them plus a direct copy of surfaceCache.GetCards() into 3 small
        // host-visible SSBOs, creates the kMaxTracedEntities-slot dummy-padded sampler3D array +
        // its 1x1x1 "far" fallback volume, and allocates/writes both descriptor sets. Returns
        // false (logged) only if tracedEntities.size() would silently lose entities beyond
        // kMaxTracedEntities (still succeeds, but truncates -- see Init()'s own log).
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const GlobalSDFPass& globalSDF, const SurfaceCachePass& surfaceCache);

        void Shutdown();

        VkDescriptorSetLayout GetMeshSdfTraceSetLayout() const { return m_MeshSdfTraceSetLayout; }
        VkDescriptorSetLayout GetSurfaceCacheSamplingSetLayout() const { return m_SurfaceCacheSamplingSetLayout; }
        VkDescriptorSet GetMeshSdfTraceSet() const { return m_MeshSdfTraceSet; }
        VkDescriptorSet GetSurfaceCacheSamplingSet() const { return m_SurfaceCacheSamplingSet; }

        // The dynamically-uniform entity count every consumer must push as a push constant before
        // calling TraceMeshSDFScene (mesh_sdf_trace.glsl) -- see that function's own comment.
        uint32_t GetEntityCount() const { return m_EntityCount; }
        const std::vector<TracedEntity>& GetTracedEntities() const { return m_TracedEntities; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        uint32_t m_EntityCount = 0;
        std::vector<TracedEntity> m_TracedEntities;

        GpuBuffer m_EntityInfoBuffer;      // mesh_sdf_trace.glsl's EntityInfo[], host-visible.
        GpuBuffer m_CardBuffer;            // surface_cache_sampling.glsl's SurfaceCacheCardEntry[], host-visible.
        GpuBuffer m_EntityCardIndexBuffer; // surface_cache_sampling.glsl's g_EntityCardIndices[], host-visible.

        VkImage m_DummySdfImage = VK_NULL_HANDLE;
        VmaAllocation m_DummySdfAllocation = VK_NULL_HANDLE;
        VkImageView m_DummySdfView = VK_NULL_HANDLE;
        VkSampler m_EntitySdfSampler = VK_NULL_HANDLE;

        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_MeshSdfTraceSetLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_SurfaceCacheSamplingSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_MeshSdfTraceSet = VK_NULL_HANDLE;
        VkDescriptorSet m_SurfaceCacheSamplingSet = VK_NULL_HANDLE;
    };

}
