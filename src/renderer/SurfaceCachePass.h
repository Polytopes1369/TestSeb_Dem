#pragma once
// Surface Cache capture pass (Lumen-style): projects each entity's Fallback Mesh (the same coarse
// BVH proxy geometry built for ray tracing, geometry::BuildFallbackMesh / FallbackMeshBuilder.h)
// through its pre-packed orthographic "Cards" (geometry::SurfaceCacheCardEntry, ClusterFormat.h)
// into a shared global texture atlas, injecting albedo/normal/emissive per texel. A future GI pass
// samples this atlas by reprojecting a world position through a card's stored UV rect (uvMin/
// uvMax) instead of re-evaluating heavy cluster geometry at every light bounce.
//
// --- Why "asynchronous" ---
// Every card is captured through its own tiny, disjoint vkCmdBeginRendering/EndRendering scope
// (render area == exactly that card's atlas rect, so its LOAD_OP_CLEAR never disturbs a
// neighboring card -- geometry::PackCardsIntoAtlas guarantees the rects, gutter included, never
// overlap), but RecordCapture() only (re-)captures a bounded slice of the total card list per
// call (kCardsPerFrameBudget), round-robining through the full list across many frames/calls
// instead of blocking one frame on the whole scene -- the same idea as Lumen's own per-frame
// surface cache capture budget, and the same "one command buffer, no mid-frame submits" discipline
// renderer::ClusterRenderPipeline's own class comment documents for the rest of this pipeline
// (RecordCapture() only records into an already-open, caller-owned command buffer; it never
// submits on its own).
//
// --- Material ---
// This codebase has no texture/material-binding system (see ClusterResolve.comp's own comment) --
// SurfaceCacheCapture.frag reuses the exact same procedural-material approach every other shading
// pass in this codebase already uses (procedural_material.glsl's HashID/HsvToRgb, keyed by
// entityID) plus a small triplanar value-noise modulation so a captured card is not perfectly
// flat-shaded -- a genuinely complete (if intentionally simple) procedural material, not a stub.
//
// --- Atlas layout convention ---
// The 3 atlas images (albedo/normal/emissive) are allocated with COLOR_ATTACHMENT_BIT |
// SAMPLED_BIT and kept in VK_IMAGE_LAYOUT_GENERAL for their ENTIRE lifetime after Init() (both a
// valid color-attachment layout for dynamic rendering and a valid sampled-image layout, mirroring
// renderer::ClusterResolvePass's own output image, which is likewise written by one stage and
// meant to be sampled by a later one without ping-ponging layouts every frame) -- RecordCapture()
// never transitions them, only inserts a memory barrier at the end. The shared depth image stays
// in VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL for its entire lifetime (purely an internal scratch
// buffer for this pass, never sampled externally).

#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/ClusterFormat.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    class SurfaceCachePass {
    public:
        SurfaceCachePass() = default;

        SurfaceCachePass(const SurfaceCachePass&) = delete;
        SurfaceCachePass& operator=(const SurfaceCachePass&) = delete;

        static constexpr VkFormat kAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr VkFormat kNormalFormat = VK_FORMAT_R8G8B8A8_UNORM; // Octahedral-encoded world-space normal in RG.
        static constexpr VkFormat kEmissiveFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
        // HDR outgoing-radiance atlas: what a GI trace (SWRT/HWRT, see SurfaceCacheSWRTPass /
        // SurfaceCacheRayTracingPass) samples as "the luminance stored at this texel," and what
        // SurfaceCacheGIInject.comp read-modify-writes a secondary bounce into. Seeded at capture
        // time (SurfaceCacheCapture.frag) to emissive + albedo*kInitialRadianceAmbientProxy so the
        // very first trace before any GI injection pass has run does not sample pure black.
        // STORAGE_BIT (unlike albedo/normal/emissive) because the injection compute shader needs
        // imageLoad/imageStore, not just a sampled read.
        static constexpr VkFormat kRadianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        static constexpr float kInitialRadianceAmbientProxy = 0.15f;
        // World-space (== local-space, this codebase's entities carry no runtime transform --
        // see SurfaceCacheCapture.vert's own comment) hit position atlas: full float precision
        // because a demoscene-scale local position is not reliably representable in fp16. This is
        // the "where in 3D space does this texel's captured surface actually sit" a GI injection
        // pass needs to originate its hemisphere rays from -- the capture pass's own depth buffer
        // is a same-lifetime scratch image (see class comment) with no sampled-read usage, so it
        // cannot serve that purpose.
        static constexpr VkFormat kWorldPosFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

        // How many cards RecordCapture() (re-)captures per call -- see the class comment's
        // "asynchronous" note. Small enough that even a full command buffer's worth of capture
        // draws costs a handful of tiny (card-sized, not full-screen) rasterization scopes.
        static constexpr uint32_t kCardsPerFrameBudget = 4;

        // Reads the surface-cache card table + every fallback mesh's geometry from
        // `cacheFilePath` (written by geometry::CacheFileManager::WriteCacheFile), uploads one
        // combined vertex/index GPU buffer covering every entity's Fallback Mesh, allocates the 3
        // atlas images + 1 shared depth image (all geometry::kSurfaceCacheAtlasSize^2), clears the
        // atlas images to a neutral default (so a card not yet captured samples something sane
        // rather than undefined memory), and builds the capture graphics pipeline. A scene with
        // zero cards is valid (Init() succeeds, RecordCapture() is then a no-op) -- only an actual
        // I/O failure reading the cache file returns false.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const std::filesystem::path& cacheFilePath);

        void Shutdown();

        // Records up to kCardsPerFrameBudget cards' worth of capture draws into `cmd`, advancing
        // the internal round-robin cursor so repeated calls eventually cover every card. See the
        // class comment for the atlas images' fixed GENERAL layout contract (no caller-side
        // transition needed, before or between calls). Ends with a VkMemoryBarrier2 making every
        // texel captured THIS call visible to a later fragment/compute sampled read.
        void RecordCapture(VkCommandBuffer cmd);

        uint32_t GetCardCount() const { return static_cast<uint32_t>(m_Cards.size()); }
        const std::vector<geometry::SurfaceCacheCardEntry>& GetCards() const { return m_Cards; }

        VkImage GetAlbedoImage() const { return m_AlbedoImage; }
        VkImageView GetAlbedoView() const { return m_AlbedoView; }
        VkImage GetNormalImage() const { return m_NormalImage; }
        VkImageView GetNormalView() const { return m_NormalView; }
        VkImage GetEmissiveImage() const { return m_EmissiveImage; }
        VkImageView GetEmissiveView() const { return m_EmissiveView; }
        VkImage GetRadianceImage() const { return m_RadianceImage; }
        VkImageView GetRadianceView() const { return m_RadianceView; }
        VkImage GetWorldPosImage() const { return m_WorldPosImage; }
        VkImageView GetWorldPosView() const { return m_WorldPosView; }
        VkSampler GetAtlasSampler() const { return m_AtlasSampler; }

        // One entity's span inside the combined vertex/index buffers -- vkCmdDrawIndexed's own
        // (vertexOffset, firstIndex, indexCount) triple, so a per-card draw is one indexed draw
        // call with no further indirection. Public (unlike the rest of this class' internals) so
        // renderer::SurfaceCacheRayTracingPass can build one BLAS per entity directly against this
        // pass' own combined vertex/index buffers -- see GetVertexBuffer()/GetIndexBuffer().
        struct EntityDrawRange {
            int32_t vertexOffset = 0;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
        };
        const std::unordered_map<uint32_t, EntityDrawRange>& GetEntityRanges() const { return m_EntityRanges; }

        // The combined Fallback Mesh vertex/index buffers every entity's cards are captured from
        // (geometry::FallbackVertex / uint32_t, see EntityDrawRange). Created with
        // VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT in addition to their vertex/index-buffer usage
        // (see Init()) precisely so renderer::SurfaceCacheRayTracingPass can build BLAS geometry
        // directly against them -- no duplicate upload of the same geometry for ray tracing.
        VkBuffer GetVertexBuffer() const { return m_VertexBuffer.Handle(); }
        VkBuffer GetIndexBuffer() const { return m_IndexBuffer.Handle(); }
        VkDeviceSize GetVertexBufferSize() const { return m_VertexBuffer.Size(); }
        VkDeviceSize GetIndexBufferSize() const { return m_IndexBuffer.Size(); }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        std::vector<geometry::SurfaceCacheCardEntry> m_Cards;
        std::unordered_map<uint32_t, EntityDrawRange> m_EntityRanges; // Keyed by entityID.
        uint32_t m_CaptureCursor = 0;

        GpuBuffer m_VertexBuffer; // geometry::FallbackVertex[], GPU_ONLY.
        GpuBuffer m_IndexBuffer;  // uint32_t[], GPU_ONLY.

        VkImage m_AlbedoImage = VK_NULL_HANDLE;
        VmaAllocation m_AlbedoAllocation = VK_NULL_HANDLE;
        VkImageView m_AlbedoView = VK_NULL_HANDLE;
        VkImage m_NormalImage = VK_NULL_HANDLE;
        VmaAllocation m_NormalAllocation = VK_NULL_HANDLE;
        VkImageView m_NormalView = VK_NULL_HANDLE;
        VkImage m_EmissiveImage = VK_NULL_HANDLE;
        VmaAllocation m_EmissiveAllocation = VK_NULL_HANDLE;
        VkImageView m_EmissiveView = VK_NULL_HANDLE;
        VkImage m_RadianceImage = VK_NULL_HANDLE;
        VmaAllocation m_RadianceAllocation = VK_NULL_HANDLE;
        VkImageView m_RadianceView = VK_NULL_HANDLE;
        VkImage m_WorldPosImage = VK_NULL_HANDLE;
        VmaAllocation m_WorldPosAllocation = VK_NULL_HANDLE;
        VkImageView m_WorldPosView = VK_NULL_HANDLE;
        VkImage m_DepthImage = VK_NULL_HANDLE;
        VmaAllocation m_DepthAllocation = VK_NULL_HANDLE;
        VkImageView m_DepthView = VK_NULL_HANDLE;
        VkSampler m_AtlasSampler = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
