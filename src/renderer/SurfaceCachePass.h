#pragma once
// Surface Cache capture pass (Lumen-style): projects each entity's Fallback Mesh (the same coarse
// BVH proxy geometry built for ray tracing, geometry::BuildFallbackMesh / FallbackMeshBuilder.h)
// through its pre-packed orthographic "Cards" (geometry::SurfaceCacheCardEntry, ClusterFormat.h)
// into a shared global texture atlas, injecting albedo/normal/emissive/direct-lighting/radiance/
// world-position per texel (direct-lighting shaded against renderer::ShadowMapPass's sun shadow
// map plus any active point lights, see SetShadowMap/UpdateLighting and SurfaceCacheCapture.frag).
// renderer::SurfaceCacheSWRTPass / SurfaceCacheRayTracingPass / SurfaceCacheGIInjectPass are the
// GI consumers: they sample this atlas by reprojecting a hit position through a card's stored UV
// rect (uvMin/uvMax) instead of re-evaluating heavy cluster geometry at every light bounce, and
// SurfaceCacheGIInjectPass writes a secondary hemisphere-sampled bounce back into the radiance
// atlas (see that pass' own class comment).
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
// The 6 atlas images (albedo/normal/emissive/direct-lighting/radiance/world-position) are
// allocated with COLOR_ATTACHMENT_BIT | SAMPLED_BIT (radiance additionally STORAGE_BIT -- see
// kRadianceFormat's own comment) and kept in VK_IMAGE_LAYOUT_GENERAL for their ENTIRE lifetime
// after Init() (both a valid color-attachment layout for dynamic rendering and a valid sampled-
// image layout, mirroring renderer::ClusterResolvePass's own output image, which is likewise
// written by one stage and meant to be sampled by a later one without ping-ponging layouts every
// frame) -- RecordCapture() never transitions them, only inserts a memory barrier at the end. The
// shared depth image stays in VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL for its entire lifetime
// (purely an internal scratch buffer for this pass, never sampled externally).
//
// --- Dynamic atlas residency ---
// Cards are NOT all resident in the atlas simultaneously: UpdateVisibility() frustum-culls every
// card's entity AABB each frame and grants/revokes atlas pages via m_AtlasAllocator (geometry::
// SurfaceCacheAtlasAllocator) as cards enter/leave the camera's view, evicting a card only after
// kEvictionFrameDelay consecutive unwanted frames (absorbing frustum-edge flicker) and
// defragmenting the atlas as a last resort before failing an allocation outright (graceful
// degradation -- see UpdateVisibility()'s own comment). GetCards()' returned atlasOffset/uvMin/
// uvMax are therefore only valid while IsCardResident(cardIndex) is true; a GI trace consumer
// (renderer::SurfaceCacheTraceContext) must re-derive its own card index tables whenever residency
// changes, not just once at Init().

#include <cstdint>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/CardGenerator.h" // geometry::kSurfaceCacheAtlasSize
#include "geometry/ClusterFormat.h"
#include "geometry/SurfaceCacheAtlasAllocator.h"
#include "renderer/GpuBuffer.h"
#include "renderer/LightingTypes.h"

namespace renderer {

    class SurfaceCachePass {
    public:
        SurfaceCachePass() = default;

        SurfaceCachePass(const SurfaceCachePass&) = delete;
        SurfaceCachePass& operator=(const SurfaceCachePass&) = delete;

        static constexpr VkFormat kAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr VkFormat kNormalFormat = VK_FORMAT_R8G8B8A8_UNORM; // Octahedral-encoded world-space normal in RG.
        static constexpr VkFormat kEmissiveFormat = VK_FORMAT_R8G8B8A8_UNORM;
        // HDR-capable (unlike the 3 above, UNORM, atlas images): accumulated direct-light radiance
        // (sun + point lights, see SurfaceCacheCapture.frag's ComputeDirectLighting) can exceed 1.0
        // well before any tone mapping happens in a later GI-consuming pass. Stored UN-multiplied
        // by albedo (see that function's own comment) -- kRadianceFormat below is where the
        // albedo-multiplied, GI-ready combined value lives.
        static constexpr VkFormat kDirectLightingFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        // HDR outgoing-radiance atlas: what a GI trace (SWRT/HWRT, see SurfaceCacheSWRTPass /
        // SurfaceCacheRayTracingPass) samples as "the luminance stored at this texel," and what
        // SurfaceCacheGIInject.comp read-modify-writes a secondary bounce into. Seeded at capture
        // time (SurfaceCacheCapture.frag) to emissive + albedo*directLighting -- the same "fold
        // albedo into the lighting atlas" step ComputeDirectLighting's own comment defers to "a
        // future pass," which this atlas IS. STORAGE_BIT (unlike the other 5) because the
        // injection compute shader needs imageLoad/imageStore, not just a sampled read.
        static constexpr VkFormat kRadianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        // World-space (== local-space, this codebase's entities carry no runtime transform --
        // see SurfaceCacheCapture.vert's own comment) hit position atlas: full float precision
        // because a demoscene-scale local position is not reliably representable in fp16. This is
        // the "where in 3D space does this texel's captured surface actually sit" a GI injection
        // pass needs to originate its hemisphere rays from -- the capture pass's own depth buffer
        // is a same-lifetime scratch image (see class comment) with no sampled-read usage, so it
        // cannot serve that purpose.
        static constexpr VkFormat kWorldPosFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

        // How many cards RecordCapture() (re-)captures per call -- see the class comment's
        // "asynchronous" note. Small enough that even a full command buffer's worth of capture
        // draws costs a handful of tiny (card-sized, not full-screen) rasterization scopes.
        static constexpr uint32_t kCardsPerFrameBudget = 4;

        // How many consecutive UpdateVisibility() calls a resident card is allowed to go unwanted
        // before its atlas page is actually freed -- see UpdateVisibility()'s own comment. A short
        // grace period absorbs a card flickering in and out of the frustum (e.g. an entity near
        // the screen edge) without paying a re-capture every single time it comes back.
        static constexpr uint32_t kEvictionFrameDelay = 120;

        // Reads the surface-cache card table + every fallback mesh's geometry from
        // `cacheFilePath` (written by geometry::CacheFileManager::WriteCacheFile), uploads one
        // combined vertex/index GPU buffer covering every entity's Fallback Mesh, allocates the 6
        // atlas images + 1 shared depth image (all geometry::kSurfaceCacheAtlasSize^2), clears the
        // atlas images to a neutral default (so a card not yet captured samples something sane
        // rather than undefined memory), and builds the capture graphics pipeline. A scene with
        // zero cards is valid (Init() succeeds, RecordCapture() is then a no-op) -- only an actual
        // I/O failure reading the cache file returns false.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const std::filesystem::path& cacheFilePath);

        void Shutdown();

        // Runtime residency update (see the class comment's "Dynamic atlas residency" section):
        // tests every card's entity AABB against the 6 planes of the view frustum described by
        // `cameraPosition`/`cameraForward`/`cameraUp` (need not be re-orthonormalized -- cameraUp
        // is only used via cameraForward.Cross(cameraUp) to derive right, then re-crossed for an
        // orthonormal up, so small numerical drift in cameraUp is harmless), `fovYRadians`,
        // `aspectRatio` (width/height) and [nearZ, farZ]. A newly-visible, not-yet-resident card is
        // allocated a page from m_AtlasAllocator (evicting unwanted cards, and if still short of
        // contiguous space, fully defragmenting -- see the .cpp's AllocateCardPage) and queued for
        // capture; a card unwanted for more than kEvictionFrameDelay consecutive calls has its page
        // freed. Must be called once per frame, before RecordCapture() (which only captures cards
        // this call has queued).
        void UpdateVisibility(const maths::vec3& cameraPosition, const maths::vec3& cameraForward,
            const maths::vec3& cameraUp, float fovYRadians, float aspectRatio, float nearZ, float farZ);

        // Binds the sun's shadow map (renderer::ShadowMapPass::GetShadowMapView/GetShadowMapSampler)
        // into this pass's lighting descriptor set (set 0, binding 1 -- see SurfaceCacheCapture.frag).
        // Must be called exactly once after Init(), before the first RecordCapture() call, and
        // before any UpdateLighting() call -- the shadow map image itself is never recreated after
        // ShadowMapPass::Init(), so this binding does not need to be refreshed again afterward
        // (only the UBO contents change per frame, via UpdateLighting()).
        void SetShadowMap(VkImageView shadowMapView, VkSampler shadowMapSampler);

        // Writes `lights` and `lightViewProj` (renderer::ShadowMapPass::GetLightViewProj() from
        // the SAME frame's ShadowMapPass::RecordCapture() call) into the persistently-mapped
        // lighting UBO (set 0, binding 0) that every SurfaceCacheCapture.frag invocation reads.
        // Call once per frame before RecordCapture() -- a plain memcpy into host-visible/coherent
        // memory, no descriptor-set update needed (only SetShadowMap() touches the descriptor set
        // itself).
        void UpdateLighting(const SceneLights& lights, const maths::mat4& lightViewProj);

        // Records up to kCardsPerFrameBudget cards' worth of capture draws into `cmd`, draining
        // from the front of the dirty-card queue UpdateVisibility() fills (a card is dirty exactly
        // once, right after (re)gaining residency -- there is no periodic re-capture of an
        // already-captured resident card, since this engine's entities are static, see
        // renderer::GlobalSDFPass's own "Scope" note). See the class comment for the atlas images'
        // fixed GENERAL layout contract (no caller-side transition needed, before or between
        // calls). Ends with a VkMemoryBarrier2 making every texel captured THIS call visible to a
        // later fragment/compute sampled read.
        void RecordCapture(VkCommandBuffer cmd);

        uint32_t GetCardCount() const { return static_cast<uint32_t>(m_Cards.size()); }
        // NOTE: atlasOffset/uvMin/uvMax are now a DYNAMIC placement (see UpdateVisibility()), not
        // the static one geometry::CardGenerator::PackCardsIntoAtlas originally baked into the
        // .cache file -- valid only for a card where IsCardResident() is true.
        const std::vector<geometry::SurfaceCacheCardEntry>& GetCards() const { return m_Cards; }
        bool IsCardResident(uint32_t cardIndex) const {
            return cardIndex < m_CardStates.size() && m_CardStates[cardIndex].resident;
        }

        VkImage GetAlbedoImage() const { return m_AlbedoImage; }
        VkImageView GetAlbedoView() const { return m_AlbedoView; }
        VkImage GetNormalImage() const { return m_NormalImage; }
        VkImageView GetNormalView() const { return m_NormalView; }
        VkImage GetEmissiveImage() const { return m_EmissiveImage; }
        VkImageView GetEmissiveView() const { return m_EmissiveView; }
        VkImage GetDirectLightingImage() const { return m_DirectLightingImage; }
        VkImageView GetDirectLightingView() const { return m_DirectLightingView; }
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
        // Per-card runtime residency state -- parallel to m_Cards (same index), populated lazily:
        // a card that has never been visible has resident == false and an all-zero rect, which is
        // exactly the state it would have right after eviction, so no separate "never allocated"
        // flag is needed.
        struct CardRuntimeState {
            bool resident = false;
            bool dirty = false; // Queued in m_DirtyCardQueue, awaiting its first/next capture.
            uint32_t framesSinceVisible = 0; // Reset to 0 every UpdateVisibility() call the card is wanted in.
            geometry::AtlasRect rect{}; // The gutter-PADDED allocation (what must be passed to Free()); valid only while resident == true.
        };

        // Tries to place `card` (its atlasSize, gutter-padded) into m_AtlasAllocator, evicting
        // unwanted resident cards and, if that alone is not enough contiguous space, fully
        // defragmenting first. Returns false only if the atlas genuinely cannot fit `card` even
        // once every unwanted card is evicted and the atlas is fully repacked -- logged, not fatal
        // (see UpdateVisibility()'s class-comment note on graceful degradation).
        bool AllocateCardPage(uint32_t cardIndex, geometry::AtlasRect& outRect);

        // Marks `cardIndex` resident at the given gutter-padded atlas rect, stamping the
        // corresponding unpadded atlasOffset/uvMin/uvMax into m_Cards[cardIndex] (the public,
        // GI-facing placement -- see GetCards()'s own comment) and queuing it for capture exactly
        // once (idempotent: calling this again on an already-dirty card does not requeue it).
        void ApplyCardPlacement(uint32_t cardIndex, const geometry::AtlasRect& paddedRect);

        // Frees every resident card whose framesSinceVisible > 0 (i.e. not wanted THIS call),
        // regardless of kEvictionFrameDelay -- the fallback AllocateCardPage() reaches for once a
        // plain eviction-by-delay pass was not enough contiguous space for a new arrival.
        void EvictAllUnwantedCards();

        // Full defragmentation rebuild: resets m_AtlasAllocator to one whole-atlas free rect, then
        // re-Allocate()s every still-resident card's existing (gutter-padded) size, largest first
        // (mirrors CardGenerator::PackCardsIntoAtlas's own tallest-first ordering), writing each
        // card's new rect back into its CardRuntimeState. Only called from AllocateCardPage() as a
        // last resort, since it touches (and marks dirty for re-capture) every resident card, not
        // just the one being allocated for.
        void DefragmentAtlas();

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        std::vector<geometry::SurfaceCacheCardEntry> m_Cards;
        std::vector<CardRuntimeState> m_CardStates; // Parallel to m_Cards.
        std::unordered_map<uint32_t, EntityDrawRange> m_EntityRanges; // Keyed by entityID.
        geometry::SurfaceCacheAtlasAllocator m_AtlasAllocator{ geometry::kSurfaceCacheAtlasSize };
        std::deque<uint32_t> m_DirtyCardQueue; // Card indices queued for capture by UpdateVisibility(), drained by RecordCapture().

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
        VkImage m_DirectLightingImage = VK_NULL_HANDLE;
        VmaAllocation m_DirectLightingAllocation = VK_NULL_HANDLE;
        VkImageView m_DirectLightingView = VK_NULL_HANDLE;
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

        // Lighting: one UBO (set 0 binding 0, SurfaceCacheLightingUBO -- see the .cpp) written by
        // UpdateLighting(), one combined image sampler (set 0 binding 1, the sun's shadow map)
        // bound once by SetShadowMap(). Both stay in the SAME descriptor set for the pass's whole
        // lifetime; only the UBO's CONTENTS change per frame.
        GpuBuffer m_LightingUBO; // Persistently mapped, CPU_TO_GPU.
        VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_LightingDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_LightingDescriptorSet = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
