#pragma once
// Surface Cache capture pass (Lumen-style): projects each entity's Fallback Mesh (the same coarse
// BVH proxy geometry built for ray tracing, geometry::BuildFallbackMesh / FallbackMeshBuilder.h)
// through its pre-packed orthographic "Cards" (geometry::SurfaceCacheCardEntry, ClusterFormat.h)
// into a shared global texture atlas, injecting albedo/normal/emissive/direct-lighting/radiance/
// world-position per texel (direct-lighting shaded against renderer::VirtualShadowMapPass's sun
// clipmap and point light cube faces, see SetVirtualShadowMap/UpdateLighting and
// SurfaceCacheCapture.frag).
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
// SurfaceCacheCapture.frag samples the same Substrate material table every other shading pass in
// this codebase reads (material_params.glsl / substrate_bsdf.glsl -- see ClusterResolve.comp's own
// Step 3 comment): each entity's real materialID is looked up via core::EntityData (bound at set 0
// binding 6) and used to index g_MaterialParams (binding 7 -- the SAME SSBO renderer::
// ClusterResolvePass::Init() already fills, bound directly here via GetMaterialParamsBuffer(), no
// re-upload), plus a small triplanar value-noise modulation on top of the real albedo so a
// captured card is not perfectly flat-shaded.
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

#include "core/EntityData.h" // core::EntityTransformCPU/core::EntityData -- Phase 6's RecordCapture priority ordering + Feature 2's translucent-card exclusion.
#include "core/maths/Maths.h"
#include "core/EngineConfig.h"
#include "geometry/CardGenerator.h" // geometry::kSurfaceCacheAtlasSize
#include "geometry/ClusterFormat.h"
#include "geometry/SurfaceCacheAtlasAllocator.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/GpuImage.h"
#include "renderer/LightingTypes.h"
#include "renderer/MaterialParameterTable.h" // renderer::MaterialTable -- Feature 2's translucent-card exclusion.

namespace renderer {

    class VirtualShadowMapPass;

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
        //
        // Re-verified during the 2026-07-18 VRAM-reduction audit (an RGBA16F downgrade here was
        // in scope, then deliberately reverted): VulkanContext::GenerateTerrain(300.0f, 300.0f, ...)
        // bakes the floor/terrain entity's vertices across roughly [-150, +150] on X/Z (geom_terrain
        // .comp's xPos/zPos), and SurfaceCacheCapture.vert/.frag write that raw, un-rebased position
        // straight through (outWorldPos = inPosition) -- NOT the ~14m camera-orbit scale
        // ScreenProbeTemporal.comp's own kDisocclusionDistanceThreshold comment assumes. fp16's ULP
        // at |150| is ~0.125 world units, and SurfaceCacheGIInject.comp uses this exact value,
        // unmodified, as a hemisphere-ray origin with only a 1.0e-2 unit surface bias
        // (biasedOrigin = origin + normal * 1.0e-2) -- quantization error an order of magnitude
        // larger than that bias risks visible GI self-intersection/light-leak artifacts on the
        // terrain specifically. Kept at fp32; see renderer::ReflectionPass::kWorldPosFormat's own
        // comment for the same finding applied to its own (also-un-rebased) world-position slots.
        static constexpr VkFormat kWorldPosFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

        // How many cards RecordCapture() (re-)captures per call -- see the class comment's
        // "asynchronous" note. Small enough that even a full command buffer's worth of capture
        // draws costs a handful of tiny (card-sized, not full-screen) rasterization scopes.
        // Bumped 16->48 for the Phase 4 integration (dynamic scenes onto main): while
        // config::ENTITY_SELF_ROTATION_ENABLED is on, MarkAllRotatingEntityCardsDirty() re-queues
        // every resident card every frame (up to ~kEntityCount x 6 faces), so a too-small budget
        // would visibly lag the Surface Cache behind the actual rotation -- generous rather than
        // dimensioned to the exact worst case, adjustable via the log if still insufficient.
        static inline uint32_t kCardsPerFrameBudget = 48u;

        // How many consecutive UpdateVisibility() calls a resident card is allowed to go unwanted
        // before its atlas page is actually freed -- see UpdateVisibility()'s own comment. A short
        // grace period absorbs a card flickering in and out of the frustum (e.g. an entity near
        // the screen edge) without paying a re-capture every single time it comes back.
        static inline uint32_t kEvictionFrameDelay = 600u;

        // Reads the surface-cache card table + every fallback mesh's geometry from
        // `cacheFilePath` (written by geometry::CacheFileManager::WriteCacheFile), uploads one
        // combined vertex/index GPU buffer covering every entity's Fallback Mesh, allocates the 6
        // atlas images + 1 shared depth image (all m_AtlasSize^2, tier-scaled from
        // config::lumen::SURFACE_CACHE_ATLAS_SIZE -- see that member's own comment), clears the
        // atlas images to a neutral default (so a card not yet captured samples something sane
        // rather than undefined memory), and builds the capture graphics pipeline. A scene with
        // zero cards is valid (Init() succeeds, RecordCapture() is then a no-op) -- only an actual
        // I/O failure reading the cache file returns false.
        // `entityDataBuffer` (core::EntityData[], the SAME buffer renderer::ClusterResolvePass::Init
        // already receives) and `materialParamsBuffer` (renderer::ClusterResolvePass::
        // GetMaterialParamsBuffer(), the already-filled MaterialParams SSBO -- no re-upload here)
        // are bound at set 0 bindings 6/7 of the lighting descriptor set (see STEP 4 in the .cpp),
        // so SurfaceCacheCapture.frag can look up each entity's real materialID and sample the same
        // Substrate material table every other shading pass reads, instead of an independently
        // invented procedural color.
        //
        // Feature 2 (Lumen advanced roadmap, translucent-material cache exclusion): `entityDataCPU`
        // (renderer::VulkanContext::GetEntityData(), index == meshID == card.entityID) and
        // `materialTable` (the SAME CPU-authored table renderer::VulkanContext::GetMaterialTable()
        // returns, ClusterRenderPipelineCreateInfo::materialTable) are read ONCE here, right after
        // m_Cards is populated from the cache file, to std::erase_if every card whose owning
        // entity's REAL material alpha is < 1.0 (glass/water) -- see this method's own .cpp comment
        // for why this uses the material's real alpha, NOT core::EntityFlags::IsTransparent (a
        // VisBuffer-routing flag also forced true on the fully-opaque hero entity, which would
        // wrongly strip its cards too). A real UE5.8 Lumen Surface Cache never captures translucent
        // surfaces either -- glass/water are already lit BY this cache via TransparentForwardPass/
        // WaterForwardPass, they must not also BAKE INTO it as if opaque.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const std::filesystem::path& cacheFilePath, VkBuffer entityDataBuffer, VkBuffer materialParamsBuffer,
            const core::EntityData* entityDataCPU, const renderer::MaterialTable& materialTable);

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

        // Binds renderer::VirtualShadowMapPass's own resources (physical page atlas + sampler,
        // page table, feedback buffer, sun clipmap levels UBO, point light cube faces UBO) into
        // this pass's lighting descriptor set (set 0, bindings 1-5 -- see SurfaceCacheCapture.frag
        // / shadow_atlas_sampling.glsl / shadow_page_table.glsl / shadow_feedback.glsl /
        // shadow_sun_sampling.glsl / shadow_point_sampling.glsl). Must be called exactly once after
        // Init(), before the first RecordCapture() call, and before any UpdateLighting() call --
        // none of these buffer/image handles are ever recreated after VirtualShadowMapPass::Init(),
        // so this binding does not need to be refreshed again afterward (only their CONTENTS change
        // per frame, via VirtualShadowMapPass::RecordBeginFrame()).
        void SetVirtualShadowMap(const VirtualShadowMapPass& vsm);

        // Writes `lights` into the persistently-mapped lighting UBO (set 0, binding 0) that every
        // SurfaceCacheCapture.frag invocation reads. Call once per frame before RecordCapture() --
        // a plain memcpy into host-visible/coherent memory, no descriptor-set update needed (only
        // SetVirtualShadowMap() touches the descriptor set itself). Unlike pre-Phase-3, this no
        // longer takes a `lightViewProj` parameter -- shadow lookups read
        // renderer::VirtualShadowMapPass's own dedicated UBOs (bindings 4/5) directly instead.
        void UpdateLighting(const SceneLights& lights);

        // Records up to kCardsPerFrameBudget cards' worth of capture draws into `cmd`, draining
        // from the front of the dirty-card queue UpdateVisibility() fills (a card is normally dirty
        // exactly once, right after (re)gaining residency -- this engine's entities are static by
        // default). Phase 4 integration (dynamic scenes onto main): when
        // config::ENTITY_SELF_ROTATION_ENABLED is on, this method ALSO calls
        // MarkAllRotatingEntityCardsDirty() first, re-queuing every resident card every call, since
        // a rotating entity's captured surface data goes stale every frame. See the class comment for the atlas images'
        // fixed GENERAL layout contract (no caller-side transition needed, before or between
        // calls). Ends with a VkMemoryBarrier2 making every texel captured THIS call visible to a
        // later fragment/compute sampled read.
        //
        // Phase 6 (UE5.8 parity roadmap, adaptive/importance-sampled probes): `cameraPositionWorld`/
        // `entityTransformsCPU` (the SAME per-frame data renderer::ClusterRenderPipeline::RecordFrame
        // already has in scope for its own dynamic-entity loop, see that method's own parameters) are
        // used ONLY when the dirty queue exceeds kCardsPerFrameBudget this call, to capture the
        // closest/largest-on-screen cards first instead of strict FIFO order -- see
        // ComputeCardPriority()'s own comment. `entityTransformsCPU` is indexed by entityID (== meshID),
        // must have at least as many entries as the highest entityID any card references.
        void RecordCapture(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld,
            const core::EntityTransformCPU* entityTransformsCPU);

        uint32_t GetCardCount() const { return static_cast<uint32_t>(m_Cards.size()); }
        // NOTE: atlasOffset/uvMin/uvMax are now a DYNAMIC placement (see UpdateVisibility()), not
        // the static one geometry::CardGenerator::PackCardsIntoAtlas originally baked into the
        // .cache file -- valid only for a card where IsCardResident() is true.
        const std::vector<geometry::SurfaceCacheCardEntry>& GetCards() const { return m_Cards; }
        bool IsCardResident(uint32_t cardIndex) const {
            return cardIndex < m_CardStates.size() && m_CardStates[cardIndex].resident;
        }

        VkImage GetAlbedoImage() const { return m_Albedo.Image(); }
        VkImageView GetAlbedoView() const { return m_Albedo.View(); }
        VkImage GetNormalImage() const { return m_Normal.Image(); }
        VkImageView GetNormalView() const { return m_Normal.View(); }
        VkImage GetEmissiveImage() const { return m_Emissive.Image(); }
        VkImageView GetEmissiveView() const { return m_Emissive.View(); }
        VkImage GetDirectLightingImage() const { return m_DirectLighting.Image(); }
        VkImageView GetDirectLightingView() const { return m_DirectLighting.View(); }
        VkImage GetRadianceImage() const { return m_Radiance.Image(); }
        VkImageView GetRadianceView() const { return m_Radiance.View(); }
        VkImage GetWorldPosImage() const { return m_WorldPos.Image(); }
        VkImageView GetWorldPosView() const { return m_WorldPos.View(); }
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

        // Phase 6 (UE5.8 parity roadmap): a proxy for "how urgently this dirty card deserves
        // re-capture before the others" -- entity-local bounding radius (a stand-in for on-screen
        // size, since this codebase has no per-frame projected-size buffer to read) divided by
        // distance from `cameraPositionWorld` to the card's CURRENT world-space center (rotated via
        // `entityTransformsCPU[card.entityID]`, mirroring the exact TransformDirection() pattern
        // renderer::SurfaceCacheRayTracingPass/GlobalSDFPass already use for this same operation --
        // no new maths primitive needed). Higher = higher priority. Only the RELATIVE ORDER this
        // produces matters (it feeds a sort key, not a physically exact angular size), so no
        // FOV/aspect-ratio projection is needed. NOTE: SurfaceCacheCardEntry::localBoundsMin/Max is
        // the owning ENTITY's full AABB, identical across all of that entity's cards (see that
        // struct's own comment) -- every card of one entity therefore gets the exact same priority,
        // which is expected (RecordCapture()'s std::stable_sort keeps their relative order
        // deterministic) rather than a defect.
        float ComputeCardPriority(uint32_t cardIndex, const maths::vec3& cameraPositionWorld,
            const core::EntityTransformCPU* entityTransformsCPU) const;

        // Marks `cardIndex` resident at the given gutter-padded atlas rect, stamping the
        // corresponding unpadded atlasOffset/uvMin/uvMax into m_Cards[cardIndex] (the public,
        // GI-facing placement -- see GetCards()'s own comment) and queuing it for capture exactly
        // once (idempotent: calling this again on an already-dirty card does not requeue it).
        void ApplyCardPlacement(uint32_t cardIndex, const geometry::AtlasRect& paddedRect);

        // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): re-queues EVERY
        // already-resident card for capture, every call -- called internally from RecordCapture(),
        // gated by config::ENTITY_SELF_ROTATION_ENABLED (see that call site's own comment). Reuses
        // ApplyCardPlacement's own idempotent enqueue check (CardRuntimeState::dirty), so calling
        // this every frame while a card is still awaiting its OWN previous capture does not
        // requeue it a second time. Deliberately does NOT try to exclude the one entity
        // (VulkanContext's floor, always static) that never actually rotates -- doing so would
        // need a hardcoded entityID here duplicating a fact only VulkanContext currently owns;
        // the floor's cards being harmlessly re-captured every frame while the global rotation
        // switch is on is an accepted, bounded inefficiency (see kCardsPerFrameBudget's own bump),
        // not a correctness issue.
        void MarkAllRotatingEntityCardsDirty();

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

        // Tier-scaled runtime atlas resolution (config::lumen::SURFACE_CACHE_ATLAS_SIZE, re-read at
        // the top of Init() -- see that method's own comment), replacing what used to be a single
        // hardcoded geometry::kSurfaceCacheAtlasSize for every profile. geometry::
        // kSurfaceCacheAtlasSize itself is NOT repurposed here: it remains the offline/cook-time
        // ceiling geometry::PackCardsIntoAtlas validates against when CacheFileManager bakes the
        // .cache file (a build step with no notion of "which GPU tier will load this later"), always
        // the largest possible tier size (2048) regardless of which tier actually runs the result --
        // the runtime SurfaceCacheAtlasAllocator below is explicitly designed to hold only a SUBSET
        // of cards resident at once (see its own class comment), so a smaller runtime atlas on
        // Low/Medium is a supported degradation (more eviction pressure), never a correctness issue.
        // Default-initialized to the same 2048 ceiling purely as a pre-Init() placeholder; Init()
        // always overwrites both this and m_AtlasAllocator together so they never disagree.
        uint32_t m_AtlasSize = geometry::kSurfaceCacheAtlasSize;
        geometry::SurfaceCacheAtlasAllocator m_AtlasAllocator{ m_AtlasSize };
        std::deque<uint32_t> m_DirtyCardQueue; // Card indices queued for capture by UpdateVisibility(), drained by RecordCapture().

        GpuBuffer m_VertexBuffer; // geometry::FallbackVertex[], GPU_ONLY.
        GpuBuffer m_IndexBuffer;  // uint32_t[], GPU_ONLY.

        GpuImage m_Albedo;
        GpuImage m_Normal;
        GpuImage m_Emissive;
        GpuImage m_DirectLighting;
        GpuImage m_Radiance;
        GpuImage m_WorldPos;
        GpuImage m_Depth;
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
