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
// --- Phase 0.3 (PCG roadmap, dynamic Lumen registration) ---
// UpdateVisibility()'s dynamic atlas RESIDENCY (below) is orthogonal to what card ENTRIES even
// exist in the first place: m_Cards/m_EntityRanges/the combined vertex-index buffer were, until
// Phase 0.3, a fixed list built once at Init() from the cache file's card table + fallback-mesh
// table. RegisterEntity()/UnregisterEntity() additively extend that with a small, capacity-bounded
// (kMaxDynamicEntities) set of entities addable/removable at runtime -- see those methods' own
// comments. A registered entity's Cards start non-resident exactly like any other card and are
// picked up by the very next UpdateVisibility() call through the SAME frustum-test/
// m_AtlasAllocator/eviction machinery every other card already uses -- no new atlas-placement logic
// was needed for this, only a way to grow/shrink the card LIST itself. Consistent with
// GlobalSDFPass's own Phase 0.3 scope note: this is deliberately NOT meant for every PCG-scattered
// instance, just a handful of large/hero objects.
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

        // Phase 0.3 (PCG roadmap, dynamic Lumen registration): upper bound on how many entities may
        // be registered via RegisterEntity() at any one time -- a small, fixed headroom mirroring
        // GlobalSDFPass::kMaxDynamicEntities (both passes use the same bound for the same reason:
        // "a handful of large scattered rocks or hero trees", not an unbounded PCG instance count).
        static constexpr uint32_t kMaxDynamicEntities = 8;

        // Reads the surface-cache card table + every fallback mesh's geometry from
        // `cacheFilePath` (written by geometry::CacheFileManager::WriteCacheFile), uploads one
        // combined vertex/index GPU buffer covering every entity's Fallback Mesh, allocates the 6
        // atlas images + 1 shared depth image (all geometry::kSurfaceCacheAtlasSize^2), clears the
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

        // Phase 0.3 (PCG roadmap, dynamic Lumen registration): registers a NEW entity's Cards + a
        // dedicated small vertex/index buffer at runtime, after Init(). Rather than requiring a
        // brand-new offline bake, this REUSES `sourceEntityID`'s ALREADY-BAKED Fallback Mesh
        // geometry + AABB (read fresh from the same cache file Init() consumed, reopened here) --
        // e.g. one of the World Partition streaming pool's archetype meshes -- and registers Cards
        // for it under the caller-chosen `newEntityID` identity via geometry::GenerateEntityCards
        // (the SAME per-entity Phase-1 function CardGenerator's own offline bake calls -- see
        // CardGenerator.h's two-phase API comment; Phase 2's PackCardsIntoAtlas is NOT called here,
        // since m_AtlasAllocator already IS a live allocator, see the class comment). The new Cards
        // are appended non-resident, exactly like a freshly-loaded card, and will be picked up by
        // the very next UpdateVisibility() call.
        // `newEntityID`'s geometry is uploaded into its OWN dedicated GPU vertex/index buffer (kept
        // separate from the shared Init()-time combined buffer, which cannot be grown at runtime) --
        // RecordCapture() binds whichever buffer a given card's owning entity actually needs, per
        // draw (see EntityDrawRange::isDynamic).
        // Returns false (logged), touching no state, if: `newEntityID` already names a resident
        // entity (from Init() or a prior RegisterEntity() call); kMaxDynamicEntities dynamically-
        // registered entities are already live; `sourceEntityID` has no Fallback Mesh in the cache
        // file; the read fails; or GenerateEntityCards produces zero cards for its AABB (fully
        // degenerate).
        // NOTE (risk -- same caveat as GlobalSDFPass::RegisterEntity): `newEntityID` is NOT
        // validated against renderer::VulkanContext's own entity-transform array bounds. If
        // config::ENTITY_SELF_ROTATION_ENABLED is (or later becomes) enabled and RecordCapture() is
        // ever called with a non-null `entityTransformsCPU` while a dynamically-registered entity
        // whose `newEntityID` is outside that array's valid range is still live,
        // ComputeCardPriority()'s/MarkAllRotatingEntityCardsDirty()'s (indirect, via
        // ComputeCardPriority) `entityTransformsCPU[card.entityID]` indexing is an out-of-bounds
        // read. Callers using synthetic IDs outside the engine's normal dense entity range MUST
        // keep rotation disabled, or pass entityTransformsCPU == nullptr to RecordCapture() and keep
        // the dirty queue within kCardsPerFrameBudget (so ComputeCardPriority() is never invoked),
        // for the lifetime of any such registration -- a deliberate, documented boundary Phase 0.3
        // does not attempt to close (composite-list/card-list add/remove only).
        bool RegisterEntity(uint32_t newEntityID, uint32_t sourceEntityID, VkCommandPool commandPool, VkQueue queue);

        // Phase 0.3: reverses RegisterEntity() -- evicts (frees the atlas page of, if resident) and
        // tombstones every Card belonging to `entityID` (returning their slots to a free list a
        // LATER RegisterEntity() call can reuse, WITHOUT shifting any other card's index -- other
        // cards' indices must stay stable, since m_AtlasAllocator/m_DirtyCardQueue/external GI
        // consumers reference them by position across frames, unlike GlobalSDFPass's own
        // m_Entities list, which has no such cross-frame index dependency), erases its
        // EntityDrawRange, and destroys its dedicated vertex/index buffer. Only ever removes an
        // entity that was ITSELF added via RegisterEntity() -- a no-op (logged) for any entityID
        // that is either not currently registered at all, or is one of Init()'s own fixed-roster
        // entities (never removable through this API, by design -- see GlobalSDFPass::
        // UnregisterEntity's identical rule).
        void UnregisterEntity(uint32_t entityID);

        // Phase 0.3: number of card slots currently ACTIVE (a real Init-time card, or a currently-
        // claimed RegisterEntity() slot) -- unlike GetCardCount() (== m_Cards.size(), a monotonic
        // high-water mark that never shrinks, since freed dynamic slots are tombstoned/reused in
        // place rather than erased -- see UnregisterEntity's own comment), this returns to its
        // pre-registration value once every dynamically-registered entity is unregistered again,
        // making it the more useful before/after metric for a caller validating a register/
        // unregister round trip.
        uint32_t GetActiveCardCount() const;

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
            // Phase 0.3: true for an entity registered via RegisterEntity() -- its geometry lives in
            // that entity's OWN dedicated buffer (m_DynamicEntityGeometry[entityID]), not the shared
            // Init()-time combined m_VertexBuffer/m_IndexBuffer this struct's offsets normally index
            // into. RecordCapture() checks this to decide which buffer to bind before each draw.
            bool isDynamic = false;
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
            // Phase 0.3: true for every Init()-time real card AND for a currently-claimed
            // RegisterEntity() slot; false for a tombstoned (UnregisterEntity()'d) slot sitting in
            // m_FreeDynamicCardSlots, awaiting reuse. UpdateVisibility() skips inactive slots
            // outright (their m_Cards[] entry may hold stale data from before eviction) -- every
            // other per-card loop in this class (EvictAllUnwantedCards/DefragmentAtlas/
            // RecordCapture) already only ever touches a slot that is resident and/or dirty, both
            // of which are always false on an inactive slot, so no other change was needed.
            bool active = true;
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

        // Phase 0.3: one dynamically-registered entity's own dedicated Fallback Mesh vertex/index
        // buffers -- kept separate from the shared Init()-time combined m_VertexBuffer/m_IndexBuffer
        // (see the class comment's own "Phase 0.3" section for why: that buffer cannot be grown at
        // runtime). Same usage flags as the shared buffers (VERTEX/INDEX_BUFFER_BIT | STORAGE_BUFFER_BIT
        // | TRANSFER_DST_BIT | ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        // SHADER_DEVICE_ADDRESS_BIT) for parity, even though no current Phase 0.3 consumer needs the
        // acceleration-structure/SSBO usages -- avoids a landmine for a future consumer expecting
        // every entity's geometry buffer to share one usage contract.
        struct DynamicEntityGeometry {
            GpuBuffer vertexBuffer;
            GpuBuffer indexBuffer;
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        std::vector<geometry::SurfaceCacheCardEntry> m_Cards;
        std::vector<CardRuntimeState> m_CardStates; // Parallel to m_Cards.
        std::unordered_map<uint32_t, EntityDrawRange> m_EntityRanges; // Keyed by entityID.
        geometry::SurfaceCacheAtlasAllocator m_AtlasAllocator{ geometry::kSurfaceCacheAtlasSize };
        std::deque<uint32_t> m_DirtyCardQueue; // Card indices queued for capture by UpdateVisibility(), drained by RecordCapture().

        // Phase 0.3 (PCG roadmap, dynamic Lumen registration):
        std::filesystem::path m_CacheFilePath; // Stashed at Init() so RegisterEntity() can reopen it later.
        std::unordered_map<uint32_t, DynamicEntityGeometry> m_DynamicEntityGeometry; // Keyed by entityID.
        std::unordered_map<uint32_t, std::vector<uint32_t>> m_DynamicEntityCardSlots; // entityID -> its claimed indices into m_Cards/m_CardStates.
        std::vector<uint32_t> m_FreeDynamicCardSlots; // Tombstoned slots (see CardRuntimeState::active) available for reuse.

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
