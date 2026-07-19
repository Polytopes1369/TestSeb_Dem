#pragma once
// Phase 3 (UE5.8 parity roadmap): Virtual Shadow Maps -- replaces this codebase's pre-Phase-3
// renderer::ShadowMapPass (a single whole-scene-fit 2048x2048 orthographic map, fully redrawn every
// frame, sun only -- kept in the repo per feedback_file_deletion_blocked but no longer instantiated
// by renderer::ClusterRenderPipeline). This pass owns:
//   - A 3-level clipmap of Virtual Shadow Maps for the sun (VSM index 0 = finest, 2 = coarsest --
//     radius_L = kSunBaseRadius * 2^L), sharing one physical page pool (renderer::
//     VirtualShadowMapPool) with...
//   - Up to renderer::kMaxPointLights point lights' own 6-face cube of Virtual Shadow Maps (VSM
//     index SHADOW_SUN_LEVEL_COUNT + pointLightSlot*6 + face), built for real per this phase's own
//     "done exactly as in UE 5.8" scope decision even though this demo authors only one
//     point light for verification.
//
// --- Sun clipmap camera re-centering (Feature F14) ---
// Each sun level's projection HALF-EXTENT (radius_L = kSunBaseRadius * 2^L) is still fixed for this
// pass' entire lifetime (computed once at Init(), see m_SunLevelProj), but the window it covers now
// re-centers on the CAMERA every RecordBeginFrame() call -- mirroring renderer::GlobalSDFPass's own
// toroidal clipmap streaming (see that class' own header comment for the general technique) and
// renderer::WorldProbeGridPass's identical camera-snapped grid, adapted from a directly-addressed
// wrapped 3D image (GlobalSDF) to this pass' page-TABLE-indirected pool (VirtualShadowMapPool):
//   - Per level, per frame: project the camera onto the light's own (right, up) basis (the exact
//     s/u vectors maths::mat4::LookAt derives internally -- recomputed explicitly here so this
//     pass' own page-grid axes exactly match the view matrix's real X/Y axes, not an approximation
//     of them), then snap each axis down to whole kSunWindowSnapChunkPages-page-sized world steps
//     (RecordBeginFrame's own snapToPageChunk lambda) -- exactly GlobalSDFPass::
//     EnqueueDirtyRegionsForLevel's snapAxis lambda, just parameterized in PAGES instead of voxels.
//     Depth (the light-direction axis) is NOT snapped -- texel stability only matters for the two
//     axes perpendicular to the light, unlike GlobalSDF's isotropic 3D voxel grid.
//   - A level's local page-table index (localPageIndex, see VirtualShadowMapPool.h) is the WORLD-
//     ANCHORED page-grid coordinate WRAPPED modulo kShadowPagesPerAxis (a true toroidal address,
//     matching shadow_page_table.glsl's ShadowWrapPageCoord) -- NOT the page's RASTER position
//     within the level's current 2048x2048 virtual frustum (which shifts every time the window
//     re-centers). RenderPage()/ClassifyDynamicPages() convert wrapped<->raster using this frame's
//     m_SunLevelWindowStartPage (uploaded to shaders alongside m_SunLevelViewProj so
//     shadow_sun_sampling.glsl's SampleSunShadowVSM performs the identical raster->wrapped
//     conversion before ever touching the page table) -- see RenderPage's own comment for the
//     wrap-coordinate derivation. Because the window's width equals the wrap period exactly
//     (kShadowPagesPerAxis pages both ways), a page-aligned window shift invalidates ONLY the
//     newly-revealed band of local indices along the axis that moved (RecenterSunLevel's own
//     InvalidateSunWindowBand calls) -- the rest of the window's already-resident pages keep
//     resolving to the exact same physical layer/content they held before the shift, no re-render
//     needed. Point-light cube VSMs are entirely unaffected by any of this -- a point light's own
//     frustum is anchored to the LIGHT, not the camera, so it never needs to re-center.
//
// --- Why a page is rendered by redrawing the WHOLE scene, clipped by viewport+scissor ---
// Exactly like the pre-Phase-3 ShadowMapPass: entities are static, so one light view-projection
// (now: one PAGE's virtual viewport into a level/face's FIXED projection, see RenderPage()) is
// valid for the whole combined Fallback Mesh buffer at once. No per-entity or per-page frustum
// culling is performed before issuing the single vkCmdDrawIndexed -- the Fallback Mesh is tiny
// (low hundreds of indices total across every entity, confirmed by ShadowMapPass's own startup
// log), so the wasted vertex-shader work for geometry outside a given page's tiny viewport is
// negligible, not worth a culling pass to avoid.
//
// --- Per-frame call order a caller must follow ---
//   1. RecordBeginFrame(cmd, sunDirection, sceneLights, cameraPosition) -- clears this frame's
//      shadow-page feedback counter, drains + renders up to kMaxPagesRenderedPerFrame page
//      requests reported by LAST frame's consumers (the established one-frame-lag contract, see
//      renderer::GeometryStreamingCoordinator's own class comment for the identical contract on
//      the geometry-streaming side), and uploads this frame's VSM view-projection matrices. Call
//      early (same position ShadowMapPass::RecordCapture used to occupy) -- before any pass that
//      might sample a shadow this frame (renderer::SurfaceCachePass::RecordCapture,
//      renderer::ClusterResolvePass::RecordResolve/RecordResolveBinned).
//   2. (Elsewhere in the frame: every pass that samples a VSM -- SurfaceCacheCapture.frag,
//      ClusterResolve.comp/ClusterResolveBinned.comp -- runs, and may call
//      shadow_feedback.glsl's RequestShadowPageResidency() for a page it needed but found
//      non-resident.)
//   3. RecordEndFrame(cmd) -- captures THIS frame's page-miss reports for step 1 to consume NEXT
//      frame. Call late, after every pass from step 2 has run.

#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "core/EngineConfig.h"
#include "core/EntityData.h" // core::EntityData/EntityTransformCPU -- Feature 1's per-entity material lookup + Feature 2's per-frame overlap test.

#include "core/maths/Maths.h"
#include "renderer/streaming/FeedbackBuffer.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/LightingTypes.h"
#include "renderer/passes/VirtualShadowMapPool.h"
#include "io/StreamingRequestQueue.h"

namespace renderer {

    class VirtualShadowMapPass {
    public:
        VirtualShadowMapPass() = default;

        VirtualShadowMapPass(const VirtualShadowMapPass&) = delete;
        VirtualShadowMapPass& operator=(const VirtualShadowMapPass&) = delete;

        static constexpr uint32_t kSunLevelCount = config::lumen::VSM_SUN_LEVEL_COUNT;
        static inline float kSunBaseRadius = 2.0f; // Level 0's ortho half-extent; level L = kSunBaseRadius * 2^L.
        static constexpr float kSunNearMarginFactor = 0.05f; // Matches ShadowMapPass's own convention.
        static constexpr float kSunFarMarginFactor = 2.0f;
        // Feature F14 (sun clipmap camera re-centering): a level's covered window snaps to whole
        // multiples of this many PAGES (not world units -- page size itself scales per level, see
        // class comment) as the camera moves, so it does not re-center (and thus invalidate a
        // band) on every single frame of continuous camera motion -- exactly
        // GlobalSDFPass::kSnapChunkVoxels's own rationale, parameterized in pages instead of voxels.
        static constexpr uint32_t kSunWindowSnapChunkPages = 2u;

        static constexpr uint32_t kMaxPointLightVSMs = kMaxPointLights * 6u; // 48.
        static constexpr uint32_t kTotalVSMCount = kSunLevelCount + kMaxPointLightVSMs;

        static inline uint32_t kPhysicalPageCapacity = 4096u; // 256 * 128^2 * 4B = 16 MB.
        static inline uint32_t kMaxPagesRenderedPerFrame = 512u; // Generous: the Fallback Mesh is tiny to render.
        // VSM advanced roadmap, Feature 2 (real static-vs-dynamic page invalidation): own frame
        // budget for RecordBeginFrame()'s dynamic-page re-render block -- mirrors
        // kMaxPagesRenderedPerFrame's own exact convention above (a SEPARATE budget, not shared,
        // since the two blocks serve different purposes: one drains newly-requested pages, the
        // other re-renders already-resident ones every single frame).
        static inline uint32_t kMaxDynamicPagesRenderedPerFrame = 512u;
        static constexpr uint32_t kFeedbackCapacity = 256; // Bounded reports/frame, see FeedbackBuffer's own comment.

        // One entity's Fallback Mesh draw range + precomputed per-entity data -- reuses
        // SurfaceCachePass::EntityDrawRange's own {vertexOffset, firstIndex, indexCount} precedent
        // verbatim (see that struct's own comment), extended with what Feature 1 (live transforms),
        // Feature 2 (dynamic invalidation) and Feature 3 (transparency masks) each additionally
        // need, all precomputed ONCE in Init() (entity flags/materialID never change afterward).
        struct EntityDrawRange {
            int32_t vertexOffset = 0;
            uint32_t firstIndex = 0;
            uint32_t indexCount = 0;
            // Rest-pose (bake-time) local-space AABB, same convention as
            // geometry::FallbackMeshIndexEntry::boundsMin/boundsMax (the same space
            // ClusterRaster.vert calls "worldPos" before subtracting EntityTransform::center) --
            // Feature 2's overlap test rotates these 8 corners through the entity's CURRENT
            // EntityTransformCPU every frame before testing NDC overlap against a page.
            maths::vec3 boundsMin{};
            maths::vec3 boundsMax{};
            // Feature 1/3: this entity's TRUE (non-inflated) authored WPO sway amplitude and mask
            // slot, read once from geometry::GetEntityMaterialProperties(entityDataCPU[entityID]
            // .materialID) -- see ShadowMapCaptureAnimated.vert's own header comment for why no
            // GetOriginalWPOAmplitude() un-inflation is needed here, unlike ClusterRaster.vert.
            float maxWPOAmplitude = 0.0f;
            uint32_t maskTextureIndex = 0xFFFFFFFFu; // geometry::kInvalidMaskTextureIndex.
            // Feature 2: true if core::EntityFlags::IsDynamic | HasSplineDeformation |
            // HasEnhancedDisplacement was set on this entity at BuildEntityData() time -- only
            // entities with this true are ever tested against a page's NDC rect; every other
            // entity is implicitly "never covers dynamic content" without a per-frame test.
            bool isDynamicCandidate = false;
            // Feature 2: which deformation-specific conservative bound (if any) to additionally
            // inflate boundsMin/boundsMax by before the overlap test -- see
            // EntityAABBOverlapsPageNDC()'s own .cpp comment.
            bool hasSplineDeformation = false;
            bool hasEnhancedDisplacement = false;
            // Skeletal-animation feature (VSM shadow-capture fix): true for core::EntityFlags::
            // IsSkeletallyAnimated -- folded into isDynamicCandidate below exactly like
            // hasSplineDeformation/hasEnhancedDisplacement already are (this flag was simply never
            // added to that OR-chain when the skeletal-animation feature was introduced, a real
            // integration gap: without it, a page that had already captured the creature once would
            // never be re-rendered again, so ShadowMapCaptureAnimated.vert's own skinning fix would
            // never actually be observed). Also used by EntityAABBOverlapsPageNDC() to inflate
            // boundsMin/boundsMax by kCpuSkeletalMaxDeviation, mirroring the other two flags' own
            // conservative-bound inflation exactly.
            bool isSkeletallyAnimated = false;
        };

        // Reads every fallback mesh's geometry from `cacheFilePath` (mirrors ShadowMapPass::Init's
        // STEP 1 exactly -- same combined position-only vertex/index buffer, same scene-bounding-
        // sphere derivation, PLUS Feature 1's own per-entity EntityDrawRange population), builds the
        // two depth-only capture pipelines (ShadowMapCaptureAnimated.vert unmasked/masked, Feature
        // 1/3 -- see that shader and ShadowMapCaptureMasked.frag's own header comments;
        // ShadowMapCapture.vert itself is left untouched, still used by the separate dead-but-kept
        // ShadowMapPass), and initializes the physical page pool + feedback buffer + request queue.
        // `physicalDevice` is forwarded to VirtualShadowMapPool::Init, which queries the device's
        // actual maxArrayLayers for the pool's D32_SFLOAT image format before sizing it -- see
        // that function's own comment on why kPhysicalPageCapacity cannot just be trusted as-is.
        // `entityTransformBuffer`/`entityDataBuffer`/`wpoGlobalsBuffer`/`splineControlPointsBuffer`
        // are the SAME 4 buffers renderer::ClusterHardwareRasterPass::Init already receives (see
        // that class's own comment for what each one is) -- bound read-only, vertex-stage-only, at
        // this pass's own fresh descriptor set (bindings 0-3, this pass' own numbering, not a reuse
        // of ClusterHardwareRasterPass's binding numbers). `entityDataCPU` (index == meshID, the
        // SAME array renderer::SurfaceCachePass::Init already receives) is read ONCE here, right
        // after every EntityDrawRange's vertexOffset/firstIndex/indexCount is computed, to look up
        // each entity's real materialID -> geometry::GetEntityMaterialProperties() and flags -- see
        // EntityDrawRange's own field comments. `maskImageInfos` is renderer::
        // ProceduralMaskGenerator::GetMaskImageInfos() (Feature 3) -- bound as this descriptor set's
        // binding 4 (fragment-stage-only, COMBINED_IMAGE_SAMPLER, descriptorCount ==
        // maskImageInfos.size()), same pattern ClusterHardwareRasterPass::Init already uses.
        // `boneMatricesBuffer` (Skeletal-animation feature, VSM shadow-capture fix:
        // animation::SkeletalAnimator::GetBoneMatricesBuffer(), the SAME per-frame SSBO
        // ClusterRaster.vert already binds) is bound read-only, vertex-stage-only, at this
        // descriptor set's binding 5 -- see ShadowMapCaptureAnimated.vert's own binding comment.
        bool Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
            VkCommandPool commandPool, VkQueue queue, const std::filesystem::path& cacheFilePath,
            VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer wpoGlobalsBuffer,
            VkBuffer splineControlPointsBuffer, VkBuffer boneMatricesBuffer, const core::EntityData* entityDataCPU,
            const std::vector<VkDescriptorImageInfo>& maskImageInfos);

        void Shutdown();

        // `entityTransformsCPU` (Feature 2, VSM advanced roadmap; index == meshID, the SAME
        // per-frame array renderer::VulkanContext::UpdateEntityRotations() already computes for
        // renderer::SurfaceCachePass's own identical use) drives this call's dynamic-page
        // classification + re-render block -- see the .cpp's own comment for the full derivation.
        // A null pointer is tolerated (skips Feature 2's block entirely, degrading to Feature 1-only
        // behavior) but every real caller in this codebase always has a valid array to pass.
        void RecordBeginFrame(VkCommandBuffer cmd, const maths::vec3& sunDirection,
            const SceneLights& sceneLights, const maths::vec3& cameraPosition,
            const core::EntityTransformCPU* entityTransformsCPU);

        void RecordEndFrame(VkCommandBuffer cmd);

        // Wiring accessors for consumer descriptor sets (SurfaceCachePass, ClusterResolvePass) --
        // see shadow_page_table.glsl / shadow_feedback.glsl / shadow_atlas_sampling.glsl /
        // shadow_sun_sampling.glsl / shadow_point_sampling.glsl for the exact binding contract each
        // one is meant to satisfy.
        VkImageView GetPhysicalAtlasView() const { return m_Pool.GetPhysicalPoolArrayView(); }
        VkSampler GetPhysicalAtlasSampler() const { return m_Pool.GetSampler(); }
        VkBuffer GetPageTableBuffer() const { return m_Pool.GetPageTableBuffer(); }
        VkBuffer GetFeedbackDeviceBuffer() const { return m_Feedback.GetDeviceBuffer(); }
        VkBuffer GetSunLevelsBuffer() const { return m_SunLevelsUBO.Handle(); }
        VkBuffer GetPointFacesBuffer() const { return m_PointFacesUBO.Handle(); }

        // For the DEBUG_VIEW_SHADOW_CASCADES visualization: the sun level view-proj matrices
        // uploaded THIS frame (CPU-side copy, so a debug shader's own descriptor set can be fed the
        // same values without re-deriving them) and how many resident pages the pool currently
        // holds (a coarse but useful "is streaming keeping up" signal).
        const maths::mat4& GetSunLevelViewProj(uint32_t level) const { return m_SunLevelViewProj[level]; }
        uint32_t GetResidentPageCount() const { return m_Pool.GetResidentPageCount(); }
        uint32_t GetPhysicalPageCapacity() const { return m_Pool.GetPhysicalCapacity(); }

    private:
        // Feature 1 (live per-entity transforms): renders EVERY entity's Fallback Mesh range into
        // this page (one vkCmdDrawIndexed per entity, opaque or masked pipeline selected per
        // entity's own precomputed maskTextureIndex -- Feature 3), instead of the old single
        // monolithic draw. Still "redraw the whole (tiny) scene per page, clipped by viewport/
        // scissor" -- see this class' own header comment -- just per-entity now instead of merged
        // into one draw call, which is what actually lets each entity's CURRENT (deformed/rotated)
        // geometry reach the shadow page instead of a frozen rest-pose snapshot.
        void RenderPage(VkCommandBuffer cmd, uint32_t vsmIndex, uint32_t localPageIndex, uint32_t physicalLayer,
            const maths::mat4& viewProj);

        // Feature 2 (real static-vs-dynamic page invalidation): re-tests EVERY currently-resident
        // page (any VSM -- sun cascade or point light face) against every dynamic-candidate
        // entity's CURRENT world-space AABB, writing the verdict into m_Pool via
        // SetPageCoversDynamicContent(). Called once per RecordBeginFrame() call, before that
        // method's own dynamic-page re-render block reads m_Pool.GetResidentDynamicPageIDs(). See
        // the .cpp for the full NDC-overlap derivation.
        void ClassifyDynamicPages(const core::EntityTransformCPU* entityTransformsCPU);

        // True if `range`'s current (rotated about its own EntityTransformCPU pivot, deformation-
        // inflated) world-space AABB overlaps the page NDC sub-rectangle [ndcXMin,ndcXMax] x
        // [ndcYMin,ndcYMax] under `viewProj` -- see the .cpp definition for the exact derivation
        // (mirrors RenderPage's own virtual-viewport math in reverse to recover a page's NDC
        // sub-rect, and PostProcessPass.cpp's own manual clip-space projection pattern, since
        // maths::mat4 has no vec4 type/operator* to lean on -- see that file's own comment).
        bool EntityAABBOverlapsPageNDC(const EntityDrawRange& range, const core::EntityTransformCPU& xform,
            const maths::mat4& viewProj, float ndcXMin, float ndcXMax, float ndcYMin, float ndcYMax) const;

        // Feature F14 (sun clipmap camera re-centering): recomputes level `level`'s chunk-snapped
        // window center from `cameraPosition` (projected onto `lightRight`/`lightUpForPaging` --
        // see class comment on why the Y-axis projection direction is pre-flipped to match
        // OrthoVulkan's own NDC.y sign convention), invalidates (via m_Pool.InvalidatePage) exactly
        // the newly-revealed band of local page indices if the window moved since the last call,
        // recomputes this level's m_SunLevelViewProj[level] centered on the new window, and updates
        // m_SunLevelWindowStartPage[level] for both RenderPage()/ClassifyDynamicPages() and this
        // frame's GPU upload. Returns how many resident pages were invalidated this call (purely
        // for RecordBeginFrame's own validation logging).
        uint32_t RecenterSunLevel(VkCommandBuffer cmd, uint32_t level, const maths::vec3& cameraPosition,
            const maths::vec3& lightDir, const maths::vec3& lightRight, const maths::vec3& lightUpForPaging,
            const maths::vec3& up);

        // Feature F14: invalidates every (vsmIndex=level, wrapped local page index) whose moved-axis
        // wrapped coordinate falls in [bandStartWorldPage, bandStartWorldPage + bandCount), for
        // EVERY local index along the other (untouched) axis -- the exact per-axis slab shape
        // GlobalSDFPass::EnqueueDirtyRegionsForLevel enqueues as a dirty compositing region, here
        // applied directly as an eviction (VSM pages regenerate lazily via the existing feedback/
        // miss path -- no eager refill needed, unlike GlobalSDF's clipmap texels). `movedAxis` is 0
        // for the light-right axis, 1 for the light-up(-for-paging) axis. Returns how many
        // (already-resident) pages were actually invalidated.
        uint32_t InvalidateSunWindowBand(VkCommandBuffer cmd, uint32_t level, int32_t bandStartWorldPage,
            int32_t bandCount, int32_t movedAxis);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        GpuBuffer m_VertexBuffer; // geometry::FallbackVertex[], GPU_ONLY (position + uv both read now, Feature 1/3) -- own copy, see class comment.
        GpuBuffer m_IndexBuffer;  // uint32_t[], GPU_ONLY.
        uint32_t m_TotalIndexCount = 0;

        // Feature 1: one entry per entity present in the Fallback Mesh table -- see EntityDrawRange's
        // own comment. Keyed by entityID (== meshID).
        std::unordered_map<uint32_t, EntityDrawRange> m_EntityRanges;

        maths::vec3 m_SceneBoundsCenter{};
        float m_SceneBoundsRadius = 0.0f;

        VirtualShadowMapPool m_Pool;
        FeedbackBuffer m_Feedback;
        geometry::StreamingRequestQueue m_RequestQueue; // Fully generic (opaque uint32 IDs) -- reused as-is, own instance.

        // Feature 1/3: one descriptor set (EntityTransformBuffer/EntityDataBuffer/WPOGlobalsUBO/
        // SplineControlPointsSSBO at bindings 0-3, all vertex-stage-only, + the bindless mask array
        // at binding 4, fragment-stage-only) shared by both pipelines below -- mirrors
        // ClusterHardwareRasterPass::Init's own layout pattern (see that class's comment), written
        // once at Init(), never updated again (only the buffers' own CONTENTS change per frame,
        // exactly like ClusterHardwareRasterPass's identical descriptor set).
        VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;       // ShadowMapCaptureAnimated.vert only -- unmasked entities (Feature 1).
        VkPipeline m_MaskedPipeline = VK_NULL_HANDLE; // + ShadowMapCaptureMasked.frag -- masked entities (Feature 3).

        // This frame's VSM view-projection matrices, recomputed every RecordBeginFrame() call --
        // consumed both by RenderPage() (for whatever pages get (re)rendered this frame) and
        // uploaded verbatim into m_SunLevelsUBO/m_PointFacesUBO for every consumer shader to read.
        maths::mat4 m_SunLevelProj[kSunLevelCount]{};      // Fixed at Init() (radius never changes, see class comment).
        maths::mat4 m_SunLevelViewProj[kSunLevelCount]{};  // proj * view(sunDirection, camera-recentered window), recomputed every frame.
        maths::mat4 m_PointFaceViewProj[kMaxPointLightVSMs]{}; // Recomputed every frame from SceneLights::pointLights.
        uint32_t m_ActivePointLightCount = 0;

        // Feature F14 (sun clipmap camera re-centering): one level's chunk-snapped window-center
        // page-grid coordinate (world-anchored, see class comment) -- persists across frames so
        // RecenterSunLevel() can detect a shift and invalidate only the newly-revealed band.
        struct SunClipmapWindow {
            bool hasValidWindow = false;
            int32_t centerPage[2] = { 0, 0 }; // [0]=light-right axis, [1]=light-up(-for-paging) axis.
        };
        SunClipmapWindow m_SunWindows[kSunLevelCount]{};

        // Feature F14: this frame's window-start page-grid coordinate per sun level (== centerPage
        // - kShadowPagesPerAxis/2), i.e. the world-page-grid coordinate that raster index 0 along
        // that axis currently resolves to -- consumed by RenderPage()/ClassifyDynamicPages() to
        // convert a wrapped local page index back to this frame's raster position, AND uploaded
        // into m_SunLevelsUBO so shadow_sun_sampling.glsl's SampleSunShadowVSM can perform the
        // identical raster->wrapped conversion GPU-side before ever touching the page table.
        int32_t m_SunLevelWindowStartPage[kSunLevelCount][2]{};

        GpuBuffer m_SunLevelsUBO;   // See SunLevelsUBOData (VirtualShadowMapPass.cpp) for the exact std140-mirrored layout uploaded here.
        GpuBuffer m_PointFacesUBO; // mat4[kMaxPointLightVSMs], CPU_TO_GPU mapped.
    };

}
