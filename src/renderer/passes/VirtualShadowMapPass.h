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
// --- Why the sun's clipmap windows never move (deliberate simplification vs. real UE5.8) ---
// A real VSM clipmap re-centers its levels on the CAMERA every frame (renderer::GlobalSDFPass's own
// SDF clipmap does exactly this, toroidally). This pass does not: this engine's scene is static and
// bounded (a single ~6.4m-radius sphere, see the scene-bounds read in Init()) and the demo camera
// only ever orbits it, so a level whose FIXED radius already covers the whole bounding sphere never
// needs to move to keep up with the camera -- there is nothing outside it to ever come into view.
// Each level's projection is therefore computed ONCE at Init() (only the VIEW half, which depends
// on the sun's current direction, is recomputed every RecordBeginFrame() call). This is a
// deliberate, documented scope trim for THIS bounded demo scene, not an oversight -- see this
// phase's own plan for the full rationale, and Phase 4 (dynamic scenes) as the natural place to
// revisit camera-following if this engine ever grows an unbounded/open scene.
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
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "core/EngineConfig.h"

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

        static constexpr uint32_t kMaxPointLightVSMs = kMaxPointLights * 6u; // 48.
        static constexpr uint32_t kTotalVSMCount = kSunLevelCount + kMaxPointLightVSMs;

        static inline uint32_t kPhysicalPageCapacity = 4096u; // 256 * 128^2 * 4B = 16 MB.
        static inline uint32_t kMaxPagesRenderedPerFrame = 512u; // Generous: the Fallback Mesh is tiny to render.
        static constexpr uint32_t kFeedbackCapacity = 256; // Bounded reports/frame, see FeedbackBuffer's own comment.

        // Reads every fallback mesh's geometry from `cacheFilePath` (mirrors ShadowMapPass::Init's
        // STEP 1 exactly -- same combined position-only vertex/index buffer, same scene-bounding-
        // sphere derivation), builds the depth-only capture pipeline (reuses ShadowMapCapture.vert
        // unchanged), and initializes the physical page pool + feedback buffer + request queue.
        // `physicalDevice` is forwarded to VirtualShadowMapPool::Init, which queries the device's
        // actual maxArrayLayers for the pool's D32_SFLOAT image format before sizing it -- see
        // that function's own comment on why kPhysicalPageCapacity cannot just be trusted as-is.
        bool Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
            VkCommandPool commandPool, VkQueue queue, const std::filesystem::path& cacheFilePath);

        void Shutdown();

        void RecordBeginFrame(VkCommandBuffer cmd, const maths::vec3& sunDirection,
            const SceneLights& sceneLights, const maths::vec3& cameraPosition);

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
        void RenderPage(VkCommandBuffer cmd, uint32_t vsmIndex, uint32_t localPageIndex, uint32_t physicalLayer,
            const maths::mat4& viewProj);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        GpuBuffer m_VertexBuffer; // geometry::FallbackVertex[], GPU_ONLY (only .position read) -- own copy, see class comment.
        GpuBuffer m_IndexBuffer;  // uint32_t[], GPU_ONLY.
        uint32_t m_TotalIndexCount = 0;

        maths::vec3 m_SceneBoundsCenter{};
        float m_SceneBoundsRadius = 0.0f;

        VirtualShadowMapPool m_Pool;
        FeedbackBuffer m_Feedback;
        geometry::StreamingRequestQueue m_RequestQueue; // Fully generic (opaque uint32 IDs) -- reused as-is, own instance.

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        // This frame's VSM view-projection matrices, recomputed every RecordBeginFrame() call --
        // consumed both by RenderPage() (for whatever pages get (re)rendered this frame) and
        // uploaded verbatim into m_SunLevelsUBO/m_PointFacesUBO for every consumer shader to read.
        maths::mat4 m_SunLevelProj[kSunLevelCount]{};      // Fixed at Init() (radius never changes, see class comment).
        maths::mat4 m_SunLevelViewProj[kSunLevelCount]{};  // proj * view(sunDirection), recomputed every frame.
        maths::mat4 m_PointFaceViewProj[kMaxPointLightVSMs]{}; // Recomputed every frame from SceneLights::pointLights.
        uint32_t m_ActivePointLightCount = 0;

        GpuBuffer m_SunLevelsUBO;   // mat4[kSunLevelCount], CPU_TO_GPU mapped.
        GpuBuffer m_PointFacesUBO; // mat4[kMaxPointLightVSMs], CPU_TO_GPU mapped.
    };

}
