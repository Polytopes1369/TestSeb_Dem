#pragma once
// Directional ("sun") shadow map for the Lumen-style Surface Cache capture pipeline (see
// renderer::SurfaceCachePass / SurfaceCacheCapture.frag, which sample this pass's depth image to
// shadow-test the sun contribution while shading a captured card).
//
// --- Scope: single map, whole-scene fit, sun only ---
// Entities are static in this engine (see renderer::GlobalSDFPass's own class comment), so the
// scene's world-space AABB is fixed and known once at Init() (the union of every entity's
// geometry::FallbackMeshIndexEntry bounds, read from the same .cache file every other Lumen-style
// pass in this codebase already reads). One orthographic map, fit to a bounding SPHERE around that
// AABB (so any sun direction reuses the same fixed projection size without needing to
// recompute a tight box per-direction), is sufficient for a single bounded demoscene scene -- a
// full cascaded shadow map (multiple maps at increasing texel density near the camera) is out of
// scope here, matching the project's existing pattern of documenting a deliberately staged scope
// rather than building unbounded generality (e.g. GlobalSDFPass.h's own "Scope" section).
// Point lights are NOT shadowed by this pass at all (see renderer::LightingTypes.h's PointLight
// comment) -- a correct point-light shadow needs a 6-face cube map per light, a materially bigger
// feature this pass does not attempt.
//
// --- Why re-read the fallback mesh geometry independently ---
// renderer::SurfaceCachePass already loads the exact same combined Fallback Mesh vertex/index data
// for its own capture draws. This pass duplicates that read (into its own, separate GPU buffers)
// rather than sharing SurfaceCachePass's buffers, matching this codebase's established "deliberately
// self-contained pass" convention (see SurfaceCachePass.cpp / GlobalSDFPass.cpp's own identical
// comment on ReadShaderFile) -- the two passes have entirely different vertex layouts in mind
// (this one only ever reads position; SurfaceCachePass's pipeline also consumes normal/uv) and
// different lifetimes conceptually, so sharing a buffer would couple them for no real benefit.
//
// --- Why one single draw call ---
// Since entities are static (local space == world space, no per-entity transform to apply -- see
// GlobalSDFPass.h's own "Scope" note), every entity's geometry can share the exact same light
// view-projection matrix. RecordCapture() therefore issues exactly one vkCmdDrawIndexed() call
// covering every entity's fallback mesh at once, unlike SurfaceCachePass's own per-card draws
// (which each need a different orthographic projection).

#include <cstdint>
#include <filesystem>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/RenderPass.h"

namespace renderer {

    // Migrated to RenderPass<Derived> (see renderer/vulkan/RenderPass.h): Init()/Shutdown() are
    // inherited. Note InitImpl() still calls the inherited Shutdown() as its own first statement
    // (preserved from the original Init(), which self-reinitializes: calling Init() again on an
    // already-initialized instance releases the old GPU resources before rebuilding) -- InitImpl()
    // re-assigns m_Device/m_Allocator right after that call since Shutdown() clears them.
    class ShadowMapPass : public RenderPass<ShadowMapPass> {
        friend class RenderPass<ShadowMapPass>; // Lets Init() call our private InitImpl().

    public:
        ShadowMapPass() = default;

        ShadowMapPass(const ShadowMapPass&) = delete;
        ShadowMapPass& operator=(const ShadowMapPass&) = delete;

        static constexpr uint32_t kShadowMapSize = 2048u;
        static constexpr VkFormat kShadowFormat = VK_FORMAT_D32_SFLOAT;

        // How far outside the scene's bounding sphere the light's near plane starts, and how much
        // margin is added past the far side -- both expressed as a fraction of the bounding
        // sphere's own radius so they scale with scene size instead of being fixed world units.
        static constexpr float kNearMarginFactor = 0.05f;
        static constexpr float kFarMarginFactor = 2.0f;

        // Reads every fallback mesh's geometry from `cacheFilePath` (see class comment), uploads
        // one combined position-only vertex/index GPU buffer covering every entity, computes the
        // scene's world-space bounding sphere from the fallback-mesh index table's AABBs, allocates
        // the depth image (kShadowMapSize^2, kept in VK_IMAGE_LAYOUT_GENERAL for its entire
        // lifetime -- valid for BOTH a depth attachment write AND a later sampled read, so it never
        // needs to ping-pong layouts between RecordCapture() and a consumer's shadow lookup, exactly
        // like renderer::SurfaceCachePass's own 3 atlas color images), and builds the depth-only capture pipeline (vertex
        // stage only, no fragment shader -- there is nothing to write but depth). Returns false
        // (logged) only on an actual cache-file I/O failure; a scene with zero fallback meshes is
        // valid (Init() succeeds, the scene bounding sphere degenerates to radius 0 at the origin,
        // RecordCapture() is then a depth-clear-only no-geometry no-op).
        // Init(VkDevice, VmaAllocator, VkCommandPool, VkQueue, const std::filesystem::path&) -> bool
        // and Shutdown() are inherited from RenderPass<ShadowMapPass>; see InitImpl() below.

        // Recomputes the light's view-projection matrix from `sunDirection` (fit to the scene's
        // bounding sphere computed at Init(), via mat4::LookAt + mat4::OrthoVulkan -- see the class
        // comment's "single map, whole-scene fit" note) and records one depth-only draw of every
        // entity's fallback mesh into the shadow map. Safe to call every frame with a changing sun
        // direction (e.g. a day/night cycle); the whole recorded workload is one draw call, so
        // there is no per-call budget to spread across frames the way SurfaceCachePass::
        // RecordCapture / GlobalSDFPass::RecordUpdate document for their own, much larger
        // workloads. Must be called into an already-open, caller-owned command buffer (never
        // submits on its own), and before any SurfaceCachePass::RecordCapture call in the same
        // frame that should see this call's shadow-map contents (there is a memory barrier at the
        // end of this call, but no cross-pass synchronization beyond that).
        void RecordCapture(VkCommandBuffer cmd, const maths::vec3& sunDirection);

        VkImageView GetShadowMapView() const { return m_ShadowView; }
        VkSampler GetShadowMapSampler() const { return m_ShadowSampler; }
        // The light view-projection matrix used by the MOST RECENT RecordCapture() call --
        // SurfaceCachePass::UpdateLighting needs this exact matrix to transform a captured
        // texel's world position into the same clip space this pass rendered into.
        const maths::mat4& GetLightViewProj() const { return m_LightViewProj; }

    private:
        bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const std::filesystem::path& cacheFilePath);

        // m_Device / m_Allocator are inherited (protected) from RenderPass<ShadowMapPass>.

        GpuBuffer m_VertexBuffer; // geometry::FallbackVertex[], GPU_ONLY (only .position is read).
        GpuBuffer m_IndexBuffer;  // uint32_t[], GPU_ONLY.
        uint32_t m_TotalIndexCount = 0;

        maths::vec3 m_SceneBoundsCenter{};
        float m_SceneBoundsRadius = 0.0f;

        VkImage m_ShadowImage = VK_NULL_HANDLE;
        VmaAllocation m_ShadowAllocation = VK_NULL_HANDLE;
        VkImageView m_ShadowView = VK_NULL_HANDLE;
        VkSampler m_ShadowSampler = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        maths::mat4 m_LightViewProj;
    };

}
