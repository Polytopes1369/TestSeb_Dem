#pragma once
// Lumen-style two-tier SDF ray march (see geometry::EntityBVH.h's own "why this exists" note): a
// full-screen compute pass that, for every pixel, sphere-traces the coarse Global SDF clipmap
// (renderer::GlobalSDFPass) in large steps until it is near a surface, then narrows down to the
// handful of per-entity Mesh SDFs actually nearby (via a CPU-built geometry::EntityBVH, traversed
// on the GPU with a small fixed stack) and sphere-traces THOSE at full precision to find the exact
// hit point, shading it (or a sky gradient on a miss) into a full-screen debug-visualization image.
//
// --- Why this pass builds its OWN per-entity Mesh SDF textures, not renderer::GlobalSDFPass's ---
// GlobalSDFPass decodes each entity's Mesh SDF at a DELIBERATELY coarse kEntitySDFResolution (24^3,
// "enough to drive cone tracing's coarse empty-space skipping, not fine per-object surface detail"
// -- see that class's own comment) purely so ITS clipmap compositing has something cheap to
// min-combine. This pass's whole point is the opposite: precise local sampling once the coarse
// clipmap says a surface is near. So it re-reads the same fallback-mesh geometry and re-decodes each
// entity's Mesh SDF itself, at the surface-detail-oriented geometry::kMeshSDFResolution default --
// matching this codebase's established "self-contained pass, duplicate rather than share" convention
// (see renderer::ShadowMapPass's own class comment on why it re-reads SurfaceCachePass's geometry
// too) -- rather than depend on GlobalSDFPass's coarser textures and blur the two tiers together.
//
// --- Fixed-size, bindless-indexed entity sampler array ---
// SDFRayMarch.comp declares a plain, compile-time-sized `sampler3D uEntitySDFs[kMaxEntitySDFs]`
// array (ordinary core Vulkan 1.0 descriptor array, not VK_EXT_descriptor_indexing's runtime-sized
// arrays), with every slot written at Init() (a real entity's view for slots < the entity count, a
// shared 1x1x1 "always far" placeholder texture for the rest) so validation never sees an
// unwritten descriptor. Which slot a given ray march step samples is still a genuinely
// per-invocation decision (the BVH traversal result), so the shader indexes it with
// nonuniformEXT() (GL_EXT_nonuniform_qualifier) -- see VulkanContext.cpp's
// shaderSampledImageArrayNonUniformIndexing feature enable, which this requires regardless of the
// array itself being fixed-size. A scene with more entities than kMaxEntitySDFs truncates (logged)
// rather than growing the array at runtime -- a documented scope limit, matching kMaxPointLights/
// kLevelCount's own fixed-cap convention elsewhere in this codebase.
//
// --- Coupling to renderer::GlobalSDFPass ---
// Unlike the geometry-loading duplication above, this pass DOES directly depend on GlobalSDFPass's
// public accessors (GetClipmapView/GetLevelVoxelSize/GetLevelSnappedCenterVoxel) -- that class's own
// header comment already documents itself as existing for exactly this: "A future GI cone-tracing
// pass walks these clipmaps directly, skipping empty space in large strides." SetGlobalSDFViews()
// binds its 4 level views (with THIS pass's own nearest-filter sampler, not GlobalSDFPass's -- see
// SetGlobalSDFViews()'s own comment); RecordRayMarch() reads its per-level voxel size/window every
// call, since those can change every frame as the camera moves.
//
// debug-only (whole file compiled out in Release) ---
// This pass's entire output is, by its own class comment, "a full-screen debug-visualization
// image" -- it never feeds production lighting, unlike renderer::SurfaceCachePass/GlobalSDFPass
// (real GI infrastructure that stays compiled in Release). Matches CLAUDE.md's build-separation
// rule for "visualization modes (Lumen/Nanite)" and this codebase's existing convention for
// debug-only passes (see renderer::debug::ClusterTriangleStatsPass's own identical guard).

#ifndef NDEBUG

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "geometry/EntityBVH.h"
#include "renderer/passes/GlobalSDFPass.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/RenderPass.h"

namespace renderer {

    // Migrated to RenderPass<Derived> (see renderer/vulkan/RenderPass.h): Init()/Shutdown() are
    // inherited. Note InitImpl() still calls the inherited Shutdown() as its own first statement
    // (self-reinit, same pattern/reason as ShadowMapPass's own migration comment). One pool
    // allocates 2 DIFFERENT-layout sets (entity + clipmap, maxSets=2) -- same reasoning as
    // ReflectionPass/SurfaceCacheTraceContext: left as raw Vulkan calls wrapped in
    // RegisterResource() rather than extending the single-set helper for this one caller.
    class SDFRayMarchPass : public RenderPass<SDFRayMarchPass> {
        friend class RenderPass<SDFRayMarchPass>; // Lets Init() call our private InitImpl().

    public:
        SDFRayMarchPass() = default;

        SDFRayMarchPass(const SDFRayMarchPass&) = delete;
        SDFRayMarchPass& operator=(const SDFRayMarchPass&) = delete;

        // Upper bound on entities this pass can sample per ray march -- see the class comment's
        // "fixed-size entity sampler array" note.
        static constexpr uint32_t kMaxEntitySDFs = 128u;

        static constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;

        // Reads every fallback mesh's geometry from `cacheFilePath`, builds one CPU-side
        // geometry::MeshSDF per entity at geometry::kMeshSDFResolution (see class comment), decodes
        // each into its own plain filterable 3D texture, builds a geometry::EntityBVH over the
        // SAME (post-filtering) entity list so BVH leaf indices, the entity-parameter SSBO, and the
        // entity sampler array all share one consistent slot ordering, and allocates the
        // (outputWidth x outputHeight) RGBA8 output storage image (kept in VK_IMAGE_LAYOUT_GENERAL
        // for its entire lifetime, both a valid storage-image layout and a valid sampled-image
        // layout for whatever later pass displays it). Returns false (logged) only on an actual
        // cache-file I/O failure; a scene with zero (or more than kMaxEntitySDFs, truncated and
        // logged) fallback meshes is valid.
        // Init(VkDevice, VmaAllocator, VkCommandPool, VkQueue, const std::filesystem::path&,
        // uint32_t, uint32_t) -> bool and Shutdown() are inherited from
        // RenderPass<SDFRayMarchPass>; see InitImpl() below.

        // Binds `globalSDF`'s 4 clipmap level views into this pass's own descriptor set, using this
        // pass's OWN nearest-filter sampler (NOT GlobalSDFPass::GetEntitySDFSampler-equivalent --
        // GlobalSDFPass does not expose one; more importantly, nearest avoids blending across a
        // toroidally-wrapped clipmap's wrap seam, which a linear filter would do incorrectly -- see
        // SDFRayMarch.comp's own SampleClipmap comment). Must be called exactly once after
        // `globalSDF.Init()`, before the first RecordRayMarch() call; the 4 views themselves are
        // never recreated after GlobalSDFPass::Init(), so this binding never needs to be refreshed.
        void SetGlobalSDFViews(const GlobalSDFPass& globalSDF);

        // Atmos weather system, Subtask 2 (atmos_integration_plan.md): binds renderer::AtmosSkyPass's
        // Sky-View LUT view/sampler into this pass's own descriptor set (set 1, binding 2) -- same
        // "deferred, called once after the producer pass' own Init()" convention as
        // SetGlobalSDFViews() above, since renderer::AtmosSkyPass is Init'd independently. Must be
        // called exactly once before the first RecordRayMarch() call.
        void SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler);

        // Ray marches one full frame into the output image: `cameraPosition`/`cameraForward`/
        // `cameraUp` describe the camera (same convention as renderer::SurfaceCachePass::
        // UpdateVisibility: cameraUp need not be re-orthonormalized, only non-parallel to
        // cameraForward), `fovYRadians`/`aspectRatio` set up this frame's per-pixel ray
        // reconstruction, and [nearZ, farZ] bound how far a ray marches before being treated as a
        // sky miss. Reads `globalSDF`'s CURRENT per-level voxel size and streamed window (NOT
        // cached from SetGlobalSDFViews() -- these change every frame as the camera moves) to keep
        // the shader's toroidal-wrap addressing in agreement with whatever GlobalSDFPass most
        // recently wrote. Must be called into an already-open, caller-owned command buffer (never
        // submits on its own); ends with a VkMemoryBarrier2 making the output image's writes visible
        // to a later sampled/compute read. `coarseOnly` (false = normal two-tier march, matching
        // DEBUG_VIEW_LUMEN's full result; true = DEBUG_VIEW_GLOBAL_SDF's own coarse-clipmap-only
        // variant, skipping the fine per-entity BVH refinement tier entirely) makes this pass's
        // output visually distinct between the two debug views that both blit-swap to it -- see
        // renderer::ClusterRenderPipeline::RecordFrame's own [14] blit-source-swap comment.
        // `sunDirectionWorld` (Atmos weather system, Subtask 2): threaded into the Sky-View LUT's
        // sun-relative azimuth mapping on a miss -- see SDFRayMarch.comp's own GetSunDirection().
        void RecordRayMarch(VkCommandBuffer cmd, const GlobalSDFPass& globalSDF,
            const maths::vec3& cameraPosition, const maths::vec3& cameraForward, const maths::vec3& cameraUp,
            float fovYRadians, float aspectRatio, float nearZ, float farZ, bool coarseOnly,
            const maths::vec3& sunDirectionWorld);

        uint32_t GetOutputWidth() const { return m_OutputWidth; }
        uint32_t GetOutputHeight() const { return m_OutputHeight; }
        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        struct EntitySDF {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            maths::vec3 volumeMin{};
            float voxelSize = 0.0f;
            uint32_t resolution = 0;
            uint32_t entityID = 0;
        };

        bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const std::filesystem::path& cacheFilePath, uint32_t outputWidth, uint32_t outputHeight);

        // m_Device / m_Allocator are inherited (protected) from RenderPass<SDFRayMarchPass>.

        uint32_t m_OutputWidth = 0;
        uint32_t m_OutputHeight = 0;
        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        std::vector<EntitySDF> m_Entities; // Slot order == geometry::EntityBVH leaf-index order.
        VkImage m_PlaceholderImage = VK_NULL_HANDLE; // 1x1x1, value == GlobalSDFPass::kFarValue, bound to every unused sampler-array slot.
        VmaAllocation m_PlaceholderAllocation = VK_NULL_HANDLE;
        VkImageView m_PlaceholderView = VK_NULL_HANDLE;
        VkSampler m_EntitySampler = VK_NULL_HANDLE; // Linear -- precise local refine wants a smooth SDF gradient.
        VkSampler m_ClipmapSampler = VK_NULL_HANDLE; // Nearest -- see SetGlobalSDFViews()'s own comment.

        GpuBuffer m_BVHNodeBuffer;         // geometry::BVHNode[], GPU_ONLY (>= 1 element even if empty, see .cpp).
        GpuBuffer m_LeafEntityIndexBuffer; // uint32_t[], GPU_ONLY (>= 1 element even if empty).
        GpuBuffer m_EntityParamsBuffer;    // EntitySDFParamsGPU[] (SDFRayMarchPass.cpp), GPU_ONLY (>= 1 element even if empty).
        int32_t m_RootNodeIndex = -1;

        VkDescriptorSetLayout m_EntitySetLayout = VK_NULL_HANDLE;  // set 0: BVH + entity SDFs.
        VkDescriptorSetLayout m_ClipmapSetLayout = VK_NULL_HANDLE; // set 1: Global SDF clipmaps + output image.
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_EntitySet = VK_NULL_HANDLE;
        VkDescriptorSet m_ClipmapSet = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}

#endif // NDEBUG
