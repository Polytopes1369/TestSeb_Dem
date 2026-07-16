#pragma once
// Deferred material-resolve pass: the decoupled second half of the Visibility Buffer pipeline --
// see src/shaders/src/Renderer/ClusterResolve.comp for the compute shader this class drives, which
// reconciles renderer::ClusterHardwareRasterPass's (ClusterID/TriangleID + depth) output against
// renderer::ClusterSoftwareRasterPass's (atomic depth+visibility) output per pixel, reconstructs
// each winning triangle's barycentric coordinates from the pixel position and the triangle's own
// screen-space plane, and shades the result into an owned output color image.
//
// This class owns none of its three VisBuffer inputs (they belong to the two rasterization passes
// that filled them) -- only the output color image, the view-params UBO, and the depth sampler.
// Exactly like every other piece of this Nanite-style pipeline, this is a self-contained building
// block -- Init()/Shutdown()/RecordResolve() only -- not wired into VulkanContext/main.cpp by this
// change; in particular, GetOutputColorImage() is not blitted to the swapchain here (a future
// integration step's job, exactly like the gap already documented when the flat/legacy draw path
// was first converted to write a VisBuffer instead of shading directly).
//
// --- Per-frame sequence a caller must record, in order ---
//   1. Caller: transition the hardware ClusterID/TriangleID VisBuffer images (whatever
//      renderer::VulkanContext's kVisBufferFormat images -- or renderer::ClusterHardwareRasterPass
//      -- rendered into as color attachments) from VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL to
//      VK_IMAGE_LAYOUT_GENERAL (this class binds them as plain storage images, whose descriptor
//      writes are fixed at Init() time to imageLayout = GENERAL -- see Init()'s doc comment), and
//      the hardware depth image to VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL. Also make sure
//      renderer::ClusterSoftwareRasterPass::RecordRaster()'s own barrier has already been recorded
//      this frame -- this pass reads all three VisBuffer sources as of whatever they contain at
//      the point RecordResolve() is called.
//   2. RecordResolve(cmd, viewProj) -- must use the exact same view-projection matrix both
//      rasterization passes used this frame, so the re-projected triangle used for barycentric
//      reconstruction exactly matches the one that determined pixel coverage originally. Ends with
//      the barrier making GetOutputColorImage() visible to a later sampled read or blit.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/GpuBuffer.h"

namespace renderer {

    class ClusterResolvePass {
    public:
        ClusterResolvePass() = default;

        ClusterResolvePass(const ClusterResolvePass&) = delete;
        ClusterResolvePass& operator=(const ClusterResolvePass&) = delete;

        // RGBA8: this codebase's simplest, most broadly supported color-attachment-and-storage-
        // capable format (mirrors why VulkanContext::kVisBufferFormat picks a mandatory-support
        // format over an exotic one), sufficient for the procedural, non-HDR shading
        // ClusterResolve.comp currently performs (see that shader's class comment on why no real
        // material/texture system exists yet).
        static constexpr VkFormat kOutputColorFormat = VK_FORMAT_R8G8B8A8_UNORM;

        // Allocates the output color image (sized to `renderExtent`, transitioned once to
        // VK_IMAGE_LAYOUT_GENERAL via a blocking one-time submit, mirroring HZBPass::Init /
        // renderer::ClusterSoftwareRasterPass::Init's own one-shot transition pattern), the
        // nearest-filter depth sampler, the view-params UBO, and the compute pipeline/descriptor
        // set. None of `clusterMetadataBuffer` / `compressedPhysicalPoolBuffer` /
        // `hwClusterIDView` / `hwTriangleIDView` / `hwDepthView` / `swVisBufferAtomicView` are
        // owned here -- they must all come from the same frame's
        // renderer::ClusterHardwareRasterPass / ClusterSoftwareRasterPass /
        // GpuGeometryPagePool / HZBPass-driven depth image instances. `hwClusterIDView` /
        // `hwTriangleIDView`'s descriptor writes are fixed here at Init() time to
        // imageLayout = VK_IMAGE_LAYOUT_GENERAL and `hwDepthView`'s to
        // VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL -- the caller must guarantee those images
        // are actually in those exact layouts every time RecordResolve() executes (see the class
        // comment's per-frame sequence), since a descriptor's declared imageLayout is fixed at
        // write time, not re-validated per use.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
            VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
            VkImageView hwClusterIDView, VkImageView hwTriangleIDView, VkImageView hwDepthView,
            VkImageView swVisBufferAtomicView);

        void Shutdown();

        // Uploads `viewProj` into the view-params UBO and dispatches one invocation per output
        // pixel (local_size_x/y = 8). Ends with the barrier making GetOutputColorImage() visible to
        // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT / VK_ACCESS_2_SHADER_SAMPLED_READ_BIT (a future
        // sampled read) and VK_PIPELINE_STAGE_2_COPY_BIT / VK_ACCESS_2_TRANSFER_READ_BIT (a future
        // blit to the swapchain) -- whichever the caller ends up using.
        void RecordResolve(VkCommandBuffer cmd, const maths::mat4& viewProj);

        VkImage GetOutputColorImage() const { return m_OutputColorImage; }
        VkImageView GetOutputColorView() const { return m_OutputColorView; }

    private:
        static constexpr uint32_t kWorkgroupSize = 8; // Matches ClusterResolve.comp's local_size_x/y.

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only to vmaDestroyImage() in Shutdown().
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_OutputColorImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputColorAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputColorView = VK_NULL_HANDLE;

        VkSampler m_DepthSampler = VK_NULL_HANDLE; // Nearest filtering, matching HZBPass's own depth-sampling convention.

        GpuBuffer m_ViewParamsBuffer; // ResolveViewParamsUBO, std140, GPU_ONLY.

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
