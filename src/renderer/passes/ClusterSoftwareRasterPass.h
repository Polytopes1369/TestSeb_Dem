#pragma once
// Ultra-fast software micro-triangle rasterizer -- the Nanite-style dual rasterization path's
// compute half (see renderer::ClusterHardwareRasterPass for the fixed-function half it complements,
// and renderer::ClusterOcclusionCullingPass::GetSoftwareClusterListBuffer(), whose entries this
// class exclusively consumes). See src/shaders/src/Culling/ClusterSoftwareRaster.comp for the
// per-triangle half-space rasterization + atomic depth-test logic this class drives.
//
// --- Why a dedicated 64-bit atomic image ---
// Hardware rasterization gets "nearest triangle wins" for free from the fixed-function depth test;
// a software rasterizer has to implement that itself, and the standard, correct way to do so when
// many threads (potentially from many different triangles/clusters) may cover the same pixel is a
// single atomic read-modify-write per pixel combining depth and the visibility payload into one
// word (see ClusterSoftwareRaster.comp's own comment for the exact packing). This genuinely
// requires a 64-bit atomic (imageAtomicMax), which needs VK_FORMAT_R64_UINT as a *storage image*
// plus the VK_EXT_shader_image_atomic_int64 device extension / shaderImageInt64Atomics feature
// (enabled in VulkanContext::CreateLogicalDevice) -- a deliberately narrower, better-justified
// exception to the "avoid R64_UINT, prefer two R32_UINT" choice made for the *hardware* path's
// VisBuffer (VulkanContext::kVisBufferFormat), which never needs atomics since the fixed-function
// depth test already serializes writes for it.
//
// --- Per-frame sequence a caller must record, in order ---
//   1. RecordClear(cmd) -- resets the atomic image to 0 (the "nothing rasterized here" sentinel)
//      via a dedicated compute dispatch (ClearVisBufferAtomic.comp) -- not vkCmdClearColorImage,
//      whose VkClearColorValue is only well-defined for <= 32-bit image channels.
//   2. RecordRaster(cmd, viewProj) -- uploads this frame's view-projection matrix, builds this
//      dispatch's indirect args from GetSoftwareClusterListBuffer()'s own atomic count (no CPU
//      round-trip, see BuildDispatchIndirectArgs.comp), and rasterizes every listed cluster's
//      triangles via one atomicMax per covered pixel.
//   3. Caller (renderer::ClusterResolvePass): reads GetVisBufferAtomicView() alongside the
//      hardware path's own VisBuffer + depth buffer to decide, per pixel, which rasterization path
//      actually won and shade accordingly.
//
// Exactly like every other piece of this Nanite-style pipeline, this class is a self-contained
// building block -- Init()/Shutdown()/per-frame Record*() only; renderer::ClusterRenderPipeline is
// what actually wires it into the frame loop main.cpp drives.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class ClusterSoftwareRasterPass {
    public:
        ClusterSoftwareRasterPass() = default;

        ClusterSoftwareRasterPass(const ClusterSoftwareRasterPass&) = delete;
        ClusterSoftwareRasterPass& operator=(const ClusterSoftwareRasterPass&) = delete;

        // R64_UINT: see the class comment for why this format (over the hardware path's two
        // R32_UINT images) is required here.
        static constexpr VkFormat kVisBufferAtomicFormat = VK_FORMAT_R64_UINT;

        // Allocates the atomic VisBuffer image (sized to `renderExtent`, transitioned once to
        // VK_IMAGE_LAYOUT_GENERAL via a blocking one-time submit on `queue`/`commandPool`,
        // mirroring HZBPass::Init's own one-shot transition pattern -- the image then stays in
        // GENERAL for its entire lifetime, since it is only ever touched by compute shaders), the
        // view-params UBO, the dispatch-args buffer, and all 3 compute pipelines (clear, raster,
        // build-dispatch-args). `clusterMetadataBuffer`/`compressedPhysicalPoolBuffer`/
        // `softwareClusterListBuffer`/`wpoGlobalsBuffer` are NOT owned here -- they must come from
        // the same renderer::ClusterOcclusionCullingPass instance/frame whose
        // GetClusterMetadataBuffer() / GetSoftwareClusterListBuffer() populate them
        // (`compressedPhysicalPoolBuffer` is renderer::GpuGeometryPagePool::GetPhysicalPoolBuffer(),
        // `wpoGlobalsBuffer` is renderer::ClusterRenderPipeline's own WPOGlobalsUBO -- see
        // wpo_deformation.glsl), exactly like renderer::ClusterHardwareRasterPass::Init's own
        // external-buffer parameters. `maskImageInfos` is renderer::ProceduralMaskGenerator::
        // GetMaskImageInfos(), bound as binding 6 for ClusterSoftwareRaster.comp's opacity-mask
        // cutout (mask_sampling.glsl).
        // `softwareClusterListOpaqueBuffer` is renderer::ClusterOcclusionCullingPass::
        // GetSoftwareClusterListOpaqueBuffer() -- the opaque-only counterpart list, consumed by a
        // second, mask-sampling-free compute pipeline (ClusterSoftwareRasterOpaque.comp) this
        // Init() also builds; see RecordRaster().
        // `splineControlPointsBuffer` is renderer::ClusterRenderPipeline's own
        // m_SplineControlPointsBuffer (see ClusterHardwareRasterPass::Init's identical parameter
        // comment) -- bound read-only at the first free slot past each set's existing bindings:
        // binding 9 for the masked raster set (ClusterSoftwareRaster.comp), binding 8 for the
        // opaque raster set (ClusterSoftwareRasterOpaque.comp).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
            VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer, VkBuffer softwareClusterListBuffer,
            VkBuffer softwareClusterListOpaqueBuffer, VkBuffer wpoGlobalsBuffer, const std::vector<VkDescriptorImageInfo>& maskImageInfos,
            VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer splineControlPointsBuffer);

        void Shutdown();

        // Resets the atomic VisBuffer to 0 via ClearVisBufferAtomic.comp, plus the barrier making
        // that write visible to RecordRaster()'s own atomic reads/writes. Must be recorded once
        // per frame, before RecordRaster().
        void RecordClear(VkCommandBuffer cmd);

        // Uploads `viewProj` into the view-params UBO, then for EACH of the masked and opaque
        // software cluster lists in turn: builds that dispatch's VkDispatchIndirectCommand from the
        // list's own atomic count (via BuildDispatchIndirectArgs.comp, perElementMultiplier ==
        // geometry::kMaxClusterTriangles -- see ClusterSoftwareRaster.comp's dispatch-shape
        // comment), then records the indirect rasterization dispatch against that list's own
        // pipeline (masked: ClusterSoftwareRaster.comp, samples the mask array; opaque:
        // ClusterSoftwareRasterOpaque.comp, no mask sampling at all). Ends with the barrier making
        // the atomic VisBuffer's writes visible to a later compute read
        // (renderer::ClusterResolvePass).
        void RecordRaster(VkCommandBuffer cmd, const maths::mat4& viewProj);

        VkImage GetVisBufferAtomicImage() const { return m_VisBufferAtomicImage; }
        VkImageView GetVisBufferAtomicView() const { return m_VisBufferAtomicView; }

    private:
        static constexpr uint32_t kWorkgroupSize = 64;   // Matches ClusterSoftwareRaster.comp's local_size_x.
        static constexpr uint32_t kClearWorkgroupSize = 8; // Matches ClearVisBufferAtomic.comp's local_size_x/y.

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only to vmaDestroyImage() in Shutdown().
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_VisBufferAtomicImage = VK_NULL_HANDLE;
        VmaAllocation m_VisBufferAtomicAllocation = VK_NULL_HANDLE;
        VkImageView m_VisBufferAtomicView = VK_NULL_HANDLE;

        GpuBuffer m_ViewParamsBuffer;   // SoftwareRasterViewParamsUBO, std140, GPU_ONLY.
        GpuBuffer m_DispatchArgsBuffer; // VkDispatchIndirectCommand (3x uint32), GPU_ONLY, masked list.
        GpuBuffer m_DispatchArgsOpaqueBuffer; // VkDispatchIndirectCommand (3x uint32), GPU_ONLY, opaque list.

        // Masked raster descriptor set/pipeline (ClusterSoftwareRaster.comp -- samples the mask array).
        VkDescriptorSetLayout m_RasterSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_RasterDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_RasterPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_RasterPipeline = VK_NULL_HANDLE;

        // Opaque raster descriptor set/pipeline (ClusterSoftwareRasterOpaque.comp -- no mask array
        // binding at all, genuinely smaller descriptor set than the masked one above).
        VkDescriptorSetLayout m_OpaqueRasterSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_OpaqueRasterDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_OpaqueRasterPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_OpaqueRasterPipeline = VK_NULL_HANDLE;

        // Clear descriptor set/pipeline (ClearVisBufferAtomic.comp).
        VkDescriptorSetLayout m_ClearSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_ClearDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ClearPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ClearPipeline = VK_NULL_HANDLE;

        // BuildDispatchIndirectArgs descriptor set/pipeline (shared shader with
        // renderer::ClusterOcclusionCullingPass, own separate pipeline/descriptor instances --
        // one set for the masked list, one for the opaque list, both against the same pipeline
        // since the shader itself is generic over which SourceCountSSBO/DispatchArgsSSBO pair it's
        // bound to).
        VkDescriptorSetLayout m_BuildArgsSetLayout = VK_NULL_HANDLE;
        VkDescriptorSet m_BuildArgsDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet m_BuildArgsOpaqueDescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_BuildArgsPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_BuildArgsPipeline = VK_NULL_HANDLE;

        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE; // Shared by all 3 descriptor sets above.
    };

}
