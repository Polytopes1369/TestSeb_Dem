#pragma once

#include <vulkan/vulkan.h>
#include <functional>

namespace renderer {

    class VulkanUtils {
    public:
        // Allocates a primary one-time submit command buffer, executes the provided record function,
        // submits it to the queue, blocks until completion, and frees the command buffer.
        static void ExecuteOneShotCommands(
            VkDevice device,
            VkCommandPool commandPool,
            VkQueue queue,
            const std::function<void(VkCommandBuffer)>& recordFunc
        );

        // Records a single global VkMemoryBarrier2 (not an image/buffer barrier -- no layout
        // transition or ownership transfer, just an execution + memory dependency) via
        // vkCmdPipelineBarrier2. Covers the "producer wrote via stage X, consumer reads/writes via
        // stage Y" idiom repeated throughout the cluster culling/LOD/raster passes: UBO uploads
        // (vkCmdUpdateBuffer -> compute read), buffer clears (vkCmdFillBuffer -> compute
        // read/write), and compute-output-visible-to-indirect-draw handoffs.
        static void RecordMemoryBarrier(
            VkCommandBuffer cmd,
            VkPipelineStageFlags2 srcStage,
            VkAccessFlags2 srcAccess,
            VkPipelineStageFlags2 dstStage,
            VkAccessFlags2 dstAccess
        );

        // Performs a Vulkan 2 image layout transition using vkCmdPipelineBarrier2.
        static void TransitionImageLayout(
            VkCommandBuffer cmd,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkPipelineStageFlags2 srcStage,
            VkAccessFlags2 srcAccess,
            VkPipelineStageFlags2 dstStage,
            VkAccessFlags2 dstAccess,
            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            uint32_t baseMip = 0,
            uint32_t mipCount = 1,
            uint32_t baseLayer = 0,
            uint32_t layerCount = 1
        );

        // Convenience wrapper: allocates/submits/waits on a one-shot command buffer (via
        // ExecuteOneShotCommands) whose sole content is a single TransitionImageLayout call. Used
        // by passes that only need one image transitioned outside their normal per-frame recording
        // (e.g. an UNDEFINED -> GENERAL transition performed once at pass construction time).
        static void TransitionImageLayoutOneShot(
            VkDevice device,
            VkCommandPool commandPool,
            VkQueue queue,
            VkImage image,
            VkImageLayout oldLayout,
            VkImageLayout newLayout,
            VkPipelineStageFlags2 srcStage,
            VkAccessFlags2 srcAccess,
            VkPipelineStageFlags2 dstStage,
            VkAccessFlags2 dstAccess,
            VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            uint32_t baseMip = 0,
            uint32_t mipCount = 1,
            uint32_t baseLayer = 0,
            uint32_t layerCount = 1
        );

        // Creates a simple nearest-filter, clamp-to-edge sampler with no mip/anisotropic filtering
        // beyond a caller-provided max LOD -- the common "point-sample this compute-written image
        // with no wraparound" convention used by HZB/occlusion/resolve/shading-bin passes.
        static VkSampler CreateNearestSampler(VkDevice device, float maxLod = 0.0f);

        // Writes bindings 0/1 of `raySet` to a caller-supplied ray-request/ray-result storage
        // buffer pair -- the "point set 0 at this frame's caller-owned ray buffers" convention
        // shared by SurfaceCacheSWRTPass and SurfaceCacheRayTracingPass, whose SetRayBuffers()
        // methods are otherwise interchangeable back-ends for the same trace request protocol.
        static void WriteRayBuffersDescriptorSet(
            VkDevice device,
            VkDescriptorSet raySet,
            VkBuffer rayBuffer,
            VkDeviceSize rayBufferSize,
            VkBuffer resultBuffer,
            VkDeviceSize resultBufferSize
        );

        // Writes 4 consecutive bindings -- TLAS at `baseBinding`, then the SurfaceCachePass vertex
        // buffer, index buffer, and SurfaceCacheRayTracingPass draw-range buffer at
        // baseBinding+1/+2/+3 -- shared by every GI pass that ray/SDF-traces the scene's HWRT
        // fallback geometry (SurfaceCacheGIInjectPass, WorldProbeGridPass, ScreenProbeGIPass,
        // ReflectionPass). Issued as its own vkUpdateDescriptorSets call, independent of whatever
        // other bindings the caller writes to the same set, so callers can freely interleave this
        // with their own pass-specific writes in any order.
        static void WriteSharedGeometryBindings(
            VkDevice device,
            VkDescriptorSet set,
            uint32_t baseBinding,
            VkAccelerationStructureKHR tlas,
            VkBuffer vertexBuffer,
            VkBuffer indexBuffer,
            VkBuffer drawRangeBuffer
        );
    };

}
