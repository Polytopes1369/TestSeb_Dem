#pragma once
// Temporal Anti-Aliasing (TAA) & Temporal Super Resolution (TSR) pass.
// Sized and resolved in Display (Native Swapchain) space from low-res rendered color and depth targets.
// Owns the two high-resolution ping-pong history images, the view UBO, and the compute pipeline.

#include <array>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class TAATSRPass {
    public:
        TAATSRPass() = default;
        ~TAATSRPass() { Shutdown(); }

        TAATSRPass(const TAATSRPass&) = delete;
        TAATSRPass& operator=(const TAATSRPass&) = delete;

        // History color buffer format. R16G16B16A16_SFLOAT is optimal for precision retention,
        // matching Unreal Engine 5's own internal TSR accumulation pipeline format.
        static constexpr VkFormat kHistoryFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, VkExtent2D displayExtent,
            VkImageView lowResColorView, VkImageView lowResDepthView);

        void Shutdown();

        // Recreates descriptor sets when low-res color or depth image views are changed/resized
        void UpdateDescriptorSets(VkImageView lowResColorView, VkImageView lowResDepthView);

        // Records the temporal blend and upscaling shader dispatches.
        void RecordPass(VkCommandBuffer cmd,
            const maths::mat4& viewProj,
            const maths::mat4& prevViewProj,
            const maths::mat4& invViewProj,
            float jitterX, float jitterY,
            uint32_t frameIndex,
            bool resetHistory);

        VkImage GetOutputImage() const { return m_HistoryImages[m_CurrentHistoryIndex]; }
        VkImageView GetOutputView() const { return m_HistoryImageViews[m_CurrentHistoryIndex]; }

    private:
        struct TAATSRViewParamsUBO {
            maths::mat4 viewProj;
            maths::mat4 prevViewProj;
            maths::mat4 invViewProj;
            maths::vec2 renderExtent;  // Low-resolution rendering dimensions
            maths::vec2 displayExtent; // Native display dimensions
            maths::vec2 jitterOffset;  // Camera subpixel offset in pixels
            uint32_t frameIndex;
            uint32_t resetHistory;
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };
        VkExtent2D m_DisplayExtent{ 0, 0 };

        // Ping-pong history resources (display/swapchain resolution)
        VkImage m_HistoryImages[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VmaAllocation m_HistoryAllocations[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VkImageView m_HistoryImageViews[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        uint32_t m_CurrentHistoryIndex = 0; // Index of the newly written slot

        GpuBuffer m_ViewParamsBuffer;
        VkSampler m_LinearSampler = VK_NULL_HANDLE;
        VkSampler m_NearestSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE }; // [0] writes ping[0] reading ping[1], [1] writes ping[1] reading ping[0]

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

} // namespace renderer
