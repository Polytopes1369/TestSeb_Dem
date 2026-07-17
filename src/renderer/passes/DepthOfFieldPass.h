#pragma once
// Phase PP3 (post-process stack roadmap -- see this repo's own project memory
// project_postprocess_stack_roadmap.md): physically-derived Depth of Field. See DepthOfField.comp's
// own header comment for the full thin-lens Circle of Confusion + Poisson-disk gather technique and
// why this runs immediately after renderer::TAATSRPass, before renderer::BloomPass.
//
// --- Per-frame call order a caller must follow ---
//   1. UpdateSourceDescriptor(hdrColorView) -- renderer::TAATSRPass's own output view ping-pongs
//      between 2 images every frame, so this pass' own binding to it must be re-written every frame
//      too (same reason renderer::PostProcessPass/renderer::BloomPass both already do this).
//   2. RecordGenerate(cmd, invViewProj, cameraPositionWorld, settings) -- single dispatch.
//
// GetOutputView() is what renderer::BloomPass and renderer::PostProcessPass both read from instead
// of renderer::TAATSRPass's own output directly (see renderer::ClusterRenderPipeline::RecordFrame's
// own [13d]/[13e] ordering comment).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/GpuImage.h"

namespace renderer {

    class DepthOfFieldPass {
    public:
        DepthOfFieldPass() = default;
        ~DepthOfFieldPass() { Shutdown(); }

        DepthOfFieldPass(const DepthOfFieldPass&) = delete;
        DepthOfFieldPass& operator=(const DepthOfFieldPass&) = delete;

        static constexpr VkFormat kOutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        struct Settings {
            float focalLengthMM = 50.0f;             // 50mm -- a "normal" (non-wide, non-tele) lens.
            float focusDistanceWorldUnits = 10.0f;    // World-space distance from the camera that's in perfect focus.
            float maxCoCRadiusPixels = 24.0f;         // Caps the gather kernel's cost/footprint regardless of how far out of focus a pixel is.
        };

        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D displayExtent, VkImageView hdrColorView, VkImageView depthView);

        void Shutdown();

        void UpdateSourceDescriptor(VkImageView hdrColorView);

        // `aperture` is PostProcessPass::Settings::aperture -- the SAME Physical Camera f-stop
        // drives both exposure and this pass' own Circle of Confusion, exactly like a real lens.
        void RecordGenerate(VkCommandBuffer cmd, const maths::mat4& invViewProj,
            const maths::vec3& cameraPositionWorld, float aperture, const Settings& settings);

        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        struct ParamsUBO {
            maths::mat4 invViewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f, _pad0 = 0.0f;
            float aperture = 4.0f;
            float focalLengthMM = 50.0f;
            float focusDistanceWorldUnits = 10.0f;
            float maxCoCRadiusPixels = 24.0f;
        };

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_DisplayExtent{ 0, 0 };

        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        VkSampler m_LinearSampler = VK_NULL_HANDLE;

        GpuBuffer m_ParamsBuffer;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
