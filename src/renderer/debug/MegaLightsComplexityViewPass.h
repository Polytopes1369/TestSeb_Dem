#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below): UE5.8
// "MegaLights > Light Complexity" view-mode parity (ImGui "View Modes" tab). Bakes a per-pixel
// "how many MegaLights actually affect this surface point" heatmap into an owned RGBA8 image,
// feeding straight into the existing renderer::debug::DebugBufferViewPass "Buffer Viewer" dropdown
// with VisualizationMode::kPassthrough -- the exact same "compute-bake an image, feed it through
// the shared generic viewer" pattern renderer::debug::ParticleDebugViewPass already established
// (see ClusterRenderPipeline::RecordDebugBufferView's own index table).
//
// The count is a REAL lights-per-pixel overlap count (falloff-radius + angular-shaping tests
// against the full light population, reconstructed from the GBuffer depth), not a ReSTIR
// reservoir-M readout -- see src/shaders/src/Debug/MegaLightsComplexityView.comp's own header
// comment for why M would carry no information here.
#ifndef NDEBUG

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer::debug {

    class MegaLightsComplexityViewPass {
    public:
        MegaLightsComplexityViewPass() = default;

        MegaLightsComplexityViewPass(const MegaLightsComplexityViewPass&) = delete;
        MegaLightsComplexityViewPass& operator=(const MegaLightsComplexityViewPass&) = delete;

        static constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr uint32_t kWorkgroupSize = 8; // Matches MegaLightsComplexityView.comp's local_size_x/y.

        // Allocates the owned render-extent RGBA8 output image (kept in VK_IMAGE_LAYOUT_GENERAL for
        // its entire lifetime, same convention as every other Buffer Viewer candidate image) plus
        // the compute pipeline/descriptor set. The light-SSBO and GBuffer-depth bindings are left
        // pointing at small Init()-time dummies (never actually dispatched against before the first
        // real RecordBake() call rewrites them) purely so the descriptor set starts in a valid,
        // fully-bound state -- same idiom as renderer::debug::ParticleDebugViewPass's own dummy
        // buffer and renderer::debug::DebugBufferViewPass's own 1x1 dummy image.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent);

        void Shutdown();

        // Rewrites the light-SSBO binding to `lightBuffer` (renderer::MegaLightsPass::
        // GetLightBufferHandle()/GetLightBufferSize() -- the 16-byte-header + MegaLight[] layout
        // megalights_ris.glsl's own g_Lights block mirrors) and the depth binding to `gbufferDepthView`
        // (renderer::ClusterResolvePass::GetOutputDepthView(), r32f, reversed-Z), then dispatches
        // MegaLightsComplexityView.comp over the full render extent. Both inputs must already be
        // visible to VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT reads -- true at the Buffer Viewer's own
        // dispatch site ([13a] on cmdLate), which runs long after this frame's resolve and light
        // upload barriers (the light SSBO is filled once at MegaLightsPass::Init and never rewritten
        // -- see that class' own m_LightBuffer comment). Ends with a trailing barrier making the
        // output image visible to renderer::debug::DebugBufferViewPass::RecordView's own sampled
        // read, matching ParticleDebugViewPass::RecordBake's contract exactly.
        void RecordBake(VkCommandBuffer cmd, VkBuffer lightBuffer, VkDeviceSize lightBufferSizeBytes,
            VkImageView gbufferDepthView, const maths::mat4& invViewProj);

        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        // Init()-time dummies the light/depth bindings start out pointing at -- see Init()'s own
        // comment. The dummy buffer is sized to one empty g_Lights header (16 bytes, lightCount 0);
        // the dummy image is a 1x1 r32f storage image.
        GpuBuffer m_DummyLightBuffer;
        VkImage m_DummyDepthImage = VK_NULL_HANDLE;
        VmaAllocation m_DummyDepthAllocation = VK_NULL_HANDLE;
        VkImageView m_DummyDepthView = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
#endif
