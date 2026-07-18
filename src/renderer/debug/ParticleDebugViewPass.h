#pragma once
// Debug-only (subtask E3, Niagara-parity particle system roadmap -- Debug Buffer Viewer extension;
// whole file compiled out in Release, see the #ifndef NDEBUG guard below): bakes renderer::
// ParticleSystemPass's own raw GpuParticle buffer into an owned RGBA8 image, feeding straight into
// the existing renderer::debug::DebugBufferViewPass "Buffer Viewer" dropdown -- this pass' own
// output is already display-ready RGB, selected with VisualizationMode::kPassthrough, exactly the
// same "compute-bake an image, feed it through the shared generic viewer" pattern every other
// Buffer Viewer entry already uses (see ClusterRenderPipeline::RecordDebugBufferView's own 0-14
// index table -- this becomes entry 15).
//
// One pixel = one particle SLOT (kGridWidth * kGridHeight == renderer::ParticleSystemPass::
// kMaxParticles exactly, statically asserted in the .cpp) -- fuses an "alive-list occupancy
// heatmap" (a dead slot, Particle.life <= 0, renders as a faint uniform gray so the grid's own
// structure stays visible) with a per-emitter color-coded overlay (an alive slot's hue keys off
// its own stored Particle.emitterIndex, brightness off its normalized remaining life) into a
// single view. See src/shaders/src/Debug/ParticleDebugView.comp for the exact per-pixel logic.
#ifndef NDEBUG

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "renderer/vulkan/GpuBuffer.h"

namespace renderer::debug {

    class ParticleDebugViewPass {
    public:
        ParticleDebugViewPass() = default;

        ParticleDebugViewPass(const ParticleDebugViewPass&) = delete;
        ParticleDebugViewPass& operator=(const ParticleDebugViewPass&) = delete;

        static constexpr VkFormat kOutputFormat = VK_FORMAT_R8G8B8A8_UNORM;
        static constexpr uint32_t kWorkgroupSize = 8;
        // One pixel per particle slot -- MUST multiply out to exactly renderer::ParticleSystemPass::
        // kMaxParticles (65536 today, itself required to be a power of two by ParticleSort.comp's
        // own bitonic network -- see that constant's own declaration comment). The .cpp's Init()
        // static_asserts this equality directly against renderer::ParticleSystemPass::kMaxParticles,
        // so a future change to that constant that breaks this exact 256x256 factorization fails
        // the BUILD, rather than silently producing an incomplete/overflowing debug view.
        static constexpr uint32_t kGridWidth = 256;
        static constexpr uint32_t kGridHeight = 256;

        // Allocates the owned kGridWidth x kGridHeight rgba8 output image (kept in
        // VK_IMAGE_LAYOUT_GENERAL for its entire lifetime, same convention as every other Buffer
        // Viewer candidate image -- see renderer::debug::DebugBufferViewPass's own class comment)
        // plus the compute pipeline/descriptor set. The descriptor's own g_Particles binding is
        // left pointing at a small Init()-time dummy buffer (never actually sampled from before the
        // first real RecordBake() call rewrites it) purely so the descriptor set starts in a valid,
        // fully-bound state -- same idiom as renderer::debug::DebugBufferViewPass's own 1x1 dummy
        // image.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // Rewrites this pass' own g_Particles binding to `particleBuffer` (a single
        // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER write, same "cheap enough to redo unconditionally every
        // call" convention as renderer::debug::DebugBufferViewPass::RecordView's own g_Source
        // rewrite), then dispatches ParticleDebugView.comp over the full kGridWidth x kGridHeight
        // grid. `particleBuffer`/`particleBufferSizeBytes` must be
        // renderer::ParticleSystemPass::GetParticleBufferHandleForDebugView()/
        // GetParticleBufferSizeForDebugView()'s own return values (or an equivalent GpuParticle
        // array) and must already be visible to VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT /
        // VK_ACCESS_2_SHADER_STORAGE_READ_BIT reads -- the caller owns that barrier (already
        // satisfied every frame this is called, see ClusterRenderPipeline::RecordDebugBufferView's
        // own call-site comment: this always runs after RecordSimulate()'s own trailing barrier,
        // whose scope covers every later command in the same command buffer). `maxEmitters` bounds
        // this call's own emitter-color-palette lookup (renderer::ParticleSystemPass::kMaxEmitters).
        // Ends with a trailing barrier making the output image visible to a following sampled read
        // (renderer::debug::DebugBufferViewPass::RecordView, which this pass' own output feeds
        // directly into as its `sourceView`).
        void RecordBake(VkCommandBuffer cmd, VkBuffer particleBuffer, VkDeviceSize particleBufferSizeBytes, uint32_t maxEmitters);

        VkImage GetOutputImage() const { return m_OutputImage; }
        VkImageView GetOutputView() const { return m_OutputView; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        VkImage m_OutputImage = VK_NULL_HANDLE;
        VmaAllocation m_OutputAllocation = VK_NULL_HANDLE;
        VkImageView m_OutputView = VK_NULL_HANDLE;

        // Init()-time dummy the g_Particles binding starts out pointing at -- see Init()'s own
        // comment. Sized to exactly one GpuParticle (80 bytes, see the .cpp's own
        // kDummyParticleBufferBytes).
        GpuBuffer m_DummyParticleBuffer;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
#endif
