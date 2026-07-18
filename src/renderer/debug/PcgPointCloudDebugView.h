#pragma once
// Debug-only (whole file compiled out in Release -- see the #ifndef NDEBUG guard below): PCG
// Point Cloud Debug Visualization (PCG editor-tooling roadmap, Phase 7.2). Draws each
// pcg::PcgPoint in a supplied point set (the output of any sampler/filter/spawner chain -- see
// src/pcg/PcgPointData.h's own header comment) as a wireframe AABB gizmo directly in the live 3D
// scene -- UE5.8 PCG debug-draw parity: a developer/artist needs to SEE what a PCG graph is
// actually producing (point positions, per-point orientation/scale, per-point density) before any
// real mesh gets spawned from it.
//
// Density -> color mapping (this class' own documented design decision, not a literal UE5.8
// source port -- Epic's own PCG framework source is not available to this codebase): a simple
// linear RGB lerp from red (density == 0.0, "this point contributes nothing") to green
// (density == 1.0, "this point is at full strength"), computed in PcgPointCloudDebug.vert. Chosen
// for at-a-glance readability ("red/green == bad/good") over a perceptually-uniform colormap,
// matching this project's general "debug visualization clarity over physical accuracy" convention
// (e.g. debug::DebugBufferViewPass's own raw-value VisualizationMode dumps).
//
// Rendering approach: follows renderer::debug::DebugTextOverlay's own house-style template
// EXACTLY (see that class' own header comment) -- a single small, self-contained graphics
// pipeline built with dynamic rendering (VK_KHR_dynamic_rendering, no VkRenderPass/VkFramebuffer),
// driven by Init()/Shutdown()/RecordDraw() only. UNLIKE DebugTextOverlay (a 2D screen-space
// overlay with no depth), this pass draws INTO THE LIVE 3D SCENE: it shares the same forward
// color+depth target every other forward pass in ClusterRenderPipeline's own [13c] block draws
// onto, and is recorded LAST in that block (right after renderer::ParticleSystemPass::RecordDraw)
// so gizmos composite on top of every opaque/transparent/particle draw this frame. Depth-TESTED
// (gizmos correctly hide behind real opaque geometry) but NOT depth-WRITING (a debug gizmo must
// never occlude anything -- there is nothing recorded after it in the frame anyway, but this also
// matches VegetationScatterPass's own Debug-only wireframe pipeline convention, see that class'
// own m_WireframePipeline comment, and ParticleSystemPass::RecordDraw's identical
// depthWriteInEnable=false + DEPTH_STENCIL_READ_ONLY_OPTIMAL contract for a pass recorded this
// late in the forward block).
//
// Geometry: 12 box edges (24 vertices), VK_PRIMITIVE_TOPOLOGY_LINE_LIST, entirely generated in
// PcgPointCloudDebug.vert from gl_VertexIndex/gl_InstanceIndex -- no vertex/index buffer of its
// own, same "bindless, SSBO-driven, no VkBuffer vertex input" convention as DebugText.vert. Each
// instance's box corners come from that point's OWN boundsMin/boundsMax, transformed by that
// point's OWN localToWorld matrix -- CPU-composed via pcg::PcgPoint::GetLocalToWorld() (see
// PcgPointData.h's own comment for the exact Translate*FromQuat*Scale composition), NOT
// re-derived in the shader, per this phase's own task brief.
#ifndef NDEBUG

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "pcg/PcgPointData.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer::debug {

    class PcgPointCloudDebugView {
    public:
        PcgPointCloudDebugView() = default;

        PcgPointCloudDebugView(const PcgPointCloudDebugView&) = delete;
        PcgPointCloudDebugView& operator=(const PcgPointCloudDebugView&) = delete;

        // Hard cap on drawable points (instance SSBO capacity) -- generously covers every
        // ClusterRenderPipeline::RunPcgFullPipelineSmokeTest()-sized point set (order of tens of
        // points, see that method's own kPruneMinDistance-thinned grid) with wide headroom for a
        // future, denser real PCG graph result. SetPoints() truncates (logs a warning) rather than
        // overflowing if a caller ever supplies more than this.
        static constexpr uint32_t kMaxPoints = 8192;

        // Allocates the owned instance SSBO (kMaxPoints capacity, GPU_ONLY, empty until the first
        // SetPoints() call) plus this pass' own descriptor set/pipeline layout/graphics pipeline
        // (dynamic rendering; `colorFormat`/`depthFormat` must match the shared forward color/
        // depth target every other forward pass in ClusterRenderPipeline's [13c] block draws onto
        // -- GICompositePass::kOutputFormat / the caller's own depth format, exactly the pair
        // VegetationScatterPass::Init/WaterForwardPass::Init already receive at their own call
        // sites).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkFormat colorFormat, VkFormat depthFormat);

        void Shutdown();

        // Converts `points` (pcg::PcgPoint::GetLocalToWorld() + boundsMin/boundsMax/density --
        // exactly the fields this pass visualizes) into this pass' own GPU instance layout and
        // uploads them via one blocking one-shot staged copy (mirrors DebugTextOverlay::Init's own
        // font-buffer upload convention exactly: a CPU_ONLY mapped staging buffer, a vkCmdCopyBuffer,
        // and a trailing TRANSFER_WRITE -> VERTEX_SHADER/SHADER_STORAGE_READ barrier). NOT a
        // per-frame re-upload -- the point set this phase visualizes (RunPcgFullPipelineSmokeTest's
        // own bake-time result) never changes after startup, so this is called once, not from
        // RecordDraw(). Safe to call again later (e.g. a future re-bake); re-uploads from scratch.
        // Truncates to kMaxPoints (logs a warning) if `points.size()` exceeds it. A zero-size
        // `points` is legal and simply makes GetPointCount() report 0 (RecordDraw() then no-ops).
        void SetPoints(VkCommandPool commandPool, VkQueue queue, const std::vector<pcg::PcgPoint>& points);

        // Records the wireframe box gizmo draw directly into the live forward color+depth target.
        // `colorImage`/`colorView` must be the SAME image ParticleSystemPass::RecordDraw's own
        // `colorImage` parameter targets this frame (already left in VK_IMAGE_LAYOUT_GENERAL on
        // that call's exit -- this method transitions it to COLOR_ATTACHMENT_OPTIMAL for its own
        // draw and restores GENERAL before returning, identical to every other [13c] forward pass'
        // own dance). `depthImage`/`depthView` are read at VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
        // with NO transition of its own (depthWriteEnable=false means this pass never needs
        // exclusive access -- same "already read-only by this point in the frame" assumption
        // ParticleSystemPass::RecordDraw's own depth attachment already documents). `viewProj` is
        // proj*view (this frame's camera), matching every other forward pass' own convention (e.g.
        // VegetationScatterPass::RecordDraw). No-op (skips the whole draw + both barrier dances)
        // when SetPoints() has never been called, or was last called with zero points.
        void RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView,
            VkImage depthImage, VkImageView depthView, VkExtent2D extent, const maths::mat4& viewProj);

        // Last uploaded point count (SetPoints()'s own, post-truncation, result) -- read by the
        // ImGui toggle call site (main.cpp) to show a live count next to the checkbox, and by this
        // class' own RecordDraw() to decide whether to skip the draw entirely.
        uint32_t GetPointCount() const { return m_PointCount; }

    private:
        // Byte-for-byte mirror of PcgPointCloudDebug.vert's own PcgPointGizmoInstance struct
        // (std430 layout): a mat4 (64 bytes, always a whole number of 16-byte column slots)
        // followed by 2x (vec3 + trailing scalar packed into the SAME 16-byte std430 slot -- the
        // same idiom every other std430 struct in this codebase already uses, e.g.
        // GpuVegetationInstance's own comment). maths::mat4 stores its 16 floats column-major in a
        // plain std::array<float,16> with no implicit padding, matching std430's own mat4 layout
        // bit-for-bit (already relied upon by every existing CameraPushConstants use).
        struct GpuPointGizmoInstance {
            maths::mat4 localToWorld{};
            float boundsMinX = -0.5f, boundsMinY = -0.5f, boundsMinZ = -0.5f, density = 1.0f;
            float boundsMaxX = 0.5f, boundsMaxY = 0.5f, boundsMaxZ = 0.5f, _pad0 = 0.0f;
        };
        static_assert(sizeof(GpuPointGizmoInstance) == 96,
            "GpuPointGizmoInstance must match PcgPointCloudDebug.vert's own PcgPointGizmoInstance exactly (std430 layout)");

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Borrowed, stored only so SetPoints() can allocate its own one-shot staging buffer.

        GpuBuffer m_InstanceBuffer; // GpuPointGizmoInstance[kMaxPoints], GPU_ONLY, (re)written by SetPoints().
        uint32_t m_PointCount = 0;  // Last SetPoints()'s post-truncation instance count.

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
#endif // NDEBUG
