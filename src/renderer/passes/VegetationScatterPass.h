#pragma once
// GPU-instanced procedural vegetation scatter (UE5.8 rendering-parity gap G2): grass tufts, shrubs,
// and small rocks scattered PCG-style across the terrain. This is a deliberately DIFFERENT technique
// from renderer::ProceduralTreePass (which just merged): a tree is baked as a full, individual
// Nanite entity (its own cluster DAG, its own draw) at VulkanContext::GenerateGeometry() -- correct
// for a small, bounded number of hero trees, but it does NOT scale to the hundreds/thousands of
// scatter instances foliage needs (one Nanite entity per grass blade would blow up the entity/
// cluster-DAG budget). Instead this class follows renderer::ParticleSystemPass's structural template:
//
//   - A few small procedural BASE MESHES (3 archetypes: a crossed-card grass tuft, a noise-perturbed
//     bush blob, a noise-perturbed rock blob), each generated ONCE by a geom_*.comp compute shader
//     into this pass' OWN dedicated vertex/index buffers -- never the shared Nanite geometry SSBO, so
//     they never enter the cluster DAG (see geom_scatter_grass.comp / geom_scatter_blob.comp).
//   - A GPU-resident PER-INSTANCE buffer (position/yaw/scale/archetype/tint), generated once by a
//     scatter compute shader that samples the SAME analytic terrain heightfield + biome bands the
//     terrain itself uses, so density correlates with the painted ground (VegetationScatterGen.comp).
//   - A per-frame GPU-DRIVEN CULL (per-instance frustum + HZB occlusion, the same Hierarchical-Z
//     pyramid the Nanite cluster cull samples) compacting survivors into per-archetype indirect draw
//     commands (VegetationInstanceCull.comp) -- never an unculled brute-force draw.
//   - An INDIRECT, INSTANCED draw feeding a lightweight dedicated forward shading path (VSM-shadowed
//     sun + World Probe Grid indirect diffuse, exactly like TransparentForwardPass/TessellationPass/
//     WaterForwardPass/ParticleSystemPass) rather than the Nanite deferred visbuffer path (which is
//     keyed on cluster/triangle IDs that instanced non-cluster geometry cannot produce). See
//     VegetationInstanced.vert / .frag and RecordDraw() for the full justification.
//
// Lifecycle: like ProceduralTreePass, scatter generation is bake-time (once at Init, re-runnable via
// GenerateScatter() for Debug density tuning) -- NOT a per-frame simulation (unlike particles). Only
// RecordCull()/RecordDraw() run per frame.

#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"
#include "renderer/vulkan/RenderPass.h"

namespace renderer {

    class VirtualShadowMapPass;
    class WorldProbeGridPass;

    // Byte-for-byte mirror of VegetationCommon.glsl's VegetationInstance (32 bytes, std430). vec3 +
    // trailing float packs into one 16-byte slot; the four trailing 4-byte fields close a second.
    struct GpuVegetationInstance {
        float positionX = 0.0f, positionY = 0.0f, positionZ = 0.0f;
        float scale = 1.0f;
        float yaw = 0.0f;
        uint32_t archetype = 0;
        uint32_t tintSeed = 0;
        float _pad0 = 0.0f;
    };
    static_assert(sizeof(GpuVegetationInstance) == 32,
        "GpuVegetationInstance must match VegetationCommon.glsl's VegetationInstance exactly (std430 layout)");

    // Migrated to RenderPass<Derived> (see renderer/vulkan/RenderPass.h): Init()/Shutdown() are
    // inherited. One shared descriptor pool allocates 6 DIFFERENT-layout sets -- same reasoning as
    // ReflectionPass/SurfaceCacheTraceContext/MegaLightsPass: left as raw Vulkan calls wrapped in
    // RegisterResource() rather than extending the single-set helper. m_CommandPool/m_Queue are
    // borrowed (not owned) handles, and m_HZBMip0Extent/m_HZBMipCount/m_Ranges were never reset by
    // the original Shutdown() either -- preserved exactly, not "fixed", to avoid a scope-creeping
    // behavior change during a mechanical migration.
    class VegetationScatterPass : public RenderPass<VegetationScatterPass> {
        friend class RenderPass<VegetationScatterPass>; // Lets Init() call our private InitImpl().

    public:
        VegetationScatterPass() = default;
        VegetationScatterPass(const VegetationScatterPass&) = delete;
        VegetationScatterPass& operator=(const VegetationScatterPass&) = delete;

        // Grass / Bush / Rock (indices match VegetationCommon.glsl's kVegArchetype* constants).
        static constexpr uint32_t kArchetypeCount = 3;

        // Hard cap on scattered instances (buffer capacity). The scatter generator backs off
        // gracefully once this is reached (same "share the pool, back off under pressure" convention
        // the particle free-list uses). 24000 * 32 bytes == 750 KB, comfortably real-time after the
        // per-frame frustum/HZB cull knocks the drawn count far below this.
        static constexpr uint32_t kMaxInstances = 24000;

        // `vsm`/`worldProbes` supply the forward lighting set (bound once here, never re-written --
        // same borrowed-and-stable convention as ParticleSystemPass's own set 3). `hzbView`/
        // `hzbMip0Extent`/`hzbMipCount` are renderer::HZBPass's pyramid (already Init'd by the time
        // ClusterRenderPipeline reaches this pass), sampled by the per-instance occlusion cull.
        // `colorFormat`/`depthFormat` match the shared forward color/depth target every other forward
        // pass draws onto.
        // Init(VkDevice, VmaAllocator, VkCommandPool, VkQueue, const VirtualShadowMapPass&,
        // const WorldProbeGridPass&, VkImageView, VkExtent2D, uint32_t, VkFormat, VkFormat) -> bool
        // and Shutdown() are inherited from RenderPass<VegetationScatterPass>; see InitImpl() below.

        // (Re)generates the per-instance scatter buffer from the current config::vegetation:: knobs,
        // via a single blocking one-shot submit (mirrors ProceduralTreePass's own bake-time
        // convention). Safe to call again at runtime ONLY after the device is idle -- the caller
        // (ClusterRenderPipeline::RegenerateVegetationScatter) guarantees that. Refreshes the CPU-side
        // instance count (GetInstanceCount()).
        void GenerateScatter();

        // Per-instance frustum + HZB occlusion cull -> per-archetype indirect draw commands. Records
        // the CullParams UBO update, the per-archetype indirect-command reset, the cull dispatch, and
        // the trailing barrier making the indirect commands / visible-index buffer visible to the
        // subsequent RecordDraw()'s DRAW_INDIRECT / VERTEX_SHADER reads. Must be recorded (outside any
        // dynamic-rendering scope) immediately before RecordDraw(), after this frame's HZB rebuild.
        void RecordCull(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const maths::vec3& cameraPositionWorld, bool occlusionCullEnabled);

        // Indirect instanced draw into the shared forward color/depth target. OPAQUE and depth-
        // WRITING (unlike glass/particles) -- mirrors renderer::TessellationPass::RecordDraw's own
        // GENERAL<->COLOR_ATTACHMENT + READ_ONLY<->DEPTH_ATTACHMENT barrier dance and restore exactly.
        // `debugWireframe` selects the Debug-only wireframe/bounds pipeline (ignored in Release, where
        // that pipeline is never created -- CLAUDE.md build-separation rule 8).
        void RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView,
            VkImage depthImage, VkImageView depthView, VkExtent2D renderExtent,
            const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
            const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
            bool debugWireframe);

        uint32_t GetInstanceCount() const { return m_InstanceCount; }

    private:
        bool InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
            VkImageView hzbView, VkExtent2D hzbMip0Extent, uint32_t hzbMipCount,
            VkFormat colorFormat, VkFormat depthFormat);

        // m_Device / m_Allocator are inherited (protected) from RenderPass<VegetationScatterPass>.
        VkCommandPool m_CommandPool = VK_NULL_HANDLE;
        VkQueue m_Queue = VK_NULL_HANDLE;

        VkExtent2D m_HZBMip0Extent{ 0, 0 };
        uint32_t m_HZBMipCount = 0;

        uint32_t m_InstanceCount = 0; // Last generated instance count (CPU-side readback).

        // Per-archetype draw range into the combined index buffer (vertexOffset is always 0 -- the
        // geom generators bake global vertex indices, like every other geom_*.comp).
        struct ArchetypeRange { uint32_t firstIndex = 0; uint32_t indexCount = 0; };
        std::array<ArchetypeRange, kArchetypeCount> m_Ranges{};

        // --- Owned buffers ---
        GpuBuffer m_ArchetypeVertexBuffer; // struct_custo.glsl Vertex[], all 3 archetypes back-to-back, GPU_ONLY (STORAGE).
        GpuBuffer m_ArchetypeIndexBuffer;  // uint[], global indices, GPU_ONLY (STORAGE | INDEX).
        GpuBuffer m_InstanceBuffer;        // GpuVegetationInstance[kMaxInstances], GPU_ONLY (STORAGE).
        GpuBuffer m_CounterBuffer;         // single uint instance count, GPU_ONLY (STORAGE | TRANSFER_DST | TRANSFER_SRC).
        GpuBuffer m_VisibleIndexBuffer;    // uint[kArchetypeCount * kMaxInstances], per-archetype compacted segments, GPU_ONLY (STORAGE).
        GpuBuffer m_IndirectCommandBuffer; // VkDrawIndexedIndirectCommand[kArchetypeCount], GPU_ONLY (STORAGE | INDIRECT | TRANSFER_DST).
        GpuBuffer m_CullParamsBuffer;      // std140 UBO, GPU_ONLY (UNIFORM | TRANSFER_DST).
        GpuBuffer m_RenderParamsBuffer;    // std140 UBO, GPU_ONLY (UNIFORM | TRANSFER_DST).
        GpuBuffer m_WorldProbeGridParamsBuffer; // std140 UBO, filled once, GPU_ONLY (UNIFORM | TRANSFER_DST).
        GpuBuffer m_CountReadbackBuffer;   // single uint, CPU_TO_GPU mapped -- one-shot post-generation count readback.

        VkSampler m_HZBSampler = VK_NULL_HANDLE; // Nearest / nearest-mip -- see hzb_occlusion.glsl's own sampler note.

        // --- Descriptor infrastructure (one pool for every set this pass owns) ---
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_GeomSetLayout = VK_NULL_HANDLE;   // set 0 for both geom generators (vtx + idx SSBO).
        VkDescriptorSet m_GeomSet = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_ScatterGenSetLayout = VK_NULL_HANDLE; // instance + counter SSBO.
        VkDescriptorSet m_ScatterGenSet = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_CullSetLayout = VK_NULL_HANDLE;   // instance/visible/indirect/hzb/cullparams/counter.
        VkDescriptorSet m_CullSet = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_RenderInstanceSetLayout = VK_NULL_HANDLE; // set 0 for render (vtx/instance/visible).
        VkDescriptorSet m_RenderInstanceSet = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_RenderParamsSetLayout = VK_NULL_HANDLE;   // set 1 for render (render params UBO).
        VkDescriptorSet m_RenderParamsSet = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;       // set 2 for render (VSM + World Probe grid).
        VkDescriptorSet m_LightingSet = VK_NULL_HANDLE;

        // --- Pipelines ---
        VkPipelineLayout m_GeomPipelineLayout = VK_NULL_HANDLE; // shared by both gen pipelines (64-byte push range).
        VkPipeline m_GrassGenPipeline = VK_NULL_HANDLE;
        VkPipeline m_BlobGenPipeline = VK_NULL_HANDLE;

        VkPipelineLayout m_ScatterGenPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ScatterGenPipeline = VK_NULL_HANDLE;

        VkPipelineLayout m_CullPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CullPipeline = VK_NULL_HANDLE;

        VkPipelineLayout m_RenderPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_RenderPipeline = VK_NULL_HANDLE;
#ifndef NDEBUG
        // Debug-only wireframe/bounds visualization pipeline -- identical to m_RenderPipeline but
        // VK_POLYGON_MODE_LINE. Never created in Release (CLAUDE.md build-separation rule 8).
        VkPipeline m_WireframePipeline = VK_NULL_HANDLE;
#endif

        // Bakes the 3 archetype base meshes into m_ArchetypeVertexBuffer/IndexBuffer via one blocking
        // one-shot submit (dispatches geom_scatter_grass.comp then geom_scatter_blob.comp x2).
        void BakeArchetypeGeometry();
    };

}
