#pragma once
// GPU-instanced procedural fur/hair strands (UE5.8 rendering-parity gap G10a): a dedicated Hair/Fur
// shading model grown off the skinned creature entity's animated surface. UE5.8's Substrate
// documentation (and this codebase's own substrate_bsdf.glsl header) explicitly list "Hair/Eye/Water
// shading models" as SEPARATE, DEDICATED shading models OUTSIDE Substrate's Slab BSDF -- so this is
// NOT forced into substrate_bsdf.glsl; it has its own evaluation path (include/hair_bsdf.glsl), the
// same way this codebase already gives Water (WaterForward.frag) and Terrain (terrain_shading.glsl)
// their own dedicated shading.
//
// Structurally this is renderer::VegetationScatterPass (gap G2) applied to hair: many thin, cheap,
// GPU-generated/instanced primitives that must NOT each become a Nanite cluster-DAG entity, kept in
// this pass' OWN dedicated buffers and drawn by its OWN lightweight forward pass (VSM-shadowed sun +
// World Probe Grid indirect diffuse), culled per-strand (frustum + HZB) independently of the Nanite
// path. The differences from VegetationScatterPass:
//   - No baked base mesh: each strand is a procedural camera-facing ribbon synthesized entirely from
//     gl_VertexIndex (kFurSegments quads), so there is no archetype vertex/index buffer at all -- the
//     draw is a single NON-indexed vkCmdDrawIndirect.
//   - Skinned roots: strand roots are baked once in the creature's rest-pose local space with the
//     creature's own 2-bone LBS weights (FurStrandGen.comp), then re-skinned every frame in the cull
//     and vertex shaders via the identical transform composition ClusterRaster.vert applies to the
//     creature's own vertices (see FurCommon.glsl's FurSkinRestToWorld) -- so the fur tracks the
//     creature's idle undulation with no CPU update step. This is why the bone-matrices +
//     entity-transform SSBOs are bound into both the cull and render descriptor sets.
//
// Why the SKINNED creature (not a static primitive) was chosen as the showcase surface: it is the
// engine's one already-rigged, already-animated entity, and a furry creature whose pelt visibly
// undulates reads far better as a demo of a hair/fur shading model than a static fur patch would --
// and the geometry generation is genuinely compatible, because the creature's bind pose is fully
// analytic (geom_creature.comp) so roots can be placed on its surface and skinned exactly like its
// own vertices, with no mesh readback.
//
// Lifecycle: like VegetationScatterPass, strand generation is bake-time (once at Init, re-runnable
// via GenerateStrands() for Debug density/length tuning). Only RecordCull()/RecordDraw() run per frame.

#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class VirtualShadowMapPass;
    class WorldProbeGridPass;

    // The subset of the creature's geom_creature.comp bake parameters the fur generator needs to
    // place strand roots on the EXACT same analytic bind-pose surface (see FurStrandGen.comp). Filled
    // by the caller from VulkanContext's own creature constants (kCreatureClearingX/kCreatureGroundY/
    // kCreatureRadiusMin/Max) + animation::SkeletalAnimator::kBoneCount/kSegmentLength.
    struct CreatureFurGeometry {
        maths::vec3 worldOffset{ 0.0f, 0.0f, 0.0f }; // geom_creature.comp's worldOffsetX/Y/Z (baked into the creature's own vertices too).
        float radiusMin = 0.06f;
        float radiusMax = 0.30f;
        float segmentLength = 0.32f;                  // animation::SkeletalAnimator::kSegmentLength.
        uint32_t boneCount = 16u;                     // animation::SkeletalAnimator::kBoneCount.
        uint32_t creatureMeshID = 0u;                 // Index into the EntityTransform buffer for the creature (== its meshID).
    };

    // Byte-for-byte mirror of FurCommon.glsl's FurStrandRoot (48 bytes, std430).
    struct GpuFurStrandRoot {
        float rootRestPosX = 0.0f, rootRestPosY = 0.0f, rootRestPosZ = 0.0f;
        float lengthScale = 1.0f;
        float growthNormalX = 0.0f, growthNormalY = 1.0f, growthNormalZ = 0.0f;
        float curlPhase = 0.0f;
        uint32_t boneIndicesPacked = 0u;
        uint32_t boneWeightsPacked = 0u;
        float tint = 1.0f;
        float _pad = 0.0f;
    };
    static_assert(sizeof(GpuFurStrandRoot) == 48,
        "GpuFurStrandRoot must match FurCommon.glsl's FurStrandRoot exactly (std430 layout)");

    class FurStrandPass {
    public:
        FurStrandPass() = default;
        FurStrandPass(const FurStrandPass&) = delete;
        FurStrandPass& operator=(const FurStrandPass&) = delete;

        // Hard cap on strand count (buffer capacity). 32768 * 48 bytes == 1.5 MB, comfortably
        // real-time after the per-frame frustum/HZB cull knocks the drawn count far below this.
        static constexpr uint32_t kMaxStrands = 32768u;

        // Longitudinal ribbon resolution -- MUST equal FurStrand.vert's own kFurSegments. Each strand
        // draws kFurSegments quads == kFurSegments * 6 vertices (the indirect draw's vertexCount).
        static constexpr uint32_t kFurSegments = 4u;

        // `vsm`/`worldProbes` supply the forward lighting set (bound once, never re-written -- same
        // borrowed-and-stable convention as VegetationScatterPass). `hzbView`/`hzbMip0Extent`/
        // `hzbMipCount` are renderer::HZBPass's pyramid, sampled by the per-strand occlusion cull.
        // `boneMatricesBuffer` is animation::SkeletalAnimator::GetBoneMatricesBuffer() (updated in
        // place once per frame -- the handle is stable), `entityTransformBuffer` the shared per-entity
        // transform SSBO; both are read to re-skin each root every frame. `creatureGeom` locates the
        // creature's analytic bind-pose surface for root placement. `colorFormat`/`depthFormat` match
        // the shared forward color/depth target every other forward pass draws onto.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
            VkImageView hzbView, VkExtent2D hzbMip0Extent, uint32_t hzbMipCount,
            VkBuffer boneMatricesBuffer, VkBuffer entityTransformBuffer,
            const CreatureFurGeometry& creatureGeom,
            VkFormat colorFormat, VkFormat depthFormat);

        void Shutdown();

        // (Re)generates the per-strand root buffer from the current config::fur:: knobs via a single
        // blocking one-shot submit (mirrors VegetationScatterPass::GenerateScatter). Safe to call
        // again at runtime ONLY after the device is idle -- the caller (ClusterRenderPipeline::
        // RegenerateFur) guarantees that. Refreshes the CPU-side strand count (GetStrandCount()).
        void GenerateStrands();

        // Per-strand frustum + HZB occlusion cull -> the single indirect draw command. Records the
        // CullParams UBO update, the indirect-command reset, the cull dispatch, and the trailing
        // barrier making the indirect command / visible-index buffer visible to RecordDraw()'s
        // DRAW_INDIRECT / VERTEX_SHADER reads. Must be recorded (outside any dynamic-rendering scope)
        // immediately before RecordDraw(), after this frame's HZB rebuild AND after this frame's
        // SkeletalAnimator::RecordUpdate (the cull reads the animated bone matrices).
        void RecordCull(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const maths::vec3& cameraPositionWorld, bool occlusionCullEnabled);

        // Indirect instanced draw into the shared forward color/depth target. OPAQUE and depth-
        // WRITING -- mirrors renderer::VegetationScatterPass::RecordDraw's own GENERAL<->COLOR and
        // READ_ONLY<->DEPTH barrier dance and restore exactly. `debugWireframe` selects the Debug-only
        // wireframe pipeline (ignored in Release, where that pipeline is never created).
        void RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView,
            VkImage depthImage, VkImageView depthView, VkExtent2D renderExtent,
            const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
            const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
            float globalTimeSeconds, bool debugWireframe);

        uint32_t GetStrandCount() const { return m_StrandCount; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkCommandPool m_CommandPool = VK_NULL_HANDLE;
        VkQueue m_Queue = VK_NULL_HANDLE;

        VkExtent2D m_HZBMip0Extent{ 0, 0 };
        uint32_t m_HZBMipCount = 0;

        CreatureFurGeometry m_CreatureGeom{};
        uint32_t m_StrandCount = 0; // Last generated strand count (CPU-side readback).

        // Borrowed handles (not owned): stable for this pass' lifetime.
        VkBuffer m_BoneMatricesBuffer = VK_NULL_HANDLE;
        VkBuffer m_EntityTransformBuffer = VK_NULL_HANDLE;

        // --- Owned buffers ---
        GpuBuffer m_StrandRootBuffer;      // GpuFurStrandRoot[kMaxStrands], GPU_ONLY (STORAGE).
        GpuBuffer m_CounterBuffer;         // single uint strand count, GPU_ONLY (STORAGE | TRANSFER_DST | TRANSFER_SRC).
        GpuBuffer m_VisibleIndexBuffer;    // uint[kMaxStrands], compacted survivors, GPU_ONLY (STORAGE).
        GpuBuffer m_IndirectCommandBuffer; // single VkDrawIndirectCommand, GPU_ONLY (STORAGE | INDIRECT | TRANSFER_DST).
        GpuBuffer m_CullParamsBuffer;      // std140 UBO, GPU_ONLY (UNIFORM | TRANSFER_DST).
        GpuBuffer m_RenderParamsBuffer;    // std140 UBO, GPU_ONLY (UNIFORM | TRANSFER_DST).
        GpuBuffer m_WorldProbeGridParamsBuffer; // std140 UBO, filled once, GPU_ONLY (UNIFORM | TRANSFER_DST).
        GpuBuffer m_CountReadbackBuffer;   // single uint, CPU_TO_GPU mapped -- one-shot post-generation count readback.

        VkSampler m_HZBSampler = VK_NULL_HANDLE; // Nearest / nearest-mip -- see hzb_occlusion.glsl's own sampler note.

        // --- Descriptor infrastructure ---
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_GenSetLayout = VK_NULL_HANDLE;   // strand-root + counter SSBO.
        VkDescriptorSet m_GenSet = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_CullSetLayout = VK_NULL_HANDLE;  // root/visible/indirect/hzb/cullparams/counter/bones/transforms.
        VkDescriptorSet m_CullSet = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_RenderInstanceSetLayout = VK_NULL_HANDLE; // set 0 render (root/visible/bones/transforms).
        VkDescriptorSet m_RenderInstanceSet = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_RenderParamsSetLayout = VK_NULL_HANDLE;   // set 1 render (render params UBO).
        VkDescriptorSet m_RenderParamsSet = VK_NULL_HANDLE;
        VkDescriptorSetLayout m_LightingSetLayout = VK_NULL_HANDLE;       // set 2 render (VSM + World Probe grid).
        VkDescriptorSet m_LightingSet = VK_NULL_HANDLE;

        // --- Pipelines ---
        VkPipelineLayout m_GenPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_GenPipeline = VK_NULL_HANDLE;

        VkPipelineLayout m_CullPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_CullPipeline = VK_NULL_HANDLE;

        VkPipelineLayout m_RenderPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_RenderPipeline = VK_NULL_HANDLE;
#ifndef NDEBUG
        // Debug-only wireframe visualization -- identical to m_RenderPipeline but VK_POLYGON_MODE_LINE.
        // Never created in Release (CLAUDE.md build-separation rule 8).
        VkPipeline m_WireframePipeline = VK_NULL_HANDLE;
#endif
    };

}
