#pragma once
// UE5.8-parity gap G1 (Decal system): the deferred projected-decal compositing pass. Drives
// src/shaders/src/Renderer/DecalProject.comp -- see that shader's own header comment for the full
// per-pixel algorithm (depth->world reconstruction, decal-BVH traversal, procedural material
// evaluation, GBuffer + direct-color read-modify-write via albedo de-modulation).
//
// --- Placement in the frame graph ---
// Recorded at the very top of renderer::ClusterRenderPipeline::RecordFrameLate, right after the
// async-compute ownership acquire and BEFORE the deferred lighting passes (GTAO / Contact Shadows /
// Reflections / MegaLights / GI Composite) consume the GBuffer this frame. This is exactly a UE
// deferred decal's "written into the GBuffer before it is consumed by lighting" placement: every
// downstream pass this frame sees the decaled GBuffer, and the resolve pass's already-baked sun-lit
// color is corrected in place via albedo de-modulation (see the shader header).
//
// --- Ownership ---
// This pass owns NONE of the 5 GBuffer/color images it reads-and-writes (they belong to renderer::
// ClusterResolvePass, kept in VK_IMAGE_LAYOUT_GENERAL for their whole lifetime, bound here as plain
// storage images -- the exact same borrowing convention renderer::ReflectionPass uses to RMW that
// same color image). It owns only the per-frame view-params UBO and the three decal data SSBOs (the
// decal instance array + the flattened decal BVH nodes + the BVH leaf index array), all uploaded once
// at Init() into host-visible mapped buffers (the same lightweight upload convention renderer::
// MegaLightsPass uses for its own light BVH -- the decal set is static/immutable after Init()).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/DecalTypes.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class DecalProjectionPass {
    public:
        DecalProjectionPass() = default;

        DecalProjectionPass(const DecalProjectionPass&) = delete;
        DecalProjectionPass& operator=(const DecalProjectionPass&) = delete;

        static constexpr uint32_t kWorkgroupSize = 8; // Matches DecalProject.comp's local_size_x/y.

        // `depthView`/`normalView`/`albedoView`/`roughnessMetallicView`/`colorView` are renderer::
        // ClusterResolvePass's own GBuffer + output-color views (all VK_IMAGE_LAYOUT_GENERAL storage
        // images). `sceneDecals` is the fixed showcase decal set (renderer::GenerateShowcaseDecals())
        // -- its instance array is uploaded into this pass's decal SSBO and its parallel world AABBs
        // are fed to geometry::BuildDecalBVH here, then that BVH is uploaded too. Copied, not retained
        // by reference (the caller's own copy need not outlive this call).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, VkImageView depthView, VkImageView normalView,
            VkImageView albedoView, VkImageView roughnessMetallicView, VkImageView colorView,
            const DecalSceneData& sceneDecals);

        void Shutdown();

        // Uploads this frame's DecalViewParamsUBO (invViewProj/cameraPos/viewport/time + the decal
        // count and, Debug-only, the bounds-overlay flag) and dispatches one invocation per output
        // pixel. Ends with a global memory barrier making the decaled GBuffer + color images visible
        // to the downstream deferred-lighting passes' COMPUTE_SHADER storage/sampled reads. A no-op
        // (records nothing) when the decal set is empty.
        void RecordDecals(VkCommandBuffer cmd, const maths::mat4& invViewProj,
            const maths::vec3& cameraPositionWorld, float timeSeconds
#ifndef NDEBUG
            , bool debugShowBounds
#endif
        );

        uint32_t GetDecalCount() const { return m_DecalCount; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        uint32_t m_DecalCount = 0;

        GpuBuffer m_ViewParamsBuffer;   // DecalViewParamsUBO, std140, GPU_ONLY (vkCmdUpdateBuffer per frame).
        GpuBuffer m_DecalBuffer;        // DecalInstance[], std430, host-visible mapped (uploaded once at Init).
        GpuBuffer m_BVHNodesBuffer;     // DecalBVHNode[], std430, host-visible mapped (uploaded once at Init).
        GpuBuffer m_BVHIndicesBuffer;   // uint[], std430, host-visible mapped (uploaded once at Init).

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
