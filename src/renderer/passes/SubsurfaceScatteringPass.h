#pragma once
// Screen-space Subsurface Scattering (UE5.8 rendering-parity gap G4, "Subsurface Profile" shading
// model): a separable diffusion-profile blur (the Jimenez/Burley "Separable Subsurface Scattering"
// technique UE5.8's own Subsurface Profile is built on) applied to the fully-lit opaque scene color,
// gated per-pixel by a material's SubstrateSlab::sssProfileScale, so light visibly bleeds through and
// softly wraps past the silhouettes/terminators of thin organic (skin/wax/foliage) geometry instead
// of falling off sharply like a plain Lambertian material. See src/shaders/src/GI/
// SubsurfaceScattering.comp for the actual per-tap diffusion profile + bilateral edge-stopping (and
// the one documented simplification: this pipeline composites specular into the same HDR target, so
// the full radiance of SSS-flagged pixels is diffused rather than a strict diffuse-only term).
//
// --- Where it runs, and the in-place ping-pong design ---
// renderer::ClusterRenderPipeline records this right AFTER renderer::GICompositePass (the point the
// opaque scene is fully lit: direct sun + MegaLights + reflections + denoised indirect GI) and BEFORE
// the forward-transparent passes / TAA -- SSS is an opaque-material response that must be temporally
// resolved together with everything else, and the forward geometry (glass/water/particles) is not yet
// in the image, so it is correctly never diffused. The blur is separable (horizontal then vertical):
//   pass 0: sample GICompositePass's output image  -> write this pass' own owned scratch image
//   pass 1: sample the scratch image               -> write BACK INTO GICompositePass's output image
// Writing the final result back in place means ZERO downstream rewiring -- every later consumer
// (forward passes' color attachment, TAA's sampled read) keeps reading GICompositePass's output view
// exactly as before, now carrying the diffused result. Only ONE extra full-res image (the scratch) is
// allocated; a per-pixel non-SSS pixel is passed through byte-for-byte by both passes, so the image is
// unchanged wherever no SSS material is present.
//
// Follows renderer::ATrousDenoisePass's compute-pass shape (Init / RecordUpdate / Shutdown), reuses
// its exact bilateral-rejection discipline (depth + normal edge stops so the blur never crosses a
// silhouette), and -- like that pass -- keeps every image permanently in VK_IMAGE_LAYOUT_GENERAL and
// synchronizes with plain VkMemoryBarrier2 global barriers (all producers/consumers here are compute
// storage writes / sampled reads, no layout transition ever needed).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"

namespace renderer {

    class SubsurfaceScatteringPass {
    public:
        SubsurfaceScatteringPass() = default;

        SubsurfaceScatteringPass(const SubsurfaceScatteringPass&) = delete;
        SubsurfaceScatteringPass& operator=(const SubsurfaceScatteringPass&) = delete;

        // Matches SubsurfaceScattering.comp's local_size_x/y exactly.
        static constexpr uint32_t kWorkgroupSize = 8;
        // R16G16B16A16_SFLOAT (linear HDR): must equal renderer::GICompositePass::kOutputFormat -- the
        // scratch image ping-pongs the exact same composited HDR radiance that image holds, and an
        // 8-bit UNORM scratch would hard-clip it to [0,1] mid-blur (the same "burned highlights" trap
        // documented on renderer::ClusterResolvePass::kOutputColorFormat).
        static constexpr VkFormat kScratchFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

        // `sceneColorView` is renderer::GICompositePass::GetOutputView() -- read as pass 0's input AND
        // written as pass 1's output (see the class comment's ping-pong description); it must have both
        // SAMPLED and STORAGE usage (GICompositePass's own output image already carries both). `depthView`
        // / `normalView` / `materialIDView` are renderer::ClusterResolvePass's own GBuffer views
        // (GetOutputDepthView / GetOutputNormalView / GetOutputMaterialIDView). `materialParamsBuffer` is
        // renderer::ClusterResolvePass::GetMaterialParamsBuffer() -- the SSBO this pass reads each pixel's
        // sssProfileScale from. All borrowed; must stay valid for this pass' lifetime (bound once here).
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, VkImageView sceneColorView, VkImageView depthView,
            VkImageView normalView, VkImageView materialIDView, VkBuffer materialParamsBuffer);

        void Shutdown();

        // Records the two separable dispatches (horizontal then vertical) with a VkMemoryBarrier2
        // between them (pass 0's scratch STORAGE_WRITE -> pass 1's SHADER_SAMPLED_READ, which also
        // orders pass 0's read of the scene-color image before pass 1 writes it back -- a WAR
        // execution dependency the same barrier covers), and ends with a VkMemoryBarrier2 making the
        // scene-color image (now diffused, written by pass 1) visible to the caller's downstream reads
        // (forward color attachment + TAA sampled read). A no-op (records nothing, leaving
        // GICompositePass's output untouched) when `enabled` is false -- the Debug-only A/B toggle;
        // Release always passes true. `invViewProj` / `cameraPositionWorld` / `projScaleY` size the
        // depth-scaled blur radius; `radiusScale` is the Debug-only tuning multiplier (1.0 in Release).
        void RecordUpdate(VkCommandBuffer cmd, const maths::mat4& invViewProj,
            const maths::vec3& cameraPositionWorld, float projScaleY, float radiusScale, bool enabled);

    private:
        // Byte-for-byte mirror of SubsurfaceScattering.comp's push_constant block.
        struct SSSPushConstants {
            maths::mat4 invViewProj;
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f, radiusScale = 1.0f;
            float viewportWidth = 0.0f, viewportHeight = 0.0f;
            float projScaleY = 1.0f;
            uint32_t direction = 0u;
        };
        static_assert(sizeof(SSSPushConstants) == 96,
            "SSSPushConstants must match SubsurfaceScattering.comp's push_constant block (std430) exactly");

        void RecordOnePass(VkCommandBuffer cmd, VkDescriptorSet set, const SSSPushConstants& pc);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        // The single horizontal-blur intermediate (pass 0 writes it, pass 1 reads it).
        VkImage m_ScratchImage = VK_NULL_HANDLE;
        VmaAllocation m_ScratchAllocation = VK_NULL_HANDLE;
        VkImageView m_ScratchView = VK_NULL_HANDLE;

        VkSampler m_NearestSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        // [0] = horizontal (scene color -> scratch); [1] = vertical (scratch -> scene color).
        VkDescriptorSet m_Sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
