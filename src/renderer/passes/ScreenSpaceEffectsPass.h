#pragma once
// Phase PP4 (post-process stack roadmap -- see this repo's own project memory
// project_postprocess_stack_roadmap.md): bundles the three screen-space lighting effects that all
// read renderer::ClusterResolvePass's own GBuffer and either write their own owned output image
// (GTAO.comp) or read-modify-write that pass' output color image directly (ContactShadows.comp,
// SSRFallback.comp) -- see src/shaders/src/Renderer/GTAO.comp, ContactShadows.comp, SSRFallback.comp
// for the actual algorithms.
//
// Exposes 3 independent Record*() entry points, NOT a single RecordAll(), because their correct
// call sites in renderer::ClusterRenderPipeline::RecordFrame are NOT adjacent to each other:
//   - RecordAmbientOcclusion: anytime after [12] Resolve, before renderer::GICompositePass's own
//     RecordComposite() (which multiplies its denoised indirect GI term by this pass' AO image).
//   - RecordContactShadows: immediately after [12] Resolve, BEFORE [12b2]/[12b3]
//     (Reflections/MegaLights) -- see ContactShadows.comp's own header comment for why.
//   - RecordSSRFallback: AFTER renderer::ReflectionPass's own RecordTrace/RecordTemporal/
//     RecordGather trio has already run this frame (needs that pass' hit-mask).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h"

namespace renderer {

    class ClusterResolvePass;
    class ReflectionPass;

    class ScreenSpaceEffectsPass {
    public:
        ScreenSpaceEffectsPass() = default;

        ScreenSpaceEffectsPass(const ScreenSpaceEffectsPass&) = delete;
        ScreenSpaceEffectsPass& operator=(const ScreenSpaceEffectsPass&) = delete;

        static constexpr uint32_t kWorkgroupSize = 8;
        static constexpr VkFormat kAOFormat = VK_FORMAT_R8_UNORM;

        // All user/scene tunable -- Release-time artistic + quality controls, not debug tooling
        // (same rationale as renderer::PostProcessPass::Settings' own comment).
        struct Settings {
            float aoRadiusWorld = 1.0f;
            float aoIntensity = 1.0f;
            float aoPower = 1.5f;

            float contactShadowLengthWorld = 1.0f;
            float contactShadowIntensity = 0.8f;
            float contactShadowThicknessWorld = 0.3f;

            float ssrFallbackMaxDistanceWorld = 20.0f;
            float ssrFallbackThicknessWorld = 0.5f;
            float ssrFallbackIntensity = 1.0f;
        };

        // `resolvePass` (GBuffer normal/depth/roughness-metallic/albedo + own output color image)
        // and `reflectionPass` (Phase PP4's hit-mask) must both already be Init'd and must outlive
        // this pass -- borrowed unmodified into this pass' own descriptor sets, the same convention
        // renderer::ReflectionPass itself already uses for `resolvePass`.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, const ClusterResolvePass& resolvePass, const ReflectionPass& reflectionPass);

        void Shutdown();

        // Dispatches GTAO.comp -- `fovYRadians` (renderer::CameraFrameInfo::fovYRadians) drives the
        // world-radius -> screen-pixel-radius conversion (GTAO.comp's own `projScale`).
        void RecordAmbientOcclusion(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const maths::vec3& cameraPositionWorld, float fovYRadians, const Settings& settings);

        // Dispatches ContactShadows.comp. `sunDirection` -- see renderer::ClusterResolvePass::
        // RecordResolve's own comment (points FROM the light TOWARD the scene).
        void RecordContactShadows(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const maths::vec3& cameraPositionWorld, const maths::vec3& sunDirection, const Settings& settings);

        // Atmos weather system, Subtask 5: binds renderer::AtmosSkyPass's Sky-View LUT into the SSR
        // Fallback set's binding 7 -- must be called exactly once after both Init() and
        // AtmosSkyPass's own Init(), before the first RecordSSRFallback() call.
        void SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler);

        // Dispatches SSRFallback.comp. `sunDirectionWorld` (Atmos Subtask 5): fed to the Sky-View
        // LUT sample used when even the screen-space march itself finds nothing.
        void RecordSSRFallback(VkCommandBuffer cmd, const maths::mat4& viewProj,
            const maths::vec3& cameraPositionWorld, const maths::vec3& sunDirectionWorld, const Settings& settings);

        VkImageView GetAOView() const { return m_AOView; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_RenderExtent{ 0, 0 };

        VkImage m_AOImage = VK_NULL_HANDLE;
        VmaAllocation m_AOAllocation = VK_NULL_HANDLE;
        VkImageView m_AOView = VK_NULL_HANDLE;

        GpuBuffer m_AOParamsBuffer;
        GpuBuffer m_ContactShadowParamsBuffer;
        GpuBuffer m_SSRFallbackParamsBuffer;

        // Stage 1: GTAO.
        VkDescriptorSetLayout m_AOSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_AODescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_AOSet = VK_NULL_HANDLE;
        VkPipelineLayout m_AOPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_AOPipeline = VK_NULL_HANDLE;

        // Stage 2: Contact Shadows.
        VkDescriptorSetLayout m_ContactShadowSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_ContactShadowDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_ContactShadowSet = VK_NULL_HANDLE;
        VkPipelineLayout m_ContactShadowPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_ContactShadowPipeline = VK_NULL_HANDLE;

        // Stage 3: SSR Fallback.
        VkDescriptorSetLayout m_SSRFallbackSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_SSRFallbackDescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_SSRFallbackSet = VK_NULL_HANDLE;
        VkPipelineLayout m_SSRFallbackPipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_SSRFallbackPipeline = VK_NULL_HANDLE;
    };

}
