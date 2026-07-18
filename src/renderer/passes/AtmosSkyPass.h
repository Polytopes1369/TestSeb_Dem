#pragma once
// Atmos weather system, Subtask 2 (atmos_integration_plan.md): Physically Based Sky, Atmosphere and
// Cloud Rendering (Sebastien Hillaire, 2020) -- generates 3 LUTs on the GPU (AtmosSkyLUTs.comp, see
// that shader's own header comment for the full per-mode derivation):
//   - Transmittance LUT (256x64, R16G16B16A16_SFLOAT): optical depth from any (height, view-zenith-
//     angle) to the atmosphere's top boundary. Pure medium property, independent of the sun.
//   - Multi-Scattering LUT (32x32, R16G16B16A16_SFLOAT): isotropic multi-bounce scattering estimate,
//     parameterized by (sun-zenith-cosine, height).
//   - Sky-View LUT (200x200, R16G16B16A16_SFLOAT): the actual sky appearance this frame, baked
//     directly against THIS frame's real sun direction (azimuth-relative-to-sun + horizon-detail
//     elevation mapping -- see atmos_sky_lut_mapping.glsl's own header comment).
//
// --- Cache policy ---
// Transmittance and Multi-Scattering depend only on the sun's ZENITH angle (not azimuth, not the
// camera), so RecordUpdate() only re-dispatches those two modes when the sun direction has moved
// more than a small epsilon since the last call that regenerated them (or on the very first call) --
// Sky-View is baked fresh every single call, since it is cheap relative to the other two AND is what
// actually needs to track the sun's live azimuth for correct shadow-band placement.
//
// --- Per-frame call order a caller must follow ---
//   RecordUpdate(cmd, sunDirectionWorld, sunIlluminanceLux) -- records 1-3 dispatches (Sky-View
//   always; Transmittance+Multi-Scattering only when dirty) with the barriers between them, ending
//   with a trailing VkMemoryBarrier2 making the Sky-View LUT visible to whatever the caller's own
//   later consumers (PostProcessComposite.comp, SDFRayMarch.comp) sample it with next.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"

namespace renderer {

    class AtmosSkyPass {
    public:
        AtmosSkyPass() = default;

        AtmosSkyPass(const AtmosSkyPass&) = delete;
        AtmosSkyPass& operator=(const AtmosSkyPass&) = delete;

        static constexpr VkFormat kLUTFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
        static constexpr VkExtent2D kTransmittanceExtent{ 256, 64 };
        static constexpr VkExtent2D kMultiScatteringExtent{ 32, 32 };
        static constexpr VkExtent2D kSkyViewExtent{ 200, 200 };
        static constexpr uint32_t kWorkgroupSize = 8; // Matches AtmosSkyLUTs.comp's local_size_x/y.

        // Sun direction change (dot product against the last-regenerated direction) beyond which
        // Transmittance/Multi-Scattering are considered dirty -- see class comment's cache policy.
        static constexpr float kSunDirtyDotThreshold = 0.9999f;

        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // `sunDirectionWorld` points FROM the light TOWARD the scene (renderer::ClusterResolvePass's
        // own convention, matches renderer::SceneLights::sun.direction). `sunIlluminanceLux` is the
        // same real-photometric value renderer::SceneLights::sun.intensity already carries (see
        // LightingTypes.h's own comment on this engine's lux/candela recalibration) -- reused
        // directly here so the sky's own brightness stays consistent with the rest of the lighting
        // pipeline's exposure/tonemap chain without a second, hand-tuned constant.
        void RecordUpdate(VkCommandBuffer cmd, const maths::vec3& sunDirectionWorld, float sunIlluminanceLux);

        VkImageView GetSkyViewLUTView() const { return m_SkyView.view; }
        VkSampler GetLUTSampler() const { return m_LUTSampler; }

    private:
        struct LUTImage {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };

        void DispatchMode(VkCommandBuffer cmd, int32_t mode, VkExtent2D extent,
            const maths::vec3& sunDirectionWorld, float sunIlluminanceLux);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        LUTImage m_Transmittance;
        LUTImage m_MultiScattering;
        LUTImage m_SkyView;

        VkSampler m_LUTSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        bool m_HasGeneratedStaticLUTs = false;
        maths::vec3 m_LastStaticSunDirection{};
    };

}
