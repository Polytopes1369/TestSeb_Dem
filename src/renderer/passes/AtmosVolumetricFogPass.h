#pragma once
// Atmos weather system, Subtask 3 (atmos_integration_plan.md): camera-aligned Froxel Volumetric Fog
// -- a 160x90x64 view-space-frustum-aligned voxel grid, exponentially distributed along depth (see
// atmos_volumetric_fog_mapping.glsl's own header comment for why), filled by 3 sequential compute
// kernels in AtmosVolumetricFog.comp (mode-branched, same "one shader, mode selects the kernel"
// convention AtmosSkyLUTs.comp already establishes):
//   0 (InjectMediaProps): height-fog decay + AtmosClimatePass's own dew-point/LCL condensation
//     boost + wind-scrolled fbm noise -> scattering/extinction coefficients per froxel.
//   1 (InjectLight): sun contribution (renderer::VirtualShadowMapPass's shadow lookup +
//     renderer::AtmosSkyPass-scale illuminance, Henyey-Greenstein phase) + ONE MegaLights point
//     light per froxel (a self-contained RIS reservoir sample, isotropic weight -- see
//     AtmosVolumetricFog.comp's own SelectVolumeLightRIS comment for why this doesn't reuse
//     megalights_ris.glsl's surface-oriented SelectLightRIS as-is).
//   2 (AccumulateFog): one thread per (x, y) column, walks all 64 Z slices serially, integrating
//     Beer-Lambert transmittance + in-scattered radiance into the final sampled texture.
//
// --- Per-frame call order a caller must follow ---
//   RecordUpdate(cmd, ...) -- records all 3 kernels with the barriers between them, ending with a
//   trailing VkMemoryBarrier2 making the final integrated fog texture visible to whatever the
//   caller's own later consumer (PostProcessComposite.comp) samples it with next.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"
#include "renderer/vulkan/GpuBuffer.h" // Local Fog Volumes (G8): RAII owner of the volumes SSBO.

namespace renderer {

    class AtmosClimatePass;
    class MegaLightsPass;
    class VirtualShadowMapPass;

    class AtmosVolumetricFogPass {
    public:
        AtmosVolumetricFogPass() = default;

        AtmosVolumetricFogPass(const AtmosVolumetricFogPass&) = delete;
        AtmosVolumetricFogPass& operator=(const AtmosVolumetricFogPass&) = delete;

        static constexpr uint32_t kGridWidth = 160;
        static constexpr uint32_t kGridHeight = 90;
        static constexpr uint32_t kGridDepth = 64;
        static constexpr VkFormat kMediaFormat = VK_FORMAT_R16G16_SFLOAT;      // (scatteringCoeff, extinctionCoeff), per km^-1-equivalent world units.
        static constexpr VkFormat kRadianceFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // rawLight (rgb) / integratedFog (rgb = light, a = transmittance).
        // GLSL only allows ONE local_size declaration per shader file, so all 3 kernels share this
        // 4x4x4 workgroup shape even though AccumulateFog (mode 2) only needs a 2D (x, y) dispatch --
        // see AtmosVolumetricFog.comp's own mode-2 "only gl_LocalInvocationID.z == 0 does real work"
        // comment for how the wasted z-threads are handled (cheap: only 160*90 columns total either way).
        static constexpr uint32_t kWorkgroupSize3D = 4;

        // `atmosClimate`/`vsm`/`megaLights` are borrowed, unmodified, and must already be Init'd --
        // same convention as e.g. renderer::MegaLightsPass::Init's own `resolvePass`/`rtPass`
        // parameters. Binds AtmosClimatePass's AtmosGlobalsUBO, MegaLightsPass's own light SSBO, and
        // VirtualShadowMapPass's 4 shadow resources (atlas/page-table/feedback/sun-levels) directly
        // into this pass' own descriptor set -- no deferred SetXxx() call needed (unlike
        // SDFRayMarchPass's own convention) since all 3 producers are guaranteed already Init'd by
        // the time this pass is (see ClusterRenderPipeline::Init's own call-order comment).
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            const AtmosClimatePass& atmosClimate, const MegaLightsPass& megaLights, const VirtualShadowMapPass& vsm);

        void Shutdown();

        // `cameraPosition`/`cameraForward`/`upHint` drive each froxel's world-position reconstruction
        // (view-space depth along `cameraForward`, NOT Euclidean ray length -- see
        // atmos_volumetric_fog_mapping.glsl's own comment); `upHint` need not be orthogonal to
        // `cameraForward` (right/up are re-derived internally, same "right = forward x upHint,
        // up = right x forward" convention as renderer::SDFRayMarchPass::RecordRayMarch's own
        // comment, itself matching renderer::SurfaceCachePass::UpdateVisibility). `sunDirectionWorld`/
        // `sunColor`/`sunIlluminanceLux` mirror renderer::SceneLights::sun's own fields. `frameIndex`
        // seeds InjectLight's per-froxel RIS candidate sequence (Halton23), same decorrelation
        // convention as every other stochastic sampler in this codebase. Must be called at most once
        // per frame into an already-open, caller-owned command buffer (never submits on its own).
        void RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPosition,
            const maths::vec3& cameraForward, const maths::vec3& upHint,
            float fovYRadians, float aspectRatio, const maths::vec3& sunDirectionWorld,
            const maths::vec3& sunColor, float sunIlluminanceLux, uint32_t frameIndex);

        VkImageView GetIntegratedFogView() const { return m_IntegratedFog.view; }
        VkSampler GetFogSampler() const { return m_FogSampler; }

    private:
        struct FroxelImage {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };

        void CreateFroxelImage(VmaAllocator allocator, VkDevice device, VkFormat format, FroxelImage& outImage);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        FroxelImage m_MediaProps;   // RG16F: (scatteringCoeff, extinctionCoeff).
        FroxelImage m_RawLight;     // RGBA16F: unintegrated per-voxel in-scattered radiance (rgb).
        FroxelImage m_IntegratedFog; // RGBA16F: final accumulated light (rgb) + transmittance (a).

        VkSampler m_FogSampler = VK_NULL_HANDLE; // Linear, clamp-to-edge -- shared by all 3 images' sampled-read bindings and by consumers.

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;

        // Local Fog Volumes (G8): std430 SSBO (binding 11) of config::localfog::VOLUMES' active
        // entries, built ONCE from config at Init (they are static authored scene content, like the
        // showcase zone layout). Host-visible/mapped, written once -- same "tiny static data, no
        // staging needed" convention as renderer::MegaLightsPass::m_LightBuffer. m_LocalFogVolumeCount
        // is the number of active entries actually uploaded (pushed to the shader each frame, or 0
        // when config::localfog::ENABLE is false -- see RecordUpdate).
        GpuBuffer m_LocalFogVolumes;
        uint32_t m_LocalFogVolumeCount = 0u;
    };

}
