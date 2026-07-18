#pragma once
// Atmos weather system, Subtask 4 (atmos_integration_plan.md): Procedural Volumetric Clouds --
// half-resolution raymarch through a spherical-shell cloud layer (same planet-local frame
// AtmosSkyPass's own LUT generation uses: ground radius kGroundRadiusKm, camera fixed at
// kCameraHeightAboveGroundKm -- see AtmosClouds.comp's own header comment for why these two
// systems deliberately share that frame instead of each inventing their own).
//
// Two 3D noise textures are generated ONCE at Init() (this demo's "no data in the .exe" constraint
// means these must be procedural, not baked assets -- see AtmosClouds.comp's own GenerateShapeNoise/
// GenerateDetailNoise comments for the Perlin-Worley combination technique):
//   - Shape noise (128^3, R8_UNORM): low-frequency Perlin FBM eroded by inverted Worley FBM (the
//     classic Schneider/Horizon-Zero-Dawn "billowy" cloud base shape).
//   - Detail noise (32^3, R8_UNORM): high-frequency pure Worley FBM, subtracted from the shape
//     density near cloud edges for wispy erosion detail.
//
// --- Per-frame call order a caller must follow ---
//   RecordUpdate(cmd, ...) -- records ONLY the raymarch dispatch (the two noise textures are never
//   regenerated after Init()), ending with a trailing VkMemoryBarrier2 making the half-resolution
//   cloud buffer visible to whatever the caller's own later consumer (PostProcessComposite.comp)
//   samples it with next.
//
// --- Scope note: half-resolution, no dedicated temporal reprojection ---
// atmos_integration_plan.md's own Subtask 4 objective #4 suggests pairing the half-res raymarch with
// a dedicated cloud-specific temporal reconstruction pass. This implementation instead relies on a
// plain bilinear upsample at composite time (PostProcessComposite.comp) plus the existing
// renderer::TAATSRPass's own whole-frame temporal smoothing downstream -- a deliberately simpler,
// still-complete choice consistent with how renderer::BloomPass/renderer::DepthOfFieldPass already
// upsample their own lower-resolution intermediates in this codebase, rather than standing up a
// second bespoke temporal-accumulation subsystem for one effect.

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "core/maths/Maths.h"

namespace renderer {

    class AtmosClimatePass;

    class AtmosCloudsPass {
    public:
        AtmosCloudsPass() = default;

        AtmosCloudsPass(const AtmosCloudsPass&) = delete;
        AtmosCloudsPass& operator=(const AtmosCloudsPass&) = delete;

        static constexpr uint32_t kShapeNoiseResolution = 128;
        static constexpr uint32_t kDetailNoiseResolution = 32;
        static constexpr VkFormat kNoiseFormat = VK_FORMAT_R8_UNORM;
        static constexpr VkFormat kOutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT; // rgb = in-scattered cloud radiance, a = transmittance.
        static constexpr uint32_t kResolutionDivisor = 2; // Half-resolution -- see class comment's scope note.
        // GLSL only allows ONE local_size declaration per shader file -- all 3 modes (2 noise-gen,
        // 1 raymarch) share this 4x4x4 workgroup shape, same convention renderer::
        // AtmosVolumetricFogPass::kWorkgroupSize3D already establishes for the identical constraint.
        static constexpr uint32_t kWorkgroupSize3D = 4;

        // `renderExtent` is the FULL-resolution render target size -- this pass derives its own
        // half-res output extent internally (kResolutionDivisor). `atmosClimate` is borrowed,
        // unmodified, and must already be Init'd (same convention as renderer::
        // AtmosVolumetricFogPass::Init's own `atmosClimate` parameter) -- binds its AtmosGlobalsUBO
        // directly into this pass' own descriptor set. Dispatches the 2 one-time noise-generation
        // kernels here (GenerateShapeNoise, GenerateDetailNoise); RecordUpdate() never re-dispatches
        // them.
        bool Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
            VkExtent2D renderExtent, const AtmosClimatePass& atmosClimate);

        void Shutdown();

        // `cameraPosition`/`cameraForward`/`upHint` and `sunDirectionWorld`/`sunColor`/
        // `sunIlluminanceLux` follow the exact same conventions as renderer::
        // AtmosVolumetricFogPass::RecordUpdate's own identically-named parameters (right/up
        // re-derived internally from `upHint`, sun direction points FROM the light TOWARD the
        // scene). Must be called at most once per frame into an already-open, caller-owned command
        // buffer (never submits on its own).
        void RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPosition,
            const maths::vec3& cameraForward, const maths::vec3& upHint,
            float fovYRadians, float aspectRatio, const maths::vec3& sunDirectionWorld,
            const maths::vec3& sunColor, float sunIlluminanceLux);

        VkImageView GetCloudView() const { return m_CloudOutput.view; }
        VkSampler GetCloudSampler() const { return m_LinearSampler; }

    private:
        struct Image3D {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };
        struct Image2D {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };

        void DispatchMode(VkCommandBuffer cmd, int32_t mode, const maths::vec3& cameraPosition,
            const maths::vec3& cameraForward, const maths::vec3& cameraRight, const maths::vec3& cameraUp,
            float tanHalfFovY, float aspectRatio, const maths::vec3& sunDirectionWorld,
            const maths::vec3& sunColor, float sunIlluminanceLux);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;
        VkExtent2D m_OutputExtent{ 0, 0 };

        Image3D m_ShapeNoise;
        Image3D m_DetailNoise;
        Image2D m_CloudOutput;

        VkSampler m_NoiseSampler = VK_NULL_HANDLE; // Linear, REPEAT (noise tiles) -- see Init()'s own comment.
        VkSampler m_LinearSampler = VK_NULL_HANDLE; // Linear, CLAMP_TO_EDGE -- for m_CloudOutput's own sampled reads.

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_Set = VK_NULL_HANDLE;
        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
