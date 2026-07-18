#include "renderer/passes/AtmosVolumetricFogPass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <span>
#include <vector>

#include "core/EngineConfig.h"
#include "core/Logger.h"
#include "renderer/passes/AtmosClimatePass.h"
#include "renderer/passes/MegaLightsPass.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of AtmosVolumetricFog.comp's AtmosVolumetricFogPC push_constant block.
        struct AtmosVolumetricFogPC {
            int32_t mode = 0;
            float cameraPositionX = 0.0f, cameraPositionY = 0.0f, cameraPositionZ = 0.0f;
            float cameraForwardX = 0.0f, cameraForwardY = 0.0f, cameraForwardZ = 0.0f;
            float cameraRightX = 0.0f, cameraRightY = 0.0f, cameraRightZ = 0.0f;
            float cameraUpX = 0.0f, cameraUpY = 0.0f, cameraUpZ = 0.0f;
            float tanHalfFovY = 0.0f;
            float aspectRatio = 0.0f;
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
            float sunColorR = 0.0f, sunColorG = 0.0f, sunColorB = 0.0f;
            float sunIlluminance = 0.0f;
            uint32_t frameIndex = 0u;
            uint32_t localFogVolumeCount = 0u; // Local Fog Volumes (G8): count injected this frame (0 = feature off).
            uint32_t debugVizBounds = 0u;      // Debug-only bounds visualization flag (always 0 in Release).
        };
        static_assert(sizeof(AtmosVolumetricFogPC) == 100,
            "AtmosVolumetricFogPC must match AtmosVolumetricFog.comp's own push_constant block exactly");

        // std430 mirror of local_fog_volumes.glsl's LocalFogVolume (6 x vec4 == 96 bytes, natural
        // std430 stride, no padding). alignas(16) so an array of these keeps the vec4 alignment the
        // std430 layout on the GPU side assumes. Orientation is stored as three world-space
        // orthonormal axes (right/up/forward) so the shader's world->local transform is three dot
        // products with no inverse-rotation math -- see that header's own comment.
        struct alignas(16) LocalFogVolumeGPU {
            float centerX = 0.0f, centerY = 0.0f, centerZ = 0.0f, shape = 0.0f;      // xyz center, w shape (0 box / 1 sphere).
            float halfX = 0.0f, halfY = 0.0f, halfZ = 0.0f, sphereRadius = 0.0f;     // xyz box half-extents, w sphere radius.
            float rightX = 1.0f, rightY = 0.0f, rightZ = 0.0f, density = 0.0f;       // xyz right axis, w base extinction density.
            float upX = 0.0f, upY = 1.0f, upZ = 0.0f, heightFalloff = 0.0f;          // xyz up axis, w vertical falloff rate.
            float fwdX = 0.0f, fwdY = 0.0f, fwdZ = 1.0f, edgeSoftness = 0.0f;        // xyz forward axis, w edge softness.
            float colorR = 1.0f, colorG = 1.0f, colorB = 1.0f, receivesSunShadow = 0.0f; // rgb scattering tint, w shadow flag.
        };
        static_assert(sizeof(LocalFogVolumeGPU) == 96,
            "LocalFogVolumeGPU must match local_fog_volumes.glsl's LocalFogVolume std430 layout exactly");

    } // namespace

    void AtmosVolumetricFogPass::CreateFroxelImage(VmaAllocator allocator, VkDevice device, VkFormat format, FroxelImage& outImage) {
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.format = format;
        imageInfo.extent = { kGridWidth, kGridHeight, kGridDepth };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage.image, &outImage.allocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = outImage.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = format;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outImage.view));
    }

    bool AtmosVolumetricFogPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const AtmosClimatePass& atmosClimate, const MegaLightsPass& megaLights, const VirtualShadowMapPass& vsm) {
        CreateFroxelImage(allocator, device, kMediaFormat, m_MediaProps);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_MediaProps.view, nullptr);
            vmaDestroyImage(m_Allocator, m_MediaProps.image, m_MediaProps.allocation);
        });
        CreateFroxelImage(allocator, device, kRadianceFormat, m_RawLight);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_RawLight.view, nullptr);
            vmaDestroyImage(m_Allocator, m_RawLight.image, m_RawLight.allocation);
        });
        CreateFroxelImage(allocator, device, kRadianceFormat, m_IntegratedFog);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_IntegratedFog.view, nullptr);
            vmaDestroyImage(m_Allocator, m_IntegratedFog.image, m_IntegratedFog.allocation);
        });

        VkClearColorValue zeroClear{}; zeroClear.float32[0] = zeroClear.float32[1] = zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 1.0f;
        VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_MediaProps.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_RawLight.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_IntegratedFog.image, zeroClear);
            });

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_FogSampler));
        RegisterResource([this] { vkDestroySampler(m_Device, m_FogSampler, nullptr); });

        // --- Local Fog Volumes (G8): build the binding-11 std430 SSBO ONCE from config. ---
        // These are static authored scene content (like the VulkanContext zone layout), so they are
        // translated to their GPU form and uploaded a single time here; only the injected COUNT is
        // dynamic at runtime (config::localfog::ENABLE, applied per-frame in RecordUpdate).
        {
            std::vector<LocalFogVolumeGPU> gpuVolumes;
            gpuVolumes.reserve(config::localfog::kMaxLocalFogVolumes);
            for (uint32_t i = 0u; i < config::localfog::kMaxLocalFogVolumes; ++i) {
                const config::localfog::LocalFogVolumeConfig& src = config::localfog::VOLUMES[i];
                if (!src.active) {
                    continue;
                }

                LocalFogVolumeGPU dst{};
                dst.centerX = src.centerX; dst.centerY = src.centerY; dst.centerZ = src.centerZ;
                dst.shape = static_cast<float>(src.shape);
                dst.halfX = src.halfX; dst.halfY = src.halfY; dst.halfZ = src.halfZ;
                dst.sphereRadius = src.sphereRadius;

                // Orthonormal orientation basis from a yaw about world +Y (oriented box). The up
                // axis stays world up so vertical falloff always tracks TRUE world height; the right
                // and forward axes span the horizontal plane. right = (cos, 0, sin),
                // forward = (-sin, 0, cos) -- dot(right, forward) == 0, all unit length (orthonormal).
                const float yawRad = src.yawDegrees * 0.01745329252f; // pi / 180.
                const float c = std::cos(yawRad);
                const float s = std::sin(yawRad);
                dst.rightX = c;    dst.rightY = 0.0f; dst.rightZ = s;
                dst.upX = 0.0f;    dst.upY = 1.0f;    dst.upZ = 0.0f;
                dst.fwdX = -s;     dst.fwdY = 0.0f;   dst.fwdZ = c;

                dst.density = src.density;
                dst.heightFalloff = src.heightFalloff;
                dst.edgeSoftness = src.edgeSoftness;
                dst.colorR = src.colorR; dst.colorG = src.colorG; dst.colorB = src.colorB;
                dst.receivesSunShadow = src.receivesSunShadow ? 1.0f : 0.0f;
                gpuVolumes.push_back(dst);
            }
            m_LocalFogVolumeCount = static_cast<uint32_t>(gpuVolumes.size());

            // A Vulkan buffer must be non-zero-sized and binding 11 must always resolve to a valid
            // buffer even with zero active volumes -- allocate at least one (never-read, since the
            // shader's loop bound is pushed as 0) slot in that defensive case.
            const VkDeviceSize volumesBytes =
                static_cast<VkDeviceSize>(std::max<uint32_t>(1u, m_LocalFogVolumeCount)) * sizeof(LocalFogVolumeGPU);
            m_LocalFogVolumes.Create(allocator, volumesBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
            RegisterResource([this] {
                m_LocalFogVolumes.Destroy();
                m_LocalFogVolumeCount = 0u;
            });
            if (m_LocalFogVolumeCount > 0u) {
                std::memcpy(m_LocalFogVolumes.MappedData(), gpuVolumes.data(),
                    gpuVolumes.size() * sizeof(LocalFogVolumeGPU));
            } else {
                LocalFogVolumeGPU dummy{}; // Inert placeholder, never read (loop bound is 0).
                std::memcpy(m_LocalFogVolumes.MappedData(), &dummy, sizeof(dummy));
            }
            LOG_INFO(std::format("[AtmosVolumetricFogPass] Local fog volumes: {} active (of {} max).",
                m_LocalFogVolumeCount, config::localfog::kMaxLocalFogVolumes));
        }

        // --- Descriptor set: 12 bindings -- see AtmosVolumetricFog.comp's own binding comments. ---
        std::array<VkDescriptorSetLayoutBinding, 12> bindings{ {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_MediaPropsStorage
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_MediaPropsSampler
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_RawLightStorage
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_RawLightSampler
            { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_IntegratedFogStorage
            { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // AtmosGlobalsUBO
            { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // MegaLightsSSBO
            { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // Shadow atlas
            { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // Shadow page table
            { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },         // Shadow feedback
            { 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },        // Shadow sun levels
            { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },        // Local Fog Volumes SSBO (G8)
        } };
        std::array<VkDescriptorPoolSize, 4> poolSizes{ {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 }, // MegaLights + shadow page table + shadow feedback + local fog volumes.
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 }, // AtmosGlobalsUBO + shadow sun levels.
        } };
        auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device, bindings, poolSizes);
        m_SetLayout = descSet.layout;
        m_DescriptorPool = descSet.pool;
        m_Set = descSet.set;
        RegisterResource([this] {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        });

        VkDescriptorImageInfo mediaStorageInfo{ VK_NULL_HANDLE, m_MediaProps.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo mediaSamplerInfo{ m_FogSampler, m_MediaProps.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo rawStorageInfo{ VK_NULL_HANDLE, m_RawLight.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo rawSamplerInfo{ m_FogSampler, m_RawLight.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo integratedStorageInfo{ VK_NULL_HANDLE, m_IntegratedFog.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo atmosGlobalsInfo{ atmosClimate.GetGlobalsBufferHandle(), 0, atmosClimate.GetGlobalsBufferSize() };
        VkDescriptorBufferInfo megaLightsInfo{ megaLights.GetLightBufferHandle(), 0, megaLights.GetLightBufferSize() };
        VkDescriptorImageInfo shadowAtlasInfo{ vsm.GetPhysicalAtlasSampler(), vsm.GetPhysicalAtlasView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo shadowPageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo shadowFeedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo shadowSunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo localFogVolumesInfo{ m_LocalFogVolumes.Handle(), 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 12> writes{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mediaStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &mediaSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &rawStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &rawSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &integratedStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &atmosGlobalsInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &megaLightsInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowAtlasInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowPageTableInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowFeedbackInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &shadowSunLevelsInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &localFogVolumesInfo, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AtmosVolumetricFogPC) };
        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &m_SetLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_PipelineLayout));
        RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); });

        VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/AtmosVolumetricFog.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shader);
        vkDestroyShaderModule(m_Device, shader, nullptr);
        RegisterResource([this] { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); });

        LOG_INFO(std::format("[AtmosVolumetricFogPass] Initialized ({}x{}x{} froxel grid).", kGridWidth, kGridHeight, kGridDepth));
        return true;
    }

    // Shutdown() is inherited from RenderPass<AtmosVolumetricFogPass>: runs the RegisterResource()
    // cleanups above in reverse (pipeline -> pipeline layout -> descriptor pool+layout -> local fog
    // volumes SSBO -> sampler -> integrated-fog image -> raw-light image -> media-props image), the
    // same dependency-safe order the hand-written Shutdown() used.

    void AtmosVolumetricFogPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPosition,
        const maths::vec3& cameraForward, const maths::vec3& upHint,
        float fovYRadians, float aspectRatio, const maths::vec3& sunDirectionWorld,
        const maths::vec3& sunColor, float sunIlluminanceLux, uint32_t frameIndex) {

        // Same orthonormal-basis derivation as renderer::SDFRayMarchPass::RecordRayMarch's own
        // comment (itself matching renderer::SurfaceCachePass::UpdateVisibility).
        const maths::vec3 forward = cameraForward.Normalize();
        const maths::vec3 right = forward.Cross(upHint).Normalize();
        const maths::vec3 up = right.Cross(forward);

        AtmosVolumetricFogPC pc{};
        pc.cameraPositionX = cameraPosition.x; pc.cameraPositionY = cameraPosition.y; pc.cameraPositionZ = cameraPosition.z;
        pc.cameraForwardX = forward.x; pc.cameraForwardY = forward.y; pc.cameraForwardZ = forward.z;
        pc.cameraRightX = right.x; pc.cameraRightY = right.y; pc.cameraRightZ = right.z;
        pc.cameraUpX = up.x; pc.cameraUpY = up.y; pc.cameraUpZ = up.z;
        pc.tanHalfFovY = std::tan(fovYRadians * 0.5f);
        pc.aspectRatio = aspectRatio;
        pc.sunDirX = sunDirectionWorld.x; pc.sunDirY = sunDirectionWorld.y; pc.sunDirZ = sunDirectionWorld.z;
        pc.sunColorR = sunColor.x; pc.sunColorG = sunColor.y; pc.sunColorB = sunColor.z;
        pc.sunIlluminance = sunIlluminanceLux;
        pc.frameIndex = frameIndex;

        // Local Fog Volumes (G8): inject the uploaded volumes unless the master runtime toggle is
        // off (which zeroes the injected count for this frame -- the SSBO itself stays untouched,
        // matching volumetrics::_VOLUMETRIC_FOG_ENABLE's own live-toggle convention).
        pc.localFogVolumeCount = config::localfog::ENABLE ? m_LocalFogVolumeCount : 0u;

        // Debug-only bounds visualization: the config read that drives the shader flag is compiled
        // ONLY into a Debug build (CLAUDE.md rule 8 -- visualization modes must generate no Release
        // code); in Release pc.debugVizBounds stays 0, so the shader's viz branch is never entered.
        pc.debugVizBounds = 0u;
#ifndef NDEBUG
        if (config::debugview::LOCAL_FOG_VOLUME_BOUNDS_VIZ) {
            pc.debugVizBounds = 1u;
        }
#endif

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

        const uint32_t groups3DX = (kGridWidth + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
        const uint32_t groups3DY = (kGridHeight + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;
        const uint32_t groups3DZ = (kGridDepth + kWorkgroupSize3D - 1u) / kWorkgroupSize3D;

        // --- Kernel 0: InjectMediaProps ---
        pc.mode = 0;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups3DX, groups3DY, groups3DZ);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // --- Kernel 1: InjectLight (reads m_MediaProps, writes m_RawLight) ---
        pc.mode = 1;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups3DX, groups3DY, groups3DZ);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        // --- Kernel 2: AccumulateFog (one thread per (x, y) column, reads m_MediaProps + m_RawLight, writes m_IntegratedFog) ---
        // groups3DZ deliberately NOT included here (only 1 Z-group dispatched) -- see this class'
        // own kWorkgroupSize3D comment for why mode 2 only needs a 2D dispatch despite sharing the
        // 3D-shaped workgroup.
        pc.mode = 2;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups3DX, groups3DY, 1);

        // Trailing barrier: makes m_IntegratedFog visible to this frame's later consumer
        // (PostProcessComposite.comp), which reads it via COMPUTE_SHADER combined-image-sampler.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
