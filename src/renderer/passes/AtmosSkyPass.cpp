#include "renderer/passes/AtmosSkyPass.h"

#include <array>
#include <format>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of AtmosSkyLUTs.comp's AtmosSkyLUTsPC push_constant block. Push
        // constants have no std140/std430 alignment rules (each scalar is simply 4-byte-aligned in
        // declaration order), unlike every UBO/SSBO struct elsewhere in this codebase.
        struct AtmosSkyLUTsPC {
            int32_t mode = 0;
            int32_t outputWidth = 0;
            int32_t outputHeight = 0;
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
            float sunIlluminance = 0.0f;
        };
        static_assert(sizeof(AtmosSkyLUTsPC) == 28,
            "AtmosSkyLUTsPC must match AtmosSkyLUTs.comp's own push_constant block exactly");

    } // namespace

    bool AtmosSkyPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
        // Note: m_Device / m_Allocator are already set by RenderPass<AtmosSkyPass>::Init() before
        // this runs.

        // --- 3 LUT images: STORAGE (this shader's own imageStore) | SAMPLED (Transmittance/Multi-
        // Scattering read back within THIS shader for later modes; Sky-View read by PostProcess-
        // Composite.comp/SDFRayMarch.comp) -- see VulkanUtils::CreateStorageSampledImage2D's own
        // comment for why both usage bits are bundled together. All 3 stay in GENERAL layout for
        // their entire lifetime, same convention as every other compute-written-and-sampled image in
        // this codebase (e.g. renderer::MegaLightsPass's own raw radiance image). ---
        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kLUTFormat, kTransmittanceExtent,
            m_Transmittance.image, m_Transmittance.allocation, m_Transmittance.view);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_Transmittance.view, nullptr);
            vmaDestroyImage(m_Allocator, m_Transmittance.image, m_Transmittance.allocation);
            m_Transmittance.view = VK_NULL_HANDLE;
            m_Transmittance.image = VK_NULL_HANDLE;
            m_Transmittance.allocation = VK_NULL_HANDLE;
        });

        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kLUTFormat, kMultiScatteringExtent,
            m_MultiScattering.image, m_MultiScattering.allocation, m_MultiScattering.view);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_MultiScattering.view, nullptr);
            vmaDestroyImage(m_Allocator, m_MultiScattering.image, m_MultiScattering.allocation);
            m_MultiScattering.view = VK_NULL_HANDLE;
            m_MultiScattering.image = VK_NULL_HANDLE;
            m_MultiScattering.allocation = VK_NULL_HANDLE;
        });

        VulkanUtils::CreateStorageSampledImage2D(allocator, device, kLUTFormat, kSkyViewExtent,
            m_SkyView.image, m_SkyView.allocation, m_SkyView.view);
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_SkyView.view, nullptr);
            vmaDestroyImage(m_Allocator, m_SkyView.image, m_SkyView.allocation);
            m_SkyView.view = VK_NULL_HANDLE;
            m_SkyView.image = VK_NULL_HANDLE;
            m_SkyView.allocation = VK_NULL_HANDLE;
        });

        VkClearColorValue zeroClear{}; zeroClear.float32[0] = zeroClear.float32[1] = zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 1.0f;
        VulkanUtils::ExecuteOneShotCommands(device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_Transmittance.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_MultiScattering.image, zeroClear);
            VulkanUtils::ClearComputeImageToGeneral(cmd, m_SkyView.image, zeroClear);
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
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_LUTSampler));
        RegisterResource([this] { vkDestroySampler(m_Device, m_LUTSampler, nullptr); m_LUTSampler = VK_NULL_HANDLE; });

        // --- Descriptor set: 5 bindings -- see AtmosSkyLUTs.comp's own binding comments. Storage +
        // sampler pairs on the SAME underlying view are safe here because mode is uniform across an
        // entire dispatch: a mode-0 dispatch only ever imageStores into binding 0, never sampling
        // binding 1 in that same dispatch, and so on -- see that shader's own header comment. ---
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{ {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_TransmittanceStorage
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_TransmittanceSampler
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_MultiScatteringStorage
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_MultiScatteringSampler
            { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_SkyViewStorage
        } };
        std::array<VkDescriptorPoolSize, 2> poolSizes{ {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
        } };
        // Was 3 separate calls (vkCreateDescriptorSetLayout / vkCreateDescriptorPool /
        // vkAllocateDescriptorSets) -- same repeated 3-call sequence as 40+ other passes, now bundled
        // in VulkanUtils::CreateDescriptorSetLayoutPoolAndSet.
        auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device, bindings, poolSizes);
        m_SetLayout = descSet.layout;
        m_DescriptorPool = descSet.pool;
        m_Set = descSet.set;
        RegisterResource([this] {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            m_DescriptorPool = VK_NULL_HANDLE;
            m_SetLayout = VK_NULL_HANDLE;
            m_Set = VK_NULL_HANDLE;
        });

        VkDescriptorImageInfo transStorageInfo{ VK_NULL_HANDLE, m_Transmittance.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo transSamplerInfo{ m_LUTSampler, m_Transmittance.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo msStorageInfo{ VK_NULL_HANDLE, m_MultiScattering.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo msSamplerInfo{ m_LUTSampler, m_MultiScattering.view, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo skyViewStorageInfo{ VK_NULL_HANDLE, m_SkyView.view, VK_IMAGE_LAYOUT_GENERAL };
        std::array<VkWriteDescriptorSet, 5> writes{ {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &transStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &transSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &msStorageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &msSamplerInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &skyViewStorageInfo, nullptr, nullptr },
        } };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AtmosSkyLUTsPC) };
        VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plInfo.setLayoutCount = 1;
        plInfo.pSetLayouts = &m_SetLayout;
        plInfo.pushConstantRangeCount = 1;
        plInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &plInfo, nullptr, &m_PipelineLayout));
        RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); m_PipelineLayout = VK_NULL_HANDLE; });

        VkShaderModule shader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/AtmosSkyLUTs.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shader);
        vkDestroyShaderModule(m_Device, shader, nullptr);
        RegisterResource([this] { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); m_Pipeline = VK_NULL_HANDLE; });

        // Not a Vulkan handle -- but the original hand-written Shutdown() reset this cache state too,
        // so a Shutdown() followed by a fresh Init() on the same instance doesn't start out believing
        // stale Transmittance/Multi-Scattering LUTs (from before the shutdown) are still valid.
        RegisterResource([this] {
            m_HasGeneratedStaticLUTs = false;
            m_LastStaticSunDirection = {};
        });

        LOG_INFO(std::format("[AtmosSkyPass] Initialized (Transmittance {}x{}, Multi-Scattering {}x{}, Sky-View {}x{}).",
            kTransmittanceExtent.width, kTransmittanceExtent.height,
            kMultiScatteringExtent.width, kMultiScatteringExtent.height,
            kSkyViewExtent.width, kSkyViewExtent.height));
        return true;
    }

    // Shutdown() is inherited from RenderPass<AtmosSkyPass>: it runs every RegisterResource()
    // cleanup registered above, in reverse order (pipeline -> pipeline layout -> descriptor
    // pool+layout -> sampler -> Sky-View -> Multi-Scattering -> Transmittance), which destroys
    // exactly the same handles the hand-written Shutdown() used to, in the same dependency-safe
    // order (consumers before the resources they depend on).

    void AtmosSkyPass::DispatchMode(VkCommandBuffer cmd, int32_t mode, VkExtent2D extent,
        const maths::vec3& sunDirectionWorld, float sunIlluminanceLux) {
        AtmosSkyLUTsPC pc{};
        pc.mode = mode;
        pc.outputWidth = static_cast<int32_t>(extent.width);
        pc.outputHeight = static_cast<int32_t>(extent.height);
        pc.sunDirX = sunDirectionWorld.x;
        pc.sunDirY = sunDirectionWorld.y;
        pc.sunDirZ = sunDirectionWorld.z;
        pc.sunIlluminance = sunIlluminanceLux;

        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        uint32_t groupsX = (extent.width + kWorkgroupSize - 1u) / kWorkgroupSize;
        uint32_t groupsY = (extent.height + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    void AtmosSkyPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& sunDirectionWorld, float sunIlluminanceLux) {
        bool sunDirty = !m_HasGeneratedStaticLUTs
            || sunDirectionWorld.Dot(m_LastStaticSunDirection) < kSunDirtyDotThreshold;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

        if (sunDirty) {
            DispatchMode(cmd, 0, kTransmittanceExtent, sunDirectionWorld, sunIlluminanceLux);
            // Multi-Scattering (mode 1) samples the Transmittance LUT this same dispatch sequence
            // just wrote -- must be visible first.
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            DispatchMode(cmd, 1, kMultiScatteringExtent, sunDirectionWorld, sunIlluminanceLux);

            m_HasGeneratedStaticLUTs = true;
            m_LastStaticSunDirection = sunDirectionWorld;
        }

        // Sky-View (mode 2) samples both the Transmittance and Multi-Scattering LUTs -- needs the
        // above barrier's guarantee even on a frame that didn't regenerate them (a PRIOR frame's
        // barrier already made them visible then, and neither is ever written again until the sun
        // moves, so no new barrier is needed on a non-dirty frame -- only the dirty branch's own
        // writes need one).
        DispatchMode(cmd, 2, kSkyViewExtent, sunDirectionWorld, sunIlluminanceLux);

        // Trailing barrier: makes the Sky-View LUT (and, on a dirty frame, the other two -- harmless
        // to include unconditionally) visible to this frame's later consumers (PostProcessComposite.
        // comp, SDFRayMarch.comp), which read it via COMPUTE_SHADER combined-image-sampler.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
