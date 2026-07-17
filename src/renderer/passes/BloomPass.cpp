#include "renderer/passes/BloomPass.h"

#include <algorithm>
#include <array>
#include <format>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        constexpr uint32_t kWorkgroupSize = 8;

        uint32_t DispatchGroupCount(uint32_t dim) {
            return (dim + kWorkgroupSize - 1) / kWorkgroupSize;
        }

        VkExtent2D HalveExtent(VkExtent2D e) {
            return VkExtent2D{ std::max(1u, e.width / 2u), std::max(1u, e.height / 2u) };
        }

    } // namespace

    void BloomPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D displayExtent, VkImageView hdrSourceView) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_DisplayExtent = displayExtent;

        // --- Mip chain sizing: mip 0 is half display resolution, halving down to kMipCount levels. ---
        m_MipExtents.resize(kMipCount);
        m_MipExtents[0] = HalveExtent(displayExtent);
        for (uint32_t i = 1; i < kMipCount; ++i) {
            m_MipExtents[i] = HalveExtent(m_MipExtents[i - 1]);
        }

        // --- Downsample chain: kMipCount levels, one owned VkImage with a real mip chain. ---
        VkImageCreateInfo downsampleInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        downsampleInfo.imageType = VK_IMAGE_TYPE_2D;
        downsampleInfo.format = kFormat;
        downsampleInfo.extent = { m_MipExtents[0].width, m_MipExtents[0].height, 1 };
        downsampleInfo.mipLevels = kMipCount;
        downsampleInfo.arrayLayers = 1;
        downsampleInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        downsampleInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        downsampleInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        downsampleInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        downsampleInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        m_DownsampleImage.Create(allocator, device, downsampleInfo, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_ASPECT_COLOR_BIT);

        m_DownsampleMipViews.resize(kMipCount);
        for (uint32_t level = 0; level < kMipCount; ++level) {
            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_DownsampleImage.Image();
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = kFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_DownsampleMipViews[level]));
        }

        // --- Upsample chain: kMipCount - 1 levels (mip (kMipCount - 1) is never needed -- see the
        // class comment for why the smallest level is read straight from the downsample chain). ---
        const uint32_t upsampleLevelCount = kMipCount - 1u;
        VkImageCreateInfo upsampleInfo = downsampleInfo;
        upsampleInfo.mipLevels = upsampleLevelCount;
        m_UpsampleImage.Create(allocator, device, upsampleInfo, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_ASPECT_COLOR_BIT);

        m_UpsampleMipViews.resize(upsampleLevelCount);
        for (uint32_t level = 0; level < upsampleLevelCount; ++level) {
            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_UpsampleImage.Image();
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = kFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_UpsampleMipViews[level]));
        }

        // --- One-time UNDEFINED -> GENERAL transition, both images, every mip at once. Neither
        // image ever leaves GENERAL afterward (pure compute read/write for their whole lifetime). ---
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            std::array<VkImageMemoryBarrier2, 2> barriers{};
            for (int i = 0; i < 2; ++i) {
                barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                barriers[i].srcAccessMask = 0;
                barriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                // STORAGE_WRITE: every mip's own imageStore. SAMPLED_READ (not STORAGE_READ): every
                // mip is read back through a sampler2D, never imageLoad -- see BarrierAfterMipWrite's
                // own comment.
                barriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            }
            barriers[0].image = m_DownsampleImage.Image();
            barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, kMipCount, 0, 1 };
            barriers[1].image = m_UpsampleImage.Image();
            barriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, upsampleLevelCount, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            depInfo.pImageMemoryBarriers = barriers.data();
            vkCmdPipelineBarrier2(cmd, &depInfo);
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
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_LinearClampSampler));

        // --- Descriptor set layouts ---
        std::array<VkDescriptorSetLayoutBinding, 2> downsampleBindings{ {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_Source
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_Output
        } };
        VkDescriptorSetLayoutCreateInfo downsampleLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        downsampleLayoutInfo.bindingCount = static_cast<uint32_t>(downsampleBindings.size());
        downsampleLayoutInfo.pBindings = downsampleBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &downsampleLayoutInfo, nullptr, &m_DownsampleSetLayout));

        std::array<VkDescriptorSetLayoutBinding, 4> upsampleBindings{ {
            { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_SmallerBlurred
            { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_Detail
            { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // g_FlareSource
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // g_Output
        } };
        VkDescriptorSetLayoutCreateInfo upsampleLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        upsampleLayoutInfo.bindingCount = static_cast<uint32_t>(upsampleBindings.size());
        upsampleLayoutInfo.pBindings = upsampleBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &upsampleLayoutInfo, nullptr, &m_UpsampleSetLayout));

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMipCount + upsampleLevelCount * 3u };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMipCount + upsampleLevelCount };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = kMipCount + upsampleLevelCount;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        // --- Downsample sets: one per mip, index == mip level ---
        m_DownsampleSets.resize(kMipCount);
        {
            std::vector<VkDescriptorSetLayout> layouts(kMipCount, m_DownsampleSetLayout);
            VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = m_DescriptorPool;
            allocInfo.descriptorSetCount = kMipCount;
            allocInfo.pSetLayouts = layouts.data();
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, m_DownsampleSets.data()));
        }
        // Mip 0's source binding (the external HDR view) is written by UpdateSourceDescriptor()
        // below; mips 1..kMipCount-1 read the mip directly above them, written once here.
        {
            std::vector<VkDescriptorImageInfo> srcInfos(kMipCount);
            std::vector<VkDescriptorImageInfo> dstInfos(kMipCount);
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(kMipCount * 2);
            for (uint32_t level = 1; level < kMipCount; ++level) {
                srcInfos[level] = { m_LinearClampSampler, m_DownsampleMipViews[level - 1], VK_IMAGE_LAYOUT_GENERAL };
                dstInfos[level] = { VK_NULL_HANDLE, m_DownsampleMipViews[level], VK_IMAGE_LAYOUT_GENERAL };
                writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DownsampleSets[level], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &srcInfos[level], nullptr, nullptr });
                writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DownsampleSets[level], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dstInfos[level], nullptr, nullptr });
            }
            // Mip 0's own dst-only write (src written by UpdateSourceDescriptor).
            dstInfos[0] = { VK_NULL_HANDLE, m_DownsampleMipViews[0], VK_IMAGE_LAYOUT_GENERAL };
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DownsampleSets[0], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dstInfos[0], nullptr, nullptr });
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        // --- Upsample sets: one per output mip in [0, kMipCount - 2], index == mip level ---
        m_UpsampleSets.resize(upsampleLevelCount);
        {
            std::vector<VkDescriptorSetLayout> layouts(upsampleLevelCount, m_UpsampleSetLayout);
            VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = m_DescriptorPool;
            allocInfo.descriptorSetCount = upsampleLevelCount;
            allocInfo.pSetLayouts = layouts.data();
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, m_UpsampleSets.data()));
        }
        {
            std::vector<VkDescriptorImageInfo> smallerInfos(upsampleLevelCount);
            std::vector<VkDescriptorImageInfo> detailInfos(upsampleLevelCount);
            std::vector<VkDescriptorImageInfo> flareInfos(upsampleLevelCount);
            std::vector<VkDescriptorImageInfo> outputInfos(upsampleLevelCount);
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(upsampleLevelCount * 4);
            for (uint32_t level = 0; level < upsampleLevelCount; ++level) {
                // The very first upsample step (level == kMipCount - 2) reads the downsample
                // chain's own smallest mip as "the smaller blurred input" (no upsample-chain entry
                // exists for it); every other step reads the upsample chain's own mip (level + 1),
                // its own previously-computed result -- see the class comment.
                VkImageView smallerView = (level == upsampleLevelCount - 1u)
                    ? m_DownsampleMipViews[kMipCount - 1u]
                    : m_UpsampleMipViews[level + 1u];

                smallerInfos[level] = { m_LinearClampSampler, smallerView, VK_IMAGE_LAYOUT_GENERAL };
                detailInfos[level] = { m_LinearClampSampler, m_DownsampleMipViews[level], VK_IMAGE_LAYOUT_GENERAL };
                flareInfos[level] = { m_LinearClampSampler, m_DownsampleMipViews[kFlareSourceMip], VK_IMAGE_LAYOUT_GENERAL };
                outputInfos[level] = { VK_NULL_HANDLE, m_UpsampleMipViews[level], VK_IMAGE_LAYOUT_GENERAL };

                writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_UpsampleSets[level], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &smallerInfos[level], nullptr, nullptr });
                writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_UpsampleSets[level], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &detailInfos[level], nullptr, nullptr });
                writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_UpsampleSets[level], 2, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &flareInfos[level], nullptr, nullptr });
                writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_UpsampleSets[level], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfos[level], nullptr, nullptr });
            }
            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        // --- Pipelines ---
        VkPushConstantRange downsamplePush{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DownsamplePushConstants) };
        VkPipelineLayoutCreateInfo downsamplePLInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        downsamplePLInfo.setLayoutCount = 1;
        downsamplePLInfo.pSetLayouts = &m_DownsampleSetLayout;
        downsamplePLInfo.pushConstantRangeCount = 1;
        downsamplePLInfo.pPushConstantRanges = &downsamplePush;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &downsamplePLInfo, nullptr, &m_DownsamplePipelineLayout));

        VkPushConstantRange upsamplePush{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpsamplePushConstants) };
        VkPipelineLayoutCreateInfo upsamplePLInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        upsamplePLInfo.setLayoutCount = 1;
        upsamplePLInfo.pSetLayouts = &m_UpsampleSetLayout;
        upsamplePLInfo.pushConstantRangeCount = 1;
        upsamplePLInfo.pPushConstantRanges = &upsamplePush;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &upsamplePLInfo, nullptr, &m_UpsamplePipelineLayout));

        VkShaderModule downsampleShader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/BloomDownsample.comp.spv");
        m_DownsamplePipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_DownsamplePipelineLayout, downsampleShader);
        vkDestroyShaderModule(m_Device, downsampleShader, nullptr);

        VkShaderModule upsampleShader = VulkanPipeline::LoadShaderModule(m_Device, "shaders/BloomUpsampleComposite.comp.spv");
        m_UpsamplePipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_UpsamplePipelineLayout, upsampleShader);
        vkDestroyShaderModule(m_Device, upsampleShader, nullptr);

        UpdateSourceDescriptor(hdrSourceView);

        LOG_INFO(std::format("[BloomPass] Initialized: mip0={}x{}, mipCount={}",
            m_MipExtents[0].width, m_MipExtents[0].height, kMipCount));
    }

    void BloomPass::UpdateSourceDescriptor(VkImageView hdrSourceView) {
        VkDescriptorImageInfo sourceInfo{ m_LinearClampSampler, hdrSourceView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DownsampleSets[0], 0, 0, 1,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sourceInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
    }

    void BloomPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_DownsamplePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_DownsamplePipeline, nullptr);
            if (m_UpsamplePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_UpsamplePipeline, nullptr);
            if (m_DownsamplePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_DownsamplePipelineLayout, nullptr);
            if (m_UpsamplePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_UpsamplePipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_DownsampleSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_DownsampleSetLayout, nullptr);
            if (m_UpsampleSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_UpsampleSetLayout, nullptr);
            if (m_LinearClampSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_LinearClampSampler, nullptr);
            for (VkImageView view : m_DownsampleMipViews) vkDestroyImageView(m_Device, view, nullptr);
            for (VkImageView view : m_UpsampleMipViews) vkDestroyImageView(m_Device, view, nullptr);
        }

        m_DownsamplePipeline = VK_NULL_HANDLE; m_UpsamplePipeline = VK_NULL_HANDLE;
        m_DownsamplePipelineLayout = VK_NULL_HANDLE; m_UpsamplePipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_DownsampleSetLayout = VK_NULL_HANDLE; m_UpsampleSetLayout = VK_NULL_HANDLE;
        m_DownsampleSets.clear(); m_UpsampleSets.clear();
        m_LinearClampSampler = VK_NULL_HANDLE;
        m_DownsampleMipViews.clear(); m_UpsampleMipViews.clear();
        m_MipExtents.clear();

        m_DownsampleImage.Destroy();
        m_UpsampleImage.Destroy();
        m_DisplayExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void BloomPass::BarrierAfterMipWrite(VkCommandBuffer cmd, VkImage image, uint32_t mipLevel) const {
        VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        // SAMPLED_READ, not STORAGE_READ: every consumer of a bloom mip (the next downsample step,
        // every upsample step, and renderer::PostProcessPass's own final composite read) samples it
        // through a `sampler2D` (textureLod, for hardware bilinear filtering in the box/tent taps),
        // never `imageLoad` -- unlike renderer::HZBPass's own analogous barrier, whose HZBReduce.comp
        // reads its source mip through a STORAGE_IMAGE binding instead.
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, 1, 0, 1 };

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void BloomPass::RecordGenerate(VkCommandBuffer cmd, const Settings& settings) {
        // --- Downsample chain: mip 0 (thresholded) then mips 1..kMipCount-1 (plain box filter) ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DownsamplePipeline);
        for (uint32_t level = 0; level < kMipCount; ++level) {
            DownsamplePushConstants pc{};
            pc.outputTexelSizeX = 1.0f / static_cast<float>(m_MipExtents[level].width);
            pc.outputTexelSizeY = 1.0f / static_cast<float>(m_MipExtents[level].height);
            pc.threshold = settings.threshold;
            pc.softKnee = settings.softKnee;
            pc.isFirstMip = (level == 0) ? 1u : 0u;

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_DownsamplePipelineLayout, 0, 1, &m_DownsampleSets[level], 0, nullptr);
            vkCmdPushConstants(cmd, m_DownsamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, DispatchGroupCount(m_MipExtents[level].width), DispatchGroupCount(m_MipExtents[level].height), 1);

            BarrierAfterMipWrite(cmd, m_DownsampleImage.Image(), level);
        }

        // --- Upsample chain: mip (kMipCount - 2) down to mip 0, mip 0 also generates flare/dirt ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_UpsamplePipeline);
        const uint32_t upsampleLevelCount = kMipCount - 1u;
        for (uint32_t i = 0; i < upsampleLevelCount; ++i) {
            uint32_t level = upsampleLevelCount - 1u - i; // kMipCount-2, ..., 1, 0.
            VkExtent2D smallerExtent = (level == upsampleLevelCount - 1u) ? m_MipExtents[kMipCount - 1u] : m_MipExtents[level + 1u];

            UpsamplePushConstants pc{};
            pc.outputTexelSizeX = 1.0f / static_cast<float>(m_MipExtents[level].width);
            pc.outputTexelSizeY = 1.0f / static_cast<float>(m_MipExtents[level].height);
            pc.smallerMipTexelSizeX = 1.0f / static_cast<float>(smallerExtent.width);
            pc.smallerMipTexelSizeY = 1.0f / static_cast<float>(smallerExtent.height);
            pc.upsampleRadius = settings.upsampleRadius;
            pc.isFinalMip = (level == 0) ? 1u : 0u;
            pc.ghostIntensity = settings.ghostIntensity;
            pc.ghostCount = settings.ghostCount;
            pc.ghostSpacing = settings.ghostSpacing;
            pc.anamorphicIntensity = settings.anamorphicIntensity;
            pc.anamorphicStretch = settings.anamorphicStretch;
            pc.dirtIntensity = settings.dirtIntensity;
            pc.dirtScale = settings.dirtScale;

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_UpsamplePipelineLayout, 0, 1, &m_UpsampleSets[level], 0, nullptr);
            vkCmdPushConstants(cmd, m_UpsamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, DispatchGroupCount(m_MipExtents[level].width), DispatchGroupCount(m_MipExtents[level].height), 1);

            // Visible to either the next (smaller-level) upsample dispatch's read, or, on mip 0,
            // renderer::PostProcessPass's composite shader -- both are COMPUTE_SHADER/STORAGE_READ.
            BarrierAfterMipWrite(cmd, m_UpsampleImage.Image(), level);
        }
    }

}
