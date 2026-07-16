#include "renderer/passes/HZBPass.h"

#include <algorithm>
#include <cassert>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"
#include "renderer/vulkan/GpuImage.h"

namespace renderer {

    namespace {

        // Byte-for-byte layout match for HZBPushConstants in HZBBuildInit.comp / HZBReduce.comp:
        // two uvec2 (srcSize, dstSize), 16 bytes total, no padding needed since uint32_t x 4 is
        // already 4-byte aligned throughout and uvec2's own alignment (8 bytes) is satisfied by
        // both fields starting on an 8-byte boundary.
        struct HZBPushConstants {
            uint32_t srcWidth;
            uint32_t srcHeight;
            uint32_t dstWidth;
            uint32_t dstHeight;
        };
        static_assert(sizeof(HZBPushConstants) == 16,
            "HZBPushConstants must match HZBBuildInit.comp/HZBReduce.comp's push_constant block exactly");

        // Next mip dimension along one axis: ceil(dim / 2), floored at 1 so the pyramid always
        // terminates at a 1x1 mip instead of reaching 0.
        uint32_t NextMipDim(uint32_t dim) {
            return std::max(1u, (dim + 1) / 2);
        }

        // floor(log2(dim)) for dim >= 1 (0 for dim == 1). Used to cap the mip chain at the exact
        // count VkImageCreateInfo::mipLevels is allowed to request (see MaxMipCountForExtent) --
        // computed by repeated right-shift rather than std::log2 to avoid any floating-point
        // rounding landing on the wrong side of a power-of-two boundary.
        uint32_t FloorLog2(uint32_t dim) {
            uint32_t levels = 0;
            while (dim > 1u) {
                dim >>= 1;
                ++levels;
            }
            return levels;
        }

        // The Vulkan spec (VUID-VkImageCreateInfo-mipLevels-00958) mandates mipLevels <=
        // floor(log2(max(width, height))) + 1 -- the count of the *standard* floor-halving mip
        // chain (960 -> 480 -> ... -> 1, 10 levels). NextMipDim's ceil-halving above walks a
        // slightly different sequence (e.g. 15 -> 8 -> 4 -> ... instead of 15 -> 7 -> 3 -> ...),
        // which can take one extra step to reach 1x1 whenever an odd dimension is halved -- e.g.
        // starting from 960x540 the ceil chain needs 11 steps, one more than the 10 the spec
        // allows for that extent, and vkCreateImage rejects it outright. Capping the loop below at
        // this count (even if the last level generated isn't literally 1x1 yet) keeps every
        // requested mipLevels value spec-legal; a final mip slightly larger than 1x1 texel is
        // harmless for HZB occlusion queries, which only ever pick a coarse-enough level, never
        // require an exact 1x1 apex.
        uint32_t MaxMipCountForExtent(VkExtent2D extent) {
            return FloorLog2(std::max(extent.width, extent.height)) + 1u;
        }

        constexpr uint32_t kWorkgroupSize = 8;

        uint32_t DispatchGroupCount(uint32_t dim) {
            return (dim + kWorkgroupSize - 1) / kWorkgroupSize;
        }

    } // namespace

    void HZBPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkImageView sourceDepthView, VkExtent2D sourceDepthExtent) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_SourceExtent = sourceDepthExtent;

        // --- Mip chain sizing: mip 0 is half the source depth resolution; every following mip
        // halves the one before it until reaching 1x1 (inclusive), capped at
        // MaxMipCountForExtent(mip0) -- see that function's comment for why the ceil-halving
        // sequence below can otherwise request one more level than VkImageCreateInfo::mipLevels
        // is legally allowed to have for this extent. The cap is the *only* new behavior here: for
        // any extent whose ceil-halving chain does not include an odd intermediate dimension, it
        // never triggers and this loop is unchanged from before. ---
        VkExtent2D mip0{ NextMipDim(sourceDepthExtent.width), NextMipDim(sourceDepthExtent.height) };
        uint32_t maxMipCount = MaxMipCountForExtent(mip0);
        m_MipExtents.clear();
        VkExtent2D cur = mip0;
        for (;;) {
            m_MipExtents.push_back(cur);
            if ((cur.width == 1 && cur.height == 1) || m_MipExtents.size() >= maxMipCount) {
                break;
            }
            cur = VkExtent2D{ NextMipDim(cur.width), NextMipDim(cur.height) };
        }
        uint32_t mipCount = static_cast<uint32_t>(m_MipExtents.size());
        uint32_t reduceLevelCount = mipCount - 1; // Levels [1, mipCount) built by HZBReduce.comp.

        // --- HZB image + per-mip views ---
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kFormat;
        imageInfo.extent = { mip0.width, mip0.height, 1 };
        imageInfo.mipLevels = mipCount;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // STORAGE_BIT: both HZB shaders read/write mip levels via imageLoad/imageStore.
        // SAMPLED_BIT: lets a future occlusion-culling pass sample GetFullView() with textureLod
        // instead of having to imageLoad an explicit mip.
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        m_HZBImage.Create(allocator, device, imageInfo, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_ASPECT_COLOR_BIT);

        m_MipViews.resize(mipCount);
        for (uint32_t level = 0; level < mipCount; ++level) {
            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_HZBImage.Image();
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = kFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = level;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_MipViews[level]));
        }

        // --- One-time UNDEFINED -> GENERAL transition, covering every mip level at once. The HZB
        // image never leaves GENERAL after this (see the class comment), so this is the only
        // layout transition it will ever undergo. ---
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.srcAccessMask = 0;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_HZBImage.Image();
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipCount, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // --- Sampler used to read the source depth buffer in HZBBuildInit.comp. Nearest filtering
        // and clamp-to-edge: the shader always reads exact integer texels via texelFetch (no
        // filtering ever actually happens), clamp-to-edge is just defensive since every sample
        // coordinate is already clamped in-shader before the fetch. ---
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE; // Plain sampler2D read, not a shadow/compare sampler.
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_DepthSampler));

        // --- Descriptor set layouts: one for the init pass (source depth sampler + mip 0 storage
        // image), one for the reduce pass (src mip storage image + dst mip storage image). ---
        VkDescriptorSetLayoutBinding initBindings[2]{};
        initBindings[0].binding = 0;
        initBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        initBindings[0].descriptorCount = 1;
        initBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        initBindings[1].binding = 1;
        initBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        initBindings[1].descriptorCount = 1;
        initBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo initLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        initLayoutInfo.bindingCount = 2;
        initLayoutInfo.pBindings = initBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &initLayoutInfo, nullptr, &m_InitSetLayout));

        VkDescriptorSetLayoutBinding reduceBindings[2]{};
        reduceBindings[0].binding = 0;
        reduceBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        reduceBindings[0].descriptorCount = 1;
        reduceBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        reduceBindings[1].binding = 1;
        reduceBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        reduceBindings[1].descriptorCount = 1;
        reduceBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo reduceLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        reduceLayoutInfo.bindingCount = 2;
        reduceLayoutInfo.pBindings = reduceBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &reduceLayoutInfo, nullptr, &m_ReduceSetLayout));

        // --- Descriptor pool: 1 init set (1 combined sampler + 1 storage image) plus
        // reduceLevelCount reduce sets (2 storage images each). ---
        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 + 2 * reduceLevelCount };

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1 + reduceLevelCount;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo initSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        initSetAlloc.descriptorPool = m_DescriptorPool;
        initSetAlloc.descriptorSetCount = 1;
        initSetAlloc.pSetLayouts = &m_InitSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &initSetAlloc, &m_InitSet));

        VkDescriptorImageInfo sourceDepthInfo{};
        sourceDepthInfo.sampler = m_DepthSampler;
        sourceDepthInfo.imageView = sourceDepthView;
        sourceDepthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo mip0Info{};
        mip0Info.imageView = m_MipViews[0];
        mip0Info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet initWrites[2]{};
        initWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        initWrites[0].dstSet = m_InitSet;
        initWrites[0].dstBinding = 0;
        initWrites[0].descriptorCount = 1;
        initWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        initWrites[0].pImageInfo = &sourceDepthInfo;

        initWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        initWrites[1].dstSet = m_InitSet;
        initWrites[1].dstBinding = 1;
        initWrites[1].descriptorCount = 1;
        initWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        initWrites[1].pImageInfo = &mip0Info;

        vkUpdateDescriptorSets(m_Device, 2, initWrites, 0, nullptr);

        // One descriptor set per reduce step (level -> level+1), each binding the exact pair of
        // single-mip views that step reads from / writes to. Allocated and written once here since
        // the image never gets new views for the lifetime of this object.
        m_ReduceSets.resize(reduceLevelCount);
        if (reduceLevelCount > 0) {
            std::vector<VkDescriptorSetLayout> reduceLayouts(reduceLevelCount, m_ReduceSetLayout);
            VkDescriptorSetAllocateInfo reduceSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            reduceSetAlloc.descriptorPool = m_DescriptorPool;
            reduceSetAlloc.descriptorSetCount = reduceLevelCount;
            reduceSetAlloc.pSetLayouts = reduceLayouts.data();
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &reduceSetAlloc, m_ReduceSets.data()));

            // Image infos must outlive the vkUpdateDescriptorSets call below -- kept in a vector
            // sized up front so no reallocation invalidates earlier pointers into it.
            std::vector<VkDescriptorImageInfo> srcInfos(reduceLevelCount);
            std::vector<VkDescriptorImageInfo> dstInfos(reduceLevelCount);
            std::vector<VkWriteDescriptorSet> writes(reduceLevelCount * 2);

            for (uint32_t i = 0; i < reduceLevelCount; ++i) {
                uint32_t srcLevel = i;
                uint32_t dstLevel = i + 1;

                srcInfos[i].imageView = m_MipViews[srcLevel];
                srcInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                dstInfos[i].imageView = m_MipViews[dstLevel];
                dstInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

                writes[2 * i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[2 * i].dstSet = m_ReduceSets[i];
                writes[2 * i].dstBinding = 0;
                writes[2 * i].descriptorCount = 1;
                writes[2 * i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[2 * i].pImageInfo = &srcInfos[i];

                writes[2 * i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[2 * i + 1].dstSet = m_ReduceSets[i];
                writes[2 * i + 1].dstBinding = 1;
                writes[2 * i + 1].descriptorCount = 1;
                writes[2 * i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[2 * i + 1].pImageInfo = &dstInfos[i];
            }

            vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        // --- Pipeline layouts + pipelines ---
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(HZBPushConstants);

        VkPipelineLayoutCreateInfo initPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        initPipelineLayoutInfo.setLayoutCount = 1;
        initPipelineLayoutInfo.pSetLayouts = &m_InitSetLayout;
        initPipelineLayoutInfo.pushConstantRangeCount = 1;
        initPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &initPipelineLayoutInfo, nullptr, &m_InitPipelineLayout));

        VkPipelineLayoutCreateInfo reducePipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        reducePipelineLayoutInfo.setLayoutCount = 1;
        reducePipelineLayoutInfo.pSetLayouts = &m_ReduceSetLayout;
        reducePipelineLayoutInfo.pushConstantRangeCount = 1;
        reducePipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &reducePipelineLayoutInfo, nullptr, &m_ReducePipelineLayout));

        VkShaderModule initShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/HZBBuildInit.comp.spv");

        VkComputePipelineCreateInfo initPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        initPipelineInfo.layout = m_InitPipelineLayout;
        initPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        initPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        initPipelineInfo.stage.module = initShaderModule;
        initPipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &initPipelineInfo, nullptr, &m_InitPipeline));
        vkDestroyShaderModule(m_Device, initShaderModule, nullptr);

        VkShaderModule reduceShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/HZBReduce.comp.spv");


        VkComputePipelineCreateInfo reducePipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        reducePipelineInfo.layout = m_ReducePipelineLayout;
        reducePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        reducePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        reducePipelineInfo.stage.module = reduceShaderModule;
        reducePipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &reducePipelineInfo, nullptr, &m_ReducePipeline));
        vkDestroyShaderModule(m_Device, reduceShaderModule, nullptr);

        LOG_INFO(std::format("[HZBPass] Initialized HZB pass: sourceDepth={}x{}, mip0={}x{}, mipCount={}",
            sourceDepthExtent.width, sourceDepthExtent.height, mip0.width, mip0.height, mipCount));
    }

    void HZBPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[HZBPass] Shutting down HZB pass...");
            if (m_InitPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_InitPipeline, nullptr);
            }
            if (m_ReducePipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_Device, m_ReducePipeline, nullptr);
            }
            if (m_InitPipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_Device, m_InitPipelineLayout, nullptr);
            }
            if (m_ReducePipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_Device, m_ReducePipelineLayout, nullptr);
            }
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_InitSet and every entry of m_ReduceSets --
                // none of them are individually freed.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_InitSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_Device, m_InitSetLayout, nullptr);
            }
            if (m_ReduceSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_Device, m_ReduceSetLayout, nullptr);
            }
            if (m_DepthSampler != VK_NULL_HANDLE) {
                vkDestroySampler(m_Device, m_DepthSampler, nullptr);
            }
            for (VkImageView view : m_MipViews) {
                vkDestroyImageView(m_Device, view, nullptr);
            }
        }

        m_InitPipeline = VK_NULL_HANDLE;
        m_ReducePipeline = VK_NULL_HANDLE;
        m_InitPipelineLayout = VK_NULL_HANDLE;
        m_ReducePipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_InitSetLayout = VK_NULL_HANDLE;
        m_ReduceSetLayout = VK_NULL_HANDLE;
        m_InitSet = VK_NULL_HANDLE;
        m_ReduceSets.clear();
        m_DepthSampler = VK_NULL_HANDLE;
        m_MipViews.clear();
        m_MipExtents.clear();
        m_SourceExtent = { 0, 0 };

        m_HZBImage.Destroy();
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void HZBPass::BarrierAfterMipWrite(VkCommandBuffer cmd, uint32_t mipLevel,
        VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const {
        VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        // No layout change: the HZB image stays in GENERAL for its entire lifetime (see the class
        // comment), so this barrier is a pure execution/memory dependency.
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_HZBImage.Image();
        barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, 1, 0, 1 };

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void HZBPass::Generate(VkCommandBuffer cmd) {
        assert(!m_MipExtents.empty() && "HZBPass::Generate called before Init");

        // --- Step 1: mip 0 = 2x2 min/max reduction of the source depth buffer ---
        HZBPushConstants initPC{ m_SourceExtent.width, m_SourceExtent.height, m_MipExtents[0].width, m_MipExtents[0].height };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_InitPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_InitPipelineLayout, 0, 1, &m_InitSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_InitPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(initPC), &initPC);
        vkCmdDispatch(cmd, DispatchGroupCount(m_MipExtents[0].width), DispatchGroupCount(m_MipExtents[0].height), 1);

        // Makes mip 0's write visible either to step 2's reduce read (if mipCount > 1) or directly
        // to a future occlusion-culling read (if the pyramid has only this one level) -- both
        // consumers are VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT / VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        // so one barrier shape covers both cases.
        BarrierAfterMipWrite(cmd, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        // --- Steps 2..mipCount: each mip level = 2x2 min/max reduction of the mip directly below it ---
        if (m_MipExtents.size() > 1) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ReducePipeline);

            for (uint32_t level = 1; level < m_MipExtents.size(); ++level) {
                VkDescriptorSet reduceSet = m_ReduceSets[level - 1];
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ReducePipelineLayout, 0, 1, &reduceSet, 0, nullptr);

                HZBPushConstants reducePC{
                    m_MipExtents[level - 1].width, m_MipExtents[level - 1].height,
                    m_MipExtents[level].width, m_MipExtents[level].height
                };
                vkCmdPushConstants(cmd, m_ReducePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(reducePC), &reducePC);
                vkCmdDispatch(cmd, DispatchGroupCount(m_MipExtents[level].width), DispatchGroupCount(m_MipExtents[level].height), 1);

                // Same reasoning as mip 0's barrier: visible to either the next reduce step's read
                // or, on the last level, a future occlusion-culling read.
                BarrierAfterMipWrite(cmd, level, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            }
        }
    }

}
