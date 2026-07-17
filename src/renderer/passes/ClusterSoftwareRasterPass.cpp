#include "renderer/passes/ClusterSoftwareRasterPass.h"

#include <cassert>
#include <format>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h" // geometry::kMaxClusterTriangles
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of SoftwareRasterViewParamsUBO in ClusterSoftwareRaster.comp
        // (std140): mat4 (64 bytes) + vec2 (8 bytes, 8-byte aligned) + 2 pad floats to round the
        // struct up to a multiple of its own 16-byte base alignment (80 bytes total).
        struct SoftwareRasterViewParams {
            maths::mat4 viewProj;
            float viewportWidth = 0.0f;
            float viewportHeight = 0.0f;
            float _pad0 = 0.0f;
            float _pad1 = 0.0f;
        };
        static_assert(sizeof(SoftwareRasterViewParams) == 80,
            "SoftwareRasterViewParams must match SoftwareRasterViewParamsUBO in ClusterSoftwareRaster.comp exactly (std140 layout)");

    } // namespace

    void ClusterSoftwareRasterPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
        VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer, VkBuffer softwareClusterListBuffer,
        VkBuffer softwareClusterListOpaqueBuffer, VkBuffer wpoGlobalsBuffer, const std::vector<VkDescriptorImageInfo>& maskImageInfos,
        VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer splineControlPointsBuffer) {
        Shutdown();
        uint32_t maskTextureCount = static_cast<uint32_t>(maskImageInfos.size());

        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;

        // --- Buffers ---
        m_ViewParamsBuffer.Create(
            allocator,
            sizeof(SoftwareRasterViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_DispatchArgsBuffer.Create(
            allocator,
            sizeof(VkDispatchIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        m_DispatchArgsOpaqueBuffer.Create(
            allocator,
            sizeof(VkDispatchIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // --- Atomic VisBuffer image: R64_UINT storage image, one (depth, visibilityID) word per
        // pixel -- see the class comment for why this format/usage differs from the hardware
        // path's two R32_UINT color attachments. ---
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kVisBufferAtomicFormat;
        imageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo imageAllocInfo{};
        imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &imageAllocInfo, &m_VisBufferAtomicImage, &m_VisBufferAtomicAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_VisBufferAtomicImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kVisBufferAtomicFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_VisBufferAtomicView));

        // --- One-time UNDEFINED -> GENERAL transition (mirrors HZBPass::Init's own one-shot
        // pattern) -- the image stays in GENERAL for its entire lifetime, touched only by compute
        // shaders (imageStore/imageAtomicMax), never an attachment or sampled resource. ---
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
            barrier.image = m_VisBufferAtomicImage;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // --- Descriptor set layouts ---
        VkDescriptorSetLayoutBinding rasterBindings[10]{};
        rasterBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // ClusterCullMetadataSSBO
        rasterBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // CompressedClusterPoolSSBO
        rasterBindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SoftwareClusterListSSBO
        rasterBindings[3] = { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SoftwareRasterViewParamsUBO
        rasterBindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_VisBufferAtomic (r64ui)
        rasterBindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // WPOGlobalsUBO
        rasterBindings[6] = { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_MaskTextures[]
        rasterBindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityTransformBuffer
        rasterBindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityDataBuffer
        rasterBindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SplineControlPointsSSBO (Phase 1, Nanite advanced)

        VkDescriptorSetLayoutCreateInfo rasterLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        rasterLayoutInfo.bindingCount = 10;
        rasterLayoutInfo.pBindings = rasterBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &rasterLayoutInfo, nullptr, &m_RasterSetLayout));

        // Opaque raster set: identical to bindings 0-5 above, minus binding 6 (the mask array) --
        // ClusterSoftwareRasterOpaque.comp never references it at all.
        VkDescriptorSetLayoutBinding opaqueRasterBindings[9]{};
        opaqueRasterBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // ClusterCullMetadataSSBO
        opaqueRasterBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // CompressedClusterPoolSSBO
        opaqueRasterBindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SoftwareClusterListSSBO (opaque list)
        opaqueRasterBindings[3] = { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SoftwareRasterViewParamsUBO
        opaqueRasterBindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_VisBufferAtomic (r64ui)
        opaqueRasterBindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // WPOGlobalsUBO
        opaqueRasterBindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityTransformBuffer
        opaqueRasterBindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityDataBuffer
        opaqueRasterBindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SplineControlPointsSSBO (Phase 1, Nanite advanced)

        VkDescriptorSetLayoutCreateInfo opaqueRasterLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        opaqueRasterLayoutInfo.bindingCount = 9;
        opaqueRasterLayoutInfo.pBindings = opaqueRasterBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &opaqueRasterLayoutInfo, nullptr, &m_OpaqueRasterSetLayout));

        VkDescriptorSetLayoutBinding clearBindings[1]{};
        clearBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_VisBufferAtomic (r64ui)

        VkDescriptorSetLayoutCreateInfo clearLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        clearLayoutInfo.bindingCount = 1;
        clearLayoutInfo.pBindings = clearBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &clearLayoutInfo, nullptr, &m_ClearSetLayout));

        m_BuildArgsSetLayout = VulkanPipeline::CreateBuildDispatchIndirectArgsSetLayout(m_Device);

        // --- One shared descriptor pool for all 5 sets ---
        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 + 6 + 2 + 2 };  // Raster(6) + OpaqueRaster(6) + BuildArgs(2) + OpaqueBuildArgs(2) SSBOs (each raster set gained a SplineControlPointsSSBO binding, Phase 1 Nanite advanced).
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 + 2 };          // Raster's + OpaqueRaster's view-params UBO + WPOGlobalsUBO.
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 + 1 + 1 };       // Raster's + OpaqueRaster's + Clear's g_VisBufferAtomic.
        poolSizes[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount }; // Raster set's g_MaskTextures[] (opaque set has none).

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 5;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo rasterSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        rasterSetAlloc.descriptorPool = m_DescriptorPool;
        rasterSetAlloc.descriptorSetCount = 1;
        rasterSetAlloc.pSetLayouts = &m_RasterSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &rasterSetAlloc, &m_RasterDescriptorSet));

        VkDescriptorSetAllocateInfo opaqueRasterSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        opaqueRasterSetAlloc.descriptorPool = m_DescriptorPool;
        opaqueRasterSetAlloc.descriptorSetCount = 1;
        opaqueRasterSetAlloc.pSetLayouts = &m_OpaqueRasterSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &opaqueRasterSetAlloc, &m_OpaqueRasterDescriptorSet));

        VkDescriptorSetAllocateInfo clearSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        clearSetAlloc.descriptorPool = m_DescriptorPool;
        clearSetAlloc.descriptorSetCount = 1;
        clearSetAlloc.pSetLayouts = &m_ClearSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &clearSetAlloc, &m_ClearDescriptorSet));

        VkDescriptorSetAllocateInfo buildArgsSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        buildArgsSetAlloc.descriptorPool = m_DescriptorPool;
        buildArgsSetAlloc.descriptorSetCount = 1;
        buildArgsSetAlloc.pSetLayouts = &m_BuildArgsSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &buildArgsSetAlloc, &m_BuildArgsDescriptorSet));

        VkDescriptorSetAllocateInfo buildArgsOpaqueSetAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        buildArgsOpaqueSetAlloc.descriptorPool = m_DescriptorPool;
        buildArgsOpaqueSetAlloc.descriptorSetCount = 1;
        buildArgsOpaqueSetAlloc.pSetLayouts = &m_BuildArgsSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &buildArgsOpaqueSetAlloc, &m_BuildArgsOpaqueDescriptorSet));

        // --- Raster set descriptor writes ---
        VkDescriptorBufferInfo clusterMetadataInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo compressedPoolInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo softwareListInfo{ softwareClusterListBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo entityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo splineControlPointsInfo{ splineControlPointsBuffer, 0, VK_WHOLE_SIZE };

        VkDescriptorImageInfo atomicImageInfo{};
        atomicImageInfo.imageView = m_VisBufferAtomicView;
        atomicImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo wpoGlobalsInfo{ wpoGlobalsBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet rasterWrites[10]{};
        rasterWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetadataInfo, nullptr };
        rasterWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
        rasterWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &softwareListInfo, nullptr };
        rasterWrites[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        rasterWrites[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &atomicImageInfo, nullptr, nullptr };
        rasterWrites[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &wpoGlobalsInfo, nullptr };
        rasterWrites[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 6, 0, maskTextureCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskImageInfos.data(), nullptr, nullptr };
        rasterWrites[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
        rasterWrites[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
        rasterWrites[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_RasterDescriptorSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &splineControlPointsInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 10, rasterWrites, 0, nullptr);

        // --- Opaque raster set descriptor writes -- same shapes as the masked set's bindings 0-5,
        // bound to the opaque software cluster list and sharing the same view-params/WPO-globals/
        // atomic-image resources (no separate copies needed: both dispatches rasterize into the
        // same VisBuffer against the same frame's camera). No binding 6 (mask array) here. ---
        VkDescriptorBufferInfo softwareListOpaqueInfo{ softwareClusterListOpaqueBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet opaqueRasterWrites[9]{};
        opaqueRasterWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetadataInfo, nullptr };
        opaqueRasterWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
        opaqueRasterWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &softwareListOpaqueInfo, nullptr };
        opaqueRasterWrites[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        opaqueRasterWrites[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &atomicImageInfo, nullptr, nullptr };
        opaqueRasterWrites[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &wpoGlobalsInfo, nullptr };
        opaqueRasterWrites[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
        opaqueRasterWrites[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
        opaqueRasterWrites[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_OpaqueRasterDescriptorSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &splineControlPointsInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 9, opaqueRasterWrites, 0, nullptr);

        VkWriteDescriptorSet clearWrites[1]{};
        clearWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ClearDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &atomicImageInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 1, clearWrites, 0, nullptr);

        // Binding 0 aliases softwareClusterListBuffer, reading only its leading count word (see
        // BuildDispatchIndirectArgs.comp's SourceCountSSBO).
        VkDescriptorBufferInfo sourceCountInfo{ softwareClusterListBuffer, 0, sizeof(uint32_t) };
        VkDescriptorBufferInfo dispatchArgsInfo{ m_DispatchArgsBuffer.Handle(), 0, m_DispatchArgsBuffer.Size() };

        VkDescriptorBufferInfo sourceCountOpaqueInfo{ softwareClusterListOpaqueBuffer, 0, sizeof(uint32_t) };
        VkDescriptorBufferInfo dispatchArgsOpaqueInfo{ m_DispatchArgsOpaqueBuffer.Handle(), 0, m_DispatchArgsOpaqueBuffer.Size() };

        VkWriteDescriptorSet buildArgsOpaqueWrites[2]{};
        buildArgsOpaqueWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsOpaqueDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sourceCountOpaqueInfo, nullptr };
        buildArgsOpaqueWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsOpaqueDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dispatchArgsOpaqueInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 2, buildArgsOpaqueWrites, 0, nullptr);

        VkWriteDescriptorSet buildArgsWrites[2]{};
        buildArgsWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sourceCountInfo, nullptr };
        buildArgsWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_BuildArgsDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dispatchArgsInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 2, buildArgsWrites, 0, nullptr);

        // --- Pipeline layouts + pipelines ---
        VkPushConstantRange rasterPushConstantRange{};
        rasterPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        rasterPushConstantRange.offset = 0;
        rasterPushConstantRange.size = sizeof(uint32_t); // Matches SoftwareRasterPushConstants::maxTrianglesPerCluster.

        VkPipelineLayoutCreateInfo rasterPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        rasterPipelineLayoutInfo.setLayoutCount = 1;
        rasterPipelineLayoutInfo.pSetLayouts = &m_RasterSetLayout;
        rasterPipelineLayoutInfo.pushConstantRangeCount = 1;
        rasterPipelineLayoutInfo.pPushConstantRanges = &rasterPushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &rasterPipelineLayoutInfo, nullptr, &m_RasterPipelineLayout));

        VkShaderModule rasterShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterSoftwareRaster.comp.spv");
        m_RasterPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_RasterPipelineLayout, rasterShaderModule);
        vkDestroyShaderModule(m_Device, rasterShaderModule, nullptr);

        // Opaque raster pipeline: same push-constant shape, its own (smaller) set layout, own shader.
        VkPushConstantRange opaqueRasterPushConstantRange = rasterPushConstantRange;

        VkPipelineLayoutCreateInfo opaqueRasterPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        opaqueRasterPipelineLayoutInfo.setLayoutCount = 1;
        opaqueRasterPipelineLayoutInfo.pSetLayouts = &m_OpaqueRasterSetLayout;
        opaqueRasterPipelineLayoutInfo.pushConstantRangeCount = 1;
        opaqueRasterPipelineLayoutInfo.pPushConstantRanges = &opaqueRasterPushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &opaqueRasterPipelineLayoutInfo, nullptr, &m_OpaqueRasterPipelineLayout));

        VkShaderModule opaqueRasterShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterSoftwareRasterOpaque.comp.spv");
        m_OpaqueRasterPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_OpaqueRasterPipelineLayout, opaqueRasterShaderModule);
        vkDestroyShaderModule(m_Device, opaqueRasterShaderModule, nullptr);

        VkPushConstantRange clearPushConstantRange{};
        clearPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        clearPushConstantRange.offset = 0;
        clearPushConstantRange.size = 2 * sizeof(uint32_t); // Matches ClearPushConstants::imageSize (uvec2).

        VkPipelineLayoutCreateInfo clearPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        clearPipelineLayoutInfo.setLayoutCount = 1;
        clearPipelineLayoutInfo.pSetLayouts = &m_ClearSetLayout;
        clearPipelineLayoutInfo.pushConstantRangeCount = 1;
        clearPipelineLayoutInfo.pPushConstantRanges = &clearPushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &clearPipelineLayoutInfo, nullptr, &m_ClearPipelineLayout));

        VkShaderModule clearShaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClearVisBufferAtomic.comp.spv");
        m_ClearPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_ClearPipelineLayout, clearShaderModule);
        vkDestroyShaderModule(m_Device, clearShaderModule, nullptr);

        VulkanPipeline::CreateBuildDispatchIndirectArgsPipeline(
            m_Device, m_BuildArgsSetLayout, m_BuildArgsPipelineLayout, m_BuildArgsPipeline);

        LOG_INFO(std::format("[ClusterSoftwareRasterPass] Initialized software raster pass: size={}x{}, maskTextures={}",
            renderExtent.width, renderExtent.height, maskTextureCount));
    }

    void ClusterSoftwareRasterPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[ClusterSoftwareRasterPass] Shutting down software raster pass...");
            if (m_RasterPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_RasterPipeline, nullptr);
            if (m_OpaqueRasterPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_OpaqueRasterPipeline, nullptr);
            if (m_ClearPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ClearPipeline, nullptr);
            if (m_BuildArgsPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_BuildArgsPipeline, nullptr);
            if (m_RasterPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_RasterPipelineLayout, nullptr);
            if (m_OpaqueRasterPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_OpaqueRasterPipelineLayout, nullptr);
            if (m_ClearPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ClearPipelineLayout, nullptr);
            if (m_BuildArgsPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_BuildArgsPipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees every set allocated from it.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_RasterSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RasterSetLayout, nullptr);
            if (m_OpaqueRasterSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_OpaqueRasterSetLayout, nullptr);
            if (m_ClearSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ClearSetLayout, nullptr);
            if (m_BuildArgsSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_BuildArgsSetLayout, nullptr);
            if (m_VisBufferAtomicView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_VisBufferAtomicView, nullptr);
        }

        m_RasterPipeline = VK_NULL_HANDLE;
        m_OpaqueRasterPipeline = VK_NULL_HANDLE;
        m_ClearPipeline = VK_NULL_HANDLE;
        m_BuildArgsPipeline = VK_NULL_HANDLE;
        m_RasterPipelineLayout = VK_NULL_HANDLE;
        m_OpaqueRasterPipelineLayout = VK_NULL_HANDLE;
        m_ClearPipelineLayout = VK_NULL_HANDLE;
        m_BuildArgsPipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_RasterSetLayout = VK_NULL_HANDLE;
        m_OpaqueRasterSetLayout = VK_NULL_HANDLE;
        m_ClearSetLayout = VK_NULL_HANDLE;
        m_BuildArgsSetLayout = VK_NULL_HANDLE;
        m_RasterDescriptorSet = VK_NULL_HANDLE;
        m_OpaqueRasterDescriptorSet = VK_NULL_HANDLE;
        m_ClearDescriptorSet = VK_NULL_HANDLE;
        m_BuildArgsDescriptorSet = VK_NULL_HANDLE;
        m_BuildArgsOpaqueDescriptorSet = VK_NULL_HANDLE;
        m_VisBufferAtomicView = VK_NULL_HANDLE;

        m_ViewParamsBuffer.Destroy();
        m_DispatchArgsBuffer.Destroy();
        m_DispatchArgsOpaqueBuffer.Destroy();

        // vmaDestroyImage tolerates VK_NULL_HANDLE for both handle arguments (no-op), matching
        // GpuBuffer::Destroy()'s own null-safe convention.
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_VisBufferAtomicImage, m_VisBufferAtomicAllocation);
        }
        m_VisBufferAtomicImage = VK_NULL_HANDLE;
        m_VisBufferAtomicAllocation = VK_NULL_HANDLE;

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterSoftwareRasterPass::RecordClear(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ClearPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ClearPipelineLayout, 0, 1, &m_ClearDescriptorSet, 0, nullptr);

        uint32_t imageSize[2] = { m_RenderExtent.width, m_RenderExtent.height };
        vkCmdPushConstants(cmd, m_ClearPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(imageSize), imageSize);

        uint32_t groupCountX = (m_RenderExtent.width + kClearWorkgroupSize - 1) / kClearWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kClearWorkgroupSize - 1) / kClearWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // The clear's write must be visible to RecordRaster()'s imageAtomicMax (both a read and a
        // write of the same texel).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    void ClusterSoftwareRasterPass::RecordRaster(VkCommandBuffer cmd, const maths::mat4& viewProj) {
        // --- Upload this frame's view-projection matrix + viewport size ---
        SoftwareRasterViewParams viewParams{};
        viewParams.viewProj = viewProj;
        viewParams.viewportWidth = static_cast<float>(m_RenderExtent.width);
        viewParams.viewportHeight = static_cast<float>(m_RenderExtent.height);
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(SoftwareRasterViewParams), &viewParams);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        // --- Build BOTH dispatches' indirect args from each list's own atomic count --
        // perElementMultiplier = geometry::kMaxClusterTriangles for both, since both
        // ClusterSoftwareRaster.comp and ClusterSoftwareRasterOpaque.comp dispatch one thread per
        // (cluster, triangleSlot) pair. Two separate 1x1x1 dispatches against the same
        // m_BuildArgsPipeline/m_BuildArgsPipelineLayout (the shader is generic over which
        // SourceCountSSBO/DispatchArgsSSBO pair its descriptor set binds), one per list. ---
        uint32_t buildArgsPushConstants[2] = { kWorkgroupSize, geometry::kMaxClusterTriangles };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipelineLayout, 0, 1, &m_BuildArgsDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_BuildArgsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(buildArgsPushConstants), buildArgsPushConstants);
        vkCmdDispatch(cmd, 1, 1, 1);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BuildArgsPipelineLayout, 0, 1, &m_BuildArgsOpaqueDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_BuildArgsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(buildArgsPushConstants), buildArgsPushConstants);
        vkCmdDispatch(cmd, 1, 1, 1);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);

        // --- Indirect rasterization dispatches: masked list against the mask-sampling pipeline,
        // opaque list against the mask-free pipeline. Both write the same atomic VisBuffer image;
        // imageAtomicMax makes the relative order between the two dispatches irrelevant to
        // correctness (see ClusterSoftwareRaster.comp's own class comment on the atomic packing). ---
        uint32_t maxTrianglesPerCluster = geometry::kMaxClusterTriangles;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_RasterPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_RasterPipelineLayout, 0, 1, &m_RasterDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_RasterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &maxTrianglesPerCluster);
        vkCmdDispatchIndirect(cmd, m_DispatchArgsBuffer.Handle(), 0);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_OpaqueRasterPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_OpaqueRasterPipelineLayout, 0, 1, &m_OpaqueRasterDescriptorSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_OpaqueRasterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &maxTrianglesPerCluster);
        vkCmdDispatchIndirect(cmd, m_DispatchArgsOpaqueBuffer.Handle(), 0);

        // Both rasterization dispatches' atomic writes must be visible to a later compute read
        // (renderer::ClusterResolvePass).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

}
