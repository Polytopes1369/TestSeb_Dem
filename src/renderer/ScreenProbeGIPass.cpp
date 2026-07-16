#include "renderer/ScreenProbeGIPass.h"

#include <array>
#include <format>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "core/Logger.h"
#include "renderer/ClusterResolvePass.h"
#include "renderer/SurfaceCachePass.h"
#include "renderer/SurfaceCacheRayTracingPass.h"
#include "renderer/SurfaceCacheTraceContext.h"

namespace renderer {

    namespace {

        // Mirrors every other pass's own copy -- see GlobalSDFPass.cpp's own comment on this
        // codebase's per-pass self-containment convention.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("ScreenProbeGIPass: failed to open SPIR-V file: " + filename);
            }
            size_t fileSize = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            file.close();
            return buffer;
        }

        VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code) {
            VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &module));
            return module;
        }

        void TransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
            VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = srcStage;
            barrier.srcAccessMask = srcAccess;
            barrier.dstStageMask = dstStage;
            barrier.dstAccessMask = dstAccess;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // Byte-for-byte std140 mirror of ScreenProbeViewParamsUBO in every ScreenProbe*.comp shader.
        struct ScreenProbeViewParamsUBO {
            maths::mat4 invViewProj;
            maths::mat4 prevViewProj;
            float viewportWidth = 0.0f;
            float viewportHeight = 0.0f;
            float probeCountX = 0.0f;
            float probeCountY = 0.0f;
        };
        static_assert(sizeof(ScreenProbeViewParamsUBO) == 144,
            "ScreenProbeViewParamsUBO must match every ScreenProbe*.comp shader's UBO exactly (std140 layout)");

        struct TracePushConstants {
            uint32_t probeCountX = 0, probeCountY = 0;
            uint32_t entityCount = 0;
            uint32_t traceMode = 0;
            uint32_t frameIndex = 0;
        };
        static_assert(sizeof(TracePushConstants) == 20, "TracePushConstants must match ScreenProbeTrace.comp's push_constant block exactly");

        struct TemporalPushConstants {
            uint32_t probeCountX = 0, probeCountY = 0;
        };
        static_assert(sizeof(TemporalPushConstants) == 8, "TemporalPushConstants must match ScreenProbeTemporal.comp's push_constant block exactly");

        struct GatherPushConstants {
            uint32_t probeCountX = 0, probeCountY = 0;
#ifndef NDEBUG
            uint32_t debugViewMode = 0;
#endif
        };

        // One probe-field image: STORAGE_BIT (RecordTrace's/RecordTemporal's imageLoad/imageStore)
        // | SAMPLED_BIT (history reprojection + bilateral gather taps, both via texelFetch/texture
        // through m_ProbeSampler) | TRANSFER_DST_BIT (the one-time neutral-default clear below).
        void CreateProbeImage(VmaAllocator allocator, VkDevice device, VkFormat format, VkExtent2D extent,
            VkImage& outImage, VmaAllocation& outAllocation, VkImageView& outView) {
            VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = format;
            imageInfo.extent = { extent.width, extent.height, 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage, &outAllocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = outImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &outView));
        }

    } // namespace

    bool ScreenProbeGIPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent, const SurfaceCacheTraceContext& traceContext, const SurfaceCachePass& surfaceCache,
        const SurfaceCacheRayTracingPass& rtPass, const ClusterResolvePass& resolvePass) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;
        m_ProbeCountX = (renderExtent.width + kProbeTileSize - 1u) / kProbeTileSize;
        m_ProbeCountY = (renderExtent.height + kProbeTileSize - 1u) / kProbeTileSize;
        VkExtent2D probeExtent{ m_ProbeCountX, m_ProbeCountY };

        // =====================================================================================
        // STEP 1 -- Probe ping-pong images (2 slots x 5 fields) + shared sampler + view-params UBO.
        // =====================================================================================
        for (ProbeSlot& slot : m_Slots) {
            CreateProbeImage(allocator, device, kSHFormat, probeExtent, slot.shRImage, slot.shRAllocation, slot.shRView);
            CreateProbeImage(allocator, device, kSHFormat, probeExtent, slot.shGImage, slot.shGAllocation, slot.shGView);
            CreateProbeImage(allocator, device, kSHFormat, probeExtent, slot.shBImage, slot.shBAllocation, slot.shBView);
            CreateProbeImage(allocator, device, kWorldPosFormat, probeExtent, slot.worldPosImage, slot.worldPosAllocation, slot.worldPosView);
            CreateProbeImage(allocator, device, kNormalFormat, probeExtent, slot.normalImage, slot.normalAllocation, slot.normalView);
        }

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_ProbeSampler));

        m_ViewParamsBuffer.Create(allocator, sizeof(ScreenProbeViewParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // One-time UNDEFINED -> GENERAL transition + neutral-default clear, mirrors
        // renderer::SurfaceCachePass::Init's own STEP 3 pattern: SH channels start at zero (no
        // contribution yet), worldPos.a = 0 (never traced -- see kWorldPosFormat's own "validity"
        // comment), normal = oct-encoded +Y (an arbitrary but valid direction).
        {
            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkClearColorValue zeroClear{}; zeroClear.float32[0] = 0.0f; zeroClear.float32[1] = 0.0f; zeroClear.float32[2] = 0.0f; zeroClear.float32[3] = 0.0f;
            VkClearColorValue normalClear{}; normalClear.float32[0] = 0.5f; normalClear.float32[1] = 0.5f; normalClear.float32[2] = 0.0f; normalClear.float32[3] = 0.0f;
            VkImageSubresourceRange colorRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            for (ProbeSlot& slot : m_Slots) {
                struct { VkImage image; const VkClearColorValue* clear; } clears[5] = {
                    { slot.shRImage, &zeroClear }, { slot.shGImage, &zeroClear }, { slot.shBImage, &zeroClear },
                    { slot.worldPosImage, &zeroClear }, { slot.normalImage, &normalClear }
                };
                for (auto& entry : clears) {
                    TransitionImageLayout(cmd, entry.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                    vkCmdClearColorImage(cmd, entry.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, entry.clear, 1, &colorRange);
                    TransitionImageLayout(cmd, entry.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                }
            }

            vkEndCommandBuffer(cmd);
            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(m_Device, commandPool, 1, &cmd);
        }

        // =====================================================================================
        // STEP 2 -- Trace pipeline: set 0 (12 bindings, 2 slot-indexed variants) + set 1 (mesh SDF
        // trace scene) + set 2 (surface cache sampling), both shared unmodified from traceContext.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[12]{};
            for (uint32_t b = 0; b <= 6; ++b) {
                bindings[b].binding = b;
                bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                bindings[b].descriptorCount = 1;
                bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            bindings[11] = { 11, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 12;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_TraceSetLayout));

            VkDescriptorPoolSize poolSizes[4] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 7 * 2 },
                { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 * 2 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 4;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_TraceDescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_TraceSetLayout, m_TraceSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_TraceDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_TraceSet));

            VkAccelerationStructureKHR tlasHandle = rtPass.GetTLASHandle();
            VkDescriptorBufferInfo vertexBufferInfo{ surfaceCache.GetVertexBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo indexBufferInfo{ surfaceCache.GetIndexBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo drawRangeBufferInfo{ rtPass.GetDrawRangeBuffer(), 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                const ProbeSlot& slot = m_Slots[slotIndex];
                VkDescriptorImageInfo shRInfo{ VK_NULL_HANDLE, slot.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo shGInfo{ VK_NULL_HANDLE, slot.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo shBInfo{ VK_NULL_HANDLE, slot.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo worldPosInfo{ VK_NULL_HANDLE, slot.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo normalInfo{ VK_NULL_HANDLE, slot.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkWriteDescriptorSet writes[12]{};
                VkDescriptorImageInfo* imageInfos[7] = { &shRInfo, &shGInfo, &shBInfo, &worldPosInfo, &normalInfo, &gbufferNormalInfo, &gbufferDepthInfo };
                for (uint32_t b = 0; b <= 6; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, imageInfos[b], nullptr, nullptr };
                }
                VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
                accelWrite.accelerationStructureCount = 1;
                accelWrite.pAccelerationStructures = &tlasHandle;
                writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &accelWrite, m_TraceSet[slotIndex], 7, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr };
                writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vertexBufferInfo, nullptr };
                writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indexBufferInfo, nullptr };
                writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &drawRangeBufferInfo, nullptr };
                writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TraceSet[slotIndex], 11, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 12, writes, 0, nullptr);
            }

            VkDescriptorSetLayout traceSetLayouts[3] = { m_TraceSetLayout, traceContext.GetMeshSdfTraceSetLayout(), traceContext.GetSurfaceCacheSamplingSetLayout() };
            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TracePushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 3;
            pipelineLayoutInfo.pSetLayouts = traceSetLayouts;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_TracePipelineLayout));

            VkShaderModule shaderModule = CreateShaderModule(m_Device, ReadShaderFile("shaders/ScreenProbeTrace.comp.spv"));
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_TracePipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_TracePipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 3 -- Temporal pipeline: set 0 (11 bindings, 2 slot-indexed variants).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[11]{};
            for (uint32_t b = 0; b <= 4; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            for (uint32_t b = 5; b <= 9; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[10] = { 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 11;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_TemporalSetLayout));

            VkDescriptorPoolSize poolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 * 2 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 3;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_TemporalDescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_TemporalSetLayout, m_TemporalSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_TemporalDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_TemporalSet));

            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                const ProbeSlot& current = m_Slots[slotIndex];
                const ProbeSlot& history = m_Slots[1 - slotIndex];

                VkDescriptorImageInfo curSHR{ VK_NULL_HANDLE, current.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHG{ VK_NULL_HANDLE, current.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHB{ VK_NULL_HANDLE, current.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curWorldPos{ VK_NULL_HANDLE, current.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curNormal{ VK_NULL_HANDLE, current.normalView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histSHR{ m_ProbeSampler, history.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histSHG{ m_ProbeSampler, history.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histSHB{ m_ProbeSampler, history.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histWorldPos{ m_ProbeSampler, history.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo histNormal{ m_ProbeSampler, history.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkDescriptorImageInfo* storageInfos[5] = { &curSHR, &curSHG, &curSHB, &curWorldPos, &curNormal };
                VkDescriptorImageInfo* sampledInfos[5] = { &histSHR, &histSHG, &histSHB, &histWorldPos, &histNormal };

                VkWriteDescriptorSet writes[11]{};
                for (uint32_t b = 0; b <= 4; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b], nullptr, nullptr };
                }
                for (uint32_t b = 5; b <= 9; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampledInfos[b - 5], nullptr, nullptr };
                }
                writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_TemporalSet[slotIndex], 10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 11, writes, 0, nullptr);
            }

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TemporalPushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_TemporalSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_TemporalPipelineLayout));

            VkShaderModule shaderModule = CreateShaderModule(m_Device, ReadShaderFile("shaders/ScreenProbeTemporal.comp.spv"));
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_TemporalPipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_TemporalPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // STEP 4 -- Gather pipeline: set 0 (10 bindings, 2 slot-indexed variants).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[10]{};
            for (uint32_t b = 0; b <= 4; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            for (uint32_t b = 5; b <= 8; ++b) {
                bindings[b] = { b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            }
            bindings[9] = { 9, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 10;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_GatherSetLayout));

            VkDescriptorPoolSize poolSizes[3] = {
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 * 2 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * 2 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 * 2 }
            };
            VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            poolInfo.maxSets = 2;
            poolInfo.poolSizeCount = 3;
            poolInfo.pPoolSizes = poolSizes;
            VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_GatherDescriptorPool));

            VkDescriptorSetLayout setLayouts[2] = { m_GatherSetLayout, m_GatherSetLayout };
            VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAllocInfo.descriptorPool = m_GatherDescriptorPool;
            setAllocInfo.descriptorSetCount = 2;
            setAllocInfo.pSetLayouts = setLayouts;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, m_GatherSet));

            VkDescriptorImageInfo gbufferNormalInfo{ VK_NULL_HANDLE, resolvePass.GetOutputNormalView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferDepthInfo{ VK_NULL_HANDLE, resolvePass.GetOutputDepthView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo gbufferAlbedoInfo{ VK_NULL_HANDLE, resolvePass.GetOutputAlbedoView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, resolvePass.GetOutputColorView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

            for (uint32_t slotIndex = 0; slotIndex < 2; ++slotIndex) {
                const ProbeSlot& current = m_Slots[slotIndex];
                VkDescriptorImageInfo curSHR{ m_ProbeSampler, current.shRView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHG{ m_ProbeSampler, current.shGView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curSHB{ m_ProbeSampler, current.shBView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curWorldPos{ m_ProbeSampler, current.worldPosView, VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo curNormal{ m_ProbeSampler, current.normalView, VK_IMAGE_LAYOUT_GENERAL };

                VkDescriptorImageInfo* sampledInfos[5] = { &curSHR, &curSHG, &curSHB, &curWorldPos, &curNormal };
                VkDescriptorImageInfo* storageInfos[4] = { &gbufferNormalInfo, &gbufferDepthInfo, &gbufferAlbedoInfo, &outputColorInfo };

                VkWriteDescriptorSet writes[10]{};
                for (uint32_t b = 0; b <= 4; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sampledInfos[b], nullptr, nullptr };
                }
                for (uint32_t b = 5; b <= 8; ++b) {
                    writes[b] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageInfos[b - 5], nullptr, nullptr };
                }
                writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_GatherSet[slotIndex], 9, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
                vkUpdateDescriptorSets(m_Device, 10, writes, 0, nullptr);
            }

            VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GatherPushConstants) };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_GatherSetLayout;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_GatherPipelineLayout));

            VkShaderModule shaderModule = CreateShaderModule(m_Device, ReadShaderFile("shaders/ScreenProbeGather.comp.spv"));
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_GatherPipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_GatherPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        m_CurrentSlotIndex = 0;
        LOG_INFO(std::format("[ScreenProbeGIPass] Initialized: {} x {} probes.", m_ProbeCountX, m_ProbeCountY));
        return true;
    }

    void ScreenProbeGIPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_GatherPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_GatherPipeline, nullptr);
            if (m_GatherPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_GatherPipelineLayout, nullptr);
            if (m_GatherDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_GatherDescriptorPool, nullptr);
            if (m_GatherSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_GatherSetLayout, nullptr);

            if (m_TemporalPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_TemporalPipeline, nullptr);
            if (m_TemporalPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_TemporalPipelineLayout, nullptr);
            if (m_TemporalDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_TemporalDescriptorPool, nullptr);
            if (m_TemporalSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_TemporalSetLayout, nullptr);

            if (m_TracePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_TracePipeline, nullptr);
            if (m_TracePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_TracePipelineLayout, nullptr);
            if (m_TraceDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_TraceDescriptorPool, nullptr);
            if (m_TraceSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_TraceSetLayout, nullptr);

            if (m_ProbeSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ProbeSampler, nullptr);

            for (ProbeSlot& slot : m_Slots) {
                if (slot.shRView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shRView, nullptr);
                if (slot.shGView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shGView, nullptr);
                if (slot.shBView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.shBView, nullptr);
                if (slot.worldPosView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.worldPosView, nullptr);
                if (slot.normalView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, slot.normalView, nullptr);
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            for (ProbeSlot& slot : m_Slots) {
                vmaDestroyImage(m_Allocator, slot.shRImage, slot.shRAllocation);
                vmaDestroyImage(m_Allocator, slot.shGImage, slot.shGAllocation);
                vmaDestroyImage(m_Allocator, slot.shBImage, slot.shBAllocation);
                vmaDestroyImage(m_Allocator, slot.worldPosImage, slot.worldPosAllocation);
                vmaDestroyImage(m_Allocator, slot.normalImage, slot.normalAllocation);
            }
        }
        m_ViewParamsBuffer.Destroy();

        m_GatherPipeline = VK_NULL_HANDLE; m_GatherPipelineLayout = VK_NULL_HANDLE; m_GatherDescriptorPool = VK_NULL_HANDLE; m_GatherSetLayout = VK_NULL_HANDLE;
        m_GatherSet[0] = VK_NULL_HANDLE; m_GatherSet[1] = VK_NULL_HANDLE;
        m_TemporalPipeline = VK_NULL_HANDLE; m_TemporalPipelineLayout = VK_NULL_HANDLE; m_TemporalDescriptorPool = VK_NULL_HANDLE; m_TemporalSetLayout = VK_NULL_HANDLE;
        m_TemporalSet[0] = VK_NULL_HANDLE; m_TemporalSet[1] = VK_NULL_HANDLE;
        m_TracePipeline = VK_NULL_HANDLE; m_TracePipelineLayout = VK_NULL_HANDLE; m_TraceDescriptorPool = VK_NULL_HANDLE; m_TraceSetLayout = VK_NULL_HANDLE;
        m_TraceSet[0] = VK_NULL_HANDLE; m_TraceSet[1] = VK_NULL_HANDLE;
        m_ProbeSampler = VK_NULL_HANDLE;
        m_Slots[0] = ProbeSlot{};
        m_Slots[1] = ProbeSlot{};
        m_CurrentSlotIndex = 0;
        m_ProbeCountX = 0;
        m_ProbeCountY = 0;
        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ScreenProbeGIPass::RecordUpdateViewParams(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj) {
        ScreenProbeViewParamsUBO ubo{};
        ubo.invViewProj = viewProj.Inverse();
        ubo.prevViewProj = prevViewProj;
        ubo.viewportWidth = static_cast<float>(m_RenderExtent.width);
        ubo.viewportHeight = static_cast<float>(m_RenderExtent.height);
        ubo.probeCountX = static_cast<float>(m_ProbeCountX);
        ubo.probeCountY = static_cast<float>(m_ProbeCountY);
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void ScreenProbeGIPass::RecordTrace(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
        uint32_t entityCount, uint32_t traceMode, uint32_t frameIndex) {
        // This frame writes into whichever slot held the PREVIOUS frame's history -- see this
        // method's own header comment.
        m_CurrentSlotIndex = 1 - m_CurrentSlotIndex;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TracePipeline);
        VkDescriptorSet sets[3] = { m_TraceSet[m_CurrentSlotIndex], traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TracePipelineLayout, 0, 3, sets, 0, nullptr);

        TracePushConstants pc{};
        pc.probeCountX = m_ProbeCountX;
        pc.probeCountY = m_ProbeCountY;
        pc.entityCount = entityCount;
        pc.traceMode = traceMode;
        pc.frameIndex = frameIndex;
        vkCmdPushConstants(cmd, m_TracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        vkCmdDispatch(cmd, m_ProbeCountX, m_ProbeCountY, 1);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void ScreenProbeGIPass::RecordTemporal(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TemporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_TemporalPipelineLayout, 0, 1, &m_TemporalSet[m_CurrentSlotIndex], 0, nullptr);

        TemporalPushConstants pc{};
        pc.probeCountX = m_ProbeCountX;
        pc.probeCountY = m_ProbeCountY;
        vkCmdPushConstants(cmd, m_TemporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupCountX = (m_ProbeCountX + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        uint32_t groupCountY = (m_ProbeCountY + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    void ScreenProbeGIPass::RecordGather(VkCommandBuffer cmd
#ifndef NDEBUG
        , uint32_t debugViewMode
#endif
    ) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GatherPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GatherPipelineLayout, 0, 1, &m_GatherSet[m_CurrentSlotIndex], 0, nullptr);

        GatherPushConstants pc{};
        pc.probeCountX = m_ProbeCountX;
        pc.probeCountY = m_ProbeCountY;
#ifndef NDEBUG
        pc.debugViewMode = debugViewMode;
#endif
        vkCmdPushConstants(cmd, m_GatherPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        uint32_t groupCountX = (m_RenderExtent.width + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kGridWorkgroupSize - 1u) / kGridWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

}
