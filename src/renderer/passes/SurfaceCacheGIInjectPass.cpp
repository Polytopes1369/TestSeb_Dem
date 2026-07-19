#include "renderer/passes/SurfaceCacheGIInjectPass.h"

#include <algorithm>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h"
#include "renderer/passes/SurfaceCachePass.h"
#include "renderer/passes/SurfaceCacheRayTracingPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte layout match for GIInjectPushConstants in SurfaceCacheGIInject.comp.
        struct GIInjectPushConstants {
            uint32_t atlasOffsetX = 0, atlasOffsetY = 0;
            uint32_t atlasSizeX = 0, atlasSizeY = 0;
            uint32_t entityCount = 0;
            uint32_t sampleCount = 0;
            uint32_t frameIndex = 0;
            uint32_t traceMode = 0;
            // Atmos weather system, Subtask 5.
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f;
        };
        static_assert(sizeof(GIInjectPushConstants) == 44,
            "GIInjectPushConstants must match SurfaceCacheGIInject.comp's push_constant block exactly");

    } // namespace

    bool SurfaceCacheGIInjectPass::Init(VkDevice device, VmaAllocator /*allocator*/, const SurfaceCacheTraceContext& traceContext,
        const SurfaceCachePass& surfaceCache, const SurfaceCacheRayTracingPass& rtPass) {
        kCardsPerFrameBudget = config::lumen::CARDS_PER_FRAME_BUDGET;
        kSampleCountPerTexel = config::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;

        Shutdown();
        m_Device = device;

        // =====================================================================================
        // STEP 1 -- set 0's layout: the 10 bindings SurfaceCacheGIInject.comp declares (9 original
        // + binding 9, Atmos weather system Subtask 5's g_SkyViewLUT).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[10]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (uint32_t b = 1; b <= 4; ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        bindings[5].binding = 5;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        for (uint32_t b = 6; b <= 8; ++b) {
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        // Atmos weather system, Subtask 5 -- see SetAtmosSkyView()'s own comment.
        bindings[9].binding = 9;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 10;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[4] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 }, // + g_SkyViewLUT (Atmos Subtask 5).
            { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_Set));

        // =====================================================================================
        // STEP 2 -- write set 0: every binding here is a STATIC resource for this pass' whole
        // lifetime (the 5 Surface Cache atlas images, the shared TLAS, the Fallback Mesh
        // vertex/index/draw-range buffers) -- unlike SurfaceCacheSWRTPass/SurfaceCacheRayTracingPass's
        // set 0, nothing here is caller-supplied per call, so this is written once, here.
        // =====================================================================================
        VkDescriptorImageInfo radianceStorageInfo{ VK_NULL_HANDLE, surfaceCache.GetRadianceView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo albedoInfo{ surfaceCache.GetAtlasSampler(), surfaceCache.GetAlbedoView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo normalInfo{ surfaceCache.GetAtlasSampler(), surfaceCache.GetNormalView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo worldPosInfo{ surfaceCache.GetAtlasSampler(), surfaceCache.GetWorldPosView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo emissiveInfo{ surfaceCache.GetAtlasSampler(), surfaceCache.GetEmissiveView(), VK_IMAGE_LAYOUT_GENERAL };

        VkWriteDescriptorSet writes[5]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m_Set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].pImageInfo = &radianceStorageInfo;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = m_Set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[1].pImageInfo = &albedoInfo;

        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[2].dstSet = m_Set; writes[2].dstBinding = 2; writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[2].pImageInfo = &normalInfo;

        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[3].dstSet = m_Set; writes[3].dstBinding = 3; writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[3].pImageInfo = &worldPosInfo;

        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[4].dstSet = m_Set; writes[4].dstBinding = 4; writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[4].pImageInfo = &emissiveInfo;

        vkUpdateDescriptorSets(m_Device, 5, writes, 0, nullptr);

        VulkanUtils::WriteSharedGeometryBindings(m_Device, m_Set, 5, rtPass.GetTLASHandle(),
            surfaceCache.GetVertexBuffer(), surfaceCache.GetIndexBuffer(), rtPass.GetDrawRangeBuffer());

        // =====================================================================================
        // STEP 3 -- pipeline layout (set 0 above + set 1 mesh SDF trace scene + set 2 surface
        // cache sampling, both shared unmodified from traceContext) + compute pipeline.
        // =====================================================================================
        VkDescriptorSetLayout setLayouts[3] = {
            m_SetLayout,
            traceContext.GetMeshSdfTraceSetLayout(),
            traceContext.GetSurfaceCacheSamplingSetLayout()
        };

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(GIInjectPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 3;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/SurfaceCacheGIInject.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        m_InjectCursor = 0;
        m_FrameIndex = 0;

        LOG_INFO("[SurfaceCacheGIInjectPass] Initialized GI injection pipeline.");
        return true;
    }

    void SurfaceCacheGIInjectPass::SetAtmosSkyView(VkImageView skyViewLUTView, VkSampler skyViewLUTSampler) {
        VkDescriptorImageInfo skyViewInfo{ skyViewLUTSampler, skyViewLUTView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = m_Set; write.dstBinding = 9; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo = &skyViewInfo;
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
    }

    void SurfaceCacheGIInjectPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }
        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Set = VK_NULL_HANDLE;
        m_InjectCursor = 0;
        m_FrameIndex = 0;
        m_Device = VK_NULL_HANDLE;
    }

    void SurfaceCacheGIInjectPass::RecordInject(VkCommandBuffer cmd, const SurfaceCacheTraceContext& traceContext,
        const SurfaceCachePass& surfaceCache, uint32_t traceMode, const maths::vec3& sunDirectionWorld) {
        const std::vector<geometry::SurfaceCacheCardEntry>& cards = surfaceCache.GetCards();
        if (cards.empty()) {
            return;
        }

        const uint32_t cardCount = static_cast<uint32_t>(cards.size());
        const uint32_t sliceCount = std::min(kCardsPerFrameBudget, cardCount);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        VkDescriptorSet sets[3] = { m_Set, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 3, sets, 0, nullptr);

        for (uint32_t i = 0; i < sliceCount; ++i) {
            const uint32_t cardIndex = (m_InjectCursor + i) % cardCount;

            // A non-resident card's atlasOffset/atlasSize is stale (or, worse, has already been
            // reassigned to a DIFFERENT currently-resident card by renderer::SurfaceCachePass's
            // dynamic allocator -- see that class' own "Dynamic atlas residency" comment) --
            // dispatching against it would read/write a live card's texels under the wrong
            // identity. Skipped iterations still count against this call's scanned slice (keeping
            // the round-robin cursor advance simple/bounded) but not against any GPU work.
            if (!surfaceCache.IsCardResident(cardIndex)) {
                continue;
            }
            const geometry::SurfaceCacheCardEntry& card = cards[cardIndex];

            GIInjectPushConstants pc{};
            pc.atlasOffsetX = card.atlasOffset[0];
            pc.atlasOffsetY = card.atlasOffset[1];
            pc.atlasSizeX = card.atlasSize[0];
            pc.atlasSizeY = card.atlasSize[1];
            pc.entityCount = traceContext.GetEntityCount();
            pc.sampleCount = kSampleCountPerTexel;
            pc.frameIndex = m_FrameIndex;
            pc.traceMode = traceMode;
            pc.sunDirX = sunDirectionWorld.x; pc.sunDirY = sunDirectionWorld.y; pc.sunDirZ = sunDirectionWorld.z;
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

            const uint32_t groupCountX = (card.atlasSize[0] + kWorkgroupSize - 1u) / kWorkgroupSize;
            const uint32_t groupCountY = (card.atlasSize[1] + kWorkgroupSize - 1u) / kWorkgroupSize;
            vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
        }

        m_InjectCursor = (m_InjectCursor + sliceCount) % cardCount;
        ++m_FrameIndex;
    }

}
