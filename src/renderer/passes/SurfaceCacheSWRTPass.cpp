#include "renderer/passes/SurfaceCacheSWRTPass.h"

#include "core/Logger.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"

namespace renderer {

    namespace {

        // Byte-for-byte layout match for SWRTPushConstants in SurfaceCacheTraceSWRT.comp.
        struct SWRTPushConstants {
            uint32_t rayCount = 0;
            uint32_t entityCount = 0;
        };
        static_assert(sizeof(SWRTPushConstants) == 8,
            "SWRTPushConstants must match SurfaceCacheTraceSWRT.comp's push_constant block exactly");

    } // namespace

    bool SurfaceCacheSWRTPass::Init(VkDevice device, VmaAllocator /*allocator*/, const SurfaceCacheTraceContext& traceContext) {
        Shutdown();
        m_Device = device;

        // =====================================================================================
        // STEP 1 -- set 0's layout: the ray-request (readonly) / ray-result (writeonly) SSBO pair.
        // =====================================================================================
        VkDescriptorSetLayoutBinding rayBindings[2]{};
        rayBindings[0].binding = 0;
        rayBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        rayBindings[0].descriptorCount = 1;
        rayBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        rayBindings[1].binding = 1;
        rayBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        rayBindings[1].descriptorCount = 1;
        rayBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo raySetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        raySetLayoutInfo.bindingCount = 2;
        raySetLayoutInfo.pBindings = rayBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &raySetLayoutInfo, nullptr, &m_RaySetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_RaySetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_RaySet));

        // =====================================================================================
        // STEP 2 -- pipeline layout: set 0 (above) + set 1 (mesh SDF trace scene) + set 2
        // (surface cache sampling), both from the shared, already-Init'd trace context.
        // =====================================================================================
        VkDescriptorSetLayout setLayouts[3] = {
            m_RaySetLayout,
            traceContext.GetMeshSdfTraceSetLayout(),
            traceContext.GetSurfaceCacheSamplingSetLayout()
        };

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SWRTPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 3;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        // =====================================================================================
        // STEP 3 -- compute pipeline.
        // =====================================================================================
        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/SurfaceCacheTraceSWRT.comp.spv");

        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[SurfaceCacheSWRTPass] Initialized SWRT trace pipeline.");
        return true;
    }

    void SurfaceCacheSWRTPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_RaySetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RaySetLayout, nullptr);
        }
        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_RaySetLayout = VK_NULL_HANDLE;
        m_RaySet = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void SurfaceCacheSWRTPass::SetRayBuffers(VkBuffer rayBuffer, VkDeviceSize rayBufferSize, VkBuffer resultBuffer, VkDeviceSize resultBufferSize) {
        VkDescriptorBufferInfo rayInfo{ rayBuffer, 0, rayBufferSize };
        VkDescriptorBufferInfo resultInfo{ resultBuffer, 0, resultBufferSize };

        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m_RaySet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &rayInfo;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = m_RaySet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &resultInfo;

        vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);
    }

    void SurfaceCacheSWRTPass::RecordTrace(VkCommandBuffer cmd, uint32_t rayCount, const SurfaceCacheTraceContext& traceContext) {
        if (rayCount == 0) {
            return;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);

        VkDescriptorSet sets[3] = { m_RaySet, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 3, sets, 0, nullptr);

        SWRTPushConstants pc{};
        pc.rayCount = rayCount;
        pc.entityCount = traceContext.GetEntityCount();
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        const uint32_t groupCount = (rayCount + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCount, 1, 1);
    }

}
