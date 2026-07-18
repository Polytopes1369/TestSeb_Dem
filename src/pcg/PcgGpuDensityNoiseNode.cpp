// PCG framework roadmap, Phase 5.3 ("GPU-Resident Node Execution"). See PcgGpuDensityNoiseNode.h
// for the full design rationale.

#include "pcg/PcgGpuDensityNoiseNode.h"

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"

namespace pcg {

    void PcgGpuDensityNoiseNode::Init(VkDevice device) {
        Shutdown();
        m_Device = device;

        // =========================================================================================
        // Descriptor set layout (2 storage-buffer bindings) + pool + set.
        // =========================================================================================
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 2;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_Set));

        // =========================================================================================
        // Pipeline layout (set 0 above + one push-constant range) and the compute pipeline itself.
        // =========================================================================================
        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PcgGpuDensityNoisePushConstants) };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PcgDensityNoise.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO("[PcgGpuDensityNoiseNode] Initialized (typeId='pcg.gpu.density_noise').");
    }

    void PcgGpuDensityNoiseNode::Shutdown() {
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
        m_Device = VK_NULL_HANDLE;
    }

    void PcgGpuDensityNoiseNode::RegisterGpu(PcgNodeTypeRegistry& registry) {
        registry.RegisterGpu(kTypeId, [this](VkCommandBuffer cmd, const PcgGpuPointBuffer& input,
            const PcgGpuPointBuffer& output, const PcgAttributeSet& params) {
                return Execute(cmd, input, output, params);
            });
    }

    PcgGpuNodeExecuteResult PcgGpuDensityNoiseNode::Execute(VkCommandBuffer cmd, const PcgGpuPointBuffer& input,
        const PcgGpuPointBuffer& output, const PcgAttributeSet& params) {
        if (m_Pipeline == VK_NULL_HANDLE) {
            return PcgGpuNodeExecuteResult::Error("PcgGpuDensityNoiseNode::Execute called before a successful Init()");
        }
        if (input.buffer == VK_NULL_HANDLE || output.buffer == VK_NULL_HANDLE) {
            return PcgGpuNodeExecuteResult::Error("PcgGpuDensityNoiseNode::Execute: input/output buffer must not be VK_NULL_HANDLE");
        }
        if (input.pointCount != output.pointCount) {
            return PcgGpuNodeExecuteResult::Error(
                "PcgGpuDensityNoiseNode::Execute: input.pointCount must equal output.pointCount "
                "(this node performs a strict 1:1 per-point transform, never a resize)");
        }
        if (input.pointCount == 0) {
            // Nothing to do -- not an error, mirrors this codebase's established "skip a zero-count
            // dispatch entirely" convention (e.g. ParticleSystemPass::RecordSimulate's own spawn
            // passes, "Skipped entirely when spawnCount == 0").
            return PcgGpuNodeExecuteResult::Ok();
        }

        // Rewrite this node's own descriptor bindings to this call's input/output buffers -- always
        // binds the FULL underlying VkBuffer at descriptor offset 0 (VK_WHOLE_SIZE range); the
        // caller-supplied element offset is instead threaded through as a push constant and applied
        // inside the shader -- see PcgGpuPointBuffer's own comment (PcgGraphEvaluator.h) for why
        // this sidesteps VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment concerns entirely.
        VkDescriptorBufferInfo inputInfo{ input.buffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo outputInfo{ output.buffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &inputInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &outputInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

        PcgGpuDensityNoisePushConstants pc{};
        pc.pointCount = input.pointCount;
        pc.inputOffsetElements = input.offsetElements;
        pc.outputOffsetElements = output.offsetElements;
        pc.seedOverride = static_cast<uint32_t>(params.GetOr<int32_t>("seedOverride", 0));
        pc.noiseFrequency = params.GetOr<float>("noiseFrequency", 1.0f);
        pc.noiseAmplitude = params.GetOr<float>("noiseAmplitude", 0.5f);
        pc.densityFloor = params.GetOr<float>("densityFloor", 0.0f);
        pc.densityCeil = params.GetOr<float>("densityCeil", 1.0f);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        const uint32_t groupCountX = (input.pointCount + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, 1, 1);

        return PcgGpuNodeExecuteResult::Ok();
    }

}
