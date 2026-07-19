#include "renderer/passes/ProceduralTreePass.h"

#include <cstring>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of geom_tree_bark.comp's TreeBarkParams push_constant block --
        // every member is a plain scalar (uint/float), so GLSL packs them tightly at 4-byte
        // stride with no vec4-boundary padding, exactly like this codebase's other all-scalar
        // push-constant structs (e.g. GlobalSDFPass.cpp's GlobalSDFCompositePC).
        struct TreeBarkPC {
            uint32_t seed = 0;
            uint32_t depth = 0;
            uint32_t branchFactor = 0;
            float trunkHeight = 0;
            float trunkRadius = 0;
            float lengthTaper = 0;
            float radiusTaper = 0;
            float branchAngle = 0;
            float pitchDamping = 0;
            uint32_t sides = 0;
            uint32_t nodeCount = 0;
            uint32_t meshID = 0;
            float materialID = 0;
            uint32_t vertexOffset = 0;
            uint32_t indexOffset = 0;
            float worldOffsetX = 0;
            float worldOffsetY = 0;
            float worldOffsetZ = 0;
        };
        static_assert(sizeof(TreeBarkPC) == 72,
            "TreeBarkPC must match geom_tree_bark.comp's TreeBarkParams push_constant block exactly");

        // Byte-for-byte mirror of geom_tree_leaves.comp's TreeLeavesParams push_constant block.
        struct TreeLeavesPC {
            uint32_t seed = 0;
            uint32_t depth = 0;
            uint32_t branchFactor = 0;
            float trunkHeight = 0;
            float trunkRadius = 0;
            float lengthTaper = 0;
            float radiusTaper = 0;
            float branchAngle = 0;
            float pitchDamping = 0;
            uint32_t leafNodeIDBase = 0;
            uint32_t leafNodeCount = 0;
            float leafSize = 0;
            uint32_t meshID = 0;
            float materialID = 0;
            uint32_t vertexOffset = 0;
            uint32_t indexOffset = 0;
            float worldOffsetX = 0;
            float worldOffsetY = 0;
            float worldOffsetZ = 0;
        };
        static_assert(sizeof(TreeLeavesPC) == 76,
            "TreeLeavesPC must match geom_tree_leaves.comp's TreeLeavesParams push_constant block exactly");

        constexpr uint32_t kLocalSizeX = 64u; // Matches both shaders' local_size_x = 64.

        uint32_t DispatchGroupCount(uint32_t invocationCount) {
            return (invocationCount + kLocalSizeX - 1u) / kLocalSizeX;
        }

        // (branchFactor^exponent), computed in uint64_t to avoid any intermediate overflow before
        // narrowing back to uint32_t -- depth/branchFactor stay small in practice (this pass's own
        // default is depth=4, branchFactor=3, i.e. 3^4=81) but the formula itself is general.
        uint64_t IntPow(uint64_t base, uint32_t exponent) {
            uint64_t result = 1;
            for (uint32_t i = 0; i < exponent; ++i) {
                result *= base;
            }
            return result;
        }

    } // namespace

    uint32_t ProceduralTreePass::ComputeNodeCount(uint32_t depth, uint32_t branchFactor) {
        // Closed-form node count of a complete branchFactor-ary tree spanning levels [0, depth]:
        // (branchFactor^(depth+1) - 1) / (branchFactor - 1) -- see tree_lsystem.glsl's own header
        // comment for the identical formula restated GLSL-side.
        if (branchFactor < 2u) {
            return 1u; // Degenerate (a single-child "tree" is just a chain) -- treated as trunk-only.
        }
        uint64_t numerator = IntPow(branchFactor, depth + 1u) - 1ull;
        uint64_t denominator = static_cast<uint64_t>(branchFactor) - 1ull;
        return static_cast<uint32_t>(numerator / denominator);
    }

    uint32_t ProceduralTreePass::ComputeLeafNodeIDBase(uint32_t depth, uint32_t branchFactor) {
        if (branchFactor < 2u || depth == 0u) {
            return 0u;
        }
        uint64_t numerator = IntPow(branchFactor, depth) - 1ull;
        uint64_t denominator = static_cast<uint64_t>(branchFactor) - 1ull;
        return static_cast<uint32_t>(numerator / denominator);
    }

    uint32_t ProceduralTreePass::ComputeLeafNodeCount(uint32_t depth, uint32_t branchFactor) {
        return static_cast<uint32_t>(IntPow(branchFactor, depth));
    }

    ProceduralTreePass::GeometryFootprint ProceduralTreePass::ComputeFootprint(const TreeParams& params) {
        GeometryFootprint fp{};
        uint32_t nodeCount = ComputeNodeCount(params.depth, params.branchFactor);
        uint32_t sideColumns = params.sides + 1u;
        fp.barkVertexCount = nodeCount * sideColumns * 2u;
        fp.barkIndexCount = nodeCount * params.sides * 6u;

        uint32_t leafNodeCount = ComputeLeafNodeCount(params.depth, params.branchFactor);
        fp.leafVertexCount = leafNodeCount * 8u;
        fp.leafIndexCount = leafNodeCount * 24u;
        return fp;
    }

    void ProceduralTreePass::Init(VkDevice device, VkCommandPool commandPool, VkQueue queue,
        VkBuffer vertexBuffer, VkBuffer indexBuffer) {
        Shutdown();

        m_Device = device;
        m_CommandPool = commandPool;
        m_Queue = queue;

        // --- Descriptor set layout: binding 0 = shared vertex SSBO, binding 1 = shared index SSBO
        // (both borrowed -- see this class's own header comment for why this pass does not own
        // them), matching geom_tree_bark.comp/geom_tree_leaves.comp's own `binding = 0`/`binding =
        // 1` writeonly buffer declarations. ---
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

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

        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_DescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, &m_DescriptorSet));

        VkDescriptorBufferInfo vertexInfo{ vertexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo indexInfo{ indexBuffer, 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m_DescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &vertexInfo;
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = m_DescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &indexInfo;
        vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

        // --- Bark pipeline ---
        {
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.size = sizeof(TreeBarkPC);
            VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &m_SetLayout;
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &pcRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_BarkPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/geom_tree_bark.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_BarkPipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_BarkPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // --- Leaves pipeline ---
        {
            VkPushConstantRange pcRange{};
            pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcRange.size = sizeof(TreeLeavesPC);
            VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &m_SetLayout;
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &pcRange;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_LeavesPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/geom_tree_leaves.comp.spv");
            VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            pipelineInfo.layout = m_LeavesPipelineLayout;
            pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pipelineInfo.stage.module = shaderModule;
            pipelineInfo.stage.pName = "main";
            VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_LeavesPipeline));
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        LOG_INFO("[ProceduralTreePass] Initialized bark+leaves compute pipelines.");
    }

    void ProceduralTreePass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_BarkPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_BarkPipeline, nullptr);
            if (m_BarkPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_BarkPipelineLayout, nullptr);
            if (m_LeavesPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_LeavesPipeline, nullptr);
            if (m_LeavesPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_LeavesPipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr); // Frees m_DescriptorSet too.
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }

        m_BarkPipeline = VK_NULL_HANDLE;
        m_BarkPipelineLayout = VK_NULL_HANDLE;
        m_LeavesPipeline = VK_NULL_HANDLE;
        m_LeavesPipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
        m_CommandPool = VK_NULL_HANDLE;
        m_Queue = VK_NULL_HANDLE;
    }

    void ProceduralTreePass::RecordGenerate(const TreeParams& params, uint32_t& vertexOffset, uint32_t& indexOffset) {
        if (m_Device == VK_NULL_HANDLE) {
            throw std::runtime_error("[ProceduralTreePass] RecordGenerate() called before Init()!");
        }

        uint32_t nodeCount = ComputeNodeCount(params.depth, params.branchFactor);
        uint32_t leafNodeIDBase = ComputeLeafNodeIDBase(params.depth, params.branchFactor);
        uint32_t leafNodeCount = ComputeLeafNodeCount(params.depth, params.branchFactor);
        GeometryFootprint fp = ComputeFootprint(params);

        TreeBarkPC barkPC{};
        barkPC.seed = params.seed;
        barkPC.depth = params.depth;
        barkPC.branchFactor = params.branchFactor;
        barkPC.trunkHeight = params.trunkHeight;
        barkPC.trunkRadius = params.trunkRadius;
        barkPC.lengthTaper = params.lengthTaper;
        barkPC.radiusTaper = params.radiusTaper;
        barkPC.branchAngle = params.branchAngleRadians;
        barkPC.pitchDamping = params.pitchDamping;
        barkPC.sides = params.sides;
        barkPC.nodeCount = nodeCount;
        barkPC.meshID = params.barkMeshID;
        barkPC.materialID = params.barkMaterialID;
        barkPC.vertexOffset = vertexOffset;
        barkPC.indexOffset = indexOffset;
        barkPC.worldOffsetX = params.worldOffsetX;
        barkPC.worldOffsetY = params.worldOffsetY;
        barkPC.worldOffsetZ = params.worldOffsetZ;

        uint32_t barkVertexOffset = vertexOffset;
        uint32_t barkIndexOffset = indexOffset;
        vertexOffset += fp.barkVertexCount;
        indexOffset += fp.barkIndexCount;

        TreeLeavesPC leavesPC{};
        leavesPC.seed = params.seed;
        leavesPC.depth = params.depth;
        leavesPC.branchFactor = params.branchFactor;
        leavesPC.trunkHeight = params.trunkHeight;
        leavesPC.trunkRadius = params.trunkRadius;
        leavesPC.lengthTaper = params.lengthTaper;
        leavesPC.radiusTaper = params.radiusTaper;
        leavesPC.branchAngle = params.branchAngleRadians;
        leavesPC.pitchDamping = params.pitchDamping;
        leavesPC.leafNodeIDBase = leafNodeIDBase;
        leavesPC.leafNodeCount = leafNodeCount;
        leavesPC.leafSize = params.leafSize;
        leavesPC.meshID = params.leafMeshID;
        leavesPC.materialID = params.leafMaterialID;
        leavesPC.vertexOffset = vertexOffset;
        leavesPC.indexOffset = indexOffset;
        leavesPC.worldOffsetX = params.worldOffsetX;
        leavesPC.worldOffsetY = params.worldOffsetY;
        leavesPC.worldOffsetZ = params.worldOffsetZ;

        vertexOffset += fp.leafVertexCount;
        indexOffset += fp.leafIndexCount;

        uint32_t barkGroups = DispatchGroupCount(fp.barkVertexCount);
        uint32_t leavesGroups = DispatchGroupCount(leafNodeCount);

        VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_Queue, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BarkPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_BarkPipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_BarkPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(barkPC), &barkPC);
            vkCmdDispatch(cmd, barkGroups, 1, 1);

            // The leaves dispatch reads no data the bark dispatch wrote (they write disjoint vertex/
            // index ranges and geom_tree_leaves.comp re-derives its own node transforms from scratch
            // via tree_lsystem.glsl rather than reading anything geom_tree_bark.comp produced), so
            // strictly no barrier is REQUIRED between them for correctness of the leaves dispatch
            // itself -- this barrier exists purely so the final "compute writes visible to the vertex
            // shader" barrier below has a single well-defined completion point covering both
            // dispatches, matching VulkanContext::DispatchGeometryCompute's own barrier placement
            // convention (one barrier immediately after each compute dispatch that touches these
            // shared SSBOs).
            VkMemoryBarrier2 midBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            midBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            midBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            midBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            midBarrier.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            VkDependencyInfo midDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            midDep.memoryBarrierCount = 1;
            midDep.pMemoryBarriers = &midBarrier;
            vkCmdPipelineBarrier2(cmd, &midDep);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LeavesPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_LeavesPipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_LeavesPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(leavesPC), &leavesPC);
            vkCmdDispatch(cmd, leavesGroups, 1, 1);

            // Compute writes to the shared Vertex/Index SSBOs must be visible to the vertex shader
            // stage that reads them at draw time -- same barrier VulkanContext::
            // DispatchGeometryCompute records after every other primitive's own dispatch.
            VkMemoryBarrier2 finalBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            finalBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            finalBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo finalDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            finalDep.memoryBarrierCount = 1;
            finalDep.pMemoryBarriers = &finalBarrier;
            vkCmdPipelineBarrier2(cmd, &finalDep);
            });

        LOG_INFO(std::format(
            "[ProceduralTreePass] Generated tree (seed={}, nodes={}, leaves={}) -- bark: {}v/{}i at "
            "vOff={}/iOff={}, leaves: {}v/{}i at vOff={}/iOff={}.",
            params.seed, nodeCount, leafNodeCount,
            fp.barkVertexCount, fp.barkIndexCount, barkVertexOffset, barkIndexOffset,
            fp.leafVertexCount, fp.leafIndexCount, leavesPC.vertexOffset, leavesPC.indexOffset));
    }

}
