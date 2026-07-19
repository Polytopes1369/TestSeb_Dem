#include "renderer/passes/DecalProjectionPass.h"

#include <array>
#include <cstring>
#include <format>
#include <vector>

#include "core/Logger.h"
#include "geometry/DecalBVH.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of DecalViewParamsUBO in DecalProject.comp (std140).
        struct DecalViewParams {
            maths::mat4 invViewProj;            // offset 0
            float viewportWidth = 0.0f;         // offset 64
            float viewportHeight = 0.0f;        // offset 68
            uint32_t decalCount = 0u;           // offset 72
            uint32_t debugShowBounds = 0u;      // offset 76
            float cameraPosX = 0.0f;            // offset 80
            float cameraPosY = 0.0f;            // offset 84
            float cameraPosZ = 0.0f;            // offset 88
            float timeSeconds = 0.0f;           // offset 92
        };
        static_assert(sizeof(DecalViewParams) == 96,
            "DecalViewParams must match DecalProject.comp's DecalViewParamsUBO exactly (std140 layout)");

    } // namespace

    void DecalProjectionPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool,
        VkQueue queue, VkExtent2D renderExtent, VkImageView depthView, VkImageView normalView,
        VkImageView albedoView, VkImageView roughnessMetallicView, VkImageView colorView,
        const DecalSceneData& sceneDecals) {
        Shutdown();
        m_Device = device;
        m_RenderExtent = renderExtent;
        m_DecalCount = static_cast<uint32_t>(sceneDecals.instances.size());
        (void)commandPool;
        (void)queue;

        // Build the decal BVH from the scene's parallel world AABBs (index-aligned 1:1 with the GPU
        // instance array -- see renderer::DecalSceneData).
        geometry::DecalBVH bvh = geometry::BuildDecalBVH(sceneDecals.bounds.data(), m_DecalCount);
        const uint32_t nodeCount = static_cast<uint32_t>(bvh.nodes.size());
        const uint32_t indexCount = static_cast<uint32_t>(bvh.decalIndices.size());

        // --- Per-frame view-params UBO (GPU-only; filled via vkCmdUpdateBuffer each frame). ---
        m_ViewParamsBuffer.Create(allocator, sizeof(DecalViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // --- Static decal data SSBOs (host-visible mapped, uploaded once -- the decal set is
        // immutable after Init, same lightweight convention as renderer::MegaLightsPass's light BVH).
        // Every buffer is sized to at least one element so a zero-decal scene still yields valid,
        // bindable (never zero-sized) descriptors -- mirrors MegaLightsPass's own dummy-node fallback.
        const VkDeviceSize decalBytes =
            static_cast<VkDeviceSize>(std::max<uint32_t>(1u, m_DecalCount)) * sizeof(DecalInstanceGPU);
        const VkDeviceSize nodeBytes =
            static_cast<VkDeviceSize>(std::max<uint32_t>(1u, nodeCount)) * sizeof(geometry::DecalBVHNode);
        const VkDeviceSize indexBytes =
            static_cast<VkDeviceSize>(std::max<uint32_t>(1u, indexCount)) * sizeof(uint32_t);

        m_DecalBuffer.Create(allocator, decalBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        m_BVHNodesBuffer.Create(allocator, nodeBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        m_BVHIndicesBuffer.Create(allocator, indexBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        if (m_DecalCount > 0u) {
            std::memcpy(m_DecalBuffer.MappedData(), sceneDecals.instances.data(),
                static_cast<size_t>(m_DecalCount) * sizeof(DecalInstanceGPU));
        } else {
            DecalInstanceGPU dummy{};
            std::memcpy(m_DecalBuffer.MappedData(), &dummy, sizeof(dummy));
        }
        if (nodeCount > 0u) {
            std::memcpy(m_BVHNodesBuffer.MappedData(), bvh.nodes.data(),
                static_cast<size_t>(nodeCount) * sizeof(geometry::DecalBVHNode));
        } else {
            geometry::DecalBVHNode dummy{};
            std::memcpy(m_BVHNodesBuffer.MappedData(), &dummy, sizeof(dummy));
        }
        if (indexCount > 0u) {
            std::memcpy(m_BVHIndicesBuffer.MappedData(), bvh.decalIndices.data(),
                static_cast<size_t>(indexCount) * sizeof(uint32_t));
        } else {
            uint32_t dummy = 0u;
            std::memcpy(m_BVHIndicesBuffer.MappedData(), &dummy, sizeof(dummy));
        }

        // --- Descriptor set layout: 5 storage images + 1 UBO + 3 storage buffers. ---
        std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
        for (uint32_t i = 0; i < 5; ++i) {
            bindings[i] = { i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        }
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        setLayoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_SetLayout));

        std::array<VkDescriptorPoolSize, 3> poolSizes{ {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
        } };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, &m_Set));

        // All 5 GBuffer/color images are bound as storage images in VK_IMAGE_LAYOUT_GENERAL (their
        // permanent layout -- see renderer::ClusterResolvePass / ReflectionPass conventions).
        std::array<VkDescriptorImageInfo, 5> imageInfos{ {
            { VK_NULL_HANDLE, depthView,             VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, normalView,            VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, albedoView,            VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, roughnessMetallicView, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, colorView,             VK_IMAGE_LAYOUT_GENERAL },
        } };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorBufferInfo decalInfo{ m_DecalBuffer.Handle(), 0, m_DecalBuffer.Size() };
        VkDescriptorBufferInfo nodesInfo{ m_BVHNodesBuffer.Handle(), 0, m_BVHNodesBuffer.Size() };
        VkDescriptorBufferInfo indicesInfo{ m_BVHIndicesBuffer.Handle(), 0, m_BVHIndicesBuffer.Size() };

        std::array<VkWriteDescriptorSet, 9> writes{};
        for (uint32_t i = 0; i < 5; ++i) {
            writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, i, 0, 1,
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfos[i], nullptr, nullptr };
        }
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 5, 0, 1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 6, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &decalInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 7, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodesInfo, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 8, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indicesInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/DecalProject.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        LOG_INFO(std::format("[DecalProjectionPass] Initialized: {} decals, {} BVH nodes, {} indices.",
            m_DecalCount, nodeCount, indexCount));
    }

    void DecalProjectionPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }
        m_ViewParamsBuffer.Destroy();
        m_DecalBuffer.Destroy();
        m_BVHNodesBuffer.Destroy();
        m_BVHIndicesBuffer.Destroy();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_Set = VK_NULL_HANDLE;
        m_DecalCount = 0;
        m_RenderExtent = { 0, 0 };
        m_Device = VK_NULL_HANDLE;
    }

    void DecalProjectionPass::RecordDecals(VkCommandBuffer cmd, const maths::mat4& invViewProj,
        const maths::vec3& cameraPositionWorld, float timeSeconds
#ifndef NDEBUG
        , bool debugShowBounds
#endif
    ) {
        // Empty decal set -> nothing to composite (the shader would early-out on decalCount==0 anyway,
        // but skipping the whole dispatch + barrier avoids the cost entirely).
        if (m_DecalCount == 0u) {
            return;
        }

        DecalViewParams params{};
        params.invViewProj = invViewProj;
        params.viewportWidth = static_cast<float>(m_RenderExtent.width);
        params.viewportHeight = static_cast<float>(m_RenderExtent.height);
        params.decalCount = m_DecalCount;
#ifndef NDEBUG
        params.debugShowBounds = debugShowBounds ? 1u : 0u;
#else
        params.debugShowBounds = 0u;
#endif
        params.cameraPosX = cameraPositionWorld.x;
        params.cameraPosY = cameraPositionWorld.y;
        params.cameraPosZ = cameraPositionWorld.z;
        params.timeSeconds = timeSeconds;
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(DecalViewParams), &params);

        // vkCmdUpdateBuffer (transfer) -> compute uniform read.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_Set, 0, nullptr);

        const uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        const uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // Make the decaled GBuffer (normal/albedo/roughness-metallic) + direct color images visible to
        // every downstream deferred-lighting pass's compute reads this frame. The images never leave
        // VK_IMAGE_LAYOUT_GENERAL, so a single global memory barrier (no layout transition) suffices --
        // same convention the surrounding passes already use between compute stages.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

}
