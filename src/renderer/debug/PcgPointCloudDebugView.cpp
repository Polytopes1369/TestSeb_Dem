#ifndef NDEBUG

#include "renderer/debug/PcgPointCloudDebugView.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer::debug {

    void PcgPointCloudDebugView::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkFormat colorFormat, VkFormat depthFormat) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // --- Instance SSBO: capacity for kMaxPoints, empty (GPU_ONLY, no initial upload -- content
        // is only ever meaningful once SetPoints() has run at least once) until the first real
        // draw. Unlike DebugTextOverlay's font buffer (uploaded once here at Init), this pass has
        // no data to upload yet at Init time -- SetPoints() is the caller's own separate step. ---
        m_InstanceBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(GpuPointGizmoInstance)) * kMaxPoints,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // --- Descriptor set: binding 0 = PointInstancesSSBO (vertex-stage only -- the fragment
        // stage only ever reads the vertex stage's own interpolated outColor output). ---
        VkDescriptorSetLayoutBinding binding{ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &binding;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAlloc.descriptorPool = m_DescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_DescriptorSet));

        // Written once here, never rewritten -- m_InstanceBuffer's own VkBuffer identity never
        // changes across SetPoints() calls (only its CONTENTS are rewritten), so a single
        // Init()-time descriptor write is sufficient, exactly like DebugTextOverlay's own
        // GlyphInstancesSSBO binding.
        VkDescriptorBufferInfo bufferInfo{ m_InstanceBuffer.Handle(), 0, m_InstanceBuffer.Size() };
        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 0, 0, 1,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);

        // --- Pipeline layout: one push-constant range (mat4 viewProj, vertex-stage only). ---
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(maths::mat4);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        // --- Graphics pipeline: built manually (not via VulkanPipeline::CreateGraphicsPipeline,
        // which is hardcoded for the 2-attachment integer VisBuffer with no blending), matching
        // DebugTextOverlay's own manual-construction convention -- LINE_LIST topology, single color
        // attachment (`colorFormat`), depth-TESTED but NOT depth-WRITING against `depthFormat` (see
        // this class' own header comment for the full rationale). ---
        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PcgPointCloudDebug.vert.spv");
        VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/PcgPointCloudDebug.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragModule;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // Irrelevant for a LINE_LIST topology, kept FILL (no wireframe-fill-mode feature needed).
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE; // Lines have no facing.
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Opaque -- no blending, gizmo lines fully overwrite whatever pixel they land on.
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        // Reversed-Z (this codebase's own convention, see e.g. VegetationScatterPass's identical
        // depthCompareOp): depthTestEnable=TRUE so gizmos correctly hide behind real opaque
        // geometry, depthWriteEnable=FALSE so a debug gizmo never occludes anything drawn after it
        // (see this class' own header comment).
        VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkFormat colorAttachmentFormat = colorFormat;
        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 1;
        pipelineRendering.pColorAttachmentFormats = &colorAttachmentFormat;
        pipelineRendering.depthAttachmentFormat = depthFormat;

        VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.pNext = &pipelineRendering;

        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, fragModule, nullptr);

        LOG_INFO("[PcgPointCloudDebugView] Initialized PCG point cloud debug visualization (wireframe box gizmos).");

        // Silence unused-parameter warnings on the commandPool/queue arguments -- unlike
        // DebugTextOverlay's font buffer, this pass has no Init()-time upload of its own (see this
        // method's own comment above), so they are only kept in the signature for a consistent
        // Init() shape with every other Debug-only pass in this directory (all of which DO need
        // them) and to spare SetPoints()'s own callers from having to pass them separately again
        // if a future revision folds an Init()-time default upload back in.
        (void)commandPool;
        (void)queue;
    }

    void PcgPointCloudDebugView::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[PcgPointCloudDebugView] Shutting down PCG point cloud debug visualization...");
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_DescriptorSet.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;

        m_InstanceBuffer.Destroy();
        m_PointCount = 0;

        m_Device = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
    }

    void PcgPointCloudDebugView::SetPoints(VkCommandPool commandPool, VkQueue queue, const std::vector<pcg::PcgPoint>& points) {
        uint32_t count = static_cast<uint32_t>(points.size());
        if (count > kMaxPoints) {
            LOG_WARNING(std::format(
                "[PcgPointCloudDebugView] SetPoints() truncating {} point(s) to kMaxPoints={}.",
                points.size(), kMaxPoints));
            count = kMaxPoints;
        }
        m_PointCount = count;

        if (count == 0) {
            return; // Nothing to upload -- RecordDraw() will no-op against GetPointCount() == 0.
        }

        // --- CPU-side conversion: each point's OWN GetLocalToWorld() (PcgPointData.h) composes
        // its localToWorld matrix -- reused verbatim, never re-derived here. ---
        std::vector<GpuPointGizmoInstance> gpuInstances(count);
        for (uint32_t i = 0; i < count; ++i) {
            const pcg::PcgPoint& p = points[i];
            GpuPointGizmoInstance& gpu = gpuInstances[i];
            gpu.localToWorld = p.GetLocalToWorld();
            gpu.boundsMinX = p.boundsMin.x; gpu.boundsMinY = p.boundsMin.y; gpu.boundsMinZ = p.boundsMin.z;
            gpu.density = std::clamp(p.density, 0.0f, 1.0f);
            gpu.boundsMaxX = p.boundsMax.x; gpu.boundsMaxY = p.boundsMax.y; gpu.boundsMaxZ = p.boundsMax.z;
            gpu._pad0 = 0.0f;
        }

        // --- One blocking one-shot staged upload -- mirrors DebugTextOverlay::Init's own font
        // buffer upload convention exactly (CPU_ONLY mapped staging buffer, vkCmdCopyBuffer,
        // trailing memory barrier). ---
        VkDeviceSize uploadBytes = static_cast<VkDeviceSize>(sizeof(GpuPointGizmoInstance)) * count;

        GpuBuffer stagingBuffer;
        stagingBuffer.Create(m_Allocator, uploadBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
        std::memcpy(stagingBuffer.MappedData(), gpuInstances.data(), static_cast<size_t>(uploadBytes));

        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{ 0, 0, uploadBytes };
            vkCmdCopyBuffer(cmd, stagingBuffer.Handle(), m_InstanceBuffer.Handle(), 1, &copyRegion);

            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // stagingBuffer destructs (frees) here -- the one-shot submit above already blocked until
        // completion, so this is safe (matches DebugTextOverlay::Init's identical staging-buffer
        // lifetime, a local scoped-to-the-upload-block instance there vs. a local scoped-to-the-
        // whole-function instance here -- same "already synced, safe to free immediately" reasoning).

        LOG_INFO(std::format("[PcgPointCloudDebugView] SetPoints() uploaded {} PCG point(s) for wireframe box visualization.", count));
    }

    void PcgPointCloudDebugView::RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView,
        VkImage depthImage, VkImageView depthView, VkExtent2D extent, const maths::mat4& viewProj) {
        if (m_PointCount == 0) {
            return; // Nothing to draw -- skip both barrier dances entirely, mirrors DebugTextOverlay::RecordDraw's own early-out.
        }

        // `depthImage` is only referenced via `depthView` below (bound at
        // VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, no transition needed -- see this
        // method's own header comment); kept as a parameter purely for call-site symmetry with
        // every other [13c] forward pass' own RecordDraw(colorImage, colorView, depthImage,
        // depthView, ...) signature (e.g. VegetationScatterPass::RecordDraw).
        (void)depthImage;

        // Color target: GENERAL (ParticleSystemPass::RecordDraw's own exit state, this frame) ->
        // COLOR_ATTACHMENT_OPTIMAL for this draw -- identical transition every other [13c] forward
        // pass performs on entry.
        VkImageMemoryBarrier2 toAttachment{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toAttachment.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        toAttachment.srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        toAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.image = colorImage;
        toAttachment.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo toAttachmentDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDependency.imageMemoryBarrierCount = 1;
        toAttachmentDependency.pImageMemoryBarriers = &toAttachment;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDependency);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Preserve the already-composited scene underneath.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Same "already read-only by this point in the frame" assumption as
        // ParticleSystemPass::RecordDraw's own depth attachment -- see that method's own comment
        // and this class' own header comment.
        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, extent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

        maths::mat4 viewProjCopy = viewProj;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(maths::mat4), &viewProjCopy);

        VkViewport viewport{};
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // 24 vertices (12 edges) per box instance, m_PointCount instances -- no vertex/index
        // buffer, everything derived from gl_VertexIndex/gl_InstanceIndex (see
        // PcgPointCloudDebug.vert's own header comment).
        vkCmdDraw(cmd, 24, m_PointCount, 0, 0);

        vkCmdEndRendering(cmd);

        // Restore GENERAL -- this pass is recorded LAST in the [13c] forward block (right after
        // ParticleSystemPass::RecordDraw), so nothing else this frame reads `colorImage` through a
        // COLOR_ATTACHMENT_OPTIMAL-incompatible path before the eventual GENERAL-expecting
        // consumer (the post-process/blit stage) runs -- identical restore every other [13c]
        // forward pass performs on its own exit.
        VkImageMemoryBarrier2 toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toGeneral.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = colorImage;
        toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo toGeneralDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toGeneralDependency.imageMemoryBarrierCount = 1;
        toGeneralDependency.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cmd, &toGeneralDependency);
    }

}

#endif // NDEBUG
