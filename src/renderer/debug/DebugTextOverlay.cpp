#ifndef NDEBUG

#include "renderer/debug/DebugTextOverlay.h"

#include <cstring>
#include <format>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"
#include "renderer/debug/BitmapFont8x8.h"

namespace renderer::debug {

    namespace {

        constexpr uint32_t kFontCharCount = 128;
        constexpr uint32_t kFontRowsPerChar = 8;

    } // namespace

    void DebugTextOverlay::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkFormat outputColorFormat) {
        Shutdown();

        m_Device = device;

        // --- Font bitmap SSBO: flatten BitmapFont8x8.h's sparse table into 128*8 uint32 rows and
        // upload once via a blocking one-time submit (Init()-time only, matching this codebase's
        // existing one-shot-setup convention). ---
        Font8x8Table font = BuildFont8x8();
        std::vector<uint32_t> fontRows(kFontCharCount * kFontRowsPerChar, 0u);
        for (uint32_t c = 0; c < kFontCharCount; ++c) {
            for (uint32_t r = 0; r < kFontRowsPerChar; ++r) {
                fontRows[c * kFontRowsPerChar + r] = font[c][r];
            }
        }

        VkDeviceSize fontBytes = static_cast<VkDeviceSize>(sizeof(uint32_t)) * fontRows.size();
        m_FontBuffer.Create(allocator, fontBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        m_GlyphInstanceBuffer.Create(allocator, static_cast<VkDeviceSize>(sizeof(GlyphInstance)) * kMaxGlyphs,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        {
            GpuBuffer stagingBuffer;
            stagingBuffer.Create(allocator, fontBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_CPU_ONLY, /*mapped=*/true);
            std::memcpy(stagingBuffer.MappedData(), fontRows.data(), static_cast<size_t>(fontBytes));

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkBufferCopy copyRegion{ 0, 0, fontBytes };
                vkCmdCopyBuffer(cmd, stagingBuffer.Handle(), m_FontBuffer.Handle(), 1, &copyRegion);

                VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

                VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                depInfo.memoryBarrierCount = 1;
                depInfo.pMemoryBarriers = &barrier;
                vkCmdPipelineBarrier2(cmd, &depInfo);
            });
        }

        // --- Descriptor set: binding 0 = GlyphInstancesSSBO (vertex-stage), binding 1 =
        // FontBitmapSSBO (fragment-stage). ---
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
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

        VkDescriptorBufferInfo glyphInfo{ m_GlyphInstanceBuffer.Handle(), 0, m_GlyphInstanceBuffer.Size() };
        VkDescriptorBufferInfo fontInfo{ m_FontBuffer.Handle(), 0, m_FontBuffer.Size() };

        VkWriteDescriptorSet writes[2]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &glyphInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fontInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 2, writes, 0, nullptr);

        // --- Pipeline layout: one push-constant range (vec2 viewportSize, vertex-stage only). ---
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(maths::vec2);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        // --- Graphics pipeline: built manually (not via VulkanPipeline::CreateGraphicsPipeline,
        // which is hardcoded for the 2-attachment integer VisBuffer with no blending) -- single
        // RGBA8 color attachment, standard alpha blending, no depth attachment at all. ---
        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/DebugText.vert.spv");
        VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/DebugText.frag.spv");

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
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE; // Screen-space quads, winding is not meaningful here.
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Standard "over" alpha blending -- text drawn on top of whatever the resolve pass already
        // shaded, matching the fixed alpha the fragment shader outputs (0.9).
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 1;
        pipelineRendering.pColorAttachmentFormats = &outputColorFormat;
        // No depth attachment at all -- depthAttachmentFormat stays VK_FORMAT_UNDEFINED.

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

        LOG_INFO("[DebugTextOverlay] Initialized debug text overlay.");
    }

    void DebugTextOverlay::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            LOG_INFO("[DebugTextOverlay] Shutting down debug text overlay...");
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

        m_FontBuffer.Destroy();
        m_GlyphInstanceBuffer.Destroy();
        m_PendingGlyphs.clear();

        m_Device = VK_NULL_HANDLE;
    }

    void DebugTextOverlay::AppendLine(const std::string& text, float x, float y) {
        float cursorX = x;
        for (char c : text) {
            if (m_PendingGlyphs.size() >= kMaxGlyphs) {
                return;
            }
            GlyphInstance inst{};
            inst.screenPosPixels = maths::vec2{ cursorX, y };
            inst.charCode = static_cast<uint32_t>(static_cast<unsigned char>(c));
            m_PendingGlyphs.push_back(inst);
            cursorX += kGlyphAdvanceX;
        }
    }

    void DebugTextOverlay::BuildFrameText(float gpuMemUsedMB, uint32_t pendingPageLoads, float bytesPerSecond,
        uint32_t hwTriangleCount, uint32_t swTriangleCount, float fps, float viewportWidthPixels,
        bool radiosityEnabled, bool ssrtEnabled, uint32_t traceMode, bool worldProbesEnabled) {
        m_PendingGlyphs.clear();

        constexpr float kMarginX = 8.0f;
        constexpr float kMarginY = 8.0f;
        constexpr float kLineHeight = 10.0f;

        float y = kMarginY;
        AppendLine(std::format("GPU MEM: {:.1f} MB", gpuMemUsedMB), kMarginX, y); y += kLineHeight;
        AppendLine(std::format("PAGES PENDING: {}", pendingPageLoads), kMarginX, y); y += kLineHeight;
        AppendLine(std::format("READ: {:.1f} KB/S", bytesPerSecond / 1024.0f), kMarginX, y); y += kLineHeight;
        AppendLine(std::format("HW TRIS: {}", hwTriangleCount), kMarginX, y); y += kLineHeight;
        AppendLine(std::format("SW TRIS: {}", swTriangleCount), kMarginX, y); y += kLineHeight;
        AppendLine(std::format("GI: RADIOSITY={} SSRT={} TRACE={}",
            radiosityEnabled ? "ON" : "OFF", ssrtEnabled ? "ON" : "OFF", traceMode == 0u ? "SWRT" : "HWRT"),
            kMarginX, y); y += kLineHeight;
        // Unlike the RADIOSITY/SSRT/TRACE line above (all real, consumed GI terms), WORLDPROBES
        // reflects a system that is computed but has no live consumer yet -- see
        // ClusterRenderPipeline::m_DebugWorldProbesEnabled's own comment. Shown on its own line so
        // that fact stays visible instead of implying parity with the other three.
        AppendLine(std::format("WORLDPROBES={} (not yet sampled)", worldProbesEnabled ? "ON" : "OFF"),
            kMarginX, y); y += kLineHeight;

        // Top-right FPS counter -- right-aligned against the render extent's own width using
        // kGlyphAdvanceX's fixed per-glyph pixel advance to measure the line before it's drawn.
        std::string fpsText = std::format("FPS: {:.1f}", fps);
        float fpsTextWidth = static_cast<float>(fpsText.size()) * kGlyphAdvanceX;
        AppendLine(fpsText, viewportWidthPixels - fpsTextWidth - kMarginX, kMarginY);
    }

    void DebugTextOverlay::RecordDraw(VkCommandBuffer cmd, VkImage outputColorImage, VkImageView outputColorView, VkExtent2D extent) {
        if (m_PendingGlyphs.empty()) {
            return; // Nothing to draw -- skip the layout transitions entirely.
        }

        uint32_t glyphCount = static_cast<uint32_t>(std::min<size_t>(m_PendingGlyphs.size(), kMaxGlyphs));
        VkDeviceSize updateSize = static_cast<VkDeviceSize>(sizeof(GlyphInstance)) * glyphCount;
        vkCmdUpdateBuffer(cmd, m_GlyphInstanceBuffer.Handle(), 0, updateSize, m_PendingGlyphs.data());

        VkMemoryBarrier2 uploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
        uploadBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkImageMemoryBarrier2 toAttachment{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toAttachment.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toAttachment.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        toAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment.image = outputColorImage;
        toAttachment.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &uploadBarrier;
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &toAttachment;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = outputColorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Preserve the resolve pass's own shading underneath.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, extent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

        maths::vec2 viewportSize{ static_cast<float>(extent.width), static_cast<float>(extent.height) };
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(maths::vec2), &viewportSize);

        VkViewport viewport{};
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDraw(cmd, 6, glyphCount, 0, 0);

        vkCmdEndRendering(cmd);

        VkImageMemoryBarrier2 toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toGeneral.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = outputColorImage;
        toGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo backDepInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        backDepInfo.imageMemoryBarrierCount = 1;
        backDepInfo.pImageMemoryBarriers = &toGeneral;
        vkCmdPipelineBarrier2(cmd, &backDepInfo);
    }

}

#endif // NDEBUG
