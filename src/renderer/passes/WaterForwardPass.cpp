#include "renderer/passes/WaterForwardPass.h"

#include <cstring>
#include <format>
#include <span>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h" // geometry::FallbackVertex
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of WaterForwardConstants in WaterForward.vert/.frag -- flat scalar
        // fields throughout, same push-constant convention as every other pass in this codebase.
        struct WaterForwardConstants {
            maths::mat4 viewProj;
            float cameraPositionWorldX = 0, cameraPositionWorldY = 0, cameraPositionWorldZ = 0;
            float _pad0 = 0;
            uint32_t entityID = 0;
            uint32_t traceMode = 0;
            uint32_t frameIndex = 0;
            uint32_t entityCount = 0;
            float viewportWidth = 0, viewportHeight = 0;
            float timeSeconds = 0;
            float _pad1 = 0;
        };
        static_assert(sizeof(WaterForwardConstants) == 112,
            "WaterForwardConstants must match WaterForward.vert/.frag's push_constant block exactly");
        static_assert(sizeof(WaterForwardConstants) <= 128,
            "Must stay under the Vulkan-guaranteed minimum maxPushConstantsSize (128 bytes).");

    } // namespace

    bool WaterForwardPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkFormat colorFormat, VkFormat depthFormat,
        VkBuffer entityTransformBuffer,
        const MaterialParameters& waterMaterial,
        VkBuffer fallbackVertexBuffer, VkBuffer fallbackIndexBuffer,
        const SurfaceCachePass::EntityDrawRange& waterEntityDrawRange, uint32_t waterEntityID,
        VkAccelerationStructureKHR tlasHandle, VkBuffer drawRangeBuffer,
        const SurfaceCacheTraceContext& traceContext, VkExtent2D renderExtent) {
        // Self-reinit (see ShadowMapPass's own migration comment for the identical pattern):
        // Shutdown() clears m_Device/m_Allocator, which RenderPass<WaterForwardPass>::Init() already
        // set to this call's values just before invoking this function -- restore them.
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;
        m_WaterEntityDrawRange = waterEntityDrawRange;
        m_WaterEntityID = waterEntityID;
        m_FallbackVertexBuffer = fallbackVertexBuffer;
        m_FallbackIndexBuffer = fallbackIndexBuffer;

        // =====================================================================================
        // STEP 0 -- background snapshot image (see this class' own header comment for the full
        // refraction-mechanism rationale). TRANSFER_DST_BIT: this pass' own RecordDraw() blits
        // into it every frame. SAMPLED_BIT: WaterForward.frag reads it back as a texture.
        // =====================================================================================
        VkImageCreateInfo snapshotImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        snapshotImageInfo.imageType = VK_IMAGE_TYPE_2D;
        snapshotImageInfo.format = colorFormat;
        snapshotImageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
        snapshotImageInfo.mipLevels = 1;
        snapshotImageInfo.arrayLayers = 1;
        snapshotImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        snapshotImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        snapshotImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        snapshotImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        snapshotImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo snapshotAllocInfo{};
        snapshotAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        if (vmaCreateImage(allocator, &snapshotImageInfo, &snapshotAllocInfo,
                &m_BackgroundSnapshotImage, &m_BackgroundSnapshotAllocation, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate WaterForwardPass background snapshot image!");
        }

        VkImageViewCreateInfo snapshotViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        snapshotViewInfo.image = m_BackgroundSnapshotImage;
        snapshotViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        snapshotViewInfo.format = colorFormat;
        snapshotViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(m_Device, &snapshotViewInfo, nullptr, &m_BackgroundSnapshotView) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create WaterForwardPass background snapshot image view!");
        }
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_BackgroundSnapshotView, nullptr);
            vmaDestroyImage(m_Allocator, m_BackgroundSnapshotImage, m_BackgroundSnapshotAllocation);
        });

        // One-shot UNDEFINED -> SHADER_READ_ONLY_OPTIMAL transition -- so RecordDraw() never needs
        // a first-frame special case (every frame sees the exact same precondition on entry: this
        // pass' own restore-barrier-free steady state, see RecordDraw's own comment). Content is
        // undefined but the layout is valid; this pass' own first blit overwrites it wholesale
        // before anything ever samples it.
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 initBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            initBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            initBarrier.srcAccessMask = 0;
            initBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            initBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            initBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            initBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            initBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            initBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            initBarrier.image = m_BackgroundSnapshotImage;
            initBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &initBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        // LINEAR: the refraction UV-offset read benefits from smooth interpolation (distinct from
        // the NEAREST filter RecordDraw()'s own blit uses to populate this image -- a straight
        // 1:1-resolution copy has no resampling to do, so NEAREST there is simply cheaper).
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        if (vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_BackgroundSnapshotSampler) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create WaterForwardPass background snapshot sampler!");
        }
        RegisterResource([this] { vkDestroySampler(m_Device, m_BackgroundSnapshotSampler, nullptr); });

        // =====================================================================================
        // STEP 1 -- set 0 layout: 7 bindings (no shadow/World-Probe-Grid/MegaLights resources --
        // water has no diffuse/shadowed term, see this class' own header comment).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[7]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // MaterialParamsSSBO
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };           // EntityTransformSSBO (vertex-stage only)
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_TLAS
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // FallbackVertexBuffer (HWRT)
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // FallbackIndexBuffer (HWRT)
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // EntityDrawRangeBuffer (HWRT)
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_BackgroundSnapshot

        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 };             // bindings 0,1,3,4,5
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 }; // binding 2
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };     // binding 6
        auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device,
            std::span{ bindings, 7 }, std::span{ poolSizes, 3 });
        m_SetLayout = descSet.layout;
        m_DescriptorPool = descSet.pool;
        m_Set = descSet.set;
        RegisterResource([this] {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        });

        // =====================================================================================
        // STEP 2 -- writes. Single-element MaterialParams buffer -- see Init()'s own
        // `waterMaterial` parameter comment for why this pass doesn't share TransparentForwardPass's
        // own full-table upload. Everything else is already fully built by the time this Init()
        // runs (traceContext is required to already be Init'd), so written once, here.
        // =====================================================================================
        m_MaterialParamsBuffer.Create(allocator, sizeof(MaterialParameters),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        RegisterResource([this] { m_MaterialParamsBuffer.Destroy(); });
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            vkCmdUpdateBuffer(cmd, m_MaterialParamsBuffer.Handle(), 0, sizeof(MaterialParameters), &waterMaterial);
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        });
        VkDescriptorBufferInfo materialParamsInfo{ m_MaterialParamsBuffer.Handle(), 0, m_MaterialParamsBuffer.Size() };
        VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSetAccelerationStructureKHR tlasWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        tlasWrite.accelerationStructureCount = 1;
        tlasWrite.pAccelerationStructures = &tlasHandle;

        VkDescriptorBufferInfo fallbackVertexInfo{ fallbackVertexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo fallbackIndexInfo{ fallbackIndexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo drawRangeInfo{ drawRangeBuffer, 0, VK_WHOLE_SIZE };

        VkDescriptorImageInfo backgroundSnapshotInfo{};
        backgroundSnapshotInfo.sampler = m_BackgroundSnapshotSampler;
        backgroundSnapshotInfo.imageView = m_BackgroundSnapshotView;
        backgroundSnapshotInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writes[7]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr };
        writes[2].pNext = &tlasWrite;
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fallbackVertexInfo, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fallbackIndexInfo, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &drawRangeInfo, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 6, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &backgroundSnapshotInfo, nullptr, nullptr };

        vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);

        // =====================================================================================
        // STEP 3 -- pipeline layout: 3 sets (own set 0 above, plus SurfaceCacheTraceContext's set
        // 1 / set 2, borrowed unmodified -- same 3-set shape TransparentForwardPass/
        // TessellationPass already establish).
        // =====================================================================================
        VkDescriptorSetLayout setLayouts[3] = {
            m_SetLayout, traceContext.GetMeshSdfTraceSetLayout(), traceContext.GetSurfaceCacheSamplingSetLayout()
        };

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(WaterForwardConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 3;
        pipelineLayoutInfo.pSetLayouts = setLayouts;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        if (vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create WaterForwardPass pipeline layout!");
        }
        RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); });

        // =====================================================================================
        // STEP 4 -- graphics pipeline, built manually (same reasoning as TessellationPass'
        // own STEP 4): real vertex-attribute input (geometry::FallbackVertex), depth test ENABLED
        // but depth WRITE disabled (read-only against the opaque scene's own depth, like glass),
        // reversed-Z VK_COMPARE_OP_GREATER. blendEnable=FALSE -- this pass composes manually
        // in-shader against the background snapshot and writes the final color directly (see
        // this class' own header comment).
        // =====================================================================================
        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/WaterForward.vert.spv");
        VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/WaterForward.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragModule;
        stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(geometry::FallbackVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributes[3]{};
        attributes[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(geometry::FallbackVertex, position) };
        attributes[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(geometry::FallbackVertex, normal) };
        attributes[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(geometry::FallbackVertex, uv) };

        VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = 3;
        vertexInput.pVertexAttributeDescriptions = attributes;

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
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // blendEnable=FALSE: this pass composes manually in-shader against the frozen background
        // snapshot and writes the already-composited final color directly -- see this class' own
        // header comment for why fixed-function blending (glass's own approach) doesn't fit here.
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

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

        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 1;
        pipelineRendering.pColorAttachmentFormats = &colorFormat;
        pipelineRendering.depthAttachmentFormat = depthFormat;

        VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.pNext = &pipelineRendering;

        if (vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create WaterForwardPass graphics pipeline!");
        }
        RegisterResource([this] { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); });

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, fragModule, nullptr);

        LOG_INFO(std::format("[WaterForwardPass] Initialized (water entity meshID={}).", m_WaterEntityID));
        return true;
    }

    // Shutdown() is inherited from RenderPass<WaterForwardPass>: runs the RegisterResource()
    // cleanups above in reverse (pipeline -> pipeline layout -> descriptor pool+layout -> material
    // params buffer -> sampler -> background snapshot image+view), the same dependency-safe order
    // the hand-written Shutdown() used. m_FallbackVertexBuffer/m_FallbackIndexBuffer/
    // m_WaterEntityDrawRange/m_WaterEntityID left un-reset (same reasoning as AtmosCloudsPass's
    // m_OutputExtent: private, no getters, unconditionally re-set at the top of every InitImpl()).

    void WaterForwardPass::RecordDraw(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
        VkImage colorImage, VkImageView colorView, VkImage depthImage, VkImageView depthView,
        VkExtent2D extent, uint32_t traceMode, uint32_t frameIndex, float timeSeconds,
        const SurfaceCacheTraceContext& traceContext) {

        // =====================================================================================
        // STEP A -- snapshot the composited scene color into m_BackgroundSnapshotImage BEFORE
        // this pass renders anything (see this class' own header comment for the full rationale).
        // Barriers: colorImage GENERAL->GENERAL (RAW, blit source) -- srcStageMask covers BOTH the
        // denoiser compute pass AND any forward pass' own fragment writes that may have run just
        // before this one this frame (this pass is recorded LAST, after TransparentForwardPass/
        // TessellationPass -- see ClusterRenderPipeline::RecordFrame's own call-site ordering),
        // same defensive breadth as TessellationPass::RecordDraw's own entry barrier comment.
        // m_BackgroundSnapshotImage SHADER_READ_ONLY_OPTIMAL->TRANSFER_DST_OPTIMAL (WAR: this
        // frame's new blit write must wait on last frame's fragment-shader read of the same image).
        // =====================================================================================
        VkImageMemoryBarrier2 preBlitBarriers[2]{};
        preBlitBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preBlitBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        preBlitBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        preBlitBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        preBlitBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        preBlitBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        preBlitBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        preBlitBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBlitBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBlitBarriers[0].image = colorImage;
        preBlitBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        preBlitBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preBlitBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        preBlitBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        preBlitBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        preBlitBarriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preBlitBarriers[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        preBlitBarriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        preBlitBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBlitBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preBlitBarriers[1].image = m_BackgroundSnapshotImage;
        preBlitBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo preBlitDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preBlitDep.imageMemoryBarrierCount = 2;
        preBlitDep.pImageMemoryBarriers = preBlitBarriers;
        vkCmdPipelineBarrier2(cmd, &preBlitDep);

        VkImageBlit blitRegion{};
        blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blitRegion.srcOffsets[0] = { 0, 0, 0 };
        blitRegion.srcOffsets[1] = { static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1 };
        blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blitRegion.dstOffsets[0] = { 0, 0, 0 };
        blitRegion.dstOffsets[1] = { static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1 };
        // NEAREST: a 1:1-resolution copy has no resampling to do -- see m_BackgroundSnapshotSampler's
        // own comment on why the SHADER READ side uses LINEAR instead.
        vkCmdBlitImage(cmd, colorImage, VK_IMAGE_LAYOUT_GENERAL,
            m_BackgroundSnapshotImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blitRegion, VK_FILTER_NEAREST);

        // =====================================================================================
        // STEP B -- transition color (GENERAL -> COLOR_ATTACHMENT_OPTIMAL, pure WAR against the
        // blit read above -- no memory to make visible, only ordering) and depth (READ_ONLY ->
        // ATTACHMENT_OPTIMAL, test-only, identical to TessellationPass' own entry barrier
        // minus the WRITE access) for this pass' rendering scope, and finish bringing the
        // snapshot to SHADER_READ_ONLY_OPTIMAL (RAW against the blit write above).
        // =====================================================================================
        VkImageMemoryBarrier2 toAttachmentBarriers[3]{};
        toAttachmentBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachmentBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        toAttachmentBarriers[0].srcAccessMask = 0;
        toAttachmentBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachmentBarriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachmentBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toAttachmentBarriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachmentBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[0].image = colorImage;
        toAttachmentBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        toAttachmentBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachmentBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toAttachmentBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        toAttachmentBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        toAttachmentBarriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        toAttachmentBarriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        toAttachmentBarriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        toAttachmentBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[1].image = depthImage;
        toAttachmentBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        toAttachmentBarriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachmentBarriers[2].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        toAttachmentBarriers[2].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toAttachmentBarriers[2].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        toAttachmentBarriers[2].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        toAttachmentBarriers[2].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toAttachmentBarriers[2].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toAttachmentBarriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[2].image = m_BackgroundSnapshotImage;
        toAttachmentBarriers[2].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo toAttachmentDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDep.imageMemoryBarrierCount = 3;
        toAttachmentDep.pImageMemoryBarriers = toAttachmentBarriers;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDep);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Overwrite (blendEnable=false) on top of the snapshotted frame.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Test against the opaque scene's own depth -- never written (depthWriteEnable = false).
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, extent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

        VkDescriptorSet sets[3] = { m_Set, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 3, sets, 0, nullptr);

        VkDeviceSize vertexOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m_FallbackVertexBuffer, &vertexOffset);
        vkCmdBindIndexBuffer(cmd, m_FallbackIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        WaterForwardConstants pc{};
        pc.viewProj = viewProj;
        pc.cameraPositionWorldX = cameraPositionWorld.x;
        pc.cameraPositionWorldY = cameraPositionWorld.y;
        pc.cameraPositionWorldZ = cameraPositionWorld.z;
        pc.entityID = m_WaterEntityID;
        pc.traceMode = traceMode;
        pc.frameIndex = frameIndex;
        pc.entityCount = traceContext.GetEntityCount();
        pc.viewportWidth = static_cast<float>(extent.width);
        pc.viewportHeight = static_cast<float>(extent.height);
        pc.timeSeconds = timeSeconds;
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

        VkViewport viewport{};
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDrawIndexed(cmd, m_WaterEntityDrawRange.indexCount, 1, m_WaterEntityDrawRange.firstIndex,
            m_WaterEntityDrawRange.vertexOffset, 0);

        vkCmdEndRendering(cmd);

        // --- Restore color/depth to the layout the REST of the frame expects (identical to
        // TessellationPass' own restore, minus the depth-WRITE access this pass never uses --
        // see that class' own comment). The background snapshot image needs NO restore barrier: it
        // is already in SHADER_READ_ONLY_OPTIMAL from STEP B above, which is exactly the
        // precondition STEP A expects at the start of next frame's call. ---
        VkImageMemoryBarrier2 restoreBarriers[2]{};
        restoreBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restoreBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        restoreBarriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        restoreBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
        restoreBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
        restoreBarriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        restoreBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        restoreBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarriers[0].image = colorImage;
        restoreBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        restoreBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restoreBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        restoreBarriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        restoreBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        restoreBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        restoreBarriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        restoreBarriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        restoreBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarriers[1].image = depthImage;
        restoreBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        VkDependencyInfo restoreDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        restoreDep.imageMemoryBarrierCount = 2;
        restoreDep.pImageMemoryBarriers = restoreBarriers;
        vkCmdPipelineBarrier2(cmd, &restoreDep);
    }

}
