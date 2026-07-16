#include "renderer/passes/TransparentForwardPass.h"

#include <format>

#include "core/Logger.h"
#include "geometry/GpuPageTable.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of TransparentViewParamsUBO in TransparentForward.vert/.frag
        // (std140): mat4 (64 bytes) + mat4 (64 bytes) + vec3 (12 bytes) + 1 pad float rounding up
        // to a 16-byte boundary (144 bytes total) -- same shape family as ClusterResolvePass.cpp's
        // own ResolveViewParams.
        struct TransparentViewParams {
            maths::mat4 view;
            maths::mat4 proj;
            float sunDirectionX = 0.0f;
            float sunDirectionY = 0.0f;
            float sunDirectionZ = 0.0f;
            float _pad0 = 0.0f;
        };
        static_assert(sizeof(TransparentViewParams) == 144,
            "TransparentViewParams must match TransparentViewParamsUBO in TransparentForward.vert/.frag exactly (std140 layout)");

    } // namespace

    void TransparentForwardPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkBuffer pageTableBuffer, VkBuffer compressedPhysicalPoolBuffer,
        VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer wpoGlobalsBuffer,
        const std::array<MaterialParameters, kMaxMaterials>& materialTable,
        const std::vector<geometry::ClusterIndexEntry>& indexEntries,
        const std::vector<geometry::DAGNodeEntry>& dagEntries,
        VkFormat colorFormat, VkFormat depthFormat) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // --- Build the static leaf-cluster list for every transparent entity (see class comment
        // for why this is a one-time CPU walk, not a per-frame GPU LOD cut). ---
        std::vector<TransparentClusterEntry> hostEntries;
        for (size_t i = 0; i < indexEntries.size(); ++i) {
            const geometry::ClusterIndexEntry& entry = indexEntries[i];
            const geometry::DAGNodeEntry& dagNode = dagEntries[i];
            if (dagNode.level != 0u) {
                continue; // Not a leaf (full/finest geometry) -- see class comment.
            }
            uint32_t materialSlot = (entry.materialID < kMaxMaterials) ? entry.materialID : (kMaxMaterials - 1u);
            if (materialTable[materialSlot].alpha >= 1.0f) {
                continue; // Opaque -- already handled by the normal Nanite VisBuffer pipeline.
            }

            TransparentClusterEntry hostEntry{};
            hostEntry.boundsMin = maths::vec3(entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2]);
            hostEntry.boundsMax = maths::vec3(entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2]);
            hostEntry.maxWPOAmplitude = entry.maxWPOAmplitude;
            hostEntry.logicalPageID = geometry::GpuPageTable::LogicalAddressToPageID(entry.virtualAddress);
            hostEntry.indexCount = entry.indexCount;
            hostEntry.clusterID = entry.clusterID;
            hostEntry.entityID = entry.entityID;
            hostEntry.materialID = entry.materialID;
            hostEntry.maskTextureIndex = entry.maskTextureIndex;
            hostEntries.push_back(hostEntry);
        }

        m_ClusterCount = static_cast<uint32_t>(hostEntries.size());
        LOG_INFO(std::format("[TransparentForwardPass] {} transparent leaf cluster(s) found across every entity.", m_ClusterCount));
        if (m_ClusterCount == 0) {
            return; // Nothing to draw this run -- RecordDraw() checks GetTransparentClusterCount() itself.
        }

        // --- Buffers. ---
        m_ClusterEntriesBuffer.Create(allocator, sizeof(TransparentClusterEntry) * m_ClusterCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_IndirectCommandsBuffer.Create(allocator, sizeof(VkDrawIndexedIndirectCommand) * m_ClusterCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ViewParamsBuffer.Create(allocator, sizeof(TransparentViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // One-time upload of the static entries list -- mirrors every other pass's own one-shot
        // setup submit in this codebase.
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            vkCmdUpdateBuffer(cmd, m_ClusterEntriesBuffer.Handle(), 0,
                sizeof(TransparentClusterEntry) * m_ClusterCount, hostEntries.data());
        });

        // =====================================================================================
        // Descriptor pool, shared by both descriptor sets below.
        // =====================================================================================
        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 };            // Compact(3) + Forward(5: entries, pool, entityXform, entityData, materialParams).
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 };            // Forward: WPOGlobals, ViewParams, g_ShadowSunLevels (binding 10, wired in by SetVirtualShadowMap).
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };    // Forward: shadow physical atlas.

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        // =====================================================================================
        // Set/Pipeline 1: TransparentClusterCompact.comp -- bindings 0..2.
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[3]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // GeometryPageTableSSBO (borrowed)
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // TransparentClusterEntriesSSBO
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // IndirectCommandsSSBO

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 3;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_CompactSetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_CompactSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_CompactDescriptorSet));

            VkDescriptorBufferInfo pageTableInfo{ pageTableBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo entriesInfo{ m_ClusterEntriesBuffer.Handle(), 0, m_ClusterEntriesBuffer.Size() };
            VkDescriptorBufferInfo commandsInfo{ m_IndirectCommandsBuffer.Handle(), 0, m_IndirectCommandsBuffer.Size() };

            VkWriteDescriptorSet writes[3]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entriesInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_CompactDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &commandsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 3, writes, 0, nullptr);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_CompactSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_CompactPipelineLayout));

            VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/TransparentClusterCompact.comp.spv");
            m_CompactPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_CompactPipelineLayout, shaderModule);
            vkDestroyShaderModule(m_Device, shaderModule, nullptr);
        }

        // =====================================================================================
        // Set/Pipeline 2: TransparentForward.vert/.frag -- bindings 0..6 written here; 7..10
        // (Phase 3's renderer::VirtualShadowMapPass resources) left for SetVirtualShadowMap().
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[11]{};
            bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };   // TransparentClusterEntriesSSBO
            bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };   // CompressedClusterPoolSSBO
            bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };   // EntityTransformBuffer
            bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };   // EntityDataBuffer
            bindings[4] = { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };   // WPOGlobalsUBO
            bindings[5] = { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // TransparentViewParamsUBO
            bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // MaterialParamsSSBO
            bindings[7] = { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_ShadowPhysicalAtlas
            bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_ShadowPageTable
            bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_ShadowFeedback
            bindings[10] = { 10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_ShadowSunLevels

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 11;
            layoutInfo.pBindings = bindings;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_ForwardSetLayout));

            VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            setAlloc.descriptorPool = m_DescriptorPool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &m_ForwardSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_ForwardDescriptorSet));

            VkDescriptorBufferInfo entriesInfo{ m_ClusterEntriesBuffer.Handle(), 0, m_ClusterEntriesBuffer.Size() };
            VkDescriptorBufferInfo compressedPoolInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo entityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo wpoGlobalsInfo{ wpoGlobalsBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

            // renderer::MaterialParameterTable's runtime PBR table (renderer::VulkanContext::
            // GetMaterialTable(), already uploaded once into ClusterResolvePass's own SSBO) --
            // reuploaded into THIS pass's own SSBO copy rather than sharing that same buffer handle,
            // matching this codebase's convention of every pass owning its own descriptor-bound
            // resources (no cross-pass buffer-handle borrowing beyond what each Init() explicitly
            // receives as a parameter).
            m_MaterialParamsBuffer.Create(allocator, sizeof(MaterialParameters) * kMaxMaterials,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdUpdateBuffer(cmd, m_MaterialParamsBuffer.Handle(), 0, sizeof(MaterialParameters) * kMaxMaterials, materialTable.data());
            });
            VkDescriptorBufferInfo materialParamsInfo{ m_MaterialParamsBuffer.Handle(), 0, m_MaterialParamsBuffer.Size() };

            VkWriteDescriptorSet writes[7]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entriesInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &wpoGlobalsInfo, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 7, writes, 0, nullptr);
            // Bindings 7-10 (Phase 3's renderer::VirtualShadowMapPass resources) are intentionally
            // left unwritten here -- SetVirtualShadowMap() writes them once the caller has a
            // VirtualShadowMapPass to bind, same convention as renderer::ClusterResolvePass's own.

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts = &m_ForwardSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_ForwardPipelineLayout));

            VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/TransparentForward.vert.spv");
            VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/TransparentForward.frag.spv");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr };

            // Bindless: no vertex input state, same as every other cluster-driven pipeline in this
            // codebase (see ClusterHardwareRasterPass's identical convention).
            VkPipelineVertexInputStateCreateInfo vertexInputInfo{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            // No back-face culling: a fully single-sided translucent/transparent surface (only its
            // near faces rendered) looks flat and unconvincing -- rendering both faces (with no
            // depth write, see below) gives a fuller "you can see the far wall of the glass through
            // the near one" silhouette without needing a real two-pass (back-then-front) draw order.
            VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Standard "over" alpha blend against whatever is already in the color attachment (the
            // fully-composited opaque scene) -- see class comment.
            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_TRUE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;

            VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            pipelineRendering.colorAttachmentCount = 1;
            pipelineRendering.pColorAttachmentFormats = &colorFormat;
            pipelineRendering.depthAttachmentFormat = depthFormat;

            // Depth-TESTED (reversed-Z, matching every other pass, see maths::mat4::PerspectiveVulkan's
            // own comment) but NOT written -- transparent surfaces must be correctly hidden behind
            // opaque geometry, but must never occlude each other via the depth buffer (see class
            // comment on why ordering between different transparent entities is otherwise unsorted).
            VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_FALSE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;
            depthStencil.depthBoundsTestEnable = VK_FALSE;
            depthStencil.stencilTestEnable = VK_FALSE;

            VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            pipelineInfo.pNext = &pipelineRendering;
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
            pipelineInfo.layout = m_ForwardPipelineLayout;

            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_ForwardPipeline));

            vkDestroyShaderModule(m_Device, vertModule, nullptr);
            vkDestroyShaderModule(m_Device, fragModule, nullptr);
        }

        LOG_INFO(std::format("[TransparentForwardPass] Initialized ({} static leaf cluster(s)).", m_ClusterCount));
    }

    void TransparentForwardPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_ForwardPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ForwardPipeline, nullptr);
            if (m_ForwardPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ForwardPipelineLayout, nullptr);
            if (m_ForwardSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ForwardSetLayout, nullptr);

            if (m_CompactPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_CompactPipeline, nullptr);
            if (m_CompactPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_CompactPipelineLayout, nullptr);
            if (m_CompactSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_CompactSetLayout, nullptr);

            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees both descriptor sets -- not freed individually.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
        }

        m_ForwardPipeline = VK_NULL_HANDLE;
        m_ForwardPipelineLayout = VK_NULL_HANDLE;
        m_ForwardSetLayout = VK_NULL_HANDLE;
        m_ForwardDescriptorSet = VK_NULL_HANDLE;
        m_CompactPipeline = VK_NULL_HANDLE;
        m_CompactPipelineLayout = VK_NULL_HANDLE;
        m_CompactSetLayout = VK_NULL_HANDLE;
        m_CompactDescriptorSet = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;

        m_ClusterEntriesBuffer.Destroy();
        m_IndirectCommandsBuffer.Destroy();
        m_ViewParamsBuffer.Destroy();
        m_MaterialParamsBuffer.Destroy();

        m_ClusterCount = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void TransparentForwardPass::SetVirtualShadowMap(const VirtualShadowMapPass& vsm) {
        if (m_ClusterCount == 0) {
            return; // No descriptor set was ever allocated -- see Init()'s own early-out.
        }

        VkDescriptorImageInfo atlasImageInfo{};
        atlasImageInfo.sampler = vsm.GetPhysicalAtlasSampler();
        atlasImageInfo.imageView = vsm.GetPhysicalAtlasView();
        atlasImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo pageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo feedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo sunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[4]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &atlasImageInfo, nullptr, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sunLevelsInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);
    }

    void TransparentForwardPass::RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView, VkImageView depthView,
        VkExtent2D renderExtent, const maths::mat4& view, const maths::mat4& proj,
        VkBuffer decompressedIndexPoolBuffer, const maths::vec3& sunDirection) {
        if (m_ClusterCount == 0) {
            return;
        }

        // --- Upload this frame's view/proj/sunDirection. ---
        TransparentViewParams viewParams{};
        viewParams.view = view;
        viewParams.proj = proj;
        viewParams.sunDirectionX = sunDirection.x;
        viewParams.sunDirectionY = sunDirection.y;
        viewParams.sunDirectionZ = sunDirection.z;
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(TransparentViewParams), &viewParams);

        VkMemoryBarrier2 uboBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uboBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uboBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uboBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        uboBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;

        VkDependencyInfo uboDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        uboDependency.memoryBarrierCount = 1;
        uboDependency.pMemoryBarriers = &uboBarrier;
        vkCmdPipelineBarrier2(cmd, &uboDependency);

        // --- Resolve this frame's physical page + residency for every static entry (see
        // TransparentClusterCompact.comp's own comment). ---
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompactPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CompactPipelineLayout, 0, 1, &m_CompactDescriptorSet, 0, nullptr);
        uint32_t groupCount = (m_ClusterCount + kCompactWorkgroupSize - 1) / kCompactWorkgroupSize;
        vkCmdDispatch(cmd, groupCount, 1, 1);

        VkMemoryBarrier2 indirectBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        indirectBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        indirectBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        indirectBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        indirectBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

        VkDependencyInfo indirectDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        indirectDependency.memoryBarrierCount = 1;
        indirectDependency.pMemoryBarriers = &indirectBarrier;
        vkCmdPipelineBarrier2(cmd, &indirectDependency);

        // --- Transition the target color image GENERAL -> COLOR_ATTACHMENT_OPTIMAL -- mirrors
        // renderer::debug::DebugTextOverlay::RecordDraw's identical dance against the same class of
        // image (see TransparentForwardPass.h's own comment on why both this pass's candidate
        // images already carry VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT). ---
        VkImageMemoryBarrier2 toAttachment{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toAttachment.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toAttachment.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
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
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Preserve the fully-composited opaque scene underneath.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Depth is already VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL by this point in the
        // frame (see renderer::ClusterRenderPipeline::RecordFrame's own ordering) -- no transition,
        // no barrier: this pass only ever reads it (depthWriteEnable=FALSE in the pipeline state
        // above), so there is no write to synchronize against.
        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, renderExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ForwardPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ForwardPipelineLayout, 0, 1, &m_ForwardDescriptorSet, 0, nullptr);
        vkCmdBindIndexBuffer(cmd, decompressedIndexPoolBuffer, 0, VK_INDEX_TYPE_UINT32);

        VkViewport viewport{};
        viewport.width = static_cast<float>(renderExtent.width);
        viewport.height = static_cast<float>(renderExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = renderExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDrawIndexedIndirect(cmd, m_IndirectCommandsBuffer.Handle(), 0, m_ClusterCount, sizeof(VkDrawIndexedIndirectCommand));

        vkCmdEndRendering(cmd);

        VkImageMemoryBarrier2 toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toGeneral.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
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
