#include "renderer/passes/TessellationPass.h"

#include <array>
#include <cstring>
#include <format>
#include <span>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h" // geometry::FallbackVertex
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/passes/WorldProbeGridPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of TessellationConstants in Tessellation.vert/.tesc/.tese/.frag --
        // flat scalar fields throughout, matching this codebase's established push-constant
        // convention (see GlobalSDFCompositePC's own comment). `materialID` replaces the original
        // hero-only layout's trailing `_pad1` float (same 4-byte slot, now a real per-draw field --
        // see class comment on why every entity now carries its own materialID instead of this pass
        // shading a single fixed material).
        struct TessellationConstants {
            maths::mat4 viewProj;
            float cameraPositionWorldX = 0, cameraPositionWorldY = 0, cameraPositionWorldZ = 0;
            float _pad0 = 0;
            uint32_t entityID = 0;
            uint32_t traceMode = 0;
            uint32_t frameIndex = 0;
            uint32_t entityCount = 0;
            float viewportWidth = 0, viewportHeight = 0;
            float displacementScale = 0;
            uint32_t materialID = 0;
        };
        static_assert(sizeof(TessellationConstants) == 112,
            "TessellationConstants must match Tessellation.vert/.tesc/.tese/.frag's push_constant block exactly");
        static_assert(sizeof(TessellationConstants) <= 128,
            "Must stay under the Vulkan-guaranteed minimum maxPushConstantsSize (128 bytes) -- see this phase's own 'no runtime capability query' decision.");

        // Byte-for-byte mirror of Tessellation.frag's own TessellationLightingUBO (std140): only
        // the sun fields TransparentViewParamsUBO also carries -- viewProj/cameraPos are already in
        // the push constants above (needed by the vertex/tessellation stages too), so this UBO
        // does not duplicate them.
        struct TessellationLightingUBO {
            float sunDirX = 0, sunDirY = 0, sunDirZ = 0, sunIntensity = 0;
            float sunColorR = 0, sunColorG = 0, sunColorB = 0, _pad0 = 0;
        };
        static_assert(sizeof(TessellationLightingUBO) == 32,
            "TessellationLightingUBO must match Tessellation.frag's own TessellationLightingUBO exactly (std140 layout)");

        // Byte-for-byte mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO (std140) --
        // identical to renderer::TransparentForwardPass's own copy.
        struct WorldProbeGridParamsUBO {
            float gridOriginX = 0, gridOriginY = 0, gridOriginZ = 0;
            float probeSpacing = 0;
            float gridResolution = 0;
            float _pad0 = 0, _pad1 = 0, _pad2 = 0;
        };
        static_assert(sizeof(WorldProbeGridParamsUBO) == 32,
            "Must match world_probe_sampling.glsl's WorldProbeGridParamsUBO exactly (std140 layout)");

    } // namespace

    bool TessellationPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkFormat colorFormat, VkFormat depthFormat,
        VkBuffer entityTransformBuffer,
        const std::array<MaterialParameters, kMaxMaterials>& materialTable,
        VkBuffer fallbackVertexBuffer, VkBuffer fallbackIndexBuffer,
        const std::vector<TessellatedEntityDrawInfo>& entities,
        VkAccelerationStructureKHR tlasHandle, VkBuffer drawRangeBuffer,
        VkBuffer lightBuffer, VkDeviceSize lightBufferSize,
        const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
        const SurfaceCacheTraceContext& traceContext) {
        // Self-reinit (see ShadowMapPass's own migration comment for the identical pattern):
        // Shutdown() clears m_Device, which RenderPass<TessellationPass>::Init() already set to
        // this call's value just before invoking this function -- restore it.
        Shutdown();
        m_Device = device;
        m_Entities = entities;
        m_FallbackVertexBuffer = fallbackVertexBuffer;
        m_FallbackIndexBuffer = fallbackIndexBuffer;

        // =====================================================================================
        // STEP 1 -- set 0 layout: 14 bindings (own view/sun UBO, sun-only shadow resources,
        // World Probe Grid, material params, entity transform (vertex-stage), shared TLAS,
        // MegaLights' light SSBO, and the fallback-geometry trio for the optional reflection
        // trace's HWRT path) -- mirrors renderer::TransparentForwardPass's own Forward set shape.
        // Built unconditionally even if `entities` is empty -- a valid, if degenerate, pass
        // instance whose RecordDraw() is simply always a no-op (see that method's own comment); no
        // early-return here keeps Shutdown()/Init() symmetric regardless of entity count, matching
        // this codebase's own established "always own valid Vulkan objects once Init() returns
        // true" convention.
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[14]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // TessellationLightingUBO (sun dir/intensity/color)
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_ShadowPhysicalAtlas
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // g_ShadowPageTable
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // g_ShadowFeedback
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // g_ShadowSunLevels
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_WorldProbeGrid
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // WorldProbeGridParamsUBO
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };         // MaterialParamsSSBO[kMaxMaterials]
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };           // EntityTransformSSBO (vertex-stage only)
        bindings[9] = { 9, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_TLAS (shared: MegaLights TraceShadowRay + reflection TraceHWRT)
        bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };       // MegaLights g_Lights
        bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };       // FallbackVertexBuffer (HWRT)
        bindings[12] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };       // FallbackIndexBuffer (HWRT)
        bindings[13] = { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };       // EntityDrawRangeBuffer (HWRT)

        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 };             // bindings 2,3,7,8,10,11,12,13
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 };             // bindings 0,4,6
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 };     // bindings 1,5
        poolSizes[3] = { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 }; // binding 9
        auto descSet = VulkanUtils::CreateDescriptorSetLayoutPoolAndSet(m_Device,
            std::span{ bindings, 14 }, std::span{ poolSizes, 4 });
        m_SetLayout = descSet.layout;
        m_DescriptorPool = descSet.pool;
        m_Set = descSet.set;
        RegisterResource([this] {
            vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
        });

        // =====================================================================================
        // STEP 2 -- writes. Every resource is already fully built by the time this Init() runs
        // (vsm/worldProbes/traceContext/MegaLights are all required to already be Init'd) --
        // everything is written once, here, mirroring renderer::TransparentForwardPass::Init's own
        // identical STEP 2.
        // =====================================================================================
        m_ViewParamsBuffer.Create(allocator, sizeof(TessellationLightingUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        RegisterResource([this] { m_ViewParamsBuffer.Destroy(); });
        VkDescriptorBufferInfo lightingInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };

        VkDescriptorImageInfo shadowAtlasInfo{};
        shadowAtlasInfo.sampler = vsm.GetPhysicalAtlasSampler();
        shadowAtlasInfo.imageView = vsm.GetPhysicalAtlasView();
        shadowAtlasInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo shadowPageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo shadowFeedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo shadowSunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };

        VkDescriptorImageInfo worldProbeGridInfo{};
        worldProbeGridInfo.sampler = worldProbes.GetGridSampler();
        worldProbeGridInfo.imageView = worldProbes.GetGridView();
        worldProbeGridInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // world_probe_sampling.glsl's WorldProbeGridParamsUBO -- persistently mapped, refreshed
        // every RecordDraw() call (gridOrigin recenters every frame) -- see
        // m_WorldProbeGridParamsBuffer's own header comment. Allocated here so its handle can be
        // written into binding 6 below; its CONTENTS are left zero-initialized until the first
        // RecordDraw() call.
        m_WorldProbeGridParamsBuffer.Create(allocator, sizeof(WorldProbeGridParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        RegisterResource([this] { m_WorldProbeGridParamsBuffer.Destroy(); });
        VkDescriptorBufferInfo worldProbeGridParamsInfo{ m_WorldProbeGridParamsBuffer.Handle(), 0, m_WorldProbeGridParamsBuffer.Size() };

        // Full MaterialParameters[kMaxMaterials] buffer -- generalized from the original single-
        // element hero buffer (see Init()'s own `materialTable` parameter comment and class
        // comment); each entity's draw indexes its own slot via its push-constant materialID,
        // exactly mirroring renderer::TransparentForwardPass::Init's own identical upload.
        m_MaterialParamsBuffer.Create(allocator, sizeof(MaterialParameters) * kMaxMaterials,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        RegisterResource([this] { m_MaterialParamsBuffer.Destroy(); });
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            vkCmdUpdateBuffer(cmd, m_MaterialParamsBuffer.Handle(), 0, sizeof(MaterialParameters) * kMaxMaterials, materialTable.data());
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        });
        VkDescriptorBufferInfo materialParamsInfo{ m_MaterialParamsBuffer.Handle(), 0, m_MaterialParamsBuffer.Size() };
        VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSetAccelerationStructureKHR tlasWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
        tlasWrite.accelerationStructureCount = 1;
        tlasWrite.pAccelerationStructures = &tlasHandle;

        VkDescriptorBufferInfo megaLightsInfo{ lightBuffer, 0, lightBufferSize };
        VkDescriptorBufferInfo fallbackVertexInfo{ fallbackVertexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo fallbackIndexInfo{ fallbackIndexBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo drawRangeInfo{ drawRangeBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[14]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lightingInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowAtlasInfo, nullptr, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowPageTableInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowFeedbackInfo, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &shadowSunLevelsInfo, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &worldProbeGridInfo, nullptr, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 6, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &worldProbeGridParamsInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
        writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 9, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr };
        writes[9].pNext = &tlasWrite;
        writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &megaLightsInfo, nullptr };
        writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fallbackVertexInfo, nullptr };
        writes[12] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fallbackIndexInfo, nullptr };
        writes[13] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_Set, 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &drawRangeInfo, nullptr };

        vkUpdateDescriptorSets(m_Device, 14, writes, 0, nullptr);

        // =====================================================================================
        // STEP 3 -- pipeline layout: 3 sets (own set 0 above, plus SurfaceCacheTraceContext's
        // set 1 / set 2, borrowed unmodified). Push constant range spans all 4 stages this
        // pipeline uses (unlike TransparentForwardPass's vertex+fragment only).
        // =====================================================================================
        VkDescriptorSetLayout setLayouts[3] = {
            m_SetLayout, traceContext.GetMeshSdfTraceSetLayout(), traceContext.GetSurfaceCacheSamplingSetLayout()
        };

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
            | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(TessellationConstants);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 3;
        pipelineLayoutInfo.pSetLayouts = setLayouts;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));
        RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); });

        // =====================================================================================
        // STEP 4 -- graphics pipeline, built manually (hardcoded for the 4-stage vertex/
        // tessellation-control/tessellation-evaluation/fragment shape -- see this class' own
        // header comment): real vertex-attribute input (geometry::FallbackVertex, mirrors
        // renderer::TransparentForwardPass's own vertex input state exactly), PATCH_LIST topology
        // (patchControlPoints=3 -- the Fallback Mesh's own indexed triangle-list is reinterpreted
        // as 3-vertex patches, no index-buffer change needed), no blending (opaque), depth test
        // ENABLED and depth WRITE ENABLED (unlike glass -- see this class' own header comment),
        // reversed-Z VK_COMPARE_OP_GREATER matching every other 3D pipeline. ONE pipeline is shared
        // by every tessellated entity's draw (see RecordDraw()'s own comment) -- no per-entity
        // pipeline variation needed, since every entity uses the same vertex layout/topology/
        // shading algorithm, only its own materialID/entityID/geometry range differ.
        // =====================================================================================
        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/Tessellation.vert.spv");
        VkShaderModule tescModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/Tessellation.tesc.spv");
        VkShaderModule teseModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/Tessellation.tese.spv");
        VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/Tessellation.frag.spv");

        VkPipelineShaderStageCreateInfo stages[4]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        stages[1].module = tescModule;
        stages[1].pName = "main";
        stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[2].stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        stages[2].module = teseModule;
        stages[2].pName = "main";
        stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[3].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[3].module = fragModule;
        stages[3].pName = "main";

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

        // PATCH_LIST, not TRIANGLE_LIST -- the tessellation control stage consumes 3-control-point
        // patches (see Tessellation.tesc's own layout(vertices=3) out).
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        // patchControlPoints=3: comfortably under the Vulkan-guaranteed maxTessellationPatchSize
        // minimum of 32 -- see this class' own header comment.
        VkPipelineTessellationStateCreateInfo tessellationState{ VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };
        tessellationState.patchControlPoints = 3;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        // Same winding convention as VulkanPipeline::CreateGraphicsPipeline (the main camera's
        // PerspectiveVulkan Y-flip reverses apparent 2D winding -- see that function's own comment).
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

        // No blending -- fully opaque (unlike TransparentForwardPass's standard "over" alpha blend).
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        // Depth test AND WRITE enabled (unlike renderer::TransparentForwardPass's read-only test) --
        // this pass is fully opaque and must occupy real depth so TransparentForwardPass's own
        // subsequent depth test correctly occludes glass against each tessellated entity's real
        // (displaced) surface -- see this class' own header comment and RecordDraw()'s own barrier
        // comment for the full synchronization consequence.
        VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
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
        pipelineInfo.stageCount = 4;
        pipelineInfo.pStages = stages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pTessellationState = &tessellationState; // Dedicated top-level field -- NOT extensible via pNext.
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.pNext = &pipelineRendering;

        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        RegisterResource([this] { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); });

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, tescModule, nullptr);
        vkDestroyShaderModule(m_Device, teseModule, nullptr);
        vkDestroyShaderModule(m_Device, fragModule, nullptr);

        LOG_INFO(std::format("[TessellationPass] Initialized ({} tessellated entities).", m_Entities.size()));
        return true;
    }

    // Shutdown() is inherited from RenderPass<TessellationPass>: runs the RegisterResource()
    // cleanups above in reverse (pipeline -> pipeline layout -> descriptor pool+layout ->
    // material-params buffer -> world-probe-grid-params buffer -> view-params buffer), the same
    // dependency-safe order the hand-written Shutdown() used. m_FallbackVertexBuffer/
    // m_FallbackIndexBuffer/m_Entities left un-reset (same reasoning as AtmosCloudsPass's
    // m_OutputExtent: private, no getters, unconditionally re-set at the top of every InitImpl()).

    void TessellationPass::RecordDraw(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
        VkImage colorImage, VkImageView colorView, VkImage depthImage, VkImageView depthView,
        VkExtent2D extent, uint32_t traceMode, uint32_t frameIndex,
        const SurfaceCacheTraceContext& traceContext, const WorldProbeGridPass& worldProbes,
        const SceneLights& sceneLights) {

        // No tessellated entity to draw this run (see Init()'s own comment on the empty-`entities`
        // case) -- nothing to transition/render, mirrors TransparentForwardPass::RecordDraw's own
        // zero-cluster early return.
        if (m_Entities.empty()) {
            return;
        }

        // Refresh this pass' own small sun-lighting UBO (binding 0) once per call, shared by every
        // entity's draw below -- mirrors renderer::TransparentForwardPass::RecordDraw's own
        // per-frame view-params refresh.
        TessellationLightingUBO lightingUBO{};
        lightingUBO.sunDirX = sceneLights.sun.direction.x;
        lightingUBO.sunDirY = sceneLights.sun.direction.y;
        lightingUBO.sunDirZ = sceneLights.sun.direction.z;
        lightingUBO.sunIntensity = sceneLights.sun.intensity;
        lightingUBO.sunColorR = sceneLights.sun.color.x;
        lightingUBO.sunColorG = sceneLights.sun.color.y;
        lightingUBO.sunColorB = sceneLights.sun.color.z;
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(lightingUBO), &lightingUBO);

        // Refresh the World Probe Grid params UBO's only per-frame-varying field (gridOrigin --
        // see m_WorldProbeGridParamsBuffer's own header comment).
        WorldProbeGridParamsUBO gridParams{};
        const maths::vec3& gridOrigin = worldProbes.GetGridOriginWorld();
        gridParams.gridOriginX = gridOrigin.x;
        gridParams.gridOriginY = gridOrigin.y;
        gridParams.gridOriginZ = gridOrigin.z;
        gridParams.probeSpacing = WorldProbeGridPass::kProbeSpacing;
        gridParams.gridResolution = static_cast<float>(WorldProbeGridPass::kGridResolution);
        std::memcpy(m_WorldProbeGridParamsBuffer.MappedData(), &gridParams, sizeof(gridParams));

        // The UBO update above must be visible before this frame's own fragment-stage reads.
        VkMemoryBarrier2 uboBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        uboBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        uboBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        uboBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        uboBarrier.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
        VkDependencyInfo uboDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        uboDep.memoryBarrierCount = 1;
        uboDep.pMemoryBarriers = &uboBarrier;
        vkCmdPipelineBarrier2(cmd, &uboDep);

        // --- Transition color (GENERAL -> COLOR_ATTACHMENT_OPTIMAL) and depth (READ_ONLY ->
        // ATTACHMENT_OPTIMAL, test-AND-write this time) for this pass' own rendering scope --
        // mirrors renderer::TransparentForwardPass::RecordDraw's own GENERAL<->ATTACHMENT dance,
        // except dstAccessMask for depth includes WRITE (this pass writes real depth, unlike
        // glass). ---
        VkImageMemoryBarrier2 toAttachmentBarriers[2]{};
        toAttachmentBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachmentBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toAttachmentBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
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
        toAttachmentBarriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        toAttachmentBarriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        toAttachmentBarriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        toAttachmentBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachmentBarriers[1].image = depthImage;
        toAttachmentBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        VkDependencyInfo toAttachmentDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDep.imageMemoryBarrierCount = 2;
        toAttachmentDep.pImageMemoryBarriers = toAttachmentBarriers;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDep);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Composite on top of the fully opaque+GI+denoised frame.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Test against the opaque scene's own existing depth -- never cleared.
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // Unlike glass (NONE) -- this pass' own depth write must persist for later consumers.

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, extent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        // Pipeline/descriptor sets/vertex+index buffers are bound ONCE outside the per-entity loop
        // below -- every tessellated entity shares the exact same pipeline and shared Fallback Mesh
        // buffers (see class comment), only the push constants and the draw's own
        // firstIndex/vertexOffset/indexCount change per entity, so re-binding any of this state
        // inside the loop would be redundant work with zero behavioral difference.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

        VkDescriptorSet sets[3] = { m_Set, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet() };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 3, sets, 0, nullptr);

        VkDeviceSize vertexOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m_FallbackVertexBuffer, &vertexOffset);
        vkCmdBindIndexBuffer(cmd, m_FallbackIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        VkViewport viewport{};
        viewport.width = static_cast<float>(extent.width);
        viewport.height = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // One indexed draw PER tessellated entity -- see class' own "ONE pipeline, MANY entities,
        // ONE draw call per entity" comment for why this stays a simple loop of direct draws
        // instead of TransparentForwardPass's own per-frame compact/indirect-multi-draw machinery.
        for (const TessellatedEntityDrawInfo& entity : m_Entities) {
            TessellationConstants pc{};
            pc.viewProj = viewProj;
            pc.cameraPositionWorldX = cameraPositionWorld.x;
            pc.cameraPositionWorldY = cameraPositionWorld.y;
            pc.cameraPositionWorldZ = cameraPositionWorld.z;
            pc.entityID = entity.entityID;
            pc.traceMode = traceMode;
            pc.frameIndex = frameIndex;
            pc.entityCount = traceContext.GetEntityCount();
            pc.viewportWidth = static_cast<float>(extent.width);
            pc.viewportHeight = static_cast<float>(extent.height);
            pc.displacementScale = kDisplacementScale;
            pc.materialID = entity.materialID;
            vkCmdPushConstants(cmd, m_PipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT
                    | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(pc), &pc);

            vkCmdDrawIndexed(cmd, entity.drawRange.indexCount, 1, entity.drawRange.firstIndex,
                entity.drawRange.vertexOffset, 0);
        }

        vkCmdEndRendering(cmd);

        // --- Restore both images to the layout the REST of the frame expects (color back to
        // GENERAL, depth back to READ_ONLY for renderer::TransparentForwardPass's own subsequent
        // depth test and the later HZB rebuild). UNLIKE renderer::TransparentForwardPass's own
        // restore barrier (which only ever READ depth, never wrote it, so a COMPUTE_SHADER/
        // SHADER_SAMPLED_READ source scope sufficed), THIS pass' restore barrier's source scope
        // must be EARLY/LATE_FRAGMENT_TESTS + DEPTH_STENCIL_ATTACHMENT_WRITE (what this pass
        // itself just did, across every tessellated entity's draw above), and its destination scope
        // must cover BOTH the next fragment-tests consumer (TransparentForwardPass's own depth
        // test) AND the later compute-shader sampled read (the HZB rebuild) -- a mechanical copy of
        // TransparentForwardPass's own restore barrier would silently leave that depth test
        // unsynchronized against this pass' real depth write, since that call site's own entry
        // barrier is hardcoded assuming only a prior COMPUTE_SHADER/SHADER_SAMPLED_READ source
        // (correct for its own no-op-if-tessellated-entity-ran-first case, but not for what THIS
        // pass just did to the SAME image). ---
        VkImageMemoryBarrier2 restoreBarriers[2]{};
        restoreBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restoreBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        restoreBarriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        restoreBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        restoreBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        restoreBarriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        restoreBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        restoreBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarriers[0].image = colorImage;
        restoreBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        restoreBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restoreBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        restoreBarriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        restoreBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        restoreBarriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
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
