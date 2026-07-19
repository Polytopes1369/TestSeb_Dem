#include "renderer/passes/FurStrandPass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <vector>

#include "core/Logger.h"
#include "core/EngineConfig.h"
#include "renderer/passes/ClusterCullingPass.h"      // ExtractFrustumPlanes
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/passes/WorldProbeGridPass.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // --- Hardcoded fur appearance (a warm brown pelt). Kept here rather than in config:: so the
        // Debug tab stays focused on the density/length/geometry knobs gap G10a asks for; the two most
        // impactful appearance values (spec/TRT intensity) ARE exposed live via config::fur. ---
        constexpr float kRootColor[3] = { 0.11f, 0.06f, 0.03f }; // dark brown at the roots (buried, in shadow)
        constexpr float kTipColor[3] = { 0.46f, 0.30f, 0.16f };  // lighter tan at the tips (sun-exposed)
        constexpr float kRootDarken = 0.30f;                      // pelt-depth AO floor at the roots
        constexpr float kShiftR = 0.05f;                          // primary R highlight tangent shift (toward N)
        constexpr float kShiftTRT = -0.08f;                       // secondary TRT highlight tangent shift (away from N)
        constexpr float kExponentR = 120.0f;                      // primary highlight sharpness
        constexpr float kExponentTRT = 22.0f;                     // secondary highlight sharpness (broader)

        // Mirror of FurStrandGen.comp's push constant (48 bytes).
        struct FurStrandGenParams {
            float worldOffsetX, worldOffsetY, worldOffsetZ;
            float radiusMin, radiusMax;
            float segmentLength;
            uint32_t boneCount;
            uint32_t requestedStrands;
            float lengthJitter;
            float rootLift;
            uint32_t seed;
            uint32_t _pad;
        };
        static_assert(sizeof(FurStrandGenParams) == 48, "FurStrandGenParams must match FurStrandGen.comp's push constant.");

        // Mirror of FurStrandCull.comp's CullParams UBO (std140, 208 bytes).
        struct FurCullParamsUBO {
            float frustumPlanes[24]; // 6 * vec4
            float cameraPos[4];
            maths::mat4 viewProj;    // 64 bytes
            float hzbMip0Size[2];
            float hzbMipCount;
            uint32_t enableOcclusion;
            uint32_t creatureMeshID;
            uint32_t maxStrands;
            float boundingRadius;
            uint32_t _pad0;
        };
        static_assert(sizeof(FurCullParamsUBO) == 208, "FurCullParamsUBO must match FurStrandCull.comp's CullParams (std140).");

        // Mirror of FurStrand.vert/.frag's FurRenderParamsUBO (std140, 176 bytes).
        struct FurRenderParamsUBO {
            maths::mat4 viewProj;                             // 64
            float cameraPos[3]; float furLength;             // 16
            float sunDirection[3]; float sunIntensity;       // 16
            float sunColor[3]; float furWidth;               // 16
            float rootColor[3]; float rootDarken;            // 16
            float tipColor[3]; float curlAmount;             // 16
            float shiftR, shiftTRT, exponentR, exponentTRT;  // 16
            float specIntensity, trtIntensity, globalTime; uint32_t creatureMeshID; // 16
        };
        static_assert(sizeof(FurRenderParamsUBO) == 176, "FurRenderParamsUBO must match FurStrand.vert/.frag (std140).");

        // Mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO (std140, 64 bytes -- F1,
        // "Lumen Lite": one (origin, spacing) pair per renderer::WorldProbeGridPass::kLevelCount
        // clipmap level) -- identical to VegetationScatterPass's own copy.
        struct WorldProbeGridParamsUBO {
            struct Level {
                float originX = 0.0f, originY = 0.0f, originZ = 0.0f;
                float spacing = 0.0f;
            };
            Level levels[3];
            float gridResolution = 0.0f;
            float _pad0 = 0.0f, _pad1 = 0.0f, _pad2 = 0.0f;
        };
        static_assert(sizeof(WorldProbeGridParamsUBO) == 64, "WorldProbeGridParamsUBO must match world_probe_sampling.glsl (std140).");

    } // namespace

    bool FurStrandPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const VirtualShadowMapPass& vsm, const WorldProbeGridPass& worldProbes,
        VkImageView hzbView, VkExtent2D hzbMip0Extent, uint32_t hzbMipCount,
        VkBuffer boneMatricesBuffer, VkBuffer entityTransformBuffer,
        const CreatureFurGeometry& creatureGeom,
        VkFormat colorFormat, VkFormat depthFormat) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_CommandPool = commandPool;
        m_Queue = queue;
        m_HZBMip0Extent = hzbMip0Extent;
        m_HZBMipCount = hzbMipCount;
        m_BoneMatricesBuffer = boneMatricesBuffer;
        m_EntityTransformBuffer = entityTransformBuffer;
        m_CreatureGeom = creatureGeom;

        // =====================================================================================
        // STEP 1 -- Buffer allocation. Every buffer is GPU_ONLY (host-touched only via one-shot
        // staged compute at bake time / vkCmdUpdateBuffer per-frame), except the small mapped
        // count-readback buffer -- same convention as VegetationScatterPass.
        // =====================================================================================
        m_StrandRootBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxStrands) * sizeof(GpuFurStrandRoot),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_CounterBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        m_VisibleIndexBuffer.Create(allocator, static_cast<VkDeviceSize>(kMaxStrands) * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_IndirectCommandBuffer.Create(allocator, sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        m_CullParamsBuffer.Create(allocator, sizeof(FurCullParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_RenderParamsBuffer.Create(allocator, sizeof(FurRenderParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_WorldProbeGridParamsBuffer.Create(allocator, sizeof(WorldProbeGridParamsUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_CountReadbackBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        // Initialize the single indirect command once (instanceCount 0) so a frame that never culls
        // still issues a well-defined zero-instance draw rather than reading garbage.
        {
            VkDrawIndirectCommand initialCmd{ kFurSegments * 6u, 0u, 0u, 0u };
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdUpdateBuffer(cmd, m_IndirectCommandBuffer.Handle(), 0, sizeof(initialCmd), &initialCmd);
            });
        }

        // =====================================================================================
        // STEP 2 -- Descriptor set layouts + one shared pool + every set.
        // =====================================================================================
        auto makeLayout = [&](std::vector<VkDescriptorSetLayoutBinding> bindings) {
            VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            info.bindingCount = static_cast<uint32_t>(bindings.size());
            info.pBindings = bindings.data();
            VkDescriptorSetLayout layout = VK_NULL_HANDLE;
            VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &info, nullptr, &layout));
            return layout;
        };

        m_GenSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr } });
        m_CullSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // strand roots
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // visible indices
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // indirect command
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // HZB
            { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // cull params
            { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // counter
            { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },          // bone matrices
            { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr } });       // entity transforms
        m_RenderInstanceSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },           // strand roots
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },           // visible indices
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr },           // bone matrices
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr } });        // entity transforms
        m_RenderParamsSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr } });
        m_LightingSetLayout = makeLayout({
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },          // shadow page table
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },          // shadow feedback
            { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },  // shadow atlas
            { 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },          // shadow sun levels
            { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, WorldProbeGridPass::kLevelCount, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },  // world probe grid (F1: multi-level)
            { 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr } });       // world probe grid params

        VkDescriptorPoolSize poolSizes[3] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
            // F1 ("Lumen Lite"): +2 vs. pre-F1 (was 4: HZB + shadow atlas + 1 world probe grid sampler,
            // now WorldProbeGridPass::kLevelCount=3 world probe grid samplers).
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 + WorldProbeGridPass::kLevelCount },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 6 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 5;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        auto allocSet = [&](VkDescriptorSetLayout layout) {
            VkDescriptorSetAllocateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            info.descriptorPool = m_DescriptorPool;
            info.descriptorSetCount = 1;
            info.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &info, &set));
            return set;
        };
        m_GenSet = allocSet(m_GenSetLayout);
        m_CullSet = allocSet(m_CullSetLayout);
        m_RenderInstanceSet = allocSet(m_RenderInstanceSetLayout);
        m_RenderParamsSet = allocSet(m_RenderParamsSetLayout);
        m_LightingSet = allocSet(m_LightingSetLayout);

        // HZB sampler: nearest / nearest-mip across the whole pyramid (see hzb_occlusion.glsl).
        m_HZBSampler = VulkanUtils::CreateNearestSampler(m_Device, static_cast<float>(m_HZBMipCount > 0 ? m_HZBMipCount - 1u : 0u));

        // --- Descriptor writes (every binding here is stable for this pass' lifetime). ---
        VkDescriptorBufferInfo rootInfo{ m_StrandRootBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo counterInfo{ m_CounterBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo visibleInfo{ m_VisibleIndexBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo indirectInfo{ m_IndirectCommandBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo cullParamsInfo{ m_CullParamsBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo renderParamsInfo{ m_RenderParamsBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo boneInfo{ m_BoneMatricesBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo transformInfo{ m_EntityTransformBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo hzbInfo{ m_HZBSampler, hzbView, VK_IMAGE_LAYOUT_GENERAL };

        VkDescriptorBufferInfo pageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo feedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo atlasInfo{ vsm.GetPhysicalAtlasSampler(), vsm.GetPhysicalAtlasView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo sunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };
        // F1 ("Lumen Lite"): one VkDescriptorImageInfo per renderer::WorldProbeGridPass::kLevelCount
        // clipmap level.
        VkDescriptorImageInfo probeGridInfos[WorldProbeGridPass::kLevelCount]{};
        for (uint32_t level = 0; level < WorldProbeGridPass::kLevelCount; ++level) {
            probeGridInfos[level] = { worldProbes.GetGridSampler(), worldProbes.GetGridView(level), VK_IMAGE_LAYOUT_GENERAL };
        }
        VkDescriptorBufferInfo gridParamsInfo{ m_WorldProbeGridParamsBuffer.Handle(), 0, VK_WHOLE_SIZE };

        auto bufWrite = [](VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const VkDescriptorBufferInfo* bi) {
            return VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, type, nullptr, bi, nullptr };
        };
        auto imgWrite = [](VkDescriptorSet set, uint32_t binding, const VkDescriptorImageInfo* ii) {
            return VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ii, nullptr, nullptr };
        };
        auto imgArrayWrite = [](VkDescriptorSet set, uint32_t binding, uint32_t count, const VkDescriptorImageInfo* ii) {
            return VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, count, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, ii, nullptr, nullptr };
        };

        std::vector<VkWriteDescriptorSet> writes = {
            bufWrite(m_GenSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &rootInfo),
            bufWrite(m_GenSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &counterInfo),

            bufWrite(m_CullSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &rootInfo),
            bufWrite(m_CullSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &visibleInfo),
            bufWrite(m_CullSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &indirectInfo),
            imgWrite(m_CullSet, 3, &hzbInfo),
            bufWrite(m_CullSet, 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &cullParamsInfo),
            bufWrite(m_CullSet, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &counterInfo),
            bufWrite(m_CullSet, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &boneInfo),
            bufWrite(m_CullSet, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &transformInfo),

            bufWrite(m_RenderInstanceSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &rootInfo),
            bufWrite(m_RenderInstanceSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &visibleInfo),
            bufWrite(m_RenderInstanceSet, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &boneInfo),
            bufWrite(m_RenderInstanceSet, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &transformInfo),

            bufWrite(m_RenderParamsSet, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &renderParamsInfo),

            bufWrite(m_LightingSet, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &pageTableInfo),
            bufWrite(m_LightingSet, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &feedbackInfo),
            imgWrite(m_LightingSet, 2, &atlasInfo),
            bufWrite(m_LightingSet, 3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &sunLevelsInfo),
            imgArrayWrite(m_LightingSet, 4, WorldProbeGridPass::kLevelCount, probeGridInfos),
            bufWrite(m_LightingSet, 5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &gridParamsInfo),
        };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // One-time World Probe grid params upload (gridOrigin unused by SampleWorldProbeGrid's own
        // absolute-index addressing -- same static-upload simplification VegetationScatterPass uses).
        {
            WorldProbeGridParamsUBO gridParams{};
            for (uint32_t level = 0; level < WorldProbeGridPass::kLevelCount; ++level) {
                const maths::vec3& origin = worldProbes.GetGridOriginWorld(level);
                gridParams.levels[level].originX = origin.x;
                gridParams.levels[level].originY = origin.y;
                gridParams.levels[level].originZ = origin.z;
                gridParams.levels[level].spacing = WorldProbeGridPass::GetLevelSpacing(level);
            }
            gridParams.gridResolution = static_cast<float>(WorldProbeGridPass::kGridResolution);
            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                vkCmdUpdateBuffer(cmd, m_WorldProbeGridParamsBuffer.Handle(), 0, sizeof(gridParams), &gridParams);
            });
        }

        // =====================================================================================
        // STEP 3 -- Pipelines.
        // =====================================================================================
        // Strand-root generator (compute): set 0 = root/counter SSBOs + a 48-byte push range.
        {
            VkPushConstantRange push{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FurStrandGenParams) };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 1;
            li.pSetLayouts = &m_GenSetLayout;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &push;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &li, nullptr, &m_GenPipelineLayout));
            VkShaderModule module = VulkanPipeline::LoadShaderModule(m_Device, "shaders/FurStrandGen.comp.spv");
            m_GenPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_GenPipelineLayout, module);
            vkDestroyShaderModule(m_Device, module, nullptr);
        }
        // Per-strand cull (compute).
        {
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 1;
            li.pSetLayouts = &m_CullSetLayout;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &li, nullptr, &m_CullPipelineLayout));
            VkShaderModule module = VulkanPipeline::LoadShaderModule(m_Device, "shaders/FurStrandCull.comp.spv");
            m_CullPipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_CullPipelineLayout, module);
            vkDestroyShaderModule(m_Device, module, nullptr);
        }
        // Instanced forward render pipeline (+ Debug wireframe variant).
        {
            VkDescriptorSetLayout setLayouts[3] = { m_RenderInstanceSetLayout, m_RenderParamsSetLayout, m_LightingSetLayout };
            VkPipelineLayoutCreateInfo li{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            li.setLayoutCount = 3;
            li.pSetLayouts = setLayouts;
            VK_CHECK(vkCreatePipelineLayout(m_Device, &li, nullptr, &m_RenderPipelineLayout));

            VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/FurStrand.vert.spv");
            VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/FurStrand.frag.spv");
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertModule, "main", nullptr };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragModule, "main", nullptr };

            // No bound vertex attribute buffer -- the strand ribbon is synthesized from gl_VertexIndex.
            VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
            VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            // Double-sided: a camera-facing strand ribbon is viewed from both faces as the camera
            // orbits, so backface culling would blink strands out at grazing angles.
            rasterizer.cullMode = VK_CULL_MODE_NONE;
            rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

            VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            // Opaque: no blend (thin depth-writing ribbons, exactly like the grass archetype).
            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;
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

            // Opaque, depth-tested (reversed-Z) AND depth-writing -- mirrors VegetationScatterPass.
            VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER;

            VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            pipelineInfo.pNext = &pipelineRendering;
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
            pipelineInfo.layout = m_RenderPipelineLayout;
            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_RenderPipeline));

#ifndef NDEBUG
            // Debug-only wireframe: same pipeline, VK_POLYGON_MODE_LINE, depth-write OFF so it overlays
            // cleanly without polluting depth for later forward passes.
            VkPipelineRasterizationStateCreateInfo wireRaster = rasterizer;
            wireRaster.polygonMode = VK_POLYGON_MODE_LINE;
            VkPipelineDepthStencilStateCreateInfo wireDepth = depthStencil;
            wireDepth.depthWriteEnable = VK_FALSE;
            VkGraphicsPipelineCreateInfo wirePipelineInfo = pipelineInfo;
            wirePipelineInfo.pRasterizationState = &wireRaster;
            wirePipelineInfo.pDepthStencilState = &wireDepth;
            VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &wirePipelineInfo, nullptr, &m_WireframePipeline));
#endif

            vkDestroyShaderModule(m_Device, vertModule, nullptr);
            vkDestroyShaderModule(m_Device, fragModule, nullptr);
        }

        // =====================================================================================
        // STEP 4 -- Generate the initial strand roots.
        // =====================================================================================
        GenerateStrands();

        LOG_INFO(std::format("[FurStrandPass] Initialized: {} strands on the creature (cap {}), {} verts/strand.",
            m_StrandCount, kMaxStrands, kFurSegments * 6u));
        return true;
    }

    void FurStrandPass::GenerateStrands() {
        FurStrandGenParams params{};
        params.worldOffsetX = m_CreatureGeom.worldOffset.x;
        params.worldOffsetY = m_CreatureGeom.worldOffset.y;
        params.worldOffsetZ = m_CreatureGeom.worldOffset.z;
        params.radiusMin = m_CreatureGeom.radiusMin;
        params.radiusMax = m_CreatureGeom.radiusMax;
        params.segmentLength = m_CreatureGeom.segmentLength;
        params.boneCount = m_CreatureGeom.boneCount;
        params.requestedStrands = std::min(config::fur::STRAND_COUNT, kMaxStrands);
        params.lengthJitter = config::fur::LENGTH_JITTER;
        params.rootLift = config::fur::ROOT_LIFT;
        params.seed = config::fur::SEED;

        VulkanUtils::ExecuteOneShotCommands(m_Device, m_CommandPool, m_Queue, [&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, m_CounterBuffer.Handle(), 0, sizeof(uint32_t), 0u);
            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GenPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GenPipelineLayout, 0, 1, &m_GenSet, 0, nullptr);
            vkCmdPushConstants(cmd, m_GenPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
            vkCmdDispatch(cmd, (params.requestedStrands + 63u) / 64u, 1, 1);

            VulkanUtils::RecordMemoryBarrier(cmd,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            VkBufferCopy copy{ 0, 0, sizeof(uint32_t) };
            vkCmdCopyBuffer(cmd, m_CounterBuffer.Handle(), m_CountReadbackBuffer.Handle(), 1, &copy);
        });

        uint32_t count = 0;
        if (m_CountReadbackBuffer.MappedData() != nullptr) {
            std::memcpy(&count, m_CountReadbackBuffer.MappedData(), sizeof(uint32_t));
        }
        m_StrandCount = std::min(count, kMaxStrands);
        LOG_INFO(std::format("[FurStrandPass] Strands generated: {} (requested {}, length {:.3f}m, jitter {:.2f}).",
            m_StrandCount, params.requestedStrands, config::fur::LENGTH, config::fur::LENGTH_JITTER));
    }

    void FurStrandPass::RecordCull(VkCommandBuffer cmd, const maths::mat4& viewProj,
        const maths::vec3& cameraPositionWorld, bool occlusionCullEnabled) {
        if (m_StrandCount == 0) {
            return;
        }

        // --- Upload this frame's cull parameters. ---
        FurCullParamsUBO ubo{};
        std::array<std::array<float, 4>, 6> planes = ExtractFrustumPlanes(viewProj);
        for (uint32_t i = 0; i < 6; ++i) {
            for (uint32_t c = 0; c < 4; ++c) {
                ubo.frustumPlanes[i * 4u + c] = planes[i][c];
            }
        }
        ubo.cameraPos[0] = cameraPositionWorld.x;
        ubo.cameraPos[1] = cameraPositionWorld.y;
        ubo.cameraPos[2] = cameraPositionWorld.z;
        ubo.cameraPos[3] = 0.0f;
        ubo.viewProj = viewProj;
        ubo.hzbMip0Size[0] = static_cast<float>(m_HZBMip0Extent.width);
        ubo.hzbMip0Size[1] = static_cast<float>(m_HZBMip0Extent.height);
        ubo.hzbMipCount = static_cast<float>(m_HZBMipCount);
        ubo.enableOcclusion = occlusionCullEnabled ? 1u : 0u;
        ubo.creatureMeshID = m_CreatureGeom.creatureMeshID;
        ubo.maxStrands = kMaxStrands;
        // Conservative root-centered bounding radius: full length + the gravity/curl droop it can add
        // + the ribbon half-width margin. The cull shader further scales this by each strand's own
        // lengthScale, so this uses the base length here.
        ubo.boundingRadius = config::fur::LENGTH * (1.0f + config::fur::CURL_AMOUNT) + config::fur::WIDTH + 0.02f;
        vkCmdUpdateBuffer(cmd, m_CullParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);

        // Reset the single indirect command's instanceCount to 0 (fixed fields re-stamped too).
        VkDrawIndirectCommand resetCmd{ kFurSegments * 6u, 0u, 0u, 0u };
        vkCmdUpdateBuffer(cmd, m_IndirectCommandBuffer.Handle(), 0, sizeof(resetCmd), &resetCmd);

        // Make the UBO/indirect updates AND this frame's freshly-rebuilt HZB (compute imageStore
        // earlier in this same command buffer) AND the bone matrices (SkeletalAnimator::RecordUpdate,
        // whose own trailing barrier already extended visibility to COMPUTE) visible to the cull
        // dispatch's uniform/storage/sampled reads.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipelineLayout, 0, 1, &m_CullSet, 0, nullptr);
        vkCmdDispatch(cmd, (m_StrandCount + 63u) / 64u, 1, 1);

        // Cull output -> indirect draw + vertex-stage instance/vertex reads.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }

    void FurStrandPass::RecordDraw(VkCommandBuffer cmd, VkImage colorImage, VkImageView colorView,
        VkImage depthImage, VkImageView depthView, VkExtent2D renderExtent,
        const maths::mat4& viewProj, const maths::vec3& cameraPositionWorld,
        const maths::vec3& sunDirectionWorld, const maths::vec3& sunColor, float sunIntensity,
        float globalTimeSeconds, bool debugWireframe) {
        if (m_StrandCount == 0) {
            return;
        }

        // --- Per-frame render params. ---
        FurRenderParamsUBO ubo{};
        ubo.viewProj = viewProj;
        ubo.cameraPos[0] = cameraPositionWorld.x; ubo.cameraPos[1] = cameraPositionWorld.y; ubo.cameraPos[2] = cameraPositionWorld.z;
        ubo.furLength = config::fur::LENGTH;
        ubo.sunDirection[0] = sunDirectionWorld.x; ubo.sunDirection[1] = sunDirectionWorld.y; ubo.sunDirection[2] = sunDirectionWorld.z;
        ubo.sunIntensity = sunIntensity;
        ubo.sunColor[0] = sunColor.x; ubo.sunColor[1] = sunColor.y; ubo.sunColor[2] = sunColor.z;
        ubo.furWidth = config::fur::WIDTH;
        ubo.rootColor[0] = kRootColor[0]; ubo.rootColor[1] = kRootColor[1]; ubo.rootColor[2] = kRootColor[2];
        ubo.rootDarken = kRootDarken;
        ubo.tipColor[0] = kTipColor[0]; ubo.tipColor[1] = kTipColor[1]; ubo.tipColor[2] = kTipColor[2];
        ubo.curlAmount = config::fur::CURL_AMOUNT;
        ubo.shiftR = kShiftR;
        ubo.shiftTRT = kShiftTRT;
        ubo.exponentR = kExponentR;
        ubo.exponentTRT = kExponentTRT;
        ubo.specIntensity = config::fur::SPEC_INTENSITY;
        ubo.trtIntensity = config::fur::TRT_INTENSITY;
        ubo.globalTime = globalTimeSeconds;
        ubo.creatureMeshID = m_CreatureGeom.creatureMeshID;
        vkCmdUpdateBuffer(cmd, m_RenderParamsBuffer.Handle(), 0, sizeof(ubo), &ubo);
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        // --- Transition color (GENERAL -> COLOR_ATTACHMENT) and depth (READ_ONLY -> ATTACHMENT, test
        // AND write) -- exact mirror of VegetationScatterPass::RecordDraw's own entry barriers. ---
        VkImageMemoryBarrier2 toAttachment[2]{};
        toAttachment[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachment[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toAttachment[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toAttachment[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toAttachment[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toAttachment[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[0].image = colorImage;
        toAttachment[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        toAttachment[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toAttachment[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        toAttachment[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        toAttachment[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        toAttachment[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        toAttachment[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        toAttachment[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        toAttachment[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toAttachment[1].image = depthImage;
        toAttachment[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        VkDependencyInfo toAttachmentDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDep.imageMemoryBarrierCount = 2;
        toAttachmentDep.pImageMemoryBarriers = toAttachment;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDep);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, renderExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        VkPipeline pipeline = m_RenderPipeline;
#ifndef NDEBUG
        if (debugWireframe && m_WireframePipeline != VK_NULL_HANDLE) {
            pipeline = m_WireframePipeline;
        }
#else
        (void)debugWireframe;
#endif
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDescriptorSet sets[3] = { m_RenderInstanceSet, m_RenderParamsSet, m_LightingSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_RenderPipelineLayout, 0, 3, sets, 0, nullptr);

        VkViewport viewport{};
        viewport.width = static_cast<float>(renderExtent.width);
        viewport.height = static_cast<float>(renderExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.extent = renderExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Single non-indexed instanced indirect draw: the GPU cull wrote the survivor instanceCount.
        vkCmdDrawIndirect(cmd, m_IndirectCommandBuffer.Handle(), 0, 1, sizeof(VkDrawIndirectCommand));

        vkCmdEndRendering(cmd);

        // --- Restore color -> GENERAL and depth -> READ_ONLY for the subsequent forward passes'
        // depth test + the next frame's HZB rebuild -- exact mirror of VegetationScatterPass' restore. ---
        VkImageMemoryBarrier2 restore[2]{};
        restore[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restore[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        restore[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        restore[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        restore[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        restore[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        restore[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        restore[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[0].image = colorImage;
        restore[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        restore[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        restore[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        restore[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        restore[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        restore[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        restore[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        restore[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        restore[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restore[1].image = depthImage;
        restore[1].subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

        VkDependencyInfo restoreDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        restoreDep.imageMemoryBarrierCount = 2;
        restoreDep.pImageMemoryBarriers = restore;
        vkCmdPipelineBarrier2(cmd, &restoreDep);
    }

    void FurStrandPass::Shutdown() {
        if (m_Device == VK_NULL_HANDLE) {
            return;
        }

        if (m_RenderPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_RenderPipeline, nullptr);
#ifndef NDEBUG
        if (m_WireframePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_WireframePipeline, nullptr);
        m_WireframePipeline = VK_NULL_HANDLE;
#endif
        if (m_RenderPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_RenderPipelineLayout, nullptr);
        if (m_CullPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_CullPipeline, nullptr);
        if (m_CullPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_CullPipelineLayout, nullptr);
        if (m_GenPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_GenPipeline, nullptr);
        if (m_GenPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_GenPipelineLayout, nullptr);

        if (m_HZBSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_HZBSampler, nullptr);

        if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
        if (m_GenSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_GenSetLayout, nullptr);
        if (m_CullSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_CullSetLayout, nullptr);
        if (m_RenderInstanceSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RenderInstanceSetLayout, nullptr);
        if (m_RenderParamsSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_RenderParamsSetLayout, nullptr);
        if (m_LightingSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_LightingSetLayout, nullptr);

        m_StrandRootBuffer.Destroy();
        m_CounterBuffer.Destroy();
        m_VisibleIndexBuffer.Destroy();
        m_IndirectCommandBuffer.Destroy();
        m_CullParamsBuffer.Destroy();
        m_RenderParamsBuffer.Destroy();
        m_WorldProbeGridParamsBuffer.Destroy();
        m_CountReadbackBuffer.Destroy();

        m_RenderPipeline = VK_NULL_HANDLE;
        m_RenderPipelineLayout = VK_NULL_HANDLE;
        m_CullPipeline = VK_NULL_HANDLE;
        m_CullPipelineLayout = VK_NULL_HANDLE;
        m_GenPipeline = VK_NULL_HANDLE;
        m_GenPipelineLayout = VK_NULL_HANDLE;
        m_HZBSampler = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_GenSetLayout = VK_NULL_HANDLE;
        m_CullSetLayout = VK_NULL_HANDLE;
        m_RenderInstanceSetLayout = VK_NULL_HANDLE;
        m_RenderParamsSetLayout = VK_NULL_HANDLE;
        m_LightingSetLayout = VK_NULL_HANDLE;
        m_GenSet = VK_NULL_HANDLE;
        m_CullSet = VK_NULL_HANDLE;
        m_RenderInstanceSet = VK_NULL_HANDLE;
        m_RenderParamsSet = VK_NULL_HANDLE;
        m_LightingSet = VK_NULL_HANDLE;
        m_BoneMatricesBuffer = VK_NULL_HANDLE;
        m_EntityTransformBuffer = VK_NULL_HANDLE;
        m_StrandCount = 0;
        m_Device = VK_NULL_HANDLE;
    }

}
