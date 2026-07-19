#include "renderer/passes/VirtualShadowMapPass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <format>
#include <limits>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h"
#include "geometry/EntityMaterialTable.h" // geometry::GetEntityMaterialProperties -- Feature 1's per-entity CPU precompute.
#include "io/CacheFileManager.h"
#include "renderer/vulkan/VulkanPipeline.h"

namespace renderer {

    namespace {

        // Push constant for ShadowMapCaptureAnimated.vert (Feature 1/3, VSM advanced roadmap) --
        // NOT the old ShadowCaptureConstants ShadowMapCapture.vert's own dead-but-kept ShadowMapPass
        // still uses (unrelated struct, unrelated file, deliberately untouched -- see this pass' own
        // class comment). `entityID` indexes EntityDataBuffer/EntityTransformBuffer; `maxWPOAmplitude`
        // is this entity's TRUE (non-inflated) authored sway amplitude; `maskTextureIndex` selects
        // this entity's cutout mask slot (or geometry::kInvalidMaskTextureIndex for an unmasked
        // entity, which never actually reaches ShadowMapCaptureMasked.frag -- see RenderPage()'s own
        // per-entity pipeline selection). 76 bytes, comfortably under the 128-byte guaranteed minimum
        // push-constant size.
        struct ShadowCaptureConstants {
            maths::mat4 lightViewProj;
            uint32_t entityID;
            float maxWPOAmplitude;
            uint32_t maskTextureIndex;
        };
        static_assert(sizeof(ShadowCaptureConstants) == 76,
            "ShadowCaptureConstants must match ShadowMapCaptureAnimated.vert's push_constant block exactly");

        // VSM advanced roadmap, Feature 2 (real static-vs-dynamic page invalidation): fresh,
        // self-contained CPU-side copies of spline_deformation.glsl's SPLINE_MAX_DEVIATION /
        // enhanced_displacement.glsl's ENHANCED_DISPLACEMENT_MAX_AMPLITUDE -- this codebase's own
        // established convention for a GLSL #define with no C++ counterpart (see e.g.
        // VirtualShadowMapPool.h's own comment on why a fresh copy beats an awkward cross-language
        // include for a small, stable constant). Used ONLY by EntityAABBOverlapsPageNDC's own
        // conservative AABB inflation below -- must stay numerically equal to the GLSL source of
        // truth (see each constant's own GLSL header comment for the full derivation of why THAT
        // value is provably a safe upper bound).
        constexpr float kCpuSplineMaxDeviation = 1.6f;
        constexpr float kCpuEnhancedDisplacementMaxAmplitude = 0.06f;
        // Skeletal-animation feature (VSM shadow-capture fix): fresh CPU-side copy of
        // skeletal_animation.glsl's SKELETAL_MAX_DEVIATION -- same convention/rationale as the two
        // constants above (also mirrored independently in animation::SkeletalAnimator.cpp's own
        // ValidateSkeletalBounds diagnostic, kSkeletalMaxDeviationMirror -- three independent copies
        // of the same GLSL source-of-truth constant, all documented, matching this file's own
        // established "no cross-language include for a small stable constant" idiom).
        constexpr float kCpuSkeletalMaxDeviation = 1.5f;

        constexpr float kHalfPi = 1.57079632679489661923f;

        // Standard cubemap face directions/up-vectors (Vulkan/D3D convention: +X,-X,+Y,-Y,+Z,-Z),
        // consumed identically by shadow_point_sampling.glsl's ShadowCubeFaceIndex so a world
        // position's selected face here always matches the face a shadow lookup samples.
        constexpr maths::vec3 kFaceDirs[6] = {
            {  1.0f,  0.0f,  0.0f }, { -1.0f,  0.0f,  0.0f },
            {  0.0f,  1.0f,  0.0f }, {  0.0f, -1.0f,  0.0f },
            {  0.0f,  0.0f,  1.0f }, {  0.0f,  0.0f, -1.0f },
        };
        constexpr maths::vec3 kFaceUps[6] = {
            { 0.0f, -1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
            { 0.0f,  0.0f, 1.0f }, { 0.0f,  0.0f, -1.0f },
            { 0.0f, -1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
        };

    } // namespace

    bool VirtualShadowMapPass::Init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
        VkCommandPool commandPool, VkQueue queue, const std::filesystem::path& cacheFilePath,
        VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer wpoGlobalsBuffer,
        VkBuffer splineControlPointsBuffer, VkBuffer boneMatricesBuffer, const core::EntityData* entityDataCPU,
        const std::vector<VkDescriptorImageInfo>& maskImageInfos) {
        kSunBaseRadius = config::lumen::VSM_SUN_BASE_RADIUS;
        kPhysicalPageCapacity = config::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
        kMaxPagesRenderedPerFrame = config::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
        kMaxDynamicPagesRenderedPerFrame = config::lumen::VSM_MAX_DYNAMIC_PAGES_RENDERED_PER_FRAME;

        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Read every fallback mesh's geometry + derive the scene's bounding sphere.
        // Byte-for-byte the same procedure as ShadowMapPass::Init's own STEP 1 (own, separate
        // buffers -- see this pass' class comment on why a self-contained copy, not a shared read).
        // =====================================================================================
        geometry::CacheFileManager cacheManager;
        geometry::CacheFileHeader header{};
        if (!cacheManager.ReadHeader(cacheFilePath, header)) {
            LOG_ERROR(std::format("[VirtualShadowMapPass] Failed to read cache header '{}'.", cacheFilePath.string()));
            return false;
        }

        std::vector<geometry::FallbackMeshIndexEntry> fallbackTable;
        if (!cacheManager.ReadFallbackMeshTable(cacheFilePath, header, fallbackTable)) {
            LOG_ERROR("[VirtualShadowMapPass] Failed to read the fallback-mesh table.");
            return false;
        }

        maths::vec3 sceneBoundsMin{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        maths::vec3 sceneBoundsMax{ -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };

        std::vector<geometry::FallbackVertex> hostVertices;
        std::vector<uint32_t> hostIndices;
        m_EntityRanges.reserve(fallbackTable.size());
        for (const geometry::FallbackMeshIndexEntry& entry : fallbackTable) {
            std::vector<geometry::FallbackVertex> entityVertices;
            std::vector<uint32_t> entityIndices;
            if (!cacheManager.ReadFallbackMeshGeometry(cacheFilePath, entry, entityVertices, entityIndices)) {
                LOG_ERROR(std::format(
                    "[VirtualShadowMapPass] Failed to read fallback mesh geometry for entityID={}.", entry.entityID));
                return false;
            }

            // Feature 1 (live per-entity transforms): this entity's own span inside the combined
            // vertex/index buffers -- mirrors SurfaceCachePass::EntityDrawRange's own population
            // exactly (see that class's .cpp, same STEP 1 shape).
            const uint32_t indexBase = static_cast<uint32_t>(hostVertices.size());
            EntityDrawRange range{};
            range.vertexOffset = static_cast<int32_t>(indexBase);
            range.firstIndex = static_cast<uint32_t>(hostIndices.size());
            range.indexCount = static_cast<uint32_t>(entityIndices.size());
            range.boundsMin = maths::vec3{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] };
            range.boundsMax = maths::vec3{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] };

            // Feature 1/2/3: per-entity material properties + flags, read ONCE here (never change
            // after VulkanContext::BuildEntityData() ran) -- see EntityDrawRange's own field
            // comments for why no un-inflation step or per-frame lookup is needed downstream.
            if (entityDataCPU != nullptr) {
                const core::EntityData& ed = entityDataCPU[entry.entityID];
                geometry::EntityMaterialProperties materialProps = geometry::GetEntityMaterialProperties(ed.materialID);
                range.maxWPOAmplitude = materialProps.maxWPOAmplitude;
                range.maskTextureIndex = materialProps.maskTextureIndex;
                range.hasSplineDeformation = core::GetFlag(ed.flags, core::EntityFlags::HasSplineDeformation);
                range.hasEnhancedDisplacement = core::GetFlag(ed.flags, core::EntityFlags::HasEnhancedDisplacement);
                // Skeletal-animation feature (VSM shadow-capture fix): this flag was never folded
                // into isDynamicCandidate when the skeletal-animation feature was introduced (a real
                // integration gap between two previously-separate features) -- without it, a page
                // that had already captured the creature once (e.g. at first visibility) would never
                // be classified as covering dynamic content again, so it would never be re-rendered,
                // and ShadowMapCaptureAnimated.vert's own skinning fix (this file's sibling change)
                // would never actually be observed: the page's cached depth content would stay
                // frozen at whichever single pose it was last captured at, forever.
                range.isSkeletallyAnimated = core::GetFlag(ed.flags, core::EntityFlags::IsSkeletallyAnimated);
                range.isDynamicCandidate = core::GetFlag(ed.flags, core::EntityFlags::IsDynamic) ||
                    range.hasSplineDeformation || range.hasEnhancedDisplacement || range.isSkeletallyAnimated;
            }
            m_EntityRanges[entry.entityID] = range;

            hostVertices.insert(hostVertices.end(), entityVertices.begin(), entityVertices.end());
            for (uint32_t index : entityIndices) {
                hostIndices.push_back(index + indexBase);
            }

            const maths::vec3 entityMin{ entry.boundsMin[0], entry.boundsMin[1], entry.boundsMin[2] };
            const maths::vec3 entityMax{ entry.boundsMax[0], entry.boundsMax[1], entry.boundsMax[2] };
            sceneBoundsMin.x = std::min(sceneBoundsMin.x, entityMin.x);
            sceneBoundsMin.y = std::min(sceneBoundsMin.y, entityMin.y);
            sceneBoundsMin.z = std::min(sceneBoundsMin.z, entityMin.z);
            sceneBoundsMax.x = std::max(sceneBoundsMax.x, entityMax.x);
            sceneBoundsMax.y = std::max(sceneBoundsMax.y, entityMax.y);
            sceneBoundsMax.z = std::max(sceneBoundsMax.z, entityMax.z);
        }

        m_TotalIndexCount = static_cast<uint32_t>(hostIndices.size());

        if (fallbackTable.empty()) {
            m_SceneBoundsCenter = maths::vec3{ 0.0f, 0.0f, 0.0f };
            m_SceneBoundsRadius = 0.0f;
        } else {
            m_SceneBoundsCenter = (sceneBoundsMin + sceneBoundsMax) * 0.5f;
            m_SceneBoundsRadius = (sceneBoundsMax - sceneBoundsMin).Length() * 0.5f;
        }

        // =====================================================================================
        // STEP 2 -- Combined vertex/index GPU buffers (position-only reads, same convention as
        // ShadowMapPass) + one-time upload.
        // =====================================================================================
        VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(hostVertices.size()) * sizeof(geometry::FallbackVertex);
        VkDeviceSize indexBytes = static_cast<VkDeviceSize>(hostIndices.size()) * sizeof(uint32_t);

        if (vertexBytes > 0 && indexBytes > 0) {
            m_VertexBuffer.Create(allocator, vertexBytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_IndexBuffer.Create(allocator, indexBytes,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocResultInfo{};
            VkDeviceSize stagingSize = vertexBytes + indexBytes;
            VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            stagingInfo.size = stagingSize;
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagingAllocInfo{};
            stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VK_CHECK(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));
            std::memcpy(stagingAllocResultInfo.pMappedData, hostVertices.data(), static_cast<size_t>(vertexBytes));
            std::memcpy(static_cast<char*>(stagingAllocResultInfo.pMappedData) + vertexBytes, hostIndices.data(), static_cast<size_t>(indexBytes));

            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            VkCommandBuffer uploadCmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &uploadCmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(uploadCmd, &beginInfo);

            VkBufferCopy vertexCopy{ 0, 0, vertexBytes };
            vkCmdCopyBuffer(uploadCmd, stagingBuffer, m_VertexBuffer.Handle(), 1, &vertexCopy);
            VkBufferCopy indexCopy{ vertexBytes, 0, indexBytes };
            vkCmdCopyBuffer(uploadCmd, stagingBuffer, m_IndexBuffer.Handle(), 1, &indexCopy);

            VkMemoryBarrier2 uploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            uploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            uploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            uploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            uploadBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
            VkDependencyInfo uploadDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            uploadDep.memoryBarrierCount = 1;
            uploadDep.pMemoryBarriers = &uploadBarrier;
            vkCmdPipelineBarrier2(uploadCmd, &uploadDep);

            vkEndCommandBuffer(uploadCmd);
            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &uploadCmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(m_Device, commandPool, 1, &uploadCmd);
            vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        }

        // =====================================================================================
        // STEP 3 -- Physical page pool + feedback buffer (own dedicated instance, see class
        // comment on why not shared with the geometry streaming system's own).
        // =====================================================================================
        if (!m_Pool.Init(physicalDevice, device, allocator, commandPool, queue, kTotalVSMCount, kPhysicalPageCapacity)) {
            LOG_ERROR("[VirtualShadowMapPass] Failed to initialize VirtualShadowMapPool.");
            return false;
        }
        m_Feedback.Init(allocator, kFeedbackCapacity);

        // =====================================================================================
        // STEP 4 -- Sun clipmap: each level's PROJECTION is fixed for this pass' entire lifetime
        // (radius_L never changes -- see class comment on why these windows never re-center); only
        // the VIEW half (sunDirection-dependent) is recomputed every RecordBeginFrame() call.
        // =====================================================================================
        for (uint32_t level = 0; level < kSunLevelCount; ++level) {
            float radius = std::max(kSunBaseRadius * float(1u << level), 1.0e-3f);
            float nearPlane = radius * kSunNearMarginFactor;
            float farPlane = radius * (2.0f + kSunFarMarginFactor);
            m_SunLevelProj[level] = maths::mat4::OrthoVulkan(radius, radius, nearPlane, farPlane);
        }

        // =====================================================================================
        // STEP 5 -- Descriptor set (Feature 1/3, VSM advanced roadmap): 4 vertex-stage bindings
        // (EntityTransformBuffer/EntityDataBuffer/WPOGlobalsUBO/SplineControlPointsSSBO, this
        // pass's own binding numbers 0-3) + 1 fragment-stage binding (the bindless cutout mask
        // array, binding 4) -- mirrors ClusterHardwareRasterPass::Init's own layout pattern (see
        // that class's own comment).
        // =====================================================================================
        uint32_t maskTextureCount = static_cast<uint32_t>(maskImageInfos.size());

        VkDescriptorSetLayoutBinding bindings[6]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // EntityTransformBuffer
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // EntityDataBuffer
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // WPOGlobalsUBO
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // SplineControlPointsSSBO
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[4].binding = 4;
        bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // g_MaskTextures[] (Feature 3, mask_sampling.glsl)
        bindings[4].descriptorCount = maskTextureCount;
        bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        // Skeletal-animation feature (VSM shadow-capture fix): SkeletalBoneMatricesSSBO, read-only,
        // vertex-stage-only -- see ShadowMapCaptureAnimated.vert's own binding comment.
        bindings[5].binding = 5;
        bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[5].descriptorCount = 1;
        bindings[5].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo setLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        setLayoutInfo.bindingCount = 6;
        setLayoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &setLayoutInfo, nullptr, &m_DescriptorSetLayout));

        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount };
        VkDescriptorPoolCreateInfo descPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        descPoolInfo.maxSets = 1;
        descPoolInfo.poolSizeCount = 3;
        descPoolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &descPoolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo descSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        descSetAllocInfo.descriptorPool = m_DescriptorPool;
        descSetAllocInfo.descriptorSetCount = 1;
        descSetAllocInfo.pSetLayouts = &m_DescriptorSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &descSetAllocInfo, &m_DescriptorSet));

        VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo entityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo wpoGlobalsInfo{ wpoGlobalsBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo splineControlPointsInfo{ splineControlPointsBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo boneMatricesInfo{ boneMatricesBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[6]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_DescriptorSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &entityTransformInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_DescriptorSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &entityDataInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = m_DescriptorSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].pBufferInfo = &wpoGlobalsInfo;

        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = m_DescriptorSet;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &splineControlPointsInfo;

        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = m_DescriptorSet;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = maskTextureCount;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].pImageInfo = maskImageInfos.data();

        writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet = m_DescriptorSet;
        writes[5].dstBinding = 5;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[5].pBufferInfo = &boneMatricesInfo;

        vkUpdateDescriptorSets(m_Device, 6, writes, 0, nullptr);

        // =====================================================================================
        // STEP 6 -- Two depth-only capture pipelines sharing the descriptor set + layout above:
        // m_Pipeline (ShadowMapCaptureAnimated.vert only -- unmasked entities, Feature 1) and
        // m_MaskedPipeline (+ ShadowMapCaptureMasked.frag -- masked entities, Feature 3). Mirrors
        // ClusterHardwareRasterPass::Init's own opaque/masked pipeline-split precedent (see that
        // class's own comment), adapted to a depth-only (0 color attachments) render target.
        // =====================================================================================
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(ShadowCaptureConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_DescriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ShadowMapCaptureAnimated.vert.spv");
        VkShaderModule maskedFragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ShadowMapCaptureMasked.frag.spv");

        VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vertModule;
        stage.pName = "main";

        VkPipelineShaderStageCreateInfo maskedFragStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        maskedFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        maskedFragStage.module = maskedFragModule;
        maskedFragStage.pName = "main";

        VkPipelineShaderStageCreateInfo maskedStages[2] = { stage, maskedFragStage };

        // Vertex input: position (Feature 1) + uv (Feature 3, pass-through to the masked frag
        // shader's mask_sampling.glsl lookup) -- FallbackVertex's own `normal` field stays unbound
        // (unused by a depth-only capture), same convention the old single-attribute setup already
        // established for this pipeline.
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(geometry::FallbackVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributes[2]{};
        attributes[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(geometry::FallbackVertex, position) };
        attributes[1] = { 1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(geometry::FallbackVertex, uv) };

        VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = 2;
        vertexInput.pVertexAttributeDescriptions = attributes;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        // No back-face culling -- same self-shadowing rationale as ShadowMapPass's own pipeline.
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.attachmentCount = 0;

        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 0;
        pipelineRendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

        VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.stageCount = 1;
        pipelineInfo.pStages = &stage;
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
        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));

        // Masked pipeline (Feature 3): same state as above, just 2 stages (+ ShadowMapCaptureMasked
        // .frag) instead of 1 -- the frag shader's own `discard` statically disables hardware
        // early-fragment-tests for THIS pipeline only, exactly like ClusterRaster.frag's own
        // documented consequence; m_Pipeline (vertex-only, no discard) stays early-Z-eligible.
        VkGraphicsPipelineCreateInfo maskedPipelineInfo = pipelineInfo;
        maskedPipelineInfo.stageCount = 2;
        maskedPipelineInfo.pStages = maskedStages;
        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &maskedPipelineInfo, nullptr, &m_MaskedPipeline));

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, maskedFragModule, nullptr);

        LOG_INFO(std::format(
            "[VirtualShadowMapPass] Built {} entity draw range(s) (Feature 1, live transforms), "
            "{} mask texture slot(s) bound (Feature 3).", m_EntityRanges.size(), maskTextureCount));

        // =====================================================================================
        // STEP 7 -- VSM matrix UBOs: persistently-mapped CPU_TO_GPU, plain memcpy every
        // RecordBeginFrame() call -- same idiom as renderer::SurfaceCachePass::UpdateLighting's own
        // m_LightingUBO (no descriptor-set update needed after the initial bind, no barrier needed
        // since the host write always happens-before this frame's vkQueueSubmit).
        // =====================================================================================
        m_SunLevelsUBO.Create(allocator, sizeof(m_SunLevelViewProj),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        m_PointFacesUBO.Create(allocator, sizeof(m_PointFaceViewProj),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        LOG_INFO(std::format(
            "[VirtualShadowMapPass] Initialized: {} fallback mesh(es) ({} vertices, {} indices), "
            "{} sun clipmap level(s), up to {} point light(s) x 6 faces, scene bounding sphere radius={:.3f}.",
            fallbackTable.size(), hostVertices.size(), hostIndices.size(), kSunLevelCount, kMaxPointLights, m_SceneBoundsRadius));
        return true;
    }

    void VirtualShadowMapPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_MaskedPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_MaskedPipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_DescriptorSet -- not freed individually,
                // same convention as ClusterHardwareRasterPass::Shutdown.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_DescriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_DescriptorSetLayout, nullptr);
        }
        m_SunLevelsUBO.Destroy();
        m_PointFacesUBO.Destroy();
        m_Feedback.Shutdown();
        m_Pool.Shutdown();
        m_VertexBuffer.Destroy();
        m_IndexBuffer.Destroy();
        m_EntityRanges.clear();

        m_Pipeline = VK_NULL_HANDLE;
        m_MaskedPipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_DescriptorSetLayout = VK_NULL_HANDLE;
        m_TotalIndexCount = 0;
        m_SceneBoundsRadius = 0.0f;
        m_ActivePointLightCount = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void VirtualShadowMapPass::RecordBeginFrame(VkCommandBuffer cmd, const maths::vec3& sunDirection,
        const SceneLights& sceneLights, const maths::vec3& cameraPosition,
        const core::EntityTransformCPU* entityTransformsCPU) {
        (void)cameraPosition; // Unused -- see class comment on why the sun clipmap windows are fixed, not camera-following.

        // --- Recompute this frame's sun clipmap view-projection matrices (view depends on
        // sunDirection; projection was fixed once at Init(), see class comment). ---
        const maths::vec3 lightDir = sunDirection.Normalize();
        maths::vec3 up{ 0.0f, 1.0f, 0.0f };
        if (std::abs(lightDir.Dot(up)) > 0.999f) {
            up = maths::vec3{ 0.0f, 0.0f, 1.0f };
        }
        for (uint32_t level = 0; level < kSunLevelCount; ++level) {
            float radius = std::max(kSunBaseRadius * float(1u << level), 1.0e-3f);
            float nearPlane = radius * kSunNearMarginFactor;
            maths::vec3 eye = m_SceneBoundsCenter - lightDir * (radius + nearPlane);
            maths::mat4 view = maths::mat4::LookAt(eye, m_SceneBoundsCenter, up);
            m_SunLevelViewProj[level] = m_SunLevelProj[level] * view;
        }

        // --- Recompute this frame's point light cube face view-projection matrices. ---
        m_ActivePointLightCount = std::min(sceneLights.pointLightCount, kMaxPointLights);
        for (uint32_t slot = 0; slot < m_ActivePointLightCount; ++slot) {
            const PointLight& light = sceneLights.pointLights[slot];
            float farPlane = std::max(light.radius, 0.1f);
            float nearPlane = std::max(farPlane * 0.01f, 0.02f);
            maths::mat4 proj = maths::mat4::PerspectiveVulkan(kHalfPi, 1.0f, nearPlane, farPlane);
            for (uint32_t face = 0; face < 6; ++face) {
                maths::mat4 view = maths::mat4::LookAt(light.position, light.position + kFaceDirs[face], kFaceUps[face]);
                m_PointFaceViewProj[slot * 6 + face] = proj * view;
            }
        }

        std::memcpy(m_SunLevelsUBO.MappedData(), m_SunLevelViewProj, sizeof(m_SunLevelViewProj));
        std::memcpy(m_PointFacesUBO.MappedData(), m_PointFaceViewProj, sizeof(m_PointFaceViewProj));

        // --- Clear this frame's feedback counter before any consumer can write into it. ---
        m_Feedback.RecordClear(cmd);

        // --- Process LAST frame's page-miss reports (one-frame lag, see class comment): dedup,
        // allocate + render up to kMaxPagesRenderedPerFrame this frame. ---
        // config::lumen::BUILD_SHADOWS temporary kill-switch: skipping this entire block leaves
        // every shadow page permanently non-resident, which shadow_sun_sampling.glsl/
        // shadow_point_sampling.glsl already interpret as "fully lit" -- see EngineConfig.h's own
        // comment on this flag. m_SunLevelsUBO/m_PointFacesUBO above still get this frame's valid
        // view-projection matrices either way, so re-enabling this flag later needs no other change.
        bool renderedAnyPage = false;
        if (config::lumen::BUILD_SHADOWS) {
            std::vector<uint32_t> missedPages = m_Feedback.ReadRequestedClusterIDs();
            // Priority = -(vsmIndex): sun clipmap levels are indexed finest-first (kSunLevelCount
            // levels, see RecordBeginFrame's own radius = kSunBaseRadius * 2^level loop above), so
            // a lower vsmIndex is both a finer sun cascade AND, since point-light cube faces are
            // indexed immediately after every sun level, always prioritized ahead of any point
            // light's shadow pages -- one monotonic key covers both cases without a branch.
            std::vector<float> missedPriorities;
            missedPriorities.reserve(missedPages.size());
            for (uint32_t logicalPageID : missedPages) {
                uint32_t vsmIndex = logicalPageID / kShadowPagesPerVSM;
                missedPriorities.push_back(-float(vsmIndex));
            }
            m_RequestQueue.SubmitFrameRequests(missedPages, missedPriorities);

            for (uint32_t processed = 0; processed < kMaxPagesRenderedPerFrame; ++processed) {
                uint32_t logicalPageID = 0;
                if (!m_RequestQueue.PopNextRequest(logicalPageID)) {
                    break;
                }

                uint32_t vsmIndex = logicalPageID / kShadowPagesPerVSM;
                uint32_t localPageIndex = logicalPageID % kShadowPagesPerVSM;

                const maths::mat4* viewProj = nullptr;
                if (vsmIndex < kSunLevelCount) {
                    viewProj = &m_SunLevelViewProj[vsmIndex];
                } else {
                    uint32_t faceSlot = vsmIndex - kSunLevelCount;
                    if (faceSlot >= m_ActivePointLightCount * 6u) {
                        // Stale report for a now-inactive point light slot -- nothing to render.
                        m_RequestQueue.MarkRequestCompleted(logicalPageID);
                        continue;
                    }
                    viewProj = &m_PointFaceViewProj[faceSlot];
                }

                uint32_t physicalLayer = m_Pool.AllocatePage(cmd, logicalPageID);
                if (physicalLayer == kInvalidShadowPhysicalPage) {
                    m_RequestQueue.MarkRequestCompleted(logicalPageID); // Let a future miss report retry it.
                    continue;
                }

                RenderPage(cmd, vsmIndex, localPageIndex, physicalLayer, *viewProj);
                renderedAnyPage = true;
                m_RequestQueue.MarkRequestCompleted(logicalPageID);
            }
        }

        if (renderedAnyPage) {
            // Makes this call's page depth writes visible to a later fragment/vertex-shader
            // sampled read THIS SAME frame (SurfaceCacheCapture.frag, ClusterResolve.comp/
            // ClusterResolveBinned.comp) -- unlike the feedback buffer's own one-frame lag, a page
            // rendered this frame must be immediately visible, not next frame.
            VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // =====================================================================================
        // VSM advanced roadmap, Feature 2 (real static-vs-dynamic page invalidation): classify
        // EVERY currently-resident page as covering dynamic content or not from THIS frame's
        // actual (rotated) entity transforms, then unconditionally re-render every resident page
        // classified dynamic -- bypassing AllocatePage()/the feedback queue entirely (already
        // resident, no LRU/miss path needed). Own frame budget (kMaxDynamicPagesRenderedPerFrame),
        // mirroring kMaxPagesRenderedPerFrame's own convention above. Gated by the same
        // config::lumen::BUILD_SHADOWS kill-switch as the block above (nothing to reclassify or
        // re-render if shadows are off entirely) and tolerates a null entityTransformsCPU (see
        // this method's own header comment).
        // =====================================================================================
        if (config::lumen::BUILD_SHADOWS && entityTransformsCPU != nullptr) {
            ClassifyDynamicPages(entityTransformsCPU);

            std::vector<uint32_t> dynamicPageIDs = m_Pool.GetResidentDynamicPageIDs();
            bool renderedAnyDynamicPage = false;
            uint32_t dynamicPagesRendered = 0;
            for (uint32_t logicalPageID : dynamicPageIDs) {
                if (dynamicPagesRendered >= kMaxDynamicPagesRenderedPerFrame) {
                    break;
                }

                uint32_t vsmIndex = logicalPageID / kShadowPagesPerVSM;
                uint32_t localPageIndex = logicalPageID % kShadowPagesPerVSM;

                const maths::mat4* viewProj = nullptr;
                if (vsmIndex < kSunLevelCount) {
                    viewProj = &m_SunLevelViewProj[vsmIndex];
                } else {
                    uint32_t faceSlot = vsmIndex - kSunLevelCount;
                    if (faceSlot >= m_ActivePointLightCount * 6u) {
                        continue; // Stale (inactive point light slot) -- defensive, ClassifyDynamicPages already clears this case's own flag.
                    }
                    viewProj = &m_PointFaceViewProj[faceSlot];
                }

                uint32_t physicalLayer = m_Pool.GetPhysicalLayer(logicalPageID);
                if (physicalLayer == kInvalidShadowPhysicalPage) {
                    continue; // Defensive -- GetResidentDynamicPageIDs() only ever returns resident pages.
                }

                RenderPage(cmd, vsmIndex, localPageIndex, physicalLayer, *viewProj);
                // Keeps an actively-redrawn dynamic page from ever looking idle to the LRU (it IS
                // being used every frame, just not via the feedback/miss path) -- cheap (O(1)) and
                // purely defensive, since whatever consumer still samples this page's content would
                // ordinarily touch it too via the normal feedback path.
                m_Pool.TouchPage(logicalPageID);
                renderedAnyDynamicPage = true;
                ++dynamicPagesRendered;
            }

            if (renderedAnyDynamicPage) {
                // Same barrier reasoning as the miss-driven render loop's own trailing barrier
                // above -- a page rendered THIS frame (dynamic or not) must be immediately visible
                // to a later sampled read this same frame, not next frame.
                VkMemoryBarrier2 dynamicBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                dynamicBarrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                dynamicBarrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                dynamicBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                dynamicBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                VkDependencyInfo dynamicDepInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dynamicDepInfo.memoryBarrierCount = 1;
                dynamicDepInfo.pMemoryBarriers = &dynamicBarrier;
                vkCmdPipelineBarrier2(cmd, &dynamicDepInfo);
            }

            // Low-frequency diagnostic (no existing throttled-log precedent elsewhere in this
            // codebase -- a simple static frame counter is the least surprising fit): confirms
            // classification stays selective (a small fraction of resident pages, only those
            // actually touched by a rotating/deformed entity) rather than degenerating into
            // "redraw everything" -- see this phase's own plan verification step.
            static uint32_t s_DynamicPageLogFrameCounter = 0;
            if ((++s_DynamicPageLogFrameCounter % 300u) == 0u) {
                LOG_INFO(std::format(
                    "[VirtualShadowMapPass] Dynamic page classification: {} of {} resident page(s) classified dynamic.",
                    dynamicPageIDs.size(), m_Pool.GetResidentPageCount()));
            }
        }
    }

    void VirtualShadowMapPass::RecordEndFrame(VkCommandBuffer cmd) {
        // Captures this frame's page-miss reports (written by whatever ran between
        // RecordBeginFrame() and this call) for RecordBeginFrame() to consume next frame -- see
        // this class' own header comment for the full one-frame-lag contract.
        m_Feedback.RecordReadback(cmd);
    }

    void VirtualShadowMapPass::RenderPage(VkCommandBuffer cmd, uint32_t vsmIndex, uint32_t localPageIndex,
        uint32_t physicalLayer, const maths::mat4& viewProj) {
        (void)vsmIndex; // Rendering itself is VSM-agnostic -- only `viewProj` (already resolved by the caller) matters.
        uint32_t pageX = localPageIndex % kShadowPagesPerAxis;
        uint32_t pageY = localPageIndex / kShadowPagesPerAxis;

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = m_Pool.GetPhysicalLayerView(physicalLayer);
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, { kShadowPageTexels, kShadowPageTexels } };
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = &depthAttachment;
        vkCmdBeginRendering(cmd, &renderingInfo);

        // Virtual-viewport technique: a viewport LARGER than the 128x128 physical page, offset so
        // only page (pageX,pageY)'s own 128x128 region of the level/face's full 2048x2048 virtual
        // resolution falls inside [0,128)x[0,128) -- Vulkan explicitly permits viewport dimensions
        // exceeding the bound attachment (see VirtualShadowMapPool.h's own comment on why this is
        // guaranteed safe without a capability query on any conformant Vulkan 1.3 device); the
        // scissor below clips the actual rasterized output to the physical page's real bounds.
        constexpr float kVirtualResolution = float(kShadowPagesPerAxis * kShadowPageTexels); // 2048.
        VkViewport viewport{};
        viewport.x = -float(pageX) * float(kShadowPageTexels);
        viewport.y = -float(pageY) * float(kShadowPageTexels);
        viewport.width = kVirtualResolution;
        viewport.height = kVirtualResolution;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{ { 0, 0 }, { kShadowPageTexels, kShadowPageTexels } };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // Feature 1 (live per-entity transforms): one vkCmdDrawIndexed per entity instead of the
        // old single monolithic draw -- still "redraw the whole (tiny) scene per page, clipped by
        // viewport/scissor" (see this pass' own class comment), just per-entity now so each
        // entity's CURRENT (deformed/rotated) geometry actually reaches the page. Pipeline
        // (unmasked vs masked, Feature 3) is selected per entity from its own precomputed
        // maskTextureIndex; redundant rebinds across consecutive same-pipeline entities are
        // skipped, an unordered_map iteration gives no adjacency guarantee but the check is cheap
        // either way.
        if (!m_EntityRanges.empty()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

            VkBuffer vertexBuffer = m_VertexBuffer.Handle();
            VkDeviceSize vertexOffset0 = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset0);
            vkCmdBindIndexBuffer(cmd, m_IndexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

            VkPipeline currentBoundPipeline = VK_NULL_HANDLE;
            for (const auto& [entityID, range] : m_EntityRanges) {
                if (range.indexCount == 0) {
                    continue;
                }

                bool masked = range.maskTextureIndex != 0xFFFFFFFFu; // geometry::kInvalidMaskTextureIndex.
                VkPipeline pipeline = masked ? m_MaskedPipeline : m_Pipeline;
                if (pipeline != currentBoundPipeline) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                    currentBoundPipeline = pipeline;
                }

                ShadowCaptureConstants pc{};
                pc.lightViewProj = viewProj;
                pc.entityID = entityID;
                pc.maxWPOAmplitude = range.maxWPOAmplitude;
                pc.maskTextureIndex = range.maskTextureIndex;
                vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

                vkCmdDrawIndexed(cmd, range.indexCount, 1, range.firstIndex, range.vertexOffset, 0);
            }
        }

        vkCmdEndRendering(cmd);
    }

    void VirtualShadowMapPass::ClassifyDynamicPages(const core::EntityTransformCPU* entityTransformsCPU) {
        std::vector<uint32_t> residentPageIDs = m_Pool.GetResidentPageIDs();
        for (uint32_t logicalPageID : residentPageIDs) {
            uint32_t vsmIndex = logicalPageID / kShadowPagesPerVSM;
            uint32_t localPageIndex = logicalPageID % kShadowPagesPerVSM;

            const maths::mat4* viewProj = nullptr;
            if (vsmIndex < kSunLevelCount) {
                viewProj = &m_SunLevelViewProj[vsmIndex];
            } else {
                uint32_t faceSlot = vsmIndex - kSunLevelCount;
                if (faceSlot >= m_ActivePointLightCount * 6u) {
                    m_Pool.SetPageCoversDynamicContent(logicalPageID, false); // Stale/inactive point light slot.
                    continue;
                }
                viewProj = &m_PointFaceViewProj[faceSlot];
            }

            // Recovers this page's own NDC sub-rectangle from (pageX, pageY) -- the exact inverse
            // of RenderPage()'s own virtual-viewport math: a page at (pageX, pageY) covers screen
            // pixels [pageX*128, (pageX+1)*128) x [pageY*128, (pageY+1)*128) of the level/face's
            // full kShadowPagesPerAxis*kShadowPageTexels virtual resolution, which maps linearly to
            // NDC [-1, 1] regardless of the specific negative-offset viewport trick RenderPage()
            // uses to get there (the viewport only rescales NDC to pixels, it does not change what
            // NDC itself means for a given clip-space position).
            uint32_t pageX = localPageIndex % kShadowPagesPerAxis;
            uint32_t pageY = localPageIndex / kShadowPagesPerAxis;
            float ndcXMin = 2.0f * float(pageX) / float(kShadowPagesPerAxis) - 1.0f;
            float ndcXMax = 2.0f * float(pageX + 1) / float(kShadowPagesPerAxis) - 1.0f;
            float ndcYMin = 2.0f * float(pageY) / float(kShadowPagesPerAxis) - 1.0f;
            float ndcYMax = 2.0f * float(pageY + 1) / float(kShadowPagesPerAxis) - 1.0f;

            bool coversDynamic = false;
            for (const auto& [entityID, range] : m_EntityRanges) {
                if (!range.isDynamicCandidate) {
                    continue;
                }
                if (EntityAABBOverlapsPageNDC(range, entityTransformsCPU[entityID], *viewProj, ndcXMin, ndcXMax, ndcYMin, ndcYMax)) {
                    coversDynamic = true;
                    break;
                }
            }
            m_Pool.SetPageCoversDynamicContent(logicalPageID, coversDynamic);
        }
    }

    bool VirtualShadowMapPass::EntityAABBOverlapsPageNDC(const EntityDrawRange& range, const core::EntityTransformCPU& xform,
        const maths::mat4& viewProj, float ndcXMin, float ndcXMax, float ndcYMin, float ndcYMax) const {
        maths::vec3 localMin = range.boundsMin;
        maths::vec3 localMax = range.boundsMax;
        // Conservative inflation by the same worst-case bounds already established for GPU-side
        // culling (see kCpuSplineMaxDeviation/kCpuEnhancedDisplacementMaxAmplitude's own comment) --
        // symmetric per-axis, since this is a coarse CPU classification test, not precision-critical
        // GPU culling.
        if (range.hasSplineDeformation) {
            maths::vec3 dev{ kCpuSplineMaxDeviation, kCpuSplineMaxDeviation, kCpuSplineMaxDeviation };
            localMin = localMin - dev;
            localMax = localMax + dev;
        }
        if (range.hasEnhancedDisplacement) {
            maths::vec3 dev{ kCpuEnhancedDisplacementMaxAmplitude, kCpuEnhancedDisplacementMaxAmplitude, kCpuEnhancedDisplacementMaxAmplitude };
            localMin = localMin - dev;
            localMax = localMax + dev;
        }
        if (range.isSkeletallyAnimated) {
            maths::vec3 dev{ kCpuSkeletalMaxDeviation, kCpuSkeletalMaxDeviation, kCpuSkeletalMaxDeviation };
            localMin = localMin - dev;
            localMax = localMax + dev;
        }

        float entityNdcXMin = std::numeric_limits<float>::max();
        float entityNdcXMax = -std::numeric_limits<float>::max();
        float entityNdcYMin = std::numeric_limits<float>::max();
        float entityNdcYMax = -std::numeric_limits<float>::max();

        const auto& rot = xform.rotation.m;
        const auto& vp = viewProj.m;

        for (int corner = 0; corner < 8; ++corner) {
            maths::vec3 localCorner{
                (corner & 1) ? localMax.x : localMin.x,
                (corner & 2) ? localMax.y : localMin.y,
                (corner & 4) ? localMax.z : localMin.z
            };

            // worldPos = translation + center + rotation*(localCorner - center) -- same composition
            // contract as EntityTransform's own comment (struct_custo.glsl), same manual
            // column-major application SurfaceCachePass::ComputeCardPriority already uses (no
            // TransformDirection() helper exists on maths::mat4 in this codebase).
            maths::vec3 offset = localCorner - xform.center;
            maths::vec3 rotatedOffset{
                rot[0] * offset.x + rot[4] * offset.y + rot[8]  * offset.z,
                rot[1] * offset.x + rot[5] * offset.y + rot[9]  * offset.z,
                rot[2] * offset.x + rot[6] * offset.y + rot[10] * offset.z
            };
            maths::vec3 worldPos = xform.translation + xform.center + rotatedOffset;

            // Manual clip-space projection -- mirrors PostProcessPass.cpp's own god-rays sun
            // projection (see that file's comment: maths::mat4 has no vec4 type/operator* to lean
            // on, this is the same "first CPU-side forward projection" pattern).
            float clipX = vp[0] * worldPos.x + vp[4] * worldPos.y + vp[8]  * worldPos.z + vp[12];
            float clipY = vp[1] * worldPos.x + vp[5] * worldPos.y + vp[9]  * worldPos.z + vp[13];
            float clipW = vp[3] * worldPos.x + vp[7] * worldPos.y + vp[11] * worldPos.z + vp[15];

            if (clipW <= 1.0e-5f) {
                // Degenerate/behind-the-light corner (only possible for a point light's perspective
                // projection -- the sun's orthographic projection always has w==1) -- conservatively
                // treat this page as covering dynamic content rather than risk silently excluding a
                // page that should be redrawn. This is a CPU classification heuristic feeding an
                // extra-redraw decision, not GPU culling that could drop real geometry -- erring
                // toward more redraws is always safe here, erring the other way is not.
                return true;
            }
            float ndcX = clipX / clipW;
            float ndcY = clipY / clipW;
            entityNdcXMin = std::min(entityNdcXMin, ndcX);
            entityNdcXMax = std::max(entityNdcXMax, ndcX);
            entityNdcYMin = std::min(entityNdcYMin, ndcY);
            entityNdcYMax = std::max(entityNdcYMax, ndcY);
        }

        return entityNdcXMin <= ndcXMax && entityNdcXMax >= ndcXMin &&
               entityNdcYMin <= ndcYMax && entityNdcYMax >= ndcYMin;
    }

}
