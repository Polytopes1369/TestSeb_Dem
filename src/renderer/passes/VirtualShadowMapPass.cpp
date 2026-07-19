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
#include "renderer/passes/ParticleSystemPass.h" // Feature F7 (shadow-casting particles) -- SetParticleSystem()/RecordParticleShadows() call its public accessors.
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

        // Feature F14 (sun clipmap camera re-centering): CPU-side mirror of shadow_sun_sampling
        // .glsl's ShadowSunLevelsUBO -- byte-for-byte std140 layout (mat4[N] followed by ivec4[N],
        // both already naturally 16-byte-strided, so no explicit padding fields are needed between
        // or within them). windowStartPage[level].xy is this level's window-start page-grid
        // coordinate (see VirtualShadowMapPass.h's own class comment); .zw is unused (std140 pads
        // every array element of a 2-component integer type up to 16 bytes regardless, so an ivec4
        // costs nothing extra over an ivec2 here while sidestepping any doubt about array-stride
        // padding rules -- the same reasoning renderer::LightingTypes.h's own UBO mirrors already
        // apply throughout this codebase).
        struct SunLevelsUBOData {
            maths::mat4 viewProj[VirtualShadowMapPass::kSunLevelCount];
            int32_t windowStartPage[VirtualShadowMapPass::kSunLevelCount][4];
        };

        // Feature F14: wraps a page-grid coordinate into [0, kShadowPagesPerAxis) using floor-mod
        // (always non-negative, unlike C++'s truncating % for a negative left operand) -- the exact
        // CPU-side counterpart of shadow_page_table.glsl's ShadowWrapPageCoord; RenderPage() and
        // ClassifyDynamicPages() both use this to convert between a sun level's WORLD-ANCHORED
        // wrapped local page index and this frame's RASTER page position (see
        // VirtualShadowMapPass.h's own class comment for the full wrapped<->raster contract).
        int32_t WrapPageCoord(int32_t v) {
            constexpr int32_t m = static_cast<int32_t>(kShadowPagesPerAxis);
            return ((v % m) + m) % m;
        }

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

        // Feature F7 (shadow-casting particles): push constant for ParticleShadowCapture.vert -- flat
        // floats, not vec3 (avoids vec3's implicit push-constant alignment padding surprises when
        // reasoning about the byte layout by eye, same convention renderer::ParticleSystemPass::
        // EmitterParams' own header comment establishes). 88 bytes, comfortably under the 128-byte
        // guaranteed minimum push-constant size (same margin ShadowCaptureConstants above documents
        // for its own 76 bytes).
        struct ParticleShadowCaptureConstants {
            maths::mat4 lightViewProj;
            float lightRightX, lightRightY, lightRightZ;
            float lightUpX, lightUpY, lightUpZ;
        };
        static_assert(sizeof(ParticleShadowCaptureConstants) == 88,
            "ParticleShadowCaptureConstants must match ParticleShadowCapture.vert's push_constant block exactly");

        // Feature F7: coarse, deliberately generous fixed-radius bound around a shadow-casting
        // emitter's own (static) spawn position -- NOT a tight per-particle bound, which this CPU
        // code has no cheap way to compute (a particle's actual dispersion depends on GPU-simulated
        // gravity/wind/curl-noise/attractor forces this codebase deliberately never reads back to the
        // CPU per frame, see e.g. ClusterOcclusionCullingPass's own "only ever exists on the GPU"
        // comment on the identical principle for cluster visibility counts). Erring toward a larger
        // bound only costs a few extra (cheap, per-particle-degenerate-rejected in the vertex shader)
        // draw calls on pages that turn out to have no particle actually cross them -- it can never
        // DROP a real shadow, matching EntityAABBOverlapsPageNDC's own documented "erring toward more
        // redraws is always safe here" philosophy one level up.
        constexpr float kEmitterShadowBoundsRadius = 6.0f;

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
        // Feature F14: sized for SunLevelsUBOData (viewProj[] + windowStartPage[]), not just the
        // raw m_SunLevelViewProj array -- see that struct's own comment for the exact std140 shape
        // shadow_sun_sampling.glsl's ShadowSunLevelsUBO now expects.
        m_SunLevelsUBO.Create(allocator, sizeof(SunLevelsUBOData),
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
            // Feature F7: no descriptor pool/set/layout of its own to destroy here -- SetParticleSystem()
            // deliberately binds ParticleSystemPass's OWN descriptor set (GetSetLayout()/GetCurrentSet())
            // directly, see that method's own comment, so only the pipeline/pipelineLayout are this
            // pass' own resources to tear down.
            if (m_ParticleShadowPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ParticleShadowPipeline, nullptr);
            if (m_ParticleShadowPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ParticleShadowPipelineLayout, nullptr);
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
        m_ParticleShadowPipeline = VK_NULL_HANDLE;
        m_ParticleShadowPipelineLayout = VK_NULL_HANDLE;
        m_PagesRenderedThisFrame.clear();
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
        // --- Recompute this frame's sun clipmap view-projection matrices AND re-center each
        // level's window on the camera (Feature F14 -- see class comment for the full toroidal
        // wrap contract this drives). ---
        const maths::vec3 lightDir = sunDirection.Normalize();
        maths::vec3 up{ 0.0f, 1.0f, 0.0f };
        if (std::abs(lightDir.Dot(up)) > 0.999f) {
            up = maths::vec3{ 0.0f, 0.0f, 1.0f };
        }
        // The exact s/u basis maths::mat4::LookAt derives internally -- recomputed explicitly here
        // so this pass' page-grid axes exactly match the view matrix's real X/Y axes (see class
        // comment). lightUpForPaging is `u` NEGATED: OrthoVulkan flips Y (Vulkan Y-down NDC, see
        // that function's own comment), so a page's RASTER row index increases as world position
        // moves along -u, not +u -- every CPU-side "up axis" scalar projection/reconstruction below
        // uses this pre-flipped vector so the integers this pass stores/uploads as windowStartPage
        // land in the exact same sign convention shadow_sun_sampling.glsl's rasterPageCoord does.
        const maths::vec3 lightRight = lightDir.Cross(up).Normalize();
        const maths::vec3 lightUpForPaging = (lightRight.Cross(lightDir)) * -1.0f;
        // Feature F7 (shadow-casting particles): the TRUE (unflipped) sun basis for a light-facing
        // particle billboard -- lightUpForPaging above is deliberately Y-flipped for page-grid
        // arithmetic only (see this pass' own class comment), which must not leak into a billboard
        // orientation. Shared by every sun level (same lightDir/up for the whole frame).
        const maths::vec3 lightUpTrue = lightUpForPaging * -1.0f;

        uint32_t invalidatedThisFrame = 0;
        for (uint32_t level = 0; level < kSunLevelCount; ++level) {
            invalidatedThisFrame += RecenterSunLevel(cmd, level, cameraPosition, lightDir, lightRight, lightUpForPaging, up);
        }

        // --- Recompute this frame's point light cube face view-projection matrices. Unaffected by
        // Feature F14 -- a point light's own frustum is anchored to the light, never re-centers. ---
        // Feature F7: also precomputes each face's own TRUE right/up basis (the exact s/u pair
        // maths::mat4::LookAt derives internally for that face's eye/center/up below) for a
        // light-facing particle billboard drawn into that face's own pages -- cheap (6 cross
        // products/frame) regardless of whether any castShadows emitter is even active this frame.
        maths::vec3 pointFaceRight[6]{};
        maths::vec3 pointFaceUpTrue[6]{};
        for (uint32_t face = 0; face < 6; ++face) {
            pointFaceRight[face] = kFaceDirs[face].Cross(kFaceUps[face]).Normalize();
            pointFaceUpTrue[face] = pointFaceRight[face].Cross(kFaceDirs[face]);
        }
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

        // Feature F7: cleared here, repopulated by every RenderPage() call this frame's two render
        // loops below actually issue -- consumed once, later this same frame, by
        // RecordParticleShadows() (called AFTER ParticleSystemPass's own simulate/sort dispatches,
        // see that method's own comment for why it cannot simply run inline here).
        m_PagesRenderedThisFrame.clear();
        // Local helper (captures this frame's lightRight/lightUpTrue/pointFaceRight/pointFaceUpTrue
        // above by reference) -- called once per RenderPage() invocation below to append that page's
        // own PageRenderInfo + light-facing billboard basis into m_PagesRenderedThisFrame.
        auto trackRenderedPage = [&](uint32_t vsmIndex, uint32_t physicalLayer, const maths::mat4& viewProj,
            const PageRenderInfo& info) {
            RenderedPageThisFrame tracked{};
            tracked.physicalLayer = physicalLayer;
            tracked.viewProj = viewProj;
            tracked.viewport = info.viewport;
            tracked.scissor = info.scissor;
            tracked.ndcXMin = info.ndcXMin;
            tracked.ndcXMax = info.ndcXMax;
            tracked.ndcYMin = info.ndcYMin;
            tracked.ndcYMax = info.ndcYMax;
            if (vsmIndex < kSunLevelCount) {
                tracked.lightRight = lightRight;
                tracked.lightUp = lightUpTrue;
            } else {
                uint32_t face = (vsmIndex - kSunLevelCount) % 6u;
                tracked.lightRight = pointFaceRight[face];
                tracked.lightUp = pointFaceUpTrue[face];
            }
            m_PagesRenderedThisFrame.push_back(tracked);
            };

        // Feature F14: uploads both this frame's view-proj matrices AND each sun level's
        // windowStartPage -- see SunLevelsUBOData's own comment for the exact std140-mirrored
        // layout shadow_sun_sampling.glsl's ShadowSunLevelsUBO expects.
        SunLevelsUBOData sunLevelsUpload{};
        for (uint32_t level = 0; level < kSunLevelCount; ++level) {
            sunLevelsUpload.viewProj[level] = m_SunLevelViewProj[level];
            sunLevelsUpload.windowStartPage[level][0] = m_SunLevelWindowStartPage[level][0];
            sunLevelsUpload.windowStartPage[level][1] = m_SunLevelWindowStartPage[level][1];
            sunLevelsUpload.windowStartPage[level][2] = 0;
            sunLevelsUpload.windowStartPage[level][3] = 0;
        }
        std::memcpy(m_SunLevelsUBO.MappedData(), &sunLevelsUpload, sizeof(sunLevelsUpload));
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
        uint32_t missPagesRenderedThisFrame = 0; // Feature F14 validation counter -- see this function's own trailing throttled log.
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

                PageRenderInfo info = RenderPage(cmd, vsmIndex, localPageIndex, physicalLayer, *viewProj);
                trackRenderedPage(vsmIndex, physicalLayer, *viewProj, info);
                renderedAnyPage = true;
                ++missPagesRenderedThisFrame;
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

                PageRenderInfo info = RenderPage(cmd, vsmIndex, localPageIndex, physicalLayer, *viewProj);
                trackRenderedPage(vsmIndex, physicalLayer, *viewProj, info);
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

        // Feature F14 validation diagnostic: proves smooth camera motion re-centers the sun clipmap
        // WITHOUT invalidating the whole pool every frame -- `invalidatedThisFrame` should stay a
        // small fraction of m_Pool.GetResidentPageCount() (only the newly-revealed band per moved
        // axis, see RecenterSunLevel/InvalidateSunWindowBand's own comments), not the whole
        // resident set, under continuous flight. Logged whenever a re-center actually invalidated
        // something (the interesting event) OR periodically otherwise, same throttle cadence as the
        // Feature 2 diagnostic just above.
        static uint32_t s_RecenterLogFrameCounter = 0;
        bool shouldLogRecenter = invalidatedThisFrame > 0 || ((++s_RecenterLogFrameCounter % 300u) == 0u);
        if (shouldLogRecenter) {
            LOG_INFO(std::format(
                "[VirtualShadowMapPass] Sun clipmap re-center: {} page(s) invalidated (band-only, not pool-wide), "
                "{} page(s) miss-rendered, {} of {} physical page(s) resident.",
                invalidatedThisFrame, missPagesRenderedThisFrame, m_Pool.GetResidentPageCount(), m_Pool.GetPhysicalCapacity()));
        }
    }

    void VirtualShadowMapPass::RecordEndFrame(VkCommandBuffer cmd) {
        // Captures this frame's page-miss reports (written by whatever ran between
        // RecordBeginFrame() and this call) for RecordBeginFrame() to consume next frame -- see
        // this class' own header comment for the full one-frame-lag contract.
        m_Feedback.RecordReadback(cmd);
    }

    void VirtualShadowMapPass::SetParticleSystem(const ParticleSystemPass& particles) {
        if (m_ParticleShadowPipeline != VK_NULL_HANDLE) {
            return; // Already built -- Init()-time idempotency convention this class doesn't otherwise need, but a caller mistakenly calling this twice must not leak a pipeline.
        }

        // Pipeline layout: ONE descriptor set -- `particles`' own set 0 (ParticleCommon.glsl's
        // ParticleBuffer/DeadListBuffer/AliveListBuffer/CounterBuffer/EmitterParamsBuffer/
        // PerEmitterAliveCountBuffer, the SAME layout every real particle shader binds, see that
        // file's own header comment on why hardcoding set 0 there is correct) -- reused directly via
        // GetSetLayout() rather than this pass building a second, redundant descriptor set/pool of
        // its own for 3 of those same 6 buffers, plus this pipeline's own push constant range.
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(ParticleShadowCaptureConstants);

        VkDescriptorSetLayout particleSetLayout = particles.GetSetLayout();
        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &particleSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_ParticleShadowPipelineLayout));

        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleShadowCapture.vert.spv");
        VkShaderModule fragModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ParticleShadowCapture.frag.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragModule;
        stages[1].pName = "main";

        // No bound vertex/index buffer -- gl_VertexIndex (0-5) generates the quad corners in-shader,
        // same "zero vertex input state" convention ParticleRender.vert's own pipeline already uses.
        VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE; // A billboard quad's footprint must occlude the light regardless of winding.
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.attachmentCount = 0; // Depth-only, same as m_Pipeline/m_MaskedPipeline above.

        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 0;
        pipelineRendering.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        // depthWriteEnable=TRUE, VK_COMPARE_OP_LESS -- same convention as m_Pipeline/m_MaskedPipeline
        // above: RecordParticleShadows() uses LOAD_OP_LOAD (preserving this frame's already-captured
        // entity depth, see that method's own comment), so a particle's own depth only wins where it
        // is genuinely nearer to the light than whatever entity geometry (if any) was already
        // rasterized into that texel -- standard, correct depth-test behavior, nothing special-cased.
        VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

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
        pipelineInfo.layout = m_ParticleShadowPipelineLayout;
        pipelineInfo.pNext = &pipelineRendering;
        // Routed through the persisted VkPipelineCache (same convention m_Pipeline/m_MaskedPipeline
        // above now use, landed on main concurrently with this feature) instead of VK_NULL_HANDLE.
        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_ParticleShadowPipeline));

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, fragModule, nullptr);

        LOG_INFO("[VirtualShadowMapPass] Feature F7: particle-shadow-capture pipeline built (bound to ParticleSystemPass's own descriptor set layout).");
    }

    void VirtualShadowMapPass::RecordParticleShadows(VkCommandBuffer cmd, const ParticleSystemPass& particles) {
        if (m_ParticleShadowPipeline == VK_NULL_HANDLE || m_PagesRenderedThisFrame.empty()) {
            return; // SetParticleSystem() not called yet, or RecordBeginFrame() rendered nothing this frame.
        }

        // Explicit barrier (CLAUDE.md's synchronization discipline): `particles`' own
        // RecordSimulate()/RecordSort() compute dispatches (already recorded earlier THIS SAME
        // command buffer -- see renderer::ClusterRenderPipeline::RecordFrameEarly's own call-ordering
        // comment, and this method's own class-comment cross-reference for why RecordParticleShadows
        // must run AFTER them) wrote ParticleBuffer/AliveListBuffer/CounterBuffer/EmitterParamsBuffer;
        // make those writes visible to this call's own vertex-shader storage-buffer reads AND
        // vkCmdDrawIndirect's own indirect-command read of the SAME m_IndirectDrawBuffer
        // ParticleSystemPass::RecordDraw() also consumes, before any draw below issues.
        VkMemoryBarrier2 particleBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        particleBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        particleBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        particleBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        particleBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo particleDepInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        particleDepInfo.memoryBarrierCount = 1;
        particleDepInfo.pMemoryBarriers = &particleBarrier;
        vkCmdPipelineBarrier2(cmd, &particleDepInfo);

        VkDescriptorSet particleSet = particles.GetCurrentSet();
        VkBuffer indirectDrawBuffer = particles.GetIndirectDrawBufferHandle();

        bool boundPipelineThisCall = false;
        uint32_t pagesTouched = 0;
        for (const RenderedPageThisFrame& page : m_PagesRenderedThisFrame) {
            // Page budget (this feature's own explicit requirement): skip any page whose frustum does
            // not overlap ANY castShadows-enabled emitter's own bounds -- the common case once the
            // scene has more than a handful of pages resident, see AnyShadowCastingEmitterOverlapsPageNDC's
            // own comment for the overlap test itself.
            if (!AnyShadowCastingEmitterOverlapsPageNDC(page.viewProj, page.ndcXMin, page.ndcXMax, page.ndcYMin, page.ndcYMax)) {
                continue;
            }

            VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            depthAttachment.imageView = m_Pool.GetPhysicalLayerView(page.physicalLayer);
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            // LOAD, not CLEAR: this page's entity depth (RenderPage(), earlier THIS SAME frame) must
            // survive -- a particle shadow is drawn ON TOP OF the static/dynamic scene's own shadow
            // casters, not instead of them.
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            renderingInfo.renderArea = { { 0, 0 }, { kShadowPageTexels, kShadowPageTexels } };
            renderingInfo.layerCount = 1;
            renderingInfo.pDepthAttachment = &depthAttachment;
            vkCmdBeginRendering(cmd, &renderingInfo);

            vkCmdSetViewport(cmd, 0, 1, &page.viewport);
            vkCmdSetScissor(cmd, 0, 1, &page.scissor);

            if (!boundPipelineThisCall) {
                // Bound once, reused for every touched page -- same "skip redundant rebinds" idiom
                // RenderPage's own per-entity pipeline selection above already uses, just at the
                // per-page granularity here instead (this pipeline never changes across pages).
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ParticleShadowPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ParticleShadowPipelineLayout, 0, 1, &particleSet, 0, nullptr);
                boundPipelineThisCall = true;
            }

            ParticleShadowCaptureConstants pc{};
            pc.lightViewProj = page.viewProj;
            pc.lightRightX = page.lightRight.x; pc.lightRightY = page.lightRight.y; pc.lightRightZ = page.lightRight.z;
            pc.lightUpX = page.lightUp.x; pc.lightUpY = page.lightUp.y; pc.lightUpZ = page.lightUp.z;
            vkCmdPushConstants(cmd, m_ParticleShadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            // Reuses the SAME indirect-draw args buffer ParticleSystemPass::RecordDraw() draws the
            // color pass from (vertexCount=6, instanceCount=this frame's real alive-particle count --
            // see that buffer's own declaration comment) -- every alive particle is instanced;
            // ParticleShadowCapture.vert's own per-particle emitter check degenerates the vertex for
            // any particle whose emitter does not have castShadows set, the same "cheap per-vertex
            // reject, no separate compaction pass" convention ParticleRender.vert's own render-mode
            // gating (B1/B2/B3) already established -- see that shader's own header comment.
            vkCmdDrawIndirect(cmd, indirectDrawBuffer, 0, 1, sizeof(VkDrawIndirectCommand));

            vkCmdEndRendering(cmd);
            ++pagesTouched;
        }

        if (pagesTouched > 0) {
            // Same barrier reasoning as RecordBeginFrame's own two trailing barriers -- a page's
            // depth written THIS frame (here: a particle shadow layered on top) must be immediately
            // visible to a later sampled read this same frame, not next frame.
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

        // Validation diagnostic (this feature's own explicit ask: "measure the added VSM page
        // cost"): how many of this frame's rendered pages actually got a particle-shadow draw call,
        // out of how many were candidates -- proves the page-budget overlap test stays selective
        // (a small fraction) rather than degenerating into "draw particles on every page".
        static uint32_t s_ParticleShadowLogFrameCounter = 0;
        if (pagesTouched > 0 && ((++s_ParticleShadowLogFrameCounter % 60u) == 0u)) {
            LOG_INFO(std::format(
                "[VirtualShadowMapPass] Particle shadows: {} of {} rendered page(s) this frame got a particle-shadow draw call.",
                pagesTouched, m_PagesRenderedThisFrame.size()));
        }
    }

    uint32_t VirtualShadowMapPass::RecenterSunLevel(VkCommandBuffer cmd, uint32_t level, const maths::vec3& cameraPosition,
        const maths::vec3& lightDir, const maths::vec3& lightRight, const maths::vec3& lightUpForPaging,
        const maths::vec3& up) {
        float radius = std::max(kSunBaseRadius * float(1u << level), 1.0e-3f);
        float nearPlane = radius * kSunNearMarginFactor;
        // World units per page at this level -- the level's full covered width (2*radius) split
        // into kShadowPagesPerAxis pages, same convention RenderPage's own virtual-viewport math
        // assumes (kShadowPagesPerAxis pages spanning the level's whole [-radius, radius] extent).
        float pageWorldSize = (2.0f * radius) / float(kShadowPagesPerAxis);

        // Snap the camera's projection onto this axis down to whole kSunWindowSnapChunkPages-page
        // steps -- exactly GlobalSDFPass::EnqueueDirtyRegionsForLevel's own snapAxis lambda,
        // parameterized in pages instead of voxels (see class comment).
        auto snapToPageChunk = [&](float axisWorldPos) -> int32_t {
            int32_t pageIndex = static_cast<int32_t>(std::floor(axisWorldPos / pageWorldSize));
            int32_t chunk = static_cast<int32_t>(kSunWindowSnapChunkPages);
            int32_t chunkIndex = static_cast<int32_t>(std::floor(static_cast<float>(pageIndex) / static_cast<float>(chunk)));
            return chunkIndex * chunk;
            };

        float camRight = cameraPosition.Dot(lightRight);
        float camUpForPaging = cameraPosition.Dot(lightUpForPaging);
        int32_t newCenterPage[2] = { snapToPageChunk(camRight), snapToPageChunk(camUpForPaging) };

        constexpr int32_t kHalfPagesPerAxis = static_cast<int32_t>(kShadowPagesPerAxis) / 2; // 8.

        SunClipmapWindow& window = m_SunWindows[level];
        uint32_t invalidatedCount = 0;
        if (!window.hasValidWindow) {
            // First call for this level: nothing is resident yet (a freshly Init()'d pool, or a
            // level that has simply never had a page requested) -- no stale content can possibly
            // exist to invalidate, see class comment.
            window.centerPage[0] = newCenterPage[0];
            window.centerPage[1] = newCenterPage[1];
            window.hasValidWindow = true;
        } else {
            for (int axis = 0; axis < 2; ++axis) {
                int32_t delta = newCenterPage[axis] - window.centerPage[axis];
                if (delta == 0) {
                    continue;
                }
                if (std::abs(delta) >= static_cast<int32_t>(kShadowPagesPerAxis)) {
                    // Moved farther than the whole covered window in one update (a large camera
                    // jump/teleport) -- every wrapped index along this axis is stale; mirrors
                    // GlobalSDFPass::EnqueueDirtyRegionsForLevel's identical large-jump case.
                    invalidatedCount += InvalidateSunWindowBand(cmd, level, newCenterPage[axis] - kHalfPagesPerAxis,
                        static_cast<int32_t>(kShadowPagesPerAxis), axis);
                } else if (delta > 0) {
                    invalidatedCount += InvalidateSunWindowBand(cmd, level, window.centerPage[axis] + kHalfPagesPerAxis, delta, axis);
                } else {
                    invalidatedCount += InvalidateSunWindowBand(cmd, level, newCenterPage[axis] - kHalfPagesPerAxis, -delta, axis);
                }
            }
            window.centerPage[0] = newCenterPage[0];
            window.centerPage[1] = newCenterPage[1];
        }

        m_SunLevelWindowStartPage[level][0] = window.centerPage[0] - kHalfPagesPerAxis;
        m_SunLevelWindowStartPage[level][1] = window.centerPage[1] - kHalfPagesPerAxis;

        // World-space window center: the camera's own position, with its light-right/
        // light-up-for-paging components replaced by the snapped page-grid center (texel-stable
        // across sub-chunk motion), keeping the light-DIRECTION component exactly at the camera's
        // real position -- depth needs no snapping, only the two axes perpendicular to the light do
        // (see class comment).
        maths::vec3 windowCenterWorld = cameraPosition
            - lightRight * camRight - lightUpForPaging * camUpForPaging
            + lightRight * (static_cast<float>(window.centerPage[0]) * pageWorldSize)
            + lightUpForPaging * (static_cast<float>(window.centerPage[1]) * pageWorldSize);

        maths::vec3 eye = windowCenterWorld - lightDir * (radius + nearPlane);
        maths::mat4 view = maths::mat4::LookAt(eye, windowCenterWorld, up);
        m_SunLevelViewProj[level] = m_SunLevelProj[level] * view;

        return invalidatedCount;
    }

    uint32_t VirtualShadowMapPass::InvalidateSunWindowBand(VkCommandBuffer cmd, uint32_t level,
        int32_t bandStartWorldPage, int32_t bandCount, int32_t movedAxis) {
        uint32_t invalidatedCount = 0;
        for (int32_t i = 0; i < bandCount; ++i) {
            int32_t worldCoord = bandStartWorldPage + i;
            uint32_t wrapped = static_cast<uint32_t>(WrapPageCoord(worldCoord));
            for (uint32_t other = 0; other < kShadowPagesPerAxis; ++other) {
                // movedAxis 0 (light-right) is the local index's X component; movedAxis 1
                // (light-up-for-paging) is its Y component -- matches ShadowLocalPageIndex's own
                // `y * SHADOW_PAGES_PER_AXIS + x` flattening (shadow_page_table.glsl).
                uint32_t localPageIndex = (movedAxis == 0)
                    ? (other * kShadowPagesPerAxis + wrapped)
                    : (wrapped * kShadowPagesPerAxis + other);
                uint32_t logicalPageID = level * kShadowPagesPerVSM + localPageIndex;
                bool wasResident = m_Pool.IsResident(logicalPageID);
                m_Pool.InvalidatePage(cmd, logicalPageID);
                if (wasResident) {
                    ++invalidatedCount;
                }
            }
        }
        return invalidatedCount;
    }

    VirtualShadowMapPass::PageRenderInfo VirtualShadowMapPass::RenderPage(VkCommandBuffer cmd, uint32_t vsmIndex, uint32_t localPageIndex,
        uint32_t physicalLayer, const maths::mat4& viewProj) {
        uint32_t pageX = 0;
        uint32_t pageY = 0;
        if (vsmIndex < kSunLevelCount) {
            // Sun clipmap (Feature F14): `localPageIndex` is the WORLD-ANCHORED wrapped page-table
            // slot (stable across a window shift), not this frame's raster position -- recover the
            // raster position (where in the current 2048x2048 virtual frustum this page's content
            // actually belongs, for the virtual-viewport trick below) via the inverse of the wrap
            // this level's CURRENT window applies. See class comment for the full wrapped<->raster
            // derivation and shadow_sun_sampling.glsl's ShadowWrapPageCoord for the GPU-side
            // forward direction of the exact same transform.
            uint32_t wrappedX = localPageIndex % kShadowPagesPerAxis;
            uint32_t wrappedY = localPageIndex / kShadowPagesPerAxis;
            pageX = static_cast<uint32_t>(WrapPageCoord(static_cast<int32_t>(wrappedX) - m_SunLevelWindowStartPage[vsmIndex][0]));
            pageY = static_cast<uint32_t>(WrapPageCoord(static_cast<int32_t>(wrappedY) - m_SunLevelWindowStartPage[vsmIndex][1]));
        } else {
            // Point light cube face -- unaffected by Feature F14, direct (unwrapped) mapping.
            pageX = localPageIndex % kShadowPagesPerAxis;
            pageY = localPageIndex / kShadowPagesPerAxis;
        }

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

        // Feature F7 (shadow-casting particles): this page's own NDC sub-rectangle, same formula
        // ClassifyDynamicPages uses to recover one from (pageX, pageY) -- returned (not just used
        // internally) so RecordBeginFrame's own callers can register this page in
        // m_PagesRenderedThisFrame for RecordParticleShadows()'s later per-page overlap test,
        // without duplicating this derivation a second time at each call site.
        PageRenderInfo info{};
        info.viewport = viewport;
        info.scissor = scissor;
        info.ndcXMin = 2.0f * float(pageX) / float(kShadowPagesPerAxis) - 1.0f;
        info.ndcXMax = 2.0f * float(pageX + 1) / float(kShadowPagesPerAxis) - 1.0f;
        info.ndcYMin = 2.0f * float(pageY) / float(kShadowPagesPerAxis) - 1.0f;
        info.ndcYMax = 2.0f * float(pageY + 1) / float(kShadowPagesPerAxis) - 1.0f;
        return info;
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
            // Feature F14: for a sun level, `localPageIndex` is the wrapped (world-anchored) slot,
            // not the raster position `viewProj` (THIS frame's re-centered matrix) actually implies
            // -- same wrapped->raster conversion as RenderPage's own, see that function's comment.
            uint32_t pageX, pageY;
            if (vsmIndex < kSunLevelCount) {
                uint32_t wrappedX = localPageIndex % kShadowPagesPerAxis;
                uint32_t wrappedY = localPageIndex / kShadowPagesPerAxis;
                pageX = static_cast<uint32_t>(WrapPageCoord(static_cast<int32_t>(wrappedX) - m_SunLevelWindowStartPage[vsmIndex][0]));
                pageY = static_cast<uint32_t>(WrapPageCoord(static_cast<int32_t>(wrappedY) - m_SunLevelWindowStartPage[vsmIndex][1]));
            } else {
                pageX = localPageIndex % kShadowPagesPerAxis;
                pageY = localPageIndex / kShadowPagesPerAxis;
            }
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
            // Feature F7 (shadow-casting particles) extension: a page also counts as covering
            // dynamic content if a shadow-casting emitter's own bounds overlap it -- see this
            // method's own class-comment cross-reference for why this is required (without it, a
            // page whose underlying entity geometry is static would only ever be classified dynamic
            // via the entity test above, so RecordParticleShadows() would lose its chance to keep
            // re-drawing a moving particle's shadow onto it after the very first frame).
            if (!coversDynamic) {
                coversDynamic = AnyShadowCastingEmitterOverlapsPageNDC(*viewProj, ndcXMin, ndcXMax, ndcYMin, ndcYMax);
            }
            m_Pool.SetPageCoversDynamicContent(logicalPageID, coversDynamic);
        }
    }

    bool VirtualShadowMapPass::AnyShadowCastingEmitterOverlapsPageNDC(const maths::mat4& viewProj,
        float ndcXMin, float ndcXMax, float ndcYMin, float ndcYMax) const {
        for (uint32_t i = 0; i < config::particles::kMaxEmitters; ++i) {
            const config::particles::EmitterConfig& emitter = config::particles::EMITTERS[i];
            if (!emitter.active || !emitter.castShadows) {
                continue;
            }

            maths::vec3 center{ emitter.positionX, emitter.positionY, emitter.positionZ };
            maths::vec3 localMin = center - maths::vec3{ kEmitterShadowBoundsRadius, kEmitterShadowBoundsRadius, kEmitterShadowBoundsRadius };
            maths::vec3 localMax = center + maths::vec3{ kEmitterShadowBoundsRadius, kEmitterShadowBoundsRadius, kEmitterShadowBoundsRadius };

            float entityNdcXMin = std::numeric_limits<float>::max();
            float entityNdcXMax = -std::numeric_limits<float>::max();
            float entityNdcYMin = std::numeric_limits<float>::max();
            float entityNdcYMax = -std::numeric_limits<float>::max();
            const auto& vp = viewProj.m;
            bool behindLight = false;

            for (int corner = 0; corner < 8; ++corner) {
                maths::vec3 worldPos{
                    (corner & 1) ? localMax.x : localMin.x,
                    (corner & 2) ? localMax.y : localMin.y,
                    (corner & 4) ? localMax.z : localMin.z
                };

                // Manual clip-space projection -- same convention EntityAABBOverlapsPageNDC's own
                // corner loop above (and PostProcessPass.cpp's god-rays sun projection) already
                // establishes: maths::mat4 has no vec4 type/operator* to lean on.
                float clipX = vp[0] * worldPos.x + vp[4] * worldPos.y + vp[8] * worldPos.z + vp[12];
                float clipY = vp[1] * worldPos.x + vp[5] * worldPos.y + vp[9] * worldPos.z + vp[13];
                float clipW = vp[3] * worldPos.x + vp[7] * worldPos.y + vp[11] * worldPos.z + vp[15];

                if (clipW <= 1.0e-5f) {
                    // Degenerate/behind-the-light corner (point light only, sun's ortho projection
                    // always has w==1) -- conservatively treat this page as overlapping rather than
                    // risk silently excluding a page that should get a particle-shadow draw, same
                    // rationale as EntityAABBOverlapsPageNDC's own identical case.
                    behindLight = true;
                    break;
                }
                float ndcX = clipX / clipW;
                float ndcY = clipY / clipW;
                entityNdcXMin = std::min(entityNdcXMin, ndcX);
                entityNdcXMax = std::max(entityNdcXMax, ndcX);
                entityNdcYMin = std::min(entityNdcYMin, ndcY);
                entityNdcYMax = std::max(entityNdcYMax, ndcY);
            }

            if (behindLight) {
                return true;
            }
            if (entityNdcXMin <= ndcXMax && entityNdcXMax >= ndcXMin &&
                entityNdcYMin <= ndcYMax && entityNdcYMax >= ndcYMin) {
                return true;
            }
        }
        return false;
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
