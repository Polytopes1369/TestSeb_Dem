#include "renderer/passes/TransparentForwardPass.h"

#include <cstring>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/GpuPageTable.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/passes/WorldProbeGridPass.h"
#include "renderer/passes/SurfaceCacheTraceContext.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of TransparentViewParamsUBO in TransparentForward.vert/.frag
        // (std140): mat4 (64) + mat4 (64) + cameraPositionWorld vec3+pad (16) + sunDirectionAndIntensity
        // vec4 (16) + sunColor vec4 (16, only .rgb used) = 176 bytes. Phase 5: grew from a bare
        // sunDirection to a camera position (feeds the optional reflection trace's view-direction
        // reconstruction) plus sun color/intensity, matching SurfaceCacheLightingUBO's own sun
        // fields (SurfaceCacheCapture.frag) so TransparentForward.frag's ported ComputeDirectLighting
        // reads the identical layout. No point-light fields here (unlike the first Phase 5 draft) --
        // MegaLights' own RIS-selected point light + traced shadow ray (megalights_ris.glsl) is the
        // point-light path for this pass now, reconciled after `main` landed MegaLights Phase A
        // concurrently; see TransparentForward.frag's own header comment.
        struct TransparentViewParams {
            maths::mat4 view;
            maths::mat4 proj;
            // Phase PP3: globalTime repurposes what used to be pure std140 padding after
            // cameraPositionWorld -- feeds TransparentForward.frag's own animated refraction noise.
            float cameraPosX = 0.0f, cameraPosY = 0.0f, cameraPosZ = 0.0f, globalTime = 0.0f;
            float sunDirX = 0.0f, sunDirY = 0.0f, sunDirZ = 0.0f, sunIntensity = 0.0f;
            float sunColorR = 0.0f, sunColorG = 0.0f, sunColorB = 0.0f, _sunColorPad = 0.0f;
        };
        static_assert(sizeof(TransparentViewParams) == 176,
            "TransparentViewParams must match TransparentViewParamsUBO in TransparentForward.vert/.frag exactly (std140 layout)");

        // renderer::WorldProbeGridParams (WorldProbeGridPass.h) is reused as-is here -- already the
        // exact shared mirror of world_probe_sampling.glsl's WorldProbeGridParamsUBO every other
        // consumer (GICompositePass, ScreenTracePass) uses, no need for a local duplicate.

        // Fragment-stage-only push constants for the optional per-material reflection trace (see
        // TransparentForward.frag's own comment) -- same field set as renderer::ReflectionPass's
        // own TracePushConstants (ReflectionTrace.comp). MegaLights' own RIS candidate decorrelation
        // reuses this same frameIndex field (see TransparentForward.frag's own comment) rather than
        // a second UBO/push-constant copy.
        struct TransparentPushConstants {
            uint32_t entityCount = 0;
            uint32_t traceMode = 0;
            uint32_t frameIndex = 0;
        };

    } // namespace

    // `tlas` is renderer::SurfaceCacheRayTracingPass::GetTLASHandle() -- ONE handle, bound once at
    // binding 11, shared by both the optional per-material reflection trace's TraceHWRT and
    // MegaLights' own TraceShadowRay (see class comment). `lightBuffer`/`lightBufferSize` are
    // renderer::MegaLightsPass::GetLightBufferHandle()/GetLightBufferSize(), bound at binding 12.
    void TransparentForwardPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        VkExtent2D renderExtent,
        VkBuffer pageTableBuffer, VkBuffer compressedPhysicalPoolBuffer,
        VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer, VkBuffer wpoGlobalsBuffer,
        const std::array<MaterialParameters, kMaxMaterials>& materialTable,
        const std::vector<geometry::ClusterIndexEntry>& indexEntries,
        const std::vector<geometry::DAGNodeEntry>& dagEntries,
        VkFormat colorFormat, VkFormat depthFormat,
        const WorldProbeGridPass& worldProbes, const SurfaceCacheTraceContext& traceContext,
        VkAccelerationStructureKHR tlas, VkBuffer lightBuffer, VkDeviceSize lightBufferSize,
        VkBuffer fallbackVertexBuffer, VkBuffer fallbackIndexBuffer, VkBuffer drawRangeBuffer) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // --- Phase PP3: g_RefractionOffset, created UNCONDITIONALLY (before the possible zero-
        // transparent-cluster early-return below) -- renderer::PostProcessPass needs a valid view
        // to sample every frame regardless of whether this run's fixed random seed happened to
        // produce any transparent geometry at all. COLOR_ATTACHMENT_BIT: this pass' own draw target
        // (see RecordDraw()'s own comment). SAMPLED_BIT: renderer::PostProcessPass's composite
        // shader reads it through a sampler2D. ---
        {
            VkImageCreateInfo refractionInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            refractionInfo.imageType = VK_IMAGE_TYPE_2D;
            refractionInfo.format = kRefractionOffsetFormat;
            refractionInfo.extent = { renderExtent.width, renderExtent.height, 1 };
            refractionInfo.mipLevels = 1;
            refractionInfo.arrayLayers = 1;
            refractionInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            refractionInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            refractionInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            refractionInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            refractionInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            m_RefractionOffsetImage.Create(allocator, device, refractionInfo, VMA_MEMORY_USAGE_GPU_ONLY, VK_IMAGE_ASPECT_COLOR_BIT);

            VulkanUtils::TransitionImageLayoutOneShot(device, commandPool, queue, m_RefractionOffsetImage.Image(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

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
        m_DrawCountBuffer.Create(allocator, sizeof(uint32_t),
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_ViewParamsBuffer.Create(allocator, sizeof(TransparentViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_WorldProbeGridParamsBuffer.Create(allocator, sizeof(WorldProbeGridParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        // One-time upload of the static entries list, the fixed draw count (see m_DrawCountBuffer's
        // own comment), and the World Probe Grid's static addressing -- mirrors every other pass's
        // own one-shot setup submit in this codebase.
        WorldProbeGridParams gridParams{};
        gridParams.gridOriginX = worldProbes.GetGridOriginWorld().x;
        gridParams.gridOriginY = worldProbes.GetGridOriginWorld().y;
        gridParams.gridOriginZ = worldProbes.GetGridOriginWorld().z;
        gridParams.probeSpacing = WorldProbeGridPass::kProbeSpacing;
        gridParams.gridResolution = static_cast<float>(WorldProbeGridPass::kGridResolution);

        // m_ClusterEntriesBuffer's byte size scales with m_ClusterCount (scene-composition-
        // dependent -- every entity with an alpha < 1.0 material contributes its leaf clusters, see
        // this function's own loop above), so vkCmdUpdateBuffer's hard 65536-byte spec ceiling
        // (VUID-vkCmdUpdateBuffer-dataSize-00037) cannot be assumed to hold for it the way it safely
        // does for the two small, fixed-size buffers below. Staged through a temporary host-visible
        // buffer + vkCmdCopyBuffer instead, mirroring VulkanContext::UploadEntityData's own identical
        // staging pattern for the same reason (its own upload size is scene/entity-count-dependent).
        const VkDeviceSize clusterEntriesBytes = sizeof(TransparentClusterEntry) * m_ClusterCount;
        VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        stagingInfo.size = clusterEntriesBytes;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo stagingAllocInfo{};
        stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        VmaAllocationInfo stagingAllocResultInfo{};
        if (vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo,
                            &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate TransparentForwardPass cluster-entries staging buffer!");
        }
        std::memcpy(stagingAllocResultInfo.pMappedData, hostEntries.data(), static_cast<size_t>(clusterEntriesBytes));

        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkBufferCopy copyRegion{0, 0, clusterEntriesBytes};
            vkCmdCopyBuffer(cmd, stagingBuffer, m_ClusterEntriesBuffer.Handle(), 1, &copyRegion);
            vkCmdUpdateBuffer(cmd, m_DrawCountBuffer.Handle(), 0, sizeof(uint32_t), &m_ClusterCount);
            vkCmdUpdateBuffer(cmd, m_WorldProbeGridParamsBuffer.Handle(), 0, sizeof(WorldProbeGridParams), &gridParams);

            // The copy's writes must be visible before the SSBO is read (matches
            // VulkanContext::UploadEntityData's own barrier for the same staged-copy pattern).
            VkMemoryBarrier2 copyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            copyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            copyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            copyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            copyBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &copyBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        });

        vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);

        // =====================================================================================
        // Descriptor pool, shared by both descriptor sets below.
        // =====================================================================================
        // Phase 5 + MegaLights Phase A follow-up, reconciled (see class comment): Forward bindings
        // 11-17 -- shared TLAS (reflection trace + MegaLights shadow ray), MegaLights light SSBO,
        // World Probe Grid params UBO + sampler, and the Fallback Mesh vertex/index/drawRange trio.
        // No point-light-shadow-faces UBO any more (MegaLights supersedes this pass's own point-
        // light loop). See project_sun_shadow_random_materials.md's own documented lesson: this is
        // exactly the class of bug -- a pool undercounted for bindings written after Init()'s own
        // first pass -- to not repeat here.
        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12 };           // Compact(3) + Forward(9: entries, pool, entityXform, entityData, materialParams, MegaLights g_Lights, fallbackVertex, fallbackIndex, drawRange).
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 };            // Forward: WPOGlobals, ViewParams, g_ShadowSunLevels, WorldProbeGridParams.
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 };    // Forward: shadow physical atlas, World Probe Grid sampler.
        poolSizes[3] = { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 }; // Forward: shared g_TLAS.

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 4;
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
        // Set/Pipeline 2: TransparentForward.vert/.frag -- bindings 0..6, 11..17 written here;
        // 7..10 (Phase 3's renderer::VirtualShadowMapPass resources) left for SetVirtualShadowMap().
        // Bindings 11-17 are new (indirect diffuse GI + optional per-material reflection trace +
        // MegaLights point-light shading, see class comment).
        // =====================================================================================
        {
            VkDescriptorSetLayoutBinding bindings[18]{};
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
            // Bindings 11/12 (MegaLights Phase A follow-up + Phase 5 reflection trace, reconciled):
            // unlike 7-10 above (deferred to SetVirtualShadowMap()), the TLAS/light SSBO ARE already
            // available here -- renderer::ClusterRenderPipeline::Init() Init()s renderer::
            // SurfaceCacheRayTracingPass and renderer::MegaLightsPass before this pass specifically
            // so both can be written immediately below, not deferred. `g_TLAS` (11) is shared by
            // MegaLights' own TraceShadowRay AND this pass's own optional reflection TraceHWRT --
            // only ONE acceleration-structure binding for both.
            bindings[11] = { 11, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_TLAS (shared)
            bindings[12] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // MegaLightsSSBO (g_Lights)
            bindings[13] = { 13, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // WorldProbeGridParamsUBO
            bindings[14] = { 14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // g_WorldProbeGrid
            bindings[15] = { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // FallbackVertexBuffer
            bindings[16] = { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // FallbackIndexBuffer
            bindings[17] = { 17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr }; // EntityDrawRangeBuffer

            VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            layoutInfo.bindingCount = 18;
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

            // World Probe Grid sampler + params (bindings 14, 13) -- static addressing/image handle,
            // written once here like every other Init()-time binding (only the grid's own texel
            // CONTENTS change frame to frame, sampled directly through this same view/sampler).
            VkDescriptorImageInfo worldProbeGridInfo{ worldProbes.GetGridSampler(), worldProbes.GetGridView(), VK_IMAGE_LAYOUT_GENERAL };
            VkDescriptorBufferInfo worldProbeGridParamsInfo{ m_WorldProbeGridParamsBuffer.Handle(), 0, m_WorldProbeGridParamsBuffer.Size() };

            // MegaLights light SSBO (binding 12).
            VkDescriptorBufferInfo lightBufferInfo{ lightBuffer, 0, lightBufferSize };

            VkWriteDescriptorSet writes[10]{};
            writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entriesInfo, nullptr };
            writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
            writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
            writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
            writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &wpoGlobalsInfo, nullptr };
            writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
            writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
            writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr };
            writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 13, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &worldProbeGridParamsInfo, nullptr };
            writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 14, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &worldProbeGridInfo, nullptr, nullptr };
            vkUpdateDescriptorSets(m_Device, 10, writes, 0, nullptr);
            // Bindings 7-10 (Phase 3's renderer::VirtualShadowMapPass resources) are intentionally
            // left unwritten here -- SetVirtualShadowMap() writes them once the caller has a
            // VirtualShadowMapPass to bind, same convention as renderer::ClusterResolvePass's own.

            // Binding 11 (shared g_TLAS) -- needs its own pNext chain, issued as a separate
            // vkUpdateDescriptorSets call, same pattern renderer::MegaLightsPass::Init() already uses.
            VkWriteDescriptorSetAccelerationStructureKHR accelWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
            accelWrite.accelerationStructureCount = 1;
            accelWrite.pAccelerationStructures = &tlas;
            VkWriteDescriptorSet accelDescriptorWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            accelDescriptorWrite.pNext = &accelWrite;
            accelDescriptorWrite.dstSet = m_ForwardDescriptorSet;
            accelDescriptorWrite.dstBinding = 11;
            accelDescriptorWrite.descriptorCount = 1;
            accelDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(m_Device, 1, &accelDescriptorWrite, 0, nullptr);

            // Bindings 15-17 (Fallback Mesh vertex/index/drawRange, for the optional reflection
            // trace's HWRT path) -- written manually (not via VulkanUtils::WriteSharedGeometryBindings,
            // which assumes 4 CONSECUTIVE bindings starting with its own TLAS write; binding 11's
            // TLAS is shared/already written above and these 3 aren't adjacent to it here).
            VkDescriptorBufferInfo fallbackVertexInfo{ fallbackVertexBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo fallbackIndexInfo{ fallbackIndexBuffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo drawRangeInfo{ drawRangeBuffer, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet fallbackWrites[3]{};
            fallbackWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fallbackVertexInfo, nullptr };
            fallbackWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 16, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fallbackIndexInfo, nullptr };
            fallbackWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ForwardDescriptorSet, 17, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &drawRangeInfo, nullptr };
            vkUpdateDescriptorSets(m_Device, 3, fallbackWrites, 0, nullptr);

            // 3-set pipeline layout: this pass's own set 0 + traceContext's mesh_sdf_trace (set 1) /
            // surface_cache_sampling (set 2) layouts, fixed here -- same shape as renderer::
            // ReflectionPass::Init()'s own identical 3-set layout (see class comment). A fragment-
            // stage push constant range carries entityCount/traceMode/frameIndex, consumed by both
            // the optional per-material reflection trace and MegaLights' own RIS decorrelation.
            VkDescriptorSetLayout forwardSetLayouts[3] = {
                m_ForwardSetLayout, traceContext.GetMeshSdfTraceSetLayout(), traceContext.GetSurfaceCacheSamplingSetLayout()
            };
            VkPushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(TransparentPushConstants);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pipelineLayoutInfo.setLayoutCount = 3;
            pipelineLayoutInfo.pSetLayouts = forwardSetLayouts;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
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

            // Phase PP3: g_RefractionOffset (attachment 1) -- a plain overwrite (blendEnable=FALSE),
            // NOT alpha-blended like the color attachment above: a distortion vector doesn't
            // "compose" over whatever the previous transparent surface at this pixel wrote the same
            // way color does, and this attachment is CLEARed to (0,0) every frame anyway (see
            // RecordDraw()'s own comment), so a plain overwrite from the last-drawn surface at this
            // pixel is the correct, simplest behavior.
            VkPipelineColorBlendAttachmentState refractionBlendAttachment{};
            refractionBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
            refractionBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState colorBlendAttachments[2] = { colorBlendAttachment, refractionBlendAttachment };
            VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            colorBlending.attachmentCount = 2;
            colorBlending.pAttachments = colorBlendAttachments;

            VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamicState.dynamicStateCount = 2;
            dynamicState.pDynamicStates = dynamicStates;

            VkFormat colorFormats[2] = { colorFormat, kRefractionOffsetFormat };
            VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
            pipelineRendering.colorAttachmentCount = 2;
            pipelineRendering.pColorAttachmentFormats = colorFormats;
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
        m_DrawCountBuffer.Destroy();
        m_ViewParamsBuffer.Destroy();
        m_MaterialParamsBuffer.Destroy();
        m_WorldProbeGridParamsBuffer.Destroy();
        m_RefractionOffsetImage.Destroy();

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
        VkBuffer decompressedIndexPoolBuffer, const maths::vec3& cameraPositionWorld,
        const SceneLights& sceneLights, float globalTimeSeconds,
        const SurfaceCacheTraceContext& traceContext, uint32_t traceMode, uint32_t frameIndex) {
        if (m_ClusterCount == 0) {
            return;
        }

        // --- Upload this frame's view/proj/camera position/sun lighting (Phase 5: was just
        // sunDirection; point lights removed -- MegaLights handles those now, see class comment). ---
        TransparentViewParams viewParams{};
        viewParams.view = view;
        viewParams.proj = proj;
        viewParams.cameraPosX = cameraPositionWorld.x;
        viewParams.cameraPosY = cameraPositionWorld.y;
        viewParams.cameraPosZ = cameraPositionWorld.z;
        viewParams.globalTime = globalTimeSeconds;
        viewParams.sunDirX = sceneLights.sun.direction.x;
        viewParams.sunDirY = sceneLights.sun.direction.y;
        viewParams.sunDirZ = sceneLights.sun.direction.z;
        viewParams.sunIntensity = sceneLights.sun.intensity;
        viewParams.sunColorR = sceneLights.sun.color.x;
        viewParams.sunColorG = sceneLights.sun.color.y;
        viewParams.sunColorB = sceneLights.sun.color.z;
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

        // Phase PP3: g_RefractionOffset undergoes the exact same GENERAL -> COLOR_ATTACHMENT_OPTIMAL
        // dance as colorImage above, batched into the same barrier call -- its own last reader was
        // renderer::PostProcessPass's composite shader (COMPUTE_SHADER/SAMPLED_READ), not a compute
        // write, so its own srcStageMask/srcAccessMask differ from colorImage's.
        VkImageMemoryBarrier2 refractionToAttachment{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        refractionToAttachment.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        refractionToAttachment.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        refractionToAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        refractionToAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        refractionToAttachment.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        refractionToAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        refractionToAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        refractionToAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        refractionToAttachment.image = m_RefractionOffsetImage.Image();
        refractionToAttachment.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkImageMemoryBarrier2 toAttachmentBarriers[2] = { toAttachment, refractionToAttachment };
        VkDependencyInfo toAttachmentDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toAttachmentDependency.imageMemoryBarrierCount = 2;
        toAttachmentDependency.pImageMemoryBarriers = toAttachmentBarriers;
        vkCmdPipelineBarrier2(cmd, &toAttachmentDependency);

        VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView = colorView;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // Preserve the fully-composited opaque scene underneath.
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Phase PP3: g_RefractionOffset -- CLEARed to (0,0) every frame, unlike colorAttachment
        // above: a pixel's distortion must reset to "none" wherever no heat-distortion surface
        // covers it THIS frame (entity motion/rotation/camera movement changes screen coverage
        // frame to frame even though m_ClusterCount itself is static -- see GetRefractionOffsetView()'s
        // own comment).
        VkRenderingAttachmentInfo refractionAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        refractionAttachment.imageView = m_RefractionOffsetImage.View();
        refractionAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        refractionAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        refractionAttachment.clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        refractionAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        // Depth is already VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL by this point in the
        // frame (see renderer::ClusterRenderPipeline::RecordFrame's own ordering) -- no transition,
        // no barrier: this pass only ever reads it (depthWriteEnable=FALSE in the pipeline state
        // above), so there is no write to synchronize against.
        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = depthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo colorAttachments[2] = { colorAttachment, refractionAttachment };
        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, renderExtent };
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 2;
        renderingInfo.pColorAttachments = colorAttachments;
        renderingInfo.pDepthAttachment = &depthAttachment;

        vkCmdBeginRendering(cmd, &renderingInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ForwardPipeline);
        // 3 sets: this pass's own (set 0) + traceContext's mesh_sdf_trace (set 1) / surface_cache_
        // sampling (set 2) -- see Init()'s own comment on why the pipeline layout has 3 sets.
        VkDescriptorSet forwardSets[3] = {
            m_ForwardDescriptorSet, traceContext.GetMeshSdfTraceSet(), traceContext.GetSurfaceCacheSamplingSet()
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ForwardPipelineLayout, 0, 3, forwardSets, 0, nullptr);
        vkCmdBindIndexBuffer(cmd, decompressedIndexPoolBuffer, 0, VK_INDEX_TYPE_UINT32);

        TransparentPushConstants pushConstants{};
        pushConstants.entityCount = traceContext.GetEntityCount();
        pushConstants.traceMode = traceMode;
        pushConstants.frameIndex = frameIndex;
        vkCmdPushConstants(cmd, m_ForwardPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstants), &pushConstants);

        VkViewport viewport{};
        viewport.width = static_cast<float>(renderExtent.width);
        viewport.height = static_cast<float>(renderExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = renderExtent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // *IndirectCount, not the plain indirect draw -- m_ClusterCount can exceed 1 and this
        // device does not enable multiDrawIndirect (vkCmdDrawIndexedIndirect's own drawCount is
        // capped at 1 without it; the opaque pipeline already uses this exact variant for the same
        // reason, see ClusterHardwareRasterPass::RecordDraw). m_DrawCountBuffer holds the same
        // fixed m_ClusterCount value written once at Init() -- see that buffer's own comment.
        vkCmdDrawIndexedIndirectCount(cmd, m_IndirectCommandsBuffer.Handle(), 0,
            m_DrawCountBuffer.Handle(), 0, m_ClusterCount, sizeof(VkDrawIndexedIndirectCommand));

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

        // Phase PP3: g_RefractionOffset back to GENERAL too, ready for renderer::PostProcessPass's
        // composite shader to sample this same frame (COMPUTE_SHADER/SAMPLED_READ -- it runs later
        // in renderer::ClusterRenderPipeline::RecordFrame's own ordering, after TAA/Bloom).
        VkImageMemoryBarrier2 refractionToGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        refractionToGeneral.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        refractionToGeneral.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        refractionToGeneral.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        refractionToGeneral.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        refractionToGeneral.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        refractionToGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        refractionToGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        refractionToGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        refractionToGeneral.image = m_RefractionOffsetImage.Image();
        refractionToGeneral.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkImageMemoryBarrier2 toGeneralBarriers[2] = { toGeneral, refractionToGeneral };
        VkDependencyInfo toGeneralDependency{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toGeneralDependency.imageMemoryBarrierCount = 2;
        toGeneralDependency.pImageMemoryBarriers = toGeneralBarriers;
        vkCmdPipelineBarrier2(cmd, &toGeneralDependency);
    }

}
