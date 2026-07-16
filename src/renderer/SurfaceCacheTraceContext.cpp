#include "renderer/SurfaceCacheTraceContext.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>
#include <unordered_map>

#include "core/Logger.h"
#include "renderer/VulkanUtils.h"
#include "geometry/ClusterFormat.h"

namespace renderer {

    namespace {

        // std430-exact mirror of mesh_sdf_trace.glsl's EntityInfo struct -- see that file's own
        // struct-packing comment. Plain scalars only (no vec3), so this uploads byte-for-byte with
        // no std430 vec-alignment surprises.
        struct GpuEntityInfo {
            float volumeMinX = 0.0f, volumeMinY = 0.0f, volumeMinZ = 0.0f;
            float voxelSize = 0.0f;
            uint32_t resolution = 0;
            uint32_t entityID = 0;
            uint32_t firstCardIndex = 0;
            uint32_t cardCount = 0;
        };
        static_assert(sizeof(GpuEntityInfo) == 32, "GpuEntityInfo must match mesh_sdf_trace.glsl's EntityInfo layout exactly");

    } // namespace

    bool SurfaceCacheTraceContext::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const GlobalSDFPass& globalSDF, const SurfaceCachePass& surfaceCache) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Gather traced-entity info (CPU side): clamp to kMaxTracedEntities. Persisted
        // in m_TracedEntityInfos/m_TracedEntities (unlike a plain STEP-1 local) so RefreshCardTable()
        // can rebuild the per-entity card grouping every frame without re-querying GlobalSDFPass --
        // see both members' own header comments for why that grouping (not this list itself) is
        // the part that needs re-deriving.
        // =====================================================================================
        m_TracedEntityInfos = globalSDF.GetTracedEntityInfos();
        if (m_TracedEntityInfos.size() > kMaxTracedEntities) {
            LOG_ERROR(std::format(
                "[SurfaceCacheTraceContext] {} traced entities exceeds kMaxTracedEntities={}; truncating -- "
                "raise kMaxTracedEntities in both SurfaceCacheTraceContext.h and mesh_sdf_trace.glsl.",
                m_TracedEntityInfos.size(), kMaxTracedEntities));
            m_TracedEntityInfos.resize(kMaxTracedEntities);
        }
        m_EntityCount = static_cast<uint32_t>(m_TracedEntityInfos.size());

        const std::unordered_map<uint32_t, SurfaceCachePass::EntityDrawRange>& entityRanges = surfaceCache.GetEntityRanges();
        m_TracedEntities.clear();
        m_TracedEntities.reserve(m_EntityCount);
        for (const GlobalSDFPass::TracedEntityInfo& entity : m_TracedEntityInfos) {
            TracedEntity traced{};
            traced.entityID = entity.entityID;
            auto rangeIt = entityRanges.find(entity.entityID);
            if (rangeIt != entityRanges.end()) {
                traced.drawRange = rangeIt->second;
            }
            m_TracedEntities.push_back(traced);
        }

        LOG_INFO(std::format("[SurfaceCacheTraceContext] Traced entity count: {}.", m_EntityCount));

        // =====================================================================================
        // STEP 2 -- Allocate the 3 host-visible SSBOs at their WORST-CASE capacity (every card
        // resident at once, for the card/card-index buffers) so RefreshCardTable() never needs to
        // recreate them -- only memcpy new contents in, every frame, cheaply. Populated below via
        // the first RefreshCardTable() call (every card starts non-resident, see
        // renderer::SurfaceCachePass::Init()'s own comment, so this Init()-time population is
        // trivially empty -- the caller's render loop is expected to call RefreshCardTable() again
        // after its first real SurfaceCachePass::UpdateVisibility() call, per that method's own
        // documented contract).
        // =====================================================================================
        const std::vector<geometry::SurfaceCacheCardEntry>& cards = surfaceCache.GetCards();
        const VkDeviceSize cardCapacityBytes = static_cast<VkDeviceSize>(std::max<size_t>(cards.size(), 1)) * sizeof(geometry::SurfaceCacheCardEntry);
        const VkDeviceSize cardIndexCapacityBytes = static_cast<VkDeviceSize>(std::max<size_t>(cards.size(), 1)) * sizeof(uint32_t);
        const VkDeviceSize entityInfoBytes = static_cast<VkDeviceSize>(std::max<uint32_t>(m_EntityCount, 1)) * sizeof(GpuEntityInfo);

        m_CardBuffer.Create(allocator, cardCapacityBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        m_EntityCardIndexBuffer.Create(allocator, cardIndexCapacityBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        m_EntityInfoBuffer.Create(allocator, entityInfoBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);
        std::memset(m_CardBuffer.MappedData(), 0, static_cast<size_t>(cardCapacityBytes));
        std::memset(m_EntityCardIndexBuffer.MappedData(), 0, static_cast<size_t>(cardIndexCapacityBytes));
        std::memset(m_EntityInfoBuffer.MappedData(), 0, static_cast<size_t>(entityInfoBytes));

        // =====================================================================================
        // STEP 3 -- 1x1x1 dummy Mesh SDF volume (value kDummyFarDistance everywhere), used to pad
        // out g_EntitySDF's unused [entityCount, kMaxTracedEntities) array slots so every one of
        // the fixed kMaxTracedEntities descriptor slots is always bound to a valid image view.
        // =====================================================================================
        {
            VkImageCreateInfo dummyImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
            dummyImageInfo.imageType = VK_IMAGE_TYPE_3D;
            dummyImageInfo.format = VK_FORMAT_R32_SFLOAT;
            dummyImageInfo.extent = { 1, 1, 1 };
            dummyImageInfo.mipLevels = 1;
            dummyImageInfo.arrayLayers = 1;
            dummyImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            dummyImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            dummyImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            dummyImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            dummyImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo gpuOnlyAlloc{};
            gpuOnlyAlloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            VK_CHECK(vmaCreateImage(allocator, &dummyImageInfo, &gpuOnlyAlloc, &m_DummySdfImage, &m_DummySdfAllocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_DummySdfImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            viewInfo.format = VK_FORMAT_R32_SFLOAT;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_DummySdfView));

            VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.minLod = 0.0f;
            samplerInfo.maxLod = 0.0f;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_EntitySdfSampler));

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocResultInfo{};
            VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            stagingInfo.size = sizeof(float);
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagingAllocInfo{};
            stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VK_CHECK(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));
            *static_cast<float*>(stagingAllocResultInfo.pMappedData) = kDummyFarDistance;

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkImageMemoryBarrier2 toDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                toDst.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                toDst.srcAccessMask = 0;
                toDst.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                toDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toDst.image = m_DummySdfImage;
                toDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                VkDependencyInfo toDstDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                toDstDep.imageMemoryBarrierCount = 1;
                toDstDep.pImageMemoryBarriers = &toDst;
                vkCmdPipelineBarrier2(cmd, &toDstDep);

                VkBufferImageCopy copyRegion{};
                copyRegion.bufferOffset = 0;
                copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                copyRegion.imageExtent = { 1, 1, 1 };
                vkCmdCopyBufferToImage(cmd, stagingBuffer, m_DummySdfImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

                VkImageMemoryBarrier2 toRead{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                toRead.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                toRead.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                toRead.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                toRead.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                toRead.image = m_DummySdfImage;
                toRead.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                VkDependencyInfo toReadDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                toReadDep.imageMemoryBarrierCount = 1;
                toReadDep.pImageMemoryBarriers = &toRead;
                vkCmdPipelineBarrier2(cmd, &toReadDep);
            });

            vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        }

        // =====================================================================================
        // STEP 4 -- Descriptor set layouts (set 1: mesh SDF trace scene, set 2: surface cache
        // sampling -- see both shared .glsl files' own binding-convention header comments) +
        // pool + allocation + writes. Buffer descriptors reference VK_WHOLE_SIZE of each already-
        // allocated (STEP 2) buffer, so they remain valid across every future RefreshCardTable()
        // call (which only memcpy's new contents, never recreates the buffers).
        // =====================================================================================
        VkDescriptorSetLayoutBinding meshSdfBindings[2]{};
        meshSdfBindings[0].binding = 0;
        meshSdfBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        meshSdfBindings[0].descriptorCount = 1;
        meshSdfBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        meshSdfBindings[1].binding = 1;
        meshSdfBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        meshSdfBindings[1].descriptorCount = kMaxTracedEntities;
        meshSdfBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo meshSdfLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        meshSdfLayoutInfo.bindingCount = 2;
        meshSdfLayoutInfo.pBindings = meshSdfBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &meshSdfLayoutInfo, nullptr, &m_MeshSdfTraceSetLayout));

        VkDescriptorSetLayoutBinding samplingBindings[3]{};
        samplingBindings[0].binding = 0;
        samplingBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        samplingBindings[0].descriptorCount = 1;
        samplingBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        samplingBindings[1].binding = 1;
        samplingBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplingBindings[1].descriptorCount = 1;
        samplingBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        samplingBindings[2].binding = 2;
        samplingBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        samplingBindings[2].descriptorCount = 1;
        samplingBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo samplingLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        samplingLayoutInfo.bindingCount = 3;
        samplingLayoutInfo.pBindings = samplingBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &samplingLayoutInfo, nullptr, &m_SurfaceCacheSamplingSetLayout));

        VkDescriptorPoolSize poolSizes[2] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxTracedEntities + 1 }
        };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout layoutsToAllocate[2] = { m_MeshSdfTraceSetLayout, m_SurfaceCacheSamplingSetLayout };
        VkDescriptorSet allocatedSets[2] = {};
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 2;
        setAllocInfo.pSetLayouts = layoutsToAllocate;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, allocatedSets));
        m_MeshSdfTraceSet = allocatedSets[0];
        m_SurfaceCacheSamplingSet = allocatedSets[1];

        // set 1 writes: EntityInfo SSBO + the kMaxTracedEntities-slot sampler3D array (real entity
        // SDF views for [0, entityCount), the dummy "far" volume for [entityCount, kMaxTracedEntities)).
        std::vector<VkDescriptorImageInfo> sdfImageInfos(kMaxTracedEntities);
        for (uint32_t i = 0; i < kMaxTracedEntities; ++i) {
            sdfImageInfos[i].sampler = m_EntitySdfSampler;
            sdfImageInfos[i].imageView = (i < m_EntityCount) ? m_TracedEntityInfos[i].sdfView : m_DummySdfView;
            sdfImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkDescriptorBufferInfo entityInfoBufferInfo{ m_EntityInfoBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo cardBufferInfo{ m_CardBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo cardIndexBufferInfo{ m_EntityCardIndexBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo radianceImageInfo{ surfaceCache.GetAtlasSampler(), surfaceCache.GetRadianceView(), VK_IMAGE_LAYOUT_GENERAL };

        VkWriteDescriptorSet writes[5]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[0].dstSet = m_MeshSdfTraceSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &entityInfoBufferInfo;

        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[1].dstSet = m_MeshSdfTraceSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = kMaxTracedEntities;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = sdfImageInfos.data();

        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[2].dstSet = m_SurfaceCacheSamplingSet;
        writes[2].dstBinding = 0;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &cardBufferInfo;

        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[3].dstSet = m_SurfaceCacheSamplingSet;
        writes[3].dstBinding = 1;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &radianceImageInfo;

        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[4].dstSet = m_SurfaceCacheSamplingSet;
        writes[4].dstBinding = 2;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[4].pBufferInfo = &cardIndexBufferInfo;

        vkUpdateDescriptorSets(m_Device, 5, writes, 0, nullptr);

        // Populate the card table/indices for whatever residency state exists right now (every
        // card non-resident, in practice, this early -- see this function's own STEP 2 comment).
        RefreshCardTable(surfaceCache);

        return true;
    }

    void SurfaceCacheTraceContext::RefreshCardTable(const SurfaceCachePass& surfaceCache) {
        const std::vector<geometry::SurfaceCacheCardEntry>& cards = surfaceCache.GetCards();

        // Re-copy the full card array (atlasOffset/uvMin/uvMax may have moved -- see
        // renderer::SurfaceCachePass's own "Dynamic atlas residency" comment) -- always exactly
        // cards.size() elements, matching the capacity m_CardBuffer was allocated at in Init().
        if (!cards.empty()) {
            std::memcpy(m_CardBuffer.MappedData(), cards.data(), cards.size() * sizeof(geometry::SurfaceCacheCardEntry));
        }

        // Rebuild the per-entity RESIDENT card grouping -- a non-resident card's atlasOffset is
        // meaningless (stale or about to be overwritten by whatever card gets placed there next),
        // so it must never appear in a trace shader's reachable [firstCardIndex, firstCardIndex +
        // cardCount) range (see surface_cache_sampling.glsl's own indirection-table comment).
        std::unordered_map<uint32_t, std::vector<uint32_t>> cardsByEntity;
        cardsByEntity.reserve(m_TracedEntityInfos.size());
        for (uint32_t i = 0; i < cards.size(); ++i) {
            if (surfaceCache.IsCardResident(i)) {
                cardsByEntity[cards[i].entityID].push_back(i);
            }
        }

        std::vector<uint32_t> hostCardIndices;
        std::vector<GpuEntityInfo> hostEntities;
        hostEntities.reserve(m_TracedEntityInfos.size());
        for (const GlobalSDFPass::TracedEntityInfo& entity : m_TracedEntityInfos) {
            GpuEntityInfo info{};
            info.volumeMinX = entity.volumeMin.x;
            info.volumeMinY = entity.volumeMin.y;
            info.volumeMinZ = entity.volumeMin.z;
            info.voxelSize = entity.voxelSize;
            info.resolution = entity.resolution;
            info.entityID = entity.entityID;

            auto cardIt = cardsByEntity.find(entity.entityID);
            if (cardIt != cardsByEntity.end()) {
                info.firstCardIndex = static_cast<uint32_t>(hostCardIndices.size());
                info.cardCount = static_cast<uint32_t>(cardIt->second.size());
                hostCardIndices.insert(hostCardIndices.end(), cardIt->second.begin(), cardIt->second.end());
            }
            hostEntities.push_back(info);
        }

        if (!hostEntities.empty()) {
            std::memcpy(m_EntityInfoBuffer.MappedData(), hostEntities.data(), hostEntities.size() * sizeof(GpuEntityInfo));
        }
        if (!hostCardIndices.empty()) {
            std::memcpy(m_EntityCardIndexBuffer.MappedData(), hostCardIndices.data(), hostCardIndices.size() * sizeof(uint32_t));
        }
    }

    void SurfaceCacheTraceContext::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_MeshSdfTraceSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_MeshSdfTraceSetLayout, nullptr);
            if (m_SurfaceCacheSamplingSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SurfaceCacheSamplingSetLayout, nullptr);
            if (m_EntitySdfSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_EntitySdfSampler, nullptr);
            if (m_DummySdfView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_DummySdfView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            if (m_DummySdfImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_DummySdfImage, m_DummySdfAllocation);
        }
        m_EntityInfoBuffer.Destroy();
        m_CardBuffer.Destroy();
        m_EntityCardIndexBuffer.Destroy();

        m_DescriptorPool = VK_NULL_HANDLE;
        m_MeshSdfTraceSetLayout = VK_NULL_HANDLE;
        m_SurfaceCacheSamplingSetLayout = VK_NULL_HANDLE;
        m_MeshSdfTraceSet = VK_NULL_HANDLE;
        m_SurfaceCacheSamplingSet = VK_NULL_HANDLE;
        m_EntitySdfSampler = VK_NULL_HANDLE;
        m_DummySdfImage = VK_NULL_HANDLE; m_DummySdfAllocation = VK_NULL_HANDLE; m_DummySdfView = VK_NULL_HANDLE;
        m_EntityCount = 0;
        m_TracedEntities.clear();
        m_TracedEntityInfos.clear();
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

}
