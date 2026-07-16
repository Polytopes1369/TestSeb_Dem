#include "renderer/SDFRayMarchPass.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/MeshSDFGenerator.h"
#include "io/CacheFileManager.h"

namespace renderer {

    namespace {

        // Mirrors every other pass's own copy -- duplicated rather than shared, see this class's
        // own header comment on self-containment.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("SDFRayMarchPass: failed to open SPIR-V file: " + filename);
            }
            size_t fileSize = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            file.close();
            return buffer;
        }

        VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code) {
            VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &module));
            return module;
        }

        void TransitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect,
            VkImageLayout oldLayout, VkImageLayout newLayout,
            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
            VkImageMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask = srcStage;
            barrier.srcAccessMask = srcAccess;
            barrier.dstStageMask = dstStage;
            barrier.dstAccessMask = dstAccess;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = { aspect, 0, 1, 0, 1 };
            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        // Byte-for-byte std430 layout match for EntitySDFParamsGPU in SDFRayMarch.comp -- flat
        // scalar fields throughout (see GlobalSDFCompositePC's own comment on why this codebase
        // always mirrors a GPU struct this way).
        struct EntitySDFParamsGPU {
            float volumeMin[3];
            float voxelSize;
            int32_t resolution;
            uint32_t entityID;
            int32_t _pad0[2];
        };
        static_assert(sizeof(EntitySDFParamsGPU) == 32,
            "EntitySDFParamsGPU must match SDFRayMarch.comp's EntitySDFParams exactly");

        // Byte-for-byte push-constant layout match for SDFRayMarchPC in SDFRayMarch.comp.
        struct SDFRayMarchPC {
            float cameraPosition[3];
            float tanHalfFovY;
            float cameraForward[3];
            float aspectRatio;
            float cameraRight[3];
            float nearZ;
            float cameraUp[3];
            float farZ;
            float levelVoxelSize[GlobalSDFPass::kLevelCount];
            int32_t levelCenterVoxel[GlobalSDFPass::kLevelCount * 3];
            int32_t clipmapResolution;
            int32_t rootNodeIndex;
            int32_t entitySamplerCount;
            int32_t _pad0;
        };
        static_assert(sizeof(SDFRayMarchPC) == 144,
            "SDFRayMarchPC must match SDFRayMarch.comp's push_constant block exactly");

    } // namespace

    bool SDFRayMarchPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const std::filesystem::path& cacheFilePath, uint32_t outputWidth, uint32_t outputHeight) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_OutputWidth = outputWidth;
        m_OutputHeight = outputHeight;

        // =====================================================================================
        // STEP 1 -- Read every fallback mesh's geometry and build one CPU-side Mesh SDF per
        // entity at the surface-detail-oriented default resolution (see class comment on why NOT
        // GlobalSDFPass's own, coarser, per-entity SDFs). Entities beyond kMaxEntitySDFs are
        // dropped (logged) -- see class comment on the fixed-size entity sampler array.
        // =====================================================================================
        geometry::CacheFileManager cacheManager;
        geometry::CacheFileHeader header{};
        if (!cacheManager.ReadHeader(cacheFilePath, header)) {
            LOG_ERROR(std::format("[SDFRayMarchPass] Failed to read cache header '{}'.", cacheFilePath.string()));
            return false;
        }
        std::vector<geometry::FallbackMeshIndexEntry> fallbackTable;
        if (!cacheManager.ReadFallbackMeshTable(cacheFilePath, header, fallbackTable)) {
            LOG_ERROR("[SDFRayMarchPass] Failed to read the fallback-mesh table.");
            return false;
        }

        struct PendingUpload {
            uint32_t entityID = 0;
            std::vector<float> decodedGrid; // resolution^3, voxel-linear (x fastest), world-unit distances.
            maths::vec3 volumeMin{};
            float voxelSize = 0.0f;
            uint32_t resolution = 0;
        };
        std::vector<PendingUpload> uploads;
        // Same order as `uploads` -- the exact sub-list geometry::BuildEntityBVH() builds over, so
        // its leaf indices land on the same slots as `uploads`/the eventual entity sampler array.
        std::vector<geometry::FallbackMeshIndexEntry> survivingEntries;
        uploads.reserve(std::min<size_t>(fallbackTable.size(), kMaxEntitySDFs));
        survivingEntries.reserve(uploads.capacity());

        uint32_t truncatedCount = 0;
        for (const geometry::FallbackMeshIndexEntry& entry : fallbackTable) {
            if (uploads.size() >= kMaxEntitySDFs) {
                ++truncatedCount;
                continue;
            }

            std::vector<geometry::FallbackVertex> vertices;
            std::vector<uint32_t> indices;
            if (!cacheManager.ReadFallbackMeshGeometry(cacheFilePath, entry, vertices, indices)) {
                LOG_ERROR(std::format(
                    "[SDFRayMarchPass] Failed to read fallback mesh geometry for entityID={}.", entry.entityID));
                return false;
            }
            if (vertices.empty() || indices.size() < 3) {
                continue;
            }

            std::vector<maths::vec3> positions;
            positions.reserve(vertices.size());
            for (const geometry::FallbackVertex& v : vertices) {
                positions.push_back(maths::vec3{ v.position[0], v.position[1], v.position[2] });
            }

            geometry::MeshSDF sdf = geometry::BuildMeshSDF(positions, indices); // Default resolution == geometry::kMeshSDFResolution.
            if (sdf.resolution == 0) {
                LOG_WARNING(std::format(
                    "[SDFRayMarchPass] BuildMeshSDF produced an empty SDF for entityID={}; skipping.", entry.entityID));
                continue;
            }

            PendingUpload upload;
            upload.entityID = entry.entityID;
            upload.volumeMin = sdf.volumeMin;
            upload.voxelSize = sdf.voxelSize;
            upload.resolution = sdf.resolution;
            upload.decodedGrid.resize(static_cast<size_t>(sdf.resolution) * sdf.resolution * sdf.resolution);
            for (uint32_t z = 0; z < sdf.resolution; ++z) {
                for (uint32_t y = 0; y < sdf.resolution; ++y) {
                    for (uint32_t x = 0; x < sdf.resolution; ++x) {
                        upload.decodedGrid[(static_cast<size_t>(z) * sdf.resolution + y) * sdf.resolution + x] =
                            geometry::DecodeMeshSDFVoxel(sdf, x, y, z);
                    }
                }
            }
            uploads.push_back(std::move(upload));
            survivingEntries.push_back(entry);
        }

        if (truncatedCount > 0) {
            LOG_WARNING(std::format(
                "[SDFRayMarchPass] Scene has more than kMaxEntitySDFs={} fallback meshes; {} entity/entities dropped from ray marching.",
                kMaxEntitySDFs, truncatedCount));
        }

        geometry::EntityBVH bvh = geometry::BuildEntityBVH(survivingEntries);
        m_RootNodeIndex = bvh.nodes.empty() ? -1 : 0;

        LOG_INFO(std::format(
            "[SDFRayMarchPass] Built {} entity Mesh SDF(s) at {}^3 and a {}-node BVH for ray marching.",
            uploads.size(), uploads.empty() ? 0u : uploads[0].resolution, bvh.nodes.size()));

        // =====================================================================================
        // STEP 2 -- GPU resources: per-entity sampled 3D SDF textures, the 1x1x1 placeholder
        // texture, BVH/leaf-index/entity-params storage buffers, the output storage image, both
        // samplers, descriptor set layouts/pool/sets, the ray march compute pipeline.
        // =====================================================================================
        m_Entities.resize(uploads.size());

        VkImageCreateInfo entityImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        entityImageInfo.imageType = VK_IMAGE_TYPE_3D;
        entityImageInfo.format = GlobalSDFPass::kClipmapFormat; // Plain R32_SFLOAT, matching GlobalSDFPass's own entity SDF textures.
        entityImageInfo.mipLevels = 1;
        entityImageInfo.arrayLayers = 1;
        entityImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        entityImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        entityImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        entityImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        entityImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo gpuOnlyAlloc{};
        gpuOnlyAlloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        for (size_t i = 0; i < uploads.size(); ++i) {
            EntitySDF& entitySdf = m_Entities[i];
            entitySdf.volumeMin = uploads[i].volumeMin;
            entitySdf.voxelSize = uploads[i].voxelSize;
            entitySdf.resolution = uploads[i].resolution;
            entitySdf.entityID = uploads[i].entityID;

            entityImageInfo.extent = { entitySdf.resolution, entitySdf.resolution, entitySdf.resolution };
            VK_CHECK(vmaCreateImage(allocator, &entityImageInfo, &gpuOnlyAlloc, &entitySdf.image, &entitySdf.allocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = entitySdf.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            viewInfo.format = GlobalSDFPass::kClipmapFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &entitySdf.view));
        }

        VkImageCreateInfo placeholderImageInfo = entityImageInfo;
        placeholderImageInfo.extent = { 1, 1, 1 };
        VK_CHECK(vmaCreateImage(allocator, &placeholderImageInfo, &gpuOnlyAlloc, &m_PlaceholderImage, &m_PlaceholderAllocation, nullptr));
        VkImageViewCreateInfo placeholderViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        placeholderViewInfo.image = m_PlaceholderImage;
        placeholderViewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        placeholderViewInfo.format = GlobalSDFPass::kClipmapFormat;
        placeholderViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &placeholderViewInfo, nullptr, &m_PlaceholderView));

        VkImageCreateInfo outputImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        outputImageInfo.imageType = VK_IMAGE_TYPE_2D;
        outputImageInfo.format = kOutputFormat;
        outputImageInfo.extent = { m_OutputWidth, m_OutputHeight, 1 };
        outputImageInfo.mipLevels = 1;
        outputImageInfo.arrayLayers = 1;
        outputImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        outputImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        outputImageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        outputImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        outputImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vmaCreateImage(allocator, &outputImageInfo, &gpuOnlyAlloc, &m_OutputImage, &m_OutputAllocation, nullptr));

        VkImageViewCreateInfo outputViewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        outputViewInfo.image = m_OutputImage;
        outputViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        outputViewInfo.format = kOutputFormat;
        outputViewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &outputViewInfo, nullptr, &m_OutputView));

        VkSamplerCreateInfo entitySamplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        entitySamplerInfo.magFilter = VK_FILTER_LINEAR;
        entitySamplerInfo.minFilter = VK_FILTER_LINEAR;
        entitySamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        entitySamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        entitySamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        entitySamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        entitySamplerInfo.minLod = 0.0f;
        entitySamplerInfo.maxLod = 0.0f;
        entitySamplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &entitySamplerInfo, nullptr, &m_EntitySampler));

        VkSamplerCreateInfo clipmapSamplerInfo = entitySamplerInfo;
        // Nearest -- see SetGlobalSDFViews()'s own comment on why a toroidally-wrapped clipmap must
        // not be linearly filtered (it would blend across the wrap seam).
        clipmapSamplerInfo.magFilter = VK_FILTER_NEAREST;
        clipmapSamplerInfo.minFilter = VK_FILTER_NEAREST;
        VK_CHECK(vkCreateSampler(m_Device, &clipmapSamplerInfo, nullptr, &m_ClipmapSampler));

        // --- Storage buffers: BVH nodes, leaf entity indices, entity params -- each at least 1
        // element even for an empty scene, so the descriptor set always has a valid buffer bound
        // (SDFRayMarch.comp's push-constant rootNodeIndex == -1 then simply skips all BVH work). ---
        const size_t nodeCount = std::max<size_t>(bvh.nodes.size(), 1);
        const size_t leafIndexCount = std::max<size_t>(bvh.entityIndices.size(), 1);
        const size_t entityParamCount = std::max<size_t>(uploads.size(), 1);

        m_BVHNodeBuffer.Create(allocator, nodeCount * sizeof(geometry::BVHNode),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_LeafEntityIndexBuffer.Create(allocator, leafIndexCount * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        m_EntityParamsBuffer.Create(allocator, entityParamCount * sizeof(EntitySDFParamsGPU),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        std::vector<EntitySDFParamsGPU> entityParams(entityParamCount);
        for (size_t i = 0; i < uploads.size(); ++i) {
            EntitySDFParamsGPU& p = entityParams[i];
            p.volumeMin[0] = uploads[i].volumeMin.x;
            p.volumeMin[1] = uploads[i].volumeMin.y;
            p.volumeMin[2] = uploads[i].volumeMin.z;
            p.voxelSize = uploads[i].voxelSize;
            p.resolution = static_cast<int32_t>(uploads[i].resolution);
            p.entityID = uploads[i].entityID;
        }

        // --- Descriptor set layouts. ---
        VkDescriptorSetLayoutBinding entityBindings[4]{};
        entityBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        entityBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        entityBindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        entityBindings[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxEntitySDFs, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo entitySetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        entitySetLayoutInfo.bindingCount = 4;
        entitySetLayoutInfo.pBindings = entityBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &entitySetLayoutInfo, nullptr, &m_EntitySetLayout));

        VkDescriptorSetLayoutBinding clipmapBindings[2]{};
        clipmapBindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, GlobalSDFPass::kLevelCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        clipmapBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
        VkDescriptorSetLayoutCreateInfo clipmapSetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        clipmapSetLayoutInfo.bindingCount = 2;
        clipmapSetLayoutInfo.pBindings = clipmapBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &clipmapSetLayoutInfo, nullptr, &m_ClipmapSetLayout));

        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxEntitySDFs + GlobalSDFPass::kLevelCount };
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo entitySetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        entitySetAllocInfo.descriptorPool = m_DescriptorPool;
        entitySetAllocInfo.descriptorSetCount = 1;
        entitySetAllocInfo.pSetLayouts = &m_EntitySetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &entitySetAllocInfo, &m_EntitySet));

        VkDescriptorSetAllocateInfo clipmapSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        clipmapSetAllocInfo.descriptorPool = m_DescriptorPool;
        clipmapSetAllocInfo.descriptorSetCount = 1;
        clipmapSetAllocInfo.pSetLayouts = &m_ClipmapSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &clipmapSetAllocInfo, &m_ClipmapSet));

        VkDescriptorBufferInfo bvhNodeBufferInfo{ m_BVHNodeBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo leafIndexBufferInfo{ m_LeafEntityIndexBuffer.Handle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo entityParamsBufferInfo{ m_EntityParamsBuffer.Handle(), 0, VK_WHOLE_SIZE };

        std::vector<VkDescriptorImageInfo> entitySamplerInfos(kMaxEntitySDFs);
        for (uint32_t i = 0; i < kMaxEntitySDFs; ++i) {
            entitySamplerInfos[i].sampler = m_EntitySampler;
            entitySamplerInfos[i].imageView = (i < m_Entities.size()) ? m_Entities[i].view : m_PlaceholderView;
            entitySamplerInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkWriteDescriptorSet entityWrites[4]{};
        entityWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EntitySet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bvhNodeBufferInfo, nullptr };
        entityWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EntitySet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &leafIndexBufferInfo, nullptr };
        entityWrites[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EntitySet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityParamsBufferInfo, nullptr };
        entityWrites[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_EntitySet, 3, 0, kMaxEntitySDFs, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, entitySamplerInfos.data(), nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 4, entityWrites, 0, nullptr);

        // Output image binding (set 1, binding 1) is fixed for this pass's whole lifetime -- write
        // it now. Binding 0 (the 4 Global SDF clipmap views) is deferred to SetGlobalSDFViews().
        VkDescriptorImageInfo outputImageInfo2{};
        outputImageInfo2.imageView = m_OutputView;
        outputImageInfo2.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet outputWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        outputWrite.dstSet = m_ClipmapSet;
        outputWrite.dstBinding = 1;
        outputWrite.descriptorCount = 1;
        outputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        outputWrite.pImageInfo = &outputImageInfo2;
        vkUpdateDescriptorSets(m_Device, 1, &outputWrite, 0, nullptr);

        // =====================================================================================
        // STEP 3 -- One-time setup command buffer: upload every entity's decoded SDF grid + the
        // placeholder's single far-value texel + the 3 storage buffers, then transition every
        // image to its permanent layout.
        // =====================================================================================
        {
            VkDeviceSize totalUploadBytes = sizeof(float); // The placeholder's own single texel.
            for (const PendingUpload& upload : uploads) {
                totalUploadBytes += static_cast<VkDeviceSize>(upload.decodedGrid.size()) * sizeof(float);
            }
            const VkDeviceSize bvhNodeBytes = nodeCount * sizeof(geometry::BVHNode);
            const VkDeviceSize leafIndexBytes = leafIndexCount * sizeof(uint32_t);
            const VkDeviceSize entityParamsBytes = entityParamCount * sizeof(EntitySDFParamsGPU);
            totalUploadBytes += bvhNodeBytes + leafIndexBytes + entityParamsBytes;

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocResultInfo{};
            VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            stagingInfo.size = totalUploadBytes;
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagingAllocInfo{};
            stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VK_CHECK(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));

            size_t writeOffset = 0;
            auto stageCopy = [&](const void* data, size_t bytes) {
                std::memcpy(static_cast<char*>(stagingAllocResultInfo.pMappedData) + writeOffset, data, bytes);
                const size_t offset = writeOffset;
                writeOffset += bytes;
                return static_cast<VkDeviceSize>(offset);
                };

            std::vector<VkDeviceSize> entityGridOffsets(uploads.size());
            for (size_t i = 0; i < uploads.size(); ++i) {
                entityGridOffsets[i] = stageCopy(uploads[i].decodedGrid.data(), uploads[i].decodedGrid.size() * sizeof(float));
            }
            const float farValueTexel = GlobalSDFPass::kFarValue;
            const VkDeviceSize placeholderOffset = stageCopy(&farValueTexel, sizeof(float));
            const VkDeviceSize bvhNodeOffset = stageCopy(bvh.nodes.empty() ? nullptr : bvh.nodes.data(), bvh.nodes.empty() ? 0 : bvhNodeBytes);
            const VkDeviceSize leafIndexOffset = stageCopy(bvh.entityIndices.empty() ? nullptr : bvh.entityIndices.data(), bvh.entityIndices.empty() ? 0 : leafIndexBytes);
            const VkDeviceSize entityParamsOffset = stageCopy(entityParams.data(), entityParamsBytes);

            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            for (size_t i = 0; i < uploads.size(); ++i) {
                TransitionImageLayout(cmd, m_Entities[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

                VkBufferImageCopy copyRegion{};
                copyRegion.bufferOffset = entityGridOffsets[i];
                copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                copyRegion.imageExtent = { m_Entities[i].resolution, m_Entities[i].resolution, m_Entities[i].resolution };
                vkCmdCopyBufferToImage(cmd, stagingBuffer, m_Entities[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

                TransitionImageLayout(cmd, m_Entities[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }

            TransitionImageLayout(cmd, m_PlaceholderImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            VkBufferImageCopy placeholderCopy{};
            placeholderCopy.bufferOffset = placeholderOffset;
            placeholderCopy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            placeholderCopy.imageExtent = { 1, 1, 1 };
            vkCmdCopyBufferToImage(cmd, stagingBuffer, m_PlaceholderImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &placeholderCopy);
            TransitionImageLayout(cmd, m_PlaceholderImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            VkBufferCopy bvhNodeCopy{ bvhNodeOffset, 0, bvhNodeBytes };
            vkCmdCopyBuffer(cmd, stagingBuffer, m_BVHNodeBuffer.Handle(), 1, &bvhNodeCopy);
            VkBufferCopy leafIndexCopy{ leafIndexOffset, 0, leafIndexBytes };
            vkCmdCopyBuffer(cmd, stagingBuffer, m_LeafEntityIndexBuffer.Handle(), 1, &leafIndexCopy);
            VkBufferCopy entityParamsCopy{ entityParamsOffset, 0, entityParamsBytes };
            vkCmdCopyBuffer(cmd, stagingBuffer, m_EntityParamsBuffer.Handle(), 1, &entityParamsCopy);

            VkMemoryBarrier2 bufferUploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            bufferUploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            bufferUploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            bufferUploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bufferUploadBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo bufferUploadDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            bufferUploadDep.memoryBarrierCount = 1;
            bufferUploadDep.pMemoryBarriers = &bufferUploadBarrier;
            vkCmdPipelineBarrier2(cmd, &bufferUploadDep);

            TransitionImageLayout(cmd, m_OutputImage, VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));

            vkFreeCommandBuffers(m_Device, commandPool, 1, &cmd);
            vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
        }

        // =====================================================================================
        // STEP 4 -- Ray march compute pipeline.
        // =====================================================================================
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SDFRayMarchPC);

        VkDescriptorSetLayout setLayouts[2] = { m_EntitySetLayout, m_ClipmapSetLayout };
        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 2;
        layoutInfo.pSetLayouts = setLayouts;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        std::vector<char> compCode = ReadShaderFile("shaders/SDFRayMarch.comp.spv");
        VkShaderModule compModule = CreateShaderModule(m_Device, compCode);

        VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = compModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage = stage;
        pipelineInfo.layout = m_PipelineLayout;
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));

        vkDestroyShaderModule(m_Device, compModule, nullptr);

        return true;
    }

    void SDFRayMarchPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_EntitySetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_EntitySetLayout, nullptr);
            if (m_ClipmapSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ClipmapSetLayout, nullptr);
            if (m_EntitySampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_EntitySampler, nullptr);
            if (m_ClipmapSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_ClipmapSampler, nullptr);
            if (m_OutputView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputView, nullptr);
            if (m_PlaceholderView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_PlaceholderView, nullptr);
            for (EntitySDF& entitySdf : m_Entities) {
                if (entitySdf.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, entitySdf.view, nullptr);
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            if (m_OutputImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_OutputImage, m_OutputAllocation);
            if (m_PlaceholderImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_PlaceholderImage, m_PlaceholderAllocation);
            for (EntitySDF& entitySdf : m_Entities) {
                if (entitySdf.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, entitySdf.image, entitySdf.allocation);
            }
        }
        m_BVHNodeBuffer.Destroy();
        m_LeafEntityIndexBuffer.Destroy();
        m_EntityParamsBuffer.Destroy();
        m_Entities.clear();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_EntitySetLayout = VK_NULL_HANDLE;
        m_ClipmapSetLayout = VK_NULL_HANDLE;
        m_EntitySet = VK_NULL_HANDLE;
        m_ClipmapSet = VK_NULL_HANDLE;
        m_EntitySampler = VK_NULL_HANDLE;
        m_ClipmapSampler = VK_NULL_HANDLE;
        m_OutputImage = VK_NULL_HANDLE; m_OutputAllocation = VK_NULL_HANDLE; m_OutputView = VK_NULL_HANDLE;
        m_PlaceholderImage = VK_NULL_HANDLE; m_PlaceholderAllocation = VK_NULL_HANDLE; m_PlaceholderView = VK_NULL_HANDLE;
        m_RootNodeIndex = -1;
        m_OutputWidth = 0;
        m_OutputHeight = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void SDFRayMarchPass::SetGlobalSDFViews(const GlobalSDFPass& globalSDF) {
        VkDescriptorImageInfo clipmapImageInfos[GlobalSDFPass::kLevelCount]{};
        for (uint32_t level = 0; level < GlobalSDFPass::kLevelCount; ++level) {
            clipmapImageInfos[level].sampler = m_ClipmapSampler;
            clipmapImageInfos[level].imageView = globalSDF.GetClipmapView(level);
            clipmapImageInfos[level].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = m_ClipmapSet;
        write.dstBinding = 0;
        write.descriptorCount = GlobalSDFPass::kLevelCount;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = clipmapImageInfos;
        vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
    }

    void SDFRayMarchPass::RecordRayMarch(VkCommandBuffer cmd, const GlobalSDFPass& globalSDF,
        const maths::vec3& cameraPosition, const maths::vec3& cameraForward, const maths::vec3& cameraUp,
        float fovYRadians, float aspectRatio, float nearZ, float farZ) {

        // Same orthonormal-basis derivation as renderer::SurfaceCachePass::UpdateVisibility (right =
        // forward x upHint, up = right x forward) -- see that method's own comment.
        const maths::vec3 forward = cameraForward.Normalize();
        const maths::vec3 right = forward.Cross(cameraUp).Normalize();
        const maths::vec3 up = right.Cross(forward);

        SDFRayMarchPC pc{};
        pc.cameraPosition[0] = cameraPosition.x; pc.cameraPosition[1] = cameraPosition.y; pc.cameraPosition[2] = cameraPosition.z;
        pc.tanHalfFovY = std::tan(fovYRadians * 0.5f);
        pc.cameraForward[0] = forward.x; pc.cameraForward[1] = forward.y; pc.cameraForward[2] = forward.z;
        pc.aspectRatio = aspectRatio;
        pc.cameraRight[0] = right.x; pc.cameraRight[1] = right.y; pc.cameraRight[2] = right.z;
        pc.nearZ = nearZ;
        pc.cameraUp[0] = up.x; pc.cameraUp[1] = up.y; pc.cameraUp[2] = up.z;
        pc.farZ = farZ;

        for (uint32_t level = 0; level < GlobalSDFPass::kLevelCount; ++level) {
            pc.levelVoxelSize[level] = globalSDF.GetLevelVoxelSize(level);
            int32_t centerVoxel[3]{};
            globalSDF.GetLevelSnappedCenterVoxel(level, centerVoxel);
            pc.levelCenterVoxel[level * 3 + 0] = centerVoxel[0];
            pc.levelCenterVoxel[level * 3 + 1] = centerVoxel[1];
            pc.levelCenterVoxel[level * 3 + 2] = centerVoxel[2];
        }
        pc.clipmapResolution = static_cast<int32_t>(GlobalSDFPass::kClipmapResolution);
        pc.rootNodeIndex = m_RootNodeIndex;
        pc.entitySamplerCount = static_cast<int32_t>(std::min<size_t>(m_Entities.size(), kMaxEntitySDFs));

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        VkDescriptorSet sets[2] = { m_EntitySet, m_ClipmapSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 2, sets, 0, nullptr);
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        constexpr uint32_t kWorkgroupSize = 8;
        const uint32_t groupsX = (m_OutputWidth + kWorkgroupSize - 1u) / kWorkgroupSize;
        const uint32_t groupsY = (m_OutputHeight + kWorkgroupSize - 1u) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

}
