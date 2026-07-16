#include "renderer/GlobalSDFPass.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/MeshSDFGenerator.h"
#include "io/CacheFileManager.h"

namespace renderer {

    namespace {

        // Mirrors HZBPass::ReadShaderFile / every other pass's own copy -- duplicated rather than
        // shared because this class is deliberately self-contained, matching this codebase's
        // existing per-pass convention.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("GlobalSDFPass: failed to open SPIR-V file: " + filename);
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

        // Byte-for-byte layout match for GlobalSDFCompositePC in GlobalSDFComposite.comp -- flat
        // scalar fields throughout (no vec3/ivec3 members), matching HZBPushConstants's own
        // established convention in this codebase for push-constant structs that must line up
        // with a GLSL block with zero ambiguity about std430-vs-plain alignment rules.
        struct GlobalSDFCompositePC {
            int32_t dispatchMinTexelX = 0, dispatchMinTexelY = 0, dispatchMinTexelZ = 0;
            int32_t mode = 0; // 0 = clear-to-farValue, 1 = composite-min-object.
            float dispatchWorldMinX = 0, dispatchWorldMinY = 0, dispatchWorldMinZ = 0;
            float clipmapVoxelSize = 0;
            float objectVolumeMinX = 0, objectVolumeMinY = 0, objectVolumeMinZ = 0;
            float objectVoxelSize = 0;
            int32_t objectResolution = 0;
            float farValue = 0;
            int32_t clipmapResolution = 0;
            int32_t _pad0 = 0;
        };
        static_assert(sizeof(GlobalSDFCompositePC) == 64,
            "GlobalSDFCompositePC must match GlobalSDFComposite.comp's push_constant block exactly");

        constexpr uint32_t kWorkgroupSize = 4; // Matches GlobalSDFComposite.comp's local_size_x/y/z.

        uint32_t DispatchGroupCount(uint32_t extent) {
            return (extent + kWorkgroupSize - 1u) / kWorkgroupSize;
        }

        // One contiguous, wrap-resolved piece of an axis range: `worldVoxelStart` is the absolute
        // world-voxel index (see GlobalSDFPass.h's fixed-lattice convention) the piece begins at,
        // `wrappedTexelStart` is where that same voxel currently lands in the toroidally-addressed
        // GPU image, and `count` is how many consecutive voxels (and texels) the piece covers --
        // guaranteed to not cross the image's own wrap boundary, so a single dispatch can address
        // it with a plain additive offset (see GlobalSDFComposite.comp).
        struct WrappedPiece {
            int32_t worldVoxelStart = 0;
            int32_t wrappedTexelStart = 0;
            int32_t count = 0;
        };

        // Splits the absolute world-voxel range [minInclusive, maxExclusive) into 1-2
        // wrap-contiguous pieces against a toroidal image of `resolution` texels per axis. See the
        // WrappedPiece comment for why each returned piece is independently dispatch-safe.
        std::vector<WrappedPiece> SplitWrappedRange(int32_t minInclusive, int32_t maxExclusive, int32_t resolution) {
            std::vector<WrappedPiece> pieces;
            int32_t remaining = maxExclusive - minInclusive;
            int32_t cur = minInclusive;
            while (remaining > 0) {
                int32_t wrapped = ((cur % resolution) + resolution) % resolution;
                int32_t pieceLen = std::min(remaining, resolution - wrapped);
                pieces.push_back(WrappedPiece{ cur, wrapped, pieceLen });
                cur += pieceLen;
                remaining -= pieceLen;
            }
            return pieces;
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

        // Per-entity Mesh SDF resolution: deliberately coarser than the surface-detail-oriented
        // geometry::kMeshSDFResolution default -- the Global SDF only needs enough resolution to
        // drive cone tracing's coarse empty-space skipping, not fine per-object surface detail
        // (that remains the job of the per-cluster geometry / surface cache, not this clipmap).
        constexpr uint32_t kEntitySDFResolution = 24u;

    } // namespace

    bool GlobalSDFPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const std::filesystem::path& cacheFilePath) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Read every fallback mesh's geometry and build one CPU-side Mesh SDF per
        // entity (geometry::BuildMeshSDF, see MeshSDFGenerator.h) from it.
        // =====================================================================================
        geometry::CacheFileManager cacheManager;
        geometry::CacheFileHeader header{};
        if (!cacheManager.ReadHeader(cacheFilePath, header)) {
            LOG_ERROR(std::format("[GlobalSDFPass] Failed to read cache header '{}'.", cacheFilePath.string()));
            return false;
        }
        std::vector<geometry::FallbackMeshIndexEntry> fallbackTable;
        if (!cacheManager.ReadFallbackMeshTable(cacheFilePath, header, fallbackTable)) {
            LOG_ERROR("[GlobalSDFPass] Failed to read the fallback-mesh table.");
            return false;
        }

        struct PendingUpload {
            std::vector<float> decodedGrid; // resolution^3, voxel-linear (x fastest), world-unit distances.
            maths::vec3 volumeMin{};
            float voxelSize = 0.0f;
            uint32_t resolution = 0;
        };
        std::vector<PendingUpload> uploads;
        uploads.reserve(fallbackTable.size());

        for (const geometry::FallbackMeshIndexEntry& entry : fallbackTable) {
            std::vector<geometry::FallbackVertex> vertices;
            std::vector<uint32_t> indices;
            if (!cacheManager.ReadFallbackMeshGeometry(cacheFilePath, entry, vertices, indices)) {
                LOG_ERROR(std::format(
                    "[GlobalSDFPass] Failed to read fallback mesh geometry for entityID={}.", entry.entityID));
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

            geometry::MeshSDF sdf = geometry::BuildMeshSDF(positions, indices, kEntitySDFResolution);
            if (sdf.resolution == 0) {
                LOG_WARNING(std::format(
                    "[GlobalSDFPass] BuildMeshSDF produced an empty SDF for entityID={}; skipping.", entry.entityID));
                continue;
            }

            PendingUpload upload;
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
        }

        LOG_INFO(std::format("[GlobalSDFPass] Built {} entity Mesh SDF(s) at {}^3 for Global SDF compositing.",
            uploads.size(), kEntitySDFResolution));

        // =====================================================================================
        // STEP 2 -- GPU resources: per-entity sampled 3D SDF textures, kLevelCount clipmap
        // storage/sampled 3D images, one shared sampler, descriptor pool/sets, the compositing
        // compute pipeline.
        // =====================================================================================
        m_Entities.resize(uploads.size());

        VkImageCreateInfo entityImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        entityImageInfo.imageType = VK_IMAGE_TYPE_3D;
        entityImageInfo.format = kClipmapFormat;
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

            entityImageInfo.extent = { entitySdf.resolution, entitySdf.resolution, entitySdf.resolution };
            VK_CHECK(vmaCreateImage(allocator, &entityImageInfo, &gpuOnlyAlloc, &entitySdf.image, &entitySdf.allocation, nullptr));

            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = entitySdf.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            viewInfo.format = kClipmapFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &entitySdf.view));
        }

        VkImageCreateInfo clipmapImageInfo = entityImageInfo;
        clipmapImageInfo.extent = { kClipmapResolution, kClipmapResolution, kClipmapResolution };
        clipmapImageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        for (uint32_t level = 0; level < kLevelCount; ++level) {
            VK_CHECK(vmaCreateImage(allocator, &clipmapImageInfo, &gpuOnlyAlloc, &m_Levels[level].image, &m_Levels[level].allocation, nullptr));
            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = m_Levels[level].image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            viewInfo.format = kClipmapFormat;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_Levels[level].view));
            // Sentinel: forces EnqueueDirtyRegionsForLevel to treat this level as fully dirty the
            // first time RecordUpdate() runs (see that function + the class comment).
            m_Levels[level].hasValidWindow = false;
        }

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
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_EntitySampler));

        // --- Descriptor set layouts: set 0 (one storage-image binding, one set per clipmap
        // level), set 1 (one combined-image-sampler binding, one set per entity). ---
        VkDescriptorSetLayoutBinding levelBinding{};
        levelBinding.binding = 0;
        levelBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        levelBinding.descriptorCount = 1;
        levelBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo levelLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        levelLayoutInfo.bindingCount = 1;
        levelLayoutInfo.pBindings = &levelBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &levelLayoutInfo, nullptr, &m_LevelSetLayout));

        VkDescriptorSetLayoutBinding entityBinding{};
        entityBinding.binding = 0;
        entityBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        entityBinding.descriptorCount = 1;
        entityBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo entityLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        entityLayoutInfo.bindingCount = 1;
        entityLayoutInfo.pBindings = &entityBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &entityLayoutInfo, nullptr, &m_EntitySetLayout));

        uint32_t entityCount = static_cast<uint32_t>(m_Entities.size());
        // At least 1 set/descriptor reserved even with zero entities, so a valid (if never
        // sampled) set 1 can still be bound during clear-mode dispatches -- see RecordSlab.
        uint32_t entitySetCount = std::max(entityCount, 1u);

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kLevelCount };
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, entitySetCount };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = kLevelCount + entitySetCount;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        for (uint32_t level = 0; level < kLevelCount; ++level) {
            VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = m_DescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_LevelSetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, &m_Levels[level].descriptorSet));

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageView = m_Levels[level].view;
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.dstSet = m_Levels[level].descriptorSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
        }

        // Placeholder set (bound but never sampled from) so clear-mode dispatches always have a
        // valid set 1 bound even when this scene has zero entities -- allocated regardless of
        // entityCount via entitySetCount's floor of 1 above. Reuses entity 0's view if any entity
        // exists, otherwise stays unwritten (never read, since mode == 0 never samples it).
        for (uint32_t i = 0; i < entityCount; ++i) {
            EntitySDF& entitySdf = m_Entities[i];
            VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = m_DescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_EntitySetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, &entitySdf.descriptorSet));

            VkDescriptorImageInfo imageInfo{};
            imageInfo.sampler = m_EntitySampler;
            imageInfo.imageView = entitySdf.view;
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.dstSet = entitySdf.descriptorSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfo;
            vkUpdateDescriptorSets(m_Device, 1, &write, 0, nullptr);
        }
        if (entityCount == 0) {
            // No real entity set exists to reuse for clear-mode dispatches (see
            // m_PlaceholderEntitySet's doc comment) -- allocate a dedicated one. Left
            // deliberately unwritten: it is bound only during mode==0 dispatches, which never
            // sample set 1 (see GlobalSDFComposite.comp).
            VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = m_DescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_EntitySetLayout;
            VK_CHECK(vkAllocateDescriptorSets(m_Device, &allocInfo, &m_PlaceholderEntitySet));
        }

        // =====================================================================================
        // STEP 3 -- One-time setup command buffer: upload every entity's decoded SDF grid, then
        // transition every image (entity SDFs -> SHADER_READ_ONLY_OPTIMAL, clipmap levels ->
        // GENERAL) to its permanent layout.
        // =====================================================================================
        {
            VkDeviceSize totalUploadBytes = 0;
            for (const PendingUpload& upload : uploads) {
                totalUploadBytes += static_cast<VkDeviceSize>(upload.decodedGrid.size()) * sizeof(float);
            }

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocResultInfo{};
            if (totalUploadBytes > 0) {
                VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                stagingInfo.size = totalUploadBytes;
                stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                VmaAllocationCreateInfo stagingAllocInfo{};
                stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
                VK_CHECK(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));

                size_t writeOffset = 0;
                for (const PendingUpload& upload : uploads) {
                    size_t bytes = upload.decodedGrid.size() * sizeof(float);
                    std::memcpy(static_cast<char*>(stagingAllocResultInfo.pMappedData) + writeOffset, upload.decodedGrid.data(), bytes);
                    writeOffset += bytes;
                }
            }

            VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            VkCommandBuffer cmd;
            VK_CHECK(vkAllocateCommandBuffers(m_Device, &cmdAllocInfo, &cmd));

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            VkDeviceSize copyOffset = 0;
            for (size_t i = 0; i < uploads.size(); ++i) {
                TransitionImageLayout(cmd, m_Entities[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

                VkBufferImageCopy copyRegion{};
                copyRegion.bufferOffset = copyOffset;
                copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                copyRegion.imageExtent = { uploads[i].resolution, uploads[i].resolution, uploads[i].resolution };
                vkCmdCopyBufferToImage(cmd, stagingBuffer, m_Entities[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
                copyOffset += static_cast<VkDeviceSize>(uploads[i].decodedGrid.size()) * sizeof(float);

                TransitionImageLayout(cmd, m_Entities[i].image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }

            for (uint32_t level = 0; level < kLevelCount; ++level) {
                TransitionImageLayout(cmd, m_Levels[level].image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            }

            vkEndCommandBuffer(cmd);
            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(queue));
            vkFreeCommandBuffers(m_Device, commandPool, 1, &cmd);
            if (stagingBuffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
            }
        }

        // =====================================================================================
        // STEP 4 -- Compositing compute pipeline.
        // =====================================================================================
        VkDescriptorSetLayout setLayouts[2] = { m_LevelSetLayout, m_EntitySetLayout };
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.size = sizeof(GlobalSDFCompositePC);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 2;
        pipelineLayoutInfo.pSetLayouts = setLayouts;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        std::vector<char> shaderCode = ReadShaderFile("shaders/GlobalSDFComposite.comp.spv");
        VkShaderModule shaderModule = CreateShaderModule(m_Device, shaderCode);
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        return true;
    }

    void GlobalSDFPass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr); // Frees every allocated set.
            if (m_LevelSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_LevelSetLayout, nullptr);
            if (m_EntitySetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_EntitySetLayout, nullptr);
            if (m_EntitySampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_EntitySampler, nullptr);
            for (ClipmapLevel& level : m_Levels) {
                if (level.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, level.view, nullptr);
            }
            for (EntitySDF& entitySdf : m_Entities) {
                if (entitySdf.view != VK_NULL_HANDLE) vkDestroyImageView(m_Device, entitySdf.view, nullptr);
            }
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            for (ClipmapLevel& level : m_Levels) {
                if (level.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, level.image, level.allocation);
            }
            for (EntitySDF& entitySdf : m_Entities) {
                if (entitySdf.image != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, entitySdf.image, entitySdf.allocation);
            }
        }

        for (ClipmapLevel& level : m_Levels) { level = ClipmapLevel{}; }
        m_Entities.clear();
        m_PendingSlabs.clear();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_LevelSetLayout = VK_NULL_HANDLE;
        m_EntitySetLayout = VK_NULL_HANDLE;
        m_PlaceholderEntitySet = VK_NULL_HANDLE; // Freed implicitly by vkDestroyDescriptorPool above.
        m_EntitySampler = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void GlobalSDFPass::EnqueueDirtyRegionsForLevel(uint32_t level, const maths::vec3& cameraPositionWorld) {
        ClipmapLevel& clipmap = m_Levels[level];
        const float voxelSize = GetLevelVoxelSize(level);
        const int32_t res = static_cast<int32_t>(kClipmapResolution);
        const int32_t halfRes = res / 2;
        const int32_t chunk = static_cast<int32_t>(kSnapChunkVoxels);

        // Snap the camera's world position down to the nearest kSnapChunkVoxels-aligned voxel
        // index, per axis -- see the class comment's toroidal-streaming rationale.
        auto snapAxis = [&](float worldPos) -> int32_t {
            int32_t voxelIndex = static_cast<int32_t>(std::floor(worldPos / voxelSize));
            int32_t chunkIndex = static_cast<int32_t>(std::floor(static_cast<float>(voxelIndex) / static_cast<float>(chunk)));
            return chunkIndex * chunk;
            };
        int32_t newCenter[3] = {
            snapAxis(cameraPositionWorld.x), snapAxis(cameraPositionWorld.y), snapAxis(cameraPositionWorld.z)
        };

        if (!clipmap.hasValidWindow) {
            DirtySlab slab{};
            slab.level = level;
            for (int axis = 0; axis < 3; ++axis) {
                slab.voxelMin[axis] = newCenter[axis] - halfRes;
                slab.voxelMax[axis] = newCenter[axis] + halfRes;
            }
            m_PendingSlabs.push_back(slab);
            clipmap.snappedCenterVoxel[0] = newCenter[0];
            clipmap.snappedCenterVoxel[1] = newCenter[1];
            clipmap.snappedCenterVoxel[2] = newCenter[2];
            clipmap.hasValidWindow = true;
            return;
        }

        for (int axis = 0; axis < 3; ++axis) {
            int32_t delta = newCenter[axis] - clipmap.snappedCenterVoxel[axis];
            if (delta == 0) {
                continue;
            }

            DirtySlab slab{};
            slab.level = level;
            for (int otherAxis = 0; otherAxis < 3; ++otherAxis) {
                if (otherAxis == axis) continue;
                // Full extent on the two axes that did not move this call -- the standard
                // per-axis slab decomposition for a toroidal clipmap update (see the class
                // comment); any diagonal motion this frame is covered by the UNION of the
                // per-axis slabs enqueued here, one per moved axis.
                slab.voxelMin[otherAxis] = newCenter[otherAxis] - halfRes;
                slab.voxelMax[otherAxis] = newCenter[otherAxis] + halfRes;
            }

            if (std::abs(delta) >= res) {
                // Moved farther than the whole covered window in one update (a large camera
                // jump) -- cheaper to refill the entire level than to compute a slab wider than
                // the window itself.
                slab.voxelMin[axis] = newCenter[axis] - halfRes;
                slab.voxelMax[axis] = newCenter[axis] + halfRes;
            }
            else if (delta > 0) {
                slab.voxelMin[axis] = clipmap.snappedCenterVoxel[axis] + halfRes;
                slab.voxelMax[axis] = newCenter[axis] + halfRes;
            }
            else {
                slab.voxelMin[axis] = newCenter[axis] - halfRes;
                slab.voxelMax[axis] = clipmap.snappedCenterVoxel[axis] - halfRes;
            }
            m_PendingSlabs.push_back(slab);
        }

        clipmap.snappedCenterVoxel[0] = newCenter[0];
        clipmap.snappedCenterVoxel[1] = newCenter[1];
        clipmap.snappedCenterVoxel[2] = newCenter[2];
    }

    void GlobalSDFPass::RecordSlab(VkCommandBuffer cmd, const DirtySlab& slab) {
        const ClipmapLevel& clipmap = m_Levels[slab.level];
        const float voxelSize = GetLevelVoxelSize(slab.level);
        const int32_t res = static_cast<int32_t>(kClipmapResolution);

        std::vector<WrappedPiece> piecesX = SplitWrappedRange(slab.voxelMin[0], slab.voxelMax[0], res);
        std::vector<WrappedPiece> piecesY = SplitWrappedRange(slab.voxelMin[1], slab.voxelMax[1], res);
        std::vector<WrappedPiece> piecesZ = SplitWrappedRange(slab.voxelMin[2], slab.voxelMax[2], res);

        // Slab world AABB (used to decide which entities overlap it at all) -- computed once,
        // shared by every wrap piece below.
        maths::vec3 slabWorldMin{
            static_cast<float>(slab.voxelMin[0]) * voxelSize,
            static_cast<float>(slab.voxelMin[1]) * voxelSize,
            static_cast<float>(slab.voxelMin[2]) * voxelSize };
        maths::vec3 slabWorldMax{
            static_cast<float>(slab.voxelMax[0]) * voxelSize,
            static_cast<float>(slab.voxelMax[1]) * voxelSize,
            static_cast<float>(slab.voxelMax[2]) * voxelSize };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);

        for (const WrappedPiece& px : piecesX) {
            for (const WrappedPiece& py : piecesY) {
                for (const WrappedPiece& pz : piecesZ) {
                    GlobalSDFCompositePC pc{};
                    pc.dispatchMinTexelX = px.wrappedTexelStart;
                    pc.dispatchMinTexelY = py.wrappedTexelStart;
                    pc.dispatchMinTexelZ = pz.wrappedTexelStart;
                    pc.dispatchWorldMinX = static_cast<float>(px.worldVoxelStart) * voxelSize;
                    pc.dispatchWorldMinY = static_cast<float>(py.worldVoxelStart) * voxelSize;
                    pc.dispatchWorldMinZ = static_cast<float>(pz.worldVoxelStart) * voxelSize;
                    pc.clipmapVoxelSize = voxelSize;
                    pc.farValue = kFarValue;
                    pc.clipmapResolution = res;
                    pc.mode = 0;

                    uint32_t groupsX = DispatchGroupCount(static_cast<uint32_t>(px.count));
                    uint32_t groupsY = DispatchGroupCount(static_cast<uint32_t>(py.count));
                    uint32_t groupsZ = DispatchGroupCount(static_cast<uint32_t>(pz.count));

                    // Clear mode never samples set 1 (see GlobalSDFComposite.comp), but the
                    // pipeline layout still declares it, so a compatible set must be bound
                    // regardless -- entity 0's (arbitrary, unread) if any entity exists,
                    // otherwise the dedicated placeholder allocated in Init().
                    VkDescriptorSet clearSets[2] = {
                        clipmap.descriptorSet,
                        m_Entities.empty() ? m_PlaceholderEntitySet : m_Entities[0].descriptorSet
                    };
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 2, clearSets, 0, nullptr);
                    vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);

                    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    depInfo.memoryBarrierCount = 1;
                    depInfo.pMemoryBarriers = &barrier;
                    vkCmdPipelineBarrier2(cmd, &depInfo);

                    // One min-composite dispatch per entity whose Mesh SDF volume overlaps this
                    // slab at all -- sequential (barrier between every dispatch above/below), so
                    // no two dispatches ever race on the same texel: each fully completes,
                    // including making its write visible, before the next one's read.
                    for (const EntitySDF& entitySdf : m_Entities) {
                        maths::vec3 entityMax = entitySdf.volumeMin + maths::vec3{
                            entitySdf.voxelSize * static_cast<float>(entitySdf.resolution),
                            entitySdf.voxelSize * static_cast<float>(entitySdf.resolution),
                            entitySdf.voxelSize * static_cast<float>(entitySdf.resolution) };
                        bool overlaps =
                            entitySdf.volumeMin.x < slabWorldMax.x && entityMax.x > slabWorldMin.x &&
                            entitySdf.volumeMin.y < slabWorldMax.y && entityMax.y > slabWorldMin.y &&
                            entitySdf.volumeMin.z < slabWorldMax.z && entityMax.z > slabWorldMin.z;
                        if (!overlaps) {
                            continue;
                        }

                        pc.mode = 1;
                        pc.objectVolumeMinX = entitySdf.volumeMin.x;
                        pc.objectVolumeMinY = entitySdf.volumeMin.y;
                        pc.objectVolumeMinZ = entitySdf.volumeMin.z;
                        pc.objectVoxelSize = entitySdf.voxelSize;
                        pc.objectResolution = static_cast<int32_t>(entitySdf.resolution);

                        VkDescriptorSet compositeSets[2] = { clipmap.descriptorSet, entitySdf.descriptorSet };
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 2, compositeSets, 0, nullptr);
                        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                        vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
                        vkCmdPipelineBarrier2(cmd, &depInfo);
                    }
                }
            }
        }
    }

    void GlobalSDFPass::DrainAndRecordSlabs(VkCommandBuffer cmd) {
        uint32_t processed = 0;
        while (processed < kMaxDirtySlabsPerCall && !m_PendingSlabs.empty()) {
            DirtySlab slab = m_PendingSlabs.front();
            m_PendingSlabs.pop_front();
            RecordSlab(cmd, slab);
            ++processed;
        }
    }

    void GlobalSDFPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld) {
        for (uint32_t level = 0; level < kLevelCount; ++level) {
            EnqueueDirtyRegionsForLevel(level, cameraPositionWorld);
        }
        DrainAndRecordSlabs(cmd);
    }

}
