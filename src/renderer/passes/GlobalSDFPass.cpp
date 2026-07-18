#include "renderer/passes/GlobalSDFPass.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>
#include <thread>

#include "core/EngineConfig.h" // config::ENTITY_SELF_ROTATION_ENABLED
#include "core/Logger.h"
#include "geometry/MeshSDFGenerator.h"
#include "io/CacheFileManager.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

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
            // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): object-space
            // rotation support -- pivotCenter + the object's INVERSE rotation (CPU-computed via
            // maths::mat4::Inverse(), see GlobalSDFPass::RecordUpdate's own call site), so
            // GlobalSDFComposite.comp's mode==1 branch can un-rotate a world-space query point back
            // to the entity's rest pose before sampling its (rest-pose-baked) Mesh SDF. Identity
            // (invRot = I) whenever the entity is static or config::ENTITY_SELF_ROTATION_ENABLED is
            // off -- see that call site's own comment for why identity makes pivotCenter's exact
            // value irrelevant in that case.
            float pivotCenterX = 0, pivotCenterY = 0, pivotCenterZ = 0;
            float invRot00 = 1, invRot01 = 0, invRot02 = 0;
            float invRot10 = 0, invRot11 = 1, invRot12 = 0;
            float invRot20 = 0, invRot21 = 0, invRot22 = 1;
        };
        static_assert(sizeof(GlobalSDFCompositePC) == 112,
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



        // Per-entity Mesh SDF resolution: deliberately coarser than the surface-detail-oriented
        // geometry::kMeshSDFResolution default -- the Global SDF only needs enough resolution to
        // drive cone tracing's coarse empty-space skipping, not fine per-object surface detail
        // (that remains the job of the per-cluster geometry / surface cache, not this clipmap).
        // Tier-scaled (config::lumen::GLOBAL_SDF_ENTITY_RESOLUTION, see
        // EngineConfig_{Low,Medium,High,Extrem}.h) exactly like GlobalSDFPass::kClipmapResolution
        // above -- assigned at the very top of Init() below, before any bake job reads it. No
        // longer `constexpr` for that reason; every existing reader (the worker-thread lambda's
        // BuildMeshSDF() call and Init()'s own log message, both further down this file) was
        // already reading it as an ordinary namespace-scope name, so dropping `constexpr` changes
        // nothing about how it is used, only that its value can now vary by tier. Must stay a
        // multiple of geometry::kSDFBlockDim (4) -- geometry::BuildMeshSDF returns an empty
        // (resolution == 0) MeshSDF otherwise, see that function's own doc comment.
        uint32_t kEntitySDFResolution = 24u;

    } // namespace

    bool GlobalSDFPass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const std::filesystem::path& cacheFilePath, core::LoadingManager& loadingManager) {
        // Tier-scaled Global SDF quality knobs -- read first, before anything below sizes an
        // image or bakes an entity SDF from them. Mirrors renderer::WorldProbeGridPass::Init()'s
        // own kGridResolution/kProbeSpacing/kProbeSampleDirections = config::lumen::... assignment
        // convention (both classes keep these as ordinary `static inline` / anonymous-namespace
        // variables rather than `constexpr`, precisely so Init() can overwrite them here).
        kClipmapResolution = config::lumen::GLOBAL_SDF_CLIPMAP_RESOLUTION;
        kEntitySDFResolution = config::lumen::GLOBAL_SDF_ENTITY_RESOLUTION;

        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Read every fallback mesh's geometry and build one CPU-side Mesh SDF per
        // entity (geometry::BuildMeshSDF, see MeshSDFGenerator.h) from it. Fanned out across
        // `loadingManager`'s worker threads (one job per entity) instead of a single-threaded
        // loop -- see Init()'s own header comment for why this stays a blocking WaitIdle() rather
        // than a progressive, frame-spread arrival.
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
            uint32_t resolution = 0; // 0 == this entity's bake failed/produced an empty mesh; skipped below.
            uint32_t entityID = 0;
            bool readFailed = false; // Distinguishes a hard I/O error (Init() must fail) from a benign empty mesh.
        };
        // Pre-sized, one disjoint slot per fallback-table entry: every worker job below only ever
        // writes to its own index, so no locking is needed between jobs (the same discipline a
        // plain parallel-for would use) -- WaitIdle() below is the only synchronization point.
        std::vector<PendingUpload> uploads(fallbackTable.size());

        for (size_t i = 0; i < fallbackTable.size(); ++i) {
            const geometry::FallbackMeshIndexEntry& entry = fallbackTable[i];
            uploads[i].entityID = entry.entityID;
            loadingManager.Submit([&uploads, i, cacheFilePath, entry]() {
                // Runs on a worker thread. geometry::CacheFileManager holds no member state (every
                // method is effectively a free function bundled in a class, see CacheFileManager.h),
                // so constructing one per job and reading the same file concurrently from many
                // threads is safe -- each call opens/reads/closes its own OS file handle.
                geometry::CacheFileManager workerCacheManager;
                std::vector<geometry::FallbackVertex> vertices;
                std::vector<uint32_t> indices;
                if (!workerCacheManager.ReadFallbackMeshGeometry(cacheFilePath, entry, vertices, indices)) {
                    uploads[i].readFailed = true;
                    return;
                }
                if (vertices.empty() || indices.size() < 3) {
                    return; // resolution stays 0 -- benign "no fallback geometry" case, not an error.
                }

                std::vector<maths::vec3> positions;
                positions.reserve(vertices.size());
                for (const geometry::FallbackVertex& v : vertices) {
                    positions.push_back(maths::vec3{ v.position[0], v.position[1], v.position[2] });
                }

                geometry::MeshSDF sdf = geometry::BuildMeshSDF(positions, indices, kEntitySDFResolution);
                if (sdf.resolution == 0) {
                    return; // BuildMeshSDF's own empty-SDF sentinel -- resolution stays 0, skipped below.
                }

                uploads[i].volumeMin = sdf.volumeMin;
                uploads[i].voxelSize = sdf.voxelSize;
                uploads[i].resolution = sdf.resolution;
                uploads[i].decodedGrid.resize(static_cast<size_t>(sdf.resolution) * sdf.resolution * sdf.resolution);
                for (uint32_t z = 0; z < sdf.resolution; ++z) {
                    for (uint32_t y = 0; y < sdf.resolution; ++y) {
                        for (uint32_t x = 0; x < sdf.resolution; ++x) {
                            uploads[i].decodedGrid[(static_cast<size_t>(z) * sdf.resolution + y) * sdf.resolution + x] =
                                geometry::DecodeMeshSDFVoxel(sdf, x, y, z);
                        }
                    }
                }
                });
        }
        // Blocks until every entity's bake above has actually finished running -- see Init()'s own
        // header comment for why this pass cannot let entities keep arriving after this point.
        loadingManager.WaitIdle();

        for (const PendingUpload& upload : uploads) {
            if (upload.readFailed) {
                LOG_ERROR(std::format(
                    "[GlobalSDFPass] Failed to read fallback mesh geometry for entityID={}.", upload.entityID));
                return false;
            }
        }
        std::erase_if(uploads, [](const PendingUpload& upload) {
            if (upload.resolution == 0) {
                LOG_WARNING(std::format(
                    "[GlobalSDFPass] BuildMeshSDF produced an empty SDF for entityID={}; skipping.", upload.entityID));
                return true;
            }
            return false;
            });

        LOG_INFO(std::format(
            "[GlobalSDFPass] Built {} entity Mesh SDF(s) at {}^3 for Global SDF compositing ({} worker thread(s)).",
            uploads.size(), kEntitySDFResolution, std::thread::hardware_concurrency()));

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
            entitySdf.entityID = uploads[i].entityID;

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

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
                VkDeviceSize copyOffset = 0;
                for (size_t i = 0; i < uploads.size(); ++i) {
                    VulkanUtils::TransitionImageLayout(cmd, m_Entities[i].image,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

                    VkBufferImageCopy copyRegion{};
                    copyRegion.bufferOffset = copyOffset;
                    copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                    copyRegion.imageExtent = { uploads[i].resolution, uploads[i].resolution, uploads[i].resolution };
                    vkCmdCopyBufferToImage(cmd, stagingBuffer, m_Entities[i].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
                    copyOffset += static_cast<VkDeviceSize>(uploads[i].decodedGrid.size()) * sizeof(float);

                    VulkanUtils::TransitionImageLayout(cmd, m_Entities[i].image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
                }

                for (uint32_t level = 0; level < kLevelCount; ++level) {
                    VulkanUtils::TransitionImageLayout(cmd, m_Levels[level].image,
                        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
                }
            });

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

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/GlobalSDFComposite.comp.spv");
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

    std::vector<GlobalSDFPass::TracedEntityInfo> GlobalSDFPass::GetTracedEntityInfos() const {
        std::vector<TracedEntityInfo> infos;
        infos.reserve(m_Entities.size());
        for (const EntitySDF& entity : m_Entities) {
            TracedEntityInfo info{};
            info.entityID = entity.entityID;
            info.sdfView = entity.view;
            info.volumeMin = entity.volumeMin;
            info.voxelSize = entity.voxelSize;
            info.resolution = entity.resolution;
            infos.push_back(info);
        }
        return infos;
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

    void GlobalSDFPass::EnqueueDirtyRegionsForEntity(uint32_t entityID, const maths::vec3& centerWorld, float boundingRadius) {
        (void)entityID; // Not needed here -- RecordSlab's own existing per-entity overlap test
                         // (against slabWorldMin/Max, see that function's own comment) already
                         // resolves which entity/entities a drained slab actually composites;
                         // this method only needs to know WHERE to mark dirty, not which entity.
        // Unlike EnqueueDirtyRegionsForLevel (which enqueues a coarse, window-sized slab sized to
        // whatever axis-aligned slice of the level's own covered window moved), this enqueues a
        // TIGHT slab per level, sized to just this entity's own bounding sphere converted to that
        // level's voxel grid -- much cheaper than a full-window refill, and correctly proportioned
        // (fewer voxels at coarser levels, since voxelSize scales up with level).
        for (uint32_t level = 0; level < kLevelCount; ++level) {
            const float voxelSize = GetLevelVoxelSize(level);

            DirtySlab slab{};
            slab.level = level;
            const float worldMin[3] = { centerWorld.x - boundingRadius, centerWorld.y - boundingRadius, centerWorld.z - boundingRadius };
            const float worldMax[3] = { centerWorld.x + boundingRadius, centerWorld.y + boundingRadius, centerWorld.z + boundingRadius };
            for (int axis = 0; axis < 3; ++axis) {
                slab.voxelMin[axis] = static_cast<int32_t>(std::floor(worldMin[axis] / voxelSize));
                slab.voxelMax[axis] = static_cast<int32_t>(std::ceil(worldMax[axis] / voxelSize));
            }
            m_PendingSlabs.push_back(slab);
        }
    }

    void GlobalSDFPass::RecordSlab(VkCommandBuffer cmd, const DirtySlab& slab, const core::EntityTransformCPU* entityTransformsCPU) {
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

                        // Phase 4 integration: object-space rotation. Identity (pc's own default
                        // member initializers) whenever the switch is off OR entityTransformsCPU
                        // wasn't supplied -- pivotCenter's exact value is then irrelevant, since
                        // invRot=I makes GlobalSDFComposite.comp's `pivotCenter + invRot*(worldPos
                        // - pivotCenter)` reduce to exactly worldPos regardless of pivotCenter.
                        if (config::ENTITY_SELF_ROTATION_ENABLED && entityTransformsCPU != nullptr) {
                            const core::EntityTransformCPU& xform = entityTransformsCPU[entitySdf.entityID];
                            maths::mat4 invRot = xform.rotation.Inverse();
                            pc.pivotCenterX = xform.center.x;
                            pc.pivotCenterY = xform.center.y;
                            pc.pivotCenterZ = xform.center.z;
                            // Column-major maths::mat4 storage (m[row + col*4], see Maths.h) --
                            // invRotXY below names row X, column Y of the 3x3 rotation block.
                            pc.invRot00 = invRot.m[0]; pc.invRot01 = invRot.m[4]; pc.invRot02 = invRot.m[8];
                            pc.invRot10 = invRot.m[1]; pc.invRot11 = invRot.m[5]; pc.invRot12 = invRot.m[9];
                            pc.invRot20 = invRot.m[2]; pc.invRot21 = invRot.m[6]; pc.invRot22 = invRot.m[10];
                        }

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

    void GlobalSDFPass::DrainAndRecordSlabs(VkCommandBuffer cmd, const core::EntityTransformCPU* entityTransformsCPU) {
        uint32_t processed = 0;
        while (processed < kMaxDirtySlabsPerCall && !m_PendingSlabs.empty()) {
            DirtySlab slab = m_PendingSlabs.front();
            m_PendingSlabs.pop_front();
            RecordSlab(cmd, slab, entityTransformsCPU);
            ++processed;
        }
    }

    void GlobalSDFPass::RecordUpdate(VkCommandBuffer cmd, const maths::vec3& cameraPositionWorld,
        const core::EntityTransformCPU* entityTransformsCPU) {
        for (uint32_t level = 0; level < kLevelCount; ++level) {
            EnqueueDirtyRegionsForLevel(level, cameraPositionWorld);
        }

        // Phase 4 integration (UE5.8 parity roadmap, dynamic scenes onto main): while entities
        // rotate, each one's baked-rest-pose Mesh SDF needs re-compositing (at its new orientation)
        // every frame -- re-enqueue every rotating entity's own (fixed, translation-invariant)
        // dirty region here, gated by the SAME kill-switch that drives the rotation itself. A
        // no-op loop (and therefore zero behavior change from main's own pre-integration behavior)
        // whenever the switch is off.
        if (config::ENTITY_SELF_ROTATION_ENABLED && entityTransformsCPU != nullptr) {
            for (const EntitySDF& entitySdf : m_Entities) {
                float boundingRadius = entitySdf.voxelSize * static_cast<float>(entitySdf.resolution) * 0.5f;
                maths::vec3 entityCenter = entitySdf.volumeMin + maths::vec3{ boundingRadius, boundingRadius, boundingRadius };
                EnqueueDirtyRegionsForEntity(entitySdf.entityID, entityCenter, boundingRadius * 1.7321f); // sqrt(3) cube-diagonal margin -- a conservative sphere around the object's cubic AABB.
            }
        }

        DrainAndRecordSlabs(cmd, entityTransformsCPU);
    }

}
