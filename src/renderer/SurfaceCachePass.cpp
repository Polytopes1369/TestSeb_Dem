#include "renderer/SurfaceCachePass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/CardGenerator.h" // geometry::kSurfaceCacheAtlasSize / kCardGutterTexels
#include "io/CacheFileManager.h"

namespace renderer {

    namespace {

        // Mirrors HZBPass::ReadShaderFile / every other pass's own copy -- duplicated rather than
        // shared because this class is deliberately self-contained, matching this codebase's
        // existing per-pass convention.
        std::vector<char> ReadShaderFile(const std::string& filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("SurfaceCachePass: failed to open SPIR-V file: " + filename);
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

        // Byte-for-byte layout match for SurfaceCaptureConstants in SurfaceCacheCapture.vert/.frag:
        // one mat4 (64 bytes, both stages read it) + one uint (entityID, read by the fragment
        // stage only for its procedural-material hash) -- no padding required, push constants use
        // tight scalar packing, not std140.
        struct SurfaceCaptureConstants {
            maths::mat4 viewProj;
            uint32_t entityID = 0;
        };
        static_assert(sizeof(SurfaceCaptureConstants) == 68,
            "SurfaceCaptureConstants must match SurfaceCacheCapture.vert/.frag's push_constant block exactly");

        // Byte-for-byte std140 layout match for PointLightGPU / SurfaceCacheLightingUBO in
        // SurfaceCacheCapture.frag -- flat float[4]/uint32_t fields and an explicit padding array
        // throughout (see GlobalSDFCompositePC's own comment on why this codebase always mirrors a
        // GPU struct this way rather than relying on maths:: vector types matching std140 rules).
        struct PointLightGPU {
            float positionAndRadius[4]; // xyz = world position, w = radius.
            float colorAndIntensity[4]; // rgb = color, a = intensity.
        };
        static_assert(sizeof(PointLightGPU) == 32,
            "PointLightGPU must match SurfaceCacheCapture.frag's PointLightGPU exactly");

        struct SurfaceCacheLightingUBO {
            maths::mat4 lightViewProj;
            float sunDirectionAndIntensity[4]; // xyz = direction (light -> scene), w = intensity.
            float sunColor[4];                 // rgb = color, a unused.
            uint32_t pointLightCount;
            uint32_t _pad0[3];
            PointLightGPU pointLights[kMaxPointLights];
        };
        static_assert(sizeof(SurfaceCacheLightingUBO) == 368,
            "SurfaceCacheLightingUBO must match SurfaceCacheCapture.frag's SurfaceCacheLightingUBO exactly");

        // Per-face world-space outward normal, orthographic "up" vector (chosen non-parallel to
        // the normal; the specific axis is otherwise arbitrary), and which local-bounds axis (0=X,
        // 1=Y, 2=Z) that face looks ALONG (its "depth" axis) -- indexed by geometry::
        // CardFaceDirection. Verified (see SurfaceCachePass.cpp's own derivation notes) to make
        // maths::mat4::LookAt's resulting screen-right/screen-up axes match CardGenerator.cpp's
        // FaceFootprint mapping exactly, which is what makes a CPU-packed card's atlasSize agree
        // with what this shader actually rasterizes into it.
        struct CardFaceBasis {
            maths::vec3 normal;
            maths::vec3 up;
            uint32_t depthAxis;
        };
        constexpr CardFaceBasis kCardFaceBasis[geometry::kMaxCardsPerEntity] = {
            /* kCardFacePosX */ { {  1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 0u },
            /* kCardFaceNegX */ { { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 0u },
            /* kCardFacePosY */ { { 0.0f,  1.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, 1u },
            /* kCardFaceNegY */ { { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 1u },
            /* kCardFacePosZ */ { { 0.0f, 0.0f,  1.0f }, { 0.0f, 1.0f, 0.0f }, 2u },
            /* kCardFaceNegZ */ { { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f, 0.0f }, 2u },
        };

        // Mirrors CardGenerator.cpp's anonymous-namespace FaceFootprint exactly (that function is
        // private to CardGenerator.cpp, so this is a deliberate, documented duplication) -- the two
        // AABB extents a given face projects onto its card plane, in the (U, V) order that maps to
        // increasing atlas (x, y).
        void FaceFootprint(uint32_t faceDirection, const maths::vec3& extent, float& outU, float& outV) {
            switch (faceDirection) {
            case geometry::kCardFacePosX:
            case geometry::kCardFaceNegX:
                outU = extent.z; outV = extent.y; break;
            case geometry::kCardFacePosY:
            case geometry::kCardFaceNegY:
                outU = extent.x; outV = extent.z; break;
            default:
                outU = extent.x; outV = extent.y; break;
            }
        }

        // Builds this card's orthographic view-projection matrix: eye sits just outside the
        // entity's AABB along the face normal, looking straight back at the AABB center. zNear/
        // zFar are sized from depthMargin so the WHOLE AABB thickness along the face's depth axis
        // is inside the encoded depth range (near the eye's own face, far past the opposite face).
        maths::mat4 BuildCardViewProj(const geometry::SurfaceCacheCardEntry& card) {
            const maths::vec3 boundsMin{ card.localBoundsMin[0], card.localBoundsMin[1], card.localBoundsMin[2] };
            const maths::vec3 boundsMax{ card.localBoundsMax[0], card.localBoundsMax[1], card.localBoundsMax[2] };
            const maths::vec3 center = (boundsMin + boundsMax) * 0.5f;
            const maths::vec3 extent = boundsMax - boundsMin;

            const CardFaceBasis& basis = kCardFaceBasis[card.faceDirection];
            const float extentAxis[3] = { extent.x, extent.y, extent.z };
            const float halfDepth = std::max(extentAxis[basis.depthAxis] * 0.5f, 1.0e-4f);

            const float maxExtent = std::max({ extent.x, extent.y, extent.z, 1.0e-4f });
            const float depthMargin = std::max(0.1f * maxExtent, 1.0e-3f);

            const maths::vec3 eye = center + basis.normal * (halfDepth + depthMargin);
            const maths::mat4 view = maths::mat4::LookAt(eye, center, basis.up);

            float footprintU = 0.0f, footprintV = 0.0f;
            FaceFootprint(card.faceDirection, extent, footprintU, footprintV);
            const float halfWidth = std::max(footprintU * 0.5f, 1.0e-4f);
            const float halfHeight = std::max(footprintV * 0.5f, 1.0e-4f);

            const float zNear = depthMargin * 0.5f;
            const float zFar = 2.0f * halfDepth + depthMargin;
            const maths::mat4 proj = maths::mat4::OrthoVulkan(halfWidth, halfHeight, zNear, zFar);

            return proj * view;
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

    } // namespace

    bool SurfaceCachePass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const std::filesystem::path& cacheFilePath) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Read the card table + every fallback mesh's geometry (CPU side).
        // =====================================================================================
        geometry::CacheFileManager cacheManager;
        geometry::CacheFileHeader header{};
        if (!cacheManager.ReadHeader(cacheFilePath, header)) {
            LOG_ERROR(std::format("[SurfaceCachePass] Failed to read cache header '{}'.", cacheFilePath.string()));
            return false;
        }
        if (!cacheManager.ReadSurfaceCacheCardTable(cacheFilePath, header, m_Cards)) {
            LOG_ERROR("[SurfaceCachePass] Failed to read the surface-cache card table.");
            return false;
        }
        // Every card starts non-resident (see CardRuntimeState's own comment) -- UpdateVisibility()
        // is what grants a card its first atlas page, the first time it is actually on screen.
        m_CardStates.assign(m_Cards.size(), CardRuntimeState{});
        m_AtlasAllocator.Reset();
        m_DirtyCardQueue.clear();

        std::vector<geometry::FallbackMeshIndexEntry> fallbackTable;
        if (!cacheManager.ReadFallbackMeshTable(cacheFilePath, header, fallbackTable)) {
            LOG_ERROR("[SurfaceCachePass] Failed to read the fallback-mesh table.");
            return false;
        }

        std::vector<geometry::FallbackVertex> hostVertices;
        std::vector<uint32_t> hostIndices;
        m_EntityRanges.reserve(fallbackTable.size());
        for (const geometry::FallbackMeshIndexEntry& entry : fallbackTable) {
            std::vector<geometry::FallbackVertex> entityVertices;
            std::vector<uint32_t> entityIndices;
            if (!cacheManager.ReadFallbackMeshGeometry(cacheFilePath, entry, entityVertices, entityIndices)) {
                LOG_ERROR(std::format(
                    "[SurfaceCachePass] Failed to read fallback mesh geometry for entityID={}.", entry.entityID));
                return false;
            }

            EntityDrawRange range{};
            range.vertexOffset = static_cast<int32_t>(hostVertices.size());
            range.firstIndex = static_cast<uint32_t>(hostIndices.size());
            range.indexCount = static_cast<uint32_t>(entityIndices.size());
            m_EntityRanges[entry.entityID] = range;

            hostVertices.insert(hostVertices.end(), entityVertices.begin(), entityVertices.end());
            hostIndices.insert(hostIndices.end(), entityIndices.begin(), entityIndices.end());
        }

        LOG_INFO(std::format(
            "[SurfaceCachePass] Loaded {} card(s) across {} fallback mesh(es) ({} vertices, {} indices total).",
            m_Cards.size(), fallbackTable.size(), hostVertices.size(), hostIndices.size()));

        // =====================================================================================
        // STEP 2 -- GPU resources: combined vertex/index buffers, 6 atlas images + 1 shared depth
        // image, one shared sampler, the capture pipeline.
        // =====================================================================================
        VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(hostVertices.size()) * sizeof(geometry::FallbackVertex);
        VkDeviceSize indexBytes = static_cast<VkDeviceSize>(hostIndices.size()) * sizeof(uint32_t);

        // ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | SHADER_DEVICE_ADDRESS_BIT: lets
        // renderer::SurfaceCacheRayTracingPass build BLAS geometry directly against these buffers
        // (see GetVertexBuffer()/GetIndexBuffer()'s own comment) without a second upload of the
        // same Fallback Mesh geometry. Valid usage even though this pass itself never builds an
        // acceleration structure, because VulkanContext::CreateLogicalDevice enables
        // VK_KHR_acceleration_structure + bufferDeviceAddress unconditionally at device creation.
        if (vertexBytes > 0 && indexBytes > 0) {
            m_VertexBuffer.Create(allocator, vertexBytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
            m_IndexBuffer.Create(allocator, indexBytes,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        }

        VkImageCreateInfo atlasImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        atlasImageInfo.imageType = VK_IMAGE_TYPE_2D;
        atlasImageInfo.extent = { geometry::kSurfaceCacheAtlasSize, geometry::kSurfaceCacheAtlasSize, 1 };
        atlasImageInfo.mipLevels = 1;
        atlasImageInfo.arrayLayers = 1;
        atlasImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        atlasImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        atlasImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        atlasImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        atlasImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo gpuOnlyAlloc{};
        gpuOnlyAlloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        atlasImageInfo.format = kAlbedoFormat;
        VK_CHECK(vmaCreateImage(allocator, &atlasImageInfo, &gpuOnlyAlloc, &m_AlbedoImage, &m_AlbedoAllocation, nullptr));
        atlasImageInfo.format = kNormalFormat;
        VK_CHECK(vmaCreateImage(allocator, &atlasImageInfo, &gpuOnlyAlloc, &m_NormalImage, &m_NormalAllocation, nullptr));
        atlasImageInfo.format = kEmissiveFormat;
        VK_CHECK(vmaCreateImage(allocator, &atlasImageInfo, &gpuOnlyAlloc, &m_EmissiveImage, &m_EmissiveAllocation, nullptr));
        atlasImageInfo.format = kDirectLightingFormat;
        VK_CHECK(vmaCreateImage(allocator, &atlasImageInfo, &gpuOnlyAlloc, &m_DirectLightingImage, &m_DirectLightingAllocation, nullptr));

        // Radiance also needs STORAGE_BIT: SurfaceCacheGIInject.comp imageLoad/imageStore's this
        // image directly (read-modify-write of the accumulated bounce), not just a sampled read.
        VkImageCreateInfo radianceImageInfo = atlasImageInfo;
        radianceImageInfo.format = kRadianceFormat;
        radianceImageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(vmaCreateImage(allocator, &radianceImageInfo, &gpuOnlyAlloc, &m_RadianceImage, &m_RadianceAllocation, nullptr));

        atlasImageInfo.format = kWorldPosFormat;
        VK_CHECK(vmaCreateImage(allocator, &atlasImageInfo, &gpuOnlyAlloc, &m_WorldPosImage, &m_WorldPosAllocation, nullptr));

        VkImageCreateInfo depthImageInfo = atlasImageInfo;
        depthImageInfo.format = kDepthFormat;
        depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        VK_CHECK(vmaCreateImage(allocator, &depthImageInfo, &gpuOnlyAlloc, &m_DepthImage, &m_DepthAllocation, nullptr));

        auto makeView = [device](VkImage image, VkFormat format, VkImageAspectFlags aspect) {
            VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = format;
            viewInfo.subresourceRange = { aspect, 0, 1, 0, 1 };
            VkImageView view;
            VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &view));
            return view;
            };
        m_AlbedoView = makeView(m_AlbedoImage, kAlbedoFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        m_NormalView = makeView(m_NormalImage, kNormalFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        m_EmissiveView = makeView(m_EmissiveImage, kEmissiveFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        m_DirectLightingView = makeView(m_DirectLightingImage, kDirectLightingFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        m_RadianceView = makeView(m_RadianceImage, kRadianceFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        m_WorldPosView = makeView(m_WorldPosImage, kWorldPosFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        m_DepthView = makeView(m_DepthImage, kDepthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        // Clamp, not repeat/mirror: sampling near a card's edge must never wrap into a
        // neighboring, unrelated card's texels -- the gutter (geometry::kCardGutterTexels)
        // combined with clamping keeps an edge sample reading only this card's own border texel.
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_AtlasSampler));

        // =====================================================================================
        // STEP 3 -- One-time setup command buffer: upload the combined geometry (if any), clear
        // the 6 atlas images to a neutral default, and transition every image to its permanent
        // layout (GENERAL for the 6 atlas images, DEPTH_ATTACHMENT_OPTIMAL for the depth image --
        // see the class comment's atlas layout convention).
        // =====================================================================================
        {
            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VmaAllocationInfo stagingAllocResultInfo{};
            VkDeviceSize stagingSize = vertexBytes + indexBytes;
            if (stagingSize > 0) {
                VkBufferCreateInfo stagingInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                stagingInfo.size = stagingSize;
                stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                VmaAllocationCreateInfo stagingAllocInfo{};
                stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
                VK_CHECK(vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, &stagingAllocResultInfo));
                std::memcpy(stagingAllocResultInfo.pMappedData, hostVertices.data(), static_cast<size_t>(vertexBytes));
                std::memcpy(static_cast<char*>(stagingAllocResultInfo.pMappedData) + vertexBytes, hostIndices.data(), static_cast<size_t>(indexBytes));
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

            if (stagingSize > 0) {
                VkBufferCopy vertexCopy{ 0, 0, vertexBytes };
                vkCmdCopyBuffer(cmd, stagingBuffer, m_VertexBuffer.Handle(), 1, &vertexCopy);
                VkBufferCopy indexCopy{ vertexBytes, 0, indexBytes };
                vkCmdCopyBuffer(cmd, stagingBuffer, m_IndexBuffer.Handle(), 1, &indexCopy);

                VkMemoryBarrier2 uploadBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                uploadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                uploadBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                uploadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
                uploadBarrier.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT;
                VkDependencyInfo uploadDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                uploadDep.memoryBarrierCount = 1;
                uploadDep.pMemoryBarriers = &uploadBarrier;
                vkCmdPipelineBarrier2(cmd, &uploadDep);
            }

            // Transition every atlas image UNDEFINED -> TRANSFER_DST_OPTIMAL, clear it to a
            // neutral default, then transition to its permanent layout.
            VkClearColorValue albedoClear{}; albedoClear.float32[0] = 0.0f; albedoClear.float32[1] = 0.0f; albedoClear.float32[2] = 0.0f; albedoClear.float32[3] = 1.0f;
            VkClearColorValue normalClear{}; normalClear.float32[0] = 0.5f; normalClear.float32[1] = 0.5f; normalClear.float32[2] = 0.0f; normalClear.float32[3] = 1.0f;
            VkClearColorValue emissiveClear{}; emissiveClear.float32[0] = 0.0f; emissiveClear.float32[1] = 0.0f; emissiveClear.float32[2] = 0.0f; emissiveClear.float32[3] = 1.0f;
            VkClearColorValue directLightingClear{}; directLightingClear.float32[0] = 0.0f; directLightingClear.float32[1] = 0.0f; directLightingClear.float32[2] = 0.0f; directLightingClear.float32[3] = 1.0f;
            // Radiance/world-position texels not yet covered by any captured card start at zero --
            // SurfaceCacheGIInject.comp only ever visits texels inside a real, RESIDENT card's rect
            // (worldPos alpha guards this, see SurfaceCacheGIInject.comp's own check), so a zeroed
            // texel outside every card rect is simply never read.
            VkClearColorValue radianceClear{}; radianceClear.float32[0] = 0.0f; radianceClear.float32[1] = 0.0f; radianceClear.float32[2] = 0.0f; radianceClear.float32[3] = 1.0f;
            VkClearColorValue worldPosClear{}; worldPosClear.float32[0] = 0.0f; worldPosClear.float32[1] = 0.0f; worldPosClear.float32[2] = 0.0f; worldPosClear.float32[3] = 0.0f;
            VkImageSubresourceRange colorRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            struct { VkImage image; const VkClearColorValue* clear; } atlasClears[6] = {
                { m_AlbedoImage, &albedoClear }, { m_NormalImage, &normalClear }, { m_EmissiveImage, &emissiveClear },
                { m_DirectLightingImage, &directLightingClear }, { m_RadianceImage, &radianceClear }, { m_WorldPosImage, &worldPosClear }
            };
            for (auto& entry : atlasClears) {
                TransitionImageLayout(cmd, entry.image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                vkCmdClearColorImage(cmd, entry.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, entry.clear, 1, &colorRange);
                // COMPUTE_SHADER stage + STORAGE read/write access included unconditionally (even
                // though only m_RadianceImage actually carries STORAGE_BIT usage): this same
                // barrier covers all 6 atlas images, and SurfaceCacheGIInject.comp's
                // imageLoad/imageStore of the radiance atlas is exactly the access this transition
                // must make visible-from/available-to, on top of every image's existing sampled-
                // read consumers (SWRT/HWRT trace shaders, SurfaceCacheCapture.frag's own MRT
                // writes on a later RecordCapture() call).
                TransitionImageLayout(cmd, entry.image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }

            TransitionImageLayout(cmd, m_DepthImage, VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

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
        // STEP 4 -- Lighting descriptor set (set 0): binding 0 = SurfaceCacheLightingUBO (written
        // every frame by UpdateLighting(), persistently mapped so that is a plain memcpy), binding
        // 1 = the sun's shadow map (bound once by SetShadowMap(), called after ShadowMapPass::Init()
        // since the shadow view/sampler live in that separate pass -- see SetShadowMap()'s own
        // comment). The UBO's descriptor is written right here, since m_LightingUBO's VkBuffer
        // handle is already known and never changes for the rest of this pass's lifetime.
        // =====================================================================================
        m_LightingUBO.Create(allocator, sizeof(SurfaceCacheLightingUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        VkDescriptorSetLayoutBinding lightingBindings[2]{};
        lightingBindings[0].binding = 0;
        lightingBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lightingBindings[0].descriptorCount = 1;
        lightingBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        lightingBindings[1].binding = 1;
        lightingBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        lightingBindings[1].descriptorCount = 1;
        lightingBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo lightingSetLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lightingSetLayoutInfo.bindingCount = 2;
        lightingSetLayoutInfo.pBindings = lightingBindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &lightingSetLayoutInfo, nullptr, &m_LightingSetLayout));

        VkDescriptorPoolSize lightingPoolSizes[2]{};
        lightingPoolSizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
        lightingPoolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
        VkDescriptorPoolCreateInfo lightingPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        lightingPoolInfo.maxSets = 1;
        lightingPoolInfo.poolSizeCount = 2;
        lightingPoolInfo.pPoolSizes = lightingPoolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &lightingPoolInfo, nullptr, &m_LightingDescriptorPool));

        VkDescriptorSetAllocateInfo lightingSetAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        lightingSetAllocInfo.descriptorPool = m_LightingDescriptorPool;
        lightingSetAllocInfo.descriptorSetCount = 1;
        lightingSetAllocInfo.pSetLayouts = &m_LightingSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &lightingSetAllocInfo, &m_LightingDescriptorSet));

        VkDescriptorBufferInfo lightingBufferInfo{};
        lightingBufferInfo.buffer = m_LightingUBO.Handle();
        lightingBufferInfo.offset = 0;
        lightingBufferInfo.range = sizeof(SurfaceCacheLightingUBO);

        VkWriteDescriptorSet lightingUboWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        lightingUboWrite.dstSet = m_LightingDescriptorSet;
        lightingUboWrite.dstBinding = 0;
        lightingUboWrite.descriptorCount = 1;
        lightingUboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lightingUboWrite.pBufferInfo = &lightingBufferInfo;
        vkUpdateDescriptorSets(m_Device, 1, &lightingUboWrite, 0, nullptr);
        // Binding 1 (the shadow map sampler) is intentionally left unwritten here -- SetShadowMap()
        // writes it once the caller has a ShadowMapPass to bind (see that method's own comment).
        // Validation layers correctly flag sampling through an unwritten descriptor, but nothing
        // samples set 0 binding 1 until SetShadowMap() has run, which every caller must do before
        // its first RecordCapture() call (documented on both methods).

        // =====================================================================================
        // STEP 5 -- Capture pipeline: plain vertex-buffer input (geometry::FallbackVertex), set 0
        // = the lighting descriptor set above, 6-attachment MRT (albedo/normal/emissive/direct-
        // lighting/radiance/world-position) + depth test.
        // =====================================================================================
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SurfaceCaptureConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &m_LightingSetLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));

        std::vector<char> vertCode = ReadShaderFile("shaders/SurfaceCacheCapture.vert.spv");
        std::vector<char> fragCode = ReadShaderFile("shaders/SurfaceCacheCapture.frag.spv");
        VkShaderModule vertModule = CreateShaderModule(m_Device, vertCode);
        VkShaderModule fragModule = CreateShaderModule(m_Device, fragCode);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragModule;
        stages[1].pName = "main";

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

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        // Same cull convention as VulkanPipeline::CreateGraphicsPipeline: mat4::OrthoVulkan
        // negates Y exactly like PerspectiveVulkan does, so the same apparent-winding flip applies
        // -- object-space CCW-wound triangles arrive CLOCKWISE on screen.
        VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachments[6]{};
        for (auto& attachment : colorBlendAttachments) {
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            attachment.blendEnable = VK_FALSE;
        }
        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.attachmentCount = 6;
        colorBlending.pAttachments = colorBlendAttachments;

        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkFormat colorFormats[6] = { kAlbedoFormat, kNormalFormat, kEmissiveFormat, kDirectLightingFormat, kRadianceFormat, kWorldPosFormat };
        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 6;
        pipelineRendering.pColorAttachmentFormats = colorFormats;
        pipelineRendering.depthAttachmentFormat = kDepthFormat;

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
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.pNext = &pipelineRendering;
        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));

        vkDestroyShaderModule(m_Device, vertModule, nullptr);
        vkDestroyShaderModule(m_Device, fragModule, nullptr);

        return true;
    }

    void SurfaceCachePass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            // Destroying the pool implicitly frees every set allocated from it (m_LightingDescriptorSet).
            if (m_LightingDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_LightingDescriptorPool, nullptr);
            if (m_LightingSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_LightingSetLayout, nullptr);
            if (m_AtlasSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_AtlasSampler, nullptr);
            if (m_AlbedoView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_AlbedoView, nullptr);
            if (m_NormalView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_NormalView, nullptr);
            if (m_EmissiveView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_EmissiveView, nullptr);
            if (m_DirectLightingView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_DirectLightingView, nullptr);
            if (m_RadianceView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_RadianceView, nullptr);
            if (m_WorldPosView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_WorldPosView, nullptr);
            if (m_DepthView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_DepthView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            if (m_AlbedoImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_AlbedoImage, m_AlbedoAllocation);
            if (m_NormalImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_NormalImage, m_NormalAllocation);
            if (m_EmissiveImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_EmissiveImage, m_EmissiveAllocation);
            if (m_DirectLightingImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_DirectLightingImage, m_DirectLightingAllocation);
            if (m_RadianceImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_RadianceImage, m_RadianceAllocation);
            if (m_WorldPosImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_WorldPosImage, m_WorldPosAllocation);
            if (m_DepthImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_DepthImage, m_DepthAllocation);
        }
        m_VertexBuffer.Destroy();
        m_IndexBuffer.Destroy();
        m_LightingUBO.Destroy();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_LightingDescriptorPool = VK_NULL_HANDLE;
        m_LightingSetLayout = VK_NULL_HANDLE;
        m_LightingDescriptorSet = VK_NULL_HANDLE;
        m_AtlasSampler = VK_NULL_HANDLE;
        m_AlbedoImage = VK_NULL_HANDLE; m_AlbedoAllocation = VK_NULL_HANDLE; m_AlbedoView = VK_NULL_HANDLE;
        m_NormalImage = VK_NULL_HANDLE; m_NormalAllocation = VK_NULL_HANDLE; m_NormalView = VK_NULL_HANDLE;
        m_EmissiveImage = VK_NULL_HANDLE; m_EmissiveAllocation = VK_NULL_HANDLE; m_EmissiveView = VK_NULL_HANDLE;
        m_DirectLightingImage = VK_NULL_HANDLE; m_DirectLightingAllocation = VK_NULL_HANDLE; m_DirectLightingView = VK_NULL_HANDLE;
        m_RadianceImage = VK_NULL_HANDLE; m_RadianceAllocation = VK_NULL_HANDLE; m_RadianceView = VK_NULL_HANDLE;
        m_WorldPosImage = VK_NULL_HANDLE; m_WorldPosAllocation = VK_NULL_HANDLE; m_WorldPosView = VK_NULL_HANDLE;
        m_DepthImage = VK_NULL_HANDLE; m_DepthAllocation = VK_NULL_HANDLE; m_DepthView = VK_NULL_HANDLE;
        m_Cards.clear();
        m_CardStates.clear();
        m_EntityRanges.clear();
        m_AtlasAllocator.Reset();
        m_DirtyCardQueue.clear();
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void SurfaceCachePass::SetShadowMap(VkImageView shadowMapView, VkSampler shadowMapSampler) {
        VkDescriptorImageInfo shadowImageInfo{};
        shadowImageInfo.sampler = shadowMapSampler;
        shadowImageInfo.imageView = shadowMapView;
        shadowImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL; // Still a valid sampled layout, matching renderer::ShadowMapPass's own permanent-layout convention.

        VkWriteDescriptorSet shadowWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        shadowWrite.dstSet = m_LightingDescriptorSet;
        shadowWrite.dstBinding = 1;
        shadowWrite.descriptorCount = 1;
        shadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowWrite.pImageInfo = &shadowImageInfo;
        vkUpdateDescriptorSets(m_Device, 1, &shadowWrite, 0, nullptr);
    }

    void SurfaceCachePass::UpdateLighting(const SceneLights& lights, const maths::mat4& lightViewProj) {
        SurfaceCacheLightingUBO ubo{};
        ubo.lightViewProj = lightViewProj;
        ubo.sunDirectionAndIntensity[0] = lights.sun.direction.x;
        ubo.sunDirectionAndIntensity[1] = lights.sun.direction.y;
        ubo.sunDirectionAndIntensity[2] = lights.sun.direction.z;
        ubo.sunDirectionAndIntensity[3] = lights.sun.intensity;
        ubo.sunColor[0] = lights.sun.color.x;
        ubo.sunColor[1] = lights.sun.color.y;
        ubo.sunColor[2] = lights.sun.color.z;
        ubo.sunColor[3] = 0.0f;

        const uint32_t pointLightCount = std::min(lights.pointLightCount, kMaxPointLights);
        ubo.pointLightCount = pointLightCount;
        for (uint32_t i = 0; i < pointLightCount; ++i) {
            const PointLight& src = lights.pointLights[i];
            PointLightGPU& dst = ubo.pointLights[i];
            dst.positionAndRadius[0] = src.position.x;
            dst.positionAndRadius[1] = src.position.y;
            dst.positionAndRadius[2] = src.position.z;
            dst.positionAndRadius[3] = src.radius;
            dst.colorAndIntensity[0] = src.color.x;
            dst.colorAndIntensity[1] = src.color.y;
            dst.colorAndIntensity[2] = src.color.z;
            dst.colorAndIntensity[3] = src.intensity;
        }

        std::memcpy(m_LightingUBO.MappedData(), &ubo, sizeof(ubo));
    }

    void SurfaceCachePass::ApplyCardPlacement(uint32_t cardIndex, const geometry::AtlasRect& paddedRect) {
        CardRuntimeState& state = m_CardStates[cardIndex];
        state.resident = true;
        state.rect = paddedRect;
        if (!state.dirty) {
            state.dirty = true;
            m_DirtyCardQueue.push_back(cardIndex);
        }

        // Stamp the unpadded (gutter-excluded) placement into the public, GI-facing card entry --
        // see GetCards()'s own comment on why atlasOffset/uvMin/uvMax are dynamic now.
        geometry::SurfaceCacheCardEntry& card = m_Cards[cardIndex];
        card.atlasOffset[0] = paddedRect.x;
        card.atlasOffset[1] = paddedRect.y;
        const float invAtlasSize = 1.0f / static_cast<float>(geometry::kSurfaceCacheAtlasSize);
        card.uvMin[0] = static_cast<float>(card.atlasOffset[0]) * invAtlasSize;
        card.uvMin[1] = static_cast<float>(card.atlasOffset[1]) * invAtlasSize;
        card.uvMax[0] = static_cast<float>(card.atlasOffset[0] + card.atlasSize[0]) * invAtlasSize;
        card.uvMax[1] = static_cast<float>(card.atlasOffset[1] + card.atlasSize[1]) * invAtlasSize;
    }

    void SurfaceCachePass::EvictAllUnwantedCards() {
        for (size_t i = 0; i < m_CardStates.size(); ++i) {
            CardRuntimeState& state = m_CardStates[i];
            if (state.resident && state.framesSinceVisible > 0) {
                m_AtlasAllocator.Free(state.rect);
                state.resident = false;
                state.rect = geometry::AtlasRect{};
            }
        }
    }

    void SurfaceCachePass::DefragmentAtlas() {
        // Gather every still-resident card index, largest gutter-padded footprint first (mirrors
        // CardGenerator::PackCardsIntoAtlas's own tallest-first shelf order -- packing large pieces
        // before small ones is what keeps guillotine splits from stranding large free rects as
        // unusable slivers by the time small cards are placed).
        std::vector<uint32_t> residentIndices;
        for (uint32_t i = 0; i < m_CardStates.size(); ++i) {
            if (m_CardStates[i].resident) {
                residentIndices.push_back(i);
            }
        }
        std::sort(residentIndices.begin(), residentIndices.end(), [this](uint32_t a, uint32_t b) {
            const uint32_t heightA = m_Cards[a].atlasSize[1];
            const uint32_t heightB = m_Cards[b].atlasSize[1];
            if (heightA != heightB) return heightA > heightB;
            return m_Cards[a].atlasSize[0] > m_Cards[b].atlasSize[0];
            });

        m_AtlasAllocator.Reset();

        for (uint32_t cardIndex : residentIndices) {
            const geometry::SurfaceCacheCardEntry& card = m_Cards[cardIndex];
            const uint32_t paddedW = card.atlasSize[0] + geometry::kCardGutterTexels;
            const uint32_t paddedH = card.atlasSize[1] + geometry::kCardGutterTexels;

            geometry::AtlasRect newRect{};
            if (m_AtlasAllocator.Allocate(paddedW, paddedH, newRect)) {
                // Repacked at a new atlas location: its previously captured texels (if any) are now
                // stale/orphaned at the OLD location, so it must be (re-)captured before a GI pass
                // ever samples the new one -- ApplyCardPlacement() queues that automatically.
                ApplyCardPlacement(cardIndex, newRect);
            }
            else {
                // Should not happen in practice: this card already fit before the repack (it was
                // resident), and repacking largest-first from an empty atlas can only pack at
                // least as tightly as whatever incremental sequence of Allocate()/Free() calls
                // preceded it. Guard against it anyway rather than leave inconsistent state.
                LOG_WARNING(std::format(
                    "[SurfaceCachePass] Card {} (entityID={}) no longer fits the atlas after defragmentation; evicting.",
                    cardIndex, card.entityID));
                m_CardStates[cardIndex].resident = false;
                m_CardStates[cardIndex].rect = geometry::AtlasRect{};
            }
        }
    }

    bool SurfaceCachePass::AllocateCardPage(uint32_t cardIndex, geometry::AtlasRect& outRect) {
        const geometry::SurfaceCacheCardEntry& card = m_Cards[cardIndex];
        const uint32_t paddedW = card.atlasSize[0] + geometry::kCardGutterTexels;
        const uint32_t paddedH = card.atlasSize[1] + geometry::kCardGutterTexels;

        if (m_AtlasAllocator.Allocate(paddedW, paddedH, outRect)) {
            return true;
        }

        // First fallback: evict every currently-unwanted resident card (not just ones already past
        // kEvictionFrameDelay) and retry -- a card that just entered the screen this very call is
        // worth more than a card that just left it a moment ago.
        EvictAllUnwantedCards();
        if (m_AtlasAllocator.Allocate(paddedW, paddedH, outRect)) {
            return true;
        }

        // Second fallback: the remaining resident set's free area may still add up to enough room
        // for this card, just fragmented across rects each individually smaller than (paddedW,
        // paddedH) -- rebuild from scratch, tightest-first. See geometry::SurfaceCacheAtlasAllocator's
        // own class comment on why Guillotine packing alone cannot always avoid this.
        DefragmentAtlas();
        return m_AtlasAllocator.Allocate(paddedW, paddedH, outRect);
    }

    void SurfaceCachePass::UpdateVisibility(const maths::vec3& cameraPosition, const maths::vec3& cameraForward,
        const maths::vec3& cameraUp, float fovYRadians, float aspectRatio, float nearZ, float farZ) {

        // Orthonormal camera basis -- same right/up derivation as maths::mat4::LookAt (right =
        // forward x upHint, up = right x forward), so a caller passing the exact vectors it also
        // feeds LookAt gets a frustum that matches its own view matrix exactly.
        const maths::vec3 forward = cameraForward.Normalize();
        const maths::vec3 right = forward.Cross(cameraUp).Normalize();
        const maths::vec3 up = right.Cross(forward);

        // Standard "camera-parameters" frustum-plane derivation (6 inward-facing point-normal
        // planes built directly from the far-plane rectangle, no matrix inversion required): see
        // https://learnopengl.com/Guest-Articles/2021/Scene/Frustum-Culling for the geometric
        // derivation this mirrors exactly.
        const float halfVSide = farZ * std::tan(fovYRadians * 0.5f);
        const float halfHSide = halfVSide * aspectRatio;
        const maths::vec3 frontMultFar = forward * farZ;

        struct Plane {
            maths::vec3 normal; // Points INTO the frustum.
            maths::vec3 point;  // Any point that lies on the plane.
        };
        const Plane planes[6] = {
            { forward, cameraPosition + forward * nearZ },                                    // Near.
            { forward * -1.0f, cameraPosition + frontMultFar },                                // Far.
            { (frontMultFar - right * halfHSide).Cross(up), cameraPosition },                  // Right.
            { up.Cross(frontMultFar + right * halfHSide), cameraPosition },                    // Left.
            { right.Cross(frontMultFar - up * halfVSide), cameraPosition },                    // Top.
            { (frontMultFar + up * halfVSide).Cross(right), cameraPosition },                  // Bottom.
        };

        for (uint32_t i = 0; i < static_cast<uint32_t>(m_Cards.size()); ++i) {
            const geometry::SurfaceCacheCardEntry& card = m_Cards[i];
            // Every card of one entity shares the entity's full AABB (see SurfaceCacheCardEntry's
            // own comment in ClusterFormat.h) -- testing it directly is exactly an entity-level
            // visibility test, done once per card rather than once per entity, which is simpler
            // than threading a separate per-entity result through and immaterial in cost at this
            // scale (at most kMaxCardsPerEntity redundant, cheap AABB-vs-6-planes tests per entity).
            const maths::vec3 boundsMin{ card.localBoundsMin[0], card.localBoundsMin[1], card.localBoundsMin[2] };
            const maths::vec3 boundsMax{ card.localBoundsMax[0], card.localBoundsMax[1], card.localBoundsMax[2] };

            bool visible = true;
            for (const Plane& plane : planes) {
                // The AABB corner furthest along the plane's inward normal ("positive vertex"): if
                // even THAT corner is on the outside half-space, the whole AABB is outside.
                maths::vec3 positive;
                positive.x = plane.normal.x >= 0.0f ? boundsMax.x : boundsMin.x;
                positive.y = plane.normal.y >= 0.0f ? boundsMax.y : boundsMin.y;
                positive.z = plane.normal.z >= 0.0f ? boundsMax.z : boundsMin.z;
                if (plane.normal.Dot(positive - plane.point) < 0.0f) {
                    visible = false;
                    break;
                }
            }

            CardRuntimeState& state = m_CardStates[i];
            if (visible) {
                state.framesSinceVisible = 0;
                if (!state.resident) {
                    geometry::AtlasRect rect{};
                    if (AllocateCardPage(i, rect)) {
                        ApplyCardPlacement(i, rect);
                    }
                    else {
                        // Atlas genuinely oversubscribed by this frame's visible set even after
                        // evicting everything unwanted and fully defragmenting -- documented
                        // graceful degradation (see UpdateVisibility()'s own header comment): this
                        // card simply stays non-resident and is retried next call, instead of
                        // crashing or growing the atlas at runtime.
                        LOG_WARNING(std::format(
                            "[SurfaceCachePass] Atlas exhausted: could not allocate a page for card {} (entityID={}) this frame.",
                            i, card.entityID));
                    }
                }
            }
            else {
                ++state.framesSinceVisible;
                if (state.resident && state.framesSinceVisible > kEvictionFrameDelay) {
                    m_AtlasAllocator.Free(state.rect);
                    state.resident = false;
                    state.rect = geometry::AtlasRect{};
                }
            }
        }
    }

    void SurfaceCachePass::RecordCapture(VkCommandBuffer cmd) {
        if (m_DirtyCardQueue.empty()) {
            return;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout, 0, 1, &m_LightingDescriptorSet, 0, nullptr);
        VkBuffer vertexBuffer = m_VertexBuffer.Handle();
        VkDeviceSize vertexOffset0 = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset0);
        vkCmdBindIndexBuffer(cmd, m_IndexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

        VkRenderingAttachmentInfo colorAttachments[6]{};
        colorAttachments[0].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachments[0].imageView = m_AlbedoView;
        colorAttachments[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorAttachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachments[0].clearValue.color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
        colorAttachments[1] = colorAttachments[0];
        colorAttachments[1].imageView = m_NormalView;
        colorAttachments[1].clearValue.color = { { 0.5f, 0.5f, 0.0f, 1.0f } };
        colorAttachments[2] = colorAttachments[0];
        colorAttachments[2].imageView = m_EmissiveView;
        colorAttachments[3] = colorAttachments[0];
        colorAttachments[3].imageView = m_DirectLightingView;
        // Radiance/world-position: same LOAD_OP_CLEAR + STORE discipline as the other 4 -- every
        // (re-)capture of a card fully re-seeds its radiance with the fresh
        // emissive+albedo*directLighting value (SurfaceCacheCapture.frag), discarding whatever a
        // prior GI injection pass had accumulated there. This is deliberate: a card is only ever
        // re-captured because its underlying material/lighting data changed (or it was just
        // (re-)placed in the atlas -- see ApplyCardPlacement()), at which point a stale
        // accumulated bounce is no longer trustworthy either -- the next
        // SurfaceCacheGIInject.comp pass over this card rebuilds it from the fresh seed.
        colorAttachments[4] = colorAttachments[0];
        colorAttachments[4].imageView = m_RadianceView;
        colorAttachments[5] = colorAttachments[0];
        colorAttachments[5].imageView = m_WorldPosView;
        colorAttachments[5].clearValue.color = { { 0.0f, 0.0f, 0.0f, 0.0f } };

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = m_DepthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // Scratch, disjoint per-card -- never read back across cards.
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        uint32_t capturedCount = 0;
        while (capturedCount < kCardsPerFrameBudget && !m_DirtyCardQueue.empty()) {
            const uint32_t cardIndex = m_DirtyCardQueue.front();
            m_DirtyCardQueue.pop_front();

            CardRuntimeState& state = m_CardStates[cardIndex];
            if (!state.resident || !state.dirty) {
                // Evicted (DefragmentAtlas()/EvictAllUnwantedCards()) after being queued but before
                // its turn came up -- nothing left to capture for it; does not count against this
                // call's budget, since no actual capture draw was recorded.
                continue;
            }

            const geometry::SurfaceCacheCardEntry& card = m_Cards[cardIndex];

            auto rangeIt = m_EntityRanges.find(card.entityID);
            if (rangeIt == m_EntityRanges.end()) {
                LOG_WARNING(std::format(
                    "[SurfaceCachePass] Card {} references entityID={} with no fallback mesh; skipping.",
                    cardIndex, card.entityID));
                state.dirty = false;
                continue;
            }
            const EntityDrawRange& range = rangeIt->second;

            VkRect2D cardRect{};
            cardRect.offset = { static_cast<int32_t>(state.rect.x), static_cast<int32_t>(state.rect.y) };
            cardRect.extent = { card.atlasSize[0], card.atlasSize[1] };

            VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            renderingInfo.renderArea = cardRect;
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 6;
            renderingInfo.pColorAttachments = colorAttachments;
            renderingInfo.pDepthAttachment = &depthAttachment;
            vkCmdBeginRendering(cmd, &renderingInfo);

            VkViewport viewport{};
            viewport.x = static_cast<float>(cardRect.offset.x);
            viewport.y = static_cast<float>(cardRect.offset.y);
            viewport.width = static_cast<float>(cardRect.extent.width);
            viewport.height = static_cast<float>(cardRect.extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &cardRect);

            SurfaceCaptureConstants pc{};
            pc.viewProj = BuildCardViewProj(card);
            pc.entityID = card.entityID;
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

            vkCmdDrawIndexed(cmd, range.indexCount, 1, range.firstIndex, range.vertexOffset, 0);

            vkCmdEndRendering(cmd);

            state.dirty = false;
            ++capturedCount;
        }

        if (capturedCount == 0) {
            return; // Every queued entry this call drained was stale (see above) -- nothing to make visible.
        }

        // Makes every texel captured this call visible to a later sampled read (a future GI
        // pass) -- no layout transition, the atlas images stay in GENERAL for their whole
        // lifetime (see the class comment).
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        // SHADER_STORAGE_READ: SurfaceCacheGIInject.comp's imageLoad of the radiance atlas it is
        // about to read-modify-write, on top of every atlas's existing sampled-read consumers
        // (SWRT/HWRT trace shaders' card sampling).
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

}
