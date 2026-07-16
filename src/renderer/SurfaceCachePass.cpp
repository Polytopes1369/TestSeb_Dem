#include "renderer/SurfaceCachePass.h"

#include <algorithm>
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
        // STEP 2 -- GPU resources: combined vertex/index buffers, 3 atlas images + 1 shared depth
        // image, one shared sampler, the capture pipeline.
        // =====================================================================================
        VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(hostVertices.size()) * sizeof(geometry::FallbackVertex);
        VkDeviceSize indexBytes = static_cast<VkDeviceSize>(hostIndices.size()) * sizeof(uint32_t);

        if (vertexBytes > 0 && indexBytes > 0) {
            m_VertexBuffer.Create(allocator, vertexBytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_IndexBuffer.Create(allocator, indexBytes,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
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
        // the 3 atlas images to a neutral default, and transition every image to its permanent
        // layout (GENERAL for the 3 atlas images, DEPTH_ATTACHMENT_OPTIMAL for the depth image --
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
            VkImageSubresourceRange colorRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            struct { VkImage image; const VkClearColorValue* clear; } atlasClears[3] = {
                { m_AlbedoImage, &albedoClear }, { m_NormalImage, &normalClear }, { m_EmissiveImage, &emissiveClear }
            };
            for (auto& entry : atlasClears) {
                TransitionImageLayout(cmd, entry.image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                vkCmdClearColorImage(cmd, entry.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, entry.clear, 1, &colorRange);
                TransitionImageLayout(cmd, entry.image, VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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
        // STEP 4 -- Capture pipeline: plain vertex-buffer input (geometry::FallbackVertex), no
        // descriptor sets at all (no SSBO/sampler reads -- pure push-constant-driven procedural
        // shading, see SurfaceCacheCapture.frag), 3-attachment MRT + depth test.
        // =====================================================================================
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(SurfaceCaptureConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 0;
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

        VkPipelineColorBlendAttachmentState colorBlendAttachments[3]{};
        for (auto& attachment : colorBlendAttachments) {
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            attachment.blendEnable = VK_FALSE;
        }
        VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.attachmentCount = 3;
        colorBlending.pAttachments = colorBlendAttachments;

        std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkFormat colorFormats[3] = { kAlbedoFormat, kNormalFormat, kEmissiveFormat };
        VkPipelineRenderingCreateInfo pipelineRendering{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        pipelineRendering.colorAttachmentCount = 3;
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
            if (m_AtlasSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_AtlasSampler, nullptr);
            if (m_AlbedoView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_AlbedoView, nullptr);
            if (m_NormalView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_NormalView, nullptr);
            if (m_EmissiveView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_EmissiveView, nullptr);
            if (m_DepthView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_DepthView, nullptr);
        }
        if (m_Allocator != VK_NULL_HANDLE) {
            if (m_AlbedoImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_AlbedoImage, m_AlbedoAllocation);
            if (m_NormalImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_NormalImage, m_NormalAllocation);
            if (m_EmissiveImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_EmissiveImage, m_EmissiveAllocation);
            if (m_DepthImage != VK_NULL_HANDLE) vmaDestroyImage(m_Allocator, m_DepthImage, m_DepthAllocation);
        }
        m_VertexBuffer.Destroy();
        m_IndexBuffer.Destroy();

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_AtlasSampler = VK_NULL_HANDLE;
        m_AlbedoImage = VK_NULL_HANDLE; m_AlbedoAllocation = VK_NULL_HANDLE; m_AlbedoView = VK_NULL_HANDLE;
        m_NormalImage = VK_NULL_HANDLE; m_NormalAllocation = VK_NULL_HANDLE; m_NormalView = VK_NULL_HANDLE;
        m_EmissiveImage = VK_NULL_HANDLE; m_EmissiveAllocation = VK_NULL_HANDLE; m_EmissiveView = VK_NULL_HANDLE;
        m_DepthImage = VK_NULL_HANDLE; m_DepthAllocation = VK_NULL_HANDLE; m_DepthView = VK_NULL_HANDLE;
        m_Cards.clear();
        m_EntityRanges.clear();
        m_CaptureCursor = 0;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void SurfaceCachePass::RecordCapture(VkCommandBuffer cmd) {
        if (m_Cards.empty()) {
            return;
        }

        const uint32_t cardCount = static_cast<uint32_t>(m_Cards.size());
        const uint32_t sliceCount = std::min(kCardsPerFrameBudget, cardCount);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
        VkBuffer vertexBuffer = m_VertexBuffer.Handle();
        VkDeviceSize vertexOffset0 = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset0);
        vkCmdBindIndexBuffer(cmd, m_IndexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

        VkRenderingAttachmentInfo colorAttachments[3]{};
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

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = m_DepthView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // Scratch, disjoint per-card -- never read back across cards.
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        for (uint32_t i = 0; i < sliceCount; ++i) {
            const uint32_t cardIndex = (m_CaptureCursor + i) % cardCount;
            const geometry::SurfaceCacheCardEntry& card = m_Cards[cardIndex];

            auto rangeIt = m_EntityRanges.find(card.entityID);
            if (rangeIt == m_EntityRanges.end()) {
                LOG_WARNING(std::format(
                    "[SurfaceCachePass] Card {} references entityID={} with no fallback mesh; skipping.",
                    cardIndex, card.entityID));
                continue;
            }
            const EntityDrawRange& range = rangeIt->second;

            VkRect2D cardRect{};
            cardRect.offset = { static_cast<int32_t>(card.atlasOffset[0]), static_cast<int32_t>(card.atlasOffset[1]) };
            cardRect.extent = { card.atlasSize[0], card.atlasSize[1] };

            VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
            renderingInfo.renderArea = cardRect;
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 3;
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
        }

        m_CaptureCursor = (m_CaptureCursor + sliceCount) % cardCount;

        // Makes every texel captured this call visible to a later sampled read (a future GI
        // pass) -- no layout transition, the atlas images stay in GENERAL for their whole
        // lifetime (see the class comment).
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

}
