#include "renderer/passes/ShadowMapPass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <stdexcept>

#include "core/Logger.h"
#include "geometry/ClusterFormat.h"
#include "io/CacheFileManager.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        struct ShadowCaptureConstants {
            maths::mat4 lightViewProj;
        };
        static_assert(sizeof(ShadowCaptureConstants) == 64,
            "ShadowCaptureConstants must match ShadowMapCapture.vert's push_constant block exactly");



    } // namespace

    bool ShadowMapPass::InitImpl(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue,
        const std::filesystem::path& cacheFilePath) {
        // Self-reinit: releases any previously-registered resources (a no-op on the very first call,
        // since m_Cleanups starts empty) before rebuilding below. Shutdown() also clears
        // m_Device/m_Allocator, which RenderPass<ShadowMapPass>::Init() already set to this call's
        // values just before invoking this function -- re-assign them here to restore that.
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // STEP 1 -- Read every fallback mesh's geometry (CPU side) and derive the scene's
        // world-space bounding sphere from the index table's per-entity AABBs.
        // =====================================================================================
        geometry::CacheFileManager cacheManager;
        geometry::CacheFileHeader header{};
        if (!cacheManager.ReadHeader(cacheFilePath, header)) {
            LOG_ERROR(std::format("[ShadowMapPass] Failed to read cache header '{}'.", cacheFilePath.string()));
            return false;
        }

        std::vector<geometry::FallbackMeshIndexEntry> fallbackTable;
        if (!cacheManager.ReadFallbackMeshTable(cacheFilePath, header, fallbackTable)) {
            LOG_ERROR("[ShadowMapPass] Failed to read the fallback-mesh table.");
            return false;
        }

        maths::vec3 sceneBoundsMin{ std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        maths::vec3 sceneBoundsMax{ -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max() };

        std::vector<geometry::FallbackVertex> hostVertices;
        std::vector<uint32_t> hostIndices;
        for (const geometry::FallbackMeshIndexEntry& entry : fallbackTable) {
            std::vector<geometry::FallbackVertex> entityVertices;
            std::vector<uint32_t> entityIndices;
            if (!cacheManager.ReadFallbackMeshGeometry(cacheFilePath, entry, entityVertices, entityIndices)) {
                LOG_ERROR(std::format(
                    "[ShadowMapPass] Failed to read fallback mesh geometry for entityID={}.", entry.entityID));
                return false;
            }

            const uint32_t indexBase = static_cast<uint32_t>(hostVertices.size());
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
            // No geometry at all: degenerate bounding sphere at the origin (see class comment).
            m_SceneBoundsCenter = maths::vec3{ 0.0f, 0.0f, 0.0f };
            m_SceneBoundsRadius = 0.0f;
        } else {
            m_SceneBoundsCenter = (sceneBoundsMin + sceneBoundsMax) * 0.5f;
            m_SceneBoundsRadius = (sceneBoundsMax - sceneBoundsMin).Length() * 0.5f;
        }

        LOG_INFO(std::format(
            "[ShadowMapPass] Loaded {} fallback mesh(es) ({} vertices, {} indices); scene bounding sphere radius={:.3f}.",
            fallbackTable.size(), hostVertices.size(), hostIndices.size(), m_SceneBoundsRadius));

        // =====================================================================================
        // STEP 2 -- GPU resources: combined vertex/index buffers, depth image, capture pipeline.
        // =====================================================================================
        VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(hostVertices.size()) * sizeof(geometry::FallbackVertex);
        VkDeviceSize indexBytes = static_cast<VkDeviceSize>(hostIndices.size()) * sizeof(uint32_t);

        if (vertexBytes > 0 && indexBytes > 0) {
            m_VertexBuffer.Create(allocator, vertexBytes,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            m_IndexBuffer.Create(allocator, indexBytes,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }
        // Registered unconditionally (matching the original Shutdown()'s unconditional
        // m_VertexBuffer.Destroy()/m_IndexBuffer.Destroy() calls): GpuBuffer::Destroy() is a safe
        // no-op on a buffer that was never Create()'d (empty-scene case above).
        RegisterResource([this] {
            m_VertexBuffer.Destroy();
            m_IndexBuffer.Destroy();
        });

        VkImageCreateInfo shadowImageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        shadowImageInfo.imageType = VK_IMAGE_TYPE_2D;
        shadowImageInfo.format = kShadowFormat;
        shadowImageInfo.extent = { kShadowMapSize, kShadowMapSize, 1 };
        shadowImageInfo.mipLevels = 1;
        shadowImageInfo.arrayLayers = 1;
        shadowImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        shadowImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        shadowImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        shadowImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        shadowImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo gpuOnlyAlloc{};
        gpuOnlyAlloc.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &shadowImageInfo, &gpuOnlyAlloc, &m_ShadowImage, &m_ShadowAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_ShadowImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kShadowFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_ShadowView));
        RegisterResource([this] {
            vkDestroyImageView(m_Device, m_ShadowView, nullptr);
            vmaDestroyImage(m_Allocator, m_ShadowImage, m_ShadowAllocation);
            m_ShadowView = VK_NULL_HANDLE;
            m_ShadowImage = VK_NULL_HANDLE;
            m_ShadowAllocation = VK_NULL_HANDLE;
        });

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        // Plain (non-comparison) sampler -- SurfaceCacheCapture.frag does its own manual PCF depth
        // comparison, matching this codebase's existing convention (see HZBPass.cpp's own note:
        // "Plain sampler2D, not shadow/compare sampler").
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        // Sampling past the map's edge must read as "far" (unshadowed), not "near" (spuriously
        // shadowed) -- a max-depth (1.0) opaque-white border achieves that, since the fragment
        // shader's comparison treats a stored depth >= its own depth as visible.
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_ShadowSampler));
        RegisterResource([this] { vkDestroySampler(m_Device, m_ShadowSampler, nullptr); m_ShadowSampler = VK_NULL_HANDLE; });

        // =====================================================================================
        // STEP 3 -- One-time setup command buffer: upload the combined geometry (if any) and
        // transition the depth image to its permanent layout.
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

            VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
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

                // GENERAL, not DEPTH_ATTACHMENT_OPTIMAL: this image is sampled by a LATER, completely
                // separate render pass (SurfaceCacheCapture.frag's shadow lookup), and DEPTH_ATTACHMENT_
                // OPTIMAL is not a valid layout for a sampled-image descriptor. GENERAL is valid for
                // BOTH depth-attachment read/write AND a sampled read, so this image never needs to
                // ping-pong layouts between the two uses -- the exact same trade renderer::
                // SurfaceCachePass's own 3 atlas color images already make (see that class's own
                // "Atlas layout convention" comment).
                VulkanUtils::TransitionImageLayout(cmd, m_ShadowImage,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
            });

            if (stagingBuffer != VK_NULL_HANDLE) {
                vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
            }
        }

        // =====================================================================================
        // STEP 4 -- Depth-only capture pipeline: vertex stage only (no fragment shader -- there is
        // nothing to write but depth), plain vertex-attribute input (position only read; the
        // binding stride still matches geometry::FallbackVertex so the SAME uploaded buffer's
        // layout is consistent, even though only offsetof(position) is consumed).
        // =====================================================================================
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(ShadowCaptureConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 0;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &layoutInfo, nullptr, &m_PipelineLayout));
        RegisterResource([this] { vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr); m_PipelineLayout = VK_NULL_HANDLE; });

        VkShaderModule vertModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ShadowMapCapture.vert.spv");

        VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vertModule;
        stage.pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(geometry::FallbackVertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attribute{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(geometry::FallbackVertex, position) };

        VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = 1;
        vertexInput.pVertexAttributeDescriptions = &attribute;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        // Same cull convention as SurfaceCachePass's own capture pipeline: mat4::OrthoVulkan
        // negates Y like PerspectiveVulkan does, so CCW object-space winding arrives CW on screen.
        VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        // No back-face culling for the shadow map: a card that only ever sees an entity's front
        // faces during Surface Cache capture must still receive a shadow cast by that SAME
        // entity's back faces from the light's point of view (e.g. self-shadowing on a concave
        // shape) -- culling back faces here would silently drop valid occluders.
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
        pipelineRendering.depthAttachmentFormat = kShadowFormat;

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
        VK_CHECK(vkCreateGraphicsPipelines(m_Device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_Pipeline));
        RegisterResource([this] { vkDestroyPipeline(m_Device, m_Pipeline, nullptr); m_Pipeline = VK_NULL_HANDLE; });

        vkDestroyShaderModule(m_Device, vertModule, nullptr);

        // Not Vulkan handles -- but the original hand-written Shutdown() reset these two too (NOT
        // m_SceneBoundsCenter, which it left untouched; preserved exactly as-is here).
        RegisterResource([this] {
            m_TotalIndexCount = 0;
            m_SceneBoundsRadius = 0.0f;
        });

        return true;
    }

    // Shutdown() is inherited from RenderPass<ShadowMapPass>: runs the RegisterResource() cleanups
    // above in reverse (state reset -> pipeline -> pipeline layout -> sampler -> view+image ->
    // vertex/index buffers), the same dependency-safe order the hand-written Shutdown() used.

    void ShadowMapPass::RecordCapture(VkCommandBuffer cmd, const maths::vec3& sunDirection) {
        // Fit an orthographic projection to the scene's bounding sphere from this direction: place
        // the eye back along -sunDirection far enough to clear the whole sphere, look at the
        // sphere's center, size the ortho box to the sphere's diameter on every axis so the
        // projection is valid regardless of which way the sun currently points.
        const maths::vec3 lightDir = sunDirection.Normalize();
        const float radius = std::max(m_SceneBoundsRadius, 1.0e-3f); // Avoid a degenerate (zero-extent) projection for an empty scene.
        const float nearPlane = radius * kNearMarginFactor;
        const float farPlane = radius * (2.0f + kFarMarginFactor);
        const maths::vec3 eye = m_SceneBoundsCenter - lightDir * (radius + nearPlane);

        // mat4::LookAt needs an `up` not parallel to the view direction -- the sun can point
        // straight down/up (e.g. local noon), where world-up would degenerate the cross product.
        maths::vec3 up{ 0.0f, 1.0f, 0.0f };
        if (std::abs(lightDir.Dot(up)) > 0.999f) {
            up = maths::vec3{ 0.0f, 0.0f, 1.0f };
        }

        const maths::mat4 view = maths::mat4::LookAt(eye, m_SceneBoundsCenter, up);
        const maths::mat4 proj = maths::mat4::OrthoVulkan(radius, radius, nearPlane, farPlane);
        m_LightViewProj = proj * view;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);

        VkRenderingAttachmentInfo depthAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView = m_ShadowView;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // See Init()'s comment on why GENERAL, not DEPTH_ATTACHMENT_OPTIMAL.
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderingInfo.renderArea = { { 0, 0 }, { kShadowMapSize, kShadowMapSize } };
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = &depthAttachment;
        vkCmdBeginRendering(cmd, &renderingInfo);

        VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(kShadowMapSize), static_cast<float>(kShadowMapSize), 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{ { 0, 0 }, { kShadowMapSize, kShadowMapSize } };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        if (m_TotalIndexCount > 0) {
            ShadowCaptureConstants pc{};
            pc.lightViewProj = m_LightViewProj;
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            VkBuffer vertexBuffer = m_VertexBuffer.Handle();
            VkDeviceSize vertexOffset0 = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset0);
            vkCmdBindIndexBuffer(cmd, m_IndexBuffer.Handle(), 0, VK_INDEX_TYPE_UINT32);

            // A single draw covers every entity: entities are static (local space == world space,
            // see class comment), so one light view-projection is valid for the whole combined
            // buffer at once -- no per-entity draw needed.
            vkCmdDrawIndexed(cmd, m_TotalIndexCount, 1, 0, 0, 0);
        }

        vkCmdEndRendering(cmd);

        // Makes this call's depth writes visible to a later fragment-shader sampled read (the
        // Surface Cache capture pass's shadow lookup).
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

}
