#include "renderer/passes/ClusterResolvePass.h"

#include <array>
#include <cstring>

#include "core/Logger.h"
#include "renderer/passes/ClusterShadingBinPass.h"
#include "renderer/MaterialParameterTable.h"
#include "renderer/passes/VirtualShadowMapPass.h"
#include "renderer/streaming/VirtualTextureManager.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Byte-for-byte mirror of ResolveViewParamsUBO in ClusterResolve.comp/ClusterResolveBinned
        // .comp (std140): mat4 (64 bytes) + mat4 (64 bytes, prevViewProj -- DEBUG_VIEW_MOTION_VECTORS'
        // own reprojection) + vec2 (8 bytes) + 2 floats rounding up to a 16-byte boundary (144
        // bytes) + vec3 sunDirection (Phase 3 -- points FROM the light TOWARD the scene, same
        // convention as renderer::DirectionalLight, needed so this shader's direct-lighting term
        // uses the SAME sun direction the shadow was rendered from) + 1 pad float rounding back up
        // to a 16-byte boundary (160 bytes) + vec3 sunColor + 1 float sunIntensity (real LUX, see
        // renderer::DirectionalLight's own comment -- physically-based recalibration, 2026-07-17)
        // rounding up to 176 bytes + vec3 cameraPositionWorld (Substrate integration:
        // EvaluateSubstrateMaterial's view-direction term) + glintIntensityScale (gap G5, which
        // reused what had been that trailing pad float) rounding up to 192 bytes. Gap G6 (horizontal
        // mixing) then appends mixMaskSharpnessScale + 3 explicit pad floats -- the first field to
        // actually grow this UBO, since no dead padding remained to reuse -- rounding it to 208 bytes
        // total. The 2 floats immediately after viewportWidth/viewportHeight were originally dead
        // std140 padding (still needed to round vec2 up to a 16-byte boundary) -- Atmos weather
        // system's surface response extension repurposes them as real data (surfaceWetness/
        // snowCoverage, see ClusterResolve.comp's own field comment) instead of growing the UBO.
        // Wave 2 (UE5.8 caustics/light-function parity): timeSeconds joins the 16-byte block G6's
        // mixMaskSharpnessScale opens below (2 of its 4 floats now real data, 2 remain dead padding)
        // rather than growing the UBO with a further block -- feeds the procedural (texture-free)
        // underwater caustics term and the sun's light-function modulation, both in ClusterResolve.
        // comp/ClusterResolveBinned.comp.
        struct ResolveViewParams {
            maths::mat4 viewProj;
            maths::mat4 prevViewProj;
            float viewportWidth = 0.0f;
            float viewportHeight = 0.0f;
            float surfaceWetness = 0.0f;
            float snowCoverage = 0.0f;
            float sunDirectionX = 0.0f;
            float sunDirectionY = 0.0f;
            float sunDirectionZ = 0.0f;
            // Glint / sparkle (UE5.8 rendering-parity gap G5): Debug-only per-material glintDensity
            // tuning multiplier, occupying what was dead std140 padding after the vec3 sunDirection
            // (same "reuse dead padding" pattern as surfaceWetness/snowCoverage above) -- 1.0 in
            // Release, driven by the Post FX ImGui "Glint Density" slider in Debug. See
            // ClusterResolve.comp's own glintDensityScale field comment.
            float glintDensityScale = 1.0f;
            float sunColorR = 0.0f;
            float sunColorG = 0.0f;
            float sunColorB = 0.0f;
            float sunIntensity = 0.0f;
            // Substrate integration: EvaluateSubstrateMaterial's view-direction term -- see
            // ClusterResolve.comp's own ResolveViewParamsUBO.cameraPositionWorld comment.
            float cameraPositionWorldX = 0.0f;
            float cameraPositionWorldY = 0.0f;
            float cameraPositionWorldZ = 0.0f;
            // Glint / sparkle (UE5.8 rendering-parity gap G5): Debug-only per-material glintIntensity
            // tuning multiplier, occupying what was the trailing dead pad float -- 1.0 in Release,
            // driven by the Post FX ImGui "Glint Intensity" slider in Debug.
            float glintIntensityScale = 1.0f;
            // Substrate horizontal mixing (UE5.8 rendering-parity gap G6): Debug-only per-material
            // blend-sharpness tuning multiplier. The FIRST field to actually GROW this UBO past its
            // previously-fully-packed 192 bytes -- glintIntensityScale above already consumed the last
            // dead pad float, so this opens a fresh std140 16-byte block. 1.0 in Release, driven by the
            // Post FX ImGui "Mix Sharpness" slider in Debug. See ClusterResolve.comp's own field comment.
            float mixMaskSharpnessScale = 1.0f;
            // Wave 2 (UE5.8 caustics/light-function parity): real elapsed time, joining this same
            // freshly-opened block (see this struct's own top comment) -- see ClusterResolve.comp's
            // own timeSeconds field comment.
            float timeSeconds = 0.0f;
            float _padMix1 = 0.0f;
            float _padMix2 = 0.0f;
        };
        static_assert(sizeof(ResolveViewParams) == 208,
            "ResolveViewParams must match ResolveViewParamsUBO in ClusterResolve.comp exactly (std140 layout)");

        // Step 4: byte-for-byte mirror of VirtualTextureVolumeUBO in ClusterResolve.comp/
        // ClusterResolveBinned.comp (std140): vec2 (8 bytes) + vec2 (8 bytes) + 4 floats (16 bytes)
        // = 32 bytes total. Unlike ResolveViewParams, this is filled ONCE by SetVirtualTexture()
        // (the volume bounds never change per-frame), not every RecordResolve() call.
        struct VTVolumeParams {
            float worldMinX = 0.0f;
            float worldMinZ = 0.0f;
            float worldMaxX = 0.0f;
            float worldMaxZ = 0.0f;
            float virtualTextureSize = 0.0f;
            float tileSize = 0.0f;
            float borderSize = 0.0f;
            float _pad0 = 0.0f;
        };
        static_assert(sizeof(VTVolumeParams) == 32,
            "VTVolumeParams must match VirtualTextureVolumeUBO in ClusterResolve.comp exactly (std140 layout)");

    } // namespace

    void ClusterResolvePass::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue, VkExtent2D renderExtent,
        VkBuffer clusterMetadataBuffer, VkBuffer compressedPhysicalPoolBuffer,
        VkImageView hwClusterIDView, VkImageView hwTriangleIDView, VkImageView hwDepthView,
        VkImageView swVisBufferAtomicView, const std::vector<VkDescriptorImageInfo>& maskImageInfos,
        VkBuffer wpoGlobalsBuffer, VkBuffer entityTransformBuffer, VkBuffer entityDataBuffer,
        const std::array<MaterialParameters, kMaxMaterials>& materialTable,
        VkBuffer splineControlPointsBuffer, VkBuffer boneMatricesBuffer) {
        Shutdown();

        m_Device = device;
        m_Allocator = allocator;
        m_RenderExtent = renderExtent;
        uint32_t maskTextureCount = static_cast<uint32_t>(maskImageInfos.size());

        // Retained purely for InitBinnedResolve() (Phase 1b), called later once
        // renderer::ClusterShadingBinPass also exists -- see that method's own comment for why.
        m_ClusterMetadataBuffer = clusterMetadataBuffer;
        m_CompressedPoolBuffer = compressedPhysicalPoolBuffer;
        m_WPOGlobalsBuffer = wpoGlobalsBuffer;
        m_EntityTransformBuffer = entityTransformBuffer;
        m_EntityDataBuffer = entityDataBuffer;
        m_SplineControlPointsBuffer = splineControlPointsBuffer;
        // Skeletal-animation feature: animation::SkeletalAnimator::GetBoneMatricesBuffer() -- same
        // "retained for InitBinnedResolve()" rationale as m_SplineControlPointsBuffer above.
        m_BoneMatricesBuffer = boneMatricesBuffer;
        m_MaskImageInfos = maskImageInfos;

        // --- Output color image: RGBA8 storage image, sized to the render target. ---
        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kOutputColorFormat;
        imageInfo.extent = { renderExtent.width, renderExtent.height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // SAMPLED_BIT/TRANSFER_SRC_BIT: a future consumer may sample this directly or blit it to
        // the swapchain -- neither is wired up by this change (see the class comment), but the
        // usage flags are pre-positioned exactly like VulkanContext's swapchain TRANSFER_DST_BIT
        // was pre-positioned ahead of the VisBuffer conversion that later needed it.
        // COLOR_ATTACHMENT_BIT: renderer::DebugTextOverlay::RecordDraw (Debug-only, see
        // ClusterRenderPipeline.cpp's #ifndef NDEBUG call site) draws the frame's debug HUD text
        // directly into this image via vkCmdBeginRendering, which requires the color-attachment
        // usage bit on top of the storage/sampled/transfer-src ones above -- without it, every
        // frame's HUD draw is a VUID-VkRenderingInfo-colorAttachmentCount-06087 validation error.
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo imageAllocInfo{};
        imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &imageAllocInfo, &m_OutputColorImage, &m_OutputColorAllocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = m_OutputColorImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kOutputColorFormat;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputColorView));

        // --- Minimal GBuffer: normal/depth/albedo/roughness-metallic, same extent, same
        // STORAGE|SAMPLED usage as the color image above (this codebase's downstream GI/lighting
        // passes -- e.g. renderer::ReflectionTrace, renderer::MegaLightsShade -- sample the
        // first 3; only this shader ever writes any of them) minus COLOR_ATTACHMENT_BIT -- none of
        // these 4 images is ever bound as a render attachment, unlike the color image above (see
        // its own usage-flags comment). ---
        VkImageCreateInfo gbufferImageInfo = imageInfo;
        gbufferImageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        gbufferImageInfo.format = kOutputNormalFormat;
        VK_CHECK(vmaCreateImage(allocator, &gbufferImageInfo, &imageAllocInfo, &m_OutputNormalImage, &m_OutputNormalAllocation, nullptr));
        gbufferImageInfo.format = kOutputDepthFormat;
        VK_CHECK(vmaCreateImage(allocator, &gbufferImageInfo, &imageAllocInfo, &m_OutputDepthImage, &m_OutputDepthAllocation, nullptr));
        gbufferImageInfo.format = kOutputAlbedoFormat;
        VK_CHECK(vmaCreateImage(allocator, &gbufferImageInfo, &imageAllocInfo, &m_OutputAlbedoImage, &m_OutputAlbedoAllocation, nullptr));
        gbufferImageInfo.format = kOutputRoughnessMetallicFormat;
        VK_CHECK(vmaCreateImage(allocator, &gbufferImageInfo, &imageAllocInfo, &m_OutputRoughnessMetallicImage, &m_OutputRoughnessMetallicAllocation, nullptr));
        // Substrate integration (Phase S3): materialID GBuffer channel -- see this class's own
        // GetOutputMaterialIDView() comment (ClusterResolvePass.h).
        gbufferImageInfo.format = kOutputMaterialIDFormat;
        VK_CHECK(vmaCreateImage(allocator, &gbufferImageInfo, &imageAllocInfo, &m_OutputMaterialIDImage, &m_OutputMaterialIDAllocation, nullptr));

        viewInfo.image = m_OutputNormalImage;
        viewInfo.format = kOutputNormalFormat;
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputNormalView));
        viewInfo.image = m_OutputDepthImage;
        viewInfo.format = kOutputDepthFormat;
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputDepthView));
        viewInfo.image = m_OutputAlbedoImage;
        viewInfo.format = kOutputAlbedoFormat;
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputAlbedoView));
        viewInfo.image = m_OutputRoughnessMetallicImage;
        viewInfo.format = kOutputRoughnessMetallicFormat;
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputRoughnessMetallicView));
        viewInfo.image = m_OutputMaterialIDImage;
        viewInfo.format = kOutputMaterialIDFormat;
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &m_OutputMaterialIDView));

        // --- One-time UNDEFINED -> GENERAL transition (mirrors HZBPass::Init /
        // ClusterSoftwareRasterPass::Init's own one-shot pattern) -- stays in GENERAL for its
        // entire lifetime, touched only by this class's own compute shader. ---
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier2 barriers[6]{};
            for (auto& barrier : barriers) {
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                barrier.srcAccessMask = 0;
                barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            }
            barriers[0].image = m_OutputColorImage;
            barriers[1].image = m_OutputNormalImage;
            barriers[2].image = m_OutputDepthImage;
            barriers[3].image = m_OutputAlbedoImage;
            barriers[4].image = m_OutputRoughnessMetallicImage;
            barriers[5].image = m_OutputMaterialIDImage;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.imageMemoryBarrierCount = 6;
            depInfo.pImageMemoryBarriers = barriers;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            // --- Material parameter table: renderer::GenerateShowcaseMaterialTable()'s result
            // (passed in via `materialTable`, generated once by VulkanContext -- see this method's
            // own doc comment), not a per-frame value -- filled once, here, in the same one-time
            // command buffer as the image transitions above (no ordering dependency between them,
            // so recording order doesn't matter). Well under vkCmdUpdateBuffer's 65536-byte limit
            // (kMaxMaterials * sizeof(MaterialParameters) = 32 * 336 = 10752 bytes, after gap G6's
            // horizontal-mix additions and the later Substrate iridescence layer grew the struct from
            // 208 to 336 bytes total -- see MaterialParameterTable.h's own static_assert). No intra-
            // command-buffer barrier is needed after this write -- ExecuteOneShotCommands' own
            // vkQueueWaitIdle fully orders it before any later-submitted command buffer's reads,
            // exactly like ClusterRenderPipeline::Init()'s own one-time setup submit.
            m_MaterialParamsBuffer.Create(allocator, sizeof(MaterialParameters) * kMaxMaterials,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            vkCmdUpdateBuffer(cmd, m_MaterialParamsBuffer.Handle(), 0,
                sizeof(MaterialParameters) * kMaxMaterials, materialTable.data());
        });

        // --- Depth sampler: nearest filtering, matching HZBPass's own depth-sampling convention
        // (the shader always reads exact integer texels via texelFetch, no filtering ever actually
        // happens). ---
        m_DepthSampler = VulkanUtils::CreateNearestSampler(m_Device);

        // --- View-params UBO ---
        m_ViewParamsBuffer.Create(
            allocator,
            sizeof(ResolveViewParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

        // Step 4: Virtual Texture volume UBO -- persistently-mapped CPU_TO_GPU, filled once by
        // SetVirtualTexture() (see that method's own comment for why, unlike m_ViewParamsBuffer,
        // this is not re-uploaded every RecordResolve() call).
        m_VTVolumeUBO.Create(allocator, sizeof(VTVolumeParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, /*mapped=*/true);

        // --- Descriptor set layout: 27 bindings, matching ClusterResolve.comp's set = 0 bindings
        // 0..26 exactly (9..11 are the GBuffer outputs, 12 is the WPOGlobalsUBO this shader needs
        // to reapply the same sway deformation the rasterizers already applied, 13 is the material
        // parameter table SSBO, 14 is the roughness/metallic GBuffer output, 15-18 are Phase 3's
        // renderer::VirtualShadowMapPass resources -- see SetVirtualShadowMap()'s own comment -- 19
        // and 20 are the per-entity rotation buffers also consumed by both rasterizers, 21-24 are
        // Step 4's renderer::VirtualTextureManager resources -- see SetVirtualTexture()'s own
        // comment, 25 is Substrate's g_OutputMaterialID, 26 is Phase 1 Nanite advanced's
        // SplineControlPointsSSBO). ---
        VkDescriptorSetLayoutBinding bindings[30]{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // ClusterCullMetadataSSBO
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // CompressedClusterPoolSSBO
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_HWClusterIDImage (r32ui)
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_HWTriangleIDImage (r32ui)
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_HWDepthTexture
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_SWVisBufferAtomic (r64ui)
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_OutputColor (rgba8)
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };         // ResolveViewParamsUBO
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_MaskTextures[]
        bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };          // g_OutputNormal (rg16f)
        bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };        // g_OutputDepth (r32f)
        bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };        // g_OutputAlbedo (rgba8)
        bindings[12] = { 12, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // WPOGlobalsUBO
        bindings[13] = { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // MaterialParamsSSBO
        bindings[14] = { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };        // g_OutputRoughnessMetallic (rg8)
        bindings[15] = { 15, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_ShadowPhysicalAtlas
        bindings[16] = { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // g_ShadowPageTable
        bindings[17] = { 17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // g_ShadowFeedback
        bindings[18] = { 18, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // g_ShadowSunLevels
        bindings[19] = { 19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // EntityTransformBuffer
        bindings[20] = { 20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // EntityDataBuffer
        bindings[21] = { 21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_PageTable
        bindings[22] = { 22, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxPhysicalPools, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_PhysicalPools[]
        bindings[23] = { 23, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // g_VTFeedback
        bindings[24] = { 24, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // VirtualTextureVolumeUBO
        bindings[25] = { 25, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };        // g_OutputMaterialID (r16ui, Substrate integration)
        // Phase 1 (Nanite advanced): binding 26 -- the first free slot past this shader's full
        // 0-25 range (0-24 original + 25 Substrate's g_OutputMaterialID). See ClusterResolve.comp's
        // own identical binding comment.
        bindings[26] = { 26, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };       // SplineControlPointsSSBO
        // Atmos weather system, Subtask 5 -- see SetAtmosCloudLighting()'s own comment.
        bindings[27] = { 27, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_SkyViewLUT
        bindings[28] = { 28, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_CloudShadowMap
        // Skeletal-animation feature: binding 29 -- the first free slot past this shader's full
        // 0-28 range. See ClusterRaster.vert's identical binding comment.
        bindings[29] = { 29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };        // SkeletalBoneMatricesSSBO

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = 30;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSizes[4]{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 };  // + material params, shadow page table, shadow feedback, entity transform, entity data, VT feedback, spline control points, bone matrices.
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9 };   // + g_OutputMaterialID (Substrate integration).
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 + maskTextureCount + kMaxPhysicalPools }; // + g_ShadowPhysicalAtlas, g_PageTable, g_PhysicalPools[], g_SkyViewLUT, g_CloudShadowMap.
        poolSizes[3] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 };  // + g_ShadowSunLevels, VirtualTextureVolumeUBO.

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 4;
        poolInfo.pPoolSizes = poolSizes;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAlloc.descriptorPool = m_DescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &m_SetLayout;
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAlloc, &m_DescriptorSet));

        VkDescriptorBufferInfo clusterMetadataInfo{ clusterMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo compressedPoolInfo{ compressedPhysicalPoolBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorBufferInfo entityTransformInfo{ entityTransformBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo entityDataInfo{ entityDataBuffer, 0, VK_WHOLE_SIZE };

        VkDescriptorImageInfo hwClusterIDInfo{};
        hwClusterIDInfo.imageView = hwClusterIDView;
        hwClusterIDInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // See the class/Init() doc comment: caller must guarantee this.

        VkDescriptorImageInfo hwTriangleIDInfo{};
        hwTriangleIDInfo.imageView = hwTriangleIDView;
        hwTriangleIDInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo hwDepthInfo{};
        hwDepthInfo.sampler = m_DepthSampler;
        hwDepthInfo.imageView = hwDepthView;
        hwDepthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo swAtomicInfo{};
        swAtomicInfo.imageView = swVisBufferAtomicView;
        swAtomicInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // renderer::ClusterSoftwareRasterPass keeps this permanently GENERAL.

        VkDescriptorImageInfo outputColorInfo{};
        outputColorInfo.imageView = m_OutputColorView;
        outputColorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputNormalInfo{};
        outputNormalInfo.imageView = m_OutputNormalView;
        outputNormalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputDepthInfo{};
        outputDepthInfo.imageView = m_OutputDepthView;
        outputDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputAlbedoInfo{};
        outputAlbedoInfo.imageView = m_OutputAlbedoView;
        outputAlbedoInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // WPOGlobalsUBO is owned by renderer::ClusterRenderPipeline (m_WPOGlobalsBuffer), uploaded
        // once per frame before any raster/resolve pass runs -- borrowed read-only here, exactly
        // like the two raster passes already borrow it.
        VkDescriptorBufferInfo wpoGlobalsInfo{ wpoGlobalsBuffer, 0, VK_WHOLE_SIZE };

        // The material table passed into Init(), already fully filled by the one-time command
        // buffer above (m_MaterialParamsBuffer.Handle() is valid the moment Create() returns -- the
        // fill itself is only ORDERED, not required to have already executed, by the time this
        // VkBuffer handle is written into a descriptor).
        VkDescriptorBufferInfo materialParamsInfo{ m_MaterialParamsBuffer.Handle(), 0, m_MaterialParamsBuffer.Size() };

        VkDescriptorImageInfo outputRoughnessMetallicInfo{};
        outputRoughnessMetallicInfo.imageView = m_OutputRoughnessMetallicView;
        outputRoughnessMetallicInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputMaterialIDInfo{};
        outputMaterialIDInfo.imageView = m_OutputMaterialIDView;
        outputMaterialIDInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Phase 1 (Nanite advanced): binding 26 -- see this shader's own binding comment above.
        VkDescriptorBufferInfo splineControlPointsInfo{ splineControlPointsBuffer, 0, VK_WHOLE_SIZE };
        // Skeletal-animation feature: binding 29 -- see this shader's own binding comment above.
        VkDescriptorBufferInfo boneMatricesInfo{ boneMatricesBuffer, 0, VK_WHOLE_SIZE };

        VkWriteDescriptorSet writes[20]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetadataInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hwClusterIDInfo, nullptr, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &hwTriangleIDInfo, nullptr, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hwDepthInfo, nullptr, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &swAtomicInfo, nullptr, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputColorInfo, nullptr, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 8, 0, maskTextureCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskImageInfos.data(), nullptr, nullptr };
        writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputNormalInfo, nullptr, nullptr };
        writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputDepthInfo, nullptr, nullptr };
        writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputAlbedoInfo, nullptr, nullptr };
        writes[12] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &wpoGlobalsInfo, nullptr };
        writes[13] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
        writes[14] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputRoughnessMetallicInfo, nullptr, nullptr };
        // Array index != dstBinding for these last two on purpose -- bindings 15-18 (Phase 3's
        // renderer::VirtualShadowMapPass resources) are intentionally left unwritten here,
        // written later by SetVirtualShadowMap() once the caller has a VirtualShadowMapPass to
        // bind (same convention as renderer::SurfaceCachePass's own SetVirtualShadowMap()) --
        // the per-entity rotation buffers are appended past them at 19/20 instead of renumbering
        // the already-verified shadow bindings.
        writes[15] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 19, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
        writes[16] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 20, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
        writes[17] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 25, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputMaterialIDInfo, nullptr, nullptr };
        writes[18] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 26, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &splineControlPointsInfo, nullptr };
        writes[19] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 29, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &boneMatricesInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 20, writes, 0, nullptr);

        // --- Pipeline layout + pipeline ---
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
#ifndef NDEBUG
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(uint32_t); // viewMode
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
#else
        pipelineLayoutInfo.pushConstantRangeCount = 0;
#endif
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/ClusterResolve.comp.spv");
        m_Pipeline = VulkanPipeline::CreateComputePipeline(m_Device, m_PipelineLayout, shaderModule);
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);
    }

    void ClusterResolvePass::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_DescriptorSet -- not freed individually.
                vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            }
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);

            // --- Phase 1b (InitBinnedResolve()'s own resources) ---
            if (m_ResolveBinnedPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_ResolveBinnedPipeline, nullptr);
            if (m_ResolveBinnedPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_ResolveBinnedPipelineLayout, nullptr);
            if (m_ResolveBinnedDescriptorPool != VK_NULL_HANDLE) {
                // Destroying the pool implicitly frees m_ResolveBinnedSet -- not freed individually.
                vkDestroyDescriptorPool(m_Device, m_ResolveBinnedDescriptorPool, nullptr);
            }
            if (m_ResolveBinnedSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_ResolveBinnedSetLayout, nullptr);
            if (m_DepthSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_DepthSampler, nullptr);
            if (m_OutputColorView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputColorView, nullptr);
            if (m_OutputNormalView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputNormalView, nullptr);
            if (m_OutputDepthView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputDepthView, nullptr);
            if (m_OutputAlbedoView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputAlbedoView, nullptr);
            if (m_OutputRoughnessMetallicView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputRoughnessMetallicView, nullptr);
            if (m_OutputMaterialIDView != VK_NULL_HANDLE) vkDestroyImageView(m_Device, m_OutputMaterialIDView, nullptr);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_DescriptorSet = VK_NULL_HANDLE;
        m_DepthSampler = VK_NULL_HANDLE;
        m_OutputColorView = VK_NULL_HANDLE;
        m_OutputNormalView = VK_NULL_HANDLE;
        m_OutputDepthView = VK_NULL_HANDLE;
        m_OutputAlbedoView = VK_NULL_HANDLE;
        m_OutputRoughnessMetallicView = VK_NULL_HANDLE;
        m_OutputMaterialIDView = VK_NULL_HANDLE;

        m_ResolveBinnedPipeline = VK_NULL_HANDLE;
        m_ResolveBinnedPipelineLayout = VK_NULL_HANDLE;
        m_ResolveBinnedDescriptorPool = VK_NULL_HANDLE;
        m_ResolveBinnedSetLayout = VK_NULL_HANDLE;
        m_ResolveBinnedSet = VK_NULL_HANDLE;
        m_ClusterMetadataBuffer = VK_NULL_HANDLE;
        m_CompressedPoolBuffer = VK_NULL_HANDLE;
        m_WPOGlobalsBuffer = VK_NULL_HANDLE;
        m_EntityTransformBuffer = VK_NULL_HANDLE;
        m_EntityDataBuffer = VK_NULL_HANDLE;
        m_MaskImageInfos.clear();

        m_ViewParamsBuffer.Destroy();
        m_VTVolumeUBO.Destroy();
        m_MaterialParamsBuffer.Destroy();

        // vmaDestroyImage tolerates VK_NULL_HANDLE for both handle arguments (no-op), matching
        // GpuBuffer::Destroy()'s own null-safe convention.
        if (m_Allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, m_OutputColorImage, m_OutputColorAllocation);
            vmaDestroyImage(m_Allocator, m_OutputNormalImage, m_OutputNormalAllocation);
            vmaDestroyImage(m_Allocator, m_OutputDepthImage, m_OutputDepthAllocation);
            vmaDestroyImage(m_Allocator, m_OutputAlbedoImage, m_OutputAlbedoAllocation);
            vmaDestroyImage(m_Allocator, m_OutputRoughnessMetallicImage, m_OutputRoughnessMetallicAllocation);
            vmaDestroyImage(m_Allocator, m_OutputMaterialIDImage, m_OutputMaterialIDAllocation);
        }
        m_OutputColorImage = VK_NULL_HANDLE;
        m_OutputColorAllocation = VK_NULL_HANDLE;
        m_OutputNormalImage = VK_NULL_HANDLE;
        m_OutputNormalAllocation = VK_NULL_HANDLE;
        m_OutputDepthImage = VK_NULL_HANDLE;
        m_OutputDepthAllocation = VK_NULL_HANDLE;
        m_OutputAlbedoImage = VK_NULL_HANDLE;
        m_OutputAlbedoAllocation = VK_NULL_HANDLE;
        m_OutputRoughnessMetallicImage = VK_NULL_HANDLE;
        m_OutputRoughnessMetallicAllocation = VK_NULL_HANDLE;
        m_OutputMaterialIDImage = VK_NULL_HANDLE;
        m_OutputMaterialIDAllocation = VK_NULL_HANDLE;

        m_RenderExtent = { 0, 0 };
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

    void ClusterResolvePass::RecordResolve(VkCommandBuffer cmd, const maths::mat4& viewProj, const maths::mat4& prevViewProj,
        const DirectionalLight& sun, const maths::vec3& cameraPositionWorld, float surfaceWetness, float snowCoverage,
        float glintDensityScale, float glintIntensityScale, float mixMaskSharpnessScale,
        float globalTimeSeconds, uint32_t debugViewMode) {
        ResolveViewParams viewParams{};
        viewParams.viewProj = viewProj;
        viewParams.prevViewProj = prevViewProj;
        viewParams.viewportWidth = static_cast<float>(m_RenderExtent.width);
        viewParams.viewportHeight = static_cast<float>(m_RenderExtent.height);
        viewParams.surfaceWetness = surfaceWetness;
        viewParams.snowCoverage = snowCoverage;
        // Glint / sparkle (UE5.8 rendering-parity gap G5): Debug-only tuning multipliers (1.0 in
        // Release) -- see the ResolveViewParams struct's own field comments above.
        viewParams.glintDensityScale = glintDensityScale;
        viewParams.glintIntensityScale = glintIntensityScale;
        // Substrate horizontal mixing (UE5.8 rendering-parity gap G6): Debug-only blend-sharpness
        // multiplier (1.0 in Release) -- see the ResolveViewParams struct's own field comment above.
        viewParams.mixMaskSharpnessScale = mixMaskSharpnessScale;
        viewParams.sunDirectionX = sun.direction.x;
        viewParams.sunDirectionY = sun.direction.y;
        viewParams.sunDirectionZ = sun.direction.z;
        viewParams.sunColorR = sun.color.x;
        viewParams.sunColorG = sun.color.y;
        viewParams.sunColorB = sun.color.z;
        viewParams.sunIntensity = sun.intensity;
        viewParams.cameraPositionWorldX = cameraPositionWorld.x;
        viewParams.cameraPositionWorldY = cameraPositionWorld.y;
        viewParams.cameraPositionWorldZ = cameraPositionWorld.z;
        viewParams.timeSeconds = globalTimeSeconds;
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ResolveViewParams), &viewParams);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &m_DescriptorSet, 0, nullptr);

#ifndef NDEBUG
        vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &debugViewMode);
#endif

        uint32_t groupCountX = (m_RenderExtent.width + kWorkgroupSize - 1) / kWorkgroupSize;
        uint32_t groupCountY = (m_RenderExtent.height + kWorkgroupSize - 1) / kWorkgroupSize;
        vkCmdDispatch(cmd, groupCountX, groupCountY, 1);

        // GetOutputColorImage() visible to a later sampled read (a future preview/blit pass) or a
        // transfer-based blit to the swapchain -- whichever a caller ends up using; both dst
        // stage/access pairs are included since this class does not itself know which.
        // SHADER_STORAGE_READ is additionally included (on top of SAMPLED_READ) because several
        // GI/lighting passes (e.g. renderer::ReflectionTrace, renderer::MegaLightsShade) read the
        // 3 GBuffer outputs (normal/depth/albedo) via plain imageLoad, not a sampler -- and
        // renderer::ReflectionPass::RecordGather read-modify-writes GetOutputColorImage() itself
        // the same way (see that method's own comment).
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    }

    void ClusterResolvePass::InitBinnedResolve(VkDevice device, VkCommandPool commandPool, VkQueue queue,
        const ClusterShadingBinPass& shadingBinPass) {
        (void)commandPool;
        (void)queue; // Reserved for parity with every other Init()-shaped method; no one-time submit needed here.
        uint32_t maskTextureCount = static_cast<uint32_t>(m_MaskImageInfos.size());

        // --- Descriptor set layout: 26 bindings, matching ClusterResolveBinned.comp's set = 0
        // bindings 0..25 exactly. Bindings 2-4 (VisBuffer/HW-depth/SW-atomic) from the original
        // 15-binding layout are absent here -- visibility is already fully resolved by
        // `shadingBinPass` before this pipeline ever runs -- replaced by bindings 2-4 for the
        // sorted-pixel-list/bin-offsets/bin-histogram buffers that carry that resolved visibility
        // forward instead. Bindings 14-17 are Phase 3's renderer::VirtualShadowMapPass resources --
        // see SetVirtualShadowMap()'s own comment. Bindings 18-19 are the per-entity rotation
        // buffers (renderer::VulkanContext's dynamic primitive spin) -- this binned/Release path
        // needs them for the exact same reason ClusterResolve.comp's own full-screen path does
        // (reapplying entity self-rotation before re-deriving barycentrics), appended past the
        // shadow bindings rather than renumbering them. Bindings 20-23 are Step 4's renderer::
        // VirtualTextureManager resources -- see SetVirtualTexture()'s own comment. Binding 24 is
        // Substrate's g_OutputMaterialID, 25 is Phase 1 Nanite advanced's SplineControlPointsSSBO. ---
        std::array<VkDescriptorSetLayoutBinding, 29> bindings{};
        bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // ClusterCullMetadataSSBO
        bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // CompressedClusterPoolSSBO
        bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_SortedPixelList
        bindings[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_BinOffsets
        bindings[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_BinHistogram
        bindings[5] = { 5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };   // g_OutputColor
        bindings[6] = { 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // ResolveViewParamsUBO
        bindings[7] = { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_MaskTextures[]
        bindings[8] = { 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // WPOGlobalsUBO
        bindings[9] = { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // MaterialParamsSSBO
        bindings[10] = { 10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_OutputNormal
        bindings[11] = { 11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_OutputDepth
        bindings[12] = { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_OutputAlbedo
        bindings[13] = { 13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_OutputRoughnessMetallic
        bindings[14] = { 14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_ShadowPhysicalAtlas
        bindings[15] = { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_ShadowPageTable
        bindings[16] = { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_ShadowFeedback
        bindings[17] = { 17, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_ShadowSunLevels
        bindings[18] = { 18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityTransformBuffer
        bindings[19] = { 19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // EntityDataBuffer
        bindings[20] = { 20, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_PageTable
        bindings[21] = { 21, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxPhysicalPools, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_PhysicalPools[]
        bindings[22] = { 22, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_VTFeedback
        bindings[23] = { 23, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // VirtualTextureVolumeUBO
        bindings[24] = { 24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };  // g_OutputMaterialID (r16ui, Substrate integration)
        // Phase 1 (Nanite advanced): binding 25 -- the first free slot past this shader's full
        // 0-24 range (0-23 original + 24 Substrate's g_OutputMaterialID). See ClusterResolveBinned.
        // comp's own identical binding comment.
        bindings[25] = { 25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // SplineControlPointsSSBO
        // Atmos weather system, Subtask 5 -- see SetAtmosCloudLighting()'s own comment.
        bindings[26] = { 26, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_SkyViewLUT
        bindings[27] = { 27, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }; // g_CloudShadowMap
        // Skeletal-animation feature: binding 28 -- the first free slot past this shader's full
        // 0-27 range. See ClusterResolveBinned.comp's own identical binding comment.
        bindings[28] = { 28, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };        // SkeletalBoneMatricesSSBO

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_ResolveBinnedSetLayout));

        std::array<VkDescriptorPoolSize, 4> poolSizes{};
        poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13 }; // cluster metadata, compressed pool, sorted list, offsets, histogram, material params, shadow page table, shadow feedback, entity transform, entity data, VT feedback, spline control points, bone matrices
        poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 6 };   // color, normal, depth, albedo, roughness-metallic, materialID
        poolSizes[2] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 };  // view params, WPO globals, shadow sun levels, VT volume
        poolSizes[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maskTextureCount + 4 + kMaxPhysicalPools }; // + shadow physical atlas, g_PageTable, g_PhysicalPools[], g_SkyViewLUT, g_CloudShadowMap

        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_ResolveBinnedDescriptorPool));

        VkDescriptorSetAllocateInfo setAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAlloc.descriptorPool = m_ResolveBinnedDescriptorPool;
        setAlloc.descriptorSetCount = 1;
        setAlloc.pSetLayouts = &m_ResolveBinnedSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(device, &setAlloc, &m_ResolveBinnedSet));

        VkDescriptorBufferInfo clusterMetaInfo{ m_ClusterMetadataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo compressedPoolInfo{ m_CompressedPoolBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo sortedListInfo{ shadingBinPass.GetSortedPixelListBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo binOffsetsInfo{ shadingBinPass.GetBinOffsetsBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo binHistogramInfo{ shadingBinPass.GetBinHistogramBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo outputColorInfo{ VK_NULL_HANDLE, m_OutputColorView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo viewParamsInfo{ m_ViewParamsBuffer.Handle(), 0, m_ViewParamsBuffer.Size() };
        VkDescriptorBufferInfo wpoGlobalsInfo{ m_WPOGlobalsBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo materialParamsInfo{ m_MaterialParamsBuffer.Handle(), 0, m_MaterialParamsBuffer.Size() };
        VkDescriptorImageInfo outputNormalInfo{ VK_NULL_HANDLE, m_OutputNormalView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outputDepthInfo{ VK_NULL_HANDLE, m_OutputDepthView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outputAlbedoInfo{ VK_NULL_HANDLE, m_OutputAlbedoView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outputRMInfo{ VK_NULL_HANDLE, m_OutputRoughnessMetallicView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outputMaterialIDInfo{ VK_NULL_HANDLE, m_OutputMaterialIDView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo entityTransformInfo{ m_EntityTransformBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo entityDataInfo{ m_EntityDataBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo splineControlPointsInfo{ m_SplineControlPointsBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo boneMatricesInfo{ m_BoneMatricesBuffer, 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 19> writes{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &clusterMetaInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &compressedPoolInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sortedListInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &binOffsetsInfo, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &binHistogramInfo, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputColorInfo, nullptr, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &viewParamsInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 7, 0, maskTextureCount, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_MaskImageInfos.data(), nullptr, nullptr };
        writes[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 8, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &wpoGlobalsInfo, nullptr };
        writes[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &materialParamsInfo, nullptr };
        writes[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputNormalInfo, nullptr, nullptr };
        writes[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputDepthInfo, nullptr, nullptr };
        writes[12] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputAlbedoInfo, nullptr, nullptr };
        writes[13] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputRMInfo, nullptr, nullptr };
        // Array index != dstBinding for these last two on purpose -- bindings 14-17 (Phase 3's
        // renderer::VirtualShadowMapPass resources) are intentionally left unwritten here (written
        // later by SetVirtualShadowMap(), same as the primary set above); the per-entity rotation
        // buffers are appended past them at 18/19.
        writes[14] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 18, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityTransformInfo, nullptr };
        writes[15] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 19, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &entityDataInfo, nullptr };
        writes[16] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 24, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputMaterialIDInfo, nullptr, nullptr };
        writes[17] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 25, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &splineControlPointsInfo, nullptr };
        writes[18] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 28, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &boneMatricesInfo, nullptr };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) }; // binIndex
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_ResolveBinnedSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_ResolveBinnedPipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(device, "shaders/ClusterResolveBinned.comp.spv");
        m_ResolveBinnedPipeline = VulkanPipeline::CreateComputePipeline(device, m_ResolveBinnedPipelineLayout, shaderModule);
        vkDestroyShaderModule(device, shaderModule, nullptr);

        LOG_INFO("[ClusterResolvePass] Initialized binned resolve path (Phase 1b shading bins).");
    }

    void ClusterResolvePass::RecordResolveBinned(VkCommandBuffer cmd, const maths::mat4& viewProj,
        const DirectionalLight& sun, const maths::vec3& cameraPositionWorld, float surfaceWetness, float snowCoverage,
        float glintDensityScale, float glintIntensityScale, float mixMaskSharpnessScale,
        float globalTimeSeconds, const ClusterShadingBinPass& shadingBinPass) {
        // prevViewProj is never read by ClusterResolveBinned.comp (this path never serves
        // DEBUG_VIEW_MOTION_VECTORS) -- `viewProj` itself is reused as a harmless placeholder value
        // rather than introducing a separate identity-matrix concept for an otherwise-dead field.
        ResolveViewParams viewParams{};
        viewParams.viewProj = viewProj;
        viewParams.prevViewProj = viewProj;
        viewParams.viewportWidth = static_cast<float>(m_RenderExtent.width);
        viewParams.viewportHeight = static_cast<float>(m_RenderExtent.height);
        viewParams.surfaceWetness = surfaceWetness;
        viewParams.snowCoverage = snowCoverage;
        // Glint / sparkle (UE5.8 rendering-parity gap G5): Debug-only tuning multipliers (1.0 in
        // Release, so the material's authored sparkle renders unchanged on this Release path).
        viewParams.glintDensityScale = glintDensityScale;
        viewParams.glintIntensityScale = glintIntensityScale;
        // Substrate horizontal mixing (UE5.8 rendering-parity gap G6): Debug-only blend-sharpness
        // multiplier (1.0 in Release, so a horizontally-mixed material blends at its authored
        // mixContrast on this Release path) -- see the ResolveViewParams struct's own field comment.
        viewParams.mixMaskSharpnessScale = mixMaskSharpnessScale;
        viewParams.sunDirectionX = sun.direction.x;
        viewParams.sunDirectionY = sun.direction.y;
        viewParams.sunDirectionZ = sun.direction.z;
        viewParams.sunColorR = sun.color.x;
        viewParams.sunColorG = sun.color.y;
        viewParams.sunColorB = sun.color.z;
        viewParams.sunIntensity = sun.intensity;
        viewParams.cameraPositionWorldX = cameraPositionWorld.x;
        viewParams.cameraPositionWorldY = cameraPositionWorld.y;
        viewParams.cameraPositionWorldZ = cameraPositionWorld.z;
        viewParams.timeSeconds = globalTimeSeconds;
        vkCmdUpdateBuffer(cmd, m_ViewParamsBuffer.Handle(), 0, sizeof(ResolveViewParams), &viewParams);

        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ResolveBinnedPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ResolveBinnedPipelineLayout, 0, 1, &m_ResolveBinnedSet, 0, nullptr);

        // One indirect dispatch per material bin -- every thread within a given dispatch shades
        // the SAME material (binIndex, a push constant), the genuine warp coherence this whole
        // pipeline exists to provide. No barrier is needed between these kMaxMaterials dispatches:
        // each writes only the disjoint set of pixels renderer::ClusterShadingBinPass routed into
        // its own bin, the same "disjoint-region writes need no inter-dispatch barrier" reasoning
        // this codebase's early/late hardware raster passes already rely on.
        for (uint32_t binIndex = 0; binIndex < kMaxMaterials; ++binIndex) {
            vkCmdPushConstants(cmd, m_ResolveBinnedPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &binIndex);
            vkCmdDispatchIndirect(cmd, shadingBinPass.GetBinDispatchArgsBuffer(),
                static_cast<VkDeviceSize>(binIndex) * sizeof(VkDispatchIndirectCommand));
        }

        // Identical trailing barrier to RecordResolve()'s own -- see that method's comment.
        VulkanUtils::RecordMemoryBarrier(cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    }

    void ClusterResolvePass::SetVirtualShadowMap(const VirtualShadowMapPass& vsm) {
        VkDescriptorImageInfo atlasImageInfo{};
        atlasImageInfo.sampler = vsm.GetPhysicalAtlasSampler();
        atlasImageInfo.imageView = vsm.GetPhysicalAtlasView();
        atlasImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo pageTableInfo{ vsm.GetPageTableBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo feedbackInfo{ vsm.GetFeedbackDeviceBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo sunLevelsInfo{ vsm.GetSunLevelsBuffer(), 0, VK_WHOLE_SIZE };

        // Both descriptor sets (the always-live binned path's m_ResolveBinnedSet, and the
        // Debug-only full-screen m_DescriptorSet used by every DEBUG_VIEW_* mode -- see
        // ClusterRenderPipeline::RecordFrame's own branch between the two) need the identical 4
        // shadow bindings written, just at different binding indices (15-18 vs 14-17 -- see each
        // set's own layout comment for why the numbering differs).
        VkWriteDescriptorSet writes[8]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 15, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &atlasImageInfo, nullptr, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 16, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 17, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 18, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sunLevelsInfo, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 14, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &atlasImageInfo, nullptr, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pageTableInfo, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 16, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 17, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sunLevelsInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, 8, writes, 0, nullptr);
    }

    void ClusterResolvePass::SetAtmosCloudLighting(VkSampler skyViewLUTSampler, VkImageView skyViewLUTView,
        VkSampler cloudShadowSampler, VkImageView cloudShadowView) {
        VkDescriptorImageInfo skyViewInfo{ skyViewLUTSampler, skyViewLUTView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo cloudShadowInfo{ cloudShadowSampler, cloudShadowView, VK_IMAGE_LAYOUT_GENERAL };

        // Same "both descriptor sets, different binding indices" convention as SetVirtualShadowMap
        // above (27/28 in m_DescriptorSet, 26/27 in m_ResolveBinnedSet -- see each set's own layout
        // comment for why the numbering differs).
        VkWriteDescriptorSet writes[4]{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 27, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &skyViewInfo, nullptr, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 28, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudShadowInfo, nullptr, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 26, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &skyViewInfo, nullptr, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 27, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cloudShadowInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_Device, 4, writes, 0, nullptr);
    }

    void ClusterResolvePass::SetVirtualTexture(const VirtualTextureManager& vt, const maths::vec2& worldMinXZ,
        const maths::vec2& worldMaxXZ, VkBuffer feedbackBuffer) {
        VTVolumeParams params{};
        params.worldMinX = worldMinXZ.x;
        params.worldMinZ = worldMinXZ.y;
        params.worldMaxX = worldMaxXZ.x;
        params.worldMaxZ = worldMaxXZ.y;
        params.virtualTextureSize = static_cast<float>(vt.GetVirtualWidth());
        params.tileSize = static_cast<float>(vt.GetTileSize());
        params.borderSize = static_cast<float>(vt.GetBorderSize());
        std::memcpy(m_VTVolumeUBO.MappedData(), &params, sizeof(params));

        VkDescriptorImageInfo pageTableInfo{};
        pageTableInfo.sampler = vt.GetPageTableSampler();
        pageTableInfo.imageView = vt.GetPageTableImageView();
        pageTableInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Only pool 0 (this demo's Albedo-only RVT layer) is real -- every one of the
        // kMaxPhysicalPools array slots is filled with THAT SAME view/sampler rather than left
        // unwritten, since descriptorBindingPartiallyBound is not enabled (see this method's own
        // header comment in ClusterResolvePass.h).
        std::array<VkDescriptorImageInfo, kMaxPhysicalPools> physicalPoolInfos{};
        for (uint32_t i = 0; i < kMaxPhysicalPools; ++i) {
            physicalPoolInfos[i].sampler = vt.GetPhysicalPoolSampler();
            physicalPoolInfos[i].imageView = vt.GetPhysicalPoolImageView(0);
            physicalPoolInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkDescriptorBufferInfo feedbackInfo{ feedbackBuffer, 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo volumeInfo{ m_VTVolumeUBO.Handle(), 0, m_VTVolumeUBO.Size() };

        // Both descriptor sets need the identical 4 VT bindings written, just at different binding
        // indices (21-24 vs 20-23 -- see each set's own layout comment for why the numbering
        // differs), same two-set-write pattern as SetVirtualShadowMap() above.
        std::array<VkWriteDescriptorSet, 8> writes{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 21, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &pageTableInfo, nullptr, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 22, 0, kMaxPhysicalPools, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, physicalPoolInfos.data(), nullptr, nullptr };
        writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 23, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
        writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_DescriptorSet, 24, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &volumeInfo, nullptr };
        writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 20, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &pageTableInfo, nullptr, nullptr };
        writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 21, 0, kMaxPhysicalPools, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, physicalPoolInfos.data(), nullptr, nullptr };
        writes[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 22, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &feedbackInfo, nullptr };
        writes[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_ResolveBinnedSet, 23, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &volumeInfo, nullptr };
        vkUpdateDescriptorSets(m_Device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

}
