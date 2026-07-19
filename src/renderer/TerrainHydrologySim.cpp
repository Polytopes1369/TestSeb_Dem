#include "renderer/TerrainHydrologySim.h"

#include <format>
#include <stdexcept>

#include "core/Logger.h"
#include "renderer/vulkan/VulkanPipeline.h"
#include "renderer/vulkan/VulkanUtils.h"

namespace renderer {

    namespace {

        // Mirrors TerrainHydrology.comp's HydrologyPC exactly.
        struct HydrologyPushConstants {
            int32_t mode = 0;
            int32_t blurSource = 0;
            int32_t blurHorizontal = 0;
            int32_t _pad0 = 0;
        };

        // Mirrors TerrainHydrology.comp's mode constants exactly.
        constexpr int32_t kModeInit         = 0;
        constexpr int32_t kModeFlux         = 1;
        constexpr int32_t kModeWater        = 2;
        constexpr int32_t kModeErodeDeposit = 3;
        constexpr int32_t kModeAdvect       = 4;
        constexpr int32_t kModeBlurSeed     = 5;
        constexpr int32_t kModeBlurPass     = 6;
        constexpr int32_t kModeFinalizeMesh = 7;
        constexpr int32_t kModeFinalize     = 8;

        constexpr uint32_t kBindingCount = 12;

    } // namespace

    TerrainHydrologySim::GridImage TerrainHydrologySim::CreateGrid(VkFormat format) {
        GridImage grid;

        VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = { kResolution, kResolution, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        // STORAGE: every grid is written via imageStore during the bake. SAMPLED: the output
        // grids are sampled by the geometry generators / resolve / water shading afterward
        // (harmlessly granted to the intermediate grids too -- one shared creation path).
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(m_Allocator, &imageInfo, &allocInfo, &grid.image, &grid.allocation, nullptr));

        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image = grid.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VK_CHECK(vkCreateImageView(m_Device, &viewInfo, nullptr, &grid.view));

        return grid;
    }

    void TerrainHydrologySim::DestroyGrid(GridImage& grid) {
        if (grid.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_Device, grid.view, nullptr);
        }
        if (grid.image != VK_NULL_HANDLE) {
            vmaDestroyImage(m_Allocator, grid.image, grid.allocation);
        }
        grid = GridImage{};
    }

    void TerrainHydrologySim::Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue) {
        Shutdown();
        m_Device = device;
        m_Allocator = allocator;

        // =====================================================================================
        // Grid images (see TerrainHydrology.comp's binding table for each grid's role).
        // =====================================================================================
        m_Height       = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_BaseHeight   = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_Water        = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_Flux         = CreateGrid(VK_FORMAT_R32G32B32A32_SFLOAT);
        m_Velocity     = CreateGrid(VK_FORMAT_R32G32_SFLOAT);
        m_SedimentA    = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_SedimentB    = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_TempA        = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_TempB        = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_MeshHeight   = CreateGrid(VK_FORMAT_R32_SFLOAT);
        m_Attributes   = CreateGrid(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_WaterSurface = CreateGrid(VK_FORMAT_R32_SFLOAT);

        m_MeshHeightView = m_MeshHeight.view;
        m_WaterSurfaceView = m_WaterSurface.view;
        m_AttributesView = m_Attributes.view;

        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        // CLAMP_TO_EDGE is load-bearing: geom_water_surface.comp's oversized sea mesh reads the
        // border ring's open-sea texels for everything past the simulated footprint.
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        VK_CHECK(vkCreateSampler(m_Device, &samplerInfo, nullptr, &m_LinearSampler));

        // =====================================================================================
        // Descriptor layout / pool / the two ping-pong sets (see m_SetEven/m_SetOdd's comment).
        // =====================================================================================
        VkDescriptorSetLayoutBinding bindings[kBindingCount]{};
        for (uint32_t i = 0; i < kBindingCount; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        layoutInfo.bindingCount = kBindingCount;
        layoutInfo.pBindings = bindings;
        VK_CHECK(vkCreateDescriptorSetLayout(m_Device, &layoutInfo, nullptr, &m_SetLayout));

        VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kBindingCount * 2 };
        VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(m_Device, &poolInfo, nullptr, &m_DescriptorPool));

        VkDescriptorSetLayout setLayouts[2] = { m_SetLayout, m_SetLayout };
        VkDescriptorSetAllocateInfo setAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        setAllocInfo.descriptorPool = m_DescriptorPool;
        setAllocInfo.descriptorSetCount = 2;
        setAllocInfo.pSetLayouts = setLayouts;
        VkDescriptorSet sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VK_CHECK(vkAllocateDescriptorSets(m_Device, &setAllocInfo, sets));
        m_SetEven = sets[0];
        m_SetOdd = sets[1];

        auto writeSet = [&](VkDescriptorSet set, VkImageView sedimentPing, VkImageView sedimentPong) {
            VkImageView views[kBindingCount] = {
                m_Height.view, m_BaseHeight.view, m_Water.view, m_Flux.view, m_Velocity.view,
                sedimentPing, sedimentPong,
                m_TempA.view, m_TempB.view,
                m_MeshHeight.view, m_Attributes.view, m_WaterSurface.view,
            };
            VkDescriptorImageInfo imageInfos[kBindingCount]{};
            VkWriteDescriptorSet writes[kBindingCount]{};
            for (uint32_t i = 0; i < kBindingCount; ++i) {
                imageInfos[i] = { VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL };
                writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1,
                              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfos[i], nullptr, nullptr };
            }
            vkUpdateDescriptorSets(m_Device, kBindingCount, writes, 0, nullptr);
        };
        writeSet(m_SetEven, m_SedimentA.view, m_SedimentB.view);
        writeSet(m_SetOdd, m_SedimentB.view, m_SedimentA.view);

        // =====================================================================================
        // Pipeline.
        // =====================================================================================
        VkPushConstantRange pushRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HydrologyPushConstants) };
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &m_SetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        VK_CHECK(vkCreatePipelineLayout(m_Device, &pipelineLayoutInfo, nullptr, &m_PipelineLayout));

        VkShaderModule shaderModule = VulkanPipeline::LoadShaderModule(m_Device, "shaders/TerrainHydrology.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.layout = m_PipelineLayout;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        VK_CHECK(vkCreateComputePipelines(m_Device, VulkanPipeline::GetPipelineCache(), 1, &pipelineInfo, nullptr, &m_Pipeline));
        vkDestroyShaderModule(m_Device, shaderModule, nullptr);

        // =====================================================================================
        // Record + synchronously submit the whole bake (see this class' own header comment for
        // why blocking here is correct: geometry generation consumes the results immediately).
        // =====================================================================================
        VulkanUtils::ExecuteOneShotCommands(m_Device, commandPool, queue, [&](VkCommandBuffer cmd) {
            RecordBake(cmd);
        });

        LOG_INFO(std::format("[TerrainHydrologySim] Baked {}x{} hydrology grids ({} erosion iterations).",
            kResolution, kResolution, kIterations));
    }

    void TerrainHydrologySim::RecordBake(VkCommandBuffer cmd) {
        // --- One-time UNDEFINED -> GENERAL transition for every grid (kept GENERAL for their
        // entire lifetime afterward, matching this codebase's storage-image convention). ---
        const GridImage* grids[kBindingCount] = {
            &m_Height, &m_BaseHeight, &m_Water, &m_Flux, &m_Velocity,
            &m_SedimentA, &m_SedimentB, &m_TempA, &m_TempB,
            &m_MeshHeight, &m_Attributes, &m_WaterSurface,
        };
        VkImageMemoryBarrier2 imageBarriers[kBindingCount]{};
        for (uint32_t i = 0; i < kBindingCount; ++i) {
            imageBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            imageBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            imageBarriers[i].srcAccessMask = 0;
            imageBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            imageBarriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            imageBarriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            imageBarriers[i].image = grids[i]->image;
            imageBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        }
        VkDependencyInfo initDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        initDep.imageMemoryBarrierCount = kBindingCount;
        initDep.pImageMemoryBarriers = imageBarriers;
        vkCmdPipelineBarrier2(cmd, &initDep);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_Pipeline);

        const uint32_t groupCount = (kResolution + kWorkgroupSize - 1) / kWorkgroupSize;

        // Full compute->compute barrier between every dispatch: each mode reads its predecessor's
        // imageStore results (often at neighbor texels, so no dispatch-local ordering suffices).
        auto barrier = [&]() {
            VkMemoryBarrier2 b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        };

        auto dispatch = [&](VkDescriptorSet set, int32_t mode, int32_t blurSource, int32_t blurHorizontal) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_PipelineLayout, 0, 1, &set, 0, nullptr);
            HydrologyPushConstants pc{ mode, blurSource, blurHorizontal, 0 };
            vkCmdPushConstants(cmd, m_PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, groupCount, groupCount, 1);
            barrier();
        };

        // --- Init, then the erosion loop (see TerrainHydrology.comp's header for the scheme).
        // The descriptor set alternates per iteration so kModeAdvect's ping-pong output becomes
        // the next iteration's input without any image copy. ---
        dispatch(m_SetEven, kModeInit, 0, 0);
        for (int32_t iter = 0; iter < kIterations; ++iter) {
            VkDescriptorSet set = (iter % 2 == 0) ? m_SetEven : m_SetOdd;
            dispatch(set, kModeFlux, 0, 0);
            dispatch(set, kModeWater, 0, 0);
            dispatch(set, kModeErodeDeposit, 0, 0);
            dispatch(set, kModeAdvect, 0, 0);
        }

        // --- Mesh-height blur chain: 3 full separable Gaussian passes low-pass the eroded height
        // to >= ~8 world-unit wavelengths (mesh-safety, see TerrainHydrology.comp's header). ---
        dispatch(m_SetEven, kModeBlurSeed, /*blurSource=*/0, 0);
        for (int i = 0; i < 3; ++i) {
            dispatch(m_SetEven, kModeBlurPass, 0, /*horizontal=*/1);
            dispatch(m_SetEven, kModeBlurPass, 0, /*horizontal=*/0);
        }
        dispatch(m_SetEven, kModeFinalizeMesh, 0, 0);

        // --- Moisture blur chain (wider: 4 passes -- wetness spreads a few meters past the
        // waterline), then the finalize pack reads its result straight out of g_TempA. ---
        dispatch(m_SetEven, kModeBlurSeed, /*blurSource=*/1, 0);
        for (int i = 0; i < 4; ++i) {
            dispatch(m_SetEven, kModeBlurPass, 0, /*horizontal=*/1);
            dispatch(m_SetEven, kModeBlurPass, 0, /*horizontal=*/0);
        }
        dispatch(m_SetEven, kModeFinalize, 0, 0);

        // --- Final visibility: the bake's storage writes must be visible to every later consumer
        // (geometry-generation compute samples, per-frame resolve/water fragment samples). The
        // one-shot submit's vkQueueWaitIdle already fully serializes this, but the explicit
        // barrier documents (and locally guarantees) the dependency regardless of how the submit
        // path evolves. ---
        VkMemoryBarrier2 finalBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        finalBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        finalBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo finalDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        finalDep.memoryBarrierCount = 1;
        finalDep.pMemoryBarriers = &finalBarrier;
        vkCmdPipelineBarrier2(cmd, &finalDep);
    }

    void TerrainHydrologySim::Shutdown() {
        if (m_Device != VK_NULL_HANDLE) {
            if (m_Pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_Device, m_Pipeline, nullptr);
            if (m_PipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_Device, m_PipelineLayout, nullptr);
            if (m_DescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_Device, m_DescriptorPool, nullptr);
            if (m_SetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_Device, m_SetLayout, nullptr);
            if (m_LinearSampler != VK_NULL_HANDLE) vkDestroySampler(m_Device, m_LinearSampler, nullptr);

            DestroyGrid(m_Height);
            DestroyGrid(m_BaseHeight);
            DestroyGrid(m_Water);
            DestroyGrid(m_Flux);
            DestroyGrid(m_Velocity);
            DestroyGrid(m_SedimentA);
            DestroyGrid(m_SedimentB);
            DestroyGrid(m_TempA);
            DestroyGrid(m_TempB);
            DestroyGrid(m_MeshHeight);
            DestroyGrid(m_Attributes);
            DestroyGrid(m_WaterSurface);
        }

        m_Pipeline = VK_NULL_HANDLE;
        m_PipelineLayout = VK_NULL_HANDLE;
        m_DescriptorPool = VK_NULL_HANDLE;
        m_SetLayout = VK_NULL_HANDLE;
        m_SetEven = VK_NULL_HANDLE;
        m_SetOdd = VK_NULL_HANDLE;
        m_LinearSampler = VK_NULL_HANDLE;
        m_MeshHeightView = VK_NULL_HANDLE;
        m_WaterSurfaceView = VK_NULL_HANDLE;
        m_AttributesView = VK_NULL_HANDLE;
        m_Allocator = VK_NULL_HANDLE;
        m_Device = VK_NULL_HANDLE;
    }

}
