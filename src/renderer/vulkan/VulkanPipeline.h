#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

class VulkanPipeline {
public:
    static VkPipelineLayout CreatePipelineLayout(VkDevice device, VkDescriptorSetLayout bindlessLayout);

    // Simplification: no need for RenderPass, we use PipelineRenderingCreateInfo
    // Visibility Buffer: 2 color attachments (ClusterID + local TriangleID, both single-channel
    // uint formats) instead of one classic lit-color attachment -- see VulkanContext.h's
    // kVisBufferFormat comment for why 2 separate R32_UINT images are used over one R64_UINT.
    static VkPipeline CreateGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule vertShader, VkShaderModule fragShader, const std::array<VkFormat, 2>& colorFormats, VkFormat depthFormat);

    // Single source of truth for SPIR-V loading -- replaces the 17 anonymous-namespace copies
    // scattered across renderer .cpp files (each carrying an identical ReadShaderFile +
    // CreateShaderModule pair). All passes must call these instead of defining their own local
    // versions. Throws std::runtime_error on I/O or Vulkan failure.
    static std::vector<char>  ReadShaderFile(const std::string& path);
    static VkShaderModule     CreateShaderModule(VkDevice device, const std::vector<char>& code);
    // Convenience: ReadShaderFile + CreateShaderModule in one call.
    static VkShaderModule     LoadShaderModule(VkDevice device, const std::string& path);

    static VkPipeline CreateComputePipeline(VkDevice device, VkPipelineLayout layout, VkShaderModule computeShader);

    // Descriptor-set-layout (2 storage-buffer bindings: SourceCountSSBO, DispatchArgsSSBO) for
    // BuildDispatchIndirectArgs.comp -- identical across every pass that owns its own instance of
    // this small "count word -> VkDispatchIndirectCommand" utility shader (ClusterLODSelectionPass,
    // ClusterOcclusionCullingPass, ClusterSoftwareRasterPass). Split out from the pipeline creation
    // below because some callers need the layout early (to size a shared descriptor pool) well
    // before the pipeline itself is created.
    static VkDescriptorSetLayout CreateBuildDispatchIndirectArgsSetLayout(VkDevice device);

    // Pipeline-layout + compute-pipeline for BuildDispatchIndirectArgs.comp, built against a
    // caller-owned `setLayout` (see CreateBuildDispatchIndirectArgsSetLayout above). Each caller
    // still allocates/writes its own VkDescriptorSet from that layout (pointing at that pass's own
    // count/args buffers), so passes remain independent at the descriptor-set/dispatch level; only
    // the layout+pipeline creation boilerplate (previously hand-copied 3x) is shared.
    static void CreateBuildDispatchIndirectArgsPipeline(
        VkDevice device,
        VkDescriptorSetLayout setLayout,
        VkPipelineLayout& outPipelineLayout,
        VkPipeline& outPipeline
    );

    // E1 (loading-time optimization): persisted VkPipelineCache, set once by VulkanContext::
    // CreatePipelineCache() (right after vkCreatePipelineCache succeeds) and reset to VK_NULL_HANDLE
    // by VulkanContext::Shutdown() (right before the cache itself is destroyed) -- see
    // VulkanContext.cpp's own CreatePipelineCache()/Shutdown() for the create/destroy lifecycle.
    // Every renderer:: pass creates its own pipeline(s) via a direct vkCreateComputePipelines/
    // vkCreateGraphicsPipelines/vkCreateRayTracingPipelinesKHR call (not through this class), so
    // routing ALL of them through one shared cache without threading a VkPipelineCache parameter
    // through every pass' constructor/Init() signature (a large, purely mechanical signature change
    // across ~30 files) is done via this static accessor instead. Acceptable as an exception to this
    // codebase's usual "no globals" preference for the same reason the existing renderer::
    // g_RTFunctions loader-style global is: there is exactly ONE VulkanContext/VkDevice/
    // VkPipelineCache alive at a time for this process's entire lifetime (never multiple concurrent
    // instances, never re-created mid-run), so a static's usual "hidden shared mutable state" risk
    // does not apply here -- every call site just wants "the one cache for the one device". Every
    // vkCreate*Pipelines* function accepts VK_NULL_HANDLE (the default, and the value during the
    // brief window before VulkanContext::CreatePipelineCache() has run, or after Shutdown() has
    // cleared it) as "no cache", so every call site stays correct in that window too -- in practice
    // VulkanContext::Init() creates the cache before any renderer:: pass exists.
    static void SetPipelineCache(VkPipelineCache cache) { s_PipelineCache = cache; }
    static VkPipelineCache GetPipelineCache() { return s_PipelineCache; }

private:
    static inline VkPipelineCache s_PipelineCache = VK_NULL_HANDLE;
};