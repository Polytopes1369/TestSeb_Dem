#pragma once
// F12 (UE5.8 rendering-parity gap: Texture-based Light Functions + projected Caustics) --
// procedurally generates the small texture set src/shaders/include/light_functions.glsl's bindless
// g_LightFunctionTextures[] and src/shaders/include/caustics_projection.glsl's g_CausticsTexture both
// need -- zero image assets, matching the project's "100% procedural, no data in the exe" rule
// (CLAUDE.md). See src/shaders/src/Streaming/GenerateLightFunctionTextures.comp for the compute
// shader this class drives.
//
// Mirrors renderer::ProceduralMaskGenerator's own class shape exactly (Init/Shutdown, one-shot
// blocking generation dispatch, ready-to-bind VkDescriptorImageInfo output) -- see that class' own
// header comment for the full Vulkan-lifecycle rationale, identical here. The one structural
// difference: this class generates kTotalSlots (5) images in ONE dispatch/descriptor set but exposes
// TWO logically distinct consumer-facing views into that same underlying array -- slots 0..
// kLightFunctionSlotCount-1 (4) are the real Light Function gobo patterns
// (GetLightFunctionImageInfos()), slot kCausticsSlot (the 5th) is the caustics pattern
// (GetCausticsImageInfo()) -- rather than running two separate generator instances/shaders for what
// is otherwise identical "small one-shot procedural pattern texture" Vulkan boilerplate.
//
// Run early in renderer::ClusterRenderPipeline::Init(), before renderer::ClusterResolvePass::Init()
// (its own consumer) -- same "producer generator Init's before any pass binds its output" convention
// as m_MaskGenerator's own call site.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace renderer {

    class ProceduralLightFunctionGenerator {
    public:
        ProceduralLightFunctionGenerator() = default;

        ProceduralLightFunctionGenerator(const ProceduralLightFunctionGenerator&) = delete;
        ProceduralLightFunctionGenerator& operator=(const ProceduralLightFunctionGenerator&) = delete;

        // Matches K_LIGHT_FUNCTION_SLOT_COUNT / K_CAUSTICS_SLOT / K_TOTAL_SLOTS in
        // GenerateLightFunctionTextures.comp.
        static constexpr uint32_t kLightFunctionSlotCount = 4;
        static constexpr uint32_t kCausticsSlot = 4;
        static constexpr uint32_t kTotalSlots = 5;
        // Matches K_LIGHT_FUNCTION_TEXTURE_SIZE in GenerateLightFunctionTextures.comp.
        static constexpr uint32_t kTextureSize = 256;
        static constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

        // Allocates every slot's image/view + one shared REPEAT sampler (both the gobo patterns and
        // the caustics pattern must tile seamlessly -- a spot light's gobo can be scaled/projected
        // arbitrarily large, and the caustics sampling UV scrolls continuously over time, see
        // caustics_projection.glsl's own ComputeCausticsModulation()), dispatches
        // GenerateLightFunctionTextures.comp once (blocking one-time submit) to fill every slot, and
        // transitions every image to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for the rest of this
        // instance's lifetime.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // Slots 0..kLightFunctionSlotCount-1: ready-to-bind for light_functions.glsl's own bindless
        // g_LightFunctionTextures[] (VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER array write).
        const std::vector<VkDescriptorImageInfo>& GetLightFunctionImageInfos() const { return m_LightFunctionImageInfos; }
        // Slot kCausticsSlot alone: ready-to-bind for caustics_projection.glsl's own single
        // g_CausticsTexture.
        const VkDescriptorImageInfo& GetCausticsImageInfo() const { return m_CausticsImageInfo; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only to vmaDestroyImage() in Shutdown().

        std::vector<VkImage> m_Images;
        std::vector<VmaAllocation> m_Allocations;
        std::vector<VkImageView> m_ImageViews;
        VkSampler m_Sampler = VK_NULL_HANDLE;

        std::vector<VkDescriptorImageInfo> m_LightFunctionImageInfos;
        VkDescriptorImageInfo m_CausticsImageInfo{};
    };

}
