#pragma once
// Procedurally generates the bindless cutout mask array used by opacity-masked clusters (tree/
// foliage cutouts) -- zero image assets, matching the project's "100% procedural, no data in the
// exe" rule. See src/shaders/src/Streaming/ProceduralMaskGenerate.comp for the compute shader this
// class drives, and src/shaders/include/mask_sampling.glsl for how consumers (ClusterRaster.frag,
// ClusterSoftwareRaster.comp, ClusterResolve.comp) sample the result.
//
// Run ONCE at startup, before renderer::ClusterRenderPipeline::Init -- exactly like
// geometry::RunVirtualGeometryCacheTest already runs before Init() today. Init() allocates
// kMaxMaskTextures small R8_UNORM images (usable both as a compute storage target during
// generation and a sampled image afterward), runs the generator shader once via a blocking
// one-time submit (mirroring HZBPass::Init's own one-shot transition pattern), transitions every
// image to SHADER_READ_ONLY_OPTIMAL, and exposes the resulting sampler+view pairs as a ready-to-
// use VkDescriptorImageInfo array (GetMaskImageInfos()) that each consumer pass's own Init()
// writes into its own descriptor set at whatever binding it reserves for the mask array -- this
// class does not own or create any consumer-facing descriptor set itself, matching how every
// other borrowed-buffer parameter in this codebase (e.g. GpuGeometryPagePool::
// GetPhysicalPoolBuffer()) is handed to consumers rather than wrapped in a shared descriptor set.

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace renderer {

    class ProceduralMaskGenerator {
    public:
        ProceduralMaskGenerator() = default;

        ProceduralMaskGenerator(const ProceduralMaskGenerator&) = delete;
        ProceduralMaskGenerator& operator=(const ProceduralMaskGenerator&) = delete;

        // Matches K_MAX_MASK_TEXTURES in mask_sampling.glsl / ProceduralMaskGenerate.comp.
        static constexpr uint32_t kMaxMaskTextures = 64;
        // Matches K_MASK_TEXTURE_SIZE in ProceduralMaskGenerate.comp.
        static constexpr uint32_t kMaskTextureSize = 128;
        static constexpr VkFormat kMaskFormat = VK_FORMAT_R8_UNORM;

        // Allocates every mask slot's image/view + one shared sampler, dispatches
        // ProceduralMaskGenerate.comp once (blocking one-time submit) to fill every slot, and
        // transitions every image to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for the rest of this
        // instance's lifetime.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // Ready-to-bind array of kMaxMaskTextures VkDescriptorImageInfo (sampler + view both
        // valid, imageLayout already SHADER_READ_ONLY_OPTIMAL) -- a consumer passes .data()/.size()
        // straight into a VkWriteDescriptorSet with descriptorCount = kMaxMaskTextures and
        // descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER.
        const std::vector<VkDescriptorImageInfo>& GetMaskImageInfos() const { return m_ImageInfos; }

    private:
        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE; // Retained only to vmaDestroyImage() in Shutdown().

        std::vector<VkImage> m_Images;
        std::vector<VmaAllocation> m_Allocations;
        std::vector<VkImageView> m_ImageViews;
        VkSampler m_Sampler = VK_NULL_HANDLE;
        std::vector<VkDescriptorImageInfo> m_ImageInfos;
    };

}
