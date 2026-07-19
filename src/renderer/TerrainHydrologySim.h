#pragma once
// Terrain hydrology feature: CPU-side driver for the GPU pipe-model water & hydraulic erosion
// bake (src/shaders/src/Renderer/TerrainHydrology.comp -- see that shader's own header comment
// for the simulation scheme). Owned by renderer::VulkanContext and run ONCE, synchronously,
// BEFORE any geometry generation: geom_terrain.comp / geom_water_surface.comp sample this bake's
// output textures at mesh-generation time (shared geometry descriptor set, binding 6), and
// ClusterResolve*.comp / WaterForward.frag sample the attributes texture every frame at shading
// time. The bake is fully deterministic (fixed iteration count, no RNG state, no wall-clock
// input), so the on-disk geometry cache built from its output stays valid across runs -- the same
// determinism contract world::PcgCellLoader's own bake-vs-runtime test enforces for PCG content.
//
// Mirrors src/shaders/include/terrain_hydrology_params.glsl's constants -- keep in sync (same
// single-source-of-truth convention water_params.glsl / kWaterLevel established).

#include <cstdint>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

namespace renderer {

    class TerrainHydrologySim {
    public:
        TerrainHydrologySim() = default;

        TerrainHydrologySim(const TerrainHydrologySim&) = delete;
        TerrainHydrologySim& operator=(const TerrainHydrologySim&) = delete;

        // Mirrors terrain_hydrology_params.glsl's kHydroResolution / kHydroIterations exactly.
        static constexpr uint32_t kResolution = 512;
        static constexpr int32_t kIterations = 260;
        static constexpr uint32_t kWorkgroupSize = 8; // Matches TerrainHydrology.comp's local_size.

        // Creates every grid image + the compute pipeline, then records and synchronously submits
        // the ENTIRE bake (init -> kIterations erosion steps -> blur chains -> finalize) via a
        // one-shot command buffer. Blocking by design: the caller (VulkanContext) generates the
        // terrain/water meshes from these textures immediately afterward. Throws on any Vulkan
        // failure, matching VulkanContext's own init-time error convention.
        void Init(VkDevice device, VmaAllocator allocator, VkCommandPool commandPool, VkQueue queue);

        void Shutdown();

        // Sampled outputs (all kResolution^2, VK_IMAGE_LAYOUT_GENERAL for their whole lifetime,
        // matching this codebase's storage-image convention):
        //  - MeshHeight (R32F): low-passed eroded terrain height -- geom_terrain.comp's vertex Y.
        //  - WaterSurface (R32F): water surface Y where submerged, tucked under terrain where dry
        //    -- geom_water_surface.comp's vertex Y.
        //  - Attributes (RGBA16F): (height, waterDepth, flow, moisture) -- terrain biome shading
        //    (ClusterResolve*.comp) and the water fragment discard/tint (WaterForward.frag).
        VkImageView GetMeshHeightView() const { return m_MeshHeightView; }
        VkImageView GetWaterSurfaceView() const { return m_WaterSurfaceView; }
        VkImageView GetAttributesView() const { return m_AttributesView; }
        VkSampler GetLinearSampler() const { return m_LinearSampler; }

    private:
        // One grid image + its view; all owned images share this shape.
        struct GridImage {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
        };

        GridImage CreateGrid(VkFormat format);
        void DestroyGrid(GridImage& grid);
        void RecordBake(VkCommandBuffer cmd);

        VkDevice m_Device = VK_NULL_HANDLE;
        VmaAllocator m_Allocator = VK_NULL_HANDLE;

        // Simulation state grids (see TerrainHydrology.comp's binding table).
        GridImage m_Height;       // binding 0, R32F
        GridImage m_BaseHeight;   // binding 1, R32F
        GridImage m_Water;        // binding 2, R32F
        GridImage m_Flux;         // binding 3, RGBA32F
        GridImage m_Velocity;     // binding 4, RG32F
        GridImage m_SedimentA;    // binding 5, R32F
        GridImage m_SedimentB;    // binding 6, R32F
        GridImage m_TempA;        // binding 7, R32F
        GridImage m_TempB;        // binding 8, R32F
        // Sampled outputs.
        GridImage m_MeshHeight;   // binding 9,  R32F
        GridImage m_Attributes;   // binding 10, RGBA16F
        GridImage m_WaterSurface; // binding 11, R32F

        VkImageView m_MeshHeightView = VK_NULL_HANDLE;   // Alias of m_MeshHeight.view (kept for the getter's clarity).
        VkImageView m_WaterSurfaceView = VK_NULL_HANDLE; // Alias of m_WaterSurface.view.
        VkImageView m_AttributesView = VK_NULL_HANDLE;   // Alias of m_Attributes.view.

        VkSampler m_LinearSampler = VK_NULL_HANDLE;

        VkDescriptorSetLayout m_SetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        // Two sets differing ONLY in bindings 5/6 (sediment ping/pong swapped): the advection step
        // writes B from A, and the next iteration must read the advected field as "A" -- alternating
        // descriptor sets per iteration swaps the pair without any image copy.
        VkDescriptorSet m_SetEven = VK_NULL_HANDLE;
        VkDescriptorSet m_SetOdd = VK_NULL_HANDLE;

        VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
        VkPipeline m_Pipeline = VK_NULL_HANDLE;
    };

}
