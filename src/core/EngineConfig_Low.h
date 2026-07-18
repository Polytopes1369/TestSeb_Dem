#pragma once
#include <cstdint>

namespace config_low {
constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr uint32_t TARGET_FPS = 30;
// Low: Spacing relaxed to reduce vertex density on the RTX 3060 compute queue.
constexpr float VERTEX_SPACING = 0.15f;
constexpr float FLOOR_VERTEX_SPACING = 1.0f;

// Temporary kill-switch
constexpr bool ENTITY_SELF_ROTATION_ENABLED = false;

namespace nanite {
// Higher threshold shifts more tiny triangles to the software rasterizer to
// relieve pipeline stress.
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 16.0f;
// Balanced LOD error threshold.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 2.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// Lower memory allocation for vertex buffers
constexpr uint64_t VERTEX_BUFFER_BYTES = 512 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 256 * 1024 * 1024;
} // namespace nanite

namespace temporal {
// Internal render scale (720p internally, reconstructed to 1080p via TSR)
constexpr float RENDER_SCALE = 0.667f;
constexpr float BLEND_ALPHA = 0.08f;
constexpr float BLEND_ALPHA_STATIC = 0.20f;
constexpr float VARIANCE_CLAMP_FACTOR = 1.5f;
constexpr uint32_t JITTER_FRAME_COUNT = 16u;
constexpr bool ENABLED_BY_DEFAULT = true;
} // namespace temporal

namespace lumen {
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 8u; // Reduced for performance
constexpr uint32_t EVICTION_FRAME_DELAY = 300u;

// Surface Cache atlas resolution (square, texels): unlike CARDS_PER_FRAME_BUDGET above (which only
// tunes the per-frame UPDATE RATE), this tunes the actual VRAM footprint of the 6 atlas images
// renderer::SurfaceCachePass::Init() allocates -- formerly a hardcoded geometry::
// kSurfaceCacheAtlasSize (2048) regardless of tier, identical on Low and Extrem. Halved from the
// original 2048 to meaningfully cut VRAM on entry-level GPUs; the runtime SurfaceCacheAtlasAllocator
// degrades gracefully under a smaller budget (evicts off-screen cards / defragments, see that
// class's own comment), so a smaller atlas here costs GI update latency under heavy scenes, never a
// crash or corruption.
constexpr uint32_t SURFACE_CACHE_ATLAS_SIZE = 1024u;

// 32^3 probe grid to keep global illumination lightweight.
constexpr uint32_t PROBE_GRID_RESOLUTION = 32u;
constexpr float PROBE_SPACING = 2.0f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

constexpr uint32_t MAX_TRACED_ENTITIES = 64u;
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 2u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 24u;

// Global SDF clipmap quality (renderer::GlobalSDFPass): voxels per axis per clipmap level, and
// per-entity Mesh SDF bake resolution respectively. Coarser on Low -- shorter/blockier cone-
// tracing empty-space skipping, matching how every other Lumen quality knob above scales down --
// even though the resulting volume is tiny in VRAM at every tier (a few hundred KB to ~2MB, see
// GlobalSDFPass.h's own kClipmapResolution comment), so this is purely a trace-quality knob, not
// a memory-pressure one. Both must stay multiples of 4 (geometry::kSDFBlockDim, the BC4-style
// compression block size geometry::BuildMeshSDF requires -- see MeshSDFGenerator.h).
constexpr uint32_t GLOBAL_SDF_CLIPMAP_RESOLUTION = 24u;
constexpr uint32_t GLOBAL_SDF_ENTITY_RESOLUTION = 16u;

constexpr uint32_t SCREEN_PROBE_TILE_SIZE =
    16u; // Larger tiles = fewer probes traced
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 32u; // Reduced ray budget
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.10f;

constexpr bool BUILD_SHADOWS = true;

// Classic Shadow Map Fallbacks
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 1024u; // Minimized
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 128u;

// UE 5.8 Global Illumination settings (Lumen Software)
// r.Lumen.HardwareRayTracing=0 (Software Ray Tracing only on the 3060)
constexpr bool HARDWARE_RAYTRACING = false;

// UE 5.8 specific feature switches disabled on Low
constexpr bool MEGALIGHTS_ENABLE = false;
} // namespace lumen

namespace postprocess {
// UE 5.8 Post-processing & Effects settings
// sg.EffectsQuality=1
constexpr uint32_t EFFECTS_QUALITY = 1;
} // namespace postprocess
} // namespace config_low