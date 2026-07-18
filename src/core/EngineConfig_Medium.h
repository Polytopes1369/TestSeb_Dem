#pragma once
#include <cstdint>

namespace config_medium {
constexpr uint32_t WINDOW_WIDTH = 1920;
constexpr uint32_t WINDOW_HEIGHT = 1080;
constexpr uint32_t TARGET_FPS = 60;
// Medium: 0.10f spacing balances high geometric detail with Ada Lovelace compute throughput.
constexpr float VERTEX_SPACING = 0.10f;
constexpr float FLOOR_VERTEX_SPACING = 1.0f;

// Temporary kill-switch
constexpr bool ENTITY_SELF_ROTATION_ENABLED = false;

namespace nanite {
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 8.0f;
// High-quality LOD error threshold.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 1.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

constexpr uint64_t VERTEX_BUFFER_BYTES = 1024 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 512 * 1024 * 1024;
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
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 12u;
constexpr uint32_t EVICTION_FRAME_DELAY = 450u;

// Surface Cache atlas resolution -- see EngineConfig_Low.h's own comment on this value. Midway
// between Low's 1024 and High/Extrem's 2048.
constexpr uint32_t SURFACE_CACHE_ATLAS_SIZE = 1536u;

// Standard 48^3 probe grid for precise GI calculations.
constexpr uint32_t PROBE_GRID_RESOLUTION = 48u;
constexpr float PROBE_SPACING = 1.5f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

constexpr uint32_t MAX_TRACED_ENTITIES = 96u;
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 3u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 48u;

// Global SDF clipmap quality (renderer::GlobalSDFPass): voxels per axis per clipmap level, and
// per-entity Mesh SDF bake resolution respectively -- see config_low's own comment on these two
// for the full rationale. Both must stay multiples of 4 (geometry::kSDFBlockDim).
constexpr uint32_t GLOBAL_SDF_CLIPMAP_RESOLUTION = 28u;
constexpr uint32_t GLOBAL_SDF_ENTITY_RESOLUTION = 20u;

constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 8u;
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 48u;
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.08f;

constexpr bool BUILD_SHADOWS = true;

// Virtual Shadow Map (VSM) settings
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 2048u;
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 256u;

// UE 5.8 Global Illumination settings (Lumen Hardware RT)
// r.Lumen.HardwareRayTracing=1 (RTX 4060 hardware RT enabled)
constexpr bool HARDWARE_RAYTRACING = true;

constexpr bool MEGALIGHTS_ENABLE = false;
} // namespace lumen

namespace postprocess {
// UE 5.8 Post-processing & Effects settings
constexpr uint32_t EFFECTS_QUALITY = 2;
} // namespace postprocess
} // namespace config_medium