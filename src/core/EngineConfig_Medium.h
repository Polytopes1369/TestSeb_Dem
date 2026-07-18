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

// --- VIEW DISTANCE ---
// sg.ViewDistanceQuality=3 (Epic view distance, fully drawing large-scale landscapes)
constexpr uint32_t VIEW_DISTANCE_QUALITY = 3;

namespace nanite {
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 8.0f;
// High-quality LOD error threshold.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 1.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

constexpr uint64_t VERTEX_BUFFER_BYTES = 1024 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 512 * 1024 * 1024;

// UE 5.8: r.Nanite.MaxPixelsPerEdge=2.0 (Sharper mesh boundaries)
constexpr float MAX_PIXELS_PER_EDGE = 2.0f;
} // namespace nanite

namespace streaming {
// UE 5.8: r.Streaming.PoolSize=3000 (Safe 3GB cap to prevent OOM crashes on the 8GB RTX 4060)
constexpr uint32_t POOL_SIZE_MB = 3000;
} // namespace streaming

namespace temporal {
// Internal render scale (720p internally, reconstructed to 1080p via TSR)
constexpr float RENDER_SCALE = 0.667f;
constexpr float BLEND_ALPHA = 0.08f;
constexpr float BLEND_ALPHA_STATIC = 0.20f;
constexpr float VARIANCE_CLAMP_FACTOR = 1.5f;
constexpr uint32_t JITTER_FRAME_COUNT = 16u;
constexpr bool ENABLED_BY_DEFAULT = true;

// UE 5.8 Anti-Aliasing & Upscaling Settings
// sg.AntiAliasingQuality=3 (High Quality AA)
constexpr uint32_t ANTI_ALIASING_QUALITY = 3;
constexpr uint32_t ANTI_ALIASING_METHOD = 4;
constexpr float SCREEN_PERCENTAGE = 100.0f;
constexpr uint32_t TEMPORAL_AA_UPSCALER = 1;
constexpr float TSR_HISTORY_SCREEN_PERCENTAGE = 100.0f;
constexpr uint32_t TSR_VELOCITY_HEADING_CONVECTIVE = 1;
} // namespace temporal

namespace lumen {
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 12u;
constexpr uint32_t EVICTION_FRAME_DELAY = 450u;

// Standard 48^3 probe grid for precise GI calculations.
constexpr uint32_t PROBE_GRID_RESOLUTION = 48u;
constexpr float PROBE_SPACING = 1.5f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

constexpr uint32_t MAX_TRACED_ENTITIES = 96u;
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 3u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 48u;

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
// sg.GlobalIlluminationQuality=2 (High)
constexpr uint32_t GI_QUALITY = 2;
// r.Lumen.HardwareRayTracing=1 (RTX 4060 hardware RT enabled)
constexpr bool HARDWARE_RAYTRACING = true;
constexpr bool TRACE_MESH_SDF = true;
constexpr bool SCREEN_SPACE_PROBE_OCCLUSION = true;
constexpr bool REFLECTIONS_ALLOW = true;
// No downsampling for cleaner hardware reflection tracing
constexpr uint32_t REFLECTIONS_DOWNSAMPLE_FACTOR = 1;

constexpr bool HARDWARE_RAYTRACING_NANITE_MODE = false; // Kept off on Medium to save performance
constexpr bool MEGALIGHTS_ENABLE = false;
} // namespace lumen

namespace reflections {
// UE 5.8 Reflections settings
// sg.ReflectionsQuality=2 (High)
constexpr uint32_t QUALITY = 2;
constexpr uint32_t METHOD = 2;
constexpr bool SCREEN_SPACE_REFLECTIONS = true;
} // namespace reflections

namespace postprocess {
// UE 5.8 Post-processing & Effects settings
// sg.PostProcessQuality=3 (High)
constexpr uint32_t QUALITY = 3;
constexpr uint32_t EFFECTS_QUALITY = 2;
constexpr uint32_t TRANSLUCENCY_LIGHTING_VOLUME_DIM = 64;
constexpr uint32_t REFRACTION_QUALITY = 2;
} // namespace postprocess

namespace volumetrics {
// UE 5.8 Volumetrics settings
// sg.TextureQuality=3
constexpr uint32_t TEXTURE_QUALITY = 3;
// sg.SkyAtmosphereQuality=2
constexpr uint32_t SKY_ATMOSPHERE_QUALITY = 2;
constexpr bool VOLUMETRIC_FOG_ENABLE = true;
// Sharper fog with smaller pixel grid size
constexpr uint32_t VOLUMETRIC_FOG_GRID_PIXEL_SIZE = 8;
constexpr float VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = 1.0f;
} // namespace volumetrics
} // namespace config_medium