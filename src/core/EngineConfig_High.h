#pragma once
#include <cstdint>

namespace config_high {
constexpr uint32_t WINDOW_WIDTH = 1920;
constexpr uint32_t WINDOW_HEIGHT = 1080;
constexpr uint32_t TARGET_FPS = 60;
// High: 0.05f spacing for dense geometry. Excellent throughput on high-end Ada Lovelace architectures.
constexpr float VERTEX_SPACING = 0.05f;
constexpr float FLOOR_VERTEX_SPACING = 1.0f;

// Temporary kill-switch
constexpr bool ENTITY_SELF_ROTATION_ENABLED = false;

// --- VIEW DISTANCE ---
// sg.ViewDistanceQuality=4 (Epic-grade view distance target)
constexpr uint32_t VIEW_DISTANCE_QUALITY = 4;

namespace nanite {
// Sharp geometry handling
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 4.0f;
// Epic quality LOD error threshold.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 0.5f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// 1.5GB vertex buffer / 768MB index buffer -- strictly between Medium (1GB/512MB) and Extrem
// (2GB/1GB). Previously byte-for-byte identical to Medium's own values (an unintentional bug: the
// single biggest VRAM knob in the engine did not differentiate these two tiers at all), fixed here
// as the arithmetic midpoint, preserving the same 2:1 vertex:index ratio every other tier uses.
constexpr uint64_t VERTEX_BUFFER_BYTES = 1536 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 768 * 1024 * 1024;

// UE 5.8: r.Nanite.MaxPixelsPerEdge=1.0 (Epic geometry resolution, maximum details)
constexpr float MAX_PIXELS_PER_EDGE = 1.0f;
} // namespace nanite

namespace streaming {
// UE 5.8: r.Streaming.PoolSize=6000 (Generous allocation for 12GB+ GPU architectures)
constexpr uint32_t POOL_SIZE_MB = 6000;
} // namespace streaming

namespace temporal {
constexpr float RENDER_SCALE = 0.667f;
constexpr float BLEND_ALPHA = 0.08f;
constexpr float BLEND_ALPHA_STATIC = 0.20f;
constexpr float VARIANCE_CLAMP_FACTOR = 1.5f;
constexpr uint32_t JITTER_FRAME_COUNT = 16u;
constexpr bool ENABLED_BY_DEFAULT = true;

// UE 5.8 Anti-Aliasing & Upscaling Settings
// sg.AntiAliasingQuality=4 (Epic Quality AA)
constexpr uint32_t ANTI_ALIASING_QUALITY = 4;
constexpr uint32_t ANTI_ALIASING_METHOD = 4;
constexpr float SCREEN_PERCENTAGE = 100.0f;
constexpr uint32_t TEMPORAL_AA_UPSCALER = 1;
constexpr float TSR_HISTORY_SCREEN_PERCENTAGE = 100.0f;
constexpr uint32_t TSR_VELOCITY_HEADING_CONVECTIVE = 1;
} // namespace temporal

namespace shadows {
// UE 5.8 Shadows settings (Epic Virtual Shadow Maps)
// sg.ShadowQuality=3 (Epic)
constexpr uint32_t QUALITY = 3;
constexpr bool VIRTUAL_ENABLE = true;
constexpr uint32_t MAX_RESOLUTION = 4096;
constexpr uint32_t CSM_MAX_CASCADES = 4;
constexpr float DISTANCE_SCALE = 1.20f;
} // namespace shadows

namespace lumen {
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 16u;
constexpr uint32_t EVICTION_FRAME_DELAY = 600u;

// Surface Cache atlas resolution -- see EngineConfig_Low.h's own comment on this value. Full
// resolution: matches the original fixed 2048 footprint this engine always ran at pre-tiering.
constexpr uint32_t SURFACE_CACHE_ATLAS_SIZE = 2048u;

// High-quality 64^3 probe grid (262k probes) for flawless global illumination.
constexpr uint32_t PROBE_GRID_RESOLUTION = 64u;
constexpr float PROBE_SPACING = 1.0f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

constexpr uint32_t MAX_TRACED_ENTITIES = 128u;
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 4u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 64u;

constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 8u;
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 64u;
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.05f;

constexpr bool BUILD_SHADOWS = true;

// Virtual Shadow Map (VSM) settings
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 4096u;
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 512u;

// UE 5.8 Global Illumination settings (Lumen Hardware RT + Epic Preset)
// sg.GlobalIlluminationQuality=3 (Epic)
constexpr uint32_t GI_QUALITY = 3;
constexpr bool HARDWARE_RAYTRACING = true;
constexpr bool TRACE_MESH_SDF = true;
constexpr bool SCREEN_SPACE_PROBE_OCCLUSION = true;
constexpr bool REFLECTIONS_ALLOW = true;
constexpr uint32_t REFLECTIONS_DOWNSAMPLE_FACTOR = 1;

// Hardware Ray Tracing on Nanite geometries enabled for high-end
constexpr bool HARDWARE_RAYTRACING_NANITE_MODE = true;
// Kept off on High to save performance
constexpr bool MEGALIGHTS_ENABLE = false;
} // namespace lumen

namespace reflections {
// UE 5.8 Reflections settings
// sg.ReflectionsQuality=3 (Epic)
constexpr uint32_t QUALITY = 3;
constexpr uint32_t METHOD = 2;
constexpr bool SCREEN_SPACE_REFLECTIONS = true;
} // namespace reflections

namespace postprocess {
// UE 5.8 Post-processing & Effects settings
// sg.PostProcessQuality=4 (Epic)
constexpr uint32_t QUALITY = 4;
constexpr uint32_t EFFECTS_QUALITY = 3;
constexpr uint32_t TRANSLUCENCY_LIGHTING_VOLUME_DIM = 64;
constexpr uint32_t REFRACTION_QUALITY = 3;
} // namespace postprocess

namespace volumetrics {
// UE 5.8 Volumetrics settings (Fog & Clouds)
// sg.TextureQuality=3
constexpr uint32_t TEXTURE_QUALITY = 3;
// sg.SkyAtmosphereQuality=3 (Epic atmosphere scattering)
constexpr uint32_t SKY_ATMOSPHERE_QUALITY = 3;
constexpr bool VOLUMETRIC_FOG_ENABLE = true;
// Sharp volumetric light shafts
constexpr uint32_t VOLUMETRIC_FOG_GRID_PIXEL_SIZE = 8;
constexpr float VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = 1.5f;
} // namespace volumetrics
} // namespace config_high