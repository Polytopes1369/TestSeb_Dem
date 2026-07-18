#pragma once
#include <cstdint>

namespace config_extrem {
constexpr uint32_t WINDOW_WIDTH = 3840;
constexpr uint32_t WINDOW_HEIGHT = 2160;
constexpr uint32_t TARGET_FPS = 60;
// Extrem: 0.02f for ultra-dense raw procedural geometry on Blackwell architectures.
constexpr float VERTEX_SPACING = 0.02f;
constexpr float FLOOR_VERTEX_SPACING = 1.0f;

// Temporary kill-switch
constexpr bool ENTITY_SELF_ROTATION_ENABLED = false;

// --- VIEW DISTANCE ---
// sg.ViewDistanceQuality=5 (Extreme cinematic view distance, maximum asset LOD caching)
constexpr uint32_t VIEW_DISTANCE_QUALITY = 5;

namespace nanite {
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 1.0f;
// Cinematic/VR quality LOD error threshold.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 0.25f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// 2GB Vertex buffer / 1GB Index buffer allocated for extreme geometry density
constexpr uint64_t VERTEX_BUFFER_BYTES = 2ULL * 1024 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 1ULL * 1024 * 1024 * 1024;

// UE 5.8: r.Nanite.MaxPixelsPerEdge=0.5 (Cinematic rendering, virtually zero triangle simplification)
constexpr float MAX_PIXELS_PER_EDGE = 0.5f;
} // namespace nanite

namespace streaming {
// UE 5.8: r.Streaming.PoolSize=12000 (Generous 12GB allocation budget for extreme high-res textures)
constexpr uint32_t POOL_SIZE_MB = 12000;
} // namespace streaming

namespace temporal {
// Internal render scale (1440p internally, reconstructed to 4K via TSR)
constexpr float RENDER_SCALE = 0.667f;
constexpr float BLEND_ALPHA = 0.08f;
constexpr float BLEND_ALPHA_STATIC = 0.20f;
constexpr float VARIANCE_CLAMP_FACTOR = 1.5f;
constexpr uint32_t JITTER_FRAME_COUNT = 16u;
constexpr bool ENABLED_BY_DEFAULT = true;

// sg.AntiAliasingQuality=5 (Extreme Quality AA)
constexpr uint32_t ANTI_ALIASING_QUALITY = 5;
constexpr uint32_t ANTI_ALIASING_METHOD = 4;
constexpr float SCREEN_PERCENTAGE = 150.0f;
constexpr uint32_t TEMPORAL_AA_UPSCALER = 1;
constexpr float TSR_HISTORY_SCREEN_PERCENTAGE = 150.0f;
constexpr uint32_t TSR_VELOCITY_HEADING_CONVECTIVE = 1;
} // namespace temporal

namespace lumen {
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 32u;
constexpr uint32_t EVICTION_FRAME_DELAY = 1200u;

// Extreme-quality 80^3 probe grid (512k probes) for flawless global illumination.
constexpr uint32_t PROBE_GRID_RESOLUTION = 80u;
constexpr float PROBE_SPACING = 0.8f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 24u;

constexpr uint32_t MAX_TRACED_ENTITIES = 256u;
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 6u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 128u; // Extreme quality bounce GI

constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 4u;
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 128u;
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.02f;

constexpr bool BUILD_SHADOWS = true;

// Virtual Shadow Map (VSM) settings
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 8192u;
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 1024u;

// sg.GlobalIlluminationQuality=5 (Extreme Cinematic)
constexpr uint32_t GI_QUALITY = 5;
constexpr bool HARDWARE_RAYTRACING = true;
constexpr bool TRACE_MESH_SDF = true;
constexpr bool REFLECTIONS_ALLOW = true;
constexpr uint32_t REFLECTIONS_DOWNSAMPLE_FACTOR = 1;

constexpr bool HARDWARE_RAYTRACING_NANITE_MODE = true;
constexpr bool MEGALIGHTS_ENABLE = true; // Enabled to handle thousands of physics-based dynamic lights
} // namespace lumen

namespace reflections {
// sg.ReflectionsQuality=5 (Extreme Cinematic)
constexpr uint32_t QUALITY = 5;
constexpr uint32_t METHOD = 2;
constexpr bool SCREEN_SPACE_REFLECTIONS = true;
} // namespace reflections

namespace postprocess {
// sg.PostProcessQuality=5 (Extreme Cinematic)
constexpr uint32_t QUALITY = 5;
constexpr uint32_t EFFECTS_QUALITY = 5;
constexpr uint32_t REFRACTION_QUALITY = 4;
} // namespace postprocess

namespace volumetrics {
// sg.TextureQuality=5
constexpr uint32_t TEXTURE_QUALITY = 5;
// sg.SkyAtmosphereQuality=4 (Extreme scattering)
constexpr uint32_t SKY_ATMOSPHERE_QUALITY = 4;
constexpr bool VOLUMETRIC_FOG_ENABLE = true;
} // namespace volumetrics
} // namespace config_extrem