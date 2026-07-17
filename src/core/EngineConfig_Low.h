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

// --- VIEW DISTANCE ---
// sg.ViewDistanceQuality=2 (Far view distance, optimized for performance)
constexpr uint32_t VIEW_DISTANCE_QUALITY = 2;

namespace nanite {
// Higher threshold shifts more tiny triangles to the software rasterizer to relieve pipeline stress.
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 16.0f;
// Balanced LOD error threshold.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 2.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// Lower memory allocation for vertex buffers
constexpr uint64_t VERTEX_BUFFER_BYTES = 512 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 256 * 1024 * 1024;

// UE 5.8: r.Nanite.MaxPixelsPerEdge=4.0
constexpr float MAX_PIXELS_PER_EDGE = 4.0f;
} // namespace nanite

namespace streaming {
// UE 5.8: r.Streaming.PoolSize=4500 (Allocating 4.5GB leveraging the 12GB VRAM on the RTX 3060)
constexpr uint32_t POOL_SIZE_MB = 4500;
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
// sg.AntiAliasingQuality=2 (Medium Quality AA with TSR upscaling)
constexpr uint32_t ANTI_ALIASING_QUALITY = 2;
// r.AntiAliasing.Method=4 (TSR / DLSS)
constexpr uint32_t ANTI_ALIASING_METHOD = 4;
// r.ScreenPercentage=66.7
constexpr float SCREEN_PERCENTAGE = 66.7f;
constexpr uint32_t TEMPORAL_AA_UPSCALER = 1;
constexpr float TSR_HISTORY_SCREEN_PERCENTAGE = 100.0f;
constexpr uint32_t TSR_VELOCITY_HEADING_CONVECTIVE = 1;
} // namespace temporal

namespace shadows {
// UE 5.8 Shadows settings (Classic Cascaded Shadow Maps fallback to save GPU cycles)
// sg.ShadowQuality=1
constexpr uint32_t QUALITY = 1;
// r.Shadow.Virtual.Enable=0 (VSM disabled, too heavy for this architecture)
constexpr bool VIRTUAL_ENABLE = false;
constexpr uint32_t MAX_RESOLUTION = 1024;
constexpr uint32_t CSM_MAX_CASCADES = 2;
constexpr float DISTANCE_SCALE = 0.85f;
} // namespace shadows

namespace lumen {
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 8u; // Reduced for performance
constexpr uint32_t EVICTION_FRAME_DELAY = 300u;

// 32^3 probe grid to keep global illumination lightweight.
constexpr uint32_t PROBE_GRID_RESOLUTION = 32u;
constexpr float PROBE_SPACING = 2.0f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

constexpr uint32_t MAX_TRACED_ENTITIES = 64u;
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 2u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 24u;

constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 16u; // Larger tiles = fewer probes traced
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 32u; // Reduced ray budget
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.10f;

constexpr bool BUILD_SHADOWS = true;

// Classic Shadow Map Fallbacks
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 1024u; // Minimized
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 128u;

// UE 5.8 Global Illumination settings (Lumen Software)
// sg.GlobalIlluminationQuality=1 (Medium)
constexpr uint32_t GI_QUALITY = 1;
// r.Lumen.HardwareRayTracing=0 (Software Ray Tracing only on the 3060)
constexpr bool HARDWARE_RAYTRACING = false;
constexpr bool TRACE_MESH_SDF = true;
constexpr bool SCREEN_SPACE_PROBE_OCCLUSION = true;
constexpr bool REFLECTIONS_ALLOW = true;
// Downsampled reflections to protect framerate
constexpr uint32_t REFLECTIONS_DOWNSAMPLE_FACTOR = 2;

// UE 5.8 specific feature switches disabled on Low
constexpr bool HARDWARE_RAYTRACING_NANITE_MODE = false;
constexpr bool MEGALIGHTS_ENABLE = false;
} // namespace lumen

namespace reflections {
// UE 5.8 Reflections settings
// sg.ReflectionsQuality=1 (Medium)
constexpr uint32_t QUALITY = 1;
// r.ReflectionMethod=2 (Lumen)
constexpr uint32_t METHOD = 2;
constexpr bool SCREEN_SPACE_REFLECTIONS = true;
} // namespace reflections

namespace postprocess {
// UE 5.8 Post-processing & Effects settings
// sg.PostProcessQuality=2 (Medium)
constexpr uint32_t QUALITY = 2;
// sg.EffectsQuality=1
constexpr uint32_t EFFECTS_QUALITY = 1;
constexpr uint32_t TRANSLUCENCY_LIGHTING_VOLUME_DIM = 32;
constexpr uint32_t REFRACTION_QUALITY = 1;
} // namespace postprocess

namespace volumetrics {
// UE 5.8 Volumetrics settings
// sg.TextureQuality=2
constexpr uint32_t TEXTURE_QUALITY = 2;
// sg.SkyAtmosphereQuality=1
constexpr uint32_t SKY_ATMOSPHERE_QUALITY = 1;
constexpr bool VOLUMETRIC_FOG_ENABLE = true;
// Larger grid size to minimize pixel shader cost
constexpr uint32_t VOLUMETRIC_FOG_GRID_PIXEL_SIZE = 16;
constexpr float VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = 0.5f;
} // namespace volumetrics
} // namespace config_low