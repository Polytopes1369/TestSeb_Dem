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
} // namespace nanite

namespace temporal {
constexpr float RENDER_SCALE = 0.667f;
constexpr float BLEND_ALPHA = 0.08f;
constexpr float BLEND_ALPHA_STATIC = 0.20f;
constexpr float VARIANCE_CLAMP_FACTOR = 1.5f;
constexpr uint32_t JITTER_FRAME_COUNT = 16u;
constexpr bool ENABLED_BY_DEFAULT = true;
} // namespace temporal

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

constexpr uint32_t MAX_TRACED_ENTITIES = 160u; // Tracks SurfaceCacheTraceContext::kMaxTracedEntities (128 -> 160, 10-tree-species scene): High traces the full scene, matching the pre-tiering baseline.
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 4u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 64u;

// Global SDF clipmap quality (renderer::GlobalSDFPass): voxels per axis per clipmap level, and
// per-entity Mesh SDF bake resolution respectively -- see config_low's own comment on these two
// for the full rationale. Both must stay multiples of 4 (geometry::kSDFBlockDim). High keeps the
// engine's original, pre-tier-scaling defaults (32 / 24) unchanged.
constexpr uint32_t GLOBAL_SDF_CLIPMAP_RESOLUTION = 32u;
constexpr uint32_t GLOBAL_SDF_ENTITY_RESOLUTION = 24u;

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
constexpr bool HARDWARE_RAYTRACING = true;

// Kept off on High to save performance
constexpr bool MEGALIGHTS_ENABLE = false;
} // namespace lumen

namespace postprocess {
// UE 5.8 Post-processing & Effects settings
constexpr uint32_t EFFECTS_QUALITY = 3;
} // namespace postprocess
} // namespace config_high