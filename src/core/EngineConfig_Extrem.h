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
} // namespace nanite

namespace temporal {
// Internal render scale (1440p internally, reconstructed to 4K via TSR)
// Extrem tier has GPU headroom for near-native TSR input resolution (user-validated audit
// decision 2026-07-19).
constexpr float RENDER_SCALE = 0.85f;
constexpr float BLEND_ALPHA = 0.08f;
constexpr float BLEND_ALPHA_STATIC = 0.20f;
constexpr float VARIANCE_CLAMP_FACTOR = 1.5f;
constexpr uint32_t JITTER_FRAME_COUNT = 16u;
constexpr bool ENABLED_BY_DEFAULT = true;
} // namespace temporal

namespace lumen {
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 32u;
constexpr uint32_t EVICTION_FRAME_DELAY = 1200u;

// Surface Cache atlas resolution -- see EngineConfig_Low.h's own comment on this value. Capped at
// the same 2048 as High rather than scaling further: atlas resolution is not this tier's GI
// differentiator (PROBE_GRID_RESOLUTION/MAX_TRACED_ENTITIES/RADIOSITY_BOUNCE_COUNT/
// SURFACE_CACHE_GI_SAMPLE_COUNT above already scale GI fidelity at Extrem), and 2048 was the
// original engine-wide fixed size, so this tier is guaranteed no worse than the pre-tiering baseline.
constexpr uint32_t SURFACE_CACHE_ATLAS_SIZE = 2048u;

// Extreme-quality 80^3 probe grid (512k probes) for flawless global illumination.
constexpr uint32_t PROBE_GRID_RESOLUTION = 80u;
constexpr float PROBE_SPACING = 0.8f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 24u;

constexpr uint32_t MAX_TRACED_ENTITIES = 256u;
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 6u;
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 128u; // Extreme quality bounce GI

// Global SDF clipmap quality (renderer::GlobalSDFPass): voxels per axis per clipmap level, and
// per-entity Mesh SDF bake resolution respectively -- see config_low's own comment on these two
// for the full rationale. Both must stay multiples of 4 (geometry::kSDFBlockDim). Entity
// resolution matches geometry::kMeshSDFResolution (32) at this tier -- Extreme's per-entity
// Global SDF bake is as fine as the surface-detail mesh SDF itself.
constexpr uint32_t GLOBAL_SDF_CLIPMAP_RESOLUTION = 48u;
constexpr uint32_t GLOBAL_SDF_ENTITY_RESOLUTION = 32u;

constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 4u;
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 128u;
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.02f;

constexpr bool BUILD_SHADOWS = true;

// Virtual Shadow Map (VSM) settings
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 8192u;
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 1024u;

constexpr bool HARDWARE_RAYTRACING = true;

constexpr bool MEGALIGHTS_ENABLE = true; // Enabled to handle thousands of physics-based dynamic lights
} // namespace lumen

namespace postprocess {
constexpr uint32_t EFFECTS_QUALITY = 5;
} // namespace postprocess
} // namespace config_extrem