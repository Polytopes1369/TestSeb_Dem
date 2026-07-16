#pragma once
#include <cstdint>

namespace config {
constexpr uint32_t WINDOW_WIDTH = 1920;
constexpr uint32_t WINDOW_HEIGHT = 1080;
// Set to 0.05f for ultra-dense procedural geometry, fully supported by the RTX 5080.
constexpr float VERTEX_SPACING = 0.05f;
constexpr float FLOOR_VERTEX_SPACING = 1.0f;

namespace nanite {
// Lower threshold shifts more tiny triangles to the software rasterizer.
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 8.0f;
// Cinematic quality LOD error threshold.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 0.5f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// Allocated 1GB / 512MB buffers to store dynamic virtual geometries.
constexpr uint64_t VERTEX_BUFFER_BYTES = 1024 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 512 * 1024 * 1024;
} // namespace nanite

namespace lumen {
// Massively increased card update budget for high-speed dynamic updates.
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 16u;
constexpr uint32_t EVICTION_FRAME_DELAY = 600u; // Kept in memory longer.

// Ultra-quality 64^3 probe grid (262k probes) for smooth high-end GI.
constexpr uint32_t PROBE_GRID_RESOLUTION = 64u;
constexpr float PROBE_SPACING = 1.0f;
// Matches WorldProbeInject.comp's kProbeSampleDirections array length exactly (14u).
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

// Tracing limit raised to 128u.
constexpr uint32_t MAX_TRACED_ENTITIES = 128u;

// Multi-bounce radiosity loop iteration budget per frame
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 4u;

// Number of hemisphere Halton samples per Surface Cache texel for secondary bounce irradiance
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 64u; // Epic quality secondary GI.

// Screen space Probe GI (Lumen Screen Probe Gather) tile dimensions (pixels)
constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 8u;

// Fibonacci-sphere rays traced per screen space probe per frame
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 64u; // Must remain 64u to match ScreenProbeTrace.comp hardcoded thread count/SH weight.

// Exponential moving-average blend factor for temporal accumulation of Screen GI
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.05f;

// Temporary kill-switch: when false, VirtualShadowMapPass never allocates/renders any shadow
// page (sun clipmap or point light cube faces alike). shadow_sun_sampling.glsl/
// shadow_point_sampling.glsl already fall back to fully-lit (1.0) for any non-resident page, so
// leaving every page permanently non-resident disables shadows cleanly with no risk to the rest
// of the pipeline (view-projection matrices, feedback buffer, etc. keep updating normally).
constexpr bool BUILD_SHADOWS = false;

// Virtual Shadow Map (VSM) settings
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u; // Kept at 3u to match shadow_sun_sampling.glsl and ClusterResolve.comp arrays.
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 4096u; // Large page capacity (4096 * 128^2 * 4B = 256 MB) to prevent any eviction.
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 512u;
} // namespace lumen
} // namespace config