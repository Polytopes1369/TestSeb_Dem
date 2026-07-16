#pragma once
#include <cstdint>

namespace config {
constexpr uint32_t WINDOW_WIDTH = 1920;
constexpr uint32_t WINDOW_HEIGHT = 1080;
// Set to 0.2f or 0.25f. Divides the generated vertex count by 4 (or more). 
// Risk of noticeable detail loss on "hero" primitives, but required by RTX 3050 bandwidth constraints.
constexpr float VERTEX_SPACING = 0.25f;
constexpr float FLOOR_VERTEX_SPACING = 4.0f;

namespace nanite {
// The RTX 3050 has fewer ALUs, so the software rasterizer is the primary bottleneck.
// We need to shift more workload towards the hardware rasterizer.
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 12.0f;
// Force a more aggressive LOD transition to limit the geometry being processed.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 2.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

constexpr uint64_t VERTEX_BUFFER_BYTES = 128 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 64 * 1024 * 1024;
} // namespace nanite

namespace lumen {
// The RTX 3050 Ampere RT cores are limited. Surface cache updates are expensive.
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 1u; // or 2u maximum
constexpr uint32_t EVICTION_FRAME_DELAY = 120u;

// This is where you save the most CPU/GPU execution time: 32^3 = 32K probes is untenable; 16^3 = 4K probes.
constexpr uint32_t PROBE_GRID_RESOLUTION = 16u;
constexpr float PROBE_SPACING = 4.0f; // Expands coverage to compensate for the lower grid resolution.
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 8u; // Fewer rays traced per probe.

constexpr uint32_t MAX_TRACED_ENTITIES = 32u;

// Multi-bounce radiosity loop iteration budget per frame
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 3u;

// Number of hemisphere Halton samples per Surface Cache texel for secondary bounce irradiance
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 8u;

// Screen space Probe GI (Lumen Screen Probe Gather) tile dimensions (pixels)
constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 8u;

// Fibonacci-sphere rays traced per screen space probe per frame
constexpr uint32_t SCREEN_PROBE_RAY_COUNT = 64u;

// Exponential moving-average blend factor for temporal accumulation of Screen GI
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.05f;

// Virtual Shadow Map (VSM) settings
constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 256u;
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 32u;
} // namespace lumen
} // namespace config