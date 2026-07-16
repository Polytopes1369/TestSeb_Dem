#pragma once
#include <cstdint>

namespace config {
constexpr uint32_t WINDOW_WIDTH = 1920;
constexpr uint32_t WINDOW_HEIGHT = 1080;
// Set to 0.1f or 0.15f. Smaller value generates more vertices/detail.
// RTX 3060 can easily handle 0.1f density due to higher bandwidth/ALUs than RTX
// 3050.
constexpr float VERTEX_SPACING = 0.1f;
constexpr float FLOOR_VERTEX_SPACING = 2.5f;

namespace nanite {
// Lower threshold shifts more tiny triangles to the optimized software
// rasterizer. UE5 standard/optimal is typically 8.0f pixels.
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 8.0f;
// UE5 standard high-quality LOD threshold is 1.0f.
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 1.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// Increased for RTX 3060 (12GB VRAM) to cache more clusters.
constexpr uint64_t VERTEX_BUFFER_BYTES = 512 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 256 * 1024 * 1024;
} // namespace nanite

namespace lumen {
// RTX 3060 has stronger RT cores, so update budget per frame can be raised.
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 4u;
constexpr uint32_t EVICTION_FRAME_DELAY = 240u;

// Raised to 32u for high resolution probe grid (32^3 = 32768 probes) matching
// UE5 high settings.
constexpr uint32_t PROBE_GRID_RESOLUTION = 32u;
// Reduced spacing to increase local GI feature details.
constexpr float PROBE_SPACING = 2.0f;
// Matches WorldProbeInject.comp's kProbeSampleDirections array length exactly
// (14u).
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

// Tracing limit raised to 64u to match SDFRayMarchPass.
constexpr uint32_t MAX_TRACED_ENTITIES = 64u;

// Multi-bounce radiosity loop iteration budget per frame
constexpr uint32_t RADIOSITY_BOUNCE_COUNT = 3u;

// Number of hemisphere Halton samples per Surface Cache texel for secondary
// bounce irradiance
constexpr uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT =
    16u; // Raised from 8 to 16 for better quality.

// Screen space Probe GI (Lumen Screen Probe Gather) tile dimensions (pixels)
constexpr uint32_t SCREEN_PROBE_TILE_SIZE = 8u;

// Fibonacci-sphere rays traced per screen space probe per frame
constexpr uint32_t SCREEN_PROBE_RAY_COUNT =
    64u; // Must remain 64u to match ScreenProbeTrace.comp hardcoded thread
         // count/SH weight.

// Exponential moving-average blend factor for temporal accumulation of Screen
// GI
constexpr float SCREEN_PROBE_TEMPORAL_ALPHA = 0.05f;

// Virtual Shadow Map (VSM) settings
constexpr uint32_t VSM_SUN_LEVEL_COUNT =
    3u; // Kept at 3u to match shadow_sun_sampling.glsl and ClusterResolve.comp
        // arrays.
constexpr float VSM_SUN_BASE_RADIUS = 2.0f;
constexpr uint32_t VSM_PHYSICAL_PAGE_CAPACITY =
    1024u; // Raised from 256u to 1024u to hold more resident pages.
constexpr uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME =
    128u; // Raised from 32u to 128u.
} // namespace lumen
} // namespace config