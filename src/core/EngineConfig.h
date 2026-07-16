#pragma once
#include <cstdint>

namespace config {
constexpr uint32_t WINDOW_WIDTH = 1920;
constexpr uint32_t WINDOW_HEIGHT = 1080;
constexpr float VERTEX_SPACING = 0.1f;
// Tessellation density for large, perfectly flat ground planes (e.g. the world
// floor). geom_plane.comp emits a constant up-normal with no displacement, so
// fine subdivision buys nothing visually; reusing VERTEX_SPACING (tuned for
// ~1-2m hero primitives) on a 300m floor would compute a 3000x3000 grid --
// 9M vertices -- and overflow the fixed-size geometry SSBOs
// (nanite::VERTEX_BUFFER_BYTES / INDEX_BUFFER_BYTES below).
constexpr float FLOOR_VERTEX_SPACING = 0.5f;

namespace nanite {
constexpr float SOFTWARE_RASTER_THRESHOLD_PIXELS = 6.0f;
constexpr float LOD_PIXEL_ERROR_THRESHOLD = 1.0f;

constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

constexpr uint64_t VERTEX_BUFFER_BYTES = 128 * 1024 * 1024;
constexpr uint64_t INDEX_BUFFER_BYTES = 64 * 1024 * 1024;
} // namespace nanite

namespace lumen {
constexpr uint32_t CARDS_PER_FRAME_BUDGET = 4u;
constexpr uint32_t EVICTION_FRAME_DELAY = 120u;

constexpr uint32_t PROBE_GRID_RESOLUTION = 32u;
constexpr float PROBE_SPACING = 2.0f;
constexpr uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

constexpr uint32_t MAX_TRACED_ENTITIES = 64u;
} // namespace lumen
} // namespace config
