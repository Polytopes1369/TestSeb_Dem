#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <fstream>
#include <filesystem>
#include <cctype>
#include <format>
#include "core/Logger.h"

#include "EngineConfig_High.h"
#include "EngineConfig_Medium.h"
#include "EngineConfig_Low.h"

namespace config {
// Window size is 1920x1080 across all profiles
inline uint32_t WINDOW_WIDTH = 1920;
inline uint32_t WINDOW_HEIGHT = 1080;
inline float VERTEX_SPACING = 0.05f;
inline float FLOOR_VERTEX_SPACING = 1.0f;

// Temporary kill-switch
inline bool ENTITY_SELF_ROTATION_ENABLED = false;

// --- VIEW DISTANCE ---
inline uint32_t _VIEW_DISTANCE_QUALITY = 4;

namespace nanite {
// Lower threshold shifts more tiny triangles to the software rasterizer.
inline float SOFTWARE_RASTER_THRESHOLD_PIXELS = 8.0f;
// Cinematic quality LOD error threshold.
inline float LOD_PIXEL_ERROR_THRESHOLD = 0.5f;

// Structural constants: identical in all profiles, must remain constexpr for static sizes
constexpr uint32_t MAX_CLUSTER_VERTICES = 64u;
constexpr uint32_t MAX_CLUSTER_TRIANGLES = 128u;
constexpr uint32_t PAGE_SIZE_BYTES = 4096u;

// Allocated buffer sizes
inline uint64_t VERTEX_BUFFER_BYTES = 1024 * 1024 * 1024;
inline uint64_t INDEX_BUFFER_BYTES = 512 * 1024 * 1024;

inline float _MAX_PIXELS_PER_EDGE = 1.0f;
} // namespace nanite

namespace streaming {
inline uint32_t _POOL_SIZE_MB = 8000;
} // namespace streaming

namespace temporal {
inline float RENDER_SCALE = 1.000f;
inline float BLEND_ALPHA = 0.08f;
inline float BLEND_ALPHA_STATIC = 0.20f;
inline float VARIANCE_CLAMP_FACTOR = 1.5f;
inline uint32_t JITTER_FRAME_COUNT = 16u;
inline bool ENABLED_BY_DEFAULT = true;

inline uint32_t _ANTI_ALIASING_QUALITY = 4;
inline uint32_t _ANTI_ALIASING_METHOD = 4;
inline float _SCREEN_PERCENTAGE = 100.0f;
inline uint32_t _TEMPORAL_AA_UPSCALER = 1;
inline float _TSR_HISTORY_SCREEN_PERCENTAGE = 100.0f;
inline uint32_t _TSR_VELOCITY_HEADING_CONVECTIVE = 1;
} // namespace temporal

namespace shadows {
inline uint32_t _QUALITY = 4;
inline bool _VIRTUAL_ENABLE = true;
inline uint32_t _MAX_RESOLUTION = 4096;
inline uint32_t _CSM_MAX_CASCADES = 6;
inline float _DISTANCE_SCALE = 1.50f;
} // namespace shadows

namespace lumen {
inline uint32_t CARDS_PER_FRAME_BUDGET = 16u;
inline uint32_t EVICTION_FRAME_DELAY = 600u;

inline uint32_t PROBE_GRID_RESOLUTION = 64u;
inline float PROBE_SPACING = 1.0f;
inline uint32_t PROBE_SAMPLE_DIRECTIONS = 14u;

inline uint32_t MAX_TRACED_ENTITIES = 128u;
inline uint32_t RADIOSITY_BOUNCE_COUNT = 4u;
inline uint32_t SURFACE_CACHE_GI_SAMPLE_COUNT = 64u;

inline uint32_t SCREEN_PROBE_TILE_SIZE = 8u;
inline uint32_t SCREEN_PROBE_RAY_COUNT = 64u;
inline float SCREEN_PROBE_TEMPORAL_ALPHA = 0.05f;

inline bool BUILD_SHADOWS = true;

constexpr uint32_t VSM_SUN_LEVEL_COUNT = 3u;
inline float VSM_SUN_BASE_RADIUS = 2.0f;
inline uint32_t VSM_PHYSICAL_PAGE_CAPACITY = 4096u;
inline uint32_t VSM_MAX_PAGES_RENDERED_PER_FRAME = 512u;

inline uint32_t _GI_QUALITY = 4;
inline bool _HARDWARE_RAYTRACING = true;
inline bool _TRACE_MESH_SDF = true;
inline bool _SCREEN_SPACE_PROBE_OCCLUSION = true;
inline bool _REFLECTIONS_ALLOW = true;
inline uint32_t _REFLECTIONS_DOWNSAMPLE_FACTOR = 1;
inline bool _HARDWARE_RAYTRACING_NANITE_MODE = true;
inline bool _MEGALIGHTS_ENABLE = true;
} // namespace lumen

namespace reflections {
inline uint32_t _QUALITY = 4;
inline uint32_t _METHOD = 2;
inline bool _SCREEN_SPACE_REFLECTIONS = true;
} // namespace reflections

namespace postprocess {
inline uint32_t _QUALITY = 4;
inline uint32_t _EFFECTS_QUALITY = 4;
inline uint32_t _TRANSLUCENCY_LIGHTING_VOLUME_DIM = 64;
inline uint32_t _REFRACTION_QUALITY = 3;
} // namespace postprocess

namespace volumetrics {
inline uint32_t _TEXTURE_QUALITY = 4;
inline uint32_t _SKY_ATMOSPHERE_QUALITY = 3;
inline bool _VOLUMETRIC_FOG_ENABLE = true;
inline uint32_t _VOLUMETRIC_FOG_GRID_PIXEL_SIZE = 4;
inline float _VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = 2.0f;
} // namespace volumetrics

// Active loaded state
inline bool g_ProfileLoaded = false;
inline std::string g_ActiveProfileName = "High"; // Default to High properties

inline void ApplyProfile(std::string_view profileName) {
    if (profileName == "High") {
        VERTEX_SPACING = config_high::VERTEX_SPACING;
        _VIEW_DISTANCE_QUALITY = config_high::VIEW_DISTANCE_QUALITY;
        nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS = config_high::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
        nanite::LOD_PIXEL_ERROR_THRESHOLD = config_high::nanite::LOD_PIXEL_ERROR_THRESHOLD;
        nanite::VERTEX_BUFFER_BYTES = config_high::nanite::VERTEX_BUFFER_BYTES;
        nanite::INDEX_BUFFER_BYTES = config_high::nanite::INDEX_BUFFER_BYTES;
        nanite::_MAX_PIXELS_PER_EDGE = config_high::nanite::MAX_PIXELS_PER_EDGE;
        streaming::_POOL_SIZE_MB = config_high::streaming::POOL_SIZE_MB;
        temporal::RENDER_SCALE = config_high::temporal::RENDER_SCALE;
        temporal::BLEND_ALPHA = config_high::temporal::BLEND_ALPHA;
        temporal::BLEND_ALPHA_STATIC = config_high::temporal::BLEND_ALPHA_STATIC;
        temporal::VARIANCE_CLAMP_FACTOR = config_high::temporal::VARIANCE_CLAMP_FACTOR;
        temporal::JITTER_FRAME_COUNT = config_high::temporal::JITTER_FRAME_COUNT;
        temporal::ENABLED_BY_DEFAULT = config_high::temporal::ENABLED_BY_DEFAULT;
        temporal::_ANTI_ALIASING_QUALITY = config_high::temporal::ANTI_ALIASING_QUALITY;
        temporal::_ANTI_ALIASING_METHOD = config_high::temporal::ANTI_ALIASING_METHOD;
        temporal::_SCREEN_PERCENTAGE = config_high::temporal::SCREEN_PERCENTAGE;
        temporal::_TEMPORAL_AA_UPSCALER = config_high::temporal::TEMPORAL_AA_UPSCALER;
        temporal::_TSR_HISTORY_SCREEN_PERCENTAGE = config_high::temporal::TSR_HISTORY_SCREEN_PERCENTAGE;
        temporal::_TSR_VELOCITY_HEADING_CONVECTIVE = config_high::temporal::TSR_VELOCITY_HEADING_CONVECTIVE;
        shadows::_QUALITY = config_high::shadows::QUALITY;
        shadows::_VIRTUAL_ENABLE = config_high::shadows::VIRTUAL_ENABLE;
        shadows::_MAX_RESOLUTION = config_high::shadows::MAX_RESOLUTION;
        shadows::_CSM_MAX_CASCADES = config_high::shadows::CSM_MAX_CASCADES;
        shadows::_DISTANCE_SCALE = config_high::shadows::DISTANCE_SCALE;
        lumen::CARDS_PER_FRAME_BUDGET = config_high::lumen::CARDS_PER_FRAME_BUDGET;
        lumen::EVICTION_FRAME_DELAY = config_high::lumen::EVICTION_FRAME_DELAY;
        lumen::PROBE_GRID_RESOLUTION = config_high::lumen::PROBE_GRID_RESOLUTION;
        lumen::PROBE_SPACING = config_high::lumen::PROBE_SPACING;
        lumen::PROBE_SAMPLE_DIRECTIONS = config_high::lumen::PROBE_SAMPLE_DIRECTIONS;
        lumen::MAX_TRACED_ENTITIES = config_high::lumen::MAX_TRACED_ENTITIES;
        lumen::RADIOSITY_BOUNCE_COUNT = config_high::lumen::RADIOSITY_BOUNCE_COUNT;
        lumen::SURFACE_CACHE_GI_SAMPLE_COUNT = config_high::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;
        lumen::SCREEN_PROBE_TILE_SIZE = config_high::lumen::SCREEN_PROBE_TILE_SIZE;
        lumen::SCREEN_PROBE_RAY_COUNT = config_high::lumen::SCREEN_PROBE_RAY_COUNT;
        lumen::SCREEN_PROBE_TEMPORAL_ALPHA = config_high::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;
        lumen::BUILD_SHADOWS = config_high::lumen::BUILD_SHADOWS;
        lumen::VSM_SUN_BASE_RADIUS = config_high::lumen::VSM_SUN_BASE_RADIUS;
        lumen::VSM_PHYSICAL_PAGE_CAPACITY = config_high::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
        lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME = config_high::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
        lumen::_GI_QUALITY = config_high::lumen::GI_QUALITY;
        lumen::_HARDWARE_RAYTRACING = config_high::lumen::HARDWARE_RAYTRACING;
        lumen::_TRACE_MESH_SDF = config_high::lumen::TRACE_MESH_SDF;
        lumen::_SCREEN_SPACE_PROBE_OCCLUSION = config_high::lumen::SCREEN_SPACE_PROBE_OCCLUSION;
        lumen::_REFLECTIONS_ALLOW = config_high::lumen::REFLECTIONS_ALLOW;
        lumen::_REFLECTIONS_DOWNSAMPLE_FACTOR = config_high::lumen::REFLECTIONS_DOWNSAMPLE_FACTOR;
        lumen::_HARDWARE_RAYTRACING_NANITE_MODE = config_high::lumen::HARDWARE_RAYTRACING_NANITE_MODE;
        lumen::_MEGALIGHTS_ENABLE = config_high::lumen::MEGALIGHTS_ENABLE;
        reflections::_QUALITY = config_high::reflections::QUALITY;
        reflections::_METHOD = config_high::reflections::METHOD;
        reflections::_SCREEN_SPACE_REFLECTIONS = config_high::reflections::SCREEN_SPACE_REFLECTIONS;
        postprocess::_QUALITY = config_high::postprocess::QUALITY;
        postprocess::_EFFECTS_QUALITY = config_high::postprocess::EFFECTS_QUALITY;
        postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM = config_high::postprocess::TRANSLUCENCY_LIGHTING_VOLUME_DIM;
        postprocess::_REFRACTION_QUALITY = config_high::postprocess::REFRACTION_QUALITY;
        volumetrics::_TEXTURE_QUALITY = config_high::volumetrics::TEXTURE_QUALITY;
        volumetrics::_SKY_ATMOSPHERE_QUALITY = config_high::volumetrics::SKY_ATMOSPHERE_QUALITY;
        volumetrics::_VOLUMETRIC_FOG_ENABLE = config_high::volumetrics::VOLUMETRIC_FOG_ENABLE;
        volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE = config_high::volumetrics::VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
        volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = config_high::volumetrics::VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE;
    } else if (profileName == "Medium") {
        VERTEX_SPACING = config_medium::VERTEX_SPACING;
        _VIEW_DISTANCE_QUALITY = config_medium::VIEW_DISTANCE_QUALITY;
        nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS = config_medium::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
        nanite::LOD_PIXEL_ERROR_THRESHOLD = config_medium::nanite::LOD_PIXEL_ERROR_THRESHOLD;
        nanite::VERTEX_BUFFER_BYTES = config_medium::nanite::VERTEX_BUFFER_BYTES;
        nanite::INDEX_BUFFER_BYTES = config_medium::nanite::INDEX_BUFFER_BYTES;
        nanite::_MAX_PIXELS_PER_EDGE = config_medium::nanite::MAX_PIXELS_PER_EDGE;
        streaming::_POOL_SIZE_MB = config_medium::streaming::POOL_SIZE_MB;
        temporal::RENDER_SCALE = config_medium::temporal::RENDER_SCALE;
        temporal::BLEND_ALPHA = config_medium::temporal::BLEND_ALPHA;
        temporal::BLEND_ALPHA_STATIC = config_medium::temporal::BLEND_ALPHA_STATIC;
        temporal::VARIANCE_CLAMP_FACTOR = config_medium::temporal::VARIANCE_CLAMP_FACTOR;
        temporal::JITTER_FRAME_COUNT = config_medium::temporal::JITTER_FRAME_COUNT;
        temporal::ENABLED_BY_DEFAULT = config_medium::temporal::ENABLED_BY_DEFAULT;
        temporal::_ANTI_ALIASING_QUALITY = config_medium::temporal::ANTI_ALIASING_QUALITY;
        temporal::_ANTI_ALIASING_METHOD = config_medium::temporal::ANTI_ALIASING_METHOD;
        temporal::_SCREEN_PERCENTAGE = config_medium::temporal::SCREEN_PERCENTAGE;
        temporal::_TEMPORAL_AA_UPSCALER = config_medium::temporal::TEMPORAL_AA_UPSCALER;
        temporal::_TSR_HISTORY_SCREEN_PERCENTAGE = config_medium::temporal::TSR_HISTORY_SCREEN_PERCENTAGE;
        temporal::_TSR_VELOCITY_HEADING_CONVECTIVE = config_medium::temporal::TSR_VELOCITY_HEADING_CONVECTIVE;
        shadows::_QUALITY = config_medium::shadows::QUALITY;
        shadows::_VIRTUAL_ENABLE = config_medium::shadows::VIRTUAL_ENABLE;
        shadows::_MAX_RESOLUTION = config_medium::shadows::MAX_RESOLUTION;
        shadows::_CSM_MAX_CASCADES = config_medium::shadows::CSM_MAX_CASCADES;
        shadows::_DISTANCE_SCALE = config_medium::shadows::DISTANCE_SCALE;
        lumen::CARDS_PER_FRAME_BUDGET = config_medium::lumen::CARDS_PER_FRAME_BUDGET;
        lumen::EVICTION_FRAME_DELAY = config_medium::lumen::EVICTION_FRAME_DELAY;
        lumen::PROBE_GRID_RESOLUTION = config_medium::lumen::PROBE_GRID_RESOLUTION;
        lumen::PROBE_SPACING = config_medium::lumen::PROBE_SPACING;
        lumen::PROBE_SAMPLE_DIRECTIONS = config_medium::lumen::PROBE_SAMPLE_DIRECTIONS;
        lumen::MAX_TRACED_ENTITIES = config_medium::lumen::MAX_TRACED_ENTITIES;
        lumen::RADIOSITY_BOUNCE_COUNT = config_medium::lumen::RADIOSITY_BOUNCE_COUNT;
        lumen::SURFACE_CACHE_GI_SAMPLE_COUNT = config_medium::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;
        lumen::SCREEN_PROBE_TILE_SIZE = config_medium::lumen::SCREEN_PROBE_TILE_SIZE;
        lumen::SCREEN_PROBE_RAY_COUNT = config_medium::lumen::SCREEN_PROBE_RAY_COUNT;
        lumen::SCREEN_PROBE_TEMPORAL_ALPHA = config_medium::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;
        lumen::BUILD_SHADOWS = config_medium::lumen::BUILD_SHADOWS;
        lumen::VSM_SUN_BASE_RADIUS = config_medium::lumen::VSM_SUN_BASE_RADIUS;
        lumen::VSM_PHYSICAL_PAGE_CAPACITY = config_medium::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
        lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME = config_medium::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
        lumen::_GI_QUALITY = config_medium::lumen::GI_QUALITY;
        lumen::_HARDWARE_RAYTRACING = config_medium::lumen::HARDWARE_RAYTRACING;
        lumen::_TRACE_MESH_SDF = config_medium::lumen::TRACE_MESH_SDF;
        lumen::_SCREEN_SPACE_PROBE_OCCLUSION = config_medium::lumen::SCREEN_SPACE_PROBE_OCCLUSION;
        lumen::_REFLECTIONS_ALLOW = config_medium::lumen::REFLECTIONS_ALLOW;
        lumen::_REFLECTIONS_DOWNSAMPLE_FACTOR = config_medium::lumen::REFLECTIONS_DOWNSAMPLE_FACTOR;
        lumen::_HARDWARE_RAYTRACING_NANITE_MODE = config_medium::lumen::HARDWARE_RAYTRACING_NANITE_MODE;
        lumen::_MEGALIGHTS_ENABLE = config_medium::lumen::MEGALIGHTS_ENABLE;
        reflections::_QUALITY = config_medium::reflections::QUALITY;
        reflections::_METHOD = config_medium::reflections::METHOD;
        reflections::_SCREEN_SPACE_REFLECTIONS = config_medium::reflections::SCREEN_SPACE_REFLECTIONS;
        postprocess::_QUALITY = config_medium::postprocess::QUALITY;
        postprocess::_EFFECTS_QUALITY = config_medium::postprocess::EFFECTS_QUALITY;
        postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM = config_medium::postprocess::TRANSLUCENCY_LIGHTING_VOLUME_DIM;
        postprocess::_REFRACTION_QUALITY = config_medium::postprocess::REFRACTION_QUALITY;
        volumetrics::_TEXTURE_QUALITY = config_medium::volumetrics::TEXTURE_QUALITY;
        volumetrics::_SKY_ATMOSPHERE_QUALITY = config_medium::volumetrics::SKY_ATMOSPHERE_QUALITY;
        volumetrics::_VOLUMETRIC_FOG_ENABLE = config_medium::volumetrics::VOLUMETRIC_FOG_ENABLE;
        volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE = config_medium::volumetrics::VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
        volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = config_medium::volumetrics::VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE;
    } else if (profileName == "Low") {
        VERTEX_SPACING = config_low::VERTEX_SPACING;
        _VIEW_DISTANCE_QUALITY = config_low::VIEW_DISTANCE_QUALITY;
        nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS = config_low::nanite::SOFTWARE_RASTER_THRESHOLD_PIXELS;
        nanite::LOD_PIXEL_ERROR_THRESHOLD = config_low::nanite::LOD_PIXEL_ERROR_THRESHOLD;
        nanite::VERTEX_BUFFER_BYTES = config_low::nanite::VERTEX_BUFFER_BYTES;
        nanite::INDEX_BUFFER_BYTES = config_low::nanite::INDEX_BUFFER_BYTES;
        nanite::_MAX_PIXELS_PER_EDGE = config_low::nanite::MAX_PIXELS_PER_EDGE;
        streaming::_POOL_SIZE_MB = config_low::streaming::POOL_SIZE_MB;
        temporal::RENDER_SCALE = config_low::temporal::RENDER_SCALE;
        temporal::BLEND_ALPHA = config_low::temporal::BLEND_ALPHA;
        temporal::BLEND_ALPHA_STATIC = config_low::temporal::BLEND_ALPHA_STATIC;
        temporal::VARIANCE_CLAMP_FACTOR = config_low::temporal::VARIANCE_CLAMP_FACTOR;
        temporal::JITTER_FRAME_COUNT = config_low::temporal::JITTER_FRAME_COUNT;
        temporal::ENABLED_BY_DEFAULT = config_low::temporal::ENABLED_BY_DEFAULT;
        temporal::_ANTI_ALIASING_QUALITY = config_low::temporal::ANTI_ALIASING_QUALITY;
        temporal::_ANTI_ALIASING_METHOD = config_low::temporal::ANTI_ALIASING_METHOD;
        temporal::_SCREEN_PERCENTAGE = config_low::temporal::SCREEN_PERCENTAGE;
        temporal::_TEMPORAL_AA_UPSCALER = config_low::temporal::TEMPORAL_AA_UPSCALER;
        temporal::_TSR_HISTORY_SCREEN_PERCENTAGE = config_low::temporal::TSR_HISTORY_SCREEN_PERCENTAGE;
        temporal::_TSR_VELOCITY_HEADING_CONVECTIVE = config_low::temporal::TSR_VELOCITY_HEADING_CONVECTIVE;
        shadows::_QUALITY = config_low::shadows::QUALITY;
        shadows::_VIRTUAL_ENABLE = config_low::shadows::VIRTUAL_ENABLE;
        shadows::_MAX_RESOLUTION = config_low::shadows::MAX_RESOLUTION;
        shadows::_CSM_MAX_CASCADES = config_low::shadows::CSM_MAX_CASCADES;
        shadows::_DISTANCE_SCALE = config_low::shadows::DISTANCE_SCALE;
        lumen::CARDS_PER_FRAME_BUDGET = config_low::lumen::CARDS_PER_FRAME_BUDGET;
        lumen::EVICTION_FRAME_DELAY = config_low::lumen::EVICTION_FRAME_DELAY;
        lumen::PROBE_GRID_RESOLUTION = config_low::lumen::PROBE_GRID_RESOLUTION;
        lumen::PROBE_SPACING = config_low::lumen::PROBE_SPACING;
        lumen::PROBE_SAMPLE_DIRECTIONS = config_low::lumen::PROBE_SAMPLE_DIRECTIONS;
        lumen::MAX_TRACED_ENTITIES = config_low::lumen::MAX_TRACED_ENTITIES;
        lumen::RADIOSITY_BOUNCE_COUNT = config_low::lumen::RADIOSITY_BOUNCE_COUNT;
        lumen::SURFACE_CACHE_GI_SAMPLE_COUNT = config_low::lumen::SURFACE_CACHE_GI_SAMPLE_COUNT;
        lumen::SCREEN_PROBE_TILE_SIZE = config_low::lumen::SCREEN_PROBE_TILE_SIZE;
        lumen::SCREEN_PROBE_RAY_COUNT = config_low::lumen::SCREEN_PROBE_RAY_COUNT;
        lumen::SCREEN_PROBE_TEMPORAL_ALPHA = config_low::lumen::SCREEN_PROBE_TEMPORAL_ALPHA;
        lumen::BUILD_SHADOWS = config_low::lumen::BUILD_SHADOWS;
        lumen::VSM_SUN_BASE_RADIUS = config_low::lumen::VSM_SUN_BASE_RADIUS;
        lumen::VSM_PHYSICAL_PAGE_CAPACITY = config_low::lumen::VSM_PHYSICAL_PAGE_CAPACITY;
        lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME = config_low::lumen::VSM_MAX_PAGES_RENDERED_PER_FRAME;
        lumen::_GI_QUALITY = config_low::lumen::GI_QUALITY;
        lumen::_HARDWARE_RAYTRACING = config_low::lumen::HARDWARE_RAYTRACING;
        lumen::_TRACE_MESH_SDF = config_low::lumen::TRACE_MESH_SDF;
        lumen::_SCREEN_SPACE_PROBE_OCCLUSION = config_low::lumen::SCREEN_SPACE_PROBE_OCCLUSION;
        lumen::_REFLECTIONS_ALLOW = config_low::lumen::REFLECTIONS_ALLOW;
        lumen::_REFLECTIONS_DOWNSAMPLE_FACTOR = config_low::lumen::REFLECTIONS_DOWNSAMPLE_FACTOR;
        lumen::_HARDWARE_RAYTRACING_NANITE_MODE = config_low::lumen::HARDWARE_RAYTRACING_NANITE_MODE;
        lumen::_MEGALIGHTS_ENABLE = config_low::lumen::MEGALIGHTS_ENABLE;
        reflections::_QUALITY = config_low::reflections::QUALITY;
        reflections::_METHOD = config_low::reflections::METHOD;
        reflections::_SCREEN_SPACE_REFLECTIONS = config_low::reflections::SCREEN_SPACE_REFLECTIONS;
        postprocess::_QUALITY = config_low::postprocess::QUALITY;
        postprocess::_EFFECTS_QUALITY = config_low::postprocess::EFFECTS_QUALITY;
        postprocess::_TRANSLUCENCY_LIGHTING_VOLUME_DIM = config_low::postprocess::TRANSLUCENCY_LIGHTING_VOLUME_DIM;
        postprocess::_REFRACTION_QUALITY = config_low::postprocess::REFRACTION_QUALITY;
        volumetrics::_TEXTURE_QUALITY = config_low::volumetrics::TEXTURE_QUALITY;
        volumetrics::_SKY_ATMOSPHERE_QUALITY = config_low::volumetrics::SKY_ATMOSPHERE_QUALITY;
        volumetrics::_VOLUMETRIC_FOG_ENABLE = config_low::volumetrics::VOLUMETRIC_FOG_ENABLE;
        volumetrics::_VOLUMETRIC_FOG_GRID_PIXEL_SIZE = config_low::volumetrics::VOLUMETRIC_FOG_GRID_PIXEL_SIZE;
        volumetrics::_VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE = config_low::volumetrics::VOLUMETRIC_CLOUD_VIEW_RAY_SAMPLE_COUNT_SCALE;
    }
}

inline void SaveProfileLocal(std::string_view profileName) {
    std::ofstream outFile("gpu_profile.cfg");
    if (outFile.is_open()) {
        outFile << profileName;
        outFile.close();
        LOG_INFO(std::format("[EngineConfig] Saved profile '{}' locally to gpu_profile.cfg", profileName));
    } else {
        LOG_WARNING("[EngineConfig] Failed to save profile locally to gpu_profile.cfg");
    }
}

inline bool LoadProfileLocal() {
    std::ifstream inFile("gpu_profile.cfg");
    if (inFile.is_open()) {
        std::string profileName;
        inFile >> profileName;
        inFile.close();
        if (profileName == "High" || profileName == "Medium" || profileName == "Low") {
            ApplyProfile(profileName);
            g_ActiveProfileName = profileName;
            g_ProfileLoaded = true;
            LOG_INFO(std::format("[EngineConfig] Loaded saved profile from local cache: {}", profileName));
            return true;
        }
    }
    return false;
}

inline void InitializeProfileFromGPU(std::string_view deviceName) {
    if (g_ProfileLoaded) {
        return; // Already loaded from cache
    }

    std::string profile = "Medium"; // Default fallback
    std::string deviceUpper = "";
    for (char c : deviceName) {
        deviceUpper += std::toupper(static_cast<unsigned char>(c));
    }

    if (deviceUpper.find("5080") != std::string::npos || 
        deviceUpper.find("5090") != std::string::npos ||
        deviceUpper.find("BLACKWELL") != std::string::npos) {
        profile = "High";
    } else if (deviceUpper.find("4090") != std::string::npos || 
               deviceUpper.find("4080") != std::string::npos || 
               deviceUpper.find("4070") != std::string::npos || 
               deviceUpper.find("4060") != std::string::npos ||
               deviceUpper.find("ADA LOVELACE") != std::string::npos) {
        profile = "Medium";
    } else if (deviceUpper.find("3090") != std::string::npos || 
               deviceUpper.find("3080") != std::string::npos || 
               deviceUpper.find("3070") != std::string::npos || 
               deviceUpper.find("3060") != std::string::npos ||
               deviceUpper.find("2080") != std::string::npos ||
               deviceUpper.find("2070") != std::string::npos ||
               deviceUpper.find("2060") != std::string::npos ||
               deviceUpper.find("AMPERE") != std::string::npos ||
               deviceUpper.find("TURING") != std::string::npos) {
        profile = "Low";
    }

    LOG_INFO(std::format("[EngineConfig] Scan complete. Selected profile based on GPU: {}", profile));
    ApplyProfile(profile);
    g_ActiveProfileName = profile;
    g_ProfileLoaded = true;
    SaveProfileLocal(profile);
}
} // namespace config