#ifndef WATER_PARAMS_GLSL
#define WATER_PARAMS_GLSL

// Phase 7c (UE5.8 parity roadmap, water/erosion): mirror of VulkanContext.cpp's
// GenerateGeometry() water block constant -- keep in sync (same convention as kTerrainAmplitude
// being shared across terrain_noise.glsl/CPU mirror/terrain_shading.glsl). Chosen relative to
// THIS scene's own terrain range (kTerrainBaseWorldY -0.8 +/- kTerrainAmplitude 0.4, i.e.
// [-1.2, -0.4]) -- sits below the terrain's average height so only its lower basins flood, not
// the old branch's own value (tuned for a completely different, much larger terrain amplitude).
const float kWaterLevel = -1.0;

#endif // WATER_PARAMS_GLSL
