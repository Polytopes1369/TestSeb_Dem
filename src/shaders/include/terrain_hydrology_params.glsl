#ifndef TERRAIN_HYDROLOGY_PARAMS_GLSL
#define TERRAIN_HYDROLOGY_PARAMS_GLSL

// Terrain hydrology feature (procedural water & erosion simulation): constants shared between the
// GPU pipe-model simulation (TerrainHydrology.comp), the terrain mesh generator
// (geom_terrain.comp), the water-surface mesh generator (geom_water_surface.comp), the terrain
// biome shading (terrain_shading.glsl via ClusterResolve*.comp) and the water fragment shader
// (WaterForward.frag). CPU mirror: renderer::TerrainHydrologySim (TerrainHydrologySim.h) -- keep
// in sync, same single-source-of-truth convention kWaterLevel/water_params.glsl established.

// Simulation grid resolution (texels per axis) and the world-space footprint it covers -- the SAME
// 300x300m footprint, centered at the world origin, that the terrain entity itself covers
// (VulkanContext::GenerateGeometry's terrain block). One texel = 300/512 =~ 0.586 world units.
const float kHydroFootprint = 300.0;
const int   kHydroResolution = 512;
const float kHydroCellSize = kHydroFootprint / float(kHydroResolution);

// Sea level in the terrain's own LOCAL height space (before the mesh's worldOffsetY anchor is
// added). water_params.glsl's kWaterLevel (-1.0, world space) minus terrain_shading.glsl's
// kTerrainBaseWorldY (-0.8) -- kept as an explicit constant so every simulation-side use is
// anchor-free.
const float kHydroSeaLevelLocal = -0.2;

// The terrain entity's world-space vertical anchor (VulkanContext::GenerateTerrain's call-site
// worldOffsetY; mirrors terrain_shading.glsl's kTerrainBaseWorldY -- keep all three in sync).
// Converts the simulation's LOCAL heights to world Y: worldY = localHeight + this.
const float kHydroTerrainAnchorWorldY = -0.8;

// Continental macro relief (terrain_noise.glsl's SampleTerrainHeight): radial bands from the world
// origin. The showcase gallery zone keeps its authored look untouched; beyond it the terrain rises
// through rolling plains into a mountain ring, then falls off a coastal shelf to a sea floor well
// below sea level, so the map reads as an island: gallery -> plains -> mountains -> beach -> sea.
const float kHydroGalleryRadius   = 34.0;  // Untouched showcase zone (existing look + no erosion carving).
const float kHydroPlainsEnd       = 78.0;  // Plains band outer edge / mountain ramp start.
const float kHydroMountainPeakIn  = 96.0;  // Mountain band full-strength inner radius.
const float kHydroMountainPeakOut = 112.0; // Mountain band full-strength outer radius.
const float kHydroCoastStart      = 126.0; // Coastal falloff start (mountain ramp fully out).
const float kHydroCoastEnd        = 142.0; // Beyond this: open sea floor.
const float kHydroMountainAmplitude = 7.0; // Peak height added by the mountain band (local units).
const float kHydroMountainUplift    = 1.1; // Base uplift under the mountain band.
const float kHydroPlainsSwell       = 0.6; // Gentle extra low-frequency relief across the plains.
const float kHydroSeaFloorLocal     = -3.2; // Local height the coastal shelf falls to.

// Pipe-model hydraulic erosion (TerrainHydrology.comp) -- classic virtual-pipes scheme
// (Mei/Decaudin/Hu 2007): per-iteration rain, 4-neighbor flux exchange, velocity from flux,
// capacity-based erosion/deposition, semi-Lagrangian sediment advection, evaporation. All rates
// are per iteration (dt folded in), tuned for kHydroIterations total steps.
const int   kHydroIterations       = 260;
const float kHydroPipeK            = 0.25;  // Flux gain per unit height difference (explicit-integration stable at <= 0.25).
const float kHydroFluxDamping      = 0.998; // Keeps standing oscillations from ringing forever.
const float kHydroRainPerIteration = 0.006;
const float kHydroEvaporation      = 0.015; // Fraction of standing water removed per iteration.
const float kHydroCapacityK        = 0.9;   // Sediment capacity = K * slope * |velocity| * water clamp.
const float kHydroErodeRate        = 0.30;  // Fraction of (capacity - sediment) eroded per iteration.
const float kHydroDepositRate      = 0.25;  // Fraction of (sediment - capacity) deposited per iteration.
// Carve/deposit clamps relative to the INITIAL (pre-erosion) height -- keeps the simulation from
// digging through the gallery floor or burying the coastline however the tuning above drifts.
const float kHydroMaxCarve   = 1.8;
const float kHydroMaxDeposit = 1.2;

// Water-surface mesh (geom_water_surface.comp): spans BEYOND the terrain footprint so the sea
// reads as extending toward the horizon past the island's edge (the water-surface texture's
// CLAMP_TO_EDGE border texels are open sea, so the overhang samples a flat sea level).
const float kHydroWaterMeshSpan = 600.0;

// Water fragments with less depth than this discard (WaterForward.frag) -- and the water-surface
// mesh tucks its vertices this far UNDER the terrain where there is no water, so dry-land water
// surface is double-guarded (depth-tested away AND discarded).
const float kHydroMinWaterDepth = 0.015;

// Converts a local terrain-space XZ position (mesh-local, world == local for this entity's XZ --
// the terrain is centered at the origin) into the hydrology textures' UV.
vec2 HydroUVFromLocalXZ(vec2 xz) {
    return xz / kHydroFootprint + 0.5;
}

#endif // TERRAIN_HYDROLOGY_PARAMS_GLSL
