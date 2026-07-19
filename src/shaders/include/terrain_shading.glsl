#ifndef TERRAIN_SHADING_GLSL
#define TERRAIN_SHADING_GLSL

// Phase 7b (UE5.8 parity roadmap, terrain heightfield) + terrain hydrology feature: procedural
// height/slope/moisture biome blending for the terrain material. A single MATERIAL_ID_TERRAIN
// recipe (MaterialParameterTable.h) supplies a flat fallback baseColor; this include overrides it
// per-pixel using the already-interpolated worldPos/normal available at ClusterResolve.comp/
// ClusterResolveBinned.comp's material-evaluation step PLUS the erosion bake's attributes texture
// (renderer::TerrainHydrologySim: height, waterDepth, flow, moisture -- sampled by the caller and
// passed in, since the binding index differs between the two resolve shaders). The hydrology
// texture is FULL resolution (~0.6 world units/texel) while the mesh is 4-unit-spaced: shading
// detail (wet river channels, beach lines, moisture darkening) therefore resolves far finer than
// the silhouette -- deliberately, see TerrainHydrology.comp's own mesh-safety comment.
#include "terrain_noise.glsl"
#include "water_params.glsl" // Phase 7c: kWaterLevel, for the beach/waterline bands below.

// Mirrors VulkanContext::GenerateGeometry()'s terrain call site (worldOffsetY), which reuses the
// same kFloorTopY anchor the flat floor plane used before this entity replaced it -- keep in sync
// (also mirrored as terrain_hydrology_params.glsl's kHydroTerrainAnchorWorldY).
const float kTerrainBaseWorldY = -0.8;

// World-unit width of the sand band above kWaterLevel. Widened from the pre-hydrology 0.06 (tuned
// for the old +/-0.4-amplitude backdrop): the continental relief now runs from a -4.0 sea floor to
// ~+7 mountain peaks, and the coastline is a real feature the beach should read clearly on.
const float kBeachBandWidth = 0.35;

vec3 ComputeTerrainAlbedo(vec3 worldPos, vec3 normal, vec3 fallbackAlbedo, vec4 hydro) {
    float slopeFactor = clamp(normal.y, 0.0, 1.0); // 1 = flat, 0 = vertical cliff

    vec3 grass    = vec3(0.24, 0.42, 0.15);
    vec3 dryGrass = vec3(0.42, 0.44, 0.20);
    vec3 rock     = vec3(0.40, 0.36, 0.32);
    vec3 snow     = vec3(0.90, 0.92, 0.96);
    vec3 sand     = vec3(0.76, 0.70, 0.50);
    vec3 sediment = vec3(0.45, 0.38, 0.28); // Eroded channel/deposit tint.

    // Large-scale grass hue variation (dry vs lush patches) -- pure albedo detail, cheap, breaks
    // up the otherwise-uniform grass expanse at plains scale. Same ValueNoise3D hash the height
    // fbm already uses (decorrelated plane y=53).
    float grassVariation = ValueNoise3D(vec3(worldPos.x * 0.05, 53.0, worldPos.z * 0.05));
    vec3 ground = mix(grass, dryGrass, smoothstep(0.35, 0.75, grassVariation));

    // Rock on steep slopes.
    float rockBlend = 1.0 - smoothstep(0.55, 0.82, slopeFactor);
    ground = mix(ground, rock, rockBlend);

    // Terrain hydrology feature: eroded flow channels read as exposed sediment -- rivers carve
    // visible dirt/rock runs down the mountainsides even where the coarse mesh can't show the
    // carve itself (hydro.b is the simulation's flow-speed-where-wet signal).
    float flowChannel = clamp(hydro.b * 0.8, 0.0, 1.0);
    ground = mix(ground, sediment, flowChannel);

    // Beach band: fades sand in near kWaterLevel on shallow slopes, fades back out on steep cliff
    // faces (a cliff plunging into the water shouldn't show a sand strip) and away from the shore.
    float beachFactor = (1.0 - smoothstep(0.0, kBeachBandWidth, worldPos.y - kWaterLevel))
                       * smoothstep(0.3, 0.6, slopeFactor);
    ground = mix(ground, sand, beachFactor);

    // Submerged: the seabed fades from wet sand (shallows) toward dark sediment (depth) --
    // WaterForward.frag's Beer-Lambert absorption then tints the whole thing blue on top.
    float submerged = smoothstep(0.0, -1.0, worldPos.y - kWaterLevel);
    ground = mix(ground, mix(sand * 0.8, sediment * 0.5, clamp(hydro.g * 0.8, 0.0, 1.0)), submerged);

    // Snow: high altitude + shallow slope only. The continental relief's mountain ring peaks near
    // worldY ~ +7 (see terrain_hydrology_params.glsl's kHydroMountainAmplitude), so snow starts
    // roughly halfway up the peaks.
    float snowBlend = smoothstep(2.6, 4.2, worldPos.y) * smoothstep(0.45, 0.75, slopeFactor);
    ground = mix(ground, snow, snowBlend);

    // Terrain hydrology feature: moisture darkening -- damp ground near any water (hydro.a is the
    // blurred water/flow mask) reads darker/saturated, selling the waterline without any texture
    // authoring. Applied last so every band above darkens consistently; snow is exempted (wet
    // snow reads as dirt, not damp ground).
    float dampness = clamp(hydro.a, 0.0, 1.0) * (1.0 - snowBlend) * (1.0 - submerged);
    ground *= (1.0 - 0.4 * dampness);

    return ground;
}

#endif // TERRAIN_SHADING_GLSL
