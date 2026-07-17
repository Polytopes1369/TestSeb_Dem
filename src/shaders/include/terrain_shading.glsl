#ifndef TERRAIN_SHADING_GLSL
#define TERRAIN_SHADING_GLSL

// Phase 7b (UE5.8 parity roadmap, terrain heightfield): procedural height/slope biome blending for
// the terrain material. A single MATERIAL_ID_TERRAIN recipe (MaterialParameterTable.h) supplies a
// flat fallback baseColor; this include overrides it per-pixel using the already-interpolated
// worldPos/normal available at ClusterResolve.comp/ClusterResolveBinned.comp's material-evaluation
// step -- no new binding, no new MaterialParameters field (kept as dedicated GLSL constants, same
// "private constant, single-entity scope" convention HeroTessellationPass established for
// displacementScale in Phase 7a).
#include "terrain_noise.glsl"
#include "water_params.glsl" // Phase 7c: kWaterLevel, for the beach band below.

// Mirrors VulkanContext::GenerateGeometry()'s terrain call site (worldOffsetY), which reuses the
// same kFloorTopY anchor the flat floor plane used before this entity replaced it -- keep in sync.
const float kTerrainBaseWorldY = -0.8;

// Phase 7c (UE5.8 parity roadmap, water/erosion): world-unit width of the sand band above
// kWaterLevel -- same technique as the existing grass/rock/snow bands, just a 4th smoothstep band,
// no new geometry/binding. Scaled down from the old branch's own 0.35 (tuned for a 2.2-amplitude
// terrain) proportionally to THIS terrain's much smaller 0.4 amplitude (0.35 * 0.4/2.2 =~ 0.064),
// so the beach doesn't visually swallow this terrain's own much shorter above-water height range.
const float kBeachBandWidth = 0.06;

vec3 ComputeTerrainAlbedo(vec3 worldPos, vec3 normal, vec3 fallbackAlbedo) {
    float heightFactor = clamp(
        (worldPos.y - kTerrainBaseWorldY + kTerrainAmplitude) / (2.0 * kTerrainAmplitude), 0.0, 1.0);
    float slopeFactor = clamp(normal.y, 0.0, 1.0); // 1 = flat, 0 = vertical cliff

    vec3 grass = vec3(0.24, 0.42, 0.15);
    vec3 rock  = vec3(0.40, 0.36, 0.32);
    vec3 snow  = vec3(0.90, 0.92, 0.96);
    vec3 sand  = vec3(0.76, 0.70, 0.50);

    float rockBlend = 1.0 - smoothstep(0.55, 0.82, slopeFactor);
    vec3 ground = mix(grass, rock, rockBlend);

    // Beach band: fades sand in near kWaterLevel on shallow slopes, fades back out on steep cliff
    // faces (a cliff plunging into the water shouldn't show a sand strip) and away from the shore.
    float beachFactor = (1.0 - smoothstep(0.0, kBeachBandWidth, worldPos.y - kWaterLevel))
                       * smoothstep(0.3, 0.6, slopeFactor);
    ground = mix(ground, sand, beachFactor);

    float snowBlend = smoothstep(0.62, 0.85, heightFactor) * smoothstep(0.45, 0.75, slopeFactor);
    return mix(ground, snow, snowBlend);
}

#endif // TERRAIN_SHADING_GLSL
