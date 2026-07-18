#ifndef ATMOS_CLOUDS_DENSITY_GLSL
#define ATMOS_CLOUDS_DENSITY_GLSL

// Atmos weather system, Subtasks 4/5 (atmos_integration_plan.md): the cloud density field itself --
// shared, byte-for-byte identical, by AtmosClouds.comp (the camera-view raymarcher, Subtask 4) and
// AtmosCloudShadows.comp (the sun-facing shadow-map raymarcher, Subtask 5). Both need to agree
// EXACTLY on where the cloud volume is dense, or a cloud casting a shadow would not match the cloud
// actually visible from the camera. Originally implemented inline in AtmosClouds.comp only (Subtask
// 4); extracted here once a second consumer needed the identical logic -- this codebase's own
// convention (see e.g. atmos_sky_lut_mapping.glsl's own header comment) is to keep small phase/trace
// functions duplicated per-shader, but promote genuinely-must-agree logic like this one to a shared
// include instead.
//
// The includer must pass its own bound shape/detail noise samplers and AtmosGlobalsUBO fields as
// plain function parameters -- no fixed binding/set assumed here, same "consumer decides the
// binding" convention as every other shared .glsl include in this codebase.

// --- Planet-local frame -- deliberately SHARED with AtmosSkyLUTs.comp's own constants (same
// kGroundRadiusKm, same fixed camera altitude) so the cloud layer and the physical sky agree on
// what "the ground" and "the camera's height in the atmosphere" mean. ---
const float ATMOS_CLOUDS_GROUND_RADIUS_KM = 6360.0;
const float ATMOS_CLOUDS_CAMERA_HEIGHT_KM = 0.5;
const float ATMOS_CLOUDS_INNER_RADIUS_KM = ATMOS_CLOUDS_GROUND_RADIUS_KM + 1.5;
const float ATMOS_CLOUDS_OUTER_RADIUS_KM = ATMOS_CLOUDS_GROUND_RADIUS_KM + 4.5;

// Must match renderer::AtmosCloudsPass::kShapeNoiseResolution/kDetailNoiseResolution exactly.
const int ATMOS_CLOUDS_SHAPE_RES = 128;
const int ATMOS_CLOUDS_DETAIL_RES = 32;

bool AtmosCloudsRaySphereIntersect(vec3 ro, vec3 rd, float radius, out float t0, out float t1) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        t0 = t1 = -1.0;
        return false;
    }
    float sq = sqrt(discriminant);
    t0 = -b - sq;
    t1 = -b + sq;
    return t1 >= 0.0;
}

float AtmosCloudsRemap(float value, float low1, float high1, float low2, float high2) {
    return low2 + (value - low1) * (high2 - low2) / max(high1 - low1, 1.0e-5);
}

// --- Tiling noise primitives (both wrap via mod(cell, freq), so `freq` MUST be an integer number
// of cells for a seamless repeat -- callers only ever pass integer literals/multiples). ---

vec3 AtmosCloudHash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453123);
}

float AtmosCloudsTilingValueNoise3D(vec3 p, float freq) {
    vec3 scaled = p * freq;
    vec3 i = floor(scaled);
    vec3 f = fract(scaled);
    vec3 u = f * f * (3.0 - 2.0 * f);

    float n000 = AtmosCloudHash33(mod(i + vec3(0, 0, 0), freq)).x;
    float n100 = AtmosCloudHash33(mod(i + vec3(1, 0, 0), freq)).x;
    float n010 = AtmosCloudHash33(mod(i + vec3(0, 1, 0), freq)).x;
    float n110 = AtmosCloudHash33(mod(i + vec3(1, 1, 0), freq)).x;
    float n001 = AtmosCloudHash33(mod(i + vec3(0, 0, 1), freq)).x;
    float n101 = AtmosCloudHash33(mod(i + vec3(1, 0, 1), freq)).x;
    float n011 = AtmosCloudHash33(mod(i + vec3(0, 1, 1), freq)).x;
    float n111 = AtmosCloudHash33(mod(i + vec3(1, 1, 1), freq)).x;

    float nx00 = mix(n000, n100, u.x), nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x), nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

float AtmosCloudsTilingWorleyNoise3D(vec3 p, float freq) {
    vec3 scaled = p * freq;
    vec3 baseCell = floor(scaled);
    vec3 f = fract(scaled);

    float minDistSq = 1.0e6;
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                vec3 offset = vec3(x, y, z);
                vec3 cell = mod(baseCell + offset, freq);
                vec3 featurePoint = offset + AtmosCloudHash33(cell) - f;
                minDistSq = min(minDistSq, dot(featurePoint, featurePoint));
            }
        }
    }
    return clamp(sqrt(minDistSq), 0.0, 1.0);
}

float AtmosCloudsPerlinFbm3D(vec3 p, float baseFreq) {
    float sum = 0.0;
    sum += AtmosCloudsTilingValueNoise3D(p, baseFreq) * 0.571;
    sum += AtmosCloudsTilingValueNoise3D(p, baseFreq * 2.0) * 0.286;
    sum += AtmosCloudsTilingValueNoise3D(p, baseFreq * 4.0) * 0.143;
    return sum;
}

float AtmosCloudsWorleyFbm3D(vec3 p, float baseFreq) {
    float sum = 0.0;
    sum += AtmosCloudsTilingWorleyNoise3D(p, baseFreq) * 0.625;
    sum += AtmosCloudsTilingWorleyNoise3D(p, baseFreq * 2.0) * 0.25;
    sum += AtmosCloudsTilingWorleyNoise3D(p, baseFreq * 4.0) * 0.125;
    return sum;
}

// World-km -> repeating noise-UVW scale: chosen so the shape texture repeats roughly every 6km of
// world space (detail every 1.5km, 4x finer) -- tunable, not derived from any physical constant.
const float ATMOS_CLOUDS_SHAPE_UV_SCALE = 1.0 / 6.0;
const float ATMOS_CLOUDS_DETAIL_UV_SCALE = 1.0 / 1.5;
const float ATMOS_CLOUDS_WIND_SCROLL_SCALE = 0.02; // km per (world wind-speed unit x time second).
const float ATMOS_CLOUDS_EROSION_STRENGTH = 0.4;
const float ATMOS_CLOUDS_EXTINCTION_SCALE = 1.2; // Per km, at density == 1.

// Atmos weather system, Subtask 5: world XZ -> AtmosCloudShadows.comp's own shadow map UV.
//
// Deliberately in the DEMO's own actual world-space units (this scene's real ~tens-of-units scale,
// matching how AtmosVolumetricFogPass's own ATMOS_FOG_NEAR_Z/FAR_Z already treat "world units" as
// this demo's real coordinate scale -- see atmos_volumetric_fog_mapping.glsl's own header comment),
// NOT the km-scale planet-local frame the rest of this file's constants use for the camera-facing
// sky/cloud rendering. Those two systems are already decoupled by design (AtmosClouds.comp's own
// RaymarchClouds() never reads the demo's real camera XZ position either -- see that function's own
// `ro` -- clouds render as a view-direction-only "sky dome", not a positioned volume in demo space),
// so a cloud shadow that visually correlates with the demo's own actual geometry positions (this
// consumer's whole purpose -- shadows fall ON real objects) necessarily uses a different, smaller-
// scale extent than the km-scale sky/cloud rendering above. AtmosCloudShadows.comp's own raymarch
// still calls the SAME SampleCloudDensity() below for genuine procedural consistency/code reuse; it
// just feeds it demo-world-unit XZ instead of km XZ, which -- since ATMOS_CLOUDS_SHAPE_UV_SCALE was
// already a tunable, not physically-derived constant (see its own comment) -- simply means the noise
// tiles at a scale appropriate to this demo's own object sizes instead of a real-world kilometer
// scale, which is the actually useful behavior for a small-scale demoscene.
const float ATMOS_CLOUD_SHADOW_HALF_EXTENT = 32.0; // World units, NOT km.

vec2 AtmosCloudShadowUVFromWorldXZ(vec2 worldXZ) {
    return clamp(worldXZ / (2.0 * ATMOS_CLOUD_SHADOW_HALF_EXTENT) + 0.5, 0.0, 1.0);
}

// `pos` is in the planet-local frame (world km, origin at planet center -- see the constants above).
float SampleCloudDensity(sampler3D shapeSampler, sampler3D detailSampler, vec3 pos,
    vec3 windDirection, float windSpeed, float time, float cloudDensityTarget, float humidity) {
    float radius = length(pos);
    float heightFrac = clamp((radius - ATMOS_CLOUDS_INNER_RADIUS_KM) / (ATMOS_CLOUDS_OUTER_RADIUS_KM - ATMOS_CLOUDS_INNER_RADIUS_KM), 0.0, 1.0);
    // Anvil-style vertical density gradient: thin at the base, full-bodied through the middle,
    // tapering again near the top.
    float heightGradient = smoothstep(0.0, 0.2, heightFrac) * (1.0 - smoothstep(0.6, 1.0, heightFrac));
    if (heightGradient <= 0.0) {
        return 0.0;
    }

    vec3 windOffset = windDirection * windSpeed * time * ATMOS_CLOUDS_WIND_SCROLL_SCALE;

    // Coverage ties Subtask 1's climate state into cloud volume (objective #2: "higher humidity
    // increases cloud volume/thickness").
    float coverage = clamp(cloudDensityTarget * (0.5 + 0.5 * humidity), 0.0, 1.0);

    vec3 shapeUVW = (pos - windOffset) * ATMOS_CLOUDS_SHAPE_UV_SCALE;
    float shape = texture(shapeSampler, shapeUVW).r;
    float baseDensity = clamp(AtmosCloudsRemap(shape * heightGradient, 1.0 - coverage, 1.0, 0.0, 1.0), 0.0, 1.0);
    if (baseDensity <= 0.0) {
        return 0.0;
    }

    vec3 detailUVW = (pos - windOffset * 1.5) * ATMOS_CLOUDS_DETAIL_UV_SCALE;
    float detail = texture(detailSampler, detailUVW).r;
    float eroded = clamp(AtmosCloudsRemap(baseDensity, detail * ATMOS_CLOUDS_EROSION_STRENGTH, 1.0, 0.0, 1.0), 0.0, 1.0);
    return eroded;
}

#endif // ATMOS_CLOUDS_DENSITY_GLSL
