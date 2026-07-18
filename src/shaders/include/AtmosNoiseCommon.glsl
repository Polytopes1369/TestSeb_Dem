#ifndef ATMOS_NOISE_COMMON_GLSL
#define ATMOS_NOISE_COMMON_GLSL

// Atmos weather system, Subtask 1: Climatic State Manager & Wind Simulation (see
// atmos_integration_plan.md, project root). Pure-function include, no bindings/descriptor sets --
// same "consumer decides the binding" convention shadow_sun_sampling.glsl already establishes,
// applied here to mean NO binding at all (every input this file needs -- wind direction/speed,
// turbulence params, world position, time -- is passed explicitly by the caller, sourced from
// whichever descriptor set that caller's own pass binds AtmosGlobalsUBO into). A future consumer
// (Froxel Volumetric Fog / Volumetric Clouds, Subtasks 3-4) `#include`s this file and its own
// AtmosGlobalsUBO declaration side by side.
//
// GLSL mirror of AtmosClimatePass.cpp's (C++, anonymous-namespace) AtmosGlobalsUBO struct -- kept
// here, not compiled into anything itself, purely as the single documented reference layout any
// future consumer's own `uniform AtmosGlobalsUBO { AtmosGlobals atmos; }` declaration must match
// field-for-field (std140, 64 bytes total). Not declared as an actual named GLSL struct assigned to
// a variable anywhere in this file, to avoid an unused-type compiler warning on backends that warn
// on that; consumers are expected to copy this field list into their own uniform block, exactly like
// shadow_sun_sampling.glsl's own ShadowSunLevelsUBO documents its own SET/BINDING macro contract
// instead of assuming one.
//
// struct AtmosGlobals {                    // std140 offsets:
//     vec3  windDirection; float windSpeed;         //  0..15
//     float temperature, humidity, dewPoint, condensationLCL; // 16..31
//     float cloudDensityTarget, fogDensityTarget, rainStrength, time; // 32..47
//     vec4  windTurbulenceParams; // (frequency, octaves, scale, roughness)  // 48..63
// };

// --- Hash / value noise ---------------------------------------------------------------------

// Cheap, deterministic 3D->3D hash (no texture lookups -- this codebase's noise stays fully
// procedural per CLAUDE.md's "no data in the .exe" constraint). Not cryptographic; only needs to
// look decorrelated enough for wind turbulence.
vec3 AtmosHash33(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Trilinearly-interpolated 3D value noise in [-1, 1], built from AtmosHash33's per-lattice-corner
// pseudo-random scalar (the corner hash's own .x channel) -- the standard, cheapest "smooth 3D
// noise" building block, sufficient for a curl-noise potential field (Subtask 1 objective #3 allows
// "a 3-tap 3D Perlin noise" as an explicit simpler alternative to analytic simplex derivatives).
float AtmosValueNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f); // Smootherstep interpolant.

    float n000 = AtmosHash33(i + vec3(0.0, 0.0, 0.0)).x;
    float n100 = AtmosHash33(i + vec3(1.0, 0.0, 0.0)).x;
    float n010 = AtmosHash33(i + vec3(0.0, 1.0, 0.0)).x;
    float n110 = AtmosHash33(i + vec3(1.0, 1.0, 0.0)).x;
    float n001 = AtmosHash33(i + vec3(0.0, 0.0, 1.0)).x;
    float n101 = AtmosHash33(i + vec3(1.0, 0.0, 1.0)).x;
    float n011 = AtmosHash33(i + vec3(0.0, 1.0, 1.0)).x;
    float n111 = AtmosHash33(i + vec3(1.0, 1.0, 1.0)).x;

    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

// --- Curl noise ------------------------------------------------------------------------------

// Divergence-free turbulent velocity field: the curl of a vector potential Psi = (n0, n1, n2)
// (three independent AtmosValueNoise3D fields, offset from one another so they decorrelate),
// evaluated via central finite differences (the "3-tap" scheme Subtask 1 objective #3 explicitly
// allows in place of analytic simplex derivatives). `eps` is the finite-difference step in the SAME
// world-unit space `p` is already in.
vec3 AtmosCurlNoise3D(vec3 p, float eps) {
    const vec3 dx = vec3(eps, 0.0, 0.0);
    const vec3 dy = vec3(0.0, eps, 0.0);
    const vec3 dz = vec3(0.0, 0.0, eps);

    // Psi.y potential, offset so it is not literally the same field as Psi.z/Psi.x.
    float psiY_dz = AtmosValueNoise3D(p + dz + vec3(17.3, 0.0, 0.0));
    float psiY_dz2 = AtmosValueNoise3D(p - dz + vec3(17.3, 0.0, 0.0));
    float psiZ_dy = AtmosValueNoise3D(p + dy + vec3(0.0, 0.0, 41.9));
    float psiZ_dy2 = AtmosValueNoise3D(p - dy + vec3(0.0, 0.0, 41.9));
    float curlX = (psiZ_dy - psiZ_dy2) - (psiY_dz - psiY_dz2);

    float psiZ_dx = AtmosValueNoise3D(p + dx + vec3(0.0, 0.0, 41.9));
    float psiZ_dx2 = AtmosValueNoise3D(p - dx + vec3(0.0, 0.0, 41.9));
    float psiX_dz = AtmosValueNoise3D(p + dz + vec3(89.1, 0.0, 0.0) + vec3(0.0, 53.7, 0.0));
    float psiX_dz2 = AtmosValueNoise3D(p - dz + vec3(89.1, 0.0, 0.0) + vec3(0.0, 53.7, 0.0));
    float curlY = (psiX_dz - psiX_dz2) - (psiZ_dx - psiZ_dx2);

    float psiY_dx = AtmosValueNoise3D(p + dx + vec3(17.3, 0.0, 0.0));
    float psiY_dx2 = AtmosValueNoise3D(p - dx + vec3(17.3, 0.0, 0.0));
    float psiX_dy = AtmosValueNoise3D(p + dy + vec3(89.1, 0.0, 0.0) + vec3(0.0, 53.7, 0.0));
    float psiX_dy2 = AtmosValueNoise3D(p - dy + vec3(89.1, 0.0, 0.0) + vec3(0.0, 53.7, 0.0));
    float curlZ = (psiY_dx - psiY_dx2) - (psiX_dy - psiX_dy2);

    return vec3(curlX, curlY, curlZ) / (2.0 * eps);
}

// Fractal (multi-octave) curl noise: each octave doubles frequency and is weighted by
// `roughness^octaveIndex` (standard fBm persistence), matching AtmosGlobals.windTurbulenceParams'
// (frequency, octaves, scale, roughness) fields. `octaveCountF` is a float (mirrors the UBO's own
// float storage, see AtmosClimatePass.cpp) truncated once here.
vec3 AtmosFractalCurlNoise3D(vec3 p, float baseFrequency, float octaveCountF, float roughness) {
    uint octaveCount = uint(max(octaveCountF, 1.0));
    vec3 sum = vec3(0.0);
    float amplitude = 1.0;
    float frequency = baseFrequency;
    float amplitudeSum = 0.0;
    for (uint o = 0u; o < octaveCount; ++o) {
        sum += AtmosCurlNoise3D(p * frequency, 0.5) * amplitude;
        amplitudeSum += amplitude;
        amplitude *= roughness;
        frequency *= 2.0;
    }
    return amplitudeSum > 0.0 ? sum / amplitudeSum : vec3(0.0);
}

// --- Public entry point ------------------------------------------------------------------------

// Combines the base wind vector (`windDirection`, world-space unit vector, `windSpeed` in m/s)
// with fractal curl-noise turbulence, for a caller sampling AtmosGlobalsUBO's own fields (see this
// file's header comment for the exact field list). The turbulence domain is scrolled by
// `windDirection * windSpeed * time` so eddies visibly advect downwind instead of just churning in
// place, matching Subtask 1 objective #3 ("combining base wind direction, wind speed, and the Curl
// noise to animate weather features"). `turbulenceParams` = (frequency, octaves, scale, roughness),
// i.e. AtmosGlobals.windTurbulenceParams verbatim.
vec3 SampleWindVelocity(vec3 worldPos, float time, vec3 windDirection, float windSpeed, vec4 turbulenceParams) {
    vec3 baseWind = windDirection * windSpeed;
    vec3 scrolledPos = worldPos - baseWind * time;
    vec3 turbulence = AtmosFractalCurlNoise3D(scrolledPos, turbulenceParams.x, turbulenceParams.y, turbulenceParams.w);
    return baseWind + turbulence * turbulenceParams.z * windSpeed;
}

#endif // ATMOS_NOISE_COMMON_GLSL
