#ifndef ATMOS_SKY_LUT_MAPPING_GLSL
#define ATMOS_SKY_LUT_MAPPING_GLSL

// Atmos weather system, Subtask 2 (atmos_integration_plan.md): the UV<->direction mapping shared,
// byte-for-byte identical, by AtmosSkyLUTs.comp's ComputeSkyViewLUT() (the WRITER -- for each output
// texel, which world-space ray direction to raymarch) and every consumer of the resulting Sky-View
// LUT (PostProcessComposite.comp, SDFRayMarch.comp -- the READERS -- for a given ray direction,
// which texel to sample). The two functions below are exact inverses of each other; if either changes,
// the other MUST change identically or every consumer's sky sample silently points at the wrong texel.
//
// Horizon-detail-favoring elevation mapping + sun-relative azimuth (Sebastien Hillaire, "A Scalable
// and Production Ready Sky and Atmosphere Rendering Technique", 2020) -- allocates more of the V axis
// near the horizon (where the sky's radiance gradient is steepest -- sunset/sunrise bands) than
// straight up/down (where radiance varies slowly), and measures azimuth relative to the CURRENT sun
// direction (not a fixed world axis) so a single 200x200 LUT stays valid as the sun rotates without
// needing a 3rd LUT axis.

const float ATMOS_PI = 3.14159265358979323846;

// World-space unit ray direction -> Sky-View LUT UV, given the current sun direction (both already
// unit vectors; `sunDir` points FROM the light TOWARD the scene, so the sun itself sits along
// `-sunDir` -- same convention as renderer::ClusterResolvePass's own sunDirection parameter, negated
// once below so "azimuth 0" faces the actual sun position, not away from it).
vec2 SkyViewLUTUVFromDirection(vec3 rayDir, vec3 sunDir) {
    vec3 sunPos = -sunDir;

    float sinElevation = clamp(rayDir.y, -1.0, 1.0);
    float elevation = asin(sinElevation);
    float vNorm = elevation / (ATMOS_PI * 0.5); // [-1, 1]
    float v = 0.5 + 0.5 * sign(vNorm) * sqrt(abs(vNorm));

    vec2 rayHoriz = vec2(rayDir.x, rayDir.z);
    vec2 sunHoriz = vec2(sunPos.x, sunPos.z);
    float rayHorizLen = length(rayHoriz);
    float sunHorizLen = length(sunHoriz);
    // Degenerate case (ray or sun pointing straight up/down -- azimuth undefined): pick azimuth 0,
    // matching the inverse mapping's own fallback below so a round-trip stays exact.
    vec2 rayHorizN = (rayHorizLen > 1.0e-5) ? (rayHoriz / rayHorizLen) : vec2(1.0, 0.0);
    vec2 sunHorizN = (sunHorizLen > 1.0e-5) ? (sunHoriz / sunHorizLen) : vec2(1.0, 0.0);

    float cosAz = clamp(dot(rayHorizN, sunHorizN), -1.0, 1.0);
    float sinAz = rayHorizN.x * sunHorizN.y - rayHorizN.y * sunHorizN.x; // 2D cross product.
    float azimuth = atan(sinAz, cosAz); // [-PI, PI]
    float u = azimuth / (2.0 * ATMOS_PI) + 0.5;

    return vec2(u, clamp(v, 0.0, 1.0));
}

// Inverse of SkyViewLUTUVFromDirection: Sky-View LUT UV + current sun direction -> world-space unit
// ray direction. Used only by AtmosSkyLUTs.comp's ComputeSkyViewLUT() to decide which direction to
// raymarch for a given output texel.
vec3 SkyViewLUTDirectionFromUV(vec2 uv, vec3 sunDir) {
    vec3 sunPos = -sunDir;

    float vNorm = (uv.y - 0.5) * 2.0; // [-1, 1]
    float elevation = sign(vNorm) * vNorm * vNorm * (ATMOS_PI * 0.5);
    float sinElevation = sin(elevation);
    float cosElevation = cos(elevation);

    float azimuth = (uv.x - 0.5) * 2.0 * ATMOS_PI;

    vec2 sunHoriz = vec2(sunPos.x, sunPos.z);
    float sunHorizLen = length(sunHoriz);
    vec2 sunHorizN = (sunHorizLen > 1.0e-5) ? (sunHoriz / sunHorizLen) : vec2(1.0, 0.0);

    float cosAzimuth = cos(azimuth);
    float sinAzimuth = sin(azimuth);
    // Rotate sunHorizN by `azimuth` (2D rotation matrix) -- the exact inverse of the atan2(cross, dot)
    // used to extract azimuth in the forward mapping above.
    vec2 rayHoriz = vec2(
        sunHorizN.x * cosAzimuth - sunHorizN.y * sinAzimuth,
        sunHorizN.x * sinAzimuth + sunHorizN.y * cosAzimuth
    );

    return normalize(vec3(rayHoriz.x * cosElevation, sinElevation, rayHoriz.y * cosElevation));
}

#endif // ATMOS_SKY_LUT_MAPPING_GLSL
