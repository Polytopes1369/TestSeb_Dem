#ifndef TEMPORAL_VARIANCE_CLAMP_GLSL
#define TEMPORAL_VARIANCE_CLAMP_GLSL

// Phase 2 (Lumen advanced roadmap): shared temporal-accumulation color-space + variance-clamp math,
// factored out of TAATSR.comp's own original inline copy so ReflectionTemporal.comp (reflection
// temporal stability, see renderer::ReflectionPass' own class comment) can reuse the EXACT same
// technique instead of a second hand-typed copy -- same "pure stateless math -> shared include"
// convention as include/octahedral.glsl / include/ggx_brdf.glsl (see those files' own header
// comments for the same rationale).
//
//   - RGBToYCoCg/YCoCgToRGB: luma/chroma color-space round trip -- variance clamping in YCoCg
//     (rather than raw RGB) keeps the box's 3 axes closer to perceptually independent, avoiding the
//     hue-shift artifacts a per-RGB-channel clamp produces.
//   - ClipToAABB: Karis' "clip towards box center" variant (SIGGRAPH 2014, "High Quality Temporal
//     Supersampling") rather than a hard per-channel clamp -- moves the rejected color along the ray
//     from the box center to its original position until it lands exactly on the box surface,
//     preserving hue/saturation far better than an independent per-channel clamp would.
//
// Both callers build the [boxMin, boxMax] AABB themselves from their own local pixel neighborhood
// (a 3x3 mean/variance box in every current caller) -- this file only provides the shared color-space
// conversion and the clip operation itself, not the neighborhood-gathering loop.

vec3 RGBToYCoCg(vec3 rgb) {
    float y = dot(rgb, vec3(0.25, 0.5, 0.25));
    float co = dot(rgb, vec3(0.5, 0.0, -0.5));
    float cg = dot(rgb, vec3(-0.25, 0.5, -0.25));
    return vec3(y, co, cg);
}

vec3 YCoCgToRGB(vec3 ycocg) {
    float y = ycocg.x;
    float co = ycocg.y;
    float cg = ycocg.z;
    float r = y + co - cg;
    float g = y + cg;
    float b = y - co - cg;
    return vec3(r, g, b);
}

// Clips `color` (already in YCoCg space) toward the box center until it lands on the surface of the
// [boxMin, boxMax] AABB -- see this file's own header comment for why "clip" rather than a per-
// channel clamp. The +0.0001 epsilon on `extents` avoids a divide-by-zero when a fully degenerate
// (single-color) neighborhood collapses boxMin == boxMax on some axis.
vec3 ClipToAABB(vec3 color, vec3 boxMin, vec3 boxMax) {
    vec3 center = 0.5 * (boxMax + boxMin);
    vec3 extents = 0.5 * (boxMax - boxMin) + 0.0001;

    vec3 dir = color - center;
    vec3 absDir = abs(dir);

    float max_t = 0.0;
    if (absDir.x > extents.x) max_t = max(max_t, (absDir.x - extents.x) / absDir.x);
    if (absDir.y > extents.y) max_t = max(max_t, (absDir.y - extents.y) / absDir.y);
    if (absDir.z > extents.z) max_t = max(max_t, (absDir.z - extents.z) / absDir.z);

    return color - max_t * dir;
}

#endif // TEMPORAL_VARIANCE_CLAMP_GLSL
