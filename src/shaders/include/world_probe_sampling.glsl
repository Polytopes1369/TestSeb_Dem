#ifndef WORLD_PROBE_SAMPLING_GLSL
#define WORLD_PROBE_SAMPLING_GLSL

// Shared "sample the World Probe grid at any world position" primitive -- mirrors
// surface_cache_sampling.glsl's own doc-comment convention. renderer::WorldProbeGridPass fills
// the grid (WorldProbeInject.comp); this header is what any OTHER shader includes to read it back
// -- renderer::ScreenTracePass's own ScreenTrace.comp (as its far-field fallback whenever the
// screen-space march misses) today, and, per the original ask ("allow the shaders of these dynamic
// objects to sample this grid"), a future particle/character shader tomorrow: neither
// needs to know anything about how the grid was built, only this one function.
//
// Binding convention every including shader must reserve -- the exact set/binding indices are
// deliberately left to the including shader to `#define` before including this file (unlike
// surface_cache_sampling.glsl's fixed "always set 2" convention), since this header's only two
// consumers so far (ScreenTrace.comp, and any future dynamic-object shader) do not otherwise share
// a common descriptor set layout the way every Surface Cache trace consumer already does via
// renderer::SurfaceCacheTraceContext -- define WORLD_PROBE_GRID_SET / WORLD_PROBE_GRID_BINDING /
// WORLD_PROBE_GRID_PARAMS_BINDING before #include-ing this file.
#ifndef WORLD_PROBE_GRID_SET
#error "world_probe_sampling.glsl: define WORLD_PROBE_GRID_SET before including this file."
#endif
#ifndef WORLD_PROBE_GRID_BINDING
#error "world_probe_sampling.glsl: define WORLD_PROBE_GRID_BINDING before including this file."
#endif
#ifndef WORLD_PROBE_GRID_PARAMS_BINDING
#error "world_probe_sampling.glsl: define WORLD_PROBE_GRID_PARAMS_BINDING before including this file."
#endif

layout(set = WORLD_PROBE_GRID_SET, binding = WORLD_PROBE_GRID_BINDING) uniform sampler3D g_WorldProbeGrid;

// Byte-for-byte mirror of renderer::ScreenTracePass.cpp's WorldProbeGridParams (std140): the grid
// origin (texel (0,0,0)'s world-space corner, NOT its center -- see
// renderer::WorldProbeGridPass::GetGridOriginWorld()'s own comment) + spacing + resolution, all a
// consuming shader needs to convert a world position into the grid's own [0,1]^3 UVW.
layout(std140, set = WORLD_PROBE_GRID_SET, binding = WORLD_PROBE_GRID_PARAMS_BINDING) uniform WorldProbeGridParamsUBO {
    vec3 gridOrigin;
    float probeSpacing;
    float gridResolution; // Float, not uint -- avoids an implicit int->float conversion at every call site below.
} g_WorldProbeGridParams;

// Trilinear sample (hardware-filtered, via g_WorldProbeGrid's own linear+CLAMP_TO_EDGE sampler --
// see renderer::WorldProbeGridPass::Init's sampler comment) of the ambient irradiance the grid
// holds at `worldPos`. Positions outside the grid's own covered volume clamp to the nearest edge
// probe layer rather than wrapping or reading black, matching CLAMP_TO_EDGE's own semantics -- an
// acceptable approximation for a low-frequency ambient volume this far from any single probe.
vec3 SampleWorldProbeGrid(vec3 worldPos) {
    vec3 gridExtent = vec3(g_WorldProbeGridParams.gridResolution * g_WorldProbeGridParams.probeSpacing);
    vec3 uvw = (worldPos - g_WorldProbeGridParams.gridOrigin) / gridExtent;
    return texture(g_WorldProbeGrid, uvw).rgb;
}

#endif
