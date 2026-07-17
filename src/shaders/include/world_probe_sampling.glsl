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

// Phase 6 (UE5.8 parity roadmap): manual 8-corner trilinear blend via texelFetch against the
// grid's TOROIDAL (wrapped) addressing, replacing the pre-Phase-6 hardware-trilinear `texture()`
// read above (renderer::WorldProbeGridPass::RecordUpdate() used to fully rebuild the grid every
// call, so texel (0,0,0) always physically WAS world position g_WorldProbeGridParams.gridOrigin --
// straightforward hardware filtering was correct). Now that the grid streams incrementally (see
// that class' own header comment), a texel's physical slot holds whichever world-probe-index most
// recently wrapped to it -- hardware trilinear filtering between two ADJACENT physical texels
// would incorrectly blend across the wrap seam whenever those two texels' CONTENTS represent
// world-probe-indices that are nowhere near each other spatially. The exact same reasoning
// renderer::SDFRayMarchPass's own clipmap sampling already relies on for renderer::GlobalSDFPass
// (see that shader's own SampleClipmap comment) -- this function is that same technique's
// trilinear-interpolated (rather than nearest-only) counterpart, needed here because a sparse
// probe grid (unlike a multi-level SDF clipmap cascade) has no coarser level to fall back on for
// smoothness between adjacent probes.
//
// Addressing note: the absolute world-probe-index (`worldPos / probeSpacing`, NOT relative to
// g_WorldProbeGridParams.gridOrigin) mod gridResolution is the correct physical texel for any
// given world position, regardless of where the covered window currently sits -- because
// renderer::WorldProbeGridPass::RecordSlab's own write side (SplitWrappedRange) computes each
// probe's physical texel the exact same way (absolute world-probe-index mod resolution, with no
// dependency on the window's own position). g_WorldProbeGridParams.gridOrigin therefore is not
// needed by the addressing math below at all (kept in the UBO purely for other consumers, e.g.
// renderer::GICompositePass's debug visualization, that still want the window's own world-space
// extent). No bounds/window-membership check is performed here -- a caller sampling a position far
// outside the currently-covered window reads whatever stale data physically occupies that wrapped
// texel, the same accepted simplification renderer::SDFRayMarchPass's own SampleClipmap relies on
// its caller (SelectLevel) to have already ruled out.
vec3 SampleWorldProbeGrid(vec3 worldPos) {
    int resolution = int(g_WorldProbeGridParams.gridResolution);
    float spacing = g_WorldProbeGridParams.probeSpacing;

    // Continuous world-probe-index space: probe i's own sample sits at world position
    // (i + 0.5) * spacing (see WorldProbeInject.comp's own +0.5 centering) -- subtracting 0.5 here
    // lands exactly on the lower-corner probe index for the 8-corner trilinear blend below.
    vec3 continuousIndex = worldPos / spacing - vec3(0.5);
    ivec3 baseIndex = ivec3(floor(continuousIndex));
    vec3 frac = continuousIndex - vec3(baseIndex);

    vec3 corner[8];
    for (int i = 0; i < 8; ++i) {
        ivec3 offset = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 worldIndex = baseIndex + offset;
        ivec3 wrapped = ((worldIndex % resolution) + resolution) % resolution;
        corner[i] = texelFetch(g_WorldProbeGrid, wrapped, 0).rgb;
    }

    vec3 c00 = mix(corner[0], corner[1], frac.x); // x=0/1, y=0, z=0
    vec3 c10 = mix(corner[2], corner[3], frac.x); // x=0/1, y=1, z=0
    vec3 c01 = mix(corner[4], corner[5], frac.x); // x=0/1, y=0, z=1
    vec3 c11 = mix(corner[6], corner[7], frac.x); // x=0/1, y=1, z=1
    vec3 c0 = mix(c00, c10, frac.y);
    vec3 c1 = mix(c01, c11, frac.y);
    return mix(c0, c1, frac.z);
}

#endif
