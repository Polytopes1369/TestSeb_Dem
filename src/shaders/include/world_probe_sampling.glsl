#ifndef WORLD_PROBE_SAMPLING_GLSL
#define WORLD_PROBE_SAMPLING_GLSL

// Shared "sample the World Probe grid at any world position" primitive -- mirrors
// surface_cache_sampling.glsl's own doc-comment convention. renderer::WorldProbeGridPass fills the
// grid (WorldProbeInject.comp); this header is what any OTHER shader includes to read it back --
// renderer::ScreenTracePass's own ScreenTrace.comp (as its far-field fallback in HighQuality GI
// mode, or as the PRIMARY term in F1's Lite mode -- see that shader's own giMode branch), and,
// Debug-only, renderer::GICompositePass's DEBUG_VIEW_SPATIAL_PROBES visualization.
//
// Binding convention every including shader must reserve -- the exact set/binding indices are
// deliberately left to the including shader to `#define` before including this file (unlike
// surface_cache_sampling.glsl's fixed "always set 2" convention), since this header's consumers do
// not otherwise share a common descriptor set layout the way every Surface Cache trace consumer
// already does via renderer::SurfaceCacheTraceContext -- define WORLD_PROBE_GRID_SET /
// WORLD_PROBE_GRID_BINDING / WORLD_PROBE_GRID_PARAMS_BINDING before #include-ing this file.
// WORLD_PROBE_GRID_OCCLUSION_BINDING is OPTIONAL (F1, "Lumen Lite"): define it too to additionally
// reserve one more sampler3D-array binding and get DDGI-style Chebyshev probe-occlusion weighting
// (see ChebyshevVisibility()'s own comment) -- renderer::ScreenTracePass's own ScreenTrace.comp and
// renderer::GICompositePass's Debug-only visualization both define it, since screen-space GI is
// exactly where wall/corner light leaking is most visible. Every OTHER consumer (forward-shaded
// vegetation/fur/particle/translucent materials sampling this grid as a cheap ambient fill --
// TransparentForward.frag, VegetationInstanced.frag, Tessellation.frag, ParticleRender.frag,
// ParticleRibbonRender.frag, ParticleMeshRender.frag, FurStrand.frag) leaves it undefined and gets
// plain (pre-F1-equivalent) trilinear blending with no occlusion test -- a deliberate scope
// reduction: none of those materials' own C++ owning pass currently binds a 7th/8th descriptor for
// this, and an ambient fill on a small dynamic/foliage surface is far less exposed to visible
// wall-leaking than the full-screen GI term is.
#ifndef WORLD_PROBE_GRID_SET
#error "world_probe_sampling.glsl: define WORLD_PROBE_GRID_SET before including this file."
#endif
#ifndef WORLD_PROBE_GRID_BINDING
#error "world_probe_sampling.glsl: define WORLD_PROBE_GRID_BINDING before including this file."
#endif
#ifndef WORLD_PROBE_GRID_PARAMS_BINDING
#error "world_probe_sampling.glsl: define WORLD_PROBE_GRID_PARAMS_BINDING before including this file."
#endif

// F1 ("Lumen Lite", UE5.8 parity roadmap): must match renderer::WorldProbeGridPass::kLevelCount
// exactly -- 3 camera-centered clipmap levels, level L covering 2x the world-space extent of level
// L-1 at the SAME probe density (see that class' own header comment).
#define WORLD_PROBE_LEVEL_COUNT 3

// One combined-image-sampler binding, descriptorCount == WORLD_PROBE_LEVEL_COUNT -- mirrors
// SDFRayMarch.comp's own `uniform sampler3D uGlobalClipmaps[4]` convention for
// renderer::GlobalSDFPass's own multi-level clipmap (see that shader's own comment). Index 0 is
// the FINEST level.
layout(set = WORLD_PROBE_GRID_SET, binding = WORLD_PROBE_GRID_BINDING) uniform sampler3D g_WorldProbeGrid[WORLD_PROBE_LEVEL_COUNT];
#ifdef WORLD_PROBE_GRID_OCCLUSION_BINDING
// F1: per-probe (mean, mean-squared) hit-distance pair, same resolution/addressing/level count as
// g_WorldProbeGrid above -- see renderer::WorldProbeGridPass.h's own "probe occlusion" comment and
// WorldProbeInject.comp's own header comment for how this is populated.
layout(set = WORLD_PROBE_GRID_SET, binding = WORLD_PROBE_GRID_OCCLUSION_BINDING) uniform sampler3D g_WorldProbeOcclusion[WORLD_PROBE_LEVEL_COUNT];
#endif

// Byte-for-byte mirror of renderer::WorldProbeGridPass.h's own WorldProbeGridParams (std140): per
// level, the grid origin (texel (0,0,0)'s world-space corner, NOT its center -- see
// renderer::WorldProbeGridPass::GetGridOriginWorld()'s own comment) + that level's own probe
// spacing, packed vec3+float (one vec4 slot per level, matching std140's array-of-struct stride) --
// plus the single shared probe-count-per-axis (identical at every level).
struct WorldProbeGridLevelParams {
    vec3 origin;
    float spacing;
};
layout(std140, set = WORLD_PROBE_GRID_SET, binding = WORLD_PROBE_GRID_PARAMS_BINDING) uniform WorldProbeGridParamsUBO {
    WorldProbeGridLevelParams levels[WORLD_PROBE_LEVEL_COUNT];
    float gridResolution; // Float, not uint -- avoids an implicit int->float conversion at every call site below.
} g_WorldProbeGridParams;

// DDGI-style (Majercik et al.) one-sided Chebyshev visibility test: `occlusion` is this probe's own
// (mean, mean-squared) hit-distance pair (WorldProbeInject.comp), `distToPoint` is the distance from
// THIS probe to the point being shaded. If the shading point is no farther than the probe's own
// mean hit distance, there is no evidence of an occluder between them -- fully visible. Otherwise, a
// one-sided variance (Chebyshev) bound estimates how likely it is that SOME occluder sits closer
// than `distToPoint`, cubed for a sharper cutoff (the standard DDGI convention -- reduces light
// bleed from a partially-occluded probe far more aggressively than a linear falloff would).
// Deliberately a per-probe AGGREGATE test (see WorldProbeInject.comp's own header comment), not a
// full per-direction visibility field -- still catches the common "probe embedded in/behind a wall"
// leak case this engine's coarse ambient grid cares about.
float ChebyshevVisibility(vec2 occlusion, float distToPoint) {
    float mean = occlusion.x;
    if (distToPoint <= mean) {
        return 1.0;
    }
    float mean2 = occlusion.y;
    float variance = abs(mean2 - mean * mean);
    float d = distToPoint - mean;
    float chebyshev = variance / (variance + d * d);
    chebyshev = clamp(chebyshev, 0.0, 1.0);
    return chebyshev * chebyshev * chebyshev;
}

// 8-corner trilinear blend of ONE clipmap level, via texelFetch against that level's TOROIDAL
// (wrapped) addressing (renderer::WorldProbeGridPass::RecordSlab's own write side, SplitWrappedRange,
// computes each probe's physical texel the exact same way: absolute world-probe-index mod
// resolution, with no dependency on the level's current window at all -- so g_WorldProbeGridParams'
// own per-level `origin` is not needed by the addressing math below, only by other consumers, e.g.
// renderer::GICompositePass's debug visualization, that want the window's own world-space extent).
// Hardware trilinear filtering is deliberately NOT used here (see renderer::WorldProbeGridPass.cpp's
// own sampler comment: it would incorrectly blend across the wrap seam) -- replaced by this manual
// 8-corner blend, additionally weighted by ChebyshevVisibility() per corner (F1 probe occlusion) on
// top of the usual bilinear/trilinear position weight. No bounds/window-membership check is
// performed here -- a caller sampling a position far outside the level's currently-covered window
// reads whatever stale data physically occupies that wrapped texel, the same accepted simplification
// renderer::SDFRayMarchPass's own SampleClipmap relies on its caller (SelectLevel) to have already
// ruled out.
vec3 SampleWorldProbeLevel(int level, vec3 worldPos, int resolution, float spacing) {
    vec3 continuousIndex = worldPos / spacing - vec3(0.5);
    ivec3 baseIndex = ivec3(floor(continuousIndex));
    vec3 frac = continuousIndex - vec3(baseIndex);

    vec3 totalIrradiance = vec3(0.0);
    float totalWeight = 0.0;
    for (int i = 0; i < 8; ++i) {
        ivec3 offset = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        ivec3 worldIndex = baseIndex + offset;
        ivec3 wrapped = ((worldIndex % resolution) + resolution) % resolution;

        float wx = (offset.x == 1) ? frac.x : (1.0 - frac.x);
        float wy = (offset.y == 1) ? frac.y : (1.0 - frac.y);
        float wz = (offset.z == 1) ? frac.z : (1.0 - frac.z);
        float trilinearWeight = wx * wy * wz;
        if (trilinearWeight <= 1.0e-6) {
            continue;
        }

#ifdef WORLD_PROBE_GRID_OCCLUSION_BINDING
        // Corner probe i's own world-space center -- (worldIndex + 0.5) * spacing, mirroring
        // WorldProbeInject.comp's own +0.5 probe-centering convention exactly.
        vec3 probeWorldPos = (vec3(worldIndex) + 0.5) * spacing;
        float distToPoint = distance(probeWorldPos, worldPos);
        vec2 occlusion = texelFetch(g_WorldProbeOcclusion[level], wrapped, 0).rg;
        float visibility = ChebyshevVisibility(occlusion, distToPoint);

        // A small floor keeps the blend from ever collapsing to a hard black seam if every one of
        // the 8 corners happens to test fully occluded this frame (e.g. a probe still mid-way
        // through its very first trace) -- a dim, not zero, result in that rare case.
        float weight = trilinearWeight * max(visibility, 0.02);
#else
        // No occlusion binding reserved by this includer -- plain trilinear weight only, exactly
        // the pre-F1 (single-level, no-occlusion) behavior. See this file's own header comment on
        // why occlusion is opt-in.
        float weight = trilinearWeight;
#endif
        vec3 irradiance = texelFetch(g_WorldProbeGrid[level], wrapped, 0).rgb;
        totalIrradiance += irradiance * weight;
        totalWeight += weight;
    }

    return totalWeight > 1.0e-6 ? totalIrradiance / totalWeight : vec3(0.0);
}

// F1: selects the finest level whose covered window comfortably contains `worldPos` (coverage is
// camera-centered, so distance FROM THE CAMERA is the right metric -- mirrors
// renderer::GlobalSDFPass's own per-level extent, centered on the camera-derived snapped voxel) and
// cross-blends across the boundary into the next-coarser level over the outer kBlendFraction of that
// level's own radius, so a query position crossing a level boundary never pops. A hard per-pixel
// level switch (like SDFRayMarch.comp's own SelectLevel) is fine for a coarse-to-refined hit search,
// where a one-voxel error is imperceptible; it is NOT fine here, since Lite mode's irradiance is the
// only GI term on screen for large stretches of the image.
vec3 SampleWorldProbeGrid(vec3 worldPos, vec3 cameraPosWorld) {
    int resolution = int(g_WorldProbeGridParams.gridResolution);
    float distFromCamera = distance(worldPos, cameraPosWorld);
    const float kBlendFraction = 0.2;

    int level = WORLD_PROBE_LEVEL_COUNT - 1;
    for (int L = 0; L < WORLD_PROBE_LEVEL_COUNT; ++L) {
        float halfExtent = float(resolution) * 0.5 * g_WorldProbeGridParams.levels[L].spacing;
        if (distFromCamera < halfExtent * (1.0 - kBlendFraction) || L == WORLD_PROBE_LEVEL_COUNT - 1) {
            level = L;
            break;
        }
    }

    vec3 sampleNear = SampleWorldProbeLevel(level, worldPos, resolution, g_WorldProbeGridParams.levels[level].spacing);
    if (level >= WORLD_PROBE_LEVEL_COUNT - 1) {
        return sampleNear; // Coarsest level -- nothing coarser to blend toward.
    }

    float halfExtent = float(resolution) * 0.5 * g_WorldProbeGridParams.levels[level].spacing;
    float blendStart = halfExtent * (1.0 - kBlendFraction);
    float blend = clamp((distFromCamera - blendStart) / max(halfExtent - blendStart, 1.0e-4), 0.0, 1.0);
    if (blend <= 0.0) {
        return sampleNear;
    }

    vec3 sampleFar = SampleWorldProbeLevel(level + 1, worldPos, resolution, g_WorldProbeGridParams.levels[level + 1].spacing);
    return mix(sampleNear, sampleFar, blend);
}

#endif
