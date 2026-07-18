#ifndef DECAL_COMMON_GLSL
#define DECAL_COMMON_GLSL

// UE5.8-parity gap G1 (Decal system): shared GPU-side decal definitions + the purely procedural
// (noise-only, zero-texture) decal-material evaluation, consumed by DecalProject.comp. Reuses this
// codebase's existing procedural-noise machinery (displacement_noise.glsl's ValueNoise3D / Fbm3D)
// so every decal pattern is generated from math alone, matching CLAUDE.md's "aucun data dans mon
// .exe" rule and the "no texture/material-binding system" precedent every other shading pass here
// already follows.
#include "displacement_noise.glsl"

// std430 mirror of renderer::DecalInstanceGPU (DecalTypes.h) -- 144 bytes, must match field-for-field.
struct DecalInstance {
    mat4 worldToDecal;    // world -> decal-local unit cube ([-1,1]^3 inside).
    vec4 axisRightWorld;  // xyz = unit box right axis; w = uvScale.
    vec4 axisUpWorld;     // xyz = unit box up axis; w = normalStrength.
    vec4 projAxisWorld;   // xyz = unit projection axis (right x up); w = opacity.
    vec4 tint;            // xyz = tint color; w = emissive strength (linear HDR).
    vec4 params;          // x = decalType (as float); y = sortPriority; z/w reserved.
};

// std430 mirror of geometry::DecalBVHNode (DecalBVH.h) -- byte-for-byte the same 32-byte
// { float[3], int, float[3], int } layout as megalights_bvh.glsl's LightBVHNode / SDFRayMarch.comp's
// own BVHNode, declared identically (the array form, not vec3, matching that established convention).
struct DecalBVHNode {
    float boundsMin[3];
    int leftFirst; // Interior (count == 0): left child is ALWAYS nodeIndex+1; this holds the RIGHT child's index. Leaf (count > 0): first index into decalIndices.
    float boundsMax[3];
    int count;     // 0 == interior node; > 0 == leaf, holding this many decals starting at leftFirst.
};

// Procedural decal material kinds -- kept in lockstep with renderer::DecalType (DecalTypes.h).
#define DECAL_TYPE_MOSS   0u
#define DECAL_TYPE_RUST   1u
#define DECAL_TYPE_CRACKS 2u
#define DECAL_TYPE_PAINT  3u

// One evaluated decal fragment: the procedural material patch plus how strongly it covers the pixel.
struct DecalSample {
    vec3 albedo;
    float coverage;     // [0,1] pre-opacity coverage of this decal at this pixel.
    float roughness;
    float metallic;
    vec3 emissive;      // Linear HDR additive self-illumination.
    float normalRelief; // Signed local height perturbation ([-1,1]) used to bend the GBuffer normal.
};

// Feathers the decal toward its box edges so it fades out smoothly instead of ending on a hard
// rectangle seam. `localXY` are the decal-local coordinates in [-1,1]; `feather` is the fraction of
// the half-extent over which the fade happens.
float DecalEdgeFade(vec2 localXY, float feather) {
    vec2 d = 1.0 - smoothstep(vec2(1.0 - feather), vec2(1.0), abs(localXY));
    return d.x * d.y;
}

// Crack coverage field ([0,1], 1 = deepest crack) as a standalone world-space function so both the
// CRACKS material branch AND DecalProject.comp's normal-perturbation finite difference sample the
// exact same surface -- keeping the shaded groove and the bent normal consistent.
float DecalCrackField(vec3 worldPos, float uvScale) {
    float veins = abs(Fbm3D(worldPos * (2.2 * uvScale)));
    float crack = 1.0 - smoothstep(0.0, 0.055, veins);
    float veins2 = abs(Fbm3D(worldPos * (5.5 * uvScale) + vec3(7.7)));
    crack = max(crack, (1.0 - smoothstep(0.0, 0.03, veins2)) * 0.7);
    return crack;
}

// Evaluates the fully-procedural decal material for one pixel. `uv01` is the decal-local face
// coordinate in [0,1]^2 (used for pattern stability across the box face), `worldPos` drives the
// noise domain so patterns stay coherent regardless of box scale, `tint` is the per-decal color, and
// `edgeFade` is DecalEdgeFade()'s result already folded in by the caller.
DecalSample EvaluateProceduralDecal(uint type, vec2 uv01, vec3 worldPos, vec3 tint, float uvScale,
                                    float emissiveStrength, float edgeFade) {
    DecalSample s;
    s.albedo = tint;
    s.coverage = edgeFade;
    s.roughness = 0.85;
    s.metallic = 0.0;
    s.emissive = vec3(0.0);
    s.normalRelief = 0.0;

    // Base multi-octave noise fields, sampled in world space so adjacent decals / large boxes never
    // show a stretched or repeating pattern. Fbm3D returns [-1,1]; remap where a [0,1] field is
    // wanted.
    float n0 = Fbm3D(worldPos * (0.9 * uvScale)) * 0.5 + 0.5;          // Broad patches.
    float n1 = Fbm3D(worldPos * (3.5 * uvScale) + vec3(11.3)) * 0.5 + 0.5; // Fine mottling.

    if (type == DECAL_TYPE_MOSS) {
        // Patchy organic growth: coverage is a soft-thresholded broad noise so the moss clusters
        // into irregular islands with eroded borders rather than covering the whole box.
        float patchMask = smoothstep(0.35, 0.62, n0);
        float mottle = mix(0.65, 1.0, n1);
        s.albedo = tint * mottle;
        // Slight desaturated dark speckle in the densest areas.
        s.albedo *= mix(1.0, 0.75, smoothstep(0.6, 0.9, n1));
        s.roughness = 0.92;
        s.coverage = edgeFade * patchMask;
    } else if (type == DECAL_TYPE_RUST) {
        // Oxidation stain with vertical drip streaks: compress the vertical (uv01.y) domain so the
        // streaks stretch downward, then blend with broad rust patches.
        float streak = Fbm3D(vec3(uv01.x * 8.0 * uvScale, uv01.y * 1.2, 3.0)) * 0.5 + 0.5;
        float patchMask = smoothstep(0.30, 0.70, n0);
        float rust = clamp(patchMask * 0.75 + streak * 0.5, 0.0, 1.0);
        // Rust hue varies from dark iron-brown to brighter orange with the fine noise.
        vec3 dark = tint * 0.5;
        vec3 bright = tint * 1.15 + vec3(0.10, 0.03, 0.0);
        s.albedo = mix(dark, bright, n1);
        s.roughness = 0.88;
        s.coverage = edgeFade * rust;
    } else if (type == DECAL_TYPE_CRACKS) {
        // Thin fracture lines: cracks live at the zero-crossings of a signed noise field, so
        // 1 - smoothstep(abs(...)) leaves a network of narrow dark veins (DecalCrackField). The crack
        // also carves a negative relief that bends the GBuffer normal (via s.normalRelief, whose
        // gradient DecalProject.comp finite-differences from DecalCrackField directly).
        float crack = DecalCrackField(worldPos, uvScale);
        s.albedo = tint * mix(0.06, 0.16, n1); // Dark crack interior.
        s.roughness = 0.95;
        s.coverage = edgeFade * crack;
        s.normalRelief = -crack; // Indented groove.
    } else { // DECAL_TYPE_PAINT
        // Hard-edged painted hazard stripes along the decal-local diagonal, with weathered edges and
        // scuffed wear so it does not read as a flat sticker.
        float stripeCoord = (uv01.x + uv01.y) * 6.0 * uvScale;
        float stripe = step(0.5, fract(stripeCoord));
        // Weathering: erode the paint where the fine noise is low (scuffs), soften stripe edges.
        float wear = smoothstep(0.25, 0.6, n1);
        float paint = stripe * wear;
        s.albedo = tint;
        s.roughness = 0.55;
        s.metallic = 0.0;
        s.emissive = tint * emissiveStrength * paint;
        s.coverage = edgeFade * paint;
    }

    s.coverage = clamp(s.coverage, 0.0, 1.0);
    return s;
}

#endif // DECAL_COMMON_GLSL
