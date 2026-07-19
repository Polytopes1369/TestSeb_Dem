// Caustics projection and animation
// Real-time caustics effect for water and underwater lighting
// Inspired by UE5.8 water material caustics

#ifndef CAUSTICS_PROJECTION_GLSL
#define CAUSTICS_PROJECTION_GLSL

// F12 (2026-07-19, UE5.8 rendering-parity gap: projected Caustics): this file was 100% dead code
// until now -- never #included, because no consumer's real pipeline layout ever allocated the
// hardcoded `set = 2, binding = 10` / `set = 3, binding = 5` bindings below assumed (this engine's
// shading passes only ever use set 0 for their own bindings -- see light_functions.glsl's own
// identical F12 fix comment for the full rationale). Fixed the same way: the caller
// (ClusterResolve.comp/ClusterResolveBinned.comp) must #define CAUSTICS_TEXTURE_SET/
// CAUSTICS_TEXTURE_BINDING/CAUSTICS_UBO_SET/CAUSTICS_UBO_BINDING before including this file, the
// same caller-defined-binding-macro convention as megalights_ris.glsl/shadow_page_table.glsl.
// g_CausticsTexture is renderer::ProceduralLightFunctionGenerator's own 5th generated slot
// (GetCausticsImageInfo(), see GenerateLightFunctionTextures.comp's own PatternCaustics()) -- a real
// procedurally-baked texture, zero data assets in the exe (CLAUDE.md).
#ifndef CAUSTICS_TEXTURE_SET
#error "Define CAUSTICS_TEXTURE_SET/CAUSTICS_TEXTURE_BINDING/CAUSTICS_UBO_SET/CAUSTICS_UBO_BINDING before including caustics_projection.glsl"
#endif

// Caustics texture (generated from noise or precomputed)
layout(set = CAUSTICS_TEXTURE_SET, binding = CAUSTICS_TEXTURE_BINDING) uniform sampler2D g_CausticsTexture;

// Caustics parameters (constant buffer). F12: filled ONCE at renderer::ClusterResolvePass::Init()
// (same "static, not re-uploaded every frame" convention as that class' own VTVolumeParams) --
// `timeSeconds` is threaded in as an explicit function parameter below instead of a UBO field
// (causticsScale.z is consequently unused/reserved), letting this UBO stay static: the caller
// already has the SAME per-frame time value on hand (ResolveViewParamsUBO.timeSeconds, already
// uploaded every frame for procedural_light_modulation.glsl's own stand-in functions this file
// replaces), so re-deriving/re-uploading a second copy of it here every frame would be redundant.
layout(std140, set = CAUSTICS_UBO_SET, binding = CAUSTICS_UBO_BINDING) uniform CausticsUBO {
    mat4 causticsProjection;    // Projection matrix for caustics coordinates
    vec4 causticsScale;         // XY: UV scale, Z: unused/reserved, W: intensity
    vec4 causticsAnimation;     // XY: scroll speed, Z: unused/reserved, W: depth falloff
} g_Caustics;

// Compute caustics modulation for a surface
// Parameters:
//   worldPos: World position of surface being lit
//   normal: Surface normal
//   waterSurfaceHeight: Y coordinate of water surface
//   timeSeconds: real elapsed time (F12: drives the scrolling animation below -- see this file's own
//     CausticsUBO comment for why this is a parameter rather than a UBO field)
// Returns:
//   Caustics modulation factor [0..1]
float ComputeCausticsModulation(vec3 worldPos, vec3 normal, float waterSurfaceHeight, float timeSeconds) {
    // Depth below water surface (positive = underwater)
    float depthUnderwater = max(0.0, waterSurfaceHeight - worldPos.y);

    // Early exit: too deep for caustics visibility
    if (depthUnderwater > g_Caustics.causticsAnimation.w) {
        return 1.0;  // No caustics modulation at depth limit
    }

    // Project world position through caustics matrix
    vec4 causticsCoord = g_Caustics.causticsProjection * vec4(worldPos, 1.0);
    vec2 uv = causticsCoord.xy / causticsCoord.w;

    // Apply time-based animation (scrolling caustics pattern)
    uv += g_Caustics.causticsAnimation.xy * timeSeconds;

    // Scale UV coordinates
    uv *= g_Caustics.causticsScale.xy;

    // Sample caustics texture (packed dual-channel for two animated layers)
    vec4 causticsLayer1 = texture(g_CausticsTexture, uv);
    vec4 causticsLayer2 = texture(g_CausticsTexture, uv + vec2(0.5));  // Offset second layer

    // Blend two caustics layers for richer pattern
    vec3 causticsColor = mix(causticsLayer1.rgb, causticsLayer2.rgb, 0.5);

    // Convert to modulation factor -- F12 fix: this must stay a strictly BRIGHTENING multiplier
    // (>= 1.0), matching this function's own "caustics as brightening, not darkening" comment (and
    // procedural_light_modulation.glsl's own ComputeUnderwaterCaustics(), the stand-in this call
    // site replaces, which likewise only ever returns >= 1.0 -- see that function's own
    // "mix(1.0, 1.0 + pattern, ...)" formula). The ORIGINAL formula here (`mix(1.0, luminance,
    // intensity)`) could actually DARKEN the surface wherever luminance < 1.0 (i.e. almost
    // everywhere, since a luminance sample is normally in [0,1]) -- contradicting its own comment.
    // An additive term (1.0 + luminance * intensity) fixes that: luminance is always >= 0 (RGB
    // texture samples), so the result can never drop below 1.0.
    float luminance = dot(causticsColor, vec3(0.299, 0.587, 0.114));
    float modulation = 1.0 + luminance * g_Caustics.causticsScale.w;  // Intensity factor.

    // Depth-based fade: caustics fade out with depth (mix toward 1.0 = no modulation, matching the
    // "beyond the depth falloff, no caustics at all" early exit above).
    float depthFade = exp(-depthUnderwater / g_Caustics.causticsAnimation.w);
    modulation = mix(1.0, modulation, depthFade);

    // Surface angle fade: caustics less visible on steep surfaces
    float normalDot = abs(dot(normal, vec3(0, 1, 0)));  // Upward-facing check
    float angleFade = normalDot;
    modulation = mix(1.0, modulation, angleFade);

    return modulation;
}

// Animated caustics pattern (procedural, no texture needed)
// Uses simplex noise for organic caustics appearance
float ProceduralCaustics(vec3 worldPos, float time, float scale) {
    // Simulate caustics using sine waves (simplified version of proper Perlin-based caustics)
    vec3 pos = worldPos * scale + time;

    // Three octaves of sine waves for caustics-like pattern
    float pattern = sin(pos.x) * cos(pos.z) * 0.5 +
                   sin(pos.z + time) * cos(pos.x + time * 0.5) * 0.3 +
                   sin((pos.x + pos.z) * 2.0 + time * 1.5) * 0.2;

    // Normalize to [0..1]
    pattern = pattern * 0.5 + 0.5;

    // Add sharpness (caustics have distinct bright regions)
    pattern = pow(pattern, 2.0);

    return pattern;
}

#endif // CAUSTICS_PROJECTION_GLSL
