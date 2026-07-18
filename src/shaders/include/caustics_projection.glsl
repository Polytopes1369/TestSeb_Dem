// Caustics projection and animation
// Real-time caustics effect for water and underwater lighting
// Inspired by UE5.8 water material caustics

#ifndef CAUSTICS_PROJECTION_GLSL
#define CAUSTICS_PROJECTION_GLSL

// Caustics texture (generated from noise or precomputed)
layout(set = 2, binding = 10) uniform sampler2D g_CausticsTexture;

// Caustics parameters (constant buffer)
layout(std140, set = 3, binding = 5) uniform CausticsUBO {
    mat4 causticsProjection;    // Projection matrix for caustics coordinates
    vec4 causticsScale;         // XY: UV scale, Z: time offset, W: intensity
    vec4 causticsAnimation;     // XY: scroll speed, Z: intensity variation, W: depth falloff
} g_Caustics;

// Compute caustics modulation for a surface
// Parameters:
//   worldPos: World position of surface being lit
//   normal: Surface normal
//   waterSurfaceHeight: Y coordinate of water surface
// Returns:
//   Caustics modulation factor [0..1]
float ComputeCausticsModulation(vec3 worldPos, vec3 normal, float waterSurfaceHeight) {
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
    float time = g_Caustics.causticsScale.z;
    uv += g_Caustics.causticsAnimation.xy * time;

    // Scale UV coordinates
    uv *= g_Caustics.causticsScale.xy;

    // Sample caustics texture (packed dual-channel for two animated layers)
    vec4 causticsLayer1 = texture(g_CausticsTexture, uv);
    vec4 causticsLayer2 = texture(g_CausticsTexture, uv + vec2(0.5));  // Offset second layer

    // Blend two caustics layers for richer pattern
    vec3 causticsColor = mix(causticsLayer1.rgb, causticsLayer2.rgb, 0.5);

    // Convert to modulation factor (caustics as brightening, not darkening)
    float modulation = dot(causticsColor, vec3(0.299, 0.587, 0.114));  // Luminance
    modulation = mix(1.0, modulation, g_Caustics.causticsScale.w);  // Intensity factor

    // Depth-based fade: caustics fade out with depth
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
