// Foam generation and wave breaking detection
// Procedural foam creation for water effects
// Based on UE5.8 water material foam system

#ifndef FOAM_GENERATION_GLSL
#define FOAM_GENERATION_GLSL

// Foam parameters
layout(std140, set = 3, binding = 6) uniform FoamUBO {
    vec4 foamColor;             // RGBA foam color (typically white with alpha)
    vec4 foamParameters;        // X: intensity, Y: wave height threshold, Z: edge fade, W: animation speed
    vec4 breakingWaveThreshold; // Wave height derivatives that trigger foam generation
} g_Foam;

// Detect wave breaking conditions
// Parameters:
//   waveHeightDerivative: Gradient of wave height (dH/dX, dH/dY)
//   waveHeight: Current wave height at position
//   threshold: Height derivative threshold for breaking detection
// Returns:
//   Foam amount [0..1] if wave is breaking, 0 otherwise
float DetectWaveBreaking(vec2 waveHeightDerivative, float waveHeight, float threshold) {
    // Wave breaks when slope is steep (high derivative)
    float slope = length(waveHeightDerivative);

    // Break detection: derivative exceeds threshold
    float breakingAmount = smoothstep(0.0, threshold, slope);

    // Height-based detection: shallow water breaks more easily
    // (assumed: world Y coordinate represents depth, lower = shallower)
    float heightFactor = clamp(waveHeight * 10.0, 0.0, 1.0);
    breakingAmount *= heightFactor;

    return breakingAmount;
}

// Generate foam pattern at wave breaking zone
// Parameters:
//   worldPos: World position
//   time: Shader time uniform
//   breakingAmount: Output from DetectWaveBreaking()
// Returns:
//   RGBA foam color with appropriate alpha
vec4 GenerateFoamPattern(vec3 worldPos, float time, float breakingAmount) {
    if (breakingAmount < 0.01) {
        return vec4(0.0);  // No foam
    }

    // Animated noise pattern for foam (turbulent appearance)
    // Using sine waves as simplified noise (proper implementation would use Perlin/Simplex)
    float foamPattern = sin(worldPos.x * 5.0 + time * 2.0) *
                       cos(worldPos.z * 3.0 - time * 1.5) *
                       sin((worldPos.x + worldPos.z) * 8.0 + time * 3.0);

    foamPattern = fract(foamPattern * 0.5 + 0.5);  // Normalize to [0..1]

    // Add turbulence (high frequency modulation)
    float turbulence = sin(worldPos.x * 50.0 + time * 20.0) * 0.1 +
                       sin(worldPos.z * 40.0 + time * 15.0) * 0.1;
    foamPattern = clamp(foamPattern + turbulence, 0.0, 1.0);

    // Sharp foam edges (caustics-like appearance)
    foamPattern = pow(foamPattern, 1.5);

    // Modulate by breaking amount
    foamPattern *= breakingAmount;

    // Add temporal variation (foam dissolves quickly)
    float foamLife = mod(time * g_Foam.foamParameters.w, 1.0);
    float dissolutionFade = 1.0 - smoothstep(0.5, 1.0, foamLife);
    foamPattern *= dissolutionFade;

    // Return foam color with computed alpha
    return vec4(g_Foam.foamColor.rgb, foamPattern * g_Foam.foamColor.a);
}

// Compute foam height displacement
// Foam floats on water surface; add small displacement to vertex position
// Parameters:
//   foamAmount: Amount of foam [0..1]
//   normal: Surface normal
// Returns:
//   Vertex displacement in world space
vec3 ComputeFoamDisplacement(float foamAmount, vec3 normal, float time) {
    // Foam bobs up and down slightly
    float bobbing = sin(time * 2.0) * 0.02;  // 2cm bobbing amplitude

    // Displace along surface normal
    return normal * bobbing * foamAmount;
}

// Blends foam over base water color
// Parameters:
//   baseWaterColor: Original water surface color (RGBA)
//   foamColor: Foam RGBA
//   foamAmount: Foam coverage [0..1]
// Returns:
//   Blended water+foam color
vec4 BlendFoam(vec4 baseWaterColor, vec4 foamColor, float foamAmount) {
    // Foam is opaque white; blend over base water
    // Using alpha-blend formula: out = foam + (1-foam.a) * base
    vec3 blendedColor = mix(baseWaterColor.rgb, foamColor.rgb, foamColor.a * foamAmount);

    // Preserve water depth in alpha channel
    float blendedAlpha = baseWaterColor.a;

    return vec4(blendedColor, blendedAlpha);
}

#endif // FOAM_GENERATION_GLSL
