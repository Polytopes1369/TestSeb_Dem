// Light Functions implementation
// Applies light-specific masking/modulation to lighting contribution
// Based on UE5.8 Light Functions pattern
// Common use cases: gobo lights, light curtains, volumetric light shafts

#ifndef LIGHT_FUNCTIONS_GLSL
#define LIGHT_FUNCTIONS_GLSL

// Light function texture descriptors (bindless)
// In C++: add to MaterialParameterTable as optional textures
layout(set = 4, binding = 0) uniform sampler2D g_LightFunctionTextures[256];  // Bindless array

// Evaluate light function mask for a given light
// Parameters:
//   lightFunctionIndex: Index into bindless texture array (-1 = disabled)
//   worldPos: World position being lit
//   lightWorldPos: Light source world position
//   shadowCoord: Shadow map coordinate (if using shadow comparison)
// Returns:
//   Modulation factor [0..1] to apply to light contribution
float EvaluateLightFunction(
    int lightFunctionIndex,
    vec3 worldPos,
    vec3 lightWorldPos,
    vec2 shadowCoord
) {
    if (lightFunctionIndex < 0) {
        return 1.0;  // No light function: full contribution
    }

    // Project world position into light function texture space
    // Simplified: use screen-space projection (can be enhanced to light-space projection)
    vec2 lightFuncCoord = shadowCoord;  // Already in [0..1] range from shadow map

    // Sample light function texture
    vec4 lightFuncSample = texture(g_LightFunctionTextures[lightFunctionIndex], lightFuncCoord);

    // Use luminance as modulation factor
    float modulation = dot(lightFuncSample.rgb, vec3(0.299, 0.587, 0.114));

    // Optional: apply alpha as additional mask
    modulation *= lightFuncSample.a;

    return modulation;
}

// Variant: Directional light function (sun/moon)
// No position projection needed; uses direction-based coordinate
float EvaluateLightFunctionDirectional(
    int lightFunctionIndex,
    vec3 worldPos,
    vec3 lightDir,
    mat4 lightViewProj
) {
    if (lightFunctionIndex < 0) {
        return 1.0;
    }

    // Project world position through light's view-projection matrix
    vec4 lightSpacePos = lightViewProj * vec4(worldPos, 1.0);
    vec2 lightFuncCoord = lightSpacePos.xy / lightSpacePos.w;
    lightFuncCoord = lightFuncCoord * 0.5 + 0.5;  // NDC to [0..1]

    // Clamp to texture bounds (prevent wrapping artifacts at edges)
    lightFuncCoord = clamp(lightFuncCoord, vec2(0.01), vec2(0.99));

    // Sample light function
    vec4 lightFuncSample = texture(g_LightFunctionTextures[lightFunctionIndex], lightFuncCoord);

    // Luminance-based modulation
    float modulation = dot(lightFuncSample.rgb, vec3(0.299, 0.587, 0.114));
    modulation *= lightFuncSample.a;

    return modulation;
}

// Common light function patterns (procedural, no texture needed)

// Volumetric god-rays / crepuscular rays effect
float LightFunctionGodrays(vec3 worldPos, vec3 lightWorldPos, float rayCount, float raySharpness) {
    vec3 toLight = normalize(lightWorldPos - worldPos);
    vec3 rayDir = normalize(lightWorldPos - worldPos);

    // Angle-based modulation
    float angle = abs(atan(worldPos.y - lightWorldPos.y, length(worldPos.xz - lightWorldPos.xz)));
    float rays = sin(angle * rayCount) * 0.5 + 0.5;

    return pow(rays, raySharpness);
}

// Gobo pattern (simple rotating checkerboard)
float LightFunctionGobo(vec3 worldPos, float rotationAngle, float frequency) {
    vec2 rotated = vec2(
        worldPos.x * cos(rotationAngle) - worldPos.z * sin(rotationAngle),
        worldPos.x * sin(rotationAngle) + worldPos.z * cos(rotationAngle)
    );

    // Checkerboard pattern
    float checkerboard = mod(floor(rotated.x * frequency) + floor(rotated.z * frequency), 2.0);

    return checkerboard;
}

// Fog/curtain effect (distance-based attenuation)
float LightFunctionFogCurtain(vec3 worldPos, vec3 curtainNormal, vec3 curtainOrigin, float thickness, float falloff) {
    // Distance from point to curtain plane
    float distanceToCurtain = abs(dot(worldPos - curtainOrigin, curtainNormal));

    // Gaussian falloff from curtain surface
    float falloffFactor = exp(-distanceToCurtain * distanceToCurtain / (thickness * thickness)) * falloff;

    return 1.0 - falloffFactor;
}

#endif // LIGHT_FUNCTIONS_GLSL
