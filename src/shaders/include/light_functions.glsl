// Light Functions implementation
// Applies light-specific masking/modulation to lighting contribution
// Based on UE5.8 Light Functions pattern
// Common use cases: gobo lights, light curtains, volumetric light shafts

#ifndef LIGHT_FUNCTIONS_GLSL
#define LIGHT_FUNCTIONS_GLSL
#extension GL_EXT_nonuniform_qualifier : require

// F12 (2026-07-19, UE5.8 rendering-parity gap: Texture-based Light Functions): this file was 100%
// dead code until now -- never #included anywhere, because no consumer's real pipeline layout ever
// allocated the descriptor set the ORIGINAL hardcoded `set = 4, binding = 0` below assumed (this
// engine's shading passes only ever use set 0, occasionally set 1/2/3 for a genuinely separate
// concern like Surface Cache tracing -- see e.g. megalights_ris.glsl's own caller-defined-binding-
// macro convention for how every other bindless/borrowed resource in this codebase actually gets
// wired in). Fixed the same way: the caller (ClusterResolve.comp/ClusterResolveBinned.comp) must
// #define LIGHT_FUNCTIONS_SET/LIGHT_FUNCTIONS_BINDING before including this file, exactly the
// convention shadow_page_table.glsl/reflection_view_params.glsl already establish. Also shrunk the
// array from a placeholder 256 down to renderer::ProceduralLightFunctionGenerator::
// kLightFunctionSlotCount (4) -- the real generator (src/shaders/src/Streaming/
// GenerateLightFunctionTextures.comp) bakes exactly that many procedural gobo patterns at startup
// (CLAUDE.md: zero data assets in the exe), so 256 was always aspirational, never backed by real
// slots; a mismatched array size here vs. the real VkWriteDescriptorSet's descriptorCount would be
// undefined behavior the first time SAMPLING code ever indexed past slot 3.
#ifndef LIGHT_FUNCTIONS_SET
#error "Define LIGHT_FUNCTIONS_SET/LIGHT_FUNCTIONS_BINDING before including light_functions.glsl"
#endif

#define LIGHT_FUNCTION_TEXTURE_COUNT 4u // Must match renderer::ProceduralLightFunctionGenerator::kLightFunctionSlotCount.

// Light function texture descriptors (bindless)
layout(set = LIGHT_FUNCTIONS_SET, binding = LIGHT_FUNCTIONS_BINDING) uniform sampler2D g_LightFunctionTextures[LIGHT_FUNCTION_TEXTURE_COUNT];  // Bindless array

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

    // Sample light function texture (nonuniformEXT: lightFunctionIndex varies per-pixel/per-light,
    // not dynamically uniform across the invocation group -- same requirement/convention as
    // mask_sampling.glsl's own SampleMaskAlpha()).
    vec4 lightFuncSample = texture(g_LightFunctionTextures[nonuniformEXT(lightFunctionIndex)], lightFuncCoord);

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
    vec4 lightFuncSample = texture(g_LightFunctionTextures[nonuniformEXT(lightFunctionIndex)], lightFuncCoord);

    // Luminance-based modulation
    float modulation = dot(lightFuncSample.rgb, vec3(0.299, 0.587, 0.114));
    modulation *= lightFuncSample.a;

    return modulation;
}

// F12: radial/polar world-space projection -- reuses this codebase's own pre-existing "angle from a
// (pseudo-)light position, projected radially" coordinate scheme (see procedural_light_modulation
// .glsl's own EvaluateSunLightFunction / this file's own LightFunctionGodrays below for the identical
// angle derivation) as the sampling coordinate for a REAL baked texture instead of a purely analytic
// sine formula. The natural choice for this engine: unlike EvaluateLightFunctionDirectional above
// (which needs a real light-space ortho/perspective matrix), nothing here requires a dedicated
// projection matrix -- only a light position (a real MegaLight position, or a pseudo-position placed
// far along a directional light's own -direction, exactly procedural_light_modulation.glsl's own
// EvaluateSunLightFunction convention). `rotationRadians` lets a caller spin the gobo pattern (e.g.
// slowly, for a "rotating gobo wheel" look); `radialScale` controls how many times the pattern
// repeats outward from the light (the sampler's own REPEAT addressing, see renderer::
// ProceduralLightFunctionGenerator::Init's own sampler comment, makes this tile seamlessly).
float EvaluateLightFunctionRadial(int lightFunctionIndex, vec3 worldPos, vec3 lightWorldPos, float rotationRadians, float radialScale) {
    if (lightFunctionIndex < 0) {
        return 1.0;
    }
    float angle = atan(worldPos.y - lightWorldPos.y, length(worldPos.xz - lightWorldPos.xz)) + rotationRadians;
    float dist = length(worldPos - lightWorldPos) * radialScale;
    vec2 uv = vec2(angle * (1.0 / (2.0 * 3.14159265358979323846)) + 0.5, dist);

    vec4 lightFuncSample = texture(g_LightFunctionTextures[nonuniformEXT(lightFunctionIndex)], uv);
    return dot(lightFuncSample.rgb, vec3(0.299, 0.587, 0.114));
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

    // Checkerboard pattern. F12 fix: `rotated` is a vec2 (see its own construction just above) --
    // the original `.z` here was a real, never-caught compile error (out-of-range swizzle on a
    // vec2), concrete proof this whole file was 100% dead code before F12 finally #included it
    // anywhere (glslc never even saw this line until now). `.y` is the intended second component.
    float checkerboard = mod(floor(rotated.x * frequency) + floor(rotated.y * frequency), 2.0);

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
