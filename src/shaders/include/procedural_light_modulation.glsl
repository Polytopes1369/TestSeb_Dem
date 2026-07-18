#ifndef PROCEDURAL_LIGHT_MODULATION_GLSL
#define PROCEDURAL_LIGHT_MODULATION_GLSL

// Wave 2 (UE5.8 caustics/light-function parity): binding-free procedural modulation terms for
// ClusterResolve.comp/ClusterResolveBinned.comp's direct sun term. Deliberately standalone rather
// than #include-ing include/caustics_projection.glsl or include/light_functions.glsl directly --
// both of those files declare real texture/UBO/bindless-array bindings at file scope
// unconditionally (g_CausticsTexture, g_Caustics, g_LightFunctionTextures[256]), and a textual
// #include has no way to pull in only one function while leaving those bindings out; the two
// consumers here only have a single descriptor set (set 0) in their real pipeline layout, so
// pulling in bindings for sets 2/3/4 that layout doesn't provide would break pipeline creation.
// This engine also authors no texture assets at all (see CLAUDE.md's "aucun data dans mon .exe"
// constraint), so the texture-free procedural variants are the only ones actually usable here
// anyway -- see caustics_projection.glsl's own ProceduralCaustics()/light_functions.glsl's own
// LightFunctionGodrays() for the fuller, texture-backed versions this could grow into if a real
// asset pipeline is ever added.

// Procedural underwater caustics -- brightens the direct sun term on submerged terrain, fading out
// with depth. Same sine-wave formula as caustics_projection.glsl's ProceduralCaustics(), reproduced
// here standalone (see this file's own header comment for why).
// Returns a modulation factor to multiply against the direct sun term: 1.0 (no-op) above the water
// surface or once depth exceeds the falloff distance.
float ComputeUnderwaterCaustics(vec3 worldPos, float timeSeconds, float waterSurfaceHeight) {
    float depthUnderwater = waterSurfaceHeight - worldPos.y;
    const float kCausticsFalloffDistance = 0.6; // World units -- shallow basins only (this scene's own small water depth range).
    if (depthUnderwater <= 0.0 || depthUnderwater > kCausticsFalloffDistance) {
        return 1.0;
    }

    const float kCausticsScale = 3.0;
    const float kCausticsIntensity = 0.35; // Subtle brightening, not a hard-edged pattern -- matches this codebase's own "gentle, not binary" weather-modulation convention.
    vec3 pos = worldPos * kCausticsScale + timeSeconds * 0.6;
    float pattern = sin(pos.x) * cos(pos.z) * 0.5 +
                   sin(pos.z + timeSeconds) * cos(pos.x + timeSeconds * 0.5) * 0.3 +
                   sin((pos.x + pos.z) * 2.0 + timeSeconds * 1.5) * 0.2;
    pattern = pattern * 0.5 + 0.5;
    pattern = pow(pattern, 2.0);

    float depthFade = 1.0 - smoothstep(0.0, kCausticsFalloffDistance, depthUnderwater);
    return mix(1.0, 1.0 + pattern, kCausticsIntensity * depthFade);
}

// Procedural sun light function -- a subtle, always-on god-rays-style angular modulation of the
// sun's direct contribution (UE5.8's own common "directional light function" use case: crepuscular
// rays / cloud-shaft-like banding, NOT a hard gobo cutout -- a hard on/off pattern across the
// entire scene's primary light would be a jarring, un-tunable change to this codebase's existing
// showcase look, unlike this deliberately gentle modulation). Same angular banding formula as
// light_functions.glsl's own LightFunctionGodrays(), reproduced standalone (see this file's own
// header comment for why) with a fixed pseudo-position far along -sunDirection (a directional
// light has no real world position for the angle-from-light computation LightFunctionGodrays needs).
// Returns a modulation factor to multiply against the direct sun term, always within [1-kIntensity, 1].
float EvaluateSunLightFunction(vec3 worldPos, vec3 sunDirectionToScene) {
    const float kRayCount = 24.0;
    const float kRaySharpness = 2.0;
    const float kIntensity = 0.12; // Subtle -- see this function's own header comment on why this must not read as a hard gobo cutout.
    const float kPseudoLightDistance = 1000.0;
    vec3 pseudoLightWorldPos = worldPos - sunDirectionToScene * kPseudoLightDistance;

    float angle = abs(atan(worldPos.y - pseudoLightWorldPos.y, length(worldPos.xz - pseudoLightWorldPos.xz)));
    float rays = sin(angle * kRayCount) * 0.5 + 0.5;
    float godrays = pow(rays, kRaySharpness);
    return mix(1.0 - kIntensity, 1.0, godrays);
}

#endif // PROCEDURAL_LIGHT_MODULATION_GLSL
