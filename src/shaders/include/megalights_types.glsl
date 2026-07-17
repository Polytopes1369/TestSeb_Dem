#ifndef MEGALIGHTS_TYPES_GLSL
#define MEGALIGHTS_TYPES_GLSL

// GLSL mirror of renderer::MegaLight (src/renderer/MegaLightsTypes.h) -- 32 bytes, std430.
struct MegaLight {
    vec3 position;
    float radius;    // Distance at which the smooth windowed attenuation reaches zero.
    vec3 color;
    float intensity;
};

#endif
