#ifndef ACES_TONEMAP_GLSL
#define ACES_TONEMAP_GLSL

// Narkowicz 2015 fast fit of the full ACES Reference Rendering Transform + Output Device
// Transform (RRT+ODT) -- the same practical single-curve approximation Unreal Engine's own
// "ACES" filmic tonemapper uses in its simplified (non-LUT) path. Operates on linear HDR
// scene-referred color that has already been scaled by exposure; input is unclamped, output is
// display-referred [0, 1] linear (still needs the separate GammaCorrection() display encode from
// color_grading.glsl afterwards).
vec3 ACESFilmicTonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

#endif
