#ifndef PROCEDURAL_MATERIAL_GLSL
#define PROCEDURAL_MATERIAL_GLSL

// Shared procedural-material helpers: this codebase has no texture/material-binding system yet
// (see ClusterResolve.comp's class comment), so every shading pass that needs a stable, distinct
// per-primitive/per-cluster color derives one from an integer ID via these two functions instead.
// Factored out of draw.frag so ClusterResolve.comp's material evaluation step can reuse the exact
// same look rather than inventing a second, independently-tuned color scheme.

// Integer hash (PCG-style bit mixer) used to scramble a small, low-entropy ID (a meshID or cluster
// slot index, typically a small dense range) into a well-distributed 32-bit value. Without this
// step, consecutive IDs would produce visually similar hues since a naive "id / count" mapping
// walks the hue wheel linearly.
uint HashID(uint id) {
    uint state = id * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Standard HSV -> RGB conversion (H in [0,1), S and V in [0,1]).
vec3 HsvToRgb(vec3 hsv) {
    vec3 rgb = clamp(abs(mod(hsv.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return hsv.z * mix(vec3(1.0), rgb, hsv.y);
}

#endif // PROCEDURAL_MATERIAL_GLSL
