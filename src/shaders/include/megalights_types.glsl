#ifndef MEGALIGHTS_TYPES_GLSL
#define MEGALIGHTS_TYPES_GLSL

// GLSL mirror of renderer::MegaLight (src/renderer/MegaLightsTypes.h) -- 32 bytes, std430.
struct MegaLight {
    vec3 position;
    float radius;    // Distance at which the smooth windowed attenuation reaches zero.
    vec3 color;
    float intensity;
};

// Feature 1 of Phase 4 (MegaLights advanced roadmap: light BVH for RIS spatial bias) -- the fixed
// capacity of the spatial candidate pool megalights_bvh.glsl's GatherSpatialLightCandidates fills
// and megalights_ris.glsl's SelectLightRIS draws from. Declared here (not in either of those two
// files) so both can see an identical value regardless of which one is #included first --
// SelectLightRIS's own comment already establishes that RIS treats its population as approximate,
// not exhaustive, so a generous fixed cap needs no exact match to the BVH's own real leaf-fill
// count for any given query.
const uint kMegaLightsSpatialPoolCapacity = 64u;

#endif
