#ifndef MEGALIGHTS_TYPES_GLSL
#define MEGALIGHTS_TYPES_GLSL

// UE5.8 rendering-parity gap G3: light-shape discriminators, value-for-value mirrors of
// renderer::MegaLightType (src/renderer/MegaLightsTypes.h). MEGALIGHT_TYPE_POINT (0) ignores every
// extended field below and behaves exactly as the pre-G3 isotropic point light did.
const uint MEGALIGHT_TYPE_POINT       = 0u;
const uint MEGALIGHT_TYPE_SPOT        = 1u;
const uint MEGALIGHT_TYPE_RECT        = 2u;
const uint MEGALIGHT_TYPE_PHOTOMETRIC = 3u;

// Procedural photometric ("IES-style") angular-falloff shapes -- mirrors renderer::
// MegaLightIESProfile. Stored in the low byte of MegaLight::iesProfileAndFlags. Plain analytic
// shapes, no baked .ies data (CLAUDE.md hard rule).
const uint MEGALIGHT_IES_NARROW = 0u;
const uint MEGALIGHT_IES_WIDE   = 1u;
const uint MEGALIGHT_IES_BARN   = 2u;

// Bit 8 of MegaLight::iesProfileAndFlags: a rect light emits from both faces when set (mirrors
// renderer::kMegaLightFlagRectTwoSided).
const uint MEGALIGHT_FLAG_RECT_TWOSIDED = 0x100u;

// GLSL mirror of renderer::MegaLight (src/renderer/MegaLightsTypes.h) -- 80 bytes, std430, five
// {vec3, scalar} blocks. See that struct's own G3 layout comment for the meaning of every extended
// field (direction/tangentU/cone/rect/profile). MEGALIGHT_TYPE_POINT reads only the first two blocks.
struct MegaLight {
    vec3 position;
    float radius;    // Distance at which the smooth windowed attenuation reaches zero.
    vec3 color;
    float intensity;
    vec3 direction;  // Spot cone axis / rect normal / photometric forward axis (points light -> scene).
    uint lightType;  // MEGALIGHT_TYPE_*.
    vec3 tangentU;   // Rect first in-plane axis (unit, orthogonal to direction); second axis = cross(direction, tangentU).
    float rectHalfExtentX;
    float spotCosInner;      // cos(inner cone half-angle).
    float spotCosOuter;      // cos(outer cone half-angle), < spotCosInner.
    float rectHalfExtentY;
    uint iesProfileAndFlags; // low byte: MEGALIGHT_IES_*; high bits: MEGALIGHT_FLAG_*.
};

// Extracts the MEGALIGHT_IES_* profile id from a light's packed iesProfileAndFlags field.
uint MegaLightIESProfile(MegaLight light) { return light.iesProfileAndFlags & 0xFFu; }
// True when a rect light emits from both faces.
bool MegaLightRectTwoSided(MegaLight light) { return (light.iesProfileAndFlags & MEGALIGHT_FLAG_RECT_TWOSIDED) != 0u; }

// Feature 1 of Phase 4 (MegaLights advanced roadmap: light BVH for RIS spatial bias) -- the fixed
// capacity of the spatial candidate pool megalights_bvh.glsl's GatherSpatialLightCandidates fills
// and megalights_ris.glsl's SelectLightRIS draws from. Declared here (not in either of those two
// files) so both can see an identical value regardless of which one is #included first --
// SelectLightRIS's own comment already establishes that RIS treats its population as approximate,
// not exhaustive, so a generous fixed cap needs no exact match to the BVH's own real leaf-fill
// count for any given query.
const uint kMegaLightsSpatialPoolCapacity = 64u;

#endif
