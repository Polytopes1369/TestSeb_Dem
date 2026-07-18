#pragma once

#include <cstdint>

namespace renderer {

// Iridescence material parameters (added to Substrate BSDF)
// This structure is mirrored in iridescence_bsdf.glsl
struct alignas(16) IridescenceMaterialParams {
    float iridescenceAmount = 0.0f;        // [0..1] blend factor for iridescence effect
    float iridescenceThickness = 0.5f;     // [0..1] film thickness (maps to 10-500nm)
    uint32_t _padding0 = 0;
    uint32_t _padding1 = 0;
};

// Light Function material parameters
// Applied per-light in MegaLightsPass to modulate light contribution
struct alignas(16) LightFunctionMaterialParams {
    int32_t lightFunctionTextureIndex = -1;  // Index into bindless texture array (-1 = disabled)
    float lightFunctionIntensity = 1.0f;     // [0..1] modulation strength
    uint32_t _padding0 = 0;
    uint32_t _padding1 = 0;
};

} // namespace renderer
