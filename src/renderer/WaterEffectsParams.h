#pragma once

#include "core/maths/Maths.h"

namespace renderer {

// Caustics parameters, mirrors CausticsUBO in caustics_projection.glsl (std140 layout)
struct alignas(16) CausticsParams {
    maths::mat4 causticsProjection{};
    maths::vec4 causticsScale{1.0f, 1.0f, 0.0f, 0.5f};       // XY: UV scale, Z: time, W: intensity
    maths::vec4 causticsAnimation{0.05f, 0.03f, 0.4f, 2.0f}; // XY: scroll speed, Z: intensity var, W: depth falloff
};

// Foam parameters, mirrors FoamUBO in foam_generation.glsl (std140 layout)
struct alignas(16) FoamParams {
    maths::vec4 foamColor{1.0f, 1.0f, 1.0f, 0.85f};
    maths::vec4 foamParameters{1.0f, 0.3f, 0.15f, 1.5f};      // X: intensity, Y: height threshold, Z: edge fade, W: anim speed
    maths::vec4 breakingWaveThreshold{0.35f, 0.0f, 0.0f, 0.0f};
};

} // namespace renderer
