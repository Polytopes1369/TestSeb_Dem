#pragma once
// Phase A of the MegaLights native-port roadmap (see the approved plan,
// C:\Users\Seb\.claude\plans\memoized-splashing-seal.md): a growable, bindless-style point-light
// population for stochastic RIS-weighted direct lighting (renderer::MegaLightsPass), independent
// from renderer::LightingTypes.h's SceneLights/kMaxPointLights (that representation stays exactly
// as-is -- it feeds only renderer::SurfaceCachePass's own capture pipeline, a genuinely different
// consumer with a deliberately small UBO-sized cap; see this file's own GenerateProceduralLights()
// comment for why the two representations are not unified).

#include <array>
#include <cstdint>

#include "core/maths/Maths.h"

namespace renderer {

    // "UE5.8 showcase scale" -- the user's explicit Phase A choice (see the approved plan). Growable
    // in a later phase simply by raising this constant; RIS iterates g_Lights.lightCount at
    // runtime, no shader-side format change is needed.
    constexpr uint32_t kMaxMegaLights = 256u;

    // GLSL-friendly, std430-compatible mirror of MegaLight in
    // src/shaders/include/megalights_types.glsl -- 32 bytes, two {vec3, float} blocks, same
    // field-ordering convention as renderer::MaterialParameters (every vec3 immediately followed by
    // the scalar filling its std430 base-alignment gap, so no explicit padding is needed).
    struct MegaLight {
        maths::vec3 position{};
        float radius = 1.0f;   // Distance at which the smooth windowed attenuation reaches zero.
        maths::vec3 color = { 1.0f, 1.0f, 1.0f };
        float intensity = 1.0f;
    };
    static_assert(sizeof(MegaLight) == 32,
        "MegaLight must match MegaLight in megalights_types.glsl exactly (std430 layout)");

    struct MegaLightsData {
        std::array<MegaLight, kMaxMegaLights> lights{};
        uint32_t count = 0;
    };

    // Procedurally scatters `kMaxMegaLights` point lights, the large majority concentrated on the
    // dedicated "MegaLights" showcase zone (VulkanContext::GridSlot slot 8, the Cylinder) so
    // MegaLights reads as its own distinct feature rather than a uniform ambient sprinkle across
    // every zone -- the other 11 primitive zones (VulkanContext::GridSlot/kEntityCount --
    // duplicated here as a small local lookup table rather than including VulkanContext.h,
    // matching this codebase's existing renderer/VulkanContext one-directional-boundary
    // convention, e.g. core::EntityTransformCPU's own duplicated rotation math) get only a sparse
    // accent count each. The floor and the 2 static Lumen-corner walls intentionally get none --
    // they're lit by the scene's own sun instead. Deterministic, fixed-seed (never
    // std::random_device) -- matches GenerateShowcaseMaterialTable's own "a demo must look
    // identical every playback" rationale.
    MegaLightsData GenerateProceduralLights(uint32_t seed);

}
