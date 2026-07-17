#pragma once
// Direct-lighting scene data shared by every direct-lighting consumer in the engine: renderer::
// ClusterResolvePass/ClusterResolve(Binned).comp (main opaque VisBuffer resolve),
// renderer::TransparentForwardPass/TransparentForward.frag (translucent forward pass), and
// renderer::SurfaceCachePass/SurfaceCacheCapture.frag (Lumen-style GI capture) -- all three now
// consume the SAME DirectionalLight::intensity/color values below, in the SAME real photometric
// units (see DirectionalLight's own comment), so direct lighting and the GI it feeds stay on one
// consistent physical scale. renderer::VirtualShadowMapPass casts both the sun's clipmap and every
// active point light's 6-face cube of shadows (Phase 3 onward).
//
// Real photometric units (2026-07-17 recalibration, matching UE5.8's own Post Process Volume /
// Physical Camera exposure model exactly -- see EngineConfig.h's EXPOSURE_* comment and
// PostProcessComposite.comp's `1.0 / (1.2 * exp2(EV100))` exposure divisor): before this, every
// light intensity here and every direct-lighting shader's own equation was an arbitrary, unitless
// "looks okay after a naive [0,1] clamp" scalar (sun intensity 3.0, no albedo/pi normalization
// anywhere) -- harmless under the old naive blit, but ~1000x too dim once a physically-correct
// exposure divisor (calibrated for real-world lux/nits) was introduced, crushing the whole scene to
// near-black. Fixed at the root: every light is now a real photometric value (lux for the
// directional light, candela for point lights -- see PointLight's own comment) and every
// direct-lighting shader now computes outgoing radiance as the standard Lambertian
// `illuminance * albedo / PI`, exactly matching how UE5.8 itself derives surface luminance from a
// physical light + physical camera, so the SAME manual Post Process Camera defaults (already
// copied verbatim from UE5.8) now produce a correctly-exposed image with zero artificial
// exposure-compensation fudge needed.

#include <array>
#include <cstdint>

#include "core/maths/Maths.h"

namespace renderer {

    // Upper bound on simultaneous point lights the capture pipeline considers -- a fixed,
    // compile-time cap mirrored exactly by SurfaceCacheCapture.frag's `pointLights[8]` UBO array
    // size, matching this codebase's existing fixed-capacity conventions (geometry::
    // kMaxCardsPerEntity, renderer::GlobalSDFPass::kLevelCount, etc.) rather than a dynamic/
    // bindless count.
    constexpr uint32_t kMaxPointLights = 8u;

    // The sun -- shadowed via renderer::VirtualShadowMapPass's 3-level clipmap (Phase 3).
    // `direction` points FROM the light TOWARD the scene (matching the codebase's other directional
    // conventions -- every direct-lighting shader negates this one to get the surface-to-light
    // direction Lambertian shading needs, see e.g. ClusterResolve.comp's own comment).
    // `intensity` is real ILLUMINANCE in LUX (matching UE5.8's own Directional Light unit exactly)
    // -- ~10,000 lux is "bright, hazy/shaded daylight" (real-world reference: overcast day
    // ~1,000-10,000 lux, direct sun ~32,000-100,000 lux), chosen to pair correctly with this
    // engine's existing manual Post Process Camera default (aperture f/4, 1/60s, ISO 100 -- see
    // EngineConfig.h's EXPOSURE_* comment) without either crushing to black or blowing every
    // surface out to pure white.
    struct DirectionalLight {
        maths::vec3 direction = { -0.4f, -1.0f, -0.3f };
        maths::vec3 color = { 1.0f, 0.95f, 0.85f };
        float intensity = 10000.0f;
    };

    // Point lights contribute distance-attenuated light, shadowed via renderer::
    // VirtualShadowMapPass's own per-light 6-face cube of Virtual Shadow Maps (Phase 3 onward --
    // pre-Phase-3 these were unshadowed, since a correct per-point-light shadow needs a cube map
    // per light, a materially bigger feature than the sun's single orthographic map that the
    // pre-Phase-3 ShadowMapPass did not attempt).
    // `intensity` is real LUMINOUS INTENSITY in CANDELA (matching UE5.8's own Point Light unit when
    // "Intensity Units" = Candela) -- 400 candela approximates UE5.8's own default new Point Light
    // (5000 lumens / 4*pi steradians for an isotropic point source ~= 398 candela), consumed via the
    // standard inverse-square `candela / distance^2` illuminance-at-a-point formula (see
    // SurfaceCacheCapture.frag's ComputeDirectLighting -- the windowed falloff there is layered on
    // top of, not instead of, this real inverse-square law).
    struct PointLight {
        maths::vec3 position{};
        maths::vec3 color = { 1.0f, 1.0f, 1.0f };
        float intensity = 400.0f;
        float radius = 10.0f; // Distance at which attenuation reaches zero (see the frag shader's windowing term).
    };

    // Every light contributing to a frame's Surface Cache capture -- a plain CPU aggregate with no
    // GPU residency of its own; SurfaceCachePass::UpdateLighting packs this into its own UBO layout
    // (SurfaceCacheLightingUBO, SurfaceCachePass.cpp) on every call.
    struct SceneLights {
        DirectionalLight sun;
        std::array<PointLight, kMaxPointLights> pointLights{};
        uint32_t pointLightCount = 0;
    };

}
