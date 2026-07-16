#pragma once
// Direct-lighting scene data consumed by the Surface Cache capture pipeline (renderer::
// VirtualShadowMapPass casts both the sun's clipmap and every active point light's 6-face cube of
// shadows, Phase 3 onward; renderer::SurfaceCachePass / SurfaceCacheCapture.frag shade each
// captured texel with them). No shared light representation exists anywhere else in this codebase
// yet -- every other shading pass hardcodes a single fixed light direction inline (e.g.
// ClusterResolve.comp's own `lightDir = normalize(vec3(0.5, 1.0, 0.5))`); this is the first one.

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

    // The Surface Cache's sun -- shadowed via renderer::VirtualShadowMapPass's 3-level clipmap
    // (Phase 3). `direction` points FROM the light TOWARD the scene (matching the codebase's other
    // directional conventions, e.g. ClusterResolve.comp's own `lightDir`, which is instead the
    // surface-to-light direction -- SurfaceCacheCapture.frag negates this one before using it, see
    // that shader's own comment).
    struct DirectionalLight {
        maths::vec3 direction = { -0.4f, -1.0f, -0.3f };
        maths::vec3 color = { 1.0f, 0.95f, 0.85f };
        float intensity = 3.0f;
    };

    // Point lights contribute distance-attenuated light, shadowed via renderer::
    // VirtualShadowMapPass's own per-light 6-face cube of Virtual Shadow Maps (Phase 3 onward --
    // pre-Phase-3 these were unshadowed, since a correct per-point-light shadow needs a cube map
    // per light, a materially bigger feature than the sun's single orthographic map that the
    // pre-Phase-3 ShadowMapPass did not attempt).
    struct PointLight {
        maths::vec3 position{};
        maths::vec3 color = { 1.0f, 1.0f, 1.0f };
        float intensity = 1.0f;
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
