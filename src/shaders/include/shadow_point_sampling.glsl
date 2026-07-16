#ifndef SHADOW_POINT_SAMPLING_GLSL
#define SHADOW_POINT_SAMPLING_GLSL

// VirtualShadowMapPass's per-light 6-face cube of Virtual Shadow Maps. Built even though this
// demo's SceneLights::pointLightCount is 0 by default (see this phase's plan -- "done exactly
// as in UE 5.8" tie-breaker) with one verification light authored in
// ClusterRenderPipeline::Init(). VSM indices [0, SHADOW_SUN_LEVEL_COUNT) belong to the sun's
// clipmap (shadow_sun_sampling.glsl); point light face VSMs start right after, at
// SHADOW_SUN_LEVEL_COUNT + pointLightSlot*6 + face -- must match renderer::VirtualShadowMapPass's
// own VSM index assignment exactly.
//
// Requires shadow_page_table.glsl (SHADOW_SUN_LEVEL_COUNT is pulled in via shadow_sun_sampling.glsl,
// which the includer must also include first), shadow_feedback.glsl, and shadow_atlas_sampling.glsl.
//
// The includer must #define SHADOW_POINT_FACES_SET / SHADOW_POINT_FACES_BINDING before including
// this header.

#define SHADOW_MAX_POINT_LIGHTS 8u // Must match renderer::kMaxPointLights (LightingTypes.h).

layout(std140, set = SHADOW_POINT_FACES_SET, binding = SHADOW_POINT_FACES_BINDING) uniform ShadowPointFacesUBO {
    mat4 viewProj[SHADOW_MAX_POINT_LIGHTS * 6u]; // Index = pointLightSlot*6 + face.
} g_ShadowPointFaces;

// Selects one of the 6 cube faces from `dirFromLight` (need not be normalized -- only the
// dominant-axis sign/magnitude comparison matters), the standard cubemap face-selection rule.
// Face order: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
uint ShadowCubeFaceIndex(vec3 dirFromLight) {
    vec3 absDir = abs(dirFromLight);
    if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
        return dirFromLight.x >= 0.0 ? 0u : 1u;
    }
    if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
        return dirFromLight.y >= 0.0 ? 2u : 3u;
    }
    return dirFromLight.z >= 0.0 ? 4u : 5u;
}

// Point light shadow visibility [0,1] for light `pointLightSlot` (its world position `lightPos`)
// at `worldPos`. Simplified fallback vs. the sun's (shadow_sun_sampling.glsl): a cube face has no
// natural "coarser parent" without a full per-face mip chain (unjustified complexity for a light
// this demo exercises exactly once, see this phase's plan's own "Hors scope" note) -- a page not
// yet resident is requested and this call returns 1.0 (unshadowed) for this one frame instead of
// falling back to a coarser level.
float SamplePointShadowVSM(uint pointLightSlot, vec3 lightPos, vec3 worldPos) {
    vec3 toPoint = worldPos - lightPos;
    uint face = ShadowCubeFaceIndex(toPoint);
    uint faceSlot = pointLightSlot * 6u + face;
    uint vsmIndex = SHADOW_SUN_LEVEL_COUNT + faceSlot;

    vec4 clip = g_ShadowPointFaces.viewProj[faceSlot] * vec4(worldPos, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0; // Outside this face's own 90-degree frustum -- should not normally happen given face selection.
    }

    vec2 pagePos = clamp((ndc.xy * 0.5 + 0.5) * float(SHADOW_PAGES_PER_AXIS),
        vec2(0.0), vec2(float(SHADOW_PAGES_PER_AXIS) - 1.0e-4));
    uvec2 pageCoord = uvec2(floor(pagePos));
    vec2 pageLocalUV = fract(pagePos);

    uint logicalPageID = ShadowLogicalPageID(vsmIndex, pageCoord);
    if (!IsShadowPageResident(logicalPageID)) {
        RequestShadowPageResidency(logicalPageID);
        return 1.0;
    }

    uint physicalLayer = GetShadowPagePhysicalLayer(logicalPageID);
    return SampleShadowPagePCF(physicalLayer, pageLocalUV, ndc.z);
}

#endif // SHADOW_POINT_SAMPLING_GLSL
