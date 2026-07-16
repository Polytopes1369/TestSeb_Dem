#ifndef SHADOW_SUN_SAMPLING_GLSL
#define SHADOW_SUN_SAMPLING_GLSL

// Phase 3 (UE5.8 parity roadmap): sun shadow lookup against renderer::VirtualShadowMapPass's
// 3-level clipmap of Virtual Shadow Maps -- replaces this codebase's pre-Phase-3 single whole-
// scene-fit orthographic map (renderer::ShadowMapPass). Each level's projection*view is FIXED for
// the whole frame (computed once CPU-side from that level's camera-snapped window, see
// renderer::VirtualShadowMapPass::RecordCapture's own comment on why per-page frustum narrowing is
// not done -- the Fallback Mesh occluder geometry is too light to justify it), so a world position
// maps directly to a page via NDC, no CPU round trip needed.
//
// Requires shadow_page_table.glsl, shadow_feedback.glsl, and shadow_atlas_sampling.glsl to be
// included first (this file does not include them itself, matching the established convention of
// leaving inclusion order/set-binding macro definition entirely to the consumer -- see e.g.
// ReflectionTrace.comp's own multi-include block).
//
// The includer must #define SHADOW_SUN_LEVELS_SET / SHADOW_SUN_LEVELS_BINDING before including
// this header.

#define SHADOW_SUN_LEVEL_COUNT 3u

layout(std140, set = SHADOW_SUN_LEVELS_SET, binding = SHADOW_SUN_LEVELS_BINDING) uniform ShadowSunLevelsUBO {
    mat4 viewProj[SHADOW_SUN_LEVEL_COUNT]; // Index 0 = finest (smallest world coverage), 2 = coarsest.
} g_ShadowSunLevels;

// Tries the sun's clipmap levels finest (0) to coarsest (SHADOW_SUN_LEVEL_COUNT-1): the first level
// whose fixed frustum contains `worldPos` AND whose covering page is already resident is sampled
// directly; a level that contains `worldPos` but whose page is NOT YET resident is requested (see
// shadow_feedback.glsl) and skipped in favor of the next, coarser level -- mirrors
// ClusterLODResidencyFallback.comp's own ancestor-walk fallback (request the fine data for a future
// frame, substitute the nearest already-available coarser data for THIS frame) applied to clipmap
// levels instead of DAG ancestors. If not even the coarsest level's page is resident yet, returns
// 1.0 (fully lit) for this one frame -- already requested, self-heals within a few frames once
// renderer::VirtualShadowMapPass's per-frame page budget catches up.
float SampleSunShadowVSM(vec3 worldPos) {
    for (uint level = 0u; level < SHADOW_SUN_LEVEL_COUNT; ++level) {
        vec4 clip = g_ShadowSunLevels.viewProj[level] * vec4(worldPos, 1.0);
        vec3 ndc = clip.xyz / clip.w;
        if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
            continue; // Outside this level's fixed frustum -- try the next, coarser level.
        }

        // Split NDC into {page index within the 16x16 grid, local UV within that page}: a single
        // (ndc*0.5+0.5)*SHADOW_PAGES_PER_AXIS value whose floor is the page and whose fractional
        // part is the local UV -- clamped just under the top edge so ndc.xy == +1.0 exactly still
        // resolves to the last valid page (15) rather than an out-of-range page index 16.
        vec2 pagePos = clamp((ndc.xy * 0.5 + 0.5) * float(SHADOW_PAGES_PER_AXIS),
            vec2(0.0), vec2(float(SHADOW_PAGES_PER_AXIS) - 1.0e-4));
        uvec2 pageCoord = uvec2(floor(pagePos));
        vec2 pageLocalUV = fract(pagePos);

        uint logicalPageID = ShadowLogicalPageID(level, pageCoord);
        if (!IsShadowPageResident(logicalPageID)) {
            RequestShadowPageResidency(logicalPageID);
            continue;
        }

        uint physicalLayer = GetShadowPagePhysicalLayer(logicalPageID);
        return SampleShadowPagePCF(physicalLayer, pageLocalUV, ndc.z);
    }
    return 1.0;
}

#endif // SHADOW_SUN_SAMPLING_GLSL
